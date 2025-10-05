from __future__ import annotations

import json
import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from service.ppmi_builder import PPMIBuilder, PPMIConfig
from service.concept_net import ConceptNet, ConceptNetConfig, deterministic_embedding
from service.concept_store import init_concept_db, write_concepts_to_db


@dataclass
class PipelineConfig:
    db_path: Path = Path("service/cne_words.db")
    window: int = 5
    min_count: int = 5
    top_k: int = 256
    max_vocab: int = 100000
    dim: int = 128
    s_mid: float = 0.6
    s_high: float = 1.2
    output_dir: Path = Path("concept_ppmi_results")


class ConceptPPMIPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.log = logging.getLogger("concept_ppmi")
        if not self.log.handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    def run(self) -> None:
        cfg = self.cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.log.info("Building PPMI index (window=%d, K=%d, max_vocab=%d)...", cfg.window, cfg.top_k, cfg.max_vocab)
        ppmi = PPMIBuilder(PPMIConfig(db_path=cfg.db_path, window=cfg.window, min_count=cfg.min_count, top_k=cfg.top_k, max_vocab=cfg.max_vocab)).build()
        self.log.info("PPMI index built for %d tokens", len(ppmi))

        # Prepare embeddings (load if trained, else deterministic)
        dim = cfg.dim
        E: Dict[str, np.ndarray] = {}
        emb_path = Path("models/embeddings.npy")
        vocab_path = Path("models/embedding_vocab.json")
        if emb_path.exists() and vocab_path.exists():
            self.log.info("Loading trained embeddings from %s", emb_path)
            emb = np.load(emb_path)
            with vocab_path.open("r", encoding="utf-8") as f:
                vocab_map = json.load(f)
            inv = {k: int(i) for k, i in vocab_map.items()}
            if emb.shape[1] != dim:
                self.log.warning("Embedding dim mismatch: expected %d got %d; proceeding", dim, emb.shape[1])
                dim = emb.shape[1]
            for w in ppmi.keys():
                if w in inv:
                    E[w] = emb[inv[w]]
                else:
                    E[w] = deterministic_embedding(w, dim)
        else:
            self.log.info("No trained embeddings found; using deterministic hash embeddings")
            for w in ppmi.keys():
                E[w] = deterministic_embedding(w, dim)

        # Concept projector (no weights by default; LayerNorm fallback)
        net = ConceptNet(ConceptNetConfig(dim=dim))

        concepts: Dict[str, Dict] = {}
        stability_values: List[float] = []
        spreads: List[float] = []
        masses: List[float] = []
        avg_ppmis: List[float] = []

        for i, (w, neigh) in enumerate(ppmi.items(), 1):
            if not neigh:
                continue
            # Weighted sum of neighbor embeddings with softclipped PPMI
            weights = [math.log1p(pp) for _, pp in neigh]
            denom = sum(weights) + 1e-8
            Xs = []
            for (cid, pp), wt in zip(neigh, weights):
                vec = E.get(cid)
                if vec is None:
                    vec = deterministic_embedding(cid, dim)
                    E[cid] = vec
                Xs.append(wt * vec)
            x = np.sum(np.vstack(Xs), axis=0) / denom
            center, log_spread = net.forward(x)
            spread = float(np.exp(log_spread))
            mass = float(sum(pp for _, pp in neigh))
            stability = mass / (1e-6 + spread)
            avg_ppmi = float(np.mean([pp for _, pp in neigh]))

            # Temporarily mark phase; will be overwritten by Schwarzschild overlay below
            phase = "stable"

            concepts[w] = {
                "center": center.tolist(),
                "spread": spread,
                "mass": mass,
                "stability": stability,
                "avg_ppmi": avg_ppmi,
                "phase": phase,
                "neighbors": [{"id": cid, "ppmi": pp} for cid, pp in neigh],
            }
            stability_values.append(stability)
            spreads.append(spread)
            masses.append(mass)
            avg_ppmis.append(avg_ppmi)
            if i % 500 == 0:
                self.log.info("Processed %d concepts...", i)

        # Fit expected PPMI vs log(mass+1) (quadratic), compute spread median
        if concepts:
            log_m = np.log(np.array(masses) + 1.0)
            y_pp = np.array(avg_ppmis)
            try:
                coeffs = np.polyfit(log_m, y_pp, 2)
            except Exception:
                coeffs = np.array([0.0, 0.0, float(np.mean(y_pp))])
            def expected_ppmi(mass_val: float) -> float:
                lm = math.log(mass_val + 1.0)
                return float(coeffs[0]*lm*lm + coeffs[1]*lm + coeffs[2])
            s_star = float(np.median(np.array(spreads))) if spreads else 1.0

            # Constants from the Laws
            a = 2.1; b = 0.52; gamma = 0.25; kappa = 0.6; delta = 0.15

            # Apply Schwarzschild overlay and reclassify phase
            for w, rec in concepts.items():
                mass = float(rec["mass"])
                spread = float(rec["spread"])
                avg_ppmi = float(rec.get("avg_ppmi", 0.0))
                rs = (a + b * math.log(mass + 1.0)) * ((spread / (s_star if s_star>0 else 1.0)) ** gamma) * (1.0 - math.tanh(kappa * (avg_ppmi - expected_ppmi(mass))))
                # guard
                if rs <= 1e-9:
                    rs = 1e-9
                phi = avg_ppmi / rs
                # Classification aligned with analysis/event_horizon_band_test.py
                if phi < 1.0 - delta:
                    phase = "collapsed"
                elif abs(phi - 1.0) <= delta:
                    phase = "event_horizon"
                else:
                    phase = "stable"
                rec["rs"] = rs
                rec["phi"] = phi
                rec["phase"] = phase

        # Persist
        with (cfg.output_dir / "concepts_ppmi.json").open("w", encoding="utf-8") as f:
            json.dump({"concepts": concepts}, f, indent=2)
        summary = {
            "tokens": len(concepts),
            "stability": {
                "mean": float(np.mean(stability_values)) if stability_values else 0.0,
                "min": float(np.min(stability_values)) if stability_values else 0.0,
                "max": float(np.max(stability_values)) if stability_values else 0.0,
            },
            "spread": {
                "mean": float(np.mean(spreads)) if spreads else 0.0,
            },
            "mass": {
                "mean": float(np.mean(masses)) if masses else 0.0,
            },
            "thresholds": {"s_mid": cfg.s_mid, "s_high": cfg.s_high},
        }
        # Add overlay constants and band stats
        if concepts:
            phases = {}
            for rec in concepts.values():
                phases[rec["phase"]] = phases.get(rec["phase"], 0) + 1
            summary["phase_counts"] = phases
            summary["schwarzschild"] = {"a": a, "b": b, "gamma": gamma, "kappa": kappa, "delta": delta, "s_star": s_star, "ppmi_fit_coeffs": coeffs.tolist()}
        with (cfg.output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        # Persist to dedicated concepts DB
        db_path = cfg.output_dir / "concepts.db"
        init_concept_db(db_path)
        write_concepts_to_db(db_path, concepts, ppmi)
        self.log.info("PPMI concept build complete: %d concepts; DB=%s", len(concepts), db_path)


def main():
    ConceptPPMIPipeline(PipelineConfig()).run()


if __name__ == "__main__":
    main()
