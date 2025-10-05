#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Ensure project root is on sys.path so `service.*` imports resolve
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from service.ppmi_builder import PPMIBuilder, PPMIConfig


@dataclass
class TrainConfig:
    db_path: Path = Path("service/cne_words.db")
    dim: int = 128
    top_k: int = 128
    max_vocab: int = 50000
    min_count: int = 5
    window: int = 5
    emb_epochs: int = 6
    emb_eta: float = 0.5
    mlp_hidden: int = 128
    mlp_epochs: int = 10
    mlp_lr: float = 1e-2
    out_dir: Path = Path("models")


def build_ppmi(cfg: TrainConfig):
    return PPMIBuilder(PPMIConfig(cfg.db_path, cfg.window, cfg.min_count, cfg.top_k, cfg.max_vocab)).build()


def train_embeddings(ppmi_index: Dict[str, List[Tuple[str, float]]], dim: int, epochs: int, eta: float, log: logging.Logger):
    vocab = list(ppmi_index.keys())
    index = {w: i for i, w in enumerate(vocab)}
    rng = np.random.default_rng(42)
    E = rng.normal(scale=0.1, size=(len(vocab), dim))
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    for ep in range(epochs):
        for w in vocab:
            i = index[w]
            neigh = ppmi_index[w]
            if not neigh:
                continue
            weights = np.array([math.log1p(pp) for _, pp in neigh], dtype=float)
            denom = float(np.sum(weights)) + 1e-8
            m = np.zeros((dim,), dtype=float)
            for (cid, _), wt in zip(neigh, weights):
                j = index.get(cid)
                if j is None:
                    continue
                m += wt * E[j]
            m /= denom
            E[i] = (1.0 - eta) * E[i] + eta * m
            # normalize
            E[i] /= (np.linalg.norm(E[i]) + 1e-8)
        log.info("Emb epoch %d/%d", ep + 1, epochs)
    return E, index


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def train_concept_mlp(ppmi_index: Dict[str, List[Tuple[str, float]]], E: np.ndarray, index: Dict[str, int], dim: int, hidden: int, epochs: int, lr: float, log: logging.Logger):
    # Initialize
    rng = np.random.default_rng(123)
    W1 = rng.normal(scale=0.1, size=(hidden, dim))
    b1 = np.zeros((hidden,), dtype=float)
    W2 = rng.normal(scale=0.1, size=(dim + 1, hidden))
    b2 = np.zeros((dim + 1,), dtype=float)

    vocab = list(ppmi_index.keys())

    for ep in range(epochs):
        loss_total = 0.0
        count = 0
        for w in vocab:
            i = index.get(w)
            if i is None:
                continue
            neigh = ppmi_index[w]
            if not neigh:
                continue
            weights = np.array([math.log1p(pp) for _, pp in neigh], dtype=float)
            denom = float(np.sum(weights)) + 1e-8
            xs = np.zeros((dim,), dtype=float)
            for (cid, _), wt in zip(neigh, weights):
                j = index.get(cid)
                if j is None:
                    continue
                xs += wt * E[j]
            xs /= denom
            # targets: center_t = xs; var_t = average squared distance
            dists = []
            for (cid, _), wt in zip(neigh, weights):
                j = index.get(cid)
                if j is None:
                    continue
                dists.append(np.sum((E[j] - xs) ** 2))
            var_t = float(np.mean(dists)) if dists else 0.0

            # Forward
            a1 = W1 @ xs + b1
            h1 = tanh(a1)
            y = W2 @ h1 + b2  # dim+1
            center = y[:dim]
            log_spread = y[-1]
            spread = float(np.exp(log_spread))

            # Loss: center MSE + var (spread) MSE
            err_c = center - xs
            err_s = spread - var_t
            loss = float(np.dot(err_c, err_c) + err_s * err_s)
            loss_total += loss
            count += 1

            # Backprop
            dcenter = 2.0 * err_c
            dlog_s = 2.0 * err_s * spread
            dy = np.zeros((dim + 1,), dtype=float)
            dy[:dim] = dcenter
            dy[-1] = dlog_s
            dW2 = np.outer(dy, h1)
            db2 = dy
            dh1 = W2.T @ dy
            da1 = dh1 * (1.0 - np.tanh(a1) ** 2)
            dW1 = np.outer(da1, xs)
            db1 = da1

            # SGD update
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1
        log.info("MLP epoch %d/%d loss=%.6f", ep + 1, epochs, loss_total / max(count, 1))

    params = {"W1": W1.tolist(), "b1": b1.tolist(), "W2": W2.tolist(), "b2": b2.tolist()}
    return params


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    log = logging.getLogger("concept_train")
    cfg = TrainConfig()
    ppmi = build_ppmi(cfg)
    log.info("PPMI size: %d", len(ppmi))
    E, index = train_embeddings(ppmi, cfg.dim, cfg.emb_epochs, cfg.emb_eta, log)
    cfg.out_dir.mkdir(exist_ok=True)
    np.save(cfg.out_dir / "embeddings.npy", E)
    with (cfg.out_dir / "embedding_vocab.json").open("w", encoding="utf-8") as f:
        json.dump({w: int(i) for w, i in index.items()}, f)
    log.info("Saved embeddings: %s", cfg.out_dir / "embeddings.npy")
    params = train_concept_mlp(ppmi, E, index, cfg.dim, cfg.mlp_hidden, cfg.mlp_epochs, cfg.mlp_lr, log)
    with (cfg.out_dir / "concept_net.json").open("w", encoding="utf-8") as f:
        json.dump(params, f)
    log.info("Saved concept net: %s", cfg.out_dir / "concept_net.json")


if __name__ == "__main__":
    main()
