from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ConceptNetConfig:
    dim: int = 128
    model_path: Path = Path("models/concept_net.json")


class ConceptNet:
    """Tiny projector MLP with LayerNorm fallback (no training required)."""

    def __init__(self, cfg: ConceptNetConfig):
        self.cfg = cfg
        self.params = None
        if cfg.model_path.exists():
            with cfg.model_path.open("r", encoding="utf-8") as f:
                self.params = json.load(f)

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mu = x.mean()
        sigma2 = ((x - mu) ** 2).mean()
        return (x - mu) / math.sqrt(sigma2 + eps)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return (center, log_spread). If no params, use LN for center and log var as spread."""
        if self.params is None:
            c = self._layer_norm(x)
            # spread from energy in the tail
            log_spread = float(np.log(np.var(x) + 1e-6))
            return c, log_spread
        # Otherwise, MLP: y = W2*GELU(W1*x+b1)+b2; split head into center, log_spread
        W1 = np.asarray(self.params.get("W1"))
        b1 = np.asarray(self.params.get("b1"))
        W2 = np.asarray(self.params.get("W2"))
        b2 = np.asarray(self.params.get("b2"))
        h = W1 @ x + b1
        # Use tanh for stability and ease of training
        h = np.tanh(h)
        y = W2 @ h + b2
        d = self.cfg.dim
        center = y[:d]
        log_spread = float(y[-1])
        return center, log_spread


def deterministic_embedding(token: str, dim: int = 128) -> np.ndarray:
    """Deterministic pseudo-embedding via hashing; avoids external deps."""
    rng = np.random.default_rng(abs(hash(token)) % (2**32))
    v = rng.normal(scale=1.0, size=(dim,))
    # Normalize
    return v / (np.linalg.norm(v) + 1e-9)
