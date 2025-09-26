#!/usr/bin/env python3
"""
Contextual Node Engine (CNE) — Deterministic Builder

Implements the end-to-end build pipeline described in the blueprint:
- Tokenize corpus
- Global stats and priors (giants/tiers/collapse/colors)
- Co-occurrence within ±k window (optionally distance-weighted)
- PPMI-based sparse shapes (top-K features; exclude collapsed + giants as features)
- Local tiering per focus, purple modal band, cosine ranking, cohorts
- SQLite persistence with minimal indices and deterministic tie-breaking
- Optional second-order spectra: per-word cosine shape vectors (fixed K), stored in `second_order_vec`, with 2D PCA projection in `second_order_proj`.

CLI:
- build <corpus.txt> [--db cne.db] [--config config.json]
- inspect <db> <word>

This module does not render the UI; the UI can load from the SQLite DB.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import re


# -----------------------
# Tokenization
# -----------------------
WORD_RE = re.compile(r"[A-Za-z’']+")


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


# -----------------------
# Config
# -----------------------
@dataclasses.dataclass(frozen=True)
class CNEConfig:
    window_k: int = 10
    collapse_tau: float = 0.03
    red_gamma: float = 0.75
    kappa_modal: float = 0.25
    kshape: int = 128
    distance_weighting: bool = False

    # second-order (optional)
    second_enable: bool = False
    second_k: int = 64
    second_neighbor_mode: str = "context_topk"  # or "purple_top"
    second_target_vocab_size: int = 10000
    second_project: bool = True  # write 2D PCA projection if True

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))


# -----------------------
# SQLite schema
# -----------------------
SCHEMA_SQL = [
    # meta table
    """
    CREATE TABLE IF NOT EXISTS meta (
      key   TEXT PRIMARY KEY,
      value TEXT
    );
    """,
    # vocab with priors
    """
    CREATE TABLE IF NOT EXISTS vocab (
      word         TEXT PRIMARY KEY,
      c            INTEGER,
      p            REAL,
      x            REAL,
      grav         REAL,
      is_giant     INTEGER,
      giant_tier   TEXT,
      is_collapsed INTEGER,
      color_global TEXT
    );
    """,
    # co-occurrence map
    """
    CREATE TABLE IF NOT EXISTS cooc (
      word TEXT,
      nbr  TEXT,
      cnt  INTEGER,
      PRIMARY KEY (word, nbr)
    );
    """,
    # shapes
    """
    CREATE TABLE IF NOT EXISTS shape_feature (
      word    TEXT,
      feature TEXT,
      w       REAL,
      PRIMARY KEY (word, feature)
    );
    """,
    # tier stats per focus
    """
    CREATE TABLE IF NOT EXISTS tier_stat (
      word         TEXT PRIMARY KEY,
      t1_n         REAL,
      t2_n         REAL,
      m_purple     REAL,
      iqr_purple   REAL,
      delta_factor REAL
    );
    """,
    # cohort items for panels
    """
    CREATE TABLE IF NOT EXISTS cohort_item (
      word    TEXT,
      section TEXT,
      nbr     TEXT,
      c_local INTEGER,
      cos     REAL,
      rank    INTEGER,
      PRIMARY KEY (word, section, nbr)
    );
    """,
    # optional convenience for fast charts
    """
    CREATE TABLE IF NOT EXISTS purple_top (
      word TEXT,
      nbr  TEXT,
      cos  REAL,
      rank INTEGER,
      PRIMARY KEY (word, nbr)
    );
    """,
    # second-order vectors (fixed-length spectrum S_w)
    """
    CREATE TABLE IF NOT EXISTS second_order_vec (
      word TEXT,
      pos  INTEGER,
      s    REAL,
      PRIMARY KEY (word, pos)
    );
    """,
    # 2D projection (e.g., PCA2)
    """
    CREATE TABLE IF NOT EXISTS second_order_proj (
      word   TEXT,
      x      REAL,
      y      REAL,
      method TEXT,
      PRIMARY KEY (word)
    );
    """,
]

INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS vocab_freq ON vocab(c DESC);",
    "CREATE INDEX IF NOT EXISTS vocab_color ON vocab(color_global);",
    "CREATE INDEX IF NOT EXISTS cooc_word ON cooc(word);",
    "CREATE INDEX IF NOT EXISTS cooc_nbr ON cooc(nbr);",
    "CREATE INDEX IF NOT EXISTS shape_word ON shape_feature(word);",
    "CREATE INDEX IF NOT EXISTS cohort_word_sec_rank ON cohort_item(word, section, rank);",
    "CREATE INDEX IF NOT EXISTS second_vec_word ON second_order_vec(word);",
    "CREATE INDEX IF NOT EXISTS second_proj_method ON second_order_proj(method);",
]


def create_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for sql in SCHEMA_SQL:
        cur.execute(sql)
    conn.commit()


def create_indices(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for sql in INDEX_SQL:
        cur.execute(sql)
    conn.commit()


# -----------------------
# Helpers
# -----------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_str(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def terciles(arr: List[float]) -> Tuple[float, float]:
    if not arr:
        return (0.0, 0.0)
    a = np.array(arr, dtype=float)
    return (float(np.percentile(a, 33)), float(np.percentile(a, 66)))


def gaussian_gravities(log_counts: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mu = float(log_counts.mean()) if log_counts.size else 0.0
    sigma = float(log_counts.std(ddof=0)) if log_counts.size else 1.0
    if sigma == 0.0:
        sigma = 1.0
    grav = np.exp(-0.5 * ((log_counts - mu) / sigma) ** 2)
    return grav, mu, sigma


def cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    # iterate over the smaller set for speed
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None:
            dot += va * vb
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# -----------------------
# Helpers for second-order spectra
def _load_shapes_for(conn: sqlite3.Connection, words: List[str]) -> Dict[str, Dict[str, float]]:
    """Load sparse PPMI shape_feature rows for a set of words into a dict[word]->{feature:w}.
    Assumes the DB already has shape_feature populated.
    """
    if not words:
        return {}
    placeholders = ",".join(["?"] * len(words))
    cur = conn.cursor()
    cur.execute(f"SELECT word, feature, w FROM shape_feature WHERE word IN ({placeholders})", words)
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for w, feat, wgt in cur.fetchall():
        out[w][feat] = float(wgt)
    return out

def _select_target_vocab(conn: sqlite3.Connection, limit: int) -> List[str]:
    """Return up to `limit` non-collapsed normal-color words ordered by global frequency desc."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT word FROM vocab
        WHERE is_collapsed=0 AND color_global IN ('green','blue','purple')
        ORDER BY c DESC
        LIMIT ?
        """,
        (int(limit),),
    )
    return [r[0] for r in cur.fetchall()]

def _neighbors_context_topk(conn: sqlite3.Connection, word: str, K: int) -> List[Tuple[str, int]]:
    """Top-K neighbors by cooc.count for `word`, excluding collapsed (black). Ordered by cnt desc, nbr asc."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.nbr, c.cnt
        FROM cooc c
        JOIN vocab v ON v.word = c.nbr
        WHERE c.word=? AND v.color_global <> 'black'
        ORDER BY c.cnt DESC, c.nbr ASC
        LIMIT ?
        """,
        (word, int(K)),
    )
    return [(r[0], int(r[1])) for r in cur.fetchall()]

def _neighbors_purple_top(conn: sqlite3.Connection, word: str, K: int) -> List[Tuple[str, float]]:
    """Top-K purple cohort neighbors by precomputed cosine for `word` (uses cohort_item/purple_top)."""
    cur = conn.cursor()
    # Prefer purple_top if available (already ranked by cos), else fall back to cohort_item
    cur.execute(
        "SELECT nbr, cos FROM purple_top WHERE word=? ORDER BY rank ASC LIMIT ?",
        (word, int(K)),
    )
    rows = cur.fetchall()
    if not rows:
        cur.execute(
            """
            SELECT nbr, cos FROM cohort_item
            WHERE word=? AND section='purple' AND cos IS NOT NULL
            ORDER BY rank ASC
            LIMIT ?
            """,
            (word, int(K)),
        )
        rows = cur.fetchall()
    return [(r[0], float(r[1])) for r in rows]

def _build_second_order(conn: sqlite3.Connection, *, K: int, mode: str, target_limit: int, project: bool) -> None:
    """Compute second-order spectra S_w for a target vocabulary and persist into second_order_vec (+ optional PCA2)."""
    mode = (mode or "context_topk").lower()
    words = _select_target_vocab(conn, target_limit)
    if not words:
        return

    # Load shapes for fast cosine; include union of targets and their neighbors (context_topk path needs neighbor shapes).
    # We'll progressively add neighbor shapes as discovered.
    shapes: Dict[str, Dict[str, float]] = _load_shapes_for(conn, words)

    S: List[List[float]] = []
    cur = conn.cursor()
    # Clear any previous results to keep determinism
    cur.execute("DELETE FROM second_order_vec")
    cur.execute("DELETE FROM second_order_proj")

    for w in words:
        spectrum: List[float] = []
        if mode == "purple_top":
            # pre-ranked by cosine; just take cos values (already cosine(φ_w, φ_u))
            nbrs = _neighbors_purple_top(conn, w, K)
            spectrum = [float(cos) for (_u, cos) in nbrs]
        else:
            # context_topk: fetch top-K by local count, then compute cosine(φ_w, φ_u)
            nbrs_ct = _neighbors_context_topk(conn, w, K)
            phi_w = shapes.get(w)
            if phi_w is None:
                # if missing (rare), load on demand
                shapes.update(_load_shapes_for(conn, [w]))
                phi_w = shapes.get(w, {})
            for u, _cnt in nbrs_ct:
                if u not in shapes:
                    shapes.update(_load_shapes_for(conn, [u]))
                phi_u = shapes.get(u, {})
                spectrum.append(cosine_sparse(phi_w, phi_u))

        # pad/truncate to K
        if len(spectrum) < K:
            spectrum.extend([0.0] * (K - len(spectrum)))
        elif len(spectrum) > K:
            spectrum = spectrum[:K]

        # L2 normalize row (shape-only)
        norm = math.sqrt(sum(x * x for x in spectrum))
        if norm > 0:
            spectrum = [x / norm for x in spectrum]
        # write per-position rows
        cur.executemany(
            "INSERT INTO second_order_vec(word,pos,s) VALUES (?,?,?)",
            [(w, i + 1, float(spectrum[i])) for i in range(K)],
        )
        S.append(spectrum)

    conn.commit()

    # Optional 2D projection via PCA (SVD on centered S)
    if project and S:
        A = np.array(S, dtype=float)
        # center rows (column-wise mean subtraction)
        A = A - A.mean(axis=0, keepdims=True)
        # economy SVD
        U, Svals, Vt = np.linalg.svd(A, full_matrices=False)
        # take first two components
        X2 = U[:, :2] * Svals[:2]
        cur.executemany(
            "INSERT INTO second_order_proj(word,x,y,method) VALUES (?,?,?,?)",
            [(words[i], float(X2[i, 0]), float(X2[i, 1]), "PCA2") for i in range(len(words))],
        )
        conn.commit()

def _cosine_list_for_word(conn: sqlite3.Connection, word: str, K_ctx: int = 64) -> List[Tuple[str, float]]:
    """Return [(nbr, cos)] for a word. Prefer precomputed purple cosine, else compute from context_topk using shapes."""
    cur = conn.cursor()
    # 1) try purple_top
    cur.execute("SELECT nbr, cos FROM purple_top WHERE word=? ORDER BY rank ASC", (word,))
    rows = cur.fetchall()
    items: List[Tuple[str, float]] = []
    if rows:
        items = [(str(n), float(c if c is not None else 0.0)) for (n, c) in rows]
    else:
        # 2) fall back to cohort_item purple
        cur.execute(
            """
            SELECT nbr, cos FROM cohort_item
            WHERE word=? AND section='purple' AND cos IS NOT NULL
            ORDER BY rank ASC
            """,
            (word,),
        )
        rows = cur.fetchall()
        if rows:
            items = [(str(n), float(c)) for (n, c) in rows]
        else:
            # 3) compute from context_topk using shapes
            nbrs = _neighbors_context_topk(conn, word, K_ctx)
            # load shapes lazily
            shapes = _load_shapes_for(conn, [word] + [u for (u, _c) in nbrs])
            phi_w = shapes.get(word, {})
            tmp: List[Tuple[str, float]] = []
            for u, _cnt in nbrs:
                phi_u = shapes.get(u, {})
                tmp.append((u, cosine_sparse(phi_w, phi_u)))
            items = tmp
    # Sort strictly by cosine desc then lex asc for determinism
    items.sort(key=lambda t: (-t[1], t[0]))
    return items

# -----------------------
# Helpers for full-context cosine shape dump
def _neighbors_full_context(conn: sqlite3.Connection, word: str) -> List[str]:
    """Return ALL context neighbors (any section) for `word`, excluding collapsed ('black') terms.
    Ordered by nbr lex asc as a stable base; ranking happens later.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.nbr
        FROM cooc c
        JOIN vocab v ON v.word = c.nbr
        WHERE c.word=? AND v.color_global <> 'black'
        ORDER BY c.nbr ASC
        """,
        (word,),
    )
    return [str(r[0]) for r in cur.fetchall()]


def _cosine_list_full_context(conn: sqlite3.Connection, word: str) -> List[Tuple[str, float]]:
    """Compute cosine list using FULL context neighbors (excluding 'black'), giants + normals included.
    Cosine is computed via sparse PPMI shapes for both sides.
    Returns [(nbr, cos)] sorted by cosine desc, then nbr lex asc.
    """
    nbrs = _neighbors_full_context(conn, word)
    # include the word and neighbors to fetch shapes; shapes may be missing for rare terms
    shapes = _load_shapes_for(conn, [word] + nbrs)
    phi_w = shapes.get(word, {})
    items: List[Tuple[str, float]] = []
    for u in nbrs:
        phi_u = shapes.get(u, {})
        items.append((u, cosine_sparse(phi_w, phi_u)))
    items.sort(key=lambda t: (-t[1], t[0]))
    return items

def _analyze_dump(conn: sqlite3.Connection, out_path: Path, k: int | None = None) -> None:
    """Write a global cosine rank-dump for every vocab word to a single text file.
    Words are ordered by their FULL-CONTEXT cosine shape vectors (giants + normals; excludes 'black'),
    using descending lexicographic over the padded, L2-normalized cosine-only list.

    If k is provided, truncate each word's neighbor list to top-k *after* ranking by cosine.
    """
    cur = conn.cursor()
    cur.execute("SELECT word FROM vocab")
    vocab_words = [str(r[0]) for r in cur.fetchall()]

    # Build per-word neighbor lists from FULL context and extract cosine-only vectors (excluding identity)
    word_items: Dict[str, List[Tuple[str, float]]] = {}
    shapes_raw: Dict[str, List[float]] = {}
    L = 0  # global max length across words (after identity removal and optional truncation)
    for w in vocab_words:
        items = _cosine_list_full_context(conn, w)
        if k is not None:
            items = items[: int(k)]
        word_items[w] = items
        cos = [float(c) for (nbr, c) in items if nbr != w]  # exclude self
        shapes_raw[w] = cos
        if len(cos) > L:
            L = len(cos)

    # Pad to L and L2-normalize each shape vector
    shapes: Dict[str, List[float]] = {}
    for w, vec in shapes_raw.items():
        if len(vec) < L:
            vec = vec + [0.0] * (L - len(vec))
        n = math.sqrt(sum(x * x for x in vec))
        if n > 0.0:
            vec = [x / n for x in vec]
        shapes[w] = vec

    # Deterministic "shape sort" key: descending lexicographic on normalized vector; tie-break by word
    def shape_key(word: str):
        v = shapes[word]
        return tuple(-x for x in v) + (word,)

    sorted_words = sorted(vocab_words, key=shape_key)

    # Emit the report in shape order; within each word, items remain cosine-desc
    with open(out_path, "w", encoding="utf-8") as f:
        for w in sorted_words:
            items = word_items[w]
            f.write(f"WORD: {w}\n")
            for i, (nbr, cos) in enumerate(items, start=1):
                f.write(f"  {i:>3}. {nbr}\tcos={cos:.3f}\n")
            f.write("\n")

# -----------------------
# Build pipeline
# -----------------------
def build_cne(corpus_path: str | Path, config: CNEConfig, db_path: str | Path = "cne.db") -> Path:
    """
    Deterministic build. Returns path to created SQLite DB.
    Overwrites existing DB if present.
    """
    corpus_path = Path(corpus_path)
    db_path = Path(db_path)

    raw_bytes = corpus_path.read_bytes()
    text = raw_bytes.decode("utf-8", errors="ignore")
    tokens = tokenize(text)

    # 1) Counts
    counts = Counter(tokens)
    vocab_words = sorted(counts.keys())  # determinism
    N = len(tokens)

    # 2) Global stats & priors
    c_arr = np.array([counts[w] for w in vocab_words], dtype=float)
    p_arr = c_arr / (N if N else 1)
    x_arr = np.log(c_arr + 1.0)
    grav_arr, mu, sigma = gaussian_gravities(x_arr)

    is_giant = (grav_arr >= config.red_gamma)

    # Tiers
    normal_idx = [i for i, g in enumerate(is_giant) if not g]
    giant_idx = [i for i, g in enumerate(is_giant) if g]

    normal_tier_map: Dict[int, str] = {}
    if normal_idx:
        normal_counts = [c_arr[i] for i in normal_idx]
        t1_n, t2_n = terciles(normal_counts)
        for i in normal_idx:
            c = c_arr[i]
            if c <= t1_n:
                normal_tier_map[i] = "green"
            elif c <= t2_n:
                normal_tier_map[i] = "blue"
            else:
                normal_tier_map[i] = "purple"

    giant_tier_map: Dict[int, str] = {}
    if giant_idx:
        giant_gravs = [grav_arr[i] for i in giant_idx]
        g1, g2 = terciles(giant_gravs)
        for i in giant_idx:
            g = grav_arr[i]
            if g <= g1:
                giant_tier_map[i] = "yellow"
            elif g <= g2:
                giant_tier_map[i] = "orange"
            else:
                giant_tier_map[i] = "red"

    # Collapse by global share
    collapsed = [1 if (p >= config.collapse_tau) else 0 for p in p_arr]
    # Global color rule
    global_color = []
    for i, w in enumerate(vocab_words):
        if collapsed[i]:
            global_color.append("black")
        elif is_giant[i]:
            global_color.append(giant_tier_map.get(i, "red"))
        else:
            global_color.append(normal_tier_map.get(i, "blue"))

    # 3) Co-occurrence map
    k = int(config.window_k)
    distance = bool(config.distance_weighting)
    cooc: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    T = len(tokens)
    for i, w in enumerate(tokens):
        start = i - k if i - k > 0 else 0
        end = i + k + 1 if i + k + 1 < T else T
        for j in range(start, end):
            if j == i:
                continue
            u = tokens[j]
            if distance:
                wgt = 1.0 / (1.0 + abs(i - j))
            else:
                wgt = 1.0
            cooc[w][u] += wgt

    # Round counts to int and compute cooc_total
    cooc_int: Dict[str, Dict[str, int]] = {}
    cooc_total = 0
    for w, nbrs in cooc.items():
        di: Dict[str, int] = {}
        for u, f in nbrs.items():
            cnt = int(round(f))
            if cnt > 0:
                di[u] = cnt
                cooc_total += cnt
        if di:
            cooc_int[w] = di

    # 4) Shapes — PPMI and top-K features
    # Build quick maps for priors
    word_to_idx = {w: i for i, w in enumerate(vocab_words)}
    color_map = {w: global_color[i] for i, w in enumerate(vocab_words)}
    p_map = {w: float(p_arr[i]) for i, w in enumerate(vocab_words)}

    Z = float(cooc_total) if cooc_total else 1.0

    def topk_ppmi_for(w: str) -> Dict[str, float]:
        nbrs = cooc_int.get(w)
        if not nbrs:
            return {}
        out: List[Tuple[str, float]] = []
        pw = p_map.get(w, 0.0)
        if pw <= 0.0:
            return {}
        for u, cnt in nbrs.items():
            # Filter features: not collapsed, and restrict to normal colors {green,blue,purple}
            col_u = color_map.get(u, "blue")
            if col_u not in {"green", "blue", "purple"}:
                continue
            pu = p_map.get(u, 0.0)
            if pu <= 0.0:
                continue
            pwu = float(cnt) / Z
            if pwu <= 0.0:
                continue
            pmi = math.log(pwu / (pw * pu))
            if pmi <= 0:
                continue
            out.append((u, pmi))
        if not out:
            return {}
        # keep top-K by weight, break ties lexicographically
        out.sort(key=lambda t: (-t[1], t[0]))
        top = out[: int(config.kshape)]
        return {u: w for (u, w) in top}

    shapes: Dict[str, Dict[str, float]] = {}
    for w in vocab_words:
        shapes[w] = topk_ppmi_for(w)

    # 5) Local tiering & modal bands & cohorts
    # cohort_item sections: green/blue/purple (normals) + yellow/orange/red (giants). Exclude collapsed (black) from cohorts.
    cohort_rows: List[Tuple[str, str, str, int, float | None, int]] = []
    tier_rows: List[Tuple[str, float, float, float, float, float]] = []
    purple_top_rows: List[Tuple[str, str, float, int]] = []

    # Precompute per-word purple ranking to extract top list (optional)
    for f in vocab_words:
        nbrs = cooc_int.get(f, {})
        if not nbrs:
            # still store tier row with NaNs/zeros
            tier_rows.append((f, 0.0, 0.0, 0.0, 0.0, float(config.kappa_modal)))
            continue
        # Partition neighbors by priors
        normal_items: List[Tuple[str, int]] = []
        giant_items: List[Tuple[str, int, str]] = []  # (nbr, c_local, giant_tier)
        for u, c_local in nbrs.items():
            col = color_map.get(u, "blue")
            if col == "black":
                # exclude collapsed from cohort
                continue
            if col in {"yellow", "orange", "red"}:
                giant_items.append((u, c_local, col))
            else:  # green/blue/purple determined locally
                normal_items.append((u, c_local))

        # Terciles for normal counts
        t1_n = t2_n = 0.0
        normal_counts = [c for _, c in normal_items]
        if normal_counts:
            t1_n, t2_n = terciles([float(c) for c in normal_counts])

        # Assign local sections to normals
        section_map: Dict[str, str] = {}
        for u, c_local in normal_items:
            if c_local <= t1_n:
                section_map[u] = "green"
            elif c_local <= t2_n:
                section_map[u] = "blue"
            else:
                section_map[u] = "purple"

        # Purple modal band
        purple_items = [(u, nbrs[u]) for u in section_map.keys() if section_map[u] == "purple"]
        m_purple = iqr_purple = 0.0
        delta = 0
        purple_ranked: List[Tuple[str, int, float]] = []  # (u, c_local, cosine)
        if purple_items:
            vals = [c for _, c in purple_items]
            vals_sorted = sorted(vals)
            m_purple = float(statistics.median(vals_sorted))
            q1 = float(np.percentile(vals_sorted, 25))
            q3 = float(np.percentile(vals_sorted, 75))
            iqr_purple = max(0.0, q3 - q1)
            delta = max(1, int(round(config.kappa_modal * max(1.0, iqr_purple))))
            # Candidate set C within band
            C = [(u, c) for (u, c) in purple_items if abs(c - m_purple) <= delta]
            # Cosine ranking by shapes
            phi_f = shapes.get(f, {})
            for (u, c) in C:
                phi_u = shapes.get(u, {})
                score = cosine_sparse(phi_f, phi_u)
                purple_ranked.append((u, c, score))
            # sort: cosine desc, |c-m| asc, c desc, lex asc
            purple_ranked.sort(key=lambda t: (-t[2], abs(t[1] - m_purple), -t[1], t[0]))

        # Write tier_stat
        tier_rows.append((f, float(t1_n), float(t2_n), float(m_purple), float(iqr_purple), float(config.kappa_modal)))

        # Populate cohort_item per section with ranks
        # Giants: sort by c desc then lex
        for section in ("yellow", "orange", "red"):
            items_sec = [(u, c) for (u, c, col) in giant_items if col == section]
            if not items_sec:
                continue
            items_sec.sort(key=lambda t: (-t[1], t[0]))
            for rank, (u, c) in enumerate(items_sec, start=1):
                cohort_rows.append((f, section, u, int(c), None, rank))

        # Green/Blue: sort by c desc then lex
        for section in ("green", "blue"):
            items_sec = [(u, nbrs[u]) for u in section_map.keys() if section_map[u] == section]
            if not items_sec:
                continue
            items_sec.sort(key=lambda t: (-t[1], t[0]))
            for rank, (u, c) in enumerate(items_sec, start=1):
                cohort_rows.append((f, section, u, int(c), None, rank))

        # Purple: ranked by purple_ranked order
        if purple_ranked:
            for rank, (u, c, cos) in enumerate(purple_ranked, start=1):
                cohort_rows.append((f, "purple", u, int(c), float(cos), rank))
            # purple_top convenience (top-N)
            TOPN = min(50, len(purple_ranked))
            for rank, (u, _c, cos) in enumerate(purple_ranked[:TOPN], start=1):
                purple_top_rows.append((f, u, float(cos), rank))

    # 6) Persist to SQLite
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    create_schema(conn)

    cur = conn.cursor()
    # meta
    meta_items = {
        "corpus_hash": sha256_bytes(raw_bytes),
        "params_hash": sha256_str(config.to_json()),
        "tokenizer_ver": "regex_v1",
        "window_k": str(config.window_k),
        "collapse_tau": str(config.collapse_tau),
        "red_gamma": str(config.red_gamma),
        "kappa_modal": str(config.kappa_modal),
        "kshape": str(config.kshape),
        "distance_weighting": "1" if config.distance_weighting else "0",
        "total_tokens": str(N),
        "cooc_total": str(cooc_total),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # second-order params
        "second_enable": "1" if config.second_enable else "0",
        "second_k": str(config.second_k),
        "second_neighbor_mode": str(config.second_neighbor_mode),
        "second_target_vocab_size": str(config.second_target_vocab_size),
        "second_project": "1" if config.second_project else "0",
    }
    cur.executemany("INSERT INTO meta(key, value) VALUES(?,?)", meta_items.items())

    # vocab rows
    vocab_rows = []
    for i, w in enumerate(vocab_words):
        vocab_rows.append(
            (
                w,
                int(c_arr[i]),
                float(p_arr[i]),
                float(x_arr[i]),
                float(grav_arr[i]),
                1 if bool(is_giant[i]) else 0,
                (giant_tier_map.get(i) if bool(is_giant[i]) else None),
                int(collapsed[i]),
                global_color[i],
            )
        )
    cur.executemany(
        "INSERT INTO vocab(word,c,p,x,grav,is_giant,giant_tier,is_collapsed,color_global) VALUES (?,?,?,?,?,?,?,?,?)",
        vocab_rows,
    )

    # cooc rows
    cooc_rows = []
    for w, nbrs in cooc_int.items():
        for u, cnt in nbrs.items():
            cooc_rows.append((w, u, int(cnt)))
    cur.executemany("INSERT INTO cooc(word,nbr,cnt) VALUES (?,?,?)", cooc_rows)

    # shapes rows
    shape_rows = []
    for w, feats in shapes.items():
        for u, wgt in feats.items():
            shape_rows.append((w, u, float(wgt)))
    if shape_rows:
        cur.executemany("INSERT INTO shape_feature(word,feature,w) VALUES (?,?,?)", shape_rows)

    # tier_stat
    cur.executemany(
        "INSERT INTO tier_stat(word,t1_n,t2_n,m_purple,iqr_purple,delta_factor) VALUES (?,?,?,?,?,?)",
        tier_rows,
    )

    # cohort_item
    if cohort_rows:
        cur.executemany(
            "INSERT INTO cohort_item(word,section,nbr,c_local,cos,rank) VALUES (?,?,?,?,?,?)",
            cohort_rows,
        )

    # purple_top
    if purple_top_rows:
        cur.executemany(
            "INSERT INTO purple_top(word,nbr,cos,rank) VALUES (?,?,?,?)",
            purple_top_rows,
        )

    # indices last
    create_indices(conn)

    # optional: second-order spectra/projection
    if config.second_enable:
        _build_second_order(
            conn,
            K=int(config.second_k),
            mode=str(config.second_neighbor_mode),
            target_limit=int(config.second_target_vocab_size),
            project=bool(config.second_project),
        )

    conn.commit()
    conn.close()
    return db_path


# -----------------------
# Inspection helpers
# -----------------------
def load_meta(conn: sqlite3.Connection) -> Dict[str, str]:
    cur = conn.cursor()
    cur.execute("SELECT key, value FROM meta")
    return {k: v for (k, v) in cur.fetchall()}


def inspect_word(db_path: str | Path, word: str) -> Dict:
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # priors
    cur.execute(
        "SELECT color_global, is_giant, giant_tier, is_collapsed, c FROM vocab WHERE word=?",
        (word,),
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return {"word": word, "found": False}
    color, is_giant, giant_tier, is_collapsed, c = row

    # purple_top
    cur.execute(
        "SELECT nbr, cos, rank FROM purple_top WHERE word=? ORDER BY rank ASC",
        (word,),
    )
    purple = [(r[0], float(r[1]), int(r[2])) for r in cur.fetchall()]

    # sections
    cur.execute(
        "SELECT section, nbr, c_local, cos, rank FROM cohort_item WHERE word=? ORDER BY section, rank",
        (word,),
    )
    sections: Dict[str, List[Tuple[str, int, float | None, int]]] = defaultdict(list)
    for sec, nbr, c_local, cos, rank in cur.fetchall():
        sections[sec].append((nbr, int(c_local), (float(cos) if cos is not None else None), int(rank)))

    conn.close()
    return {
        "word": word,
        "found": True,
        "priors": {
            "color_global": color,
            "is_giant": int(is_giant),
            "giant_tier": giant_tier,
            "is_collapsed": int(is_collapsed),
            "count": int(c),
        },
        "purple_top": purple,
        "sections": sections,
    }


def _format_inspect_text(info: Dict) -> str:
    """Create a clean, human-readable text report for an inspected word."""
    lines: List[str] = []
    w = info.get("word", "")
    lines.append(f"Word: {w}")
    if not info.get("found"):
        lines.append("Found: no")
        return "\n".join(lines) + "\n"
    lines.append("Found: yes")
    pri = info.get("priors", {})
    lines.append("Priors:")
    lines.append(f"  color_global: {pri.get('color_global')}")
    lines.append(f"  is_giant: {pri.get('is_giant')}")
    lines.append(f"  giant_tier: {pri.get('giant_tier')}")
    lines.append(f"  is_collapsed: {pri.get('is_collapsed')}")
    lines.append(f"  count: {pri.get('count')}")

    # Purple top
    purple_top = info.get("purple_top", [])
    lines.append("")
    lines.append(f"Purple Top ({len(purple_top)}):")
    for nbr, cos, rank in purple_top:
        lines.append(f"  {rank:>3}. {nbr}\tcos={cos:.3f}")

    # Sections ordered for readability
    section_order = ["yellow", "orange", "red", "green", "blue", "purple"]
    sections: Dict[str, List[Tuple[str, int, float | None, int]]] = info.get("sections", {})
    lines.append("")
    lines.append("Sections:")
    for sec in section_order:
        items = sections.get(sec) or []
        if not items:
            continue
        lines.append(f"  {sec} ({len(items)}):")
        # Ensure deterministic by rank, then lex (defensive)
        items_sorted = sorted(items, key=lambda t: (t[3], t[0]))
        for nbr, c_local, cos, rank in items_sorted:
            if cos is None:
                lines.append(f"    {rank:>3}. {nbr}\tc={c_local}")
            else:
                lines.append(f"    {rank:>3}. {nbr}\tc={c_local}\tcos={cos:.3f}")
    return "\n".join(lines) + "\n"


# -----------------------
# CLI
# -----------------------
def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CNE Deterministic Builder")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_b = sub.add_parser("build", help="Build cne.db from corpus")
    ap_b.add_argument("corpus", help="Path to input text corpus")
    ap_b.add_argument("--db", default="cne.db", help="Output SQLite path (default: cne.db)")
    ap_b.add_argument("--config", default=None, help="Optional JSON config file")
    ap_b.add_argument("--second", action="store_true", help="Compute second-order spectra and PCA projection")
    ap_b.add_argument("--second-k", type=int, default=None, help="S_w length (default from config: 64)")
    ap_b.add_argument(
        "--second-mode",
        choices=["context_topk", "purple_top"],
        default=None,
        help="Neighbor selection for S_w (context_topk or purple_top)",
    )
    ap_b.add_argument("--second-target", type=int, default=None, help="Target vocab size (default from config: 10000)")
    ap_b.add_argument("--no-second-proj", action="store_true", help="Disable PCA2 projection for second-order")

    ap_i = sub.add_parser("inspect", help="Inspect a word from DB")
    ap_i.add_argument("db", help="Path to cne.db")
    ap_i.add_argument("word", help="Word to inspect")
    ap_i.add_argument("--out", help="Write a clean text report to this file instead of JSON to stdout")

    ap_a = sub.add_parser("analyze", help="Dump, for every word, neighbors sorted by descending cosine to a text file")
    ap_a.add_argument("db", help="Path to cne.db")
    ap_a.add_argument("--out", required=True, help="Output text file path")
    ap_a.add_argument("--k", type=int, default=None, help="Top-k neighbors per word (default: all available)")

    return ap.parse_args(argv)


def _load_config(path: str | None) -> CNEConfig:
    if not path:
        return CNEConfig()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # allow partial configs
    base = dataclasses.asdict(CNEConfig())
    base.update({k: data[k] for k in data.keys() if k in base})
    return CNEConfig(**base)


def main(argv: List[str] | None = None) -> None:
    import sys
    ns = _parse_args(sys.argv[1:] if argv is None else argv)
    if ns.cmd == "build":
        cfg = _load_config(ns.config)
        if getattr(ns, "second", False):
            # enable and override config if CLI flags provided
            cfg = dataclasses.replace(
                cfg,
                second_enable=True,
                second_k=(ns.second_k if ns.second_k is not None else cfg.second_k),
                second_neighbor_mode=(ns.second_mode if ns.second_mode is not None else cfg.second_neighbor_mode),
                second_target_vocab_size=(ns.second_target if ns.second_target is not None else cfg.second_target_vocab_size),
                second_project=(False if getattr(ns, "no_second_proj", False) else cfg.second_project),
            )
        db_path = build_cne(ns.corpus, cfg, ns.db)
        print(f"Built {db_path}")
    elif ns.cmd == "inspect":
        info = inspect_word(ns.db, ns.word)
        if getattr(ns, "out", None):
            report = _format_inspect_text(info)
            out_path = Path(ns.out)
            out_path.write_text(report, encoding="utf-8")
            print(f"Wrote {out_path}")
        else:
            print(json.dumps(info, indent=2, ensure_ascii=False))
    elif ns.cmd == "analyze":
        db_path = Path(ns.db)
        conn = sqlite3.connect(str(db_path))
        create_schema(conn)
        create_indices(conn)
        out_path = Path(ns.out)
        k = getattr(ns, "k", None)
        _analyze_dump(conn, out_path, k=k)
        conn.close()
        print(f"Wrote cosine dump to {out_path}")
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
