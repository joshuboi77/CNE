from __future__ import annotations

import math
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class PPMIConfig:
    db_path: Path
    window: int = 5
    min_count: int = 5
    top_k: int = 256
    max_vocab: int = 10000  # only top-N by frequency participate


class PPMIBuilder:
    """Streaming co-occurrence and PPMI index builder (symmetric window)."""

    def __init__(self, cfg: PPMIConfig):
        self.cfg = cfg

    def _fetch_vocab(self, conn: sqlite3.Connection) -> List[Tuple[str, int]]:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT cleaned_word, COUNT(*) as f
            FROM words
            WHERE cleaned_word != ''
            GROUP BY cleaned_word
            ORDER BY f DESC
            """
        )
        rows = cur.fetchall()
        return rows[: self.cfg.max_vocab]

    def _iterate_sequences(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute(
            """
            SELECT work_id, cleaned_word
            FROM words
            WHERE cleaned_word != ''
            ORDER BY work_id, position
            """
        )
        current = None
        seq = []
        for work_id, token in cur:
            if work_id != current and seq:
                yield seq
                seq = []
            current = work_id
            seq.append(token)
        if seq:
            yield seq

    def build(self) -> Dict[str, List[Tuple[str, float]]]:
        cfg = self.cfg
        conn = sqlite3.connect(str(cfg.db_path))

        # Select active vocab
        vocab_rows = self._fetch_vocab(conn)
        vocab = [w for w, f in vocab_rows]
        vocab_set = set(vocab)
        # Per-word neighbor counts and totals
        co_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        total_co = 0  # total co-occurrence events
        n_w: Dict[str, int] = defaultdict(int)  # total co-occurrence count per word

        # Stream documents
        W = cfg.window
        for seq in self._iterate_sequences(conn):
            buf: deque[str] = deque([], maxlen=2 * W + 1)
            for tok in seq:
                if tok not in vocab_set:
                    # still slide the window but skip counting for OOV center/neighbor
                    buf.append(tok)
                    continue
                buf.append(tok)
                center_idx = len(buf) - 1
                # look back up to W previous tokens in the buffer
                for off in range(1, min(W, center_idx) + 1):
                    nbh = buf[center_idx - off]
                    if nbh in vocab_set:
                        # symmetric update
                        co_counts[tok][nbh] += 1
                        co_counts[nbh][tok] += 1
                        n_w[tok] += 1
                        n_w[nbh] += 1
                        total_co += 2

        conn.close()

        # Compute PPMI
        # P(w,x) = n_wc / total_co ; P(w) = n_w / total_co
        ppmi_index: Dict[str, List[Tuple[str, float]]] = {}
        for w, neigh in co_counts.items():
            entries: List[Tuple[str, float]] = []
            n_w_total = n_w.get(w, 0)
            if n_w_total < cfg.min_count:
                ppmi_index[w] = []
                continue
            for x, n_wc in neigh.items():
                if n_wc <= 0:
                    continue
                n_x_total = n_w.get(x, 0)
                if n_x_total < cfg.min_count:
                    continue
                # PMI = log( (n_wc * total_co) / (n_w_total * n_x_total) )
                val = (n_wc * total_co) / (n_w_total * n_x_total)
                if val <= 0:
                    continue
                pmi = math.log(val)
                if pmi > 0:
                    entries.append((x, pmi))
            # Keep top-K neighbors by PPMI
            entries.sort(key=lambda t: t[1], reverse=True)
            if cfg.top_k and len(entries) > cfg.top_k:
                entries = entries[: cfg.top_k]
            ppmi_index[w] = entries

        return ppmi_index

