from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple


def _ensure_columns(cur: sqlite3.Cursor, table: str, needed: Dict[str, str]) -> None:
    cur.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cur.fetchall()}  # column names
    for col, coltype in needed.items():
        if col not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")


def init_concept_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS concepts (
            token TEXT PRIMARY KEY,
            center_json TEXT NOT NULL,
            spread REAL NOT NULL,
            mass REAL NOT NULL,
            stability REAL NOT NULL,
            phase TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ppmi_neighbors (
            token TEXT NOT NULL,
            neighbor TEXT NOT NULL,
            ppmi REAL NOT NULL,
            rank INTEGER NOT NULL,
            PRIMARY KEY (token, neighbor)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    # Add new columns if missing
    _ensure_columns(cur, "concepts", {"rs": "REAL", "phi": "REAL", "avg_ppmi": "REAL"})
    conn.commit()
    conn.close()


def write_concepts_to_db(db_path: Path, concepts: Dict[str, Dict], ppmi_index: Dict[str, List[Tuple[str, float]]]) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    # Ensure columns present (idempotent)
    _ensure_columns(cur, "concepts", {"rs": "REAL", "phi": "REAL", "avg_ppmi": "REAL"})
    # Upsert concepts
    for token, rec in concepts.items():
        cur.execute(
            """
            INSERT INTO concepts (token, center_json, spread, mass, stability, phase, rs, phi, avg_ppmi)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(token) DO UPDATE SET
              center_json=excluded.center_json,
              spread=excluded.spread,
              mass=excluded.mass,
              stability=excluded.stability,
              phase=excluded.phase,
              rs=excluded.rs,
              phi=excluded.phi,
              avg_ppmi=excluded.avg_ppmi
            """,
            (
                token,
                json.dumps(rec.get("center", [])),
                float(rec.get("spread", 0.0)),
                float(rec.get("mass", 0.0)),
                float(rec.get("stability", 0.0)),
                str(rec.get("phase", "unstable")),
                float(rec.get("rs", 0.0)),
                float(rec.get("phi", 0.0)),
                float(rec.get("avg_ppmi", 0.0)),
            ),
        )
    # Upsert neighbors (replace strategy)
    for token, neigh in ppmi_index.items():
        for rank, (cid, pp) in enumerate(neigh, 1):
            cur.execute(
                """
                INSERT INTO ppmi_neighbors (token, neighbor, ppmi, rank)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(token, neighbor) DO UPDATE SET
                  ppmi=excluded.ppmi,
                  rank=excluded.rank
                """,
                (token, cid, float(pp), rank),
            )
    conn.commit()
    conn.close()
