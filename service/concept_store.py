from __future__ import annotations

import json
import sqlite3
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def _ensure_columns(cur: sqlite3.Cursor, table: str, needed: Dict[str, str]) -> None:
    cur.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cur.fetchall()}  # column names
    for col, coltype in needed.items():
        if col not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")


def init_concept_v1_db(db_path: Path) -> None:
    """Initialize the CB-v1 concepts database with enhanced schema."""
    log = logging.getLogger("concept_store_v1")
    log.info("Initializing CB-v1 concepts database: %s", db_path)
    
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    
    # Main concepts table with CB-v1 enhancements
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS concepts_v1 (
            token TEXT PRIMARY KEY,
            is_valid BOOLEAN NOT NULL,
            score REAL NOT NULL,
            shell_size INTEGER NOT NULL,
            center_json TEXT NOT NULL,
            radius REAL NOT NULL,
            mass REAL NOT NULL,
            spread REAL NOT NULL,
            stability REAL NOT NULL,
            phi REAL NOT NULL,
            purity REAL NOT NULL,
            margin REAL NOT NULL,
            horizon_violation BOOLEAN NOT NULL,
            scores_json TEXT NOT NULL,
            shell_members_json TEXT NOT NULL
        )
        """
    )
    
    # Concept shell members table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS concept_shell_members (
            concept_token TEXT NOT NULL,
            member_token TEXT NOT NULL,
            ppmi REAL NOT NULL,
            rank INTEGER NOT NULL,
            PRIMARY KEY (concept_token, member_token),
            FOREIGN KEY (concept_token) REFERENCES concepts_v1(token)
        )
        """
    )
    
    # Concept edges table for inter-concept relationships
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS concept_edges (
            concept_i TEXT NOT NULL,
            concept_j TEXT NOT NULL,
            weight REAL NOT NULL,
            PRIMARY KEY (concept_i, concept_j),
            FOREIGN KEY (concept_i) REFERENCES concepts_v1(token),
            FOREIGN KEY (concept_j) REFERENCES concepts_v1(token)
        )
        """
    )
    
    # Metadata table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS concept_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    
    conn.commit()
    conn.close()
    log.info("CB-v1 concepts database initialized successfully")


def write_concepts_v1_to_db(db_path: Path, concepts: List[Dict], summary_stats: Dict) -> None:
    """Write CB-v1 concepts to the enhanced database."""
    log = logging.getLogger("concept_store_v1")
    log.info("Writing %d CB-v1 concepts to database: %s", len(concepts), db_path)
    
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    
    # Clear existing data
    cur.execute("DELETE FROM concepts_v1")
    cur.execute("DELETE FROM concept_shell_members")
    cur.execute("DELETE FROM concept_edges")
    cur.execute("DELETE FROM concept_meta")
    
    # Insert concepts
    for concept in concepts:
        cur.execute(
            """
            INSERT INTO concepts_v1 (
                token, is_valid, score, shell_size, center_json, radius,
                mass, spread, stability, phi, purity, margin, horizon_violation,
                scores_json, shell_members_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                concept['token'],
                bool(concept['is_valid']),
                float(concept['score']),
                int(concept['shell_size']),
                json.dumps(concept['center']),
                float(concept['radius']),
                float(concept['mass']),
                float(concept['spread']),
                float(concept['stability']),
                float(concept['phi']),
                float(concept['purity']),
                float(concept['margin']),
                bool(concept['horizon_violation']),
                json.dumps(concept['scores']),
                json.dumps(concept['shell_members'])
            )
        )
        
        # Insert shell members
        for rank, (member_token, ppmi) in enumerate(concept['shell_members'], 1):
            cur.execute(
                """
                INSERT INTO concept_shell_members (concept_token, member_token, ppmi, rank)
                VALUES (?, ?, ?, ?)
                """,
                (concept['token'], member_token, float(ppmi), rank)
            )
    
    # Insert metadata
    for key, value in summary_stats.items():
        cur.execute(
            "INSERT INTO concept_meta (key, value) VALUES (?, ?)",
            (key, json.dumps(value))
        )
    
    conn.commit()
    conn.close()
    log.info("CB-v1 concepts written to database successfully")


def write_concepts_v1_to_json(output_dir: Path, concepts: List[Dict], summary_stats: Dict) -> None:
    """Write CB-v1 concepts to JSON files in the concepts directory."""
    log = logging.getLogger("concept_store_v1")
    log.info("Writing CB-v1 concepts to JSON files in: %s", output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Save all concepts
    concepts_data = {
        'concepts': convert_numpy_types(concepts),
        'summary': convert_numpy_types(summary_stats),
        'metadata': {
            'version': '1.0',
            'pipeline': 'CB-v1',
            'total_concepts': len(concepts),
            'valid_concepts': len([c for c in concepts if c['is_valid']])
        }
    }
    
    with (output_dir / "concepts_v1.json").open('w', encoding='utf-8') as f:
        json.dump(concepts_data, f, indent=2)
    
    # Save only valid concepts for easier access
    valid_concepts = [c for c in concepts if c['is_valid']]
    valid_data = {
        'valid_concepts': convert_numpy_types(valid_concepts),
        'summary': convert_numpy_types(summary_stats)
    }
    
    with (output_dir / "valid_concepts_v1.json").open('w', encoding='utf-8') as f:
        json.dump(valid_data, f, indent=2)
    
    # Save summary statistics
    with (output_dir / "summary_v1.json").open('w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(summary_stats), f, indent=2)
    
    log.info("CB-v1 concepts written to JSON files successfully")
