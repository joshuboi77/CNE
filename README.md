# CNE — Minimal Setup & Build

This is a bare‑bones guide to build the concepts database using the PPMI + tiny‑NN pipeline.

## 1) Activate virtualenv (recommended)

```
python3 -m venv .venv
source .venv/bin/activate
```

## 2) Initialize the corpus database (SQLite)

This scans the corpus at `data/cne_corpus/` and writes an indexed token table to `service/cne_words.db`.

```
python service/data_loader.py
```

Notes:
- Adjust the corpus path in `service/data_loader.py` if needed.

## 3) Train embeddings + concept net (numpy‑only)

This trains lightweight embeddings and a tiny MLP projector across the full vocabulary.

```
python service/train_concept_net.py
```

Artifacts:
- `models/embeddings.npy`
- `models/embedding_vocab.json`
- `models/concept_net.json`

## 4) Build concepts (PPMI + NN + Schwarzschild overlay)

This constructs PPMI neighbors, projects concepts, computes spread/mass, applies the residual‑aware Schwarzschild radius and φ‑band phase rule, and persists results.

```
python service/concept_ppmi_pipeline.py
```

Outputs:
- JSON (pretty): `concept_ppmi_results/concepts_ppmi.json`
- Concepts DB: `concept_ppmi_results/concepts.db`
- Summary: `concept_ppmi_results/summary.json`

## 5) Quick queries

Top‑level phase counts:
```
sqlite3 concept_ppmi_results/concepts.db "SELECT phase, COUNT(*) FROM concepts GROUP BY phase;"
```

Sample φ near the event horizon band:
```
sqlite3 concept_ppmi_results/concepts.db "SELECT token, phi FROM concepts WHERE ABS(phi-1) <= 0.15 ORDER BY ABS(phi-1) ASC LIMIT 20;"
```

Optional analysis & plots:
```
python analysis/stability_ppmi_analysis.py
python analysis/schwarzschild_radius_test.py
python analysis/event_horizon_band_test.py
```

Notes
- The legacy physics‑gated path lives under `legacy/` for reference.
- All constants/definitions are documented in `formulae/ENGINEERING_SPEC.md`.
