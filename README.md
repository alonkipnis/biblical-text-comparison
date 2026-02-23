Restored Streamlit App
======================

This folder contains a minimal, self-contained version of the biblical text comparison webapp.

Highlights
- **Self-contained**: All paths are relative to this folder (`restored_app`). No references to parent or sibling directories—suitable for deployment.
- Uses local OSHB XML data from `data/morphhb/wlc/` and catalog at `data/reference_data.csv` (bundled in this folder).
- Cache and processed data live inside this folder: `cache/` (piece CSVs, selected_pieces.json, presets) and `proc_data/` (full-dataset CSV, hash).
- Ad hoc mode: compare any two user-defined collections (book/chapter/verse ranges), loading only needed pieces with on-disk caching under `cache/`.
- No external Higher Criticism package: HC* and binomial allocation P-values are implemented locally.
- Avoids old user-specific imports and hardcoded paths.

Run
- Install dependencies: `pip install -r requirements.txt` (from this folder)
- Launch: `streamlit run app.py` (run from inside `restored_app`, or `streamlit run restored_app/app.py` from repo root; cache and proc_data will always be under `restored_app/`)

Notes
- Known/Test mode: on first run, processed data is cached under `proc_data/`. Subsequent runs reuse the CSV.
- Ad hoc mode: only requested pieces are read from XML; each piece is cached as CSV in `cache/` for fast reuse.
- The vocabulary is formed from the most-frequent features per collection/corpus; size is configurable in the sidebar.
