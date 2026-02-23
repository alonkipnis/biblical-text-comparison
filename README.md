# Word-Frequency Comparison of Biblical Texts

A Streamlit web application for comparing two biblical text corpora using Higher Criticism (HC) statistics. The app supports authorship attribution and stylistic analysis of Hebrew Bible passages, following the methodology developed for critical biblical studies.

---

## Background and References

The webapp implements a word-frequency comparison method for authorship analysis of biblical texts. It is based on:

1. **A. Kipnis.** "Higher criticism for discriminating word-frequency tables and authorship attribution." *The Annals of Applied Statistics* 16, no. 2 (2022): 1236–1252.

2. **D. Donoho and A. Kipnis.** "Higher criticism to compare two large frequency tables, with sensitivity to possible rare and weak differences." *The Annals of Statistics* 50, no. 3 (2022): 1447–1472.

3. **S. Faigenbaum-Golovin, A. Kipnis, A. Bühler, E. Piasetzky, T. Römer, and I. Finkelstein.** "Critical biblical studies via word frequency analysis: Unveiling text authorship." *PLOS ONE* 20, no. 6 (2025): e0322905.

The preset authorship assignments (P, Dtr, DtrH) in the app follow the Documentary Hypothesis classifications used in [3].

---

## Technical Details

### Data: Open Source Hebrew Bible (OSHB)

The application uses morphologically tagged Hebrew Bible data from the **Open Scriptures Hebrew Bible** (OSHB) project. The OSHB provides lemma and morphology attributes for each word in OSIS XML format.

- **Source:** [Open Scriptures Hebrew Bible (morphhb)](https://github.com/openscriptures/morphhb)
- **Data location:** `data/morphhb/wlc/` — one XML file per biblical book (e.g., `Gen.xml`, `Exod.xml`)
- **Attributes used:** Each word tag includes `lemma` (e.g., Strong’s-style codes), `morph` (e.g., `HC/R/Ncmsc` for part-of-speech and morphology), and the surface Hebrew `term`
- **License:** Lemma and morphology data are under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the WLC text is in the public domain

### The `reference_data` Catalog

The file `data/reference_data.csv` is a **catalog** that maps biblical passages to authorship or corpus labels. It defines which book/chapter/verse ranges belong to which corpus for comparison.

| Column      | Description                                                                 |
|------------|-----------------------------------------------------------------------------|
| `book`     | OSIS book code (e.g., `Gen`, `Exod`, `Deut`, `2Kgs`)                       |
| `chapter`  | Chapter number                                                              |
| `author`   | Corpus label (e.g., `P`, `Dtr`, `DtrH`, `Ark1`, `Ark2`)                     |
| `to_report`| Whether the passage is included in the main authorship presets              |
| `verses`   | Optional verse specification: blank = all verses; `1-21` or `1;3;5` = specific verses |

The catalog supports both **Known mode** (predefined corpora such as P, Dtr, DtrH) and **Ad hoc mode** (user-defined book/chapter/verse ranges for arbitrary comparisons).

### Main Modules

| Module                     | Purpose                                                                     |
|----------------------------|-----------------------------------------------------------------------------|
| `app.py`                   | Streamlit UI: corpus selection, vocabulary options, HC results display, Hebrew text highlighting |
| `oshb.py`                  | OSHB utilities: reads XML via the catalog, caches flat CSVs, `ProcessText` for feature extraction (prefix/suffix, n-grams, POS filtering) |
| `two_corpora_bible_loader.py` | Data loading and processing: piece loading with disk caching, vocabulary building (top-*n* features per corpus), `LemmaMapper` for Hebrew display |
| `two_corpora_hc_analysis.py`  | Higher Criticism analysis: fits `CompareDocs`, computes global HC* and per-feature p-values, per-document leave-one-out HC |
| `compare.py`               | `CompareDocs` model: two-sample binomial allocation p-values and HC threshold for frequency table comparison |

### Caching and Processed Data

- **`cache/`** — Piece-level CSVs, `selected_pieces.json`, presets (for ad hoc mode)
- **`proc_data/`** — Full-dataset CSV and hash (for known mode; reused on subsequent runs)

---

## Running the Application

1. Install dependencies: `pip install -r requirements.txt`
2. Launch: `streamlit run app.py`

For deployment, see `Procfile` (Heroku) and `Dockerfile`.
