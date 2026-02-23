"""
TwoCorporaBibleLoader — data loading, text processing, and term mapping for
two-corpus biblical text comparison.

Responsibilities:
  - Book / chapter / verse navigation (wraps OSHBData)
  - Piece loading with disk caching
  - POS-code extraction from morphology tags
  - Text processing pipeline (prefix/suffix extraction, n-gram, POS filtering)
  - Vocabulary building (top-n features per corpus)
  - Lemma-to-Hebrew-term mapping for presentation (LemmaMapper)
  - Preset and selection persistence

This module is Streamlit-agnostic; all UI logic stays in app.py.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from oshb import OSHBData, ProcessText, _read_catalog

# ---------------------------------------------------------------------------
# Data-classes for structured output
# ---------------------------------------------------------------------------

@dataclass
class ProcessingOptions:
    """Parameters controlling how raw token data is turned into features."""
    extract_prefix: bool = True
    extract_suffix: bool = False
    ng_range: Tuple[int, int] = (1, 1)
    pad: bool = False
    to_remove: List[str] = field(default_factory=list)
    to_replace: List[str] = field(default_factory=list)
    n_words: int = 3000


@dataclass
class CorpusData:
    """Result of loading and processing two corpora.

    Attributes
    ----------
    processed : pd.DataFrame
        Unigram-level processed data (used for document display).  Contains
        *feature*, *author*, *doc_id*, *token_id*, *chapter*, *verse*,
        *term*, *morph*, *POW*, etc.
    ng_processed : pd.DataFrame
        N-gram processed data used for HC comparison.  When ``ng_range`` is
        ``(1, 1)`` this is identical to *processed*; otherwise it contains
        the n-gram expansion (features are tuples).
    vocab : list
        Vocabulary (union of top-*n_words* features per corpus).
    pt : ProcessText
        The ``ProcessText`` instance used during processing.
    df_a_raw, df_b_raw : pd.DataFrame
        De-duplicated raw token data for each corpus (before feature
        processing).  Includes *lemma* and *term* columns useful for building
        a ``LemmaMapper``.
    data_a, data_b : pd.DataFrame
        Unigram processed data for each corpus (for display), with index
        reset.
    counts_a, counts_b : pd.DataFrame
        Feature-count DataFrames (columns: *cls*, *feature*, *doc_id*) ready
        for ``TwoCorporaHCAnalysis.fit()``.
    """
    processed: pd.DataFrame
    ng_processed: pd.DataFrame
    vocab: list
    pt: ProcessText
    df_a_raw: pd.DataFrame
    df_b_raw: pd.DataFrame
    data_a: pd.DataFrame
    data_b: pd.DataFrame
    counts_a: pd.DataFrame
    counts_b: pd.DataFrame


# OSHB prefix codes (conjunction, article, prepositions) → Hebrew letters
PREFIX_HEB = {'c': 'ו', 'd': 'ה', 'b': 'ב', 'l': 'ל', 'k': 'כ'}


# ---------------------------------------------------------------------------
# LemmaMapper — maps lemma codes back to Hebrew display terms
# ---------------------------------------------------------------------------

class LemmaMapper:
    """Best-effort lemma → Hebrew term resolver.

    Combines a *local* map (extracted from the loaded pieces) with a *global*
    map (built from all XML files) so that even lemmas that appear only in
    other books can be resolved.
    """

    def __init__(self, df_a: pd.DataFrame, df_b: pd.DataFrame,
                 global_map: Optional[dict] = None):
        mp: Dict[str, str] = {}
        for df_ in (df_a, df_b):
            for le, te in zip(df_['lemma'].astype(str), df_['term'].astype(str)):
                if le not in mp:
                    mp[le] = te
        self.local = mp
        self.global_map = global_map or {}
        self.keys_global = set(self.global_map.keys())

    # -- variant generators --------------------------------------------------

    @staticmethod
    def _variants(le: str):
        for suf in ['a', 'b', 'c', 'd', 'e', 'l', 'm']:
            yield f"{le} {suf}"
        for pre in ['a', 'b', 'c', 'd', 'e', 'k', 'l', 'm']:
            yield f"{pre}/{le}"

    # -- public API -----------------------------------------------------------

    def lemma_to_term(self, lemma: str) -> str:
        le = str(lemma)
        # N-gram features are space-separated. Convert each token and join.
        if ' ' in le:
            parts = []
            for tok in le.split():
                if re.match(r'^[\[<].+[\]>]$', tok):
                    parts.append(tok)
                else:
                    parts.append(self._lemma_to_term_single(tok))
            return ' '.join(parts)
        return self._lemma_to_term_single(le)

    def _lemma_to_term_single(self, le: str) -> str:
        """Map a single lemma code to Hebrew term."""
        if len(le) == 1 and le in PREFIX_HEB:
            return PREFIX_HEB[le]
        if le in self.local:
            return self.local[le]
        for key in self._variants(le):
            if key in self.local:
                return self.local[key]
        if le in self.global_map:
            return self.global_map[le]
        for key in self._variants(le):
            if key in self.global_map:
                return self.global_map[key]
        if le.isdigit():
            for k in self.keys_global:
                if k.endswith(f"/{le}"):
                    return self.global_map[k]
        return le


# ---------------------------------------------------------------------------
# TwoCorporaBibleLoader
# ---------------------------------------------------------------------------

class TwoCorporaBibleLoader:
    """Loads and processes two Bible-text corpora for comparison.

    Parameters
    ----------
    books_path : str
        Path to the directory containing OSHB ``*.xml`` files.
    cache_dir : str
        Directory for piece-level CSV caches, presets, and selections.
    catalog_path : str, optional
        Path to the reference-data catalog CSV (for catalog-based presets).
    """

    def __init__(self, books_path: str, cache_dir: str,
                 catalog_path: Optional[str] = None):
        self.books_path = books_path
        self.cache_dir = cache_dir
        self.catalog_path = catalog_path
        self._presets_dir = os.path.join(cache_dir, 'presets')
        self._sel_path = os.path.join(cache_dir, 'selected_pieces.json')

    # ---- Navigation --------------------------------------------------------

    def list_books(self) -> List[str]:
        return sorted(OSHBData.list_books(self.books_path))

    def list_chapters(self, book: str) -> List[int]:
        return OSHBData.list_chapters(self.books_path, book)

    def list_verses(self, book: str, chapter: int) -> List[int]:
        return OSHBData.list_verses(self.books_path, book, chapter)

    # ---- Piece loading -----------------------------------------------------

    def load_pieces(self, pieces: List[dict]) -> pd.DataFrame:
        """Load raw token data for the given pieces (with disk caching)."""
        return OSHBData.read_pieces_static(
            self.books_path, pieces, cache_dir=self.cache_dir,
        )

    # ---- POS-code extraction -----------------------------------------------

    def extract_pos_codes(self, pieces_a: List[dict],
                          pieces_b: List[dict]) -> List[str]:
        """Return sorted POS codes found in the morphology of both corpora.

        OSHB morph strings start with a language indicator (``H`` = Hebrew,
        ``A`` = Aramaic) followed by POS-code segments separated by ``/``.
        We skip the language indicator and extract the 1–2 character POS code
        from each segment.
        """
        df_a = self.load_pieces(pieces_a)
        df_b = self.load_pieces(pieces_b)
        df = pd.concat([df_a, df_b], ignore_index=True)
        codes: set = set()
        for m in df['morph'].dropna().astype(str).tolist():
            for code in re.findall(r'(?:^[HA]|/)([A-Za-z]{1,2})', m):
                codes.add(code)
        return sorted(codes)

    # ---- Full processing pipeline ------------------------------------------

    def process_corpora(self, pieces_a: List[dict], pieces_b: List[dict],
                        opts: ProcessingOptions) -> CorpusData:
        """Load, de-duplicate, process, and build vocabulary for two corpora.

        Returns a ``CorpusData`` bundle ready for downstream HC analysis and
        for display.
        """
        df_a = self.load_pieces(pieces_a)
        df_b = self.load_pieces(pieces_b)

        if not df_a.empty:
            df_a = df_a.drop_duplicates(subset=['verse', 'lemma', 'morph', 'term'])
        if not df_b.empty:
            df_b = df_b.drop_duplicates(subset=['verse', 'lemma', 'morph', 'term'])

        df_a = df_a.assign(author='A')
        df_b = df_b.assign(author='B')
        raw_ab = pd.concat([df_a, df_b], ignore_index=True)

        pt = ProcessText(
            extract_prefix=opts.extract_prefix,
            extract_suffix=opts.extract_suffix,
            ng_range=opts.ng_range,
            pad=opts.pad,
            to_remove=opts.to_remove,
            to_replace=opts.to_replace,
        )

        # Unigram data — always needed for word-level display
        processed = pt.proc(raw_ab).copy()
        processed['doc_id'] = processed['chapter']

        # N-gram data — used for vocabulary building and HC comparison.
        # When ng_range is (1, 1) this is just the unigram data; otherwise
        # proc_ng produces the full n-gram expansion whose features are
        # tuples.  We stringify them so they work as a plain pd.Index
        # (avoiding an accidental MultiIndex in CompareDocs).
        use_ngrams = opts.ng_range[1] > 1
        if use_ngrams:
            ng_data = pt.proc_ng(raw_ab).copy()
            ng_data['doc_id'] = ng_data['chapter']
            ng_data['feature'] = ng_data['feature'].apply(
                lambda f: ' '.join(f) if isinstance(f, tuple) else str(f)
            )
        else:
            ng_data = processed

        # Exclude removed features ([code]) from vocabulary and counts.
        # They stay in `processed` so the display layer can render them as
        # gray/ignored, but they must not participate in the HC comparison.
        # For n-grams, also exclude any feature that contains a removed token
        # (e.g. "430 [Np]" or "[Np] 1060" when Np is ignored).
        def _has_removed_token(feat: str) -> bool:
            s = str(feat)
            if re.match(r'^\[.+\]$', s):
                return True
            if ' ' in s:
                for tok in s.split():
                    if re.match(r'^\[.+\]$', tok):
                        return True
            return False

        _removed_mask = ng_data['feature'].astype(str).apply(_has_removed_token)
        countable = ng_data[~_removed_mask]

        # Vocabulary: union of top-n most-frequent features per corpus
        terms = (
            countable.groupby(['author', 'feature'])
            .size()
            .reset_index(name='_cnt')
            .sort_values('_cnt', ascending=False)
            .groupby('author')
            .head(opts.n_words)
        )
        vocab = terms.feature.unique().tolist()

        ds_counts = (
            countable
            .filter(['author', 'feature', 'doc_id'])
            .rename(columns={'author': 'cls'})
        )

        return CorpusData(
            processed=processed,
            ng_processed=ng_data,
            vocab=vocab,
            pt=pt,
            df_a_raw=df_a,
            df_b_raw=df_b,
            data_a=processed[processed.author == 'A'].reset_index(),
            data_b=processed[processed.author == 'B'].reset_index(),
            counts_a=ds_counts[ds_counts.cls == 'A'],
            counts_b=ds_counts[ds_counts.cls == 'B'],
        )

    # ---- Lemma mapping -----------------------------------------------------

    def build_lemma_mapper(self, df_a: pd.DataFrame,
                           df_b: pd.DataFrame) -> LemmaMapper:
        """Build a ``LemmaMapper`` from local piece data + global XML scan."""
        global_map = OSHBData.build_lemma_term_dict(
            self.books_path, cache_dir=self.cache_dir,
        )
        return LemmaMapper(df_a, df_b, global_map)

    # ---- Overlap detection -------------------------------------------------

    def detect_overlaps(self, pieces: List[dict]) -> List[Tuple[str, int]]:
        """Return ``(verse_id, count)`` pairs for any verse appearing > 1 time."""
        verse_counts: Dict[str, int] = {}
        for p in pieces:
            book = p['book']
            chp = int(p['chapter'])
            if p.get('verses'):
                vs = {f"{book}.{chp}.{v}" for v in p['verses']}
            else:
                vs = {f"{book}.{chp}.{v}" for v in self.list_verses(book, chp)}
            for v in vs:
                verse_counts[v] = verse_counts.get(v, 0) + 1
        return [(v, c) for v, c in verse_counts.items() if c > 1]

    # ---- Selection persistence ---------------------------------------------

    def load_selections(self) -> dict:
        """Load saved A/B piece selections from disk (or return empty)."""
        try:
            if os.path.exists(self._sel_path):
                with open(self._sel_path, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {'A': [], 'B': []}

    def save_selections(self, pieces_a: List[dict],
                        pieces_b: List[dict]) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self._sel_path, 'w') as f:
            json.dump({'A': pieces_a, 'B': pieces_b}, f)

    # ---- Preset management -------------------------------------------------

    def list_presets(self) -> List[str]:
        os.makedirs(self._presets_dir, exist_ok=True)
        return [
            os.path.splitext(f)[0]
            for f in os.listdir(self._presets_dir)
            if f.endswith('.json')
        ]

    def load_preset(self, name: str) -> dict:
        with open(os.path.join(self._presets_dir, name + '.json'), 'r') as f:
            return json.load(f)

    def save_preset(self, name: str, pieces_a: List[dict],
                    pieces_b: List[dict],
                    params: Optional[dict] = None) -> None:
        os.makedirs(self._presets_dir, exist_ok=True)
        obj: dict = {'A': pieces_a, 'B': pieces_b}
        if params:
            obj['params'] = params
        with open(os.path.join(self._presets_dir, name + '.json'), 'w') as f:
            json.dump(obj, f)

    def delete_preset(self, name: str) -> None:
        os.remove(os.path.join(self._presets_dir, name + '.json'))

    # ---- Catalog-based presets ---------------------------------------------

    def list_catalog_authors(self) -> List[str]:
        """Return sorted unique author codes from the catalog CSV."""
        if not self.catalog_path or not os.path.isfile(self.catalog_path):
            return []
        df = pd.read_csv(self.catalog_path)
        if 'author' not in df.columns:
            return []
        authors = [
            a for a in df['author'].astype(str).unique().tolist()
            if a and a.lower() != 'nan' and not a.lstrip()[0:1].isdigit()
        ]
        return sorted(authors)

    def load_catalog_preset(self, author: str) -> List[dict]:
        """Build a pieces list from the reference-data catalog for *author*."""
        if not self.catalog_path:
            raise ValueError("No catalog path configured")
        df = _read_catalog(self.catalog_path)
        df = df[df['author'] == author]
        pieces: List[dict] = []
        for (book, chapter), sub in df.groupby(['book', 'chapter']):
            verses_list = sub['verses'].astype(str).tolist()
            if any(v.lower() == 'all' for v in verses_list):
                pieces.append({
                    'book': str(book), 'chapter': int(chapter), 'verses': [],
                })
            else:
                vs_set: set = set()
                for v in verses_list:
                    v = v.strip()
                    if '-' in v:
                        a, b = v.split('-')
                        try:
                            a, b = int(a), int(b)
                            vs_set.update(range(min(a, b), max(a, b) + 1))
                        except Exception:
                            continue
                    else:
                        try:
                            vs_set.add(int(v))
                        except Exception:
                            continue
                pieces.append({
                    'book': str(book), 'chapter': int(chapter),
                    'verses': sorted(vs_set),
                })
        return pieces
