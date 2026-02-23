import logging
import base64
import re
import os
import sys

import numpy as np
# NumPy compatibility for older downstream libs (e.g., Streamlit pre-1.0)
for _alias, _typ in [('bool', np.bool_), ('int', int), ('float', float), ('object', object)]:
    if not hasattr(np, _alias):
        try:
            setattr(np, _alias, _typ)
        except Exception:
            pass
import pandas as pd
from typing import List
import streamlit as st
import matplotlib.pyplot as plt
import io
from matplotlib import font_manager as _fm

# ensure local imports work when run via `streamlit run restored_app/app.py`
sys.path.append(os.path.dirname(__file__))
# Support vendored third-party packages under restored_app/.deps
_deps = os.path.join(os.path.dirname(__file__), '.deps')
if os.path.isdir(_deps) and _deps not in sys.path:
    sys.path.insert(0, _deps)

from two_corpora_bible_loader import (
    TwoCorporaBibleLoader, ProcessingOptions, LemmaMapper,
)
from two_corpora_hc_analysis import TwoCorporaHCAnalysis


# ---------------------------------------------------------------------------
# Constants (display / legacy)
# ---------------------------------------------------------------------------

KNOWN_AUTHORS = ['P', 'Dtr', 'DtrH']

COLOR_dic = {'Dtr': '#FF0000', 'DtrH': '#008000', 'P': '#2554C7'}

STYLES = {
    'normal': {'color': 'black', 'bg': 'white', 'fr': 'white'},
    'oov': {'color': 'gray', 'bg': 'white', 'fr': 'white'},
    'replaced': {'color': 'orange', 'bg': 'white', 'fr': 'black'},
    'removed': {'color': 'gray', 'bg': 'white', 'fr': 'gray'},
    'P': {'color': 'blue'},
    'Dtr': {'color': 'red'},
    'DtrH': {'color': 'green'},
}

PREFIX_HEB = {'c': 'ו', 'd': 'ה', 'b': 'ב', 'l': 'ל', 'k': 'כ'}

# Human-readable names for POS codes
CODE_TO_NAME = {
    'Np': 'Np (proper name)', 'Nc': 'Nc (common noun)',
    'Ng': 'Ng (gentilic noun)', 'N': 'N (noun)',
    'Ac': 'Ac (cardinal number)', 'Ao': 'Ao (ordinal number)',
    'A': 'A (adjective)', 'Aa': 'Aa (adjective)',
    'V': 'V (verb stem)', 'P': 'P (pronoun)',
    'R': 'R (preposition)', 'C': 'C (conjunction)', 'D': 'D (adverb)',
    'Rd': 'Rd (definite article)',
    'S': 'S (suffix)', 'Sp': 'Sp (suffix pronominal)',
    'Sd': 'Sd (suffix directional)', 'Sa': 'Sa (suffix accusative)',
    'Sh': 'Sh (suffix paragogic)', 'Sn': 'Sn (paragogic nun)',
    'Ta': 'Ta (particle affirmation)', 'Td': 'Td (definite article)',
    'Te': 'Te (exhortation)', 'Ti': 'Ti (particle interrogative)',
    'Tj': 'Tj (particle interjection)', 'T': 'T (particle)',
    'To': 'To (direct object marker)', 'Tn': 'Tn (particle negative)',
    'Tm': 'Tm (particle demonstrative)', 'Tr': 'Tr (particle relative)',
    'Pd': 'Pd (pronoun demonstrative)', 'Pp': 'Pp (pronoun personal)',
    'VH': 'VH (verb hophal)', 'Vh': 'Vh (verb haphel)',
    'VN': 'VN (verb niphal)', 'VP': 'VP (verb pual)',
    'Vp': 'Vp (verb piel)', 'Vv': 'Vv (verb imperative)',
    'Vo': 'Vo (verb polel)', 'Vq': 'Vq (verb qal)',
    'Vt': 'Vt (verb hithpael)',
    'Vm': 'Vm (verb poel)', 'Vl': 'Vl (verb pilpel)',
}

POS_PARENT_NAMES = {
    'A': 'Adjective', 'C': 'Conjunction', 'D': 'Adverb',
    'N': 'Noun', 'P': 'Pronoun', 'R': 'Preposition',
    'S': 'Suffix', 'T': 'Particle', 'V': 'Verb',
}


def _pos_short_desc(code: str) -> str:
    """Extract the parenthesised description, e.g. 'proper name' from 'Np (proper name)'."""
    full = CODE_TO_NAME.get(code, '')
    m = re.match(r'\w+ \((.+)\)', full)
    return m.group(1) if m else ''


# ---------------------------------------------------------------------------
# Path helpers (self-contained under restored_app)
# ---------------------------------------------------------------------------

def _app_base():
    """Directory containing this app (restored_app), for self-contained deployment."""
    return os.path.dirname(os.path.abspath(__file__))


def _default_paths():
    base = _app_base()
    books = os.path.normpath(os.path.join(base, 'data', 'morphhb', 'wlc'))
    catalog = os.path.normpath(os.path.join(base, 'data', 'reference_data.csv'))
    return books, catalog


def _default_paths_relative():
    """Relative paths (from app base) for display."""
    return 'data/morphhb/wlc', 'data/reference_data.csv'


def _resolve_data_path(p: str) -> str:
    """Convert relative path to absolute; pass through if already absolute."""
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(_app_base(), p))


def _to_relative_path(p: str) -> str:
    """Convert absolute path to relative (from app base) for display."""
    base = _app_base()
    p = os.path.normpath(p)
    if p.startswith(base):
        rel = os.path.relpath(p, base)
        return rel.replace(os.sep, '/')
    return p


def _cache_dir():
    """Cache directory inside restored_app."""
    return os.path.join(_app_base(), 'cache')


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def html_color_text(txt, color='black', bg='white', fr=None):
    if fr:
        return (
            f"<span style='color:{color};background-color:{bg}; "
            f"border:3px; border-style:solid; border-color:{fr}'>{txt}</span>"
        )
    return f"<font color='{color}' style='background-color:{bg}'>{txt}</font>"


def rgb_for_scores(scores: dict, labels: List[str]):
    max_score = 20.0
    vals = [float(scores.get(lbl, 0.0)) for lbl in labels[:3]]
    if len(vals) == 2:
        vals = [vals[0], 0.0, vals[1]]
    elif len(vals) == 1:
        vals = [vals[0], 0.0, 0.0]
    rgb = [min(int(256 * (v / max_score)), 256) for v in vals[:3]]
    return tuple(rgb)


def download_link(object_to_download, download_filename, download_link_text):
    object_to_download = object_to_download.to_csv(index=False)
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return (
        f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">'
        f'{download_link_text}</a>'
    )


# ---------------------------------------------------------------------------
# Compatibility helpers for older Streamlit versions
# ---------------------------------------------------------------------------

def compat_caption(text: str):
    if hasattr(st, 'caption'):
        st.caption(text)
    else:
        st.markdown(f"<small>{text}</small>", unsafe_allow_html=True)


def download_df_button(label: str, df: pd.DataFrame, filename: str):
    csv = df.to_csv(index=False)
    if hasattr(st, 'download_button'):
        st.download_button(label=label, data=csv.encode('utf-8'),
                           file_name=filename, mime='text/csv')
    else:
        import base64 as _b64
        b64 = _b64.b64encode(csv.encode()).decode()
        st.markdown(
            f'<a href="data:text/csv;base64,{b64}" download="{filename}">'
            f'{label}</a>',
            unsafe_allow_html=True,
        )


def columns_compat(spec):
    if hasattr(st, 'columns'):
        return st.columns(spec)
    if hasattr(st, 'beta_columns'):
        try:
            return st.beta_columns(spec)
        except Exception:
            return st.beta_columns(len(spec) if not isinstance(spec, int) else spec)
    class _DummyCol:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    n = spec if isinstance(spec, int) else len(spec)
    return [_DummyCol() for _ in range(n)]


def expander_compat(label: str):
    if hasattr(st, 'expander'):
        return st.expander(label)
    if hasattr(st, 'beta_expander'):
        return st.beta_expander(label)
    class _DummyExp:
        def __init__(self, label): self.label = label
        def __enter__(self):
            st.markdown(f"**{self.label}**")
            return self
        def __exit__(self, *a): return False
    return _DummyExp(label)


_SESSION_FALLBACK: dict = {}

def get_session_state():
    return getattr(st, 'session_state', _SESSION_FALLBACK)


# ---------------------------------------------------------------------------
# Generic term-display helper (works with any object that has lemma_to_term)
# ---------------------------------------------------------------------------

def lemma_to_term(dt, lemma):
    try:
        return dt.lemma_to_term(lemma)
    except Exception:
        return lemma


# ---------------------------------------------------------------------------
# HC bar-chart (display)
# ---------------------------------------------------------------------------

def show_hc_bar(df, labels: List[str]):
    vals = []
    for auth in labels:
        col = f'HC ({auth})'
        try:
            if col in df.columns and len(df) > 0:
                vals.append(float(np.round(df[col].iloc[0], 2)))
            else:
                vals.append(float('nan'))
        except Exception:
            vals.append(float('nan'))
    palette = {l: 'gray' for l in labels}
    if set(labels) == set(['A', 'B']):
        palette.update({'A': '#d62728', 'B': '#1f77b4'})
    elif set(labels) <= set(COLOR_dic.keys()):
        for l in labels:
            palette[l] = COLOR_dic.get(l, 'gray')
    fig, ax = plt.subplots(figsize=(4, max(1.2, 0.6 * len(labels))))
    ax.barh(labels, vals, color=[palette[l] for l in labels])
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v}", va='center', ha='left', fontsize=8)
    ax.set_xlabel('HC')
    ax.set_ylabel('')
    ax.set_title('HC discrepancy')
    plt.tight_layout()
    st.pyplot(fig)


# ---------------------------------------------------------------------------
# Document highlighting (display)
# ---------------------------------------------------------------------------

def print_results_generic(dt, ds, df, pt, labels: List[str],
                          show_hc: bool = False, show_table: bool = True):
    """Render highlighted biblical text using per-feature HC results.

    Parameters
    ----------
    dt : LemmaMapper (or any object with ``lemma_to_term``)
    ds : pd.DataFrame — token-level data for the corpus to display
    df : pd.DataFrame — per-feature display frame (HCT / sign / score cols)
    pt : ProcessText — (kept for API compat; no longer used internally)
    labels : list[str] — corpus labels (e.g. ``['A', 'B']``)
    """
    lo_known = labels
    try:
        if df.index.dtype != 'object':
            df = df.copy()
            df.index = df.index.astype(str)
    except Exception:
        pass

    # When prefix/suffix extraction is active, ProcessText creates multiple
    # rows per original word (prefix, main, suffix) sharing the same
    # token_id.  For the verse display we group by token_id and render
    # prefix parts + main part so that discriminating prefixes (e.g. "c")
    # can be highlighted.
    has_pow = 'POW' in ds.columns
    ds_main = ds[ds['POW'] == 'main'] if has_pow else ds

    def _display_text(rec, override: str = None):
        """Original Hebrew word for display (falls back to lemma mapping)."""
        if override is not None:
            return override
        term = rec.get('term')
        if pd.notna(term) and str(term).strip():
            return str(term)
        return lemma_to_term(dt, rec.get('feature (org)', rec.get('feature', '')))

    def print_token(rec, display_override: str = None):
        def is_out_of_vocab(feat):
            return feat not in df.index
        def is_removed(feat):
            return re.search(r"^\[.+\]$", feat) is not None
        def is_replaced(feat):
            return re.search(r"^\<.+\>$", feat) is not None

        feat = rec['feature']
        tokens = feat if isinstance(feat, (list, tuple)) else [feat]
        display = _display_text(rec, display_override)
        scores = dict(zip(lo_known, np.zeros(len(lo_known))))
        for token in tokens:
            if is_removed(token):
                return html_color_text(display, **STYLES['removed'])
            if is_replaced(token):
                return html_color_text(display, **STYLES['replaced'])
            tok = str(token)
            if is_out_of_vocab(tok):
                return html_color_text(display, **STYLES['oov'])
            r = df.loc[df.index == tok, :]
            for auth in lo_known:
                try:
                    mag = float(r[f'score ({auth})'].values[0]) if bool(r[f'HCT ({auth})'].values[0]) else 0.0
                    direction = 1.0 if float(r[f'sign ({auth})'].values[0]) > 0 else 0.0
                    val = mag * direction
                    if val > scores[auth]:
                        scores[auth] = val
                except Exception:
                    continue

        if sum(scores.values()) <= 0:
            return html_color_text(display)
        rgb = rgb_for_scores(scores, lo_known)
        bg = f"rgb{rgb}"
        return html_color_text(display, color='black', bg=bg, fr='white')

    if show_hc:
        show_hc_bar(df, lo_known)
    st.subheader('Document with highlighted words')
    st.subheader('Legend:')
    st.markdown(html_color_text("out of vocabulary", **STYLES['oov']), unsafe_allow_html=True)
    st.markdown(html_color_text("ignored", **STYLES['removed']), unsafe_allow_html=True)
    st.markdown(html_color_text("replaced by POS code", **STYLES['replaced']), unsafe_allow_html=True)

    st.subheader('Discriminating Features:')
    st.markdown(html_color_text(f"more frequent in {lo_known[0]} ", color='black', bg=f"rgb{(256, 0, 0)}"), unsafe_allow_html=True)
    st.markdown(html_color_text(f"more frequent in {lo_known[1]} ", color='black', bg=f"rgb{(0, 0, 256)}"), unsafe_allow_html=True)

    lo_chapters = ds_main.chapter.unique()
    for chapter in lo_chapters:
        ds_chapter_full = ds[ds.chapter == chapter]
        ds_chapter_main = ds_main[ds_main.chapter == chapter]
        st.subheader("Chapter: " + chapter)
        if ds_main.author.values[0] in lo_known:
            st.write(f"(this document is part of {ds_main.author.values[0]} corpus)")
        lo_verses = ds_chapter_main.verse.unique()
        for verse in lo_verses:
            ds_verse_full = ds_chapter_full[ds_chapter_full.verse == verse]
            ds_verse_main = ds_chapter_main[ds_chapter_main.verse == verse]
            vrs = html_color_text(f"[{verse}]", "gray")
            if has_pow:
                token_ids = ds_verse_main['token_id'].unique()
                for tid in token_ids:
                    rows = ds_verse_full[ds_verse_full.token_id == tid]
                    prefix_rows = rows[rows['POW'] == 'prefix']
                    main_rows = rows[rows['POW'] == 'main']
                    main_row = main_rows.iloc[0] if len(main_rows) > 0 else None
                    term = str(main_row['term']) if main_row is not None and pd.notna(main_row.get('term')) else ''
                    parts = term.split('/') if term else []
                    for i, (_, prow) in enumerate(prefix_rows.iterrows()):
                        part = parts[i] if i < len(parts) else lemma_to_term(dt, prow['feature'])
                        vrs = vrs + " " + print_token(prow, display_override=part)
                    prefix_count = len(prefix_rows)
                    main_part = '/'.join(parts[prefix_count:]) if prefix_count < len(parts) else term
                    if main_row is not None:
                        vrs = vrs + " " + print_token(main_row, display_override=main_part if prefix_count > 0 else None)
            else:
                for _, r in ds_verse_main.iterrows():
                    vrs = vrs + " " + print_token(r)
            st.markdown(f"<div class='verse-line'>" + vrs + "</div>", unsafe_allow_html=True)

    if show_table:
        st.subheader("Selected Features:")
        df['disp'] = df.index.map(lambda x: lemma_to_term(dt, x))
        for lab in lo_known:
            col = f'HCT ({lab})'
            scol = f'sign ({lab})'
            if col in df.columns and scol in df.columns:
                df.loc[~df[col], scol] = pd.NA

        mask_any = np.zeros(len(df), dtype=bool)
        for lab in lo_known:
            col = f'HCT ({lab})'
            if col in df.columns:
                mask_any |= df[col].astype(bool)
        dfp = df[mask_any]
        dfp = (
            dfp.set_index('disp')
            .sort_values('n Tot', ascending=False)
            .filter([col for col in df.columns if any(k in col for k in ['n (', 'T (', 'sign (', 'pval ('])])
        )
        st.write(dfp)


# ---------------------------------------------------------------------------
# Main ad-hoc comparison UI
# ---------------------------------------------------------------------------

def render_adhoc():
    st.markdown(
        """
        <style>
        .block-container { max-width: 1400px !important; padding-left: 3rem; padding-right: 3rem; }
        .verse-line { overflow-x: auto; white-space: nowrap; padding: 4px 0; }
        .compare-scope + div.stButton > button,
        [data-testid="stSidebar"] .compare-scope + div.stButton > button {
            background-color: #ff1744; color: #ffffff; border: none;
            padding: 0.8rem 1.6rem; font-weight: 800; font-size: 1.2rem; border-radius: 8px;
            box-shadow: 0 2px 8px rgba(255, 23, 68, 0.4);
        }
        .compare-scope + div.stButton > button:hover,
        [data-testid="stSidebar"] .compare-scope + div.stButton > button:hover { background-color: #d50000; }

        /* Blue-tinted tags for the Corpus B multiselect (override any default red) */
        .corpus-b-marker { display: none; }
        [data-testid="column"]:has(.corpus-b-marker) [data-baseweb="tag"],
        *:has(.corpus-b-marker) + * [data-baseweb="tag"],
        *:has(#corpus-b-marker) + * [data-baseweb="tag"] {
            background-color: rgba(28, 131, 225, 0.2) !important;
            border-color: rgb(28, 131, 225) !important;
        }
        [data-testid="column"]:has(.corpus-b-marker) [data-baseweb="tag"] span,
        [data-testid="column"]:has(.corpus-b-marker) [data-baseweb="tag"] path,
        *:has(.corpus-b-marker) + * [data-baseweb="tag"] span,
        *:has(.corpus-b-marker) + * [data-baseweb="tag"] path,
        *:has(#corpus-b-marker) + * [data-baseweb="tag"] span,
        *:has(#corpus-b-marker) + * [data-baseweb="tag"] path {
            color: rgb(28, 131, 225) !important;
            fill: rgb(28, 131, 225) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Select texts to compare")
    books_rel, catalog_rel = _default_paths_relative()
    if 'adhoc_books_path' not in st.session_state:
        st.session_state['adhoc_books_path'] = books_rel
    elif os.path.isabs(st.session_state['adhoc_books_path']):
        st.session_state['adhoc_books_path'] = _to_relative_path(st.session_state['adhoc_books_path'])
    if 'adhoc_catalog_path' not in st.session_state:
        st.session_state['adhoc_catalog_path'] = catalog_rel
    elif os.path.isabs(st.session_state['adhoc_catalog_path']):
        st.session_state['adhoc_catalog_path'] = _to_relative_path(st.session_state['adhoc_catalog_path'])

    books_path = _resolve_data_path(st.session_state['adhoc_books_path'])
    catalog_path = _resolve_data_path(st.session_state['adhoc_catalog_path'])

    # Instantiate the loader (all data logic delegated here)
    loader = TwoCorporaBibleLoader(books_path, _cache_dir(), catalog_path)

    if not os.path.exists(books_path):
        st.error(f"OSHB XML path not found: {books_path}")
        return
    all_books = loader.list_books()

    # -- session-state init -------------------------------------------------
    ss = get_session_state()
    if 'pieces_a' not in ss:
        ss['pieces_a'] = []
    if 'pieces_b' not in ss:
        ss['pieces_b'] = []

    # -- persistence helpers (delegated to loader) --------------------------
    def _load_saved():
        try:
            sel = loader.load_selections()
            if not ss['pieces_a'] and not ss['pieces_b']:
                ss['pieces_a'] = sel.get('A', [])
                ss['pieces_b'] = sel.get('B', [])
        except Exception as e:
            st.info(f"Could not load saved selections: {e}")

    def _save_now():
        try:
            loader.save_selections(ss['pieces_a'], ss['pieces_b'])
        except Exception as e:
            st.info(f"Could not save selections: {e}")

    _load_saved()

    # Load first preset on startup if both corpora are empty
    preset_names = loader.list_presets()
    if (not ss['pieces_a'] and not ss['pieces_b'] and preset_names
            and st.session_state.get('last_loaded_preset') != preset_names[0]):
        try:
            obj = loader.load_preset(preset_names[0])
            ss['pieces_a'] = obj.get('A', [])
            ss['pieces_b'] = obj.get('B', [])
            for k in ('pieces_a_tags', 'pieces_b_tags'):
                st.session_state.pop(k, None)
            prm = obj.get('params', {})
            if 'adhoc_include' not in prm and 'adhoc_rem' in prm:
                st.session_state['adhoc_rem'] = prm.get('adhoc_rem', [])
            for k, v in prm.items():
                st.session_state[k] = v
            st.session_state['_pos_needs_sync'] = True
            st.session_state['last_loaded_preset'] = preset_names[0]
            st.session_state['preset_sel'] = preset_names[0]
            _save_now()
            st.rerun()
        except Exception:
            pass  # Silently skip if preset load fails on startup

    # -- corpus picker UI ---------------------------------------------------
    def picker(side_label: str, state_key: str, use_cols: bool = True,
               tag_css_class: str = ''):
        st.markdown(f"**{side_label}**")

        cur = ss[state_key]
        def fmt(p):
            if not p['verses']:
                v = 'all'
            else:
                vv = sorted(set(p['verses']))
                rngs = []
                s = e = vv[0]
                for x in vv[1:]:
                    if x == e + 1:
                        e = x
                    else:
                        rngs.append((s, e))
                        s = e = x
                rngs.append((s, e))
                v = ','.join([f"{a}-{b}" if a != b else f"{a}" for a, b in rngs])
            return f"{p['book']} {p['chapter']} {v}"

        #st.write("Current items:")
        if len(cur):
            labels = [fmt(p) for p in cur]
            tags_key = f"{state_key}_tags"
            # Keep multiselect in sync with cur: if stored selection is invalid (labels changed),
            # reset it so we don't overwrite pieces with stale selection
            if tags_key in st.session_state:
                stored = set(st.session_state[tags_key])
                valid = set(labels)
                if not stored.issubset(valid) or stored != valid:
                    st.session_state[tags_key] = labels
            if tag_css_class:
                st.markdown(f'<div id="{tag_css_class}" class="{tag_css_class}"></div>',
                            unsafe_allow_html=True)
            selected = st.multiselect(
                f"Current items ({side_label})", options=labels, default=labels,
                key=tags_key,
            )
            if set(selected) != set(labels):
                new_cur = [p for p, lab in zip(cur, labels) if lab in set(selected)]
                ss[state_key] = new_cur
                _save_now()
            try:
                df_sel = loader.load_pieces(ss[state_key])
                total_verses = df_sel['verse'].nunique()
                total_lemmas = len(df_sel)
                unique_lemmas = df_sel['lemma'].nunique()
            except Exception:
                total_verses = total_lemmas = unique_lemmas = 0
            st.caption(
                f"{len(ss[state_key])} items, {total_verses} verses, "
                f"{total_lemmas} lemmas ({unique_lemmas} unique lemmas)",
            )
        else:
            compat_caption("No items yet.")

        # Catalog preset droplist — populates picker when a preset is selected
        _preset_opts = ['', 'Deuteronomy (D)', 'Deuteronomy History (DtrH)', 'Priestly (P)']
        _preset_to_author = {'Deuteronomy (D)': 'Dtr', 'Deuteronomy History (DtrH)': 'DtrH', 'Priestly (P)': 'P'}
        _preset_key = f'{state_key}_catalog_preset'
        _prev_key = f'{state_key}_catalog_preset_prev'
        _sel = st.selectbox('Presets', _preset_opts, key=_preset_key)
        if _sel and _sel in _preset_to_author:
            if st.session_state.get(_prev_key) != _sel:
                try:
                    ss[state_key] = loader.load_catalog_preset(_preset_to_author[_sel])
                    st.session_state.pop(f"{state_key}_tags", None)
                    _save_now()
                    st.session_state[_prev_key] = _sel
                    st.session_state['preset_sel'] = ''
                    st.rerun()
                except Exception as e:
                    st.caption(f"Could not load preset: {e}")
        elif not _sel:
            st.session_state[_prev_key] = ''

        with expander_compat(f"Create a new item ({side_label})"):
            book = st.selectbox(f"Book ({side_label})", all_books, key=f"{state_key}_book")
            chapters = loader.list_chapters(book) if book else []
            ch = st.selectbox(f"Ch ({side_label})", chapters, key=f"{state_key}_ch")
            verses = loader.list_verses(book, ch) if chapters else []
            vmode = st.radio(f"Verses ({side_label})", ['All', 'Range', 'Pick'], key=f"{state_key}_vmode")
            sel_verses: List[int] = []
            if vmode == 'All':
                st.write('All verses')
            elif vmode == 'Range':
                if verses:
                    vmin, vmax = verses[0], verses[-1]
                    a = st.number_input(f"Start ({side_label})", vmin, vmax, vmin, key=f"{state_key}_vstart")
                    b = st.number_input(f"End ({side_label})", vmin, vmax, vmax, key=f"{state_key}_vend")
                    if a > b:
                        a, b = b, a
                    sel_verses = list(range(int(a), int(b) + 1))
            else:
                sel = st.multiselect(f"Pick verses ({side_label})", verses, key=f"{state_key}_vsel")
                sel_verses = [int(x) for x in sel]

            if st.button(f"Add to {side_label}", key=f"{state_key}_add"):
                piece = {'book': book, 'chapter': int(ch), 'verses': sel_verses}
                ss[state_key].append(piece)
                st.session_state.pop(f"{state_key}_tags", None)
                _save_now()
                st.rerun()

        # Preview (reuse df_sel if already loaded)
        try:
            df_prev = df_sel.copy()  # noqa: F821 — may be unbound when cur==[]
        except Exception:
            try:
                df_prev = loader.load_pieces(ss[state_key])
            except Exception:
                df_prev = pd.DataFrame()
        with expander_compat(f'Preview {side_label} (by verse)'):
            if df_prev is None or df_prev.empty:
                st.info(f'Collection {side_label} is empty')
            else:
                for chap, dch in df_prev.groupby('chapter'):
                    st.write(f"Chapter {chap}")
                    for v, dv in dch.groupby('verse'):
                        line = ' '.join(dv['term'].astype(str).tolist())
                        st.write(f"[{v}] {line}")

    colA, colB = columns_compat([1, 1])
    with colA:
        #st.markdown('### Corpus A')
        picker('Corpus A', 'pieces_a', use_cols=False)
    with colB:
        #st.markdown('### Corpus B')
        picker('Corpus B', 'pieces_b', use_cols=False, tag_css_class='corpus-b-marker')

    # -- Instructions + Compare + Citations (top of sidebar) -----------------
    with st.sidebar:
        st.markdown("<div class='compare-scope'></div>", unsafe_allow_html=True)
        run_adhoc = st.button('Compare', key='compare_btn')
        st.markdown('---')
        st.markdown("**Instructions:**")
        st.markdown("- Form two corpora (A and B) by selecting book/chapter/verses, or use the presets.")
        st.markdown("- Click 'Compare' to compare the two corpora.")
        st.markdown("- Adjust the parameters on the sidebar below to customize the comparison.")
        
    # -- file-based preset management (sidebar) -----------------------------
    st.sidebar.markdown('---')
    st.sidebar.subheader('Presets (Save/Load)')
    preset_names = loader.list_presets()
    with st.sidebar:
        _col_sel, _col_del = st.columns([3, 1])
        preset_to_load = _col_sel.selectbox('Available presets', [''] + preset_names, key='preset_sel')
        _col_del.markdown('<div style="height:1.6rem"></div>', unsafe_allow_html=True)
        delete_clicked = _col_del.button('\U0001F5D1\uFE0F', key='delete_preset_btn', help='Delete preset')

        _col_name, _col_save = st.columns([3, 1])
        preset_name = _col_name.text_input('Save as (name)', key='preset_name')
        _col_save.markdown('<div style="height:1.6rem"></div>', unsafe_allow_html=True)
        save_clicked = _col_save.button('\U0001F4BE', key='save_preset_btn', help='Save preset')

    if 'last_loaded_preset' not in st.session_state:
        st.session_state['last_loaded_preset'] = ''
    if preset_to_load and st.session_state['last_loaded_preset'] != preset_to_load:
        try:
            obj = loader.load_preset(preset_to_load)
            ss['pieces_a'] = obj.get('A', [])
            ss['pieces_b'] = obj.get('B', [])
            # Clear multiselect widget state so pickers show the new pieces (not stale values)
            for k in ('pieces_a_tags', 'pieces_b_tags'):
                st.session_state.pop(k, None)
            prm = obj.get('params', {})
            if 'adhoc_include' not in prm and 'adhoc_rem' in prm:
                st.session_state['adhoc_rem'] = prm.get('adhoc_rem', [])
            for k, v in prm.items():
                st.session_state[k] = v
            st.session_state['_pos_needs_sync'] = True
            st.session_state['last_loaded_preset'] = preset_to_load
            _save_now()
            st.sidebar.success(f"Loaded preset '{preset_to_load}'.")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed to load preset: {e}")

    if save_clicked and preset_name:
        try:
            params_obj = {
                'adhoc_n': st.session_state.get('adhoc_n', 3000),
                'adhoc_min': st.session_state.get('adhoc_min', 3),
                'adhoc_minng': st.session_state.get('adhoc_minng', 1),
                'adhoc_maxng': st.session_state.get('adhoc_maxng', 1),
                'adhoc_prefix': st.session_state.get('adhoc_prefix', 'extract'),
                'adhoc_suffix': st.session_state.get('adhoc_suffix', 'keep'),
                'adhoc_include': st.session_state.get('adhoc_include', []),
                'adhoc_rep': st.session_state.get('adhoc_rep', []),
                'adhoc_gamma': st.session_state.get('adhoc_gamma', 0.35),
                'adhoc_books_path': st.session_state.get('adhoc_books_path', books_rel),
                'adhoc_catalog_path': st.session_state.get('adhoc_catalog_path', catalog_rel),
            }
            loader.save_preset(preset_name, ss['pieces_a'], ss['pieces_b'], params_obj)
            st.sidebar.success(f"Saved preset '{preset_name}'.")
        except Exception as e:
            st.sidebar.error(f"Failed to save preset: {e}")

    if delete_clicked and preset_to_load:
        try:
            loader.delete_preset(preset_to_load)
            st.sidebar.success(f"Deleted preset '{preset_to_load}'.")
        except Exception as e:
            st.sidebar.error(f"Failed to delete preset: {e}")

    # -- Swap A/B -----------------------------------------------------------
    if st.button('Swap A/B'):
        ss['pieces_a'], ss['pieces_b'] = (ss['pieces_b'], ss['pieces_a'])
        _save_now()
        st.success('Swapped A and B collections.')

    # -- parameters sidebar -------------------------------------------------
    st.sidebar.header('Parameters')
    n_words = st.sidebar.number_input('Maximal number of features per collection', 10, 5000, 3000, key='adhoc_n')
    min_cnt = st.sidebar.number_input('Ignore n-grams appearing less than', 0, 100, 3, key='adhoc_min')
    min_ng = st.sidebar.number_input('Min n-gram', 1, 5, 1, key='adhoc_minng')
    max_ng = st.sidebar.number_input('Max n-gram', 1, 5, 1, key='adhoc_maxng')
    with st.sidebar:
        _lp, _rp = st.columns([1, 2])
        _lp.markdown('<div style="line-height:2.4rem">Prefix</div>', unsafe_allow_html=True)
        prefix_mode = _rp.radio('Prefix', ['extract', 'keep'], index=0, key='adhoc_prefix',
                                horizontal=True, label_visibility='collapsed')
        _ls, _rs = st.columns([1, 2])
        _ls.markdown('<div style="line-height:2.4rem">Suffix</div>', unsafe_allow_html=True)
        suffix_mode = _rs.radio('Suffix', ['extract', 'keep'], index=1, key='adhoc_suffix',
                                horizontal=True, label_visibility='collapsed')

    # POS codes
    try:
        avail_codes = loader.extract_pos_codes(
            ss.get('pieces_a', []), ss.get('pieces_b', []),
        )
    except Exception:
        avail_codes = sorted(['Np', 'Nc', 'Ng', 'Ac', 'V', 'S', 'Sp', 'R', 'C', 'D', 'T', 'P', 'A'])

    # Group codes by first letter (parent category)
    pos_groups: dict = {}
    for _c in sorted(avail_codes):
        pos_groups.setdefault(_c[0], []).append(_c)

    # Determine initial per-code default from legacy adhoc_include / adhoc_rep
    _inc_init = st.session_state.get('adhoc_include', None)
    if _inc_init is None:
        _rem_init = st.session_state.get('adhoc_rem', None)
        if _rem_init is not None:
            _inc_init = [c for c in avail_codes if c not in _rem_init]
        else:
            _inc_init = [c for c in avail_codes if c not in ['Np', 'Ng']]
    _inc_set = set(c for c in (_inc_init or []) if c in avail_codes)
    _rep_set = set(c for c in st.session_state.get('adhoc_rep', []) if c in avail_codes)

    # Sync pos_ radio keys on first render or after a preset load
    _pos_needs_sync = (
        not any(f'pos_{c}' in st.session_state for c in avail_codes)
        or st.session_state.pop('_pos_needs_sync', False)
    )
    if _pos_needs_sync:
        for _c in avail_codes:
            if _c in _rep_set:
                st.session_state[f'pos_{_c}'] = 'POS code only'
            elif _c in _inc_set:
                st.session_state[f'pos_{_c}'] = 'keep'
            else:
                st.session_state[f'pos_{_c}'] = 'ignore'

    # Render hierarchical POS radio table
    with st.sidebar.expander('Parts of speech', expanded=False):
        _ph, _rh = st.columns([5, 7])
        _ph.markdown('**POS**')
        _rh.markdown(
            '<div style="font-size:.75rem;color:#888;padding-top:.15rem">'
            'keep &nbsp;&bull;&nbsp; replace &nbsp;&bull;&nbsp; ignore</div>',
            unsafe_allow_html=True,
        )
        for _parent_letter in sorted(pos_groups):
            _codes = pos_groups[_parent_letter]
            _children = [c for c in _codes if len(c) > 1]
            _parent_in_list = _parent_letter in _codes

            if _children:
                # Group with subcategories
                _pname = POS_PARENT_NAMES.get(_parent_letter, _parent_letter)
                if _parent_in_list:
                    _lc, _rc = st.columns([5, 7])
                    _lc.markdown(f'**{_pname}** – {_parent_letter}')
                    _rc.radio(_parent_letter, ['keep', 'POS code only', 'ignore'],
                              key=f'pos_{_parent_letter}', horizontal=True,
                              label_visibility='collapsed')
                else:
                    st.markdown(f'**{_pname}**')
                for _code in _children:
                    _desc = _pos_short_desc(_code)
                    _lbl = (f'&nbsp;&nbsp;{_desc} ' if _desc else '') + f'({_code})'
                    _lc, _rc = st.columns([5, 7])
                    _lc.markdown(_lbl, unsafe_allow_html=True)
                    _rc.radio(_code, ['keep', 'POS code only', 'ignore'],
                              key=f'pos_{_code}', horizontal=True,
                              label_visibility='collapsed')
            elif _parent_in_list:
                # Single standalone code (no subcategories)
                _desc = _pos_short_desc(_parent_letter)
                _lbl = (f'**{_desc}** ' if _desc else '') + f'({_parent_letter})'
                _lc, _rc = st.columns([5, 7])
                _lc.markdown(_lbl)
                _rc.radio(_parent_letter, ['keep', 'POS code only', 'ignore'],
                          key=f'pos_{_parent_letter}', horizontal=True,
                          label_visibility='collapsed')
        st.caption('[Code reference](https://hb.openscriptures.org/parsing/HebrewMorphologyCodes.html)')

    # Derive include / replace lists from per-code radio values
    include_POS = []
    replace_POS = []
    for _c in avail_codes:
        _state = st.session_state.get(f'pos_{_c}', 'keep')
        if _state == 'keep':
            include_POS.append(_c)
        elif _state == 'POS code only':
            replace_POS.append(_c)
    st.session_state['adhoc_include'] = include_POS
    st.session_state['adhoc_rep'] = replace_POS
    gamma = st.sidebar.slider('gamma (HC lower fraction)', 0.1, 0.45, 0.3, key='adhoc_gamma')

    # -- Data source (bottom of sidebar) -------------------------------------
    st.sidebar.markdown('---')
    st.sidebar.subheader('Data source')
    st.sidebar.text_input('OSHB XML path', value=st.session_state.get('adhoc_books_path', books_rel), key='adhoc_books_path')
    st.sidebar.text_input('Catalog CSV path (for known and disputed authorship presets)', value=st.session_state.get('adhoc_catalog_path', catalog_rel), key='adhoc_catalog_path')

    if run_adhoc:
        with st.spinner('Loading selected pieces and comparing...'):
            pieces_a = ss['pieces_a']
            pieces_b = ss['pieces_b']
            if len(pieces_a) == 0 or len(pieces_b) == 0:
                st.error('Please specify at least one item in each collection.')
                return

            # Overlap detection (UI only — loader provides the data)
            for label, pieces in [('A', pieces_a), ('B', pieces_b)]:
                try:
                    overlaps = loader.detect_overlaps(pieces)
                    if overlaps:
                        total_overlaps = sum(c - 1 for _, c in overlaps)
                        st.warning(
                            f"Detected overlapping verses across selected items "
                            f"in {label} (overlaps counted: {total_overlaps}). "
                            f"Overlaps will not be double-counted in comparisons.",
                        )
                        with expander_compat(f"Show overlap details ({label})"):
                            df_ov = pd.DataFrame(overlaps, columns=['verse', 'count']).sort_values('verse')
                            st.dataframe(df_ov, width='stretch')
                            download_df_button(f'Download overlap ({label})', df_ov, f'overlap_{label}.csv')
                except Exception:
                    pass

            # ---- DATA LOADING (TwoCorporaBibleLoader) ---------------------
            to_remove_codes = [c for c in avail_codes if c not in include_POS and c not in replace_POS]
            opts = ProcessingOptions(
                extract_prefix=(prefix_mode == 'extract'),
                extract_suffix=(suffix_mode == 'extract'),
                ng_range=(int(min_ng), int(max_ng)),
                pad=False,
                to_remove=to_remove_codes,
                to_replace=replace_POS,
                n_words=int(n_words),
            )
            corpus = loader.process_corpora(pieces_a, pieces_b, opts)
            _save_now()

            # ---- HC ANALYSIS (TwoCorporaHCAnalysis) -----------------------
            if not corpus.vocab:
                st.error(
                    "Vocabulary is empty. This can happen if: (1) the OSHB XML path or catalog path "
                    "is incorrect (check Data source in the sidebar); (2) POS filtering removed all "
                    "features—try relaxing the Parts of speech settings; (3) the selected corpora have "
                    "no loadable text. Please verify the data paths and try again."
                )
                return

            analysis = TwoCorporaHCAnalysis(corpus.vocab, min_count=int(min_cnt))
            analysis.fit(corpus.counts_a, corpus.counts_b)
            result = analysis.compare_global(gamma=float(gamma))

            st.subheader('Comparison results')
            st.write(f"HC between Corpus A and Corpus B: {result.hc_between:.3f}")

            # Per-document leave-one-out HC scatter
            try:
                doc_hc = analysis.compare_per_document(corpus.ng_processed, gamma=float(gamma))
                df_piv = doc_hc.pivot_table(
                    index=['doc', 'of'], columns='vs', values='HCmax',
                ).reset_index()
                df_piv = df_piv.rename(columns={'A': 'HC_A', 'B': 'HC_B'})

                data2_idx = corpus.processed.reset_index()
                sizes_df = data2_idx.groupby(['doc_id', 'author']).size().reset_index(name='lemmas')
                verse_df = data2_idx.groupby(['doc_id', 'author'])['verse'].nunique().reset_index(name='verses')
                df_plot = df_piv.merge(sizes_df, left_on=['doc', 'of'], right_on=['doc_id', 'author'], how='left')
                df_plot = df_plot.merge(verse_df, left_on=['doc', 'of'], right_on=['doc_id', 'author'], how='left')
                df_plot = df_plot[df_plot['verses'].fillna(0) >= 5]

                max_lem = float(df_plot['lemmas'].max()) if 'lemmas' in df_plot and df_plot['lemmas'].notna().any() else 1.0
                sizes = 40.0 + 160.0 * (df_plot['lemmas'].fillna(1.0) / max_lem)
                colors = df_plot['of'].map({'A': '#d62728', 'B': '#1f77b4'}).fillna('gray')

                fig2, ax2 = plt.subplots(figsize=(5, 5))
                ax2.scatter(df_plot['HC_A'], df_plot['HC_B'], c=colors, s=sizes)
                for _, r in df_plot.iterrows():
                    ax2.text(r['HC_A'], r['HC_B'], str(r['doc']), fontsize=7, ha='left', va='bottom')
                ax2.set_xlabel('HC(doc vs A corpus; LOO if doc∈A)')
                ax2.set_ylabel('HC(doc vs B corpus; LOO if doc∈B)')
                ax2.set_title('Per-item HC (doc vs corpus)')
                ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig2)
                st.caption('Only items with at least 5 verses are shown in the plot.')
            except Exception as e:
                st.info(f"Could not compute LOO HC scatter: {e}")

            # ---- PRESENTATION (uses loader's LemmaMapper) -----------------
            mapper = loader.build_lemma_mapper(corpus.df_a_raw, corpus.df_b_raw)

            def add_term_column(df_in: pd.DataFrame) -> pd.DataFrame:
                out = df_in.copy()
                out = out.assign(feature=out.index)
                def _to_display_term(feat: str) -> str:
                    s = str(feat)
                    m = re.match(r'^(?P<pref>(?:[a-z]/)+)?(?P<base>\d+)(?:\s+[a-z])?$', s)
                    if m:
                        return lemma_to_term(mapper, m.group('base'))
                    return lemma_to_term(mapper, s)
                out['term'] = out['feature'].apply(_to_display_term)
                return out

            dfres = result.per_feature

            # Discriminating terms table
            st.subheader('Discriminating terms')
            df_all = dfres[dfres['thresh']].copy()
            df_all['more frequent in'] = np.where(df_all['sign'] > 0, 'A', 'B')
            df_all_disp = add_term_column(df_all).rename(columns={'pval': 'pval_raw'})
            df_all_disp = df_all_disp[['term', 'pval_raw', 'more frequent in']].sort_values('pval_raw')

            def _row_highlight(row):
                color = '#ffe5e5' if row.get('more frequent in') == 'A' else '#e5efff'
                return [f'background-color: {color}'] * len(row)

            try:
                styled = df_all_disp.style.apply(_row_highlight, axis=1)
                st.dataframe(styled, width='stretch')
            except Exception:
                st.dataframe(df_all_disp, width='stretch')

            download_df_button('Download CSV (all discriminating)', df_all_disp, 'all_discriminating_features.csv')

            st.subheader('All terms (including non-discriminating)')
            try:
                df_all_terms_disp = add_term_column(dfres.copy()).rename(columns={'pval': 'pval_raw'})
                cols = [c for c in ['HC', 'thresh', 'sign'] if c in df_all_terms_disp.columns]
                df_all_terms_disp = df_all_terms_disp[['term', 'pval_raw'] + cols]
                download_df_button('Download CSV (all terms)', df_all_terms_disp, 'all_terms.csv')
            except Exception as e:
                st.info(f"Could not prepare full terms CSV: {e}")

            # Documents with highlighted words
            st.markdown(
                "_Legend: red background → more frequent in A; blue background "
                "→ more frequent in B; gray → out of vocabulary/ignored._",
            )
            st.header('Corpus A')
            print_results_generic(mapper, corpus.data_a, result.display_frame, corpus.pt, ['A', 'B'], show_hc=False, show_table=False)
            st.header('Corpus B')
            print_results_generic(mapper, corpus.data_b, result.display_frame, corpus.pt, ['A', 'B'], show_hc=False, show_table=False)


def main():
    st.title('Word-frequency Comparison of Biblical Texts')
    st.markdown(
            "Based on the method used for authorship analysis developed in [1] and [2]. "
            "The preset for authorship analysis in the bible is according to [3]."
        )
    st.markdown(
            "[1]&nbsp;&nbsp; A. Kipnis. 'Higher criticism for discriminating word-frequency tables and authorship attribution.' "
            "The Annals of Applied Statistics 16, no. 2 (2022): 1236-1252.")
    st.markdown(
            "[2]&nbsp;&nbsp; D. Donoho and A. Kipnis. 'Higher criticism to compare two large frequency tables, with sensitivity to possible rare and weak differences.' The Annals of Statistics 50, no. 3 (2022): 1447-1472.")
    st.markdown(
            "[3]&nbsp;&nbsp; S. Faigenbaum-Golovin, A. Kipnis, A. Bühler, E. Piasetzky, T. Römer, and I. Finkelstein. "
            "'Critical biblical studies via word frequency analysis: Unveiling text authorship.' Plos one 20, no. 6 (2025): e0322905.")
    st.markdown('---')
    render_adhoc()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
