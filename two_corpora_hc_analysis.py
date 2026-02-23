"""
TwoCorporaHCAnalysis — Higher-Criticism statistical analysis for comparing
two text corpora.

Responsibilities:
  - Fit a ``CompareDocs`` model on feature counts from two corpora.
  - Compute the **global** HC statistic between the two corpora and produce
    a per-feature results table with p-values, HC thresholds, and signs.
  - Compute **per-document** (leave-one-out) HC: for every document in each
    corpus, compare it against both corpora (removing it from its own).
  - Build a *display frame* with columns suitable for the UI highlighting
    logic (``HCT (A)``, ``score (A)``, ``sign (A)``, etc.).

This module is Streamlit-agnostic; all UI logic stays in app.py.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from compare import CompareDocs

try:
    from TwoSampleHC import (
        HC as HCClass,
        two_sample_binomial_test as ts_two_sample_binomial_test,
    )
except Exception as _e:
    HCClass = None
    ts_two_sample_binomial_test = None
    _TSHC_IMPORT_ERROR = _e


def _ensure_tshc():
    if HCClass is None or ts_two_sample_binomial_test is None:
        raise ImportError(
            f"TwoSampleHC is required but could not be imported: "
            f"{_TSHC_IMPORT_ERROR}"
        )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GlobalHCResult:
    """Result of a global two-corpus HC comparison.

    Attributes
    ----------
    hc_between : float
        The overall HC* statistic between corpus A and corpus B.
    per_feature : pd.DataFrame
        Output of ``CompareDocs.compare_classes()`` — one row per vocabulary
        feature with columns *pval*, *HC*, *thresh*, *sign*, plus count
        columns.
    display_frame : pd.DataFrame
        A copy of *per_feature* augmented with per-label columns
        (``HCT (A)``, ``score (A)``, ``sign (A)``, etc.) that the UI
        highlighting logic expects.
    """
    hc_between: float
    per_feature: pd.DataFrame
    display_frame: pd.DataFrame


# ---------------------------------------------------------------------------
# TwoCorporaHCAnalysis
# ---------------------------------------------------------------------------

class TwoCorporaHCAnalysis:
    """Two-sample Higher-Criticism analysis for two text corpora.

    Typical workflow::

        analysis = TwoCorporaHCAnalysis(vocab, min_count=3)
        analysis.fit(counts_a, counts_b)
        result   = analysis.compare_global(gamma=0.3)
        doc_hc   = analysis.compare_per_document(processed, gamma=0.3)

    Parameters
    ----------
    vocabulary : list[str]
        Vocabulary to consider (features not in this list are ignored).
    min_count : int
        Minimum total count for a feature to be included.
    """

    def __init__(self, vocabulary: List[str], min_count: int = 0):
        self.vocabulary = vocabulary
        self.min_count = min_count
        self.model = CompareDocs(vocabulary=vocabulary, min_count=min_count)
        self._fitted = False

    # ---- fit ---------------------------------------------------------------

    def fit(self, counts_a: pd.DataFrame, counts_b: pd.DataFrame) -> None:
        """Fit with feature DataFrames for each corpus.

        Each DataFrame must contain a ``feature`` column (and typically also
        ``cls`` and ``doc_id``).
        """
        self.model.fit({'A': counts_a, 'B': counts_b})
        self._fitted = True

    # ---- global comparison -------------------------------------------------

    def compare_global(self, gamma: float = 0.25) -> GlobalHCResult:
        """Compute the overall HC between corpora and per-feature results.

        Returns a ``GlobalHCResult`` with the scalar *hc_between*, the raw
        *per_feature* table, and a *display_frame* augmented with per-label
        columns for the highlighting UI.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before compare_global()")

        per_feature = self.model.compare_classes(gamma=gamma)

        # Scalar HC between the two full corpora
        _ensure_tshc()
        try:
            cnt_a = self.model.counts_df['n (A)'].astype(int).to_numpy()
            cnt_b = self.model.counts_df['n (B)'].astype(int).to_numpy()
            pv = ts_two_sample_binomial_test(cnt_a, cnt_b)
            hc_vals, _ = HCClass(pv, stbl=True).HCstar(gamma=gamma)
            hc_between = float(np.nanmax(np.atleast_1d(hc_vals)))
        except Exception:
            hc_between = float('nan')

        # Display frame with per-label columns expected by print_results_generic
        display = per_feature.copy()
        for lbl in ['A', 'B']:
            display[f'pval ({lbl})'] = per_feature['pval']
            display[f'HC ({lbl})'] = per_feature['HC']
            display[f'HCT ({lbl})'] = per_feature['thresh']
            display[f'score ({lbl})'] = -2.0 * np.log(
                np.clip(per_feature['pval'], 1e-300, 1),
            )
        display['sign (A)'] = per_feature['sign']
        display['sign (B)'] = -per_feature['sign']

        return GlobalHCResult(
            hc_between=hc_between,
            per_feature=per_feature,
            display_frame=display,
        )

    # ---- per-document (leave-one-out) HC -----------------------------------

    def compare_per_document(self, processed_data: pd.DataFrame,
                             gamma: float = 0.25) -> pd.DataFrame:
        """Leave-one-out HC for every document (chapter) in each corpus.

        For each document *d* belonging to corpus *C*, computes
        ``HC(d, C \\ d)`` and ``HC(d, C')``.

        Parameters
        ----------
        processed_data : pd.DataFrame
            The full processed feature table (must contain *author*, *doc_id*,
            *feature*).
        gamma : float
            HC lower-fraction parameter.

        Returns
        -------
        pd.DataFrame
            Columns: ``doc``, ``of`` (which corpus *d* belongs to),
            ``vs`` (which corpus it is compared against), ``HCmax``.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before compare_per_document()")
        _ensure_tshc()

        vocab_index = self.model.counts_df.index

        def _counts_from_doc(docdf: pd.DataFrame) -> pd.Series:
            return (
                docdf['feature'].value_counts()
                .reindex(vocab_index).fillna(0).astype(int)
            )

        cls_counts = {
            cls: self.model.counts_df[f'n ({cls})'].astype(int)
            for cls in ['A', 'B']
        }

        data_idx = processed_data.reset_index()
        pts = []
        for cls in ['A', 'B']:
            for doc_id, ddoc in data_idx[data_idx.author == cls].groupby('doc_id'):
                cnt1 = _counts_from_doc(ddoc)
                for other in ['A', 'B']:
                    cnt2 = cls_counts[other].copy()
                    if other == cls:
                        cnt2 = (cnt2 - cnt1).clip(lower=0)
                    pv = ts_two_sample_binomial_test(
                        cnt1.to_numpy(), cnt2.to_numpy(),
                    )
                    hc_res, _ = HCClass(pv, stbl=True).HCstar(gamma=gamma)
                    hc_arr = np.atleast_1d(hc_res)
                    hc_max = (
                        float(np.nanmax(hc_arr))
                        if hc_arr.size > 0
                        else float('nan')
                    )
                    pts.append({
                        'doc': str(doc_id), 'of': cls,
                        'vs': other, 'HCmax': hc_max,
                    })

        return pd.DataFrame(pts)

    # ---- convenience properties -------------------------------------------

    @property
    def counts_df(self) -> pd.DataFrame:
        """Access the underlying ``CompareDocs`` counts table."""
        return self.model.counts_df
