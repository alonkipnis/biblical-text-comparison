"""
Minimal document-comparison utilities using TwoSampleHC for allocation P-values
and Higher Criticism threshold.
"""

from dataclasses import dataclass
import logging
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
try:
    from TwoSampleHC import HC as HCClass, two_sample_binomial_test as ts_two_sample_binomial_test
except Exception as _e:  # pragma: no cover
    HCClass = None
    ts_two_sample_binomial_test = None
    _TSHC_IMPORT_ERROR = _e


def _ensure_tshc():
    if HCClass is None or ts_two_sample_binomial_test is None:
        raise ImportError(f"TwoSampleHC is required but could not be imported: {_TSHC_IMPORT_ERROR}")


@dataclass
class CompareDocs:
    vocabulary: Iterable[str]
    min_count: int = 0

    def __post_init__(self):
        self.vocabulary = list(self.vocabulary)
        if not self.vocabulary:
            raise ValueError("Vocabulary is empty.")
        self.cls_names: List[str] = []
        self.counts_df: pd.DataFrame = pd.DataFrame(index=self.vocabulary)

    @staticmethod
    def _count_from_df(df: pd.DataFrame) -> pd.Series:
        # df expected to have a column named 'feature'
        vc = df['feature'].value_counts()
        return vc

    def fit(self, data: Dict[str, pd.DataFrame]) -> None:
        """Build counts per class over a fixed vocabulary."""
        idx = pd.Index(self.vocabulary, name='feature')
        base = pd.DataFrame({'n Tot': 0, 'T Tot': 0}, index=idx)
        df = base.copy()
        self.cls_names = []
        for cls, df_cls in data.items():
            if cls == 'tested':
                raise ValueError("'tested' is a reserved name for new documents")
            self.cls_names.append(cls)
            cnt = self._count_from_df(df_cls)
            cnt = cnt.reindex(idx).fillna(0).astype(int)
            df[f'n ({cls})'] = cnt
            df[f'T ({cls})'] = int(cnt.sum())
            df['n Tot'] = df['n Tot'] + cnt
            df['T Tot'] = df['T Tot'] + int(cnt.sum())

        self.counts_df = df.fillna(0)

    def _hct_one_vs_many(self, stbl: bool = True, gamma: float = 0.25) -> pd.DataFrame:
        dft = self.counts_df.copy()
        for cls in self.cls_names:
            n1 = dft[f'n ({cls})'].to_numpy()
            T1 = int(dft[f'T ({cls})'].iloc[0])
            n_rest = (dft['n Tot'] - dft[f'n ({cls})']).to_numpy()
            T_rest = int(dft['T Tot'].iloc[0] - T1)
            _ensure_tshc()
            pv = ts_two_sample_binomial_test(n1, n_rest)
            hc, pth = HCClass(pv, stbl=stbl).HCstar(gamma=gamma)
            dft[f'HC (ovm {cls})'] = hc
            dft[f'HCT (ovm {cls})'] = pv <= pth
            # majority sign: whether class proportion is higher than rest
            with np.errstate(divide='ignore', invalid='ignore'):
                sign = np.sign(n1 / max(T1, 1) - n_rest / max(T_rest, 1))
            dft[f'sign (ovm {cls})'] = np.nan_to_num(sign)
            dft[f'{cls}:mask'] = dft[f'HCT (ovm {cls})']
        return dft

    def test_doc(self, doc: pd.DataFrame, stbl: bool = True, gamma: float = 0.25) -> pd.DataFrame:
        """Evaluate a test document against each class; return per-feature DataFrame."""
        if doc.empty:
            raise ValueError("Test document is empty")
        # ensure mask columns exist based on ovm selection
        _ = self._hct_one_vs_many(stbl=stbl, gamma=gamma)

        idx = self.counts_df.index
        cnt_doc = doc['feature'].value_counts().reindex(idx).fillna(0).astype(int)
        df = self.counts_df.join(cnt_doc.rename('n (test)'))
        df['T (test)'] = int(df['n (test)'].sum())

        for cls in self.cls_names:
            n1 = df['n (test)'].to_numpy()
            n2 = df[f'n ({cls})'].to_numpy()
            _ensure_tshc()
            pv = ts_two_sample_binomial_test(n1, n2)
            T1 = int(n1.sum()); T2 = int(n2.sum()); p = (T1/(T1+T2)) if (T1+T2)>0 else 0.0
            df[f'pval ({cls})'] = pv
            df[f'score ({cls})'] = -2.0 * np.log(np.clip(pv, 1e-300, 1))
            hc, pth = HCClass(pv, stbl=stbl).HCstar(gamma=gamma)
            df[f'HC ({cls})'] = hc
            df[f'HCT ({cls})'] = pv <= pth
            # sign relative to pooled allocation
            with np.errstate(divide='ignore', invalid='ignore'):
                more = -np.sign(n1 - (n1 + n2) * p)
            df[f'sign ({cls})'] = np.nan_to_num(more)

        return df

    def compare_classes(self, gamma: float = 0.25) -> pd.DataFrame:
        """Compare exactly two fitted classes and return per-feature statistics.

        Columns: n (C1), T (C1), n (C2), T (C2), pval, HC, thresh, sign
        where sign = +1 if proportion in C1 > C2, -1 otherwise.
        """
        if len(self.cls_names) != 2:
            raise ValueError("compare_classes requires exactly two classes fitted.")
        c1, c2 = self.cls_names
        df = self.counts_df.copy()
        n1 = df[f'n ({c1})'].to_numpy()
        n2 = df[f'n ({c2})'].to_numpy()
        _ensure_tshc()
        pv = ts_two_sample_binomial_test(n1, n2)
        hc, pth = HCClass(pv, stbl=True).HCstar(gamma=gamma)
        with np.errstate(divide='ignore', invalid='ignore'):
            sign = np.sign(n1 / max(int(df[f'T ({c1})'].iloc[0]), 1) - n2 / max(int(df[f'T ({c2})'].iloc[0]), 1))
        df = df.assign(**{
            'pval': pv,
            'HC': hc,
            'thresh': pv <= pth,
            'sign': np.nan_to_num(sign),
        })
        return df


class MultiCorpus2:
    """Thin wrapper to match the old interface used by webapp2.py"""

    def __init__(self, data: pd.DataFrame, ui: Dict):
        self.MIN_CNT = ui['min_cnt']
        self.known_authors = ui['known_authors']
        self.vocab = list(ui['vocab'])
        self.NG_RANGE = ui['ng_range']
        self.N_WORDS = ui['n_words']

        ds = (
            data.filter(['author', 'feature', 'doc_id'])
            .rename(columns={'author': 'cls'})
        )
        # remove features explicitly marked to ignore (square brackets)
        ds = ds[~ds.feature.astype(str).str.match(r"^\[.+\]$")]

        self.doc = ds[ds.doc_id.isin(ui['test_docs'])]

        self.model = CompareDocs(vocabulary=self.vocab, min_count=self.MIN_CNT)
        lo_docs: Dict[str, pd.DataFrame] = {}
        for auth in self.known_authors:
            lo_docs[auth] = ds[ds.cls == auth]
        self.model.fit(data=lo_docs)

    def compare_texts(self, ui: Dict) -> pd.DataFrame:
        return self.model.test_doc(self.doc, stbl=ui.get('stbl', True), gamma=ui['gamma'])
