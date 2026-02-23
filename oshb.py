"""
Minimal OSHB utilities used by the restored app.

This vendors the parts of bib-scripts needed by the UI:
- OSHBData: reads OSHB XML via a catalog CSV and caches a flat CSV
- ProcessText: prepares features and optional n-grams / filters

It reuses the logic found in bib-scripts/src/biblical_scripts/pipelines/data_engineering/_OSHBload.py
but in a trimmed, dependency-light form.
"""

from xml.dom.minidom import parse
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import re
import os
from pathlib import Path
import json

try:
    # NLTK is already in the old requirements; used only for everygrams
    from nltk import everygrams
except Exception:  # pragma: no cover
    everygrams = None


def _read_catalog(catalog_file: str) -> pd.DataFrame:
    df = pd.read_csv(catalog_file)
    df.loc[:, 'verses'] = df.verses.astype(str).apply(lambda vs: vs.strip(";").split(";"))
    df = df.explode('verses')
    df.verses = df.verses.replace('nan', 'all')
    return df


def _read_from_morph(path: str, catalog: pd.DataFrame) -> pd.DataFrame:
    def read_chapter(book: str, chapter: int, verse_set: List[int] = []) -> pd.DataFrame:
        rows = []
        bookxml = parse(f"{path}/{book}.xml")
        chapterlist = bookxml.getElementsByTagName('chapter')
        chapterlist = [ch for ch in chapterlist if ch.attributes['osisID'].value == f"{book}.{chapter}"]
        for chap in chapterlist:
            verselist = chap.getElementsByTagName('verse')
            for verse in verselist:
                mywelements = verse.getElementsByTagName('w')
                for el in mywelements:
                    vrs = verse.attributes['osisID'].value
                    vrs_numeric = int(vrs.split('.')[-1])
                    if not verse_set or vrs_numeric in verse_set:
                        rows.append(
                            {
                                'lemma': el.attributes['lemma'].value,
                                'morph': el.attributes['morph'].value,
                                'term': el.firstChild.data,
                                'chapter': chap.attributes['osisID'].value,
                                'verse': vrs,
                            }
                        )
        return pd.DataFrame(rows)

    acc = []
    for (author, book, chapter), sub in catalog.groupby(['author', 'book', 'chapter']):
        for r in sub.verses:
            if r == 'all':
                vs = []
            else:
                try:
                    a, b = r.split('-')
                except ValueError:
                    a = b = r
                vs = list(range(int(a), int(b) + 1))
            logging.debug(f"Reading: author={author}, book={book}, chapter={chapter}, verse_set={vs}")
            df = read_chapter(book=book, chapter=int(chapter), verse_set=vs)
            if not df.empty:
                df.loc[:, 'author'] = author
                acc.append(df)
    if not acc:
        return pd.DataFrame(columns=['lemma', 'morph', 'term', 'chapter', 'verse', 'author'])
    return pd.concat(acc, ignore_index=True)


def _read_hash(hash_file: str) -> str:
    try:
        with open(hash_file, 'r') as fl:
            old_hash = fl.read()
        return str(old_hash)
    except Exception:
        logging.debug(f"Cannot find hash file {hash_file}. Returning '-1'")
        return "-1"


def _store_hash(hash_val: str, hash_file: str) -> None:
    with open(hash_file, 'w') as fl:
        fl.write(str(hash_val))


class OSHBData:
    def __init__(
        self,
        raw_data_path: str,
        catalog_file: str,
        saved_data_path: str = './BiblicalScript_data.csv',
        hash_file: str = './OSHB.hash',
        force_saved: bool = False,
    ) -> None:
        self.catalog_file = catalog_file
        self.raw_data_path = raw_data_path

        logging.info(f"Reading Catalog File: {catalog_file}...")
        self.catalog = _read_catalog(self.catalog_file)
        logging.info(f"Found {len(self.catalog)} entries in catalog.")

        old_hash = _read_hash(hash_file)
        curr_hash = str(hash(tuple(self.catalog.verses.values)))
        self._has_changed = old_hash != curr_hash

        if self._has_changed and not force_saved:
            logging.info(
                "Catalog changed since last read. Extracting from OSHB XML according to catalog."
            )
            self._data = _read_from_morph(self.raw_data_path, self.catalog)
            logging.info(f"Saving extracted data to {saved_data_path} ...")
            self._data.to_csv(saved_data_path, index=False)
            logging.info(f"Storing catalog hash value in {hash_file}.")
            _store_hash(curr_hash, hash_file)
        else:
            logging.info(f"Reading cached data from {saved_data_path} ...")
            self._data = pd.read_csv(saved_data_path)

        # dictionaries for conversions
        self.dictionary_morph = dict(zip(self._data.lemma.values, self._data.morph.values))
        self.dictionary_term = dict(
            list(zip(self._data.lemma.values, self._data.term.values))
            + [('c', 'ו'), ('d', 'ה'), ('b', 'ב'), ('l', 'ל'), ('k', 'כ')]
        )

    def lemma_to_term(self, lemma: str) -> str:
        d = self.dictionary_term
        return d.get(
            lemma,
            d.get(
                lemma + ' a',
                d.get(
                    lemma + ' b',
                    d.get(
                        lemma + ' c',
                        d.get(
                            lemma + ' d',
                            d.get(
                                lemma + ' e',
                                d.get(
                                    lemma + ' l',
                                    d.get(
                                        lemma + ' m',
                                        d.get(
                                            'a/' + lemma,
                                            d.get(
                                                'b/' + lemma,
                                                d.get(
                                                    'c/' + lemma,
                                                    d.get(
                                                        'c/' + lemma + ' b',
                                                        d.get(
                                                            'd/' + lemma,
                                                            d.get(
                                                                'e/' + lemma,
                                                                d.get(
                                                                    'k/' + lemma,
                                                                    d.get(
                                                                        'l/' + lemma,
                                                                        d.get('m/' + lemma, lemma),
                                                                    ),
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

    def lemma_to_morph(self, lemma: str) -> str:
        return self.dictionary_morph.get(lemma, '')

    # ---------- Ad hoc piece loading with caching ----------
    @staticmethod
    def list_books(raw_data_path: str) -> List[str]:
        return [Path(p).stem for p in Path(raw_data_path).glob('*.xml')]

    @staticmethod
    def list_chapters(raw_data_path: str, book: str) -> List[int]:
        """Return available chapter numbers for a given book by scanning the XML."""
        bookxml = parse(f"{raw_data_path}/{book}.xml")
        chapters = bookxml.getElementsByTagName('chapter')
        out = []
        for ch in chapters:
            osis = ch.attributes['osisID'].value  # e.g., Gen.1
            try:
                out.append(int(osis.split('.')[-1]))
            except Exception:
                pass
        return sorted(list(set(out)))

    @staticmethod
    def list_verses(raw_data_path: str, book: str, chapter: int) -> List[int]:
        """Return available verse numbers for a given book/chapter by scanning the XML."""
        bookxml = parse(f"{raw_data_path}/{book}.xml")
        chapters = bookxml.getElementsByTagName('chapter')
        verses = []
        for ch in chapters:
            if ch.attributes['osisID'].value != f"{book}.{chapter}":
                continue
            for verse in ch.getElementsByTagName('verse'):
                try:
                    verses.append(int(verse.attributes['osisID'].value.split('.')[-1]))
                except Exception:
                    pass
        return sorted(list(set(verses)))

    @staticmethod
    def _piece_cache_name(book: str, chapter: int, verses: List[int]) -> str:
        if not verses:
            vs = 'all'
        else:
            # compress sorted list into ranges for shorter filenames
            v = sorted(set(int(x) for x in verses))
            ranges = []
            s = e = v[0]
            for x in v[1:]:
                if x == e + 1:
                    e = x
                else:
                    ranges.append((s, e))
                    s = e = x
            ranges.append((s, e))
            vs = '_'.join([f"{a}-{b}" if a != b else f"{a}" for a, b in ranges])
        return f"{book}.{int(chapter)}.{vs}.csv"

    @staticmethod
    def _read_chapter_from_xml(path: str, book: str, chapter: int, verse_set: List[int]) -> pd.DataFrame:
        rows = []
        bookxml = parse(f"{path}/{book}.xml")
        chapterlist = bookxml.getElementsByTagName('chapter')
        chapterlist = [ch for ch in chapterlist if ch.attributes['osisID'].value == f"{book}.{chapter}"]
        for chap in chapterlist:
            verselist = chap.getElementsByTagName('verse')
            for verse in verselist:
                mywelements = verse.getElementsByTagName('w')
                for el in mywelements:
                    vrs = verse.attributes['osisID'].value
                    vrs_numeric = int(vrs.split('.')[-1])
                    if not verse_set or vrs_numeric in verse_set:
                        rows.append(
                            {
                                'lemma': el.attributes['lemma'].value,
                                'morph': el.attributes['morph'].value,
                                'term': el.firstChild.data,
                                'chapter': chap.attributes['osisID'].value,
                                'verse': vrs,
                            }
                        )
        return pd.DataFrame(rows)

    def read_pieces(self, raw_data_path: str, pieces: List[dict], cache_dir: str = './cache') -> pd.DataFrame:
        """
        pieces: list of dicts with keys {'book': str, 'chapter': int, 'verses': List[int] or [] for all}
        Returns a concatenated DataFrame of tokens for the requested pieces.
        Caches each piece under cache_dir for faster reuse.
        """
        os.makedirs(cache_dir, exist_ok=True)
        acc = []
        for piece in pieces:
            book = piece['book']
            chapter = int(piece['chapter'])
            verses = list(piece.get('verses', []))
            fname = self._piece_cache_name(book, chapter, verses)
            fpath = os.path.join(cache_dir, fname)
            if os.path.exists(fpath):
                df = pd.read_csv(fpath)
            else:
                df = OSHBData._read_chapter_from_xml(raw_data_path, book, chapter, verses)
                df.to_csv(fpath, index=False)
            acc.append(df)
        if len(acc) == 0:
            return pd.DataFrame(columns=['lemma', 'morph', 'term', 'chapter', 'verse'])
        return pd.concat(acc, ignore_index=True)

    @staticmethod
    def read_pieces_static(raw_data_path: str, pieces: List[dict], cache_dir: str = './cache') -> pd.DataFrame:
        """Static variant of read_pieces; does not require an OSHBData instance."""
        os.makedirs(cache_dir, exist_ok=True)
        acc = []
        for piece in pieces:
            book = piece['book']
            chapter = int(piece['chapter'])
            verses = list(piece.get('verses', []))
            fname = OSHBData._piece_cache_name(book, chapter, verses)
            fpath = os.path.join(cache_dir, fname)
            if os.path.exists(fpath):
                df = pd.read_csv(fpath)
            else:
                df = OSHBData._read_chapter_from_xml(raw_data_path, book, chapter, verses)
                df.to_csv(fpath, index=False)
            acc.append(df)
        if len(acc) == 0:
            return pd.DataFrame(columns=['lemma', 'morph', 'term', 'chapter', 'verse'])
        return pd.concat(acc, ignore_index=True)

    @staticmethod
    def build_lemma_term_dict(raw_data_path: str, cache_dir: str = './cache') -> dict:
        """
        Build or load a global lemma->term dictionary by scanning all XML files
        under raw_data_path. Cached to JSON for speed.
        """
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, 'lemma_map.json')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass

        mapping = {}
        def add_key(k: str, t: str):
            if k not in mapping:
                mapping[k] = t

        for xml_path in Path(raw_data_path).glob('*.xml'):
            try:
                bookxml = parse(str(xml_path))
                for chap in bookxml.getElementsByTagName('chapter'):
                    for verse in chap.getElementsByTagName('verse'):
                        for el in verse.getElementsByTagName('w'):
                            lem = el.attributes['lemma'].value
                            term = el.firstChild.data
                            add_key(lem, term)
                            # add normalized variants
                            # prefix form like 'b/1121'
                            m = re.match(r'^[a-z]/(\d+)$', lem)
                            if m:
                                add_key(m.group(1), term)
                            # suffix form '1121 a'
                            m2 = re.match(r'^(\d+)\s+[a-z]$', lem)
                            if m2:
                                add_key(m2.group(1), term)
            except Exception:
                continue
        try:
            with open(cache_file, 'w') as f:
                json.dump(mapping, f)
        except Exception:
            pass
        return mapping


class ProcessText:
    def __init__(self, **kwargs) -> None:
        self.to_remove: List[str] = kwargs.get('to_remove', [])
        self.to_replace: List[str] = kwargs.get('to_replace', [])
        self.extract_prefix: bool = kwargs.get('extract_prefix', False)
        self.extract_suffix: bool = kwargs.get('extract_suffix', False)
        self.ng_range: Tuple[int, int] = kwargs.get('ng_range', (1, 1))
        self.pad: bool = kwargs.get('pad', False)

    def _extract_prefix_suffix(self, data: pd.DataFrame) -> pd.DataFrame:
        # suffixes (only when extract_suffix is True)
        suff = data[data.morph.str.contains(r'/S[dhnp][1-3][bcfm][dps]')]
        # prefixes (definite article and prepositions; only when extract_prefix is True)
        pref = data[data.morph.str.contains(r'^[HA][A-Z][a-z]?/[^S]')]

        data.loc[:, 'feature'] = data.lemma.str.extract(r'(?:^[a-z]/)?([A-Za-z0-9]+)', expand=False)
        data.loc[:, 'morph'] = data.morph.str.extract(r'(?:[HA][A-Z][a-z]?/)?([A-Za-z0-9]+)', expand=False)
        data.loc[:, 'POW'] = 'main'

        parts = []
        if self.extract_prefix:
            pref.loc[:, 'feature'] = pref.lemma.str.extract(r'(^[a-z]|l)/?', expand=False)
            pref.loc[:, 'morph'] = pref.morph.str.extract(r'([HA][A-Z][a-z]?)/', expand=False)
            pref.loc[:, 'POW'] = 'prefix'
            parts.append(pref)
        parts.append(data)
        if self.extract_suffix:
            suff.loc[:, 'feature'] = '[' + suff.morph.str.extract(r'(S[dhnp][1-3][bcfm][dps])', expand=False) + ']'
            suff.loc[:, 'morph'] = suff.morph.str.extract(r'(S[dhnp][1-3][bcfm][dps])', expand=False)
            suff.loc[:, 'POW'] = 'suffix'
            parts.append(suff)
        return pd.concat(parts).sort_values(by='token_id')

    def _extract_ngrams(self, df: pd.DataFrame, key: str, by: List[str]) -> pd.DataFrame:
        if everygrams is None:
            raise RuntimeError("nltk.everygrams not available; please install nltk")
        if self.pad:
            seq = (
                df.groupby(by)[key]
                .apply(
                    lambda x: list(
                        everygrams(
                            x,
                            min_len=self.ng_range[0],
                            max_len=self.ng_range[1],
                            pad_left=True,
                            pad_right=True,
                            left_pad_symbol='<start>',
                            right_pad_symbol='<end>',
                        )
                    )
                )
                .explode()
                .reset_index()
            )
        else:
            seq = (
                df.groupby(by)[key]
                .apply(lambda x: list(everygrams(x, min_len=self.ng_range[0], max_len=self.ng_range[1])))
                .explode()
                .reset_index()
            )
        return seq

    def proc(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        data = raw_data.copy()
        data.loc[:, 'feature'] = data['lemma']
        data['token_id'] = data.index
        if 'POW' not in data.columns:
            data['POW'] = 'main'

        if self.extract_prefix or self.extract_suffix:
            logging.info("Extracting prefixes and suffixes")
            data = self._extract_prefix_suffix(data)

        if self.extract_suffix:
            logging.info("Counting suffixes")
            data['feature'].replace({r"\[(S[dhnp][1-3][bcfm][dps])\]": r"\1"}, inplace=True, regex=True)

        data.loc[:, 'feature (org)'] = data['feature']

        # mark features to remove or replace
        for cd in self.to_remove:
            logging.info(f"Removing {cd}")
            data.loc[data.morph.str.contains(fr'(?:^|[H/])(?:{cd})'), 'feature'] = f"[{cd}]"
        for cd in self.to_replace:
            logging.info(f"Replacing {cd}")
            data.loc[data.morph.str.contains(fr'(?:^|[H/])(?:{cd})'), 'feature'] = f"<{cd}>"

        sel = ['token_id', 'author', 'chapter', 'verse', 'feature', 'feature (org)', 'morph', 'term', 'lemma', 'POW']
        self._data = data.filter(sel)
        return self._data

    def proc_ng(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        data = self.proc(raw_data)
        data_ng = self._extract_ngrams(data, key='feature', by=['author', 'chapter', 'verse'])
        dfmap = self._extract_ngrams(data, key='token_id', by=['author', 'chapter', 'verse'])
        data_ng['token_id'] = dfmap.token_id
        self._data_ng = data_ng
        return data_ng

    def inv_trans(self, tid: int) -> pd.Series:
        return self._data[self._data.token_id == tid].iloc[0]
