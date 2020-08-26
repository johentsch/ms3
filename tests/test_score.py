#!/usr/bin/env python

"""Tests for `ms3` package."""

import pytest
import os
import tempfile
from fractions import Fraction as frac

import pandas as pd

import ms3

@pytest.fixture(
    params=['Did03M-Son_regina-1762-Sarti.mscx', 'D973deutscher01.mscx', '05_symph_fant.mscx', 'BWV_0815.mscx', 'K281-3.mscx', '76CASM34A33UM.mscx'],
    ids=['sarti', "schubert", "berlioz", 'bach', 'mozart', 'monty'])
def score_object(request):
    mscx_path = os.path.realpath(os.path.join('MS3', request.param))
    s = ms3.Score(mscx_path, parser='bs4')
    return s

class TestBasic:

    def test_init(self):
        s = ms3.Score()
        assert isinstance(s, ms3.score.Score)
        with pytest.raises(LookupError):
            s.mscx


class TestParser:


    def test_parse_and_write_back(self, score_object):
        original_mscx = score_object.full_paths['mscx']
        tmp_file = tempfile.NamedTemporaryFile(mode='r')
        score_object.output_mscx(tmp_file.name)
        original = open(original_mscx).read()
        after_parsing = tmp_file.read()
        assert_all_lines_equal(original, after_parsing, original_mscx=original_mscx, tmp_file=tmp_file)

    def test_parse_to_measurelist(self, score_object):
        fname = score_object.fnames['mscx']
        fpath = os.path.join(score_object.paths['mscx'], '..')
        old_path = os.path.join(fpath, 'measures', fname + '.tsv')
        old_measurelist = load_tsv(old_path, index_col=None)
        new_measurelist = score_object.mscx.measures
        # Exclude 'repeat' column because the old parser used startRepeat, endRepeat and newSection
        # Exclude 'offset' and 'next' because the new parser does them more correctly
        excl = ['repeats', 'offset', 'next']
        new_measurelist.next = new_measurelist.next.map(lambda l: ', '.join(str(s) for s in l))
        new_measurelist.to_csv(fname + '_measures.tsv', sep='\t', index=False)
        assert_dfs_equal(old_measurelist, new_measurelist, exclude=excl)

    def test_parse_to_notelist(self, score_object):
        fname = score_object.fnames['mscx']
        fpath = os.path.join(score_object.paths['mscx'], '..')
        old_path = os.path.join(fpath, 'notes', fname + '.tsv')
        old_notelist = load_tsv(old_path, index_col=None)
        new_notelist = score_object.mscx.notes
        # Exclude 'onset' because the new parser computes 'offset' (measure list) more correctly
        excl = ['onset']
        new_notelist.to_csv(fname + '_notes.tsv', sep='\t', index=False)
        assert_dfs_equal(old_notelist, new_notelist, exclude=excl)

    def test_parse_to_eventlist(self, score_object):
        fname = score_object.fnames['mscx']
        score_object.mscx.parsed._events.to_csv(fname + '_events.tsv', sep='\t', index=False)



def assert_all_lines_equal(before, after, original_mscx, tmp_file):
    diff = [(bef, aft) for bef, aft in zip(before.splitlines(), after.splitlines()) if bef != aft]
    assert len(diff) == 0, '\n' + '\n'.join(
        f"{a} <--before   after-->{b}" for a, b in [(original_mscx, tmp_file.name)] + diff)


def assert_dfs_equal(old, new, exclude=[]):
    old_l, new_l = len(old), len(new)
    l = min(old_l, new_l)
    if old_l != new_l:
        print(f"Old length: {old_l}, new length: {new_l}")
    old.index.rename('old_ix', inplace=True)
    new.index.rename('new_ix', inplace=True)
    cols = [col for col in set(old.columns).intersection(set(new.columns)) if col not in exclude]
    nan_eq = lambda a, b: (a == b) | pd.isna(a) & pd.isna(b)
    diff = [(i, j, ~nan_eq(o, n)) for ((i, o), (j, n)) in zip(old[cols].iterrows(), new[cols].iterrows())]
    old_bool = pd.DataFrame.from_dict({ix: bool_series for ix, _, bool_series in diff}, orient='index')
    new_bool = pd.DataFrame.from_dict({ix: bool_series for _, ix, bool_series in diff}, orient='index')
    diffs_per_col = old_bool.sum(axis=0)

    def show_diff():
        comp_str = []
        for col, n_diffs in diffs_per_col.items():
            if n_diffs > 0:
                comparison = pd.concat([old.loc[old_bool[col], ['mc', col]].reset_index(drop=True).iloc[:20],
                                        new.loc[new_bool[col], ['mc', col]].iloc[:20].reset_index(drop=True)],
                                       axis=1,
                                       keys=['old', 'new'])
                comp_str.append(
                    f"{n_diffs}/{l} ({n_diffs / l * 100:.2f} %) rows are different for {col}{' (showing first 20)' if n_diffs > 20 else ''}:\n{comparison}\n")
        return '\n'.join(comp_str)
    assert diffs_per_col.sum() == 0, show_diff()



def load_tsv(path, index_col=[0, 1], converters={}, dtypes={}, stringtype=False, **kwargs):
    """ Loads the TSV file `path` while applying correct type conversion and parsing tuples.

    Parameters
    ----------
    path : :obj:`str`
        Path to a TSV file as output by format_data().
    index_col : :obj:`list`, optional
        By default, the first two columns are loaded as MultiIndex.
        The first level distinguishes pieces and the second level the elements within.
    converters, dtypes : :obj:`dict`, optional
        Enhances or overwrites the mapping from column names to types included the constants.
    stringtype : :obj:`bool`, optional
        If you're using pandas >= 1.0.0 you might want to set this to True in order
        to be using the new `string` datatype that includes the new null type `pd.NA`.
    """

    def str2inttuple(l):
        return tuple() if l == '' else tuple(int(s) for s in l.split(', '))

    def int2bool(s):
        try:
            return bool(int(s))
        except:
            return s

    CONVERTERS = {
        'added_tones': str2inttuple,
        'act_dur': frac,
        'chord_tones': str2inttuple,
        'globalkey_is_minor': int2bool,
        'localkey_is_minor': int2bool,
        'next': str2inttuple,
        'nominal_duration': frac,
        'offset': frac,
        'onset': frac,
        'duration': frac,
        'scalar': frac, }

    DTYPES = {
        'alt_label': str,
        'barline': str,
        'bass_note': 'Int64',
        'cadence': str,
        'cadences_id': 'Int64',
        'changes': str,
        'chord': str,
        'chord_type': str,
        'dont_count': 'Int64',
        'figbass': str,
        'form': str,
        'globalkey': str,
        'gracenote': str,
        'harmonies_id': 'Int64',
        'keysig': int,
        'label': str,
        'localkey': str,
        'mc': int,
        'midi': int,
        'mn': int,
        'notes_id': 'Int64',
        'numbering_offset': 'Int64',
        'numeral': str,
        'pedal': str,
        'playthrough': int,
        'phraseend': str,
        'relativeroot': str,
        'repeats': str,
        'root': 'Int64',
        'special': str,
        'staff': int,
        'tied': 'Int64',
        'timesig': str,
        'tpc': int,
        'voice': int,
        'voices': int,
        'volta': 'Int64'
    }


    if converters is None:
        conv = None
    else:
        conv = dict(CONVERTERS)
        conv.update(converters)

    if dtypes is None:
        types = None
    else:
        types = dict(DTYPES)
        types.update(dtypes)

    if stringtype:
        types = {col: 'string' if typ == str else typ for col, typ in types.items()}
    return pd.read_csv(path, sep='\t', index_col=index_col,
                       dtype=types,
                       converters=conv, **kwargs)


