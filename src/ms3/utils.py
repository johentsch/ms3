import os, re
from collections import defaultdict
from collections.abc import Iterable
from fractions import Fraction as frac
from itertools import repeat

import pandas as pd
import numpy as np

from .logger import function_logger


def ambitus2oneliner(ambitus):
    """ Turns a ``metadata['parts'][staff_id]`` dictionary into a string."""
    return f"{ambitus['min_midi']}-{ambitus['max_midi']} ({ambitus['min_name']}-{ambitus['max_name']})"


def decode_harmonies(df, return_series=False):
    df = df.copy()
    drop_cols, compose_label = [], []
    if 'nashville' in df.columns:
        sel = df.nashville.notna()
        df.loc[sel, 'label'] = df.loc[sel, 'nashville'] + df.loc[sel, 'label'].replace('/', '')
        drop_cols.append('nashville')
    if 'leftParen' in df.columns:
        df.leftParen.replace('/', '(', inplace=True)
        compose_label.append('leftParen')
        drop_cols.append('leftParen')
    if 'root' in df.columns:
        df.root = fifths2name(df.root, ms=True)
        compose_label.append('root')
        drop_cols.append('root')
        if 'rootCase' in df.columns:
            drop_cols.append('rootCase')
        # TODO: use rootCase
    compose_label.append('label')
    if 'base' in df.columns:
        df.base = '/' + fifths2name(df.base, ms=True)
        compose_label.append('base')
        drop_cols.append('base')
    if 'rightParen' in df.columns:
        df.rightParen.replace('/', ')', inplace=True)
        compose_label.append('rightParen')
        drop_cols.append('rightParen')
    label_col = df[compose_label].fillna('').sum(axis=1).replace('', np.nan)
    if return_series:
        return label_col
    if 'label_type' in df.columns:
        df.loc[df.label_type.isin([1, 2, 3, '1', '2', '3']), 'label_type'] == 0
    df.label = label_col
    df.drop(columns=drop_cols, inplace=True)
    return df


def dict2oneliner(d):
    """ Turns a dictionary into a single-line string without brackets."""
    return ', '.join(f"{k}: {v}" for k, v in d.items())


def fifths2acc(fifths):
    """ Returns accidentals for a stack of fifths that can be combined with a
        basic representation of the seven steps."""
    return abs(fifths // 7) * 'b' if fifths < 0 else fifths // 7 * '#'



def fifths2iv(fifths):
    """ Return interval name of a stack of fifths such that
       0 = 'P1', -1 = 'P4', -2 = 'm7', 4 = 'M3' etc.
       Uses: map2elements()
    """
    if pd.isnull(fifths):
        return fifths
    if isinstance(fifths, Iterable):
        return map2elements(fifths, fifths2iv)
    interval_qualities = {0: ['P', 'P', 'P', 'M', 'M', 'M', 'M'],
                          -1: ['D', 'D', 'D', 'm', 'm', 'm', 'm']}
    fifths += 1  # making 0 = fourth, 1 = unison, 2 = fifth etc.
    pos = fifths % 7
    int_num = [4, 1, 5, 2, 6, 3, 7][pos]
    qual_region = fifths // 7
    if qual_region in interval_qualities:
        int_qual = interval_qualities[qual_region][pos]
    elif qual_region < 0:
        int_qual = (abs(qual_region) - 1) * 'D'
    else:
        int_qual = qual_region * 'A'
    return int_qual + str(int_num)



def fifths2name(fifths, midi=None, ms=False):
    """ Return note name of a stack of fifths such that
       0 = C, -1 = F, -2 = Bb, 1 = G etc.
       Uses: map2elements(), fifths2str()

    Parameters
    ----------
    fifths : :obj:`int`
        Tonal pitch class to turn into a note name.
    midi : :obj:`int`
        In order to include the octave into the note name,
        pass the corresponding MIDI pitch.
    ms : :obj:`bool`, optional
        Pass True if ``fifths`` is a MuseScore TPC, i.e. C = 14
    """
    try:
        fifths = int(float(fifths))
    except:
        if isinstance(fifths, Iterable):
            return map2elements(fifths, fifths2name, ms=ms)
        return fifths

    if ms:
        fifths -= 14
    note_names = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
    name = fifths2str(fifths, note_names, inverted=True)
    if midi is not None:
        octave = midi2octave(midi, fifths)
        return f"{name}{octave}"
    return name



def fifths2pc(fifths):
    """ Turn a stack of fifths into a chromatic pitch class.
        Uses: map2elements()
    """
    try:
        fifths = int(float(fifths))
    except:
        if isinstance(fifths, Iterable):
            return map2elements(fifths, fifths2pc)
        return fifths

    return int(7 * fifths % 12)



def fifths2rn(fifths, minor=False, auto_key=False):
    """Return Roman numeral of a stack of fifths such that
       0 = I, -1 = IV, 1 = V, -2 = bVII in major, VII in minor, etc.
       Uses: map2elements(), is_minor_mode()

    Parameters
    ----------
    auto_key : :obj:`bool`, optional
        By default, the returned Roman numerals are uppercase. Pass True to pass upper-
        or lowercase according to the position in the scale.
    """
    if pd.isnull(fifths):
        return fifths
    if isinstance(fifths, Iterable):
        return map2elements(fifths, fifths2rn, minor=minor)
    rn = ['VI', 'III', 'VII', 'IV', 'I', 'V', 'II'] if minor else ['IV', 'I', 'V', 'II', 'VI', 'III', 'VII']
    sel = fifths + 3 if minor else fifths
    res = fifths2str(sel, rn)
    if auto_key and is_minor_mode(fifths, minor):
        return res.lower()
    return res



def fifths2sd(fifths, minor=False):
    """Return scale degree of a stack of fifths such that
       0 = '1', -1 = '4', -2 = 'b7' in major, '7' in minor etc.
       Uses: map2elements(), fifths2str()
    """
    if pd.isnull(fifths):
        return fifths
    if isinstance(fifths, Iterable):
        return map2elements(fifths, fifths2sd, minor=minor)
    sd = ['6', '3', '7', '4', '1', '5', '2'] if minor else ['4', '1', '5', '2', '6', '3', '7']
    if minor:
        fifths += 3
    return fifths2str(fifths, sd)



def fifths2str(fifths, steps, inverted=False):
    """ Boiler plate used by fifths2-functions.
    """
    fifths += 1
    acc = fifths2acc(fifths)
    if inverted:
        return steps[fifths % 7] + acc
    return acc + steps[fifths % 7]


def group_id_tuples(l):
    """ Turns a list of (key, ix) into a {key: [ix]}

    """
    d = defaultdict(list)
    for k, i in l:
        if k is not None:
            d[k].append(i)
    return dict(d)


def is_minor_mode(fifths, minor=False):
    """ Returns True if the scale degree `fifths` naturally has a minor third in the scale.
    """
    thirds = [-4, -3, -2, -1, 0, 1, 2] if minor else [3, 4, 5, -1, 0, 1, 2]
    third = thirds[(fifths + 1) % 7] - fifths
    return third == -3


def iterable2str(iterable):
    return ', '.join(str(s) for s in iterable)


def load_tsv(path, index_col=None, sep='\t', converters={}, dtypes={}, stringtype=False, **kwargs):
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
        'mc_offset': frac,
        'onset': frac,
        'duration': frac,
        'scalar': frac, }

    DTYPES = {
        'alt_label': str,
        'barline': str,
        'base': 'Int64',
        'bass_note': 'Int64',
        'cadence': str,
        'cadences_id': 'Int64',
        'changes': str,
        'chord': str,
        'chord_id': 'Int64',
        'chord_type': str,
        'dont_count': 'Int64',
        'figbass': str,
        'form': str,
        'globalkey': str,
        'gracenote': str,
        'harmonies_id': 'Int64',
        'keysig': int,
        'label': str,
        'label_type': object,
        'leftParen': str,
        'localkey': str,
        'mc': int,
        'midi': int,
        'mn': int,
        'offset:x': str,
        'offset:y': str,
        'nashville': 'Int64',
        'notes_id': 'Int64',
        'numbering_offset': 'Int64',
        'numeral': str,
        'pedal': str,
        'playthrough': int,
        'phraseend': str,
        'relativeroot': str,
        'repeats': str,
        'rightParen': str,
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
    return pd.read_csv(path, sep=sep, index_col=index_col,
                       dtype=types,
                       converters=conv, **kwargs)


def make_id_tuples(key, n):
    """ For a given key, this function return index tuples in the form [(key, 0), ..., (key, n)]

    Returns
    -------
    list
        indices in the form [(key, 0), ..., (key, n)]

    """
    return list(zip(repeat(key), range(n)))


def map2elements(e, f, *args, **kwargs):
    """ If `e` is an iterable, `f` is applied to all elements.
    """
    if isinstance(e, Iterable) and not isinstance(e, str):
        return e.__class__(map2elements(x, f, *args, **kwargs) for x in e)
    return f(e, *args, **kwargs)


def metadata2series(d):
    """ Turns a metadata dict into a pd.Series() (for storing in a DataFrame)
    Uses: ambitus2oneliner(), dict2oneliner(), parts_info()

    Returns
    -------
    :obj:`pandas.Series`
        A series allowing for storing metadata as a row of a DataFrame.
    """
    d = dict(d)
    d['TimeSig'] = dict2oneliner(d['TimeSig'])
    d['KeySig'] = dict2oneliner(d['KeySig'])
    d['ambitus'] = ambitus2oneliner(d['ambitus'])
    d.update(parts_info(d['parts']))
    del (d['parts'])
    s = pd.Series(d)
    return s


@function_logger
def midi2octave(midi, fifths=None):
    """ For a given MIDI pitch, calculate the octave. Middle octave = 4
        Uses: fifths2pc(), map2elements()

    Parameters
    ----------
    midi : :obj:`int`
        MIDI pitch (positive integer)
    tpc : :obj:`int`, optional
        To be precise, for some Tonal Pitch Classes, the octave deviates
        from the simple formula ``MIDI // 12 - 1``, e.g. for B# or Cb.
    """
    try:
        midi = int(float(midi))
    except:
        if isinstance(midi, Iterable):
            return map2elements(midi, midi2octave)
        return midi
    i = -1
    if fifths is not None:
        pc = fifths2pc(fifths)
        if midi % 12 != pc:
            logger.debug(f"midi2octave(): The Tonal Pitch Class {fifths} cannot be MIDI pitch {midi} ")
        if fifths in [
            12,  # B#
            19,  # B##
            26,  # B###
            24,  # A###
        ]:
            i -= 1
        elif fifths in [
            -7,  # Cb
            -14,  # Cbb
            -21,  # Cbbb
            -19,  # Dbbb
        ]:
            i += 1
    return midi // 12 + i


@function_logger
def name2tpc(nn):
    """ Turn a note name such as `Ab` into a tonal pitch class, such that -1=F, 0=C, 1=G etc.
        Uses: split_note_name()
    """
    if nn.__class__ == int or pd.isnull(nn):
        return nn
    name_tpcs = {'C': 0, 'D': 2, 'E': 4, 'F': -1, 'G': 1, 'A': 3, 'B': 5}
    accidentals, note_name = split_note_name(nn, count=True, logger=logger)
    step_tpc = name_tpcs[note_name.upper()]
    return step_tpc + 7 * accidentals


@function_logger
def no_collections_no_booleans(df):
    """
    Cleans the DataFrame columns ['next', 'chord_tones', 'added_tones'] from tuples and the columns
    ['globalkey_is_minor', 'localkey_is_minor'] from booleans, converting them all to integers

    """
    if df is None:
        return df
    collection_cols = ['next', 'chord_tones', 'added_tones']
    try:
        cc = [c for c in collection_cols if c in df.columns]
    except:
        logger.error(f"df needs to be a DataFrame, not a {df.__class__}.")
        return df
    if len(cc) > 0:
        df = df.copy()
        df.loc[:, cc] = transform(df[cc], iterable2str, column_wise=True)
        logger.debug(f"Transformed iterables in the columns {cc} to strings.")
    bool_cols = ['globalkey_is_minor', 'localkey_is_minor']
    bc = [c for c in bool_cols if c in df.columns]
    if len(bc) > 0:
        conv = {c: int for c in bc}
        df = df.astype(conv)
    return df


def ordinal_suffix(n):
    suffixes = {
        1: 'st',
        2: 'nd',
        3: 'rd'
    }
    n = str(n)
    if n[-1] in suffixes:
        return suffixes[n[-1]]
    return 'th'


def parts_info(d):
    """
    Turns a (nested) ``metadata['parts']`` dict into a flat dict based on staves.

    Example
    -------
    >>> d = s.mscx.metadata
    >>> parts_info(d['parts'])
    {'staff_1_name': 'Piano Right Hand',
     'staff_1_ambitus': '60-87 (C4-Eb6)',
     'staff_2_name': 'Piano Left Hand',
     'staff_2_ambitus': '39-75 (Eb2-Eb5)'}
    """
    res = {}
    for name, staves in d.items():
        for staff, ambitus in staves.items():
            res[f"staff_{staff}_name"] = name
            res[f"staff_{staff}_ambitus"] = ambitus2oneliner(ambitus)
    return res



def pretty_dict(d, heading=None):
    """ Turns a dictionary into a string where the keys are printed in a column, separated by '->'.
    """
    if heading is not None:
        d = dict(KEY=str(heading), **d)
    left = max(len(str(k)) for k in d.keys())
    res = []
    for k, v in d.items():
        ks = str(k)
        if isinstance(v, pd.DataFrame) or isinstance(v, pd.Series):
            vs = v.to_string()
        else:
            vs = str(v)
        if '\n' in vs:
            lines = vs.split('\n')
            res.extend([f"{ks if i == 0 else '':{left}} -> {l}" for i, l in enumerate(lines)])
        else:
            res.append(f"{ks:{left}} -> {vs}")
    if heading is not None:
        res.insert(1, '-' * (left + len(heading) + 4))
    return '\n'.join(res)



def resolve_dir(dir):
    """ Resolves '~' to HOME directory and turns ``dir`` into an absolute path.
    """
    if dir is None:
        return None
    if '~' in dir:
        return os.path.expanduser(dir)
    return os.path.abspath(dir)


def scan_directory(dir, file_re=r".*", folder_re=r".*", exclude_re=r"^(\.|__)", recursive=True):
    """ Get a list of files.

    Parameters
    ----------
    dir : :obj:`str`
        Directory to be scanned for files.
    file_re, folder_re : :obj:`str`, optional
        Regular expressions for filtering certain file names or folder names.
        The regEx are checked with search(), not match(), allowing for fuzzy search.
    recursive : :obj:`bool`, optional
        By default, sub-directories are recursively scanned. Pass False to scan only ``dir``.

    Returns
    -------
    list
        List of full paths meeting the criteria.

    """
    def check_regex(reg, s):
        res = re.search(reg, s) is not None and re.search(exclude_re, s) is None
        return res

    res = []
    for subdir, dirs, files in os.walk(dir):
        _, current_folder = os.path.split(subdir)
        if recursive and re.search(exclude_re, current_folder) is None:
            dirs[:] = [d for d in sorted(dirs)]
        else:
            dirs[:] = []
        if check_regex(folder_re, current_folder):
            files = [os.path.join(subdir, f) for f in sorted(files) if check_regex(file_re, f)]
            res.extend(files)
    return res



def sort_tpcs(tpcs, ascending=True, start=None):
    """ Sort tonal pitch classes by order on the piano.
        Uses: fifths2pc()

    Parameters
    ----------
    tpcs : collection of :obj:`int`
        Tonal pitch classes to sort.
    ascending : :obj:`bool`, optional
        Pass False to sort by descending order.
    start : :obj:`int`, optional
        Start on or above this TPC.
    """
    res = sorted(tpcs, key=lambda x: (fifths2pc(x), -x))
    if start is not None:
        pcs = [fifths2pc(tpc) for tpc in res]
        start = fifths2pc(start)
        i = 0
        while i < len(pcs) - 1 and pcs[i] < start:
            i += 1
        res = res[i:] + res[:i]
    return res if ascending else list(reversed(res))



@function_logger
def split_note_name(nn, count=False):
    """ Splits a note name such as 'Ab' into accidentals and name.

    nn : :obj:`str`
        Note name.
    count : :obj:`bool`, optional
        Pass True to get the accidentals as integer rather than as string.
    """
    m = re.match("^([A-G]|[a-g])(#*|b*)$", str(nn))
    if m is None:
        logger.error(nn + " is not a valid scale degree.")
        return None, None
    note_name, accidentals = m.group(1), m.group(2)
    if count:
        accidentals = accidentals.count('#') - accidentals.count('b')
    return accidentals, note_name



def transform(df, func, param2col=None, column_wise=False, **kwargs):
    """ Compute a function for every row of a DataFrame, using several cols as arguments.
        The result is the same as using df.apply(lambda r: func(param1=r.col1, param2=r.col2...), axis=1)
        but it optimizes the procedure by precomputing `func` for all occurrent parameter combinations.
        Uses: inspect.getfullargspec()

    Parameters
    ----------
    df : :obj:`pandas.DataFrame` or :obj:`pandas.Series`
        Dataframe containing function parameters.
    func : :obj:`callable`
        The result of this function for every row will be returned.
    param2col : :obj:`dict` or :obj:`list`, optional
        Mapping from parameter names of `func` to column names.
        If you pass a list of column names, the columns' values are passed as positional arguments.
        Pass None if you want to use all columns as positional arguments.
    column_wise : :obj:`bool`, optional
        Pass True if you want to map ``func`` to the elements of every column separately.
        This is simply an optimized version of df.apply(func) but allows for naming
        columns to use as function arguments. If param2col is None, ``func`` is mapped
        to the elements of all columns, otherwise to all columns that are not named
        as parameters in ``param2col``.
        In the case where ``func`` does not require a positional first element and
        you want to pass the elements of the various columns as keyword argument,
        give it as param2col={'function_argument': None}
    inplace : :obj:`bool`, optional
        Pass True if you want to mutate ``df`` rather than getting an altered copy.
    **kwargs : Other parameters passed to ``func``.
    """
    if column_wise:
        if not df.__class__ == pd.core.series.Series:
            if param2col is None:
                return df.apply(transform, args=(func,), **kwargs)
            if param2col.__class__ == dict:
                var_arg = [k for k, v in param2col.items() if v is None]
                apply_cols = [col for col in df.columns if not col in param2col.values()]
                assert len(
                    var_arg) < 2, f"Name only one variable keyword argument as which {apply_cols} are used {'argument': None}."
                var_arg = var_arg[0] if len(var_arg) > 0 else getfullargspec(func).args[0]
                param2col = {k: v for k, v in param2col.items() if v is not None}
                result_cols = {col: transform(df, func, {**{var_arg: col}, **param2col}, **kwargs) for col in
                               apply_cols}
                param2col = param2col.values()
            else:
                apply_cols = [col for col in df.columns if not col in param2col]
                result_cols = {col: transform(df, func, [col] + param2col, **kwargs) for col in apply_cols}
            return pd.DataFrame(result_cols, index=df.index)

    if param2col.__class__ == dict:
        param_tuples = list(df[param2col.values()].itertuples(index=False, name=None))
        result_dict = {t: func(**{a: b for a, b in zip(param2col.keys(), t)}, **kwargs) for t in set(param_tuples)}
    else:
        if df.__class__ == pd.core.series.Series:
            if param2col is not None:
                print("When 'df' is a Series, the parameter 'param2col' has no use.")
            param_tuples = df.values
            result_dict = {t: func(t, **kwargs) for t in set(param_tuples)}
        else:
            if param2col is None:
                param_tuples = list(df.itertuples(index=False, name=None))
            else:
                param_tuples = list(df[list(param2col)].itertuples(index=False, name=None))
            result_dict = {t: func(*t, **kwargs) for t in set(param_tuples)}
    res = pd.Series([result_dict[t] for t in param_tuples], index=df.index)
    return res