from collections.abc import Iterable
from fractions import Fraction as frac

import pandas as pd
import numpy as np


def decode_harmonies(df, return_series=False):
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
    if 'harmony_type' in df.columns:
        df.loc[df.harmony_type.isin([1, 2, 3, '1', '2', '3']), 'harmony_type'] == 0
    df.label = label_col
    df.drop(columns=drop_cols, inplace=True)
    return df


def fifths2acc(fifths):
    """ Returns accidentals for a stack of fifths that can be combined with a
        basic representation of the seven steps."""
    return abs(fifths // 7) * 'b' if fifths < 0 else fifths // 7 * '#'


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
        fifths = int(fifths)
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
        fifths = int(fifths)
    except:
        if isinstance(fifths, Iterable):
            return map2elements(fifths, fifths2pc)
        return fifths

    return int(7 * fifths % 12)


def fifths2str(fifths, steps, inverted=False):
    """ Boiler plate used by fifths2-functions.
    """
    fifths += 1
    acc = fifths2acc(fifths)
    if inverted:
        return steps[fifths % 7] + acc
    return acc + steps[fifths % 7]


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
    return pd.read_csv(path, sep=sep, index_col=index_col,
                       dtype=types,
                       converters=conv, **kwargs)


def map2elements(e, f, *args, **kwargs):
    """ If `e` is an iterable, `f` is applied to all elements.
    """
    if isinstance(e, Iterable) and not isinstance(e, str):
        return e.__class__(map2elements(x, f, *args, **kwargs) for x in e)
    return f(e, *args, **kwargs)


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
        midi = int(midi)
    except:
        if isinstance(midi, Iterable):
            return map2elements(midi, midi2octave)
        return midi
    i = -1
    if fifths is not None:
        pc = fifths2pc(fifths)
        if midi % 12 != pc:
            logging.debug(f"midi2octave(): The Tonal Pitch Class {fifths} cannot be MIDI pitch {midi} ")
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
