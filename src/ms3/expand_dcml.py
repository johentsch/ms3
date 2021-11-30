""" This is the same code as in the corpora repo as copied on September 24, 2020
and then adapted.
"""
import sys, re
from collections import defaultdict

import pandas as pd
import numpy as np

from .utils import abs2rel_key, changes2list, DCML_REGEX, features2tpcs, fifths2iv, fifths2name, fifths2pc, fifths2rn, fifths2sd, \
    map2elements, name2fifths, rel2abs_key, resolve_relative_keys, roman_numeral2fifths, \
    series_is_minor, split_alternatives, transform
from .transformations import compute_chord_tones, labels2global_tonic
from .logger import function_logger


################################################################################
# Constants
################################################################################



class SliceMaker(object):
    """ This class serves for storing slice notation such as ``:3`` as a variable or
    passing it as function argument.

    Examples
    --------

    .. code-block:: python

        SM = SliceMaker()
        some_function( slice_this, SM[3:8] )

        select_all = SM[:]
        df.loc[select_all]
    """
    def __getitem__(self, item):
        return item
SM = SliceMaker()


@function_logger
def expand_labels(df, column='label', regex=None, cols={}, dropna=False, propagate=True, volta_structure=None,
                  relative_to_global=False, chord_tones=True, absolute=False, all_in_c=False):
    """
    Split harmony labels complying with the DCML syntax into columns holding their various features
    and allows for additional computations and transformations.

    Uses: :py:func:`compute_chord_tones`, :py:func:`features2type`, :py:func:`~.utils.labels2global_tonic`, :py:func:`propagate_keys`,
    :py:func:`propagate_pedal`, :py:func:`replace_special`, :py:func:`~.utils.roman_numeral2fifths`, :py:func:`~.utils.split_alternatives`, :py:func:`split_labels`,
    :py:func:`~.utils.transform`, :py:func:`transpose`


    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        Dataframe where one column contains DCML chord labels.
    column : :obj:`str`
        Name of the column that holds the harmony labels.
    regex : :obj:`re.Pattern`
        Compiled regular expression used to split the labels. It needs to have named groups.
        The group names are used as column names unless replaced by ``cols``.
    cols : :obj:`dict`, optional
        Dictionary to map the regex's group names to deviating column names of your choice.
    dropna : :obj:`bool`, optional
        Pass True if you want to drop rows where ``column`` is NaN/<NA>
    propagate: :obj:`bool`, optional
        By default, information about global and local keys and about pedal points is spread throughout
        the DataFrame. Pass False if you only want to split the labels into their features. This ignores
        all following parameters because their expansions depend on information about keys.
    volta_structure: :obj:`dict`, optional
        {first_mc -> {volta_number -> [mc1, mc2...]} } dictionary as you can get it from
        ``Score.mscx.volta_structure``. This allows for correct propagation into second and other voltas.
    relative_to_global : :obj:`bool`, optional
        Pass True if you want all labels expressed with respect to the global key.
        This levels and eliminates the features `localkey` and `relativeroot`.
    chord_tones : :obj:`bool`, optional
        Pass True if you want to add four columns that contain information about each label's
        chord, added, root, and bass tones. The pitches are expressed as intervals
        relative to the respective chord's local key or, if ``relative_to_global=True``,
        to the globalkey. The intervals are represented as integers that represent
        stacks of fifths over the tonic, such that 0 = tonic, 1 = dominant, -1 = subdominant,
        2 = supertonic etc.
    absolute : :obj:`bool`, optional
        Pass True if you want to transpose the relative `chord_tones` to the global
        key, which makes them absolute so they can be expressed as actual note names.
        This implies prior conversion of the chord_tones (but not of the labels) to
        the global tonic.
    all_in_c : :obj:`bool`, optional
        Pass True to transpose `chord_tones` to C major/minor. This performs the same
        transposition of chord tones as `relative_to_global` but without transposing
        the labels, too. This option clashes with `absolute=True`.

    Returns
    -------
    :obj:`pandas.DataFrame`
        Original DataFrame plus additional columns with split features.
    """
    assert sum((absolute, all_in_c)) < 2, "Chord tones can be either 'absolute' or 'all_in_c', not both."
    assert len(df.index.names) == 1, f"""df has a MultiIndex of {len(df.index.names)} levels, implying that it has information 
from several pieces. Apply expand_labels() to one piece at a time."""
    df = df.copy()
    if regex is None:
        regex = DCML_REGEX

    ### If the index is not unique, it has to be temporarily replaced
    tmp_index = not df.index.is_unique
    if tmp_index:
        ix = df.index
        df.reset_index(drop=True, inplace=True)


    for col in ['numeral', 'form', 'figbass', 'localkey', 'globalkey', 'phraseend']:
        if not col in cols:
            cols[col] = col
    global_minor = f"{cols['globalkey']}_is_minor"

    ### Check for too many immediate repetitions
    not_nan = df[column].dropna()
    immediate_repetitions = not_nan == not_nan.shift()
    k = immediate_repetitions.sum()
    if k > 0:
        if k / len(not_nan.index) > 0.1:
            logger.warning(
                "DataFrame has many direct repetitions of labels. This function is written for lists of labels only which should have no immediate repetitions.")
        else:
            logger.debug(f"Immediate repetition of labels:\n{not_nan[immediate_repetitions]}")




    ### Do the actual expansion
    df = split_alternatives(df, column=column, logger=logger)
    df = split_labels(df, column=column, regex=regex, cols=cols, dropna=dropna, logger=logger)
    df['chord_type'] = transform(df, features2type, [cols[col] for col in ['numeral', 'form', 'figbass']], logger=logger)
    df = replace_special(df, regex=regex, merge=True, cols=cols, logger=logger)

    ### Check phrase annotations
    p_col = df[cols['phraseend']]
    opening = p_col.fillna('').str.count('{')
    closing = p_col.fillna('').str.count('}')
    if opening.sum() != closing.sum():
        o = df.loc[(opening > 0), ['mc', cols['phraseend']]]
        c = df.loc[(closing > 0), ['mc', cols['phraseend']]]
        compare = pd.concat([o.reset_index(drop=True), c.reset_index(drop=True)], axis=1).astype({'mc': 'Int64'})
        logger.warning(f"Phrase beginning and endings don't match:\n{compare}")

    if propagate:
        key_cols = {col: cols[col] for col in ['localkey', 'globalkey']}
        try:
            df = propagate_keys(df, volta_structure=volta_structure, add_bool=True, **key_cols, logger=logger)
        except:
            logger.error(f"propagate_keys() failed with\n{sys.exc_info()[1]}")
        try:
            df = propagate_pedal(df, cols=cols, logger=logger)
        except:
            logger.error(f"propagate_pedal() failed with\n{sys.exc_info()[1]}")

        if chord_tones:
            ct = compute_chord_tones(df, expand=True, cols=cols, logger=logger)
            if relative_to_global or absolute or all_in_c:
                transpose_by = transform(df, roman_numeral2fifths, [cols['localkey'], global_minor])
                if absolute:
                    transpose_by += transform(df, name2fifths, [cols['globalkey']])
                ct = pd.DataFrame([transpose(tpcs, fifths) for tpcs, fifths in
                                   zip(ct.itertuples(index=False, name=None), transpose_by.values)], index=ct.index,
                                  columns=ct.columns)
            df = pd.concat([df, ct], axis=1)

        if relative_to_global:
            labels2global_tonic(df, inplace=True, cols=cols, logger=logger)
    else:
        if chord_tones:
            logger.info("Chord tones cannot be calculated without propagating keys.")
        if relative_to_global:
            logger.info("Cannot transpose labels without propagating keys.")

    if tmp_index:
        df.index = ix

    return df



def transpose(e, n):
    """ Add `n` to all elements `e` recursively.
    """
    return map2elements(e, lambda x: x + n)




@function_logger
def split_labels(df, column, regex, cols={}, dropna=False, inplace=False, **kwargs):
    """ Split harmony labels complying with the DCML syntax into columns holding their various features.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        Dataframe where one column contains DCML chord labels.
    column : :obj:`str`
        Name of the column that holds the harmony labels.
    regex : :obj:`re.Pattern`
        Compiled regular expression used to split the labels. It needs to have named groups.
        The group names are used as column names unless replaced by `cols`.
    cols : :obj:`dict`
        Dictionary to map the regex's group names to deviating column names.
    dropna : :obj:`bool`, optional
        Pass True if you want to drop rows where ``column`` is NaN/<NA>
    inplace : :obj:`bool`, optional
        Pass True if you want to mutate ``df``.
    """
    if regex.__class__ != re.compile('').__class__:
        regex = re.compile(regex, re.VERBOSE)
    features = regex.groupindex.keys()
    if not inplace:
        df = df.copy()
    if df[column].isna().any():
        if dropna:
            logger.debug(f"Removing NaN values from label column {column}...")
            df = df[df[column].notna()]
        else:
            logger.debug(f"{column} contains NaN values.")

    logger.debug(f"Applying RegEx to column {column}...")
    spl = df[column].str.extract(regex, expand=True)  # .astype('string') # needs pandas 1.0
    for feature in features:
        name = cols[feature] if feature in cols else feature
        df[name] = spl[feature]
    mistakes = spl.isna().apply(all, axis=1) & df[column].notna()
    if mistakes.any():
        logger.warning(f"The following labels do not match the regEx:\n{df.loc[mistakes, :column].to_string()}")
    if not inplace:
        return df


@function_logger
def features2type(numeral, form=None, figbass=None):
    """ Turns a combination of the three chord features into a chord type.

    Returns
    -------
    'M':    Major triad
    'm':    Minor triad
    'o':    Diminished triad
    '+':    Augmented triad
    'mm7':  Minor seventh chord
    'Mm7':  Dominant seventh chord
    'MM7':  Major seventh chord
    'mM7':  Minor major seventh chord
    'o7':   Diminished seventh chord
    '%7':   Half-diminished seventh chord
    '+7':   Augmented (minor) seventh chord
    '+M7':  Augmented major seventh chord
    """
    if pd.isnull(numeral) or numeral in ['Fr', 'Ger', 'It']:
        return numeral
    form, figbass = tuple('' if pd.isnull(val) else val for val in (form, figbass))
    # triads
    if figbass in ['', '6', '64']:
        if form in ['o', '+']:
            return form
        if form in ['%', 'M', '+M']:
            if figbass != '':
                logger.error(f"{form} is a seventh chord and cannot have figbass '{figbass}'")
                return None
            # else: go down, interpret as seventh chord
        else:
            return 'm' if numeral.islower() else 'M'
    # seventh chords
    if form in ['o', '%', '+', '+M']:
        return f"{form}7"
    triad = 'm' if numeral.islower() else 'M'
    seventh = 'M' if form == 'M' else 'm'
    return f"{triad}{seventh}7"


@function_logger
def replace_special(df, regex, merge=False, inplace=False, cols={}, special_map={}):
    """
    | Move special symbols in the `numeral` column to a separate column and replace them by the explicit chords they stand for.
    | In particular, this function replaces the symbols `It`, `Ger`, and `Fr`.

    Uses: :py:func:`merge_changes`

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        Dataframe containing DCML chord labels that have been split by split_labels().
    regex : :obj:`re.Pattern`
        Compiled regular expression used to split the labels replacing the special symbols.It needs to have named groups.
        The group names are used as column names unless replaced by `cols`.
    merge : :obj:`bool`, optional
        False: By default, existing values, except `figbass`, are overwritten.
        True: Merge existing with new values (for `changes` and `relativeroot`).
    cols : :obj:`dict`, optional
        The special symbols appear in the column `numeral` and are moved to the column `special`.
        In case the column names for ``['numeral','form', 'figbass', 'changes', 'relativeroot', 'special']`` deviate, pass a dict, such as

        .. code-block:: python

            {'numeral':         'numeral_col_name',
             'form':            'form_col_name
             'figbass':         'figbass_col_name',
             'changes':         'changes_col_name',
             'relativeroot':    'relativeroot_col_name',
             'special':         'special_col_name'}

    special_map : :obj:`dict`, optional
        In case you want to add or alter special symbols to be replaced, pass a replacement map, e.g.
        {'N': 'bII6'}. The column 'figbass' is only altered if it's None to allow for inversions of special chords.
    inplace : :obj:`bool`, optional
        Pass True if you want to mutate ``df``.
    """
    if not inplace:
        df = df.copy()

    ### If the index is not unique, it has to be temporarily replaced
    tmp_index = not df.index.is_unique
    if tmp_index:
        ix = df.index
        df.reset_index(drop=True, inplace=True)

    special2label = {
        'It': 'viio6(b3)/V',
        'Ger': 'viio65(b3)/V',
        'Fr': 'V7(b5)/V',
    }
    special2label.update(special_map)

    features = ['numeral', 'form', 'figbass', 'changes', 'relativeroot']
    for col in features + ['special']:
        if not col in cols:
            cols[col] = col
    feature_cols = list(cols.values())
    missing = [cols[f] for f in features if not cols[f] in df.columns]
    assert len(
        missing) == 0, f"These columns are missing from the DataFrame: {missing}. Either use split_labels() first or give correct `cols` parameter."
    select_all_special = df[df[cols['numeral']].isin(special2label.keys())].index

    logger.debug(f"Moving special symbols from {cols['numeral']} to {cols['special']}...")
    if not cols['special'] in df.columns:
        df.insert(df.columns.get_loc(cols['numeral']), cols['special'], np.nan)
    df.loc[select_all_special, cols['special']] = df.loc[select_all_special, cols['numeral']]

    def repl_spec(frame, special, instead):
        """Check if the selected parts are empty and replace ``special`` by ``instead``."""
        new_vals = re.match(regex, instead)
        if new_vals is None:
            logger.warning(f"{instead} is not a valid label which could replace {special}. Skipped.")
            return frame
        else:
            new_vals = new_vals.groupdict()
        for f in features:
            if new_vals[f] is not None:
                replace_this = SM[:]  # by default, replace entire column
                if f == 'figbass':                  # only empty figbass is replaced, with the exception of `Ger6` and `Fr6`
                    if special in ['Fr', 'Ger']:    # For these symbols, a wrong `figbass` == 6 is accepted and replaced
                        replace_this = (frame[cols['figbass']] == '6') | frame[cols['figbass']].isna()
                    else:
                        replace_this = frame[cols['figbass']].isna()
                elif f != 'numeral':  # numerals always replaced completely
                    not_empty = frame[cols[f]].notna()
                    if not_empty.any():
                        if f in ['changes', 'relativeroot'] and merge:
                            if f == 'changes':
                                frame.loc[not_empty, cols[f]] = frame.loc[not_empty, cols[f]].apply(merge_changes,
                                                                                                    args=(new_vals[f],))
                            elif f == 'relativeroot':
                                frame.loc[not_empty, cols[f]] = frame.loc[not_empty, cols[f]].apply(
                                    lambda x: f"{new_vals[f]}/{x}")
                            logger.debug(
                                f"While replacing {special}, the existing '{f}'-values have been merged with '{new_vals[f]}', resulting in :\n{frame.loc[not_empty, cols[f]]}")
                            replace_this = ~not_empty
                        else:
                            logger.warning(
                                f"While replacing {special}, the following existing '{f}'-values have been overwritten with {new_vals[f]}:\n{frame.loc[not_empty, cols[f]]}")
                frame.loc[replace_this, cols[f]] = new_vals[f]
        return frame

    for special, instead in special2label.items():
        select_special = df[cols['special']] == special
        df.loc[select_special, feature_cols] = repl_spec(df.loc[select_special, feature_cols].copy(), instead=instead, special=special)

    if df[cols['special']].isna().all():
        df.drop(columns=cols['special'], inplace=True)

    if tmp_index:
        df.index = ix

    if not inplace:
        return df


def merge_changes(left, right, *args):
    """
    Merge two `changes` into one, e.g. `b3` and `+#7` to `+#7b3`.

    Uses: :py:func:`changes2list`
    """
    all_changes = [changes2list(changes, sort=False) for changes in (left, right, *args)]
    res = sum(all_changes, [])
    res = sorted(res, key=lambda x: int(x[3]), reverse=True)
    return ''.join(e[0] for e in res)


@function_logger
def propagate_keys(df, volta_structure=None, globalkey='globalkey', localkey='localkey', add_bool=True):
    """
    | Propagate information about global keys and local keys throughout the dataframe.
    | Pass split harmonies for one piece at a time. For concatenated pieces, use apply().

    Uses: :py:func:`series_is_minor`

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        Dataframe containing DCML chord labels that have been split by split_labels().
    volta_structure: :obj:`dict`, optional
        {first_mc -> {volta_number -> [mc1, mc2...]} } dictionary as you can get it from
        ``Score.mscx.volta_structure``. This allows for correct propagation into
        second and other voltas.
    globalkey, localkey : :obj:`str`, optional
        In case you renamed the columns, pass column names.
    add_bool : :obj:`bool`, optional
        Pass True if you want to add two boolean columns which are true if the respective key is
        a minor key.
    """
    df = df.copy()
    nunique = df[globalkey].nunique()
    assert nunique > 0, "No global key specified."
    if nunique > 1:
        raise NotImplementedError("Several global keys not accepted at the moment.")

    logger.debug('Extending global key to all harmonies')
    global_key = df[globalkey].iloc[0]
    if pd.isnull(global_key):
        global_key = df[globalkey].dropna().iloc[0]
        logger.warning(
            f"Global key is not specified in the first label. Using '{global_key}' from index {df[df[globalkey] == global_key].index[0]}")
    df.loc[:, globalkey] = global_key
    global_minor = series_is_minor(df[globalkey])

    logger.debug('Extending local keys to all harmonies')
    if pd.isnull(df[localkey].iloc[0]):
        one = 'i' if global_minor.iloc[0] else 'I'
        df.iloc[0, df.columns.get_loc(localkey)] = one

    if volta_structure is not None and volta_structure != {}:
        if 'mc' in df.columns:
            volta_mcs = defaultdict(list)
            for volta_dict in volta_structure.values():
                for volta_no, mcs in volta_dict.items():
                    volta_mcs[volta_no].extend(mcs)
            volta_exclusion = {volta_no: [mc for vn, mcs in volta_mcs.items() for mc in mcs if vn != volta_no] for
                               volta_no in volta_mcs.keys()}
            for volta_no in sorted(volta_exclusion.keys(), reverse=True):
                selector = ~df.mc.isin(volta_exclusion[volta_no])
                df.loc[selector, localkey] = df.loc[selector, localkey].fillna(method='ffill')
        else:
            logger.info("Dataframe needs to have a 'mc' column. Ignoring volta_structure.")
            df[localkey].fillna(method='ffill', inplace=True)
    else:
        df[localkey].fillna(method='ffill', inplace=True)

    if add_bool:
        gm = f"{globalkey}_is_minor"
        lm = f"{localkey}_is_minor"
        df[gm] = global_minor
        if df[localkey].str.contains('/').any():
            lk = transform(df, resolve_relative_keys, [localkey, gm])
        else:
            lk = df[localkey]
        local_minor = series_is_minor(lk)
        df[lm] = local_minor
    return df


@function_logger
def propagate_pedal(df, relative=True, drop_pedalend=True, cols={}):
    """
    Propagate the pedal note for all chords within square brackets.
    By default, the note is expressed in relation to each label's localkey.

    Uses: :py:func:`rel2abs_key`, :py:func:`abs2rel_key`

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        Dataframe containing DCML chord labels that have been split by split_labels()
        and where the keys have been propagated using propagate_keys().
    relative : :obj:`bool`, optional
        Pass False if you want the pedal note to stay the same even if the localkey changes.
    drop_pedalend : :obj:`bool`, optional
        Pass False if you don't want the column with the ending brackets to be dropped.
    cols : :obj:`dict`, optional
        In case the column names for ``['pedal','pedalend', 'globalkey', 'localkey']`` deviate, pass a dict, such as

        .. code-block:: python

            {'pedal':       'pedal_col_name',
             'pedalend':    'pedalend_col_name',
             'globalkey':   'globalkey_col_name',
             'localkey':    'localkey_col_name'}
    """
    df = df.copy()
    ### If the index is not unique, it has to be temporarily replaced
    tmp_index = not df.index.is_unique
    if tmp_index:
        ix = df.index
        df.reset_index(drop=True, inplace=True)

    features = ['pedal', 'pedalend', 'globalkey', 'localkey']
    for col in features:
        if not col in cols:
            cols[col] = col
    pedal, pedalend = cols['pedal'], cols['pedalend']

    logger.debug('Extending pedal notes to concerned harmonies')
    beginnings = df.loc[df[pedal].notna(), ['mc', pedal]]
    endings = df.loc[df[pedalend].notna(), ['mc', pedalend]]
    n_b, n_e = len(beginnings), len(endings)

    def make_comparison():
        return pd.concat([beginnings.reset_index(drop=True), endings.reset_index(drop=True)], axis=1).astype({'mc': 'Int64'})

    assert n_b == n_e, f"{n_b} organ points started, {n_e} ended:\n{make_comparison()}"
    if relative:
        assert df[cols[
            'localkey']].notna().all(), "Local keys must first be propagated using propagate_keys(), no NaNs allowed."

    for (fro, ped), to in zip(beginnings[pedal].items(), endings[pedalend].index):
        try:
            section = df.loc[fro:to].index
        except:
            logger.error(
                f"Slicing of the DataFrame did not work from {fro} to {to}. Index looks like this:\n{df.head().index}")
        localkeys = df.loc[section, cols['localkey']]
        if localkeys.nunique() > 1:
            first_localkey = localkeys.iloc[0]
            globalkeys = df.loc[section, cols['globalkey']].unique()
            assert len(globalkeys) == 1, "Several globalkeys appearing within the same organ point."
            global_minor = globalkeys[0].islower()
            key2pedal = {
                key: ped if key == first_localkey else abs2rel_key(rel2abs_key(ped, first_localkey, global_minor), key,
                                                                   global_minor) for key in localkeys.unique()}
            logger.debug(
                f"Pedal note {ped} has been transposed relative to other local keys within a global {'minor' if global_minor else 'major'} context: {key2pedal}")
            pedals = pd.Series([key2pedal[key] for key in localkeys], index=section)
        else:
            pedals = pd.Series(ped, index=section)
        df.loc[section, pedal] = pedals

    if drop_pedalend:
        df = df.drop(columns=pedalend)

    if tmp_index:
        df.index = ix

    return df


########################################################################################################################
#                                           MOMENTARILY NOT IN USE:
########################################################################################################################

def transform_columns(df, func, columns=None, param2col=None, inplace=False, **kwargs):
    """ Wrapper function to use transform() on df[columns], leaving the other columns untouched.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        DataFrame where columns (or column combinations) work as function arguments.
    func : :obj:`callable`
        Function you want to apply to all elements in `columns`.
    columns : :obj:`list`
        Columns to which you want to apply `func`.
    param2col : :obj:`dict` or :obj:`list`, optional
        Mapping from parameter names of `func` to column names.
        If you pass a list of column names, the columns' values are passed as positional arguments.
        Pass None if you want to use all columns as positional arguments.
    inplace : :obj:`bool`, optional
        Pass True if you want to mutate `df` rather than getting an altered copy.
    **kwargs: keyword arguments for transform()
    """
    if not inplace:
        df = df.copy()

    param_cols = []
    if columns is None:
        columns = df.columns
    elif param2col is None:
        pass
    elif param2col.__class__ == dict:
        param_cols = list(param2col.values())
    else:
        param_cols = list(param2col)

    tmp_index = not df.index.is_unique
    if tmp_index:
        ix = df.index
        df.reset_index(drop=True, inplace=True)

    df.loc[:, columns] = transform(df[columns + param_cols], func, param2col, **kwargs)

    if tmp_index:
        df.index = ix

    if not inplace:
        return df



def transform_note_columns(df, to, note_cols=['chord_tones', 'added_tones', 'bass_note', 'root'], minor_col='localkey_is_minor', inplace=False, **kwargs):
    """
    Turns columns with line-of-fifth tonal pitch classes into another representation.

    Uses: transform_columns()

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        DataFrame where columns (or column combinations) work as function arguments.
    to : {'name', 'iv', 'pc', 'sd', 'rn'}
        The tone representation that you want to get from the `note_cols`.

        * 'name': Note names. Should only be used if the stacked fifths actually represent
                absolute tonal pitch classes rather than intervals over the local tonic.
                In other words, make sure to use 'name' only if 0 means C rather than I.
        * 'iv':   Intervals such that 0 = 'P1', 1 = 'P5', 4 = 'M3', -3 = 'm3', 6 = 'A4',
                -6 = 'D5' etc.
        * 'pc':   (Relative) chromatic pitch class, or distance from tonic in semitones.
        * 'sd':   Scale degrees such that 0 = '1', -1 = '4', -2 = 'b7' in major, '7' in minor etc.
                This representation requires a boolean column `minor_col` which is
                True in those rows where the stacks of fifths occur in a local minor
                context and False for the others. Alternatively, if all pitches are
                in the same mode or you simply want to express them as degrees of
                particular mode, you can pass the boolean keyword argument `minor`.
        * 'rn':   Roman numerals such that 0 = 'I', -2 = 'bVII' in major, 'VII' in minor etc.
                Requires boolean 'minor' values, see 'sd'.

    note_cols : :obj:`list`, optional
        List of columns that hold integers or collections of integers that represent
        stacks of fifth (0 = tonal center, 1 = fifth above, -1 = fourth above, etc).
    minor_col : :obj:`str`, optional
        If `to` is 'sd' or 'rn', specify a boolean column where the value is
        True in those rows where the stacks of fifths occur in a local minor
        context and False for the others.

    """
    transformations = {
        'name': fifths2name,
        'names': fifths2name,
        'iv': fifths2iv,
        'pc': fifths2pc,
        'sd': fifths2sd,
        'rn': fifths2rn,
    }
    assert to in transformations, "Parameter to needs to be one of {'name', 'iv', 'pc', 'sd', 'rn'}"
    cols = [col for col in note_cols if col in df.columns]
    if len(cols) < len(note_cols):
        logger.warning(f"Columns {[[col for col in note_cols if not col in df.columns]]}")
    param2col = None
    if to in ['sd', 'rn']:
        assert minor_col in df.columns or 'minor' in kwargs, f"'{to} representation requires a boolean column for the 'minor' argument, e.g. 'globalkey_is_minor'."
        if not 'minor' in kwargs:
            param2col = {'minor': minor_col}
    func = transformations[to]
    res = transform_columns(df, func, columns=note_cols, inplace=inplace, param2col=param2col, column_wise=True, **kwargs)
    if not inplace:
        return res


@function_logger
def chord2tpcs(chord, regex=None, **kwargs):
    """
    Split a chord label into its features and apply features2tpcs().

    Uses: features2tpcs()

    Parameters
    ----------
    chord : :obj:`str`
        Chord label that can be split into the features ['numeral', 'form', 'figbass', 'changes', 'relativeroot'].
    regex : :obj:`re.Pattern`, optional
        Compiled regex with named groups for the five features. By default, the current version of the DCML harmony
        annotation standard is used.
    **kwargs :
        arguments for features2tpcs (pass MC to show it in warnings!)
    """
    if regex is None:
        regex = DCML_REGEX
    chord_features = re.match(regex, chord)
    assert chord_features is not None, f"{chord} does not match the regex."
    chord_features = chord_features.groupdict()
    numeral, form, figbass, changes, relativeroot = tuple(chord_features[f] for f in ('numeral', 'form', 'figbass', 'changes', 'relativeroot'))
    return features2tpcs(numeral=numeral, form=form, figbass=figbass, changes=changes, relativeroot=relativeroot,
                         logger=logger, **kwargs)
