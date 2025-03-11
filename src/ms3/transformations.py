"""Functions for transforming DataFrames as output by ms3."""

import logging
import sys
import warnings
from fractions import Fraction as frac
from functools import reduce
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ms3.logger import get_logger

from .utils import (
    adjacency_groups,
    features2tpcs,
    fifths2iv,
    fifths2name,
    fifths2pc,
    fifths2rn,
    fifths2sd,
    make_continuous_offset_series,
    make_interval_index_from_breaks,
    make_interval_index_from_durations,
    make_offset_dict_from_measures,
    make_playthrough_info,
    midi2octave,
    name2fifths,
    nan_eq,
    overlapping_chunk_per_interval,
    reduce_dataframe_duration_to_first_row,
    rel2abs_key,
    replace_index_by_intervals,
    resolve_relative_keys,
    roman_numeral2fifths,
    roman_numeral2semitones,
    series_is_minor,
    tpc2name,
    transform,
    transpose,
    transpose_changes,
    unfold_repeats,
)

module_logger = get_logger(__name__)


def add_localkey_change_column(at, key_column="localkey"):
    key_segment, _ = adjacency_groups(at[key_column], na_values="ffill")
    return pd.concat([at, key_segment.rename("key_segment")], axis=1)


def make_note_name_and_octave_columns(
    notes: pd.DataFrame,
    staff2drums: Optional[Dict[int, Union[dict, pd.DataFrame, pd.Series]]] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Takes a notelist and maybe a {staff -> {midi_pitch -> 'instrument_name'}} mapping and returns two columns
    named 'name' and 'octave'."""
    octaves = midi2octave(notes.midi, notes.tpc).rename("octave")
    names = (tpc2name(notes.tpc) + octaves.astype(str)).rename("name")
    if staff2drums is None or len(staff2drums) == 0:
        return names, octaves
    result = pd.Series(index=notes.index, dtype="string", name="name")
    for staff, drum_map in staff2drums.items():
        if isinstance(drum_map, pd.DataFrame):
            drum_map = drum_map.name
        drum_notes = notes.staff == staff
        result.loc[drum_notes] = notes.midi[drum_notes].map(drum_map)
    not_drum = result.isna()
    if not_drum.sum() > 0:
        result = result.fillna(names)
    if not not_drum.all():
        is_drum = ~not_drum
        octaves = octaves.astype("Int64")
        octaves.loc[is_drum] = pd.NA
    return result, octaves


def add_quarterbeats_col(
    df: pd.DataFrame,
    offset_dict: pd.Series | dict,
    offset_dict_all_endings: Optional[pd.Series | dict] = None,
    interval_index: bool = False,
    name: Optional[str] = None,
    logger=None,
) -> pd.DataFrame:
    """Insert a column measuring the distance of events from MC 1 in quarter notes. If no 'mc_onset' column is present,
        the column corresponds to the values given in the offset_dict.

    Args:
        df: DataFrame with an ``mc`` or ``mc_playthrough`` column, and an ``mc_onset`` column.
        offset_dict:
            | If unfolded: {mc_playthrough -> offset}
            | Otherwise: {mc -> offset}
            | You can create the dict using the functions make_continuous_offset_series() or
              make_offset_dict_from_measures().
            | It is not required if the column 'quarterbeats' exists already.
        offset_dict_all_endings:
            Argument added later as a straightforward way to add two quarterbeats columns, the second one being the
            'quarterbeats_all_endings' which is so important that with ms3 v2.2.0 it is included by default.
            It is independent from unfolding because its main purpose is score addressability.
        interval_index:
            Defaults to False. Pass True to replace the index with an :obj:`pandas.IntervalIndex` (depends on the
            successful creation of the column ``duration_qb``).
        name:
            If specified, name of the added column. Defaults to 'quarterbeats' for normal, and
            'quarterbeats_playthrough' for unfolded dataframes.
        logger:

    Returns:
        The DataFrame with quarterbeats and duration_qb columns added.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if len(df.index) == 0:
        return df

    is_unfolded = "mc_playthrough" in df.columns
    qb_column_name = "quarterbeats"
    mc_col = "mc"
    if is_unfolded:
        mc_col = "mc_playthrough"
        qb_column_name = "quarterbeats_playthrough"
    elif "mc" in df.columns:
        mc_col = "mc"
    if name is None:
        name = qb_column_name
    if not any(col in df.columns for col in ("mc", "mc_playthrough", name)):
        logger.error(
            "Expected to have at least one column called 'mc' or 'mc_playthrough'."
        )
        return df

    has_quarterbeats = name in df.columns
    has_duration_qb = "duration_qb" in df.columns
    has_duration = "duration" in df.columns
    if (
        sum((has_quarterbeats, has_duration_qb)) == 2
        and offset_dict_all_endings is None
    ):
        if interval_index:
            logger.debug(
                f"{name} and 'duration_qb' already present, only creating IntervalIndex"
            )
            return replace_index_by_intervals(df, logger=logger)
        else:
            logger.debug(f"{name} and 'duration_qb' already present, nothing to do.")
            return df
    if offset_dict is None:
        if not has_quarterbeats:
            logger.warning("No offset_dict was passed: Not adding quarterbeats.")
            return df
        if not has_duration:
            logger.warning(
                "Could not create 'duration_qb' because no offset_dict was passed and no 'duration' column is present."
            )
            return df

    new_cols = {}
    if has_quarterbeats:
        new_cols[name] = df[name]
    if has_duration_qb:
        new_cols["duration_qb"] = df.duration_qb
    if len(new_cols) > 0:
        # removes existing columns because they will be reinserted later in their canonical position
        df = df.drop(columns=list(new_cols.keys()))

    if "mc_onset" in df.columns:
        mc_onset_column = df.mc_onset
    else:
        mc_onset_column = None
    if not has_quarterbeats and mc_col in df.columns:
        new_cols[name] = make_quarterbeats_column(
            mc_column=df[mc_col],
            mc_onset_column=mc_onset_column,
            offset_dict=offset_dict,
            name=name,
        )
    if offset_dict_all_endings is not None and "mc" in df.columns:
        new_cols["quarterbeats_all_endings"] = make_quarterbeats_column(
            mc_column=df["mc"],
            mc_onset_column=mc_onset_column,
            offset_dict=offset_dict_all_endings,
            name="quarterbeats_all_endings",
        )
    if not has_quarterbeats or not has_duration_qb:
        # recreate duration_qb also when quarterbeats had been missing
        if "duration" in df.columns:
            new_cols["duration_qb"] = (
                (df.duration * 4).astype(float).rename("duration_qb")
            )
        elif "end" in offset_dict:
            quarterbeats = new_cols[name]
            present_qb = quarterbeats.notna()
            selected_qb = quarterbeats[present_qb].astype(float)
            qb_are_sorted = (selected_qb.sort_values().index == selected_qb.index).all()
            if not qb_are_sorted:
                unordered_index_pairs = [
                    [ix_a, ix_b]
                    for (ix_a, a), (ix_b, b) in zip(
                        selected_qb.items(), selected_qb.iloc[1:].items()
                    )
                    if b < a
                ]
                n_problems = len(unordered_index_pairs)
                if n_problems > 0:
                    fails_df = pd.concat(
                        [df.loc[fail] for fail in unordered_index_pairs],
                        keys=range(n_problems),
                        names=["problem", "index"],
                    )
                    logger.warning(
                        f"In the following instances the second event occurs before the first one: \n"
                        f"{fails_df.iloc[:, :12].to_string()}\n"
                        f"Each second row will not have a 'duration_qb' value.",
                        extra={"message_id": (28,)},
                    )
                    present_and_in_the_right_order = present_qb & ~(
                        selected_qb > selected_qb.shift(-1)
                    )
                    selected_qb = quarterbeats[present_and_in_the_right_order].astype(
                        float
                    )
                else:
                    # the index changed when sorting values but only because some quarterbeats were identical
                    qb_are_sorted = True
            try:
                ivs = make_interval_index_from_breaks(
                    selected_qb, end_value=float(offset_dict["end"]), logger=logger
                )
                duration_qb = pd.Series(pd.NA, index=df.index, name="duration_qb")
                if qb_are_sorted:
                    duration_qb.loc[present_qb] = ivs.length
                else:
                    duration_qb.loc[present_and_in_the_right_order] = ivs.length
                new_cols["duration_qb"] = duration_qb
            except Exception as e:
                logger.warning(
                    f"Error while creating durations from quarterbeats column. Error:\n{e}"
                )
        else:
            logger.warning(
                "Could not create 'duration_qb' because offset_dict does not contain the key 'end'."
            )
    if len(new_cols) > 0:
        insert_after = next(
            col
            for col in ("mn_playthrough", "mc_playthrough", "mn", "mc")
            if col in df.columns
        )
        logger.debug(f"Inserting {list(new_cols.keys())} after '{insert_after}'")
        insert_position = df.columns.get_loc(insert_after) + 1
        left, right = df.iloc[:, :insert_position], df.iloc[:, insert_position:]
        df = pd.concat([left] + list(new_cols.values()) + [right], axis=1)
        logger.debug(f"Inserted the columns {list(new_cols.keys())}.")
    else:
        logger.debug("No columns were added.")
    if interval_index:
        if is_unfolded:
            position_col = name
        elif "quarterbeats_all_endings" in df.columns:
            position_col = "quarterbeats_all_endings"
        else:
            position_col = name
        if all(c in df.columns for c in (position_col, "duration_qb")):
            df = replace_index_by_intervals(
                df, position_col=position_col, logger=logger
            )
            logger.debug(f"IntervalIndex created based on the column {position_col!r}.")
        else:
            logger.warning(
                f"Cannot create interval index because column {position_col!r} is missing."
            )
    if interval_index and all(c in df.columns for c in (name, "duration_qb")):
        df = replace_index_by_intervals(df, position_col=name, logger=logger)
    return df


def make_quarterbeats_column(
    mc_column: pd.Series,
    mc_onset_column: Optional[pd.Series],
    offset_dict: pd.Series | dict,
    name: str = "quarterbeats",
) -> pd.Series:
    """Turn each combination of mc and mc_onset into a quarterbeat value using the offset_dict that maps mc to
    the measure's quarterbeat position (distance from the beginning of the piece).

    Args:
        mc_column: A sequence of MC values, each of which will be mapped to its quarterbeats value in ``offset_dict``.
        mc_onset_column: If specified, these values will be added to the mapped quarterbeats values.
        offset_dict: {mc -> quarterbeats}, can be a Series.
        name: Name of the returned Series.

    Returns:
        Quarterbeats column.
    """
    quarterbeats = mc_column.map(offset_dict)
    if mc_onset_column is not None:
        quarterbeats += mc_onset_column * 4
    return quarterbeats.rename(name)


def add_weighted_grace_durations(notes, weight=1 / 2, logger=None):
    """For a given notes table, change the 'duration' value of all grace notes, weighting it by ``weight``.

    Parameters
    ----------
    notes : :obj:`pandas.DataFrame`
        Notes table containing the columns 'duration', 'nominal_duration', 'scalar'
    weight : :obj:`Fraction` or :obj:`float`
        Value by which to weight duration of all grace notes. Defaults to a half.

    Returns
    -------
    :obj:`pandas.DataFrame`
        Copy of ``notes`` with altered duration values.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if "gracenote" not in notes.columns:
        return notes
    grace = notes.gracenote.notna()
    if grace.sum() == 0:
        return notes
    notes = notes.copy()
    assert all(
        col in notes.columns for col in ("duration", "nominal_duration", "scalar")
    ), "Needs to contain columns 'duration', 'nominal_duration', 'scalar'"
    gracenotes = notes[grace]
    new_durations = gracenotes.scalar * gracenotes.nominal_duration * frac(weight)
    notes.loc[grace, "duration"] = new_durations
    if "duration_qb" in notes.columns:
        notes.loc[grace, "duration_qb"] = (new_durations * 4).astype(float)
        idx = notes.index
        if idx.nlevels == 1:
            if isinstance(idx, pd.IntervalIndex):
                notes = replace_index_by_intervals(notes, logger=logger)
                logger.debug("Updated existing interval index.")
            else:
                logger.debug("No interval index present to be updated.")
        else:
            if "interval" in idx.names:
                interval_idx_levels = [
                    i for i, name in enumerate(idx.names) if name == "interval"
                ]
            else:
                interval_idx_levels = [
                    i
                    for i, dtype in enumerate(idx.dtypes)
                    if isinstance(dtype, pd.IntervalDtype)
                ]
            if len(interval_idx_levels) == 0:
                logger.debug("No interval index present to be updated.")
            elif len(interval_idx_levels) == 1:
                interval_idx_level = interval_idx_levels[0]
                logger.debug(
                    f"Updating the IntervalIndex at level {interval_idx_level} ('{idx.names[interval_idx_level]}')."
                )
                new_iv_idx = make_interval_index_from_durations(notes, logger=logger)
                idx_df = idx.to_frame()
                idx_df.iloc[:, interval_idx_level] = new_iv_idx
                notes.index = pd.MultiIndex.from_frame(idx_df, names=idx.names)
            else:
                logger.warning(
                    "Found several IntervalIndex levels, none of which is called 'interval'."
                )
    return notes


def compute_chord_tones(
    df: pd.DataFrame,
    bass_only: bool = False,
    expand: bool = False,
    cols: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame | pd.Series:
    """
    Compute the chord tones for DCML harmony labels. They are returned as lists
    of tonal pitch classes in close position, starting with the bass note. The
    tonal pitch classes represent intervals relative to the local tonic:

    -2: Second below tonic
    -1: fifth below tonic
    0: tonic
    1: fifth above tonic
    2: second above tonic, etc.

    The labels need to have undergone :py:func:`split_labels` and :py:func:`propagate_keys`.
    Pedal points are not taken into account.

    Uses: :py:func:`features2tpcs`

    Args:
        df:
            Dataframe containing DCML chord labels that have been split by :func:`split_labels`
            and where the keys have been propagated using propagate_keys(add_bool=True).
        bass_only: Pass True if you need only the bass note.
        expand: Pass True if you need chord tones and added tones in separate columns. Otherwise a Series is returned.
        cols:
            In case the column names for ``['mc', 'numeral', 'form', 'figbass', 'changes', 'relativeroot', 'localkey',
            'globalkey']`` deviate, pass a dict, such as
        logger:

    Returns:
        For every row of `df` one tuple with chord tones, expressed as tonal pitch classes.
        If `expand` is True, the function returns a DataFrame with four columns:
        Two with tuples for chord tones and added tones, one with the chord root,
        and one with the bass note.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if cols is None:
        cols = {}
    df = df.copy()
    # If the index is not unique, it has to be temporarily replaced
    tmp_index = not df.index.is_unique
    if tmp_index:
        ix = df.index
        df.reset_index(drop=True, inplace=True)

    features = [
        "mc",
        "numeral",
        "form",
        "figbass",
        "changes",
        "relativeroot",
        "localkey",
        "globalkey",
    ]
    for col in features:
        if col in df.columns and col not in cols:
            cols[col] = col
    local_minor, global_minor = (
        f"{cols['localkey']}_is_minor",
        f"{cols['globalkey']}_is_minor",
    )
    if local_minor not in df.columns:
        df[local_minor] = series_is_minor(df[cols["localkey"]], is_name=False)
        logger.debug(f"Boolean column '{local_minor}' created.'")
    if global_minor not in df.columns:
        df[global_minor] = series_is_minor(df[cols["globalkey"]], is_name=True)
        logger.debug(f"Boolean column '{global_minor}' created.'")

    param_cols = {
        col: cols[col]
        for col in ["numeral", "form", "figbass", "changes", "relativeroot", "mc"]
        if col in cols and cols[col] is not None
    }
    param_cols["minor"] = local_minor
    param_tuples = list(df[param_cols.values()].itertuples(index=False, name=None))
    # result_dict = {t: features2tpcs(**{a:b for a, b in zip(param_cols.keys(), t)}, bass_only=bass_only,
    # merge_tones=not expand, logger=logger) for t in set(param_tuples)}
    result_dict = {}
    if bass_only:
        default = None
    elif not expand:
        default = tuple()
    else:
        default = {
            "chord_tones": tuple(),
            "added_tones": tuple(),
            "root": None,
        }
    for t in set(param_tuples):
        try:
            result_dict[t] = features2tpcs(
                **{a: b for a, b in zip(param_cols.keys(), t)},
                bass_only=bass_only,
                merge_tones=not expand,
                logger=logger,
            )
        except Exception:
            result_dict[t] = default
            logger.warning(str(sys.exc_info()[1]))
    if expand:
        res = pd.DataFrame([result_dict[t] for t in param_tuples], index=df.index)
        res["bass_note"] = res.chord_tones.apply(
            lambda lst: pd.NA if pd.isnull(lst) or len(lst) == 0 else lst[0]
        )
        res[["root", "bass_note"]] = res[["root", "bass_note"]].astype("Int64")
    else:
        res = pd.Series([result_dict[t] for t in param_tuples], index=df.index)

    if tmp_index:
        res.index = ix

    return res


def dfs2quarterbeats(
    dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    measures: pd.DataFrame,
    unfold=False,
    quarterbeats=True,
    interval_index=True,
    logger=None,
) -> List[pd.DataFrame]:
    """Pass one or several DataFrames and one measures table to unfold repeats and/or add quarterbeats columns and/or
    index.

    Args:
      dfs: DataFrame(s) that are to be unfolded and/or receive quarterbeats.
      measures:
      unfold:
      quarterbeats:
      interval_index:

    Returns:
      Altered copies of `dfs`.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    assert (
        sum((unfold, quarterbeats, interval_index)) >= 1
    ), "At least one of the 'unfold', 'quarterbeats', and 'interval_index' arguments needs to be True."
    assert measures is not None, "measures cannot be None"
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    if interval_index:
        quarterbeats = True
    if unfold:
        playthrough_info = make_playthrough_info(measures, logger=logger)
        if playthrough_info is None or len(playthrough_info) == 0:
            logger.warning(
                "Unfolding not possible because of incorrect repeat structure."
            )
            return []
    offset_dict_all_endings = make_offset_dict_from_measures(
        measures,
        all_endings=True,
    )
    if unfold:
        dfs = [
            (
                unfold_repeats(df, playthrough_info, logger=logger)
                if df is not None
                else df
            )
            for df in dfs
        ]
        if quarterbeats:
            unfolded_measures = unfold_repeats(
                measures, playthrough_info, logger=logger
            )
            offset_dict = make_continuous_offset_series(
                unfolded_measures, logger=logger
            )

            dfs = [
                (
                    add_quarterbeats_col(
                        df,
                        offset_dict=offset_dict,
                        offset_dict_all_endings=offset_dict_all_endings,
                        interval_index=interval_index,
                        logger=logger,
                    )
                    if df is not None
                    else df
                )
                for df in dfs
            ]
    elif quarterbeats:
        if "volta" in measures.columns:
            logger.debug(
                "No quarterbeats are assigned to first endings. Pass unfold=True to "
                "compute quarterbeats for a full playthrough."
            )
            if 3 in measures.volta.values:
                logger.debug(
                    "Piece contains third endings, note that only second endings are taken into account."
                )
            measures = measures.drop(
                index=measures[measures.volta.fillna(2) != 2].index, columns="volta"
            )
        offset_dict = make_continuous_offset_series(measures, logger=logger)
        dfs = [
            (
                add_quarterbeats_col(
                    df,
                    offset_dict=offset_dict,
                    offset_dict_all_endings=offset_dict_all_endings,
                    interval_index=interval_index,
                    logger=logger,
                )
                if df is not None
                else df
            )
            for df in dfs
        ]
    return dfs


def get_chord_sequences(at, major_minor=True, level=None, column="chord", logger=None):
    """Transforms an annotation table into lists of chord symbols for n-gram analysis. If your table represents
    several pieces, make sure to pass the groupby parameter ``level`` to avoid including inexistent transitions.

    Parameters
    ----------
    at : :obj:`pandas.DataFrame`
        Annotation table.
    major_minor : :obj:`bool`, optional
        | Defaults to True: the length of the chord sequences corresponds to localkey segments. The result comes as
        dict of dicts.
        | If you pass False, chord sequences are returned as they are, potentially including incorrect transitions,
        e.g., when
          the localkey changes. The result comes as list of lists, where the sublists result from the groupby if you
          specified ``level``.
    level : :obj:`int` or :obj:`list`
        Argument passed to :py:meth:`pandas.DataFrame.groupby`. Defaults to -1, resulting in a GroupBy by all levels
        except the last. Conversely, you can pass, for instance, 2 to group by the first two levels.
    column : :obj:`str`
        Name of the column containing the chord symbols that compose the sequences.

    Returns
    -------
    :obj:`dict` of :obj:`dict` or :obj:`list` of :obj:`list`
        | If ``major_minor`` is True, the sequences are returned as {:obj:`int` -> {'localkey' -> :obj:`str`,
        'localkey_is_minor' -> :obj:`bool`, 'sequence' -> :obj:`list`} }
        | If False, the sequences are returned as a list of lists

    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if major_minor:
        if "key_segment" not in at.columns:
            logger.info(
                "The DataFrame does not include the column 'key_segment' which is used for processing localkey "
                "segments. "
                "If you need to access the regions, add the column using transform_multiple(df, 'key_segment')"
            )
            if level is not None:
                at = transform_multiple(at, "key_segment", level=level)
            else:
                at = add_localkey_change_column(at)
        res = {}
        for i, df in at.groupby("key_segment"):
            row = df.iloc[0]
            sequence = {
                "localkey": row.localkey,
                "localkey_is_minor": row.localkey_is_minor,
                "sequence": df[column].to_list(),
            }
            res[i] = sequence
        return res
    else:
        if level is not None:
            levels = _treat_level_parameter(level, at.index.nlevels)
            sequences = at.groupby(level=levels)[column].apply(list).to_list()
        else:
            sequences = [at[column].to_list()]
        return sequences


def group_annotations_by_features(at, features="numeral", logger=None):
    """Drop exact repetitions of one or several feature columns when occurring under the same localkey (and pedal
    point).
    For example, pass ``features = ['numeral', 'form', 'figbass']`` to drop rows where all three features are identical
    with the previous row _and_ the localkey stays the same. If the column ``duration_qb`` is present, it is updated
    with the new durations, as would be the IntervalIndex if there is one.
    Uses: nan_eq()

    Parameters
    ----------
    at : :obj:`pandas.DataFrame`
        Annotation table
    features : :obj:`str` or :obj:`list`
        Feature or feature combination for which to remove immediate repetitions
    dropna : :obj:`bool`
        Also subsumes rows for which all ``features`` are NaN, rather than treating them as a new value.

    Returns
    -------
    :obj:`pandas.DataFrame`

    Example
    -------

    >>> df
    +--------------+--------------+-------------+----------+---------------+---------+------+---------+---------+--------------+
    |              | quarterbeats | duration_qb | localkey | chord         | numeral | form | figbass | changes |
    relativeroot |
    +==============+==============+=============+==========+===============+=========+======+=========+=========+==============+
    | [37.5, 38.5) | 75/2         | 1.0         | I        | viio65(6b3)/V | vii     | o    | 65      | 6b3     | V
             |
    +--------------+--------------+-------------+----------+---------------+---------+------+---------+---------+--------------+
    | [38.5, 40.5) | 77/2         | 2.0         | I        | Ger           | vii     | o    | 65      | b3      | V
             |
    +--------------+--------------+-------------+----------+---------------+---------+------+---------+---------+--------------+
    | [40.5, 41.5) | 81/2         | 1.0         | I        | V(7v4)        | V       |      |         | 7v4     |
             |
    +--------------+--------------+-------------+----------+---------------+---------+------+---------+---------+--------------+
    | [41.5, 43.5) | 83/2         | 2.0         | I        | V(64)         | V       |      |         | 64      |
             |
    +--------------+--------------+-------------+----------+---------------+---------+------+---------+---------+--------------+
    | [43.5, 44.5) | 87/2         | 1.0         | I        | V7(9)         | V       |      | 7       | 9       |
             |
    +--------------+--------------+-------------+----------+---------------+---------+------+---------+---------+--------------+
    | [44.5, 46.5) | 89/2         | 2.0         | I        | V7            | V       |      | 7       |         |
             |
    +--------------+--------------+-------------+----------+---------------+---------+------+---------+---------+--------------+
    | [46.5, 48.0) | 93/2         | 1.5         | I        | I             | I       |      |         |         |
             |
    +--------------+--------------+-------------+----------+---------------+---------+------+---------+---------+--------------+

    >>> group_annotations_by_features(df)
    +--------------+--------------+-------------+----------+--------------+---------+-------+
    |              | quarterbeats | duration_qb | localkey | relativeroot | numeral | chord |
    +==============+==============+=============+==========+==============+=========+=======+
    | [37.5, 40.5) | 75/2         | 3.0         | I        | V            | vii     | vii/V |
    +--------------+--------------+-------------+----------+--------------+---------+-------+
    | [40.5, 46.5) | 81/2         | 6.0         | I        | NaN          | V       | V     |
    +--------------+--------------+-------------+----------+--------------+---------+-------+
    | [46.5, 48.0) | 93/2         | 1.5         | I        | NaN          | I       | I     |
    +--------------+--------------+-------------+----------+--------------+---------+-------+

    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if isinstance(features, str):
        features = [features]
    qb_cols = ["quarterbeats", "duration_qb"]
    safety_cols = [
        "globalkey",
        "localkey",
        "pedal",
    ]  # if present, these prevent grouping because they change meaning
    keep_cols = [
        "mc",
        "mn",
        "mc_onset",
        "mn_onset",
        "timesig",
        "staff",
        "voice",
        "volta",
        "label",
        "globalkey",
        "localkey",
        "globalkey_is_minor",
        "localkey_is_minor",
        "special",
        "pedal",
    ]
    if "numeral" in features:
        safety_cols.append("relativeroot")
        keep_cols.append("relativeroot")
    safety = [
        f
        for f in safety_cols
        if f not in features and f in at.columns and len(at[f].unique()) > 1
    ]
    cols = []
    for f in features + safety:
        if f in at.columns:
            cols.append(f)
        else:
            print(f"DataFrame has no column called {f}")
    features = [f for f in features if f in at.columns]
    if len(features) == 0:
        return at

    if all(c in at.columns for c in qb_cols):
        na_values = ["ffill" if c == "numeral" else "group" for c in cols]
        logger.info(
            f"Grouping with the following na_values settings: {dict(zip(cols, na_values))}"
        )
        res = segment_by_adjacency_groups(at, cols, na_values=na_values, logger=logger)
        keep_cols = qb_cols + keep_cols
    else:

        def column_shift_mask(a, b):
            """Sets those values of Series a to True where a value in the Series b is different from its predecessor."""
            return a | ~nan_eq(b, b.shift())

        # The change mask is True for every row where either of the feature or safety columns is different from its
        # previous value.
        change_mask = reduce(
            column_shift_mask,
            (col for _, col in at[cols].items()),
            pd.Series(False, index=at.index),
        )
        res = at[change_mask]
    keep_cols = [c for c in keep_cols if c in at.columns] + features
    res = res[keep_cols]
    chord_components = [
        f
        for f in (
            "root",
            "bass_note",
            "root_name",
            "bass_note_name",
            "numeral",
            "form",
            "chord_type",
            "figbass",
            "changes",
        )
        if f in features
    ]
    if "numeral" in res.columns:
        chord_components.append("relativeroot")
    chord_col = make_chord_col(res, chord_components)
    res = pd.concat([res, chord_col], axis=1)
    if "numeral" in res.columns:
        try:
            res = pd.concat([res, compute_chord_tones(res, expand=True)], axis=1)
        except Exception:
            pass
    return res


def labels2global_tonic(df, cols={}, inplace=False, logger=None):
    """
    Transposes all numerals to their position in the global major or minor scale.
    This eliminates localkeys and relativeroots. The resulting chords are defined
    by [`numeral`, `figbass`, `changes`, `globalkey_is_minor`] (and `pedal`).

    Uses: :py:func:`transform`, :py:func:`rel2abs_key^, :py:func:`resolve_relative_keys` -> :py:func:`str_is_minor()`
    :py:func:`transpose_changes`, :py:func:`series_is_minor`,

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        Dataframe containing DCML chord labels that have been split by split_labels()
        and where the keys have been propagated using propagate_keys(add_bool=True).
    cols : :obj:`dict`, optional
        In case the column names for ``['numeral', 'form', 'figbass', 'changes', 'relativeroot', 'localkey',
        'globalkey']`` deviate, pass a dict, such as

        .. code-block:: python

            {'chord':           'chord_col_name'
             'pedal':           'pedal_col_name',
             'numeral':         'numeral_col_name',
             'form':            'form_col_name',
             'figbass':         'figbass_col_name',
             'changes':         'changes_col_name',
             'relativeroot':    'relativeroot_col_name',
             'localkey':        'localkey_col_name',
             'globalkey':       'globalkey_col_name'}}

    inplace : :obj:`bool`, optional
        Pass True if you want to mutate the input.

    Returns
    -------
    :obj:`pandas.DataFrame`
        If `inplace=False`, the relevant features of the transposed chords are returned.
        Otherwise, the original DataFrame is mutated.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if not inplace:
        df = df.copy()

    # If the index is not unique, it has to be temporarily replaced
    tmp_index = not df.index.is_unique or isinstance(
        df.index, pd.core.indexes.interval.IntervalIndex
    )
    if tmp_index:
        ix = df.index
        df.reset_index(drop=True, inplace=True)

    features = [
        "chord",
        "pedal",
        "numeral",
        "form",
        "figbass",
        "changes",
        "relativeroot",
        "localkey",
        "globalkey",
    ]
    for col in features:
        if col in df.columns and col not in cols:
            cols[col] = col
    local_minor, global_minor = (
        f"{cols['localkey']}_is_minor",
        f"{cols['globalkey']}_is_minor",
    )
    if local_minor not in df.columns:
        df[local_minor] = series_is_minor(df[cols["localkey"]], is_name=False)
        logger.debug(f"Boolean column '{local_minor} created.'")
    if global_minor not in df.columns:
        df[global_minor] = series_is_minor(df[cols["globalkey"]], is_name=True)
        logger.debug(f"Boolean column '{global_minor} created.'")

    if df[cols["localkey"]].str.contains("/").any():
        df.loc[:, cols["localkey"]] = transform(
            df, resolve_relative_keys, [cols["localkey"], global_minor]
        )
    if df[cols["pedal"]].str.contains("/").any():
        df.loc[:, cols["pedal"]] = transform(
            df, resolve_relative_keys, [cols["pedal"], local_minor]
        )

    # Express pedals in relation to the global tonic
    param_cols = [cols[col] for col in ["pedal", "localkey"]] + [global_minor]
    df.loc[:, cols["pedal"]] = transform(df, rel2abs_key, param_cols)

    # Make relativeroots to local keys
    param_cols = [cols[col] for col in ["relativeroot", "localkey"]] + [
        local_minor,
        global_minor,
    ]
    relativeroots = df.loc[df[cols["relativeroot"]].notna(), param_cols]
    rr_tuples = list(relativeroots.itertuples(index=False, name=None))
    transposed_rr = {
        (rr, localkey, local_minor, global_minor): rel2abs_key(
            resolve_relative_keys(rr, local_minor), localkey, global_minor
        )
        for (rr, localkey, local_minor, global_minor) in set(rr_tuples)
    }
    transposed_rr = pd.Series(
        (transposed_rr[t] for t in rr_tuples), index=relativeroots.index
    )
    df.loc[relativeroots.index, cols["localkey"]] = transposed_rr
    df.loc[relativeroots.index, local_minor] = series_is_minor(
        df.loc[relativeroots.index, cols["localkey"]]
    )

    # Express numerals in relation to the global tonic
    param_cols = [cols[col] for col in ["numeral", "localkey"]] + [global_minor]
    df["abs_numeral"] = transform(df, rel2abs_key, param_cols)

    # Transpose changes to be valid with the new numeral
    param_cols = [cols[col] for col in ["changes", "numeral"]] + [
        "abs_numeral",
        local_minor,
        global_minor,
    ]
    df.loc[:, cols["changes"]] = transform(
        df, transpose_changes, param_cols, logger=logger
    )

    # Combine the new chord features
    df.loc[:, cols["chord"]] = (
        df.abs_numeral
        + df.form.fillna("")
        + df.figbass.fillna("")
        + ("(" + df.changes.astype("string") + ")").fillna("")
    )  # + ('/' + df.relativeroot).fillna('')

    if tmp_index:
        df.index = ix

    if inplace:
        df[cols["numeral"]] = df.abs_numeral
        drop_cols = [cols[col] for col in ["localkey", "relativeroot"]] + [
            "abs_numeral",
            local_minor,
        ]
        df.drop(columns=drop_cols, inplace=True)
    else:
        res_cols = (
            ["abs_numeral"]
            + [cols[col] for col in ["form", "figbass", "changes", "globalkey"]]
            + [global_minor]
        )
        res = df[res_cols].rename(columns={"abs_numeral": cols["numeral"]})
        return res


def make_chord_col(df, cols=None):
    """The 'chord' column contains the chord part of a DCML label, i.e. without indications of key, pedal, cadence, or
    phrase. This function can re-create this column, e.g. if the feature columns were changed. To that aim, the function
    takes a DataFrame and the column names that it adds together, creating new strings. Column names
    'changes' and 'relativeroot', if present, are treated specially (see the code)."""
    if cols is None:
        cols = ["numeral", "form", "figbass", "changes", "relativeroot"]
    cols = [c for c in cols if c in df.columns]
    summing_cols = [c for c in cols if c not in ("changes", "relativeroot")]
    if len(summing_cols) == 1:
        chord_col = df[summing_cols[0]].fillna("").astype("string")
    else:
        chord_col = df[summing_cols].fillna("").astype("string").sum(axis=1)
    if "changes" in cols:
        chord_col += ("(" + df.changes.astype("string") + ")").fillna("")
    if "relativeroot" in cols:
        chord_col += ("/" + df.relativeroot.astype("string")).fillna("")
    return chord_col.rename("chord")


def make_gantt_data(
    at, last_mn=None, relativeroots=True, mode_agnostic_adjacency=True, logger=None
):
    """Takes an expanded DCML annotation table and returns a DataFrame with timings of the included key segments,
    based on the column ``localkey``. The column names are suited for the plotly library.
    Uses: rel2abs_key, resolve_relative_keys, roman_numeral2fifths roman_numerals2semitones, labels2global_tonic

    Parameters
    ----------
    at : :obj:`pandas.DataFrame`
        Expanded DCML annotation table.
    last_mn : :obj:`int`, optional
        By default, the column ``quarterbeats`` is used for computing Start and Finish unless the column is not present,
        in which case a continuous version of measure numbers (MN) is used. In the latter case you should pass the last
        measure number of the piece in order to calculate the correct duration of the last key segment; otherwise it
        will go until the end of the last label's MN. As soon as you pass a value, the column ``quarterbeats`` is
        ignored
        even if present. If you want to ignore it but don't know the last MN, pass -1.
    relativeroots : :obj:`bool`, optional
        By default, additional rows are added based on the column ``relativeroot``. Pass False to prevent that.
    mode_agnostic_adjacency : :obj:`bool`, optional
        By default (if ``relativeroots`` is True), additional rows are added for labels adjacent to temporarily
        tonicized
        roots, no matter if the mode is identical or not. For example, before and after a V/V, all V _and_ v labels will
        be grouped as adjacent segments. Pass False to group only labels with the same mode (only V labels in the
        example),
        or None to include no adjacency at all.

    Returns
    -------

    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    at = at[at.numeral.notna() & (at.numeral != "@none")].copy()
    if last_mn is not None or "quarterbeats" not in at.columns:
        position_col = "mn_fraction"
        if last_mn is None or last_mn < 0:
            last_mn = at.mn.max()
        last_val = last_mn + 1.0
        if "mn_fraction" not in at.columns:
            mn_fraction = (
                at.mn + (at.mn_onset.astype(float) / at.timesig.map(frac).astype(float))
            ).astype(float)
            at.insert(at.columns.get_loc("mn") + 1, "mn_fraction", mn_fraction)
    else:
        position_col = "quarterbeats"
        at = at[at.quarterbeats.notna()].copy()
        at.quarterbeats = at.quarterbeats.astype(float)
        last_label = at.iloc[-1]
        last_val = float(last_label.quarterbeats) + last_label.duration_qb

    check_columns = ("localkey", "globalkey_is_minor")
    if any(c not in at.columns for c in check_columns):
        logger.error(
            f"Annotation table is missing the columns {', '.join(c for c in check_columns if c not in at.columns)}. "
            f"Cannot make Gantt data"
        )
        return pd.DataFrame()

    at.sort_values(position_col, inplace=True)
    at.index = make_interval_index_from_breaks(
        at[position_col], end_value=last_val, logger=logger
    )

    at["localkey_resolved"] = transform(
        at, resolve_relative_keys, ["localkey", "globalkey_is_minor"]
    )

    key_groups = at.loc[
        at.localkey != at.localkey.shift(),
        [
            position_col,
            "localkey",
            "localkey_resolved",
            "globalkey",
            "globalkey_is_minor",
        ],
    ].rename(columns={position_col: "Start"})
    iix = make_interval_index_from_breaks(
        key_groups.Start, end_value=last_val, logger=logger
    )
    key_groups.index = iix
    fifths = transform(
        key_groups, roman_numeral2fifths, ["localkey_resolved", "globalkey_is_minor"]
    ).rename("fifths")
    semitones = transform(
        key_groups,
        roman_numeral2semitones,
        ["localkey_resolved", "globalkey_is_minor"],
        logger=logger,
    ).rename("semitones")
    description = (
        "Duration: "
        + iix.length.astype(str)
        + "<br>Tonicized global scale degree: "
        + key_groups.localkey_resolved
        + "<br>Local tonic: "
        + key_groups.localkey
    )

    key_groups = pd.DataFrame(
        {
            "Start": key_groups.Start,
            "Finish": iix.right,
            "Duration": iix.length,
            "Resource": "local",
            "abs_numeral": key_groups.localkey_resolved,
            "fifths": fifths,
            "semitones": semitones,
            "localkey": key_groups.localkey,
            "globalkey": key_groups.globalkey,
            "Description": description,
        }
    )

    if not relativeroots or at.relativeroot.isna().all():
        return key_groups

    levels = list(range(at.index.nlevels))

    def select_groups(df):
        nonlocal levels
        has_applied = df.Resource.notna()
        if has_applied.any():
            df.Resource.fillna("tonic of adjacent applied chord(s)", inplace=True)
            # relativeroot gets filled in with numeral because it is needed for the Description. However,
            # if mode_agnostic_adjacency=True,
            # only the numeral of the first row of each subgroup will be displayed.
            df.relativeroot = df.relativeroot.where(has_applied, df.numeral)
            df["subgroup"] = df.Resource != df.Resource.shift()
            return df
        else:
            return pd.DataFrame(columns=levels).set_index(levels, drop=True)

    def gantt_data(df):
        frst = df.iloc[[0]]
        start, finish = df.index[0].left, df.index[-1].right
        frst["Start"] = start
        frst["Finish"] = finish
        frst["Duration"] = finish - start
        frst.index = pd.IntervalIndex.from_tuples([(start, finish)], closed="left")
        return frst

    global_numerals = labels2global_tonic(at).numeral
    at["Resource"] = pd.NA
    at.Resource = at.Resource.where(at.relativeroot.isna(), "applied")
    at["relativeroot_resolved"] = transform(
        at, resolve_relative_keys, ["relativeroot", "localkey_is_minor"]
    )
    at["abs_numeral"] = transform(
        at,
        rel2abs_key,
        ["relativeroot_resolved", "localkey_resolved", "globalkey_is_minor"],
    )
    at.abs_numeral.fillna(
        global_numerals, inplace=True
    )  # = at.abs_numeral.where(at.abs_numeral.notna(), global_numerals)
    at["fifths"] = transform(
        at, roman_numeral2fifths, ["abs_numeral", "globalkey_is_minor"]
    )
    at["semitones"] = transform(
        at,
        roman_numeral2semitones,
        ["abs_numeral", "globalkey_is_minor"],
        logger=logger,
    )

    if mode_agnostic_adjacency is not None:
        adjacent_groups = (
            (at.semitones != at.semitones.shift()).cumsum()
            if mode_agnostic_adjacency
            else (at.abs_numeral != at.abs_numeral.shift()).cumsum()
        )
        try:
            at = (
                at.groupby(adjacent_groups, group_keys=False)
                .apply(select_groups)
                .astype({"semitones": int, "fifths": int})
            )
        except Exception:
            print(at.groupby(adjacent_groups, group_keys=False).apply(select_groups))
            raise
        at.subgroup = at.subgroup.cumsum()
        at = at.groupby(["subgroup", "localkey"], group_keys=False).apply(gantt_data)
    else:
        subgroups = (
            (at.relativeroot != at.relativeroot.shift())
            | (at.localkey != at.localkey.shift())
        ).cumsum()
        at = (
            at[at.relativeroot.notna()]
            .groupby(subgroups, group_keys=False)
            .apply(gantt_data)
        )
    res = pd.concat([key_groups, at])[
        [
            "Start",
            "Finish",
            "Duration",
            "Resource",
            "abs_numeral",
            "fifths",
            "semitones",
            "localkey",
            "globalkey",
            "relativeroot",
        ]
    ]
    res.loc[:, ["Start", "Finish", "Duration"]] = res[
        ["Start", "Finish", "Duration"]
    ].round(2)
    res["Description"] = (
        "Duration: "
        + res.Duration.astype(str)
        + "<br>Tonicized global scale degree: "
        + res.abs_numeral
        + "<br>Local tonic: "
        + res.localkey
        + ("<br>Tonicized local scale degree: " + res.relativeroot).fillna("")
    )
    return res


def notes2pcvs(
    notes,
    pitch_class_format="tpc",
    normalize=False,
    long=False,
    fillna=True,
    additional_group_cols=None,
    ensure_columns=None,
    logger=None,
):
    """

    Parameters
    ----------
    notes : :obj:`pandas.DataFrame`
        Note table to be transformed into a wide or long table of Pitch Class Vectors by grouping via the first (or
        only)
        index level. The DataFrame needs containing at least the columns 'duration_qb' and 'tpc' or 'midi', depending
        on ``pitch_class_format``.
    pitch_class_format : :obj:`str`, optional
        | Defines the type of pitch classes to use for the vectors.
        | 'tpc' (default): tonal pitch class, such that -1=F, 0=C, 1=G etc.
        | 'name': tonal pitch class as spelled pitch, e.g. 'C', 'F#', 'Abb' etc.
        | 'pc': chromatic pitch classes where 0=C, 1=C#/Db, ... 11=B/Cb.
        | 'midi': original MIDI numbers; the result are pitch vectors, not pitch class vectors.
    normalize : :obj:`bool`, optional
        By default, the PCVs contain absolute durations in quarter notes. Pass True to normalize
        the PCV for each group.
    long : :obj:`bool`, optional
        By default, the resulting DataFrames have wide format, i.e. each row contains the PCV
        for one slice. Pass True if you need long format instead, i.e. with a non-unique
        :obj:`pandas.IntervalIndex` and two columns, ``[('tpc'|'midi'), 'duration_qb']``
        where the first column's name depends on ``pitch_class_format``.
    fillna : :obj:`bool`, optional
        By default, if a Pitch class does not appear in a PCV, its value will be 0. Pass False if
        you want NA instead.
    additional_group_cols : (:obj:`list` of) :obj:`str`
        If you would like to maintain some information from other columns of ``notes`` in additional index levels,
        pass their names.
    ensure_columns : :obj:`Iterable`, optional
        By default, pitch classes that don't appear don't get a column. Pass a value if you want to
        ensure the presence of particular columns, even if empty. For example, if ``pitch_class_format='pc'``
        you could pass ``ensure_columns=range(12)``.


    Returns
    -------

    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if pitch_class_format in ("tpc", "name"):
        pitch_class_grouper = notes.tpc.values
    elif pitch_class_format == "pc":
        pitch_class_grouper = (notes.midi % 12).values
    elif pitch_class_format == "midi":
        pitch_class_grouper = notes.midi.values
    else:
        logger.warning(
            "pitch_class_format needs to be one of 'tpc', 'name', 'pc', 'midi', not "
            + str(pitch_class_format)
        )
        return pd.DataFrame()
    idx = notes.index
    if isinstance(idx, pd.MultiIndex):
        idx = idx.get_level_values(0)
    grouper = [idx]
    group_levels = 1
    if additional_group_cols is not None:
        if isinstance(additional_group_cols, str):
            additional_group_cols = [additional_group_cols]
        for col in additional_group_cols:
            grouper.append(notes[col].values)
            group_levels += 1
    grouper.append(pitch_class_grouper)
    pcvs = notes.groupby(grouper, dropna=False).duration_qb.sum()
    # if long:
    #     if normalize:
    #         pcvs = pcvs.groupby(level=list(range(group_levels))).apply(lambda S: S / S.sum())
    #     pcvs = pcvs.reset_index(level=-1)
    #     if pitch_class_format == "name":
    #         pcvs.loc[:, "tpc"] = transform(pcvs.tpc, fifths2name)
    # else:
    pcvs = pcvs.unstack()
    if pitch_class_format == "name":
        pcvs.columns = fifths2name(pcvs.columns)
    if normalize:
        pcvs = pcvs.div(pcvs.sum(axis=1), axis=0)
    if ensure_columns is not None:
        missing = [c for c in ensure_columns if c not in pcvs.columns]
        if len(missing) > 0:
            new_cols = pd.DataFrame(index=pcvs.index, columns=missing)
            pcvs = pd.concat([pcvs, new_cols], axis=1).sort_index(axis=1)
    if long:
        pcvs = pcvs.stack()
    if fillna:
        return pcvs.fillna(0.0)
    return pcvs


def resolve_all_relative_numerals(at, additional_columns=None, inplace=False):
    """Resolves Roman numerals that include slash notation such as '#vii/ii' => '#i' or 'V/V/V' => 'VI' in a major and
    '#VI' in a minor key. The function expects the columns ['globalkey_is_minor', 'localkey_is_minor'] to be present.
    The former is necessary only if the column 'localkey' is present and needs resolving. Execution will be slightly
    faster if performed on the entire DataFrame rather than using :py:meth:`transform_multiple`.

    Parameters
    ----------
    at : :obj:`pandas.DataFrame`
        Annotation table.
    additional_columns : :obj:`str` or :obj:`list`
        By default, the function resolves, if present, the columns ['relativeroot', 'pedal'] but here you can name
        other columns, too. They will be resolved based on the localkey's mode.
    inplace : :obj:`bool`, optional
        By default, a manipulated copy of ``at`` is returned. Pass True to mutate instead.

    Returns
    -------

    """
    if not inplace:
        at = at.copy()
    if "localkey" in at.columns and at.localkey.str.contains("/").any():
        at.loc[:, "localkey"] = transform(
            at, resolve_relative_keys, ["localkey", "globalkey_is_minor"]
        )
    roman_numeral_cols = ["relativeroot", "pedal"]
    if additional_columns is not None:
        if isinstance(additional_columns, str):
            additional_columns = [additional_columns]
        roman_numeral_cols += list(additional_columns)
    cols = [
        c
        for c in roman_numeral_cols
        if c in at.columns and at[c].str.contains("/").any()
    ]
    if len(cols) > 0:
        resolved = transform(
            at[cols + ["localkey_is_minor"]],
            resolve_relative_keys,
            ["localkey_is_minor"],
            column_wise=True,
        )
        at.loc[:, cols] = resolved
    if all(c in at.columns for c in ("numeral", "relativeroot")):
        at.loc[:, "numeral"] = transform(
            at, rel2abs_key, ["numeral", "relativeroot", "localkey_is_minor"]
        )
        at.drop(columns="relativeroot", inplace=True)
    at.loc[:, "chord"] = make_chord_col(at)
    if not inplace:
        return at


def segment_by_adjacency_groups(
    df, cols, na_values="group", group_keys=False, logger=None
):
    """Drop exact adjacent repetitions within one or a combination of several feature columns and adapt the
    IntervalIndex and the column 'duration_qb' accordingly.
    Uses: :func:`adjacency_groups`, :func:`reduce_dataframe_duration_to_first_row`

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        DataFrame to be reduced, expected to contain the column ``duration_qb``. In order to use the result as a
        segmentation, it should have a :obj:`pandas.IntervalIndex`.
    cols : :obj:`list`
        Feature columns which exact, adjacent repetitions should be grouped to a segment, keeping only the first row.
    na_values : (:obj:`list` of) :obj:`str` or :obj:`Any`, optional
        | Either pass a list of equal length as ``cols`` or a single value that is passed to :py:func:`adjacency_groups`
        | for each. Not dealing with NA values will lead to wrongly grouped segments. The default option is the safest.
        | 'group' creates individual groups for NA values
        | 'backfill' or 'bfill' groups NA values with the subsequent group
        | 'pad', 'ffill' groups NA values with the preceding group
        | Any other value works like 'group', with the difference that the created groups will be named with this value.
    group_keys : :obj:`bool`, optional
        By default, the grouped values will be returned as an appended MultiIndex, differentiation groups via ascending
        integers. If you want to duplicate the columns' value, e.g. to account for a custom filling value for
        ``na_values``, pass True. Beware that this most often results in non-unique index levels.

    Returns
    -------
    :obj:`pandas.DataFrame`
        Reduced DataFrame with updated 'duration_qb' column and :obj:`pandas.IntervalIndex` on the first level
        (if present).

    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if "duration_qb" not in df.columns:
        logger.error("DataFrame is missing the column 'duration_qb'")
    if isinstance(cols, str):
        cols = [cols]
    N = len(
        cols
    )  # number of columns of which subsequent equal value combinations are grouped
    if not isinstance(na_values, list):
        na_values = [na_values] * N
    else:
        while len(na_values) < N:
            na_values.append("group")
    groups, names = zip(
        *(
            adjacency_groups(c, na_values=na)
            for (_, c), na in zip(df[cols].items(), na_values)
        )
    )

    grouped = df.groupby(list(groups), dropna=False).apply(
        reduce_dataframe_duration_to_first_row
    )
    if any(gr.isna().any() for gr in groups):
        logger.warning(
            f"na_values={na_values} did not take care of all NA values in {cols}. This very probably leads to wrongly "
            f"grouped rows. If in doubt, use 'group'."
        )
    if isinstance(grouped.index, pd.MultiIndex):
        grouped.index = grouped.index.reorder_levels([-1] + list(range(N)))
    else:
        logger.debug("Groupby-apply did not generate MultiIndex.")
        grouped = grouped.set_index(list(groups), append=True)
    if group_keys:

        def no_tuples(coll):
            return not any(isinstance(e, tuple) for e in coll)

        new_levels = [
            index_level.map(d) if no_tuples(d.values()) else index_level
            for index_level, d in zip(grouped.index.levels[1:], names)
        ]
        grouped.index = grouped.index.set_levels(
            new_levels, level=list(range(1, N + 1)), verify_integrity=False
        )
    grouped.index.rename("segment", level=0, inplace=True)
    return grouped


def segment_by_criterion(
    df: pd.DataFrame,
    boolean_mask: Union[pd.Series, np.array],
    warn_na: bool = False,
    logger=None,
) -> pd.DataFrame:
    """Drop all rows where the boolean mask does not match, and adapt the IntervalIndex and the column 'duration_qb'
    accordingly.

    Args:
      df: DataFrame to be reduced, expected to come with the column ``duration_qb`` and an :obj:`pandas.IntervalIndex`.
      boolean_mask: Boolean mask where every True value starts a new segment.
      warn_na: If the boolean mask starts with any number of False, this first group will be missing from the result.
          Set warn_na to True if you want the logger to throw a warning in this case.

    Returns:
      Reduced DataFrame with updated 'duration_qb' column and :obj:`pandas.IntervalIndex` on the first level.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if "quarterbeats" not in df.columns:
        raise TypeError("DataFrame is missing the column 'quarterbeats'")
    try:
        offset_dict = dict(end=df.index.right.max())
    except AttributeError:
        logger.warning(
            f"DataFrame index does not consist of one level that is an IntervalIndex:\n{df.index[:3]}"
        )
        raise
    df = df[boolean_mask].reset_index(drop=True)
    if "duration_qb" in df.columns:
        df = df.drop(columns="duration_qb")

    result = add_quarterbeats_col(df, offset_dict, interval_index=True, logger=logger)
    if warn_na and not boolean_mask.iloc[0]:
        logger.warning(
            "Boolean mask started with False, meaning that a part of the DataFrame is excluded from the segmentation."
        )
    return result


def segment_by_interval_index(df, idx, truncate=True):
    """Segment a DataFrame into chunks based on a given IntervalIndex.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        DataFrame that has a :obj:`pandas.IntervalIndex` to allow for its segmentation.
    idx : :obj:`pandas.IntervalIndex` or :obj:`pandas.MultiIndex`
        Intervals by which to segment ``df``. The index will be prepended to differentiate between segments. If ``idx``
        is a :obj:`pandas.MultiIndex`, the first level is expected to be a :obj:`pandas.IntervalIndex`.
    truncate : :obj:`bool`, optional
        By default, the intervals of the segmented DataFrame will be cut off at segment boundaries and the event's
        'duration_qb' will be adapted accordingly. Pass False to prevent that and duplicate overlapping events without
        adapting their Intervals and 'duration_qb'.

    Returns
    -------
    :obj:`pandas.DataFrame`
        A copy of ``df`` where the index levels ``idx`` have been prepended and only rows of ``df`` with overlapping
        intervals are included.

    """
    iv_idx = idx.copy()
    if isinstance(iv_idx, pd.MultiIndex):
        iv_idx = iv_idx.get_level_values(0)
    assert isinstance(
        iv_idx, pd.IntervalIndex
    ), f"idx needs to IntervalIndex, not {type(iv_idx)}"
    chunks = overlapping_chunk_per_interval(df, iv_idx, truncate=truncate)
    return pd.concat(chunks.values(), keys=idx, levels=idx.levels)


def slice_df(
    df: pd.DataFrame, quarters_per_slice: Optional[float] = None
) -> Dict[pd.Interval, pd.DataFrame]:
    """Returns a sliced version of the DataFrame. Slices appear in the IntervalIndex and the contained event's
    durations within the slice are shown in the column 'duration_qb'.
    Uses:

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        The DataFrame is expected to come with an IntervalIndex and contain the columns 'quarterbeats' and
        'duration_qb'.
        Those can be obtained through ``Parse.get_lists(interval_index=True)`` or
        ``Parse.iter_transformed(interval_index=True)``.
    quarters_per_slice : :obj:`float`, optional
        By default, the slices have variable size, from onset to onset. If you pass a value, the slices will
        have that constant size, measured in quarter notes. For example, pass 1.0 for all slices to have size 1 quarter.

    Returns
    -------
    :obj:`pandas.DataFrame`

    """
    assert isinstance(
        df.index, pd.IntervalIndex
    ), f"DataFrame is expected to have an IntervalIndex, not {type(df.index)}."
    end = max(df.index.right)
    if quarters_per_slice is None:
        lefts = sorted(set(df.index.left))
        rights = lefts[1:] + [end]
    else:
        start = min(df.index.left)
        lefts = np.arange(start, end, quarters_per_slice)
        rights = np.arange(
            start + quarters_per_slice, end + quarters_per_slice, quarters_per_slice
        )
    intervals = [pd.Interval(i, j, closed="left") for i, j in zip(lefts, rights)]
    return overlapping_chunk_per_interval(df, intervals)


def transform_multiple(df, func, level=-1, logger=None, **kwargs):
    """Applying transformation(s) separately to concatenated pieces that can be differentiated by index level(s).

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        Concatenated tables with :obj:`pandas.MultiIndex`.
    func : :obj:`~collections.abc.Callable` or :obj:`str`
        Function to be applied to the individual tables. For convenience, you can pass strings to call the standard
        transformers for a particular table type. For example, pass 'annotations' to call ``transform_annotations``.
    level : :obj:`int` or :obj:`list`
        Argument passed to :py:meth:`pandas.DataFrame.groupby`. Defaults to -1, resulting in a GroupBy by all levels
        except the last. Conversely, you can pass, for instance, 2 to group by the first two levels.
    kwargs :
        Keyword arguments passed to ``func``.

    Returns
    -------
    :obj:`pandas.DataFrame`
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if isinstance(func, str):
        kw2func = {
            "annotations": transform_annotations,
            "key_segment": add_localkey_change_column,
        }
        if func in kw2func:
            func = kw2func[func]
        else:
            raise ValueError(
                f"'{func}' is not a valid keyword. Either pass a callable (function), or one of the keywords "
                f"{', '.join(kw2func.keys())}"
            )

    levels = _treat_level_parameter(level, df.index.nlevels)

    if len(levels) > 0:
        try:
            res = df.groupby(level=levels, group_keys=True).apply(
                lambda df: func(df.droplevel(levels), **kwargs)
            )
        except Exception:
            logger.warning(
                f"Error when trying to group the index levels {list(df.index.names)} using level={levels}."
            )
            raise
    else:
        logger.info(
            f"Index had too few levels ({list(df.index.names)}) to group by parameter level={levels}. Applied"
            f"{func} to the entire DataFrame instead."
        )
        res = func(df, **kwargs)
    return res


def transform_annotations(at, groupby_features=None, resolve_relative=False):
    """Wrapper for applying several transformations to an annotation table.

    Parameters
    ----------
    at : :obj:`pandas.DataFrame`
        Annotation table corresponding to a single piece.
    groupby_features : :obj:`str` or :obj:`list`
        Argument ``features`` passed to :py:meth:`group_annotations_by_features`.
    resolve_relative : :obj:`bool`
        Resolves slash notation (e.g. 'vii/V') from Roman numerals in the columns ['localkey', 'relativeroot', 'pedal'].

    Returns
    -------
    :obj:`pandas.DataFrame`
    """
    if groupby_features is not None:
        at = group_annotations_by_features(at, groupby_features)
    if resolve_relative:
        at = resolve_all_relative_numerals(at)
    return at


def transpose_notes_to_localkey(notes):
    """Transpose the columns 'tpc' and 'midi' such that they reflect the local key as if it was C major/minor. This
    operation is typically required for creating pitch class profiles.
    Uses: :py:func:`transform`, :py:func:`name2fifths`, :py:func:`roman_numeral2fifths`

    Parameters
    ----------
    notes : :obj:`pandas.DataFrame`
        DataFrame that has at least the columns ['globalkey', 'localkey', 'tpc', 'midi'].

    Returns
    -------
    :obj:`pandas.DataFrame`
         A copy of ``notes`` where the columns 'tpc' and 'midi' are shifted in such a way that tpc=0 and midi=60
         match the local tonic (e.g. for the local key A major/minor, each pitch A will have tpc=0 and midi % 12 = 0).
    """
    notes = notes.copy()
    new_cols = []
    if "globalkey_is_minor" not in notes.columns:
        gkim = notes.globalkey.str.islower().rename("globalkey_is_minor")
        new_cols.append(gkim)
    if "localkey_is_minor" not in notes.columns:
        lkim = notes.localkey.str.islower().rename("localkey_is_minor")
        new_cols.append(lkim)
    if len(new_cols) > 0:
        notes = pd.concat([notes] + new_cols, axis=1)
    transpose_by = transform(notes.globalkey, name2fifths) + transform(
        notes, roman_numeral2fifths, ["localkey", "globalkey_is_minor"]
    )
    notes.tpc -= transpose_by
    midi_transposition = transform(transpose_by, fifths2pc)
    # For transpositions up to a diminished fifth, move pitches up,
    # for larger intervals, move pitches down.
    midi_transposition.where(
        midi_transposition <= 6, midi_transposition % -12, inplace=True
    )
    notes.midi -= midi_transposition
    notes = pd.concat([notes, transpose_by.rename("local_tonic")], axis=1)
    notes.local_tonic = transform(
        notes, fifths2name, dict(fifths="local_tonic", minor="localkey_is_minor")
    )
    return notes


def _treat_level_parameter(level, nlevels, logger=None):
    """Given a number of index levels, turn an int such as -1 into a list of all levels except the last.

    Parameters
    ----------
    level : :obj:`int`
        Integer to transform into a list. If negative, all but the last ``level`` levels are chosen. If positive,
        the first ``level`` ones.
    nlevels : :obj:`int`
        Number of present index levels.

    Returns
    -------

    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if not isinstance(level, int):
        return level
    if level == 0:
        level = nlevels
    if level < 0:
        level += nlevels
        if level < 0:
            logger.warning(
                f"Cannot group by the {level} first levels. nlevels={nlevels}"
            )
            return []
    elif level > nlevels:
        logger.warning(
            f"Supposed to group by {level} first levels but nlevels={nlevels}"
        )
        return []
    levels = list(range(level))
    return levels


def transform_columns(df, func, columns=None, param2col=None, inplace=False, **kwargs):
    """Wrapper function to use transform() on df[columns], leaving the other columns untouched.

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


def transform_note_columns(
    df,
    to,
    note_cols=["chord_tones", "added_tones", "bass_note", "root"],
    minor_col="localkey_is_minor",
    inplace=False,
    logger=None,
    **kwargs,
):
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
    **kwargs: keyword arguments for transform()
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    transformations = {
        "name": fifths2name,
        "names": fifths2name,
        "iv": fifths2iv,
        "pc": fifths2pc,
        "sd": fifths2sd,
        "rn": fifths2rn,
    }
    assert (
        to in transformations
    ), "Parameter to needs to be one of {'name', 'iv', 'pc', 'sd', 'rn'}"
    cols = [col for col in note_cols if col in df.columns]
    if len(cols) < len(note_cols):
        logger.warning(
            f"Columns {[[col for col in note_cols if not col in df.columns]]}"
        )
    param2col = None
    if to in ["sd", "rn"]:
        assert (
            minor_col in df.columns or "minor" in kwargs
        ), f"'{to} representation requires a boolean column for the 'minor' argument, e.g. 'globalkey_is_minor'."
        if "minor" not in kwargs:
            param2col = {"minor": minor_col}
    func = transformations[to]
    res = transform_columns(
        df,
        func,
        columns=note_cols,
        inplace=inplace,
        param2col=param2col,
        column_wise=True,
        **kwargs,
    )
    if not inplace:
        return res


def transpose_chord_tones_by_localkey(df, by_global=False):
    """Returns a copy of the expanded table where the scale degrees in the chord tone columns
        have been transposed by localkey (i.e. they express all chord tones as scale degrees of
        the globalkey) or, if ``by_global`` is set to True, additionally by globalkey (i.e., chord
        tones as tonal pitch classes TPC).

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        Expanded labels with chord tone columns.
    by_global : :obj:`bool`
        By default, the transformed chord tone columns express chord tones as scale degrees (or
        intervals) of the global tonic. If set to True, they correspond to tonal pitch classes
        and can be further transformed to note names using transform_note_columns().

    Returns
    -------
    :obj:`pandas.DataFrame`
    """
    df = df.copy()
    ct_cols = [
        col
        for col in ("chord_tones", "added_tones", "root", "bass_note")
        if col in df.columns
    ]
    assert (
        ct_cols
    ), "Found none of the expected chord-tone columns {'chord_tones', 'added_tones', 'root', 'bass_note'}."
    ct = df[ct_cols]
    transpose_by = transform(
        df, roman_numeral2fifths, ["localkey", "globalkey_is_minor"]
    )
    if by_global:
        transpose_by += transform(df, name2fifths, ["globalkey"])
    transposed_row_tuples = [
        transpose(tpcs, fifths)
        for tpcs, fifths in zip(
            ct.itertuples(index=False, name=None), transpose_by.values
        )
    ]
    ct = pd.DataFrame(transposed_row_tuples, index=df.index, columns=ct_cols)
    with warnings.catch_warnings():
        # Setting values in-place is fine, ignore the warning in Pandas >= 1.5.0
        # This can be removed, if Pandas 1.5.0 does not need to be supported any longer.
        # See also: https://stackoverflow.com/q/74057367/859591
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=(
                ".*will attempt to set the values inplace instead of always setting a new array. "
                "To retain the old behavior, use either.*"
            ),
        )
        df.loc[:, ct_cols] = ct
    return df
