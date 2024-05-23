from collections import defaultdict
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .logger import LoggedClass, get_logger

module_logger = get_logger(__name__)

# region helper functions


def get_volta_structure(
    measures, mc, volta_start, volta_length, frac_col=None, logger=None
) -> Dict[int, Dict[int, List[int]]]:
    """Extract volta structure from measures table.

    Uses: :func:`treat_group`

    Args:
      measures: Measures table containing the columns indicated in the other arguments.
      mc, volta_start, volta_length, frac_col: column names

    Returns:
      {first_mc -> {volta_number -> [MC] } }
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    cols = [mc, volta_start, volta_length]
    sel = measures[volta_start].notna()
    voltas = measures.loc[sel, cols]
    if voltas[volta_length].isna().sum() > 0:
        rows = voltas[voltas[volta_length].isna()]
        logger.debug(
            f"The volta in MC {rows[mc].values} has no length: A standard length of 1 is supposed."
        )
        voltas[volta_length] = voltas[volta_length].fillna(0)
    try:
        voltas = voltas.astype(int)
    except ValueError:
        logger.error(
            f"Could not compute volta structure because at least one MC contains several of them: {voltas}"
        )
        return {}
    if len(voltas) == 0:
        return {}
    if frac_col is not None:
        voltas[volta_length] += measures.loc[sel, frac_col].notna()
    voltas.loc[voltas[volta_start] == 1, "group"] = 1
    voltas.group = voltas.group.fillna(0).astype(int).cumsum()
    groups = {v[mc].iloc[0]: v[cols].to_numpy() for _, v in voltas.groupby("group")}
    res = {mc: treat_group(mc, group, logger=logger) for mc, group in groups.items()}
    logger.debug(f"Inferred volta structure: {res}")
    return res


def keep_one_row_each(
    df,
    compress_col,
    differentiating_col,
    differentiating_val=None,
    ignore_columns=None,
    fillna=True,
    drop_differentiating=True,
    logger=None,
):
    """Eliminates duplicates in `compress_col` but warns about values within the
        dropped rows which diverge from those of the remaining rows. The `differentiating_col`
        serves to identify places where information gets lost during the process.

    The result of this function is the same as `df.drop_duplicates(subset=[compress_col])`
    if `differentiating_val` is None, and `df[df[compress_col] == differentiating_val]` otherwise
    but with the difference that only adjacent duplicates are eliminated.

    Parameters
    ----------
    compress_col : :obj:`str`
        Column with duplicates (e.g. measure counts).
    differentiating_col : :obj:`str`
        Column that differentiates duplicates (e.g. staff IDs).
    differentiating_val : value, optional
        If you want to keep rows with a certain `differentiating_col` value, pass that value (e.g. a certain staff).
        Otherwise, the first row of every `compress_col` value is kept.
    ignore_columns : :obj:`Iterable`, optional
        These columns are not checked.
    fillna : :obj:`bool`, optional
        By default, missing values of kept rows are filled if the dropped rows contain
        one unique value in that particular column. Pass False to keep rows as they are.
    drop_differentiating : :obj:`bool`, optional
        By default, the column that differentiates the `compress_col` is dropped.
        Pass False to prevent that.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if ignore_columns is None:
        ignore_columns = [differentiating_col]
    else:
        ignore_columns.append(differentiating_col)
    consider_for_notna = [
        col for col in df.columns if col not in ignore_columns + [compress_col]
    ]
    consider_for_duplicated = consider_for_notna + [compress_col]
    empty_rows = df[consider_for_duplicated].isnull().all(axis=1)
    if differentiating_val is not None:
        keep_rows = df[differentiating_col] == differentiating_val
    else:
        keep_rows = df[compress_col] != df[compress_col].shift()
    drop_rows = ~keep_rows & empty_rows
    result = df.drop(df[drop_rows].index)

    def squash_staves(df):
        if len(df) == 1:
            return df.iloc[0]
        if differentiating_val is None:
            keep_row = df.iloc[[0]].copy()
            remaining = df.iloc[1:].drop_duplicates(subset=consider_for_duplicated)
        else:
            keep = df[differentiating_col] == differentiating_val
            keep_row = df[keep].copy()
            assert (
                len(keep_row) == 1
            ), "The column designated by `differentiating_col` needs to be unique."
            remaining = df[~keep].drop_duplicates(subset=consider_for_duplicated)
        if len(remaining) == 1:
            return keep_row
        which = keep_row[compress_col].iloc[0]
        dont_warn = [
            "vspacerDown",
            "vspacerUp",
            "voice/BarLine",
            "voice/BarLine/span",
        ]
        for val, (col_name, col) in zip(
            *keep_row[consider_for_notna].itertuples(index=False, name=None),
            remaining[consider_for_notna].items(),
        ):
            log_this = logger.debug if col_name in dont_warn else logger.warning
            if col.isna().all():
                continue
            vals = col[col.notna()].unique()
            if len(vals) == 1:
                if vals[0] == val:
                    continue
                new_val = vals[0]
                if pd.isnull(val) and fillna:
                    keep_row[col_name] = new_val
                    msg = (
                        f"{compress_col} {which}: The missing value in '{col_name}' was filled with '{new_val}', "
                        f"present in '{differentiating_col}' "
                        f"{remaining.loc[remaining[col_name] == new_val, differentiating_col].to_list()}. "
                        f"In rare cases, this may lead to incorrect values in the measures table because it ambiguous "
                        f"which staff contains the relevant information."
                    )  # ToDo: Currently there is nothing the user can do to influence this behavior!
                    log_this(
                        msg, extra={"message_id": (9, compress_col, which, col_name)}
                    )
                    continue
                msg = (
                    f"{compress_col} {which}: The value '{new_val}' in '{col_name}' of '{differentiating_col}' "
                    f"{remaining.loc[remaining[col_name] == new_val, differentiating_col].to_list()} is lost."
                )
                log_this(msg, extra={"message_id": (9, compress_col, which, col_name)})
                continue
            msg = (
                f"{compress_col} {which}: The values {vals} in '{col_name}' of \n '{differentiating_col}' "
                f"{remaining.loc[col.notna(), differentiating_col].to_list()} are lost."
            )
            log_this(msg, extra={"message_id": (9, compress_col, which, col_name)})
        return keep_row

    result = result.groupby(compress_col, group_keys=False).apply(squash_staves)
    return result.drop(columns=differentiating_col) if drop_differentiating else result


def make_actdur_col(
    len_col: pd.Series, timesig_col: pd.Series, name: str = "act_dur"
) -> pd.Series:
    actdur = len_col.fillna(timesig_col)
    try:
        return actdur.map(Fraction).rename(name)
    except Exception:
        print(f"Failed to turn all values into fractions: {actdur}")
        raise


def make_keysig_col(
    df: pd.DataFrame, keysig_col: str = "keysig_col", name: str = "keysig"
) -> pd.Series:
    if keysig_col in df:
        return df[keysig_col].ffill().fillna(0).astype(int).rename(name)
    return pd.Series(0, index=df.index).rename(name)


def make_mn_col(
    df: pd.DataFrame,
    dont_count: str = "dont_count",
    numbering_offset: str = "numbering_offset",
    name="mn",
) -> pd.Series:
    """Compute measure numbers where one or two columns can influence the counting.

    Args:
        df: If no other parameters are given, every row is counted, starting from 1.
        dont_count:
            This column has notna() for measures where the option "Exclude from bar count" is activated,
            NaN otherwise.
        numbering_offset:
            This column has values of the MuseScore option "Add to bar number", which adds
            notna() values to this and all subsequent measures.
        name:

    Returns:

    """
    if dont_count is None:
        mn = pd.Series(range(1, len(df) + 1), index=df.index)
    else:
        excluded = df[dont_count].fillna(0).astype(bool)
        mn = (~excluded).cumsum()
    if numbering_offset is not None:
        offset = df[numbering_offset]
        if offset.notna().any():
            offset = offset.fillna(0).astype(int).cumsum()
            mn += offset
    return mn.rename(name)


def make_next_col(
    df: pd.DataFrame,
    volta_structure: Optional[Dict[int, Dict[int, List[int]]]] = None,
    sections: bool = True,
    name="next",
    logger=None,
) -> pd.Series:
    """Uses a `NextColumnMaker` object to create a column with all MCs that can follow each MC
    (e.g. due to repetitions).

    Args:
        df: Raw measure list.
        volta_structure:
            This parameter can be computed by get_volta_structure(). It is empty if
            there are no voltas in the piece.
        sections:
            By default, pieces containing section breaks (where counting MNs restarts) receive two more columns in the
            measures table, namely ``section`` and ``ambiguous_mn`` to grant access to MNs as shown in MuseScore.
            Pass False to not add such columns.
        name:
        logger:

    Returns:

    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if volta_structure is None:
        volta_structure = {}
    if sections and (df["breaks"].fillna("") == "section").sum() == 0:
        sections = False

    # col_names = ['mc', 'breaks', 'jump_bwd', 'jump_fwd', 'markers', 'play_until', 'repeats', 'volta']
    col_names = ["mc", "repeats", "breaks"]
    sel = df[col_names[1:]].notna().any(axis=1)

    ncm = NextColumnMaker(
        df, volta_structure, sections=sections, logger_cfg={"name": logger.name}
    )
    for mc, repeats, breaks in df.loc[sel, col_names].itertuples(index=False):
        ncm.treat_input(mc, repeats, breaks == "section")

    for mc, has_repeat in ncm.check_volta_repeats.items():
        if not has_repeat:
            logger.warning(f"MC {mc} is missing an endRepeat.")

    try:
        nxt_col = df["mc"].map(ncm.next).map(tuple)
    except Exception:
        print(df["mc"])
        print(ncm.next)
        raise
    return nxt_col.rename(name)


def make_offset_col(
    df,
    mc_col: str = "mc",
    timesig: str = "timesig",
    act_dur: str = "act_dur",
    next_col: str = "next",
    section_breaks: Optional[str] = "breaks",
    name: str = "mc_offset",
    logger=None,
) -> pd.Series:
    """If one MN is composed of two MCs, the resulting column indicates the second MC's offset from the MN's beginning.

    Args:
        df: Raw measures table that comes with the indicated columns.
        mc_col, timesig, act_dur, next_col: Names of the required columns.
        section_breaks:
            If you pass the name of a column, the string 'section' is taken into account
            as ending a section and therefore potentially ending a repeated part even when
            the repeat sign is missing.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)

    nominal_duration = df[timesig].map(Fraction)
    actual_duration = df[act_dur]
    expected_completion = (nominal_duration - actual_duration).rename(
        "expected_completion"
    )
    expected_completion = expected_completion.where(
        expected_completion > 0, 0
    )  # no negative completions!!! #92
    if (expected_completion == 0).all():
        logger.debug(
            "Actual durations do not diverge from nominal durations, hence mc_offset=0 everywhere."
        )
        return pd.Series(0, index=df.index, name=name)

    def which_mcs_to_offset(section_df: pd.DataFrame) -> List[int]:
        """Takes one section of an MC-indexed measures table and returns the MCs that need to be offset."""
        section_mcs = set(section_df.index)
        shorter_than_nominal = section_df[section_df.expected_completion > 0]
        mcs_getting_offset = set()
        for irregular_mc, next_mcs in shorter_than_nominal.next.items():
            if irregular_mc in mcs_getting_offset:
                # has already been marked as completing another irregular one and therefore
                # doesn't require completion itself
                continue
            following_mcs = list(section_mcs.intersection(next_mcs))
            if len(following_mcs) == 0:
                logger.debug(
                    f"MC {irregular_mc} is not followed by any MC within the same section, not checking."
                )
                if section_df.potential_anacrusis[irregular_mc]:
                    mcs_getting_offset.add(irregular_mc)
                continue
            expected_completion = section_df.expected_completion[irregular_mc]
            following_do_complete = (
                section_df.loc[following_mcs, "actual_duration"] == expected_completion
            )
            if section_df.potential_anacrusis[irregular_mc]:
                # this is probably an anacrusis that will itself be offset, unless the following measure(s) complete it
                if following_do_complete.all():
                    mcs_getting_offset.update(following_mcs)
                elif not following_do_complete.any():
                    mcs_getting_offset.add(irregular_mc)
                else:
                    show_mcs = [irregular_mc] + following_mcs
                    logger.warning(
                        f"Some of the MCs following the potential anacrusis MC {irregular_mc} do, some don't complete "
                        f"it with the expected {expected_completion}, so I cannot decide whether it's an anacrusis or "
                        f"not. Let's say it is. Follow-up warnings may arise.\n"
                        f"{section_df.loc[show_mcs]}",
                        extra={"message_id": (3, irregular_mc)},
                    )
                    mcs_getting_offset.add(irregular_mc)
                continue
            # arrives here if not a potential anacrusis
            if following_do_complete.all():
                mcs_getting_offset.update(following_mcs)
                continue

            # not all or none of the following MCs complete the irregular MC
            # first, check for the special case where one of the following MCs has another time signature which is
            # actually completed by the two MCs in question
            nominal_duration = section_df.nominal_duration[irregular_mc]
            subsequent_nominal_durations = section_df.loc[
                following_mcs, "nominal_duration"
            ]
            different_timesig = subsequent_nominal_durations != nominal_duration
            might_be_special_case = ~following_do_complete & different_timesig
            if might_be_special_case.any():
                # if a subsequent MC has a different timesig, it may be seen as legitimate completion if its
                # nominal duration is completed by the two actual durations in question
                act_dur = section_df.actual_duration
                for special_mc, other_nominal_duration in subsequent_nominal_durations[
                    might_be_special_case
                ].items():
                    if (
                        act_dur[irregular_mc] + act_dur[special_mc]
                        == other_nominal_duration
                    ):
                        following_do_complete[special_mc] = True
            if following_do_complete.all():
                mcs_getting_offset.update(following_mcs)
                continue

            # not all or none of the following MCs complete the irregular MC
            if not following_do_complete.any():
                msg = f"None of the MCs following the irregular MC {irregular_mc} complete it."
            else:
                msg = f"Some of the MCs following the irregular MC {irregular_mc} do, some don't complete it."
                mcs_getting_offset.update(
                    following_do_complete.index[following_do_complete]
                )
            show_mcs = [irregular_mc] + following_mcs
            show_columns = [
                "nominal_duration",
                "actual_duration",
                "expected_completion",
                "next",
            ]
            msg += f"\n{section_df.loc[show_mcs, show_columns]}"
            logger.warning(
                msg,
                extra={"message_id": (3, irregular_mc)},
            )
            continue
        return sorted(mcs_getting_offset)

    columns_to_display = [mc_col, next_col]
    if section_breaks is not None:
        columns_to_display.append(section_breaks)
        has_section_break = df[section_breaks].fillna("").str.contains("section")
        if not has_section_break.any():
            logger.debug(
                f"No section breaks in column {section_breaks!r} to be taken into account."
            )
            section_breaks = None

    auxiliary_df = pd.concat(
        [
            df[columns_to_display],
            nominal_duration.rename("nominal_duration"),
            actual_duration.rename("actual_duration"),
            expected_completion,
        ],
        axis=1,
    ).set_index(mc_col)
    if section_breaks is None:
        auxiliary_df["potential_anacrusis"] = False
        auxiliary_df.loc[1, "potential_anacrusis"] = True
    else:
        auxiliary_df["potential_anacrusis"] = (
            has_section_break.shift().fillna(True).values  # has df.index
        )

    section_grouper = auxiliary_df.potential_anacrusis.cumsum()

    offset_mcs_per_section = [
        which_mcs_to_offset(section_df)
        for _, section_df in auxiliary_df.groupby(section_grouper)
    ]
    mcs_to_be_offset = sum(offset_mcs_per_section, [])
    mask = pd.Series(False, index=auxiliary_df.index)
    mask.loc[mcs_to_be_offset] = True
    offset_column = auxiliary_df.expected_completion.where(mask, 0).rename(name)
    offset_column.index = df.index
    return offset_column


def make_repeat_col(
    df: pd.DataFrame,
    startRepeat: str = "startRepeat",
    endRepeat: str = "endRepeat",
    name="repeats",
) -> pd.Series:
    repeats = df[startRepeat].copy()
    ends = df[endRepeat]
    sel = dict(
        start=repeats.notna() & ends.isna(),
        startend=repeats.notna() & ends.notna(),
        end=repeats.isna() & ends.notna(),
    )
    for case, arr in sel.items():
        repeats.loc[arr] = case
    if pd.isnull(repeats.iloc[0]):
        repeats.iloc[0] = "firstMeasure"
    if pd.isnull(repeats.iloc[-1]):
        repeats.iloc[-1] = "lastMeasure"
    return repeats.rename(name)


def make_timesig_col(
    df,
    sigN_col: str = "sigN_col",
    sigD_col: str = "sigD_col",
    name="timesig",
    logger=None,
) -> pd.Series:
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    n = pd.to_numeric(df[sigN_col]).astype("Int64").ffill().astype("string")
    d = pd.to_numeric(df[sigD_col]).astype("Int64").ffill().astype("string")
    result = (n + "/" + d).rename(name)
    missing = result.isna()
    if missing.all():
        logger.warning(
            "No time signature specified. Wild-guessing it's the default 4/4.",
            extra={"message_id": (23,)},
        )
        result = result.fillna("4/4")
    elif missing.any():
        # because of the forward fill, only initial measures can have missing values
        result.bfill(inplace=True)
        fill_value = result.iloc[0]
        if missing.sum() == 1:
            logger.info(
                f"The first measure doesn't come with a time signature (probably an incipit?) but for matters "
                f"of consistency the measure table will indicate {fill_value}"
            )
        else:
            logger.warning(
                f"The {missing.sum()} first MCs came without time signature but the measure table will "
                f"indicate the first time signature occurring in the piece for them, namely {fill_value}",
                extra={"message_id": (24,)},
            )
    return result


def make_volta_col(
    df: pd.DataFrame,
    volta_structure: Dict[int, Dict[int, List[int]]],
    mc="mc",
    name="volta",
) -> pd.Series:
    """Create the input for `volta_structure` using get_volta_structure()"""
    mc2volta = {
        mc: volta
        for group in volta_structure.values()
        for volta, mcs in group.items()
        for mc in mcs
    }
    return df[mc].map(mc2volta).astype("Int64").rename(name)


def treat_group(mc: int, group: NDArray, logger=None) -> Dict[int, List[int]]:
    """Helper function for make_volta_col()


    Args:
      mc: MC of the first bar of the first measure.
      group:
          Input example: array([[93,  1,  1], [94,  2,  2], [96,  3,  1]])
          where columns are (MC, volta number, volta length).

    Returns:

    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    n = group.shape[0]
    mcs, numbers, lengths = group.T
    # check volta numbers
    expected = np.arange(1, n + 1)
    if (numbers != expected).any():
        logger.warning(
            f"Volta group of MC {mc} should have voltas {expected.tolist()} but has {numbers.tolist()}"
        )
    # check volta lengths
    frst = lengths[0]
    if (lengths != frst).any():
        logger.warning(
            f"Volta group of MC {mc} contains voltas with different lengths: {lengths.tolist()} Check for correct "
            f"computation of MNs "
            f"and copy this message into an IGNORED_WARNINGS file to make the warning disappear.",
            extra={"message_id": (4, mc)},
        )
    # check for overlaps and holes
    boundaries = np.append(mcs, mcs[-1] + group[-1, 2])
    correct = {
        i: np.arange(fro, to).tolist()
        for i, (fro, to) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1)
    }
    in_score = {
        i: [row[0] + i for i in range(row[2])] for i, row in enumerate(group, start=1)
    }
    if in_score != correct:
        logger.warning(
            f"The incorrect structure {in_score} of the volta groupa of MC {mc} has been corrected to {correct}."
        )
    return correct


# endregion helper functions


class MeasureList(LoggedClass):
    """Turns a _MSCX_bs4._measures DataFrame into a measure list and performs a couple of consistency checks on the
    score.

    Attributes
    ----------
    df : :obj:`pandas.DataFrame`
        The input DataFrame from _MSCX_bs4.raw_measures
    sections : :obj:`bool`, default True
        By default, section breaks allow for several anacrusis measures within the piece (relevant for `mc_offset`
        column)
        and make it possible to omit a repeat sign in the following bar (relevant for `next` column).
        Set to False if you want to ignore section breaks.
    secure : :obj:`bool`, default False
        By default, measure information from lower staves is considered to contain only redundant information.
        Set to True if you want to be warned about additional measure information from lower staves that is not taken
        into account.
    reset_index : :obj:`bool`, default True
        By default, the original index of `df` is replaced. Pass False to keep original index values.

    column2xml_tag : :obj:`dict`
        Dictionary of the relevant columns in `df` as present after the parse.
    ml : :obj:`pandas.DataFrame`
        The measure list in the making; the final result.
    volta_structure : :obj:`dict`
        Keys are first MCs of volta groups, values are dictionaries of {volta_no: [mc1, mc2 ...]}

    """

    column2xml_tag = {
        "barline": "voice/BarLine/subtype",
        "breaks": "LayoutBreak/subtype",
        "dont_count": "irregular",
        "endRepeat": "endRepeat",
        "jump_bwd": "Jump/jumpTo",
        "jump_fwd": "Jump/continueAt",
        "keysig_col": "voice/KeySig/accidental",
        "len_col": "Measure:len",
        "markers": "Marker/label",
        "mc": "mc",
        "numbering_offset": "noOffset",
        "play_until": "Jump/playUntil",
        "sigN_col": "voice/TimeSig/sigN",
        "sigD_col": "voice/TimeSig/sigD",
        "staff": "staff",
        "startRepeat": "startRepeat",
        "volta_start": "voice/Spanner/Volta/endings",
        "volta_length": "voice/Spanner/next/location/measures",
        "volta_frac": "voice/Spanner/next/location/fractions",
    }

    def __init__(
        self,
        df,
        sections=True,
        secure=True,
        reset_index=True,
        columns={},
        logger_cfg={},
    ):
        """

        Parameters
        ----------
        df
        sections : :obj:`bool`, optional
            By default, pieces containing section breaks (where counting MNs restarts) receive two more columns in the
            measures list, namely ``section`` and ``ambiguous_mn`` to grant access to MNs as shown in MuseScore.
            Pass False to not add such columns.
        secure
        reset_index
        logger_cfg : :obj:`dict`, optional
            The following options are available:
            'name': LOGGER_NAME -> by default the logger name is based on the parsed file(s)
            'level': {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}
            'file': PATH_TO_LOGFILE to store all log messages under the given path.
        """
        super().__init__(subclass="MeasureList", logger_cfg=logger_cfg)
        assert len(df) > 0, "Score contains no measures."
        self.df = df
        self.ml = pd.DataFrame()
        self.sections = sections
        self.secure = secure
        self.reset_index = reset_index
        self.volta_structure = {}
        col_names = list(self.column2xml_tag.keys())
        if any(True for c in columns if c not in col_names):
            wrong = [c for c in columns if c not in col_names]
            plural_s = "s" if len(wrong) > 1 else ""
            self.logger.warning(
                f"Wrong column name{plural_s} passed: {wrong}. Only {col_names} permitted."
            )
            columns = {k: v for k, v in columns.items() if k in col_names}
        self.column2xml_tag.update(columns)
        self.make_ml()

    def make_ml(self, section_breaks=True, secure=True, reset_index=True):
        self.sections = section_breaks
        self.secure = secure
        self.reset_index = reset_index

        # drops rows for all but the first staff, warning about competing information if secure=True
        self.ml = self.get_unique_measure_list()
        renaming = {v: k for k, v in self.column2xml_tag.items()}
        self.ml.rename(columns=renaming, inplace=True)
        necessary_columns = [
            "barline",
            "breaks",
            "dont_count",
            "endRepeat",
            "jump_bwd",
            "jump_fwd",
            "len_col",
            "markers",
            "numbering_offset",
            "play_until",
            "sigD_col",
            "sigN_col",
            "startRepeat",
            "volta_start",
            "volta_length",
        ]
        # create empty columns for all missing info_cols
        initial_columns = self.ml.columns.tolist()
        initial_columns += [c for c in necessary_columns if c not in initial_columns]

        self.ml = self.ml.reindex(columns=initial_columns, fill_value=pd.NA)
        if self.ml.jump_fwd.notna().any():
            self.ml.jump_fwd = self.ml.jump_fwd.replace({"/": pd.NA})

        volta_cols = {col: col for col in ("mc", "volta_start", "volta_length")}
        if "volta_frac" in self.ml.columns:
            volta_cols["frac_col"] = "volta_frac"
        self.volta_structure = get_volta_structure(
            self.ml, **volta_cols, logger=self.logger
        )
        new_columns = []
        new_columns.append(make_mn_col(self.ml))
        new_columns.append(make_keysig_col(self.ml))
        new_columns.append(
            (timesig_col := make_timesig_col(self.ml, logger=self.logger))
        )
        new_columns.append(
            make_actdur_col(
                len_col=self.ml["len_col"],
                timesig_col=timesig_col,
            )
        )
        new_columns.append(
            make_repeat_col(
                self.ml,
            )
        )
        new_columns.append(
            make_volta_col(
                self.ml,
                self.volta_structure,
            )
        )
        # the functions computing the final two columns rely on the previous columns, hence we concatenate here:
        self.ml = pd.concat([self.ml] + new_columns, axis=1)
        # for the final two, again, the last ('offset') relies on the presence of the second last ('next')
        self.ml = pd.concat(
            [
                self.ml,
                make_next_col(
                    self.ml,
                    self.volta_structure,
                    sections=self.sections,
                    logger=self.logger,
                ),
            ],
            axis=1,
        )
        self.ml = pd.concat(
            [
                self.ml,
                make_offset_col(
                    self.ml,
                    section_breaks="breaks",
                    logger=self.logger,
                ),
            ],
            axis=1,
        )
        if reset_index:
            self.ml.reset_index(drop=True, inplace=True)
        # rn = {
        #     self.cols[col]: col for col in ["barline", "dont_count", "numbering_offset"]
        # }
        # self.ml.rename(columns=rn, inplace=True)
        cols1 = ["mc", "mn", "keysig", "timesig", "act_dur", "mc_offset", "volta"]
        cols2 = ["numbering_offset", "dont_count"]
        cols3 = ["barline", "breaks", "repeats"]
        cols4 = ["markers", "jump_bwd", "jump_fwd", "play_until"]
        chunk1 = self.ml[cols1]
        chunk2 = self.ml[cols2].apply(pd.to_numeric).astype("Int64")
        chunk3 = self.ml[cols3]
        chunk4 = self.ml[cols4]
        chunks = [chunk1, chunk2, chunk3]
        if not chunk4.isna().all().all():
            chunks.append(chunk4)
        chunks.append(self.ml["next"])
        self.ml = pd.concat(chunks, axis=1)
        self.check_measure_numbers()

    def add_col(self, func, **kwargs):
        """Inserts or appends a column created by `func(df, **kwargs)`"""
        new_cols = func(self.ml, **kwargs)
        self.ml = pd.concat([self.ml, new_cols], axis=1)

    def get_unique_measure_list(self, **kwargs):
        """Keep only the measure information from the first staff.
        Uses: keep_one_row_each()

        Parameters
        ----------
        mc_col, staff_col : :obj:`str`, optional
            DataFrame columns where MC and staff can be found. Staff is to be dropped.
        secure : :obj:`bool`
            If the dropped rows contain additional information, set `secure` to True to
            be informed about the information being lost by the function keep_one_row_each().
        **kwargs: Additional parameter passed on to keep_one_row_each(). Ignored if `secure=False`.
        """
        if not self.secure:
            return self.df.drop_duplicates(subset=self.column2xml_tag["mc"]).drop(
                columns=self.column2xml_tag["staff"]
            )
        return keep_one_row_each(
            self.df,
            compress_col=self.column2xml_tag["mc"],
            differentiating_col=self.column2xml_tag["staff"],
            logger=self.logger,
        )

    def check_measure_numbers(
        self,
        mc_col="mc",
        mn_col="mn",
        act_dur="act_dur",
        mc_offset="mc_offset",
        dont_count="dont_count",
        numbering_offset="numbering_offset",
    ):
        """Checks if ms3's conventions for counting measure-like units are respected by the score and warns about
        discrepancies. Conventions can be satisfied either by using "Exclude from bar count" or by setting values for
        "Add to bar number".

        * anacrusis has MN 0; otherwise first measure as MN 1
        * Subsequent measures with irregular length shorter than the TimeSig's nominal length should add up and only
          the first increases the measure number, the other don't so that they have the same number
        * the measure of each alternative ending (volta) need to start with the same measure number
        """

        def ordinal(i):
            if i == 1:
                return "1st"
            elif i == 2:
                return "2nd"
            elif i == 3:
                return "3rd"
            return f"{i}th"

        mc2mn = dict(self.ml[[mc_col, mn_col]].itertuples(index=False))
        # Check measure numbers in voltas
        for volta_group in self.volta_structure.values():
            for volta_count, volta_mcs in enumerate(
                zip(*volta_group.values()), start=1
            ):
                m = volta_mcs[0]
                if not (mn := mc2mn.get(m)):
                    # this may arise when we are dealing with an excerpt where the volta has been removed
                    continue
                for mc_count, mc in enumerate(volta_mcs[1:], start=2):
                    if not (current_mn := mc2mn.get(mc)):
                        # this may arise when we are dealing with an excerpt where the volta is only partially included
                        continue
                    if current_mn != mn:
                        self.logger.warning(
                            f"MC {mc}, the {ordinal(volta_count)} measure of a {ordinal(mc_count)} volta, should have "
                            f"MN {mn}, not MN {current_mn}.",
                            extra={"message_id": (2, mc)},
                        )

        # Check measure numbers for split measures
        error_mask = (
            (self.ml[mc_offset] > 0)
            & self.ml[dont_count].isna()
            & self.ml[numbering_offset].isna()
        )
        n_errors = error_mask.sum()
        if n_errors > 0:
            mcs = self.ml.loc[error_mask, mc_col]
            mcs_int = tuple(mcs.values)
            mcs_str = ", ".join(mcs.astype(str))
            context_mask = (
                error_mask
                | error_mask.shift(-1).fillna(False)
                | error_mask.shift().fillna(False)
            )
            context = self.ml.loc[
                context_mask,
                [mc_col, mn_col, act_dur, mc_offset, dont_count, numbering_offset],
            ]
            plural = n_errors > 1
            self.logger.warning(
                f"MC{'s' if plural else ''} {mcs_str} seem{'' if plural else 's'} to be offset from the MN's "
                f"beginning but ha{'ve' if plural else 's'} not been excluded from barcount. Context:\n{context}",
                extra={"message_id": (1, *mcs_int)},
            )


class NextColumnMaker(LoggedClass):
    def __init__(self, df, volta_structure, sections=True, logger_cfg=None):
        super().__init__(subclass="NextColumnMaker", logger_cfg=logger_cfg)
        self.sections = sections
        self.mc = df.mc  # Series
        if self.mc.isna().any():
            self.logger.warning(
                "MC column contains NaN which will lead to an incorrect 'next' column."
            )
        nxt = self.mc.astype("Int64").shift(-1).fillna(-1).map(lambda x: [x])
        last_row = df.iloc[-1]
        self.last_mc = last_row.mc
        self.next = {mc: nx for mc, nx in zip(self.mc, nxt)}
        fines = df.markers.fillna("").str.contains("fine")
        if fines.any():
            if fines.sum() > 1:
                self.logger.warning(
                    "ms3 currently does not deal with more than one Fine. Using last measure as Fine."
                )
            elif last_row.repeats != "end" and df.jump_bwd.isna().all():
                fine_mc = df.loc[fines, "mc"].values[0]
                self.logger.warning(
                    "Piece has a Fine but the last MC is missing a repeat sign or a D.C. (da capo) or "
                    "D.S. (dal segno). Ignoring Fine.",
                    extra={"message_id": (20, fine_mc)},
                )
            else:
                fine_mc = df[fines].iloc[0].mc
                if -1 not in self.next[fine_mc]:
                    volta_mcs = dict(df.loc[df.volta.notna(), ["mc", "volta"]].values)
                    if fine_mc in volta_mcs:
                        # voltas can be reached only a single time, hence the name
                        self.next[fine_mc] = [-1]
                    else:
                        self.next[fine_mc].append(-1)
                    if fine_mc != self.last_mc:
                        if -1 in self.next[self.last_mc]:
                            self.next[self.last_mc].remove(-1)
                        else:
                            self.logger.warning(
                                f"Which MC has -1 in the 'next' column at the moment I've set it to "
                                f"'Fine' measure {fine_mc}?"
                            )
                        self.last_mc = fine_mc
                    self.logger.debug(f"Set the Fine in MC {fine_mc} as final measure.")

        if df.jump_bwd.notna().any():
            markers = defaultdict(list)
            for t in df.loc[df.markers.notna(), ["mc", "markers"]].itertuples(
                index=False
            ):
                for marker in t.markers.split(" & "):
                    markers[marker].append(t.mc)
            # markers = {marker: mcs.to_list() for marker, mcs in df.groupby('markers').mc}

            def jump2marker(
                from_mc: int, marker: Optional[str], untill: Optional[str] = None
            ) -> Tuple[Optional[int], Optional[int]]:
                def get_marker_mc(m, untilll=False):
                    mcs = markers[m]
                    if len(mcs) > 1:
                        if untilll:
                            self.logger.warning(
                                f"After jumping from MC {mc} to {marker}, the music is supposed to play until "
                                f"label {m} but there are {len(mcs)} of them: {mcs}. Picking the first one."
                            )
                        else:
                            self.logger.warning(
                                f"MC {mc} is supposed to jump to label {m} but there are {len(mcs)} of them: {mcs}. "
                                f"Picking the first one."
                            )
                    return mcs[0]

                if marker == "start":
                    jump_to_mc = 1
                elif marker in markers:
                    jump_to_mc = get_marker_mc(marker)
                else:
                    self.logger.warning(
                        f"MC {from_mc} is supposed to jump to label {marker} but there is no corresponding marker "
                        f"in the score. Ignoring.",
                        extra={"message_id": (22, from_mc)},
                    )
                    return None, None

                if pd.isnull(untill):
                    end_of_jump_mc = None
                elif untill == "end":
                    end_of_jump_mc = self.last_mc
                elif untill in markers:
                    end_of_jump_mc = get_marker_mc(untill, True)
                else:
                    end_of_jump_mc = None
                    self.logger.warning(
                        f"After jumping from MC {from_mc} to {marker}, the music is supposed to play until "
                        f"label {untill} but there is no corresponding marker in the score. Ignoring.",
                        extra={"message_id": (21, from_mc)},
                    )
                return jump_to_mc, end_of_jump_mc

            bwd_jumps = df.loc[
                df.jump_bwd.notna(), ["mc", "jump_bwd", "jump_fwd", "play_until"]
            ]  # .copy()
            # bwd_jumps.jump_fwd = bwd_jumps.jump_fwd.replace({'/': None})
            for mc, jumpb, jumpf, until in bwd_jumps.itertuples(name=None, index=False):
                jump_to_mc, end_of_jump_mc = jump2marker(mc, jumpb, until)
                if not pd.isnull(jump_to_mc):
                    previous_value = self.next[mc]
                    if end_of_jump_mc == mc:
                        self.next[mc] = [jump_to_mc] + previous_value
                        self.logger.debug(
                            f"Backward jump to '{jumpb}' (MC {jump_to_mc}) with 'until {until}' "
                            f"resolving to the current MC {mc}: "
                            f"Prepended {jump_to_mc} to the 'next' value {previous_value} rather than "
                            f"replacing it."
                        )
                    else:
                        self.next[mc] = [jump_to_mc]
                        self.logger.debug(
                            f"Replacing 'next' value {previous_value} of MC {mc} with the '{jumpb}' in "
                            f"MC {jump_to_mc}."
                        )
                else:
                    self.logger.debug(f"Could not include backward jump from MC {mc}.")
                if not pd.isnull(jumpf):
                    if end_of_jump_mc is None:
                        if jumpf in markers:
                            reason = f"{until} was not found in the score."
                        else:
                            reason = "neither of them was found in the score."
                        if len(self.next[mc]) > 0:
                            self.logger.warning(
                                f"The jump from MC {mc} to {self.next[mc][0]} is supposed to jump "
                                f"forward from {until} to {jumpf}, but {reason}"
                            )
                    else:
                        to_mc, _ = jump2marker(end_of_jump_mc, jumpf)
                        if not pd.isnull(to_mc):
                            n_existing_next = len(self.next[end_of_jump_mc])
                            if (
                                n_existing_next > 0
                                and self.next[end_of_jump_mc][-1] == -1
                            ):
                                self.next[end_of_jump_mc] = self.next[end_of_jump_mc][
                                    :-1
                                ] + [to_mc, -1]
                            else:
                                self.next[end_of_jump_mc].append(to_mc)
                            self.logger.debug(
                                f"Included forward jump from the {until} in MC {end_of_jump_mc} to the "
                                f"{jumpf} in MC {to_mc} "
                            )
                        else:
                            self.logger.debug(
                                f"Could not include forward jump from the {jumpb} in MC {jump_to_mc}."
                            )
        else:  # no backward jumps
            bwd_jumps = pd.DataFrame(columns=["mc"])

        self.repeats = dict(df[["mc", "repeats"]].values)
        self.start = None
        self.potential_start = None
        self.potential_ending = None
        self.check_volta_repeats = {}
        self.wasp_nest = {}
        for first_mc, group in volta_structure.items():
            firsts = []
            lasts = []
            last_volta = max(group)
            last_group = group[last_volta]
            if len(last_group) == 0:
                try:
                    previous_mc = max(
                        mc
                        for volta, mcs in group.items()
                        for mc in mcs
                        if volta < last_volta
                    )
                    last_volta_mc = previous_mc + 1
                    group[last_volta] = [last_volta_mc]
                    mc_after_voltas = last_volta_mc + 1  # wild guess
                except ValueError:
                    self.logger.warning(
                        f"Last volta does not indicate any MCs: {group}. Column 'next' will probably "
                        f"be invalid and unfolding might fail."
                    )
                    mc_after_voltas = None
                    del group[last_volta]
            else:
                mc_after_voltas = max(last_group) + 1
            if mc_after_voltas not in self.next:
                mc_after_voltas = None
            for volta, mcs in group.items():
                if len(mcs) == 0:
                    continue
                # every volta except the last needs will have the `next` value replaced either by the startRepeat MC or
                # by the first MC after the last volta
                if volta < last_volta:
                    lasts.append(mcs[-1])
                # the bar before the first volta will have first bar of every volta as `next`
                firsts.append(mcs[0])
            if first_mc > 1:
                # prepend first MC of each volta to the 'next' tuple of the preceding measure
                self.next[first_mc - 1] = firsts + self.next[first_mc - 1][1:]
            # check_volta_repeats keys are last MCs of all voltas except last voltas, values are all False at the
            # beginning and they are set to True if their value has been changed to something else than the next MC
            backward_jumps = bwd_jumps.mc.to_list()
            wasp_nest = [
                last_mc
                for last_mc in lasts
                if not pd.isnull(self.repeats[last_mc])
                and self.repeats[last_mc] == "end"
                and last_mc not in backward_jumps
            ]
            for last_mc in lasts:
                has_repeat = not pd.isnull(self.repeats[last_mc])
                has_backward_jump = last_mc in backward_jumps
                if has_backward_jump:
                    self.check_volta_repeats[last_mc] = True
                elif has_repeat:
                    if self.repeats[last_mc] == "end":
                        self.check_volta_repeats[last_mc] = False
                        self.wasp_nest[last_mc] = wasp_nest
                        # for voltas with and endRepeat, the wasp_nest makes sure that once the sections' beginning is
                        # determined in end_section(), it becomes their 'next' value
                    else:
                        self.logger.warning(
                            f"MC {last_mc}, which is the last MC of a volta, has a different repeat sign "
                            f"than 'end': {self.repeats[last_mc]}"
                        )
                elif mc_after_voltas is None:
                    self.logger.warning(
                        f"MC {last_mc} is the last MC of a volta but has neither a repeat sign or jump, "
                        f"nor is there a MC after the volta group where to continue."
                    )
                else:
                    self.next[last_mc] = [mc_after_voltas]

    def start_section(self, mc):
        if self.start is not None:
            if self.potential_ending is None:
                self.logger.warning(
                    f"""The startRepeat in MC {self.start} is missing its endRepeat.
For correction, MC {mc - 1} is interpreted as such because it precedes the next startRepeat.""",
                    extra={"message_id": (5, self.start)},
                )
                self.end_section(mc - 1)
            else:
                ending, reason = self.potential_ending
                self.logger.warning(
                    f"""The startRepeat in MC {self.start} is missing its endRepeat.
For correction, MC {ending} is interpreted as such because it {reason}."""
                )
                self.end_section(ending)
        self.start = mc
        self.potential_start = None
        self.potential_ending = None

    def end_section(self, mc):
        if self.start is not None:
            start = self.start
        elif self.potential_start is not None:
            start, reason = self.potential_start
            if reason == "firstMeasure":
                self.logger.debug(
                    f"MC {start} has been inferred as startRepeat for the endRepeat in MC {mc} because it is the first "
                    f"bar of the piece."
                )
            else:
                msg = f"""The endRepeat in MC {mc} is missing its startRepeat.
For correction, MC {start} is interpreted as such because it {reason}."""
                if "section break" in msg:
                    self.logger.debug(msg)
                else:
                    self.logger.info(msg)
        else:
            start = None

        if mc in self.check_volta_repeats:
            if self.check_volta_repeats[mc]:
                # this one has a backwards_jump and doesn't need amending
                pass
            else:
                self.check_volta_repeats[mc] = True
                if mc in self.wasp_nest:
                    if start is None:
                        self.logger.error(
                            f"No starting point for the repeatEnd in MC {mc} could be determined. It is being ignored."
                        )
                    else:
                        volta_endings = self.wasp_nest[mc]
                        for e in volta_endings:
                            self.next[e] = [start]
                            del self.wasp_nest[e]
                        self.start = None
        elif start is None:
            self.logger.error(
                f"No starting point for the repeatEnd in MC {mc} could be determined. It is being "
                f"ignored."
            )
        else:
            self.next[mc] = [start] + self.next[mc]
            if self.potential_start is not None:
                pot_mc, reason = self.potential_start
                if pot_mc == mc + 1:
                    self.potential_start = (
                        pot_mc,
                        reason + " and the previous endRepeat",
                    )
                else:
                    self.potential_start = (
                        mc + 1,
                        "is the first bar after the previous endRepeat",
                    )
            else:
                self.potential_start = (
                    mc + 1,
                    "is the first bar after the previous endRepeat",
                )
            self.start = None

    def treat_input(self, mc, repeat, section_break=False) -> None:
        if not pd.isnull(section_break) and section_break and mc != self.last_mc:
            self.potential_ending = (mc, "precedes a section break")
            self.potential_start = (mc + 1, "follows a section break")
        if pd.isnull(repeat):
            return
        if repeat == "firstMeasure":
            self.potential_start = (mc, "firstMeasure")
        elif repeat == "start":
            self.start_section(mc)
        elif repeat == "startend":
            self.start_section(mc)
            self.end_section(mc)
        elif repeat == "end":
            self.end_section(mc)
        elif repeat == "lastMeasure":
            if self.start is not None:
                self.potential_ending = (mc, "is the last bar of the piece.")
                self.start_section(mc + 1)
                self.start = None
