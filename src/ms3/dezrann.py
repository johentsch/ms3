#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DCML to Dezrann
===============

Standalone script to convert score annotations from a tabular format (one row per label) into the JSON format .dez
used by the Dezrann annotation tool (https://www.dezrann.net) developed by the Algomus group. Its only dependency is
pandas, i.e. it can be copied and run without ms3 being installed::

    python dezrann.py path/to/corpus/harmonies

Input
-----

The script converts the annotation tables that ms3 extracts from MuseScore files and that every DCML corpus
(https://github.com/DCMLab/dcml_corpora) comes with:

* Label tables, by default in a folder called ``harmonies`` (``ms3 extract -X``, X for 'expanded') or ``labels``
  (``ms3 extract -L``). Using the expanded DCML harmony labels ensures that only valid DCML labels are converted and
  makes available the columns ``chord``, ``localkey``, ``phraseend``, and ``cadence``. In principle, labels of all
  kinds can be converted as raw labels (``label`` column) as long as they come in this tabular format.
* Measure tables (``ms3 extract -M``), by default in a sibling folder called ``measures``.

Both facet-suffixed file names (``<piece>.harmonies.tsv`` & ``<piece>.measures.tsv``, the current default) and
un-suffixed names (``harmonies/<piece>.tsv`` & ``measures/<piece>.tsv``, older corpus versions) are matched by piece
name. The output file is called ``<piece>.dez``.

Columns used from the label tables:

* ``mc``: measure count, i.e. the index of the XML measure (always starting from 1, unlike measure numbers ``mn``).
* ``mc_onset``: distance of the label from the beginning of the measure, as a fraction of a whole note.
* ``quarterbeats`` and, for pieces with first and second endings, ``quarterbeats_all_endings`` (see below). They are
  computed from ``mc`` and ``mc_onset`` if missing.
* the label columns ``label``, ``chord``, ``localkey``, ``phraseend``, ``cadence`` (whichever are needed).

Columns used from the measure tables: ``mc``, ``act_dur`` (actual length of a measure as a fraction of a whole note),
``quarterbeats``, and, if the piece has first and second endings, ``volta`` and ``quarterbeats_all_endings``.

Positions: quarterbeats and measure maps
----------------------------------------

The script presents a first application of what is to become a formal standard of a "measure map"; see first
discussion points at

* https://gitlab.com/algomus.fr/dezrann/dezrann/-/issues/1030#note_1122509147
* https://github.com/MarkGotham/bar-measure/

As an early proxy of a measure map, the current version uses the measure tables. They are required for computing
correct quarter-note positions without having to re-parse the entire score, they make it possible to validate the
conversion output, and they help avoiding the confusion that necessarily arises when several addressing schemes are
at play.

From a technical perspective, offsets in the sense of "distance from the origin" represent the primary mechanism of
referencing positions in a text (character counts being the default in NLP). Music scores are typically aligned with
a time line of "musical time", an alignment which is frequently expressed as float values representing an event's
distance from the score's beginning, measured in quarter notes, here referred to as quarterbeats. Dezrann, too,
positions labels in quarterbeats. The fundamental problem is ensuring that quarterbeat positions refer to the same time
line. The commonplace score encoding formats do not indicate quarterbeat positions. Instead, they structure musical
time in a sequence of containers, generally called "measures", each of which represents a time line starting from 0.
Counting measure units (of some kind) therefore represents the second prevalent way of indicating positions in a
score, together with an event's distance from the container's beginning which we call "onset" to avoid
terminological confusion.

Looking at a single score, there is an unambiguous mapping between the two types of positions:
``event_offset = measure_offset + event_onset``. Problems arise, however, when information from one score is to be set
into relation with timed information from another source. This is a wide-spread problem in music research and corpus
studies where data from different sources with different ways of expressing timestamps frequently needs to be
aligned, often in absence of the original score. Currently, there is no standardized way of storing such alignments
for later re-use. Hence the idea of a central mapping file for storing alignments between positions given as
quarterbeats, measure+onset, recording timestamps in seconds, IDs, and other data relevant for score addressability.

**Different types of quarterbeats**

All TSV files issued by the DCML come with the column ``quarterbeats`` indicating every event's offset from the
score's beginning (position 0). In the case of first/second endings ("voltas"), however, these values do not take into
account any but the last ending, with the rationale that they should represent the temporal proportion of a single
playthrough without any repetitions; events in the other endings have no ``quarterbeats`` value. Dezrann, on the other
hand, displays the score as written, i.e. with all endings laid out one after the other. For pieces with voltas, the
script therefore uses the strict, measuring-stick-based column ``quarterbeats_all_endings`` which the DCML measure
tables include in this case. The simple formula mentioned above has its analogue
``quarterbeats_all_endings = quarterbeats_all_endings(measure) + 4 * mc_onset``.

The durations of the Dezrann labels are computed as the distance to the subsequent label of the same type (the last
label lasting until the end of the score). For this to be correct in the presence of voltas, harmony, local key, and
raw labels that are active before a group of alternative endings are repeated at the beginning of the second (third...)
ending if that ending does not begin with a label of the same type. Phrase beginnings and cadences are not repeated.

Validating the output
---------------------

Going from a :class:`DcmlLabel` to a :class:`DezrannLabel` is straightforward because they exchange positions as
quarterbeats. Validation, on the other hand, requires relating the output .dez file with the score it is laid over in
Dezrann. In the interface, positions are shown to the user in terms of measure + onset, which can be compared to the
values ``mc``/``mn`` and ``mc_onset``/``mn_onset`` in the original TSV files.

Output
------

A JSON Dezrann file (.dez) per piece containing the selected labels, aligned with the score. Each type of label
(Harmony, Local Key, Phrase, Cadence, Raw label) is assigned to one of Dezrann's six annotation lines via a layout
rule. Example::

    {
      "labels": [
        {"type": "Harmony", "start": 0.0, "duration": 4.0, "tag": "I", "layers": ["DCML"]},
        {"type": "Harmony", "start": 4.0, "duration": 4.0, "tag": "V(64)", "layers": ["DCML"]},
        {"type": "Local Key", "start": 0.0, "duration": 48.0, "tag": "I", "layers": ["DCML"]},
        ...
      ],
      "meta": {
        "layout": [
          {"filter": {"type": "Harmony"}, "style": {"line": "bot.1"}},
          {"filter": {"type": "Local Key"}, "style": {"line": "bot.2"}},
          ...
        ]
      }
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from fractions import Fraction
from typing import (
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    TypeAlias,
    TypedDict,
    Union,
)

import pandas as pd

logger = logging.getLogger(__name__)

LINE_VALUES = {1: "top.1", 2: "top.2", 3: "top.3", 4: "bot.1", 5: "bot.2", 6: "bot.3"}
"""The six annotation lines of the Dezrann app, three above ('top') and three below ('bot') the score."""

DEZ_LINE_ARGS = (0, 1, 2, 3, 4, 5, 6, -1, -2, -3)
"""Accepted integer arguments for a Dezrann line. 0 means 'do not include', -1, -2, -3 are aliases for 4, 5, 6."""

DezrannLine: TypeAlias = Literal["top.1", "top.2", "top.3", "bot.1", "bot.2", "bot.3"]
DezrannLayer: TypeAlias = Union[DezrannLine, Literal[0, 1, 2, 3, 4, 5, 6, -1, -2, -3]]
"""Specifies one of Dezrann's six annotation lines either by name or by number (see :func:`transform_line_argument`).
"""

LABEL_TYPES = {
    "cadence": "Cadence",
    "chord": "Harmony",
    "localkey": "Local Key",
    "phraseend": "Phrase",
    "label": "Raw label",
}
"""Maps the label columns of a DCML label table to the Dezrann label type under which they are written. Dezrann's
types are a configurable vocabulary; label types unknown to Dezrann are displayed nonetheless. Raw labels need a type
distinct from 'Harmony' so that they can be placed on a different line via the layout."""

REPEATED_AT_VOLTA_BEGINNINGS = ("chord", "localkey", "label")
"""Label columns whose values describe segments that continue into every alternative ending. The label active before
a group of endings is repeated at the beginning of the second, third... ending if no label of the same type exists
there. Phrase beginnings and cadences (point events) are not repeated."""

LABEL_FACETS = ("harmonies", "expanded", "labels")
"""Facet suffixes of label tables recognized in file names, in order of preference."""

OTHER_FACETS = (
    "measures",
    "notes",
    "rests",
    "notes_and_rests",
    "chords",
    "events",
    "form_labels",
)
"""Facet suffixes of other TSV files that are ignored when looking for label tables."""

FRACTION_COLUMNS = (
    "mc_onset",
    "mn_onset",
    "quarterbeats",
    "quarterbeats_all_endings",
    "duration_qb",
    "act_dur",
    "mc_offset",
)
"""Columns that may contain fractions such as '1/3' and are read as :class:`fractions.Fraction`."""

_CADENCE_AND_PHRASE_REGEX = re.compile(r"\|[A-Z]+(?:\.[^{}\\]+)?|[{}\\]")
"""Matches cadence (e.g. '|PAC', '|HC.ph') and phrase ('{', '}', '}{', '\\\\') annotations within a DCML label."""


class DezrannLabel(TypedDict):
    """Represents one label in a .dez file."""

    type: str
    start: float
    duration: float
    # line: str # Determined by the meta-layout
    tag: str
    layers: List[str]


class DezrannDict(TypedDict):
    """Represents one .dez file."""

    labels: List[DezrannLabel]
    meta: Dict


class DcmlLabel(TypedDict):
    """Represents one label from a TSV annotation file, positioned on the measuring-stick timeline."""

    label: str
    quarterbeats: float
    duration: float


class ConversionSummary(NamedTuple):
    """Number of files processed by :func:`main`."""

    written: int = 0
    skipped: int = 0
    failed: int = 0


def transform_line_argument(line: Optional[Union[int, str]]) -> Optional[DezrannLine]:
    """Turns an argument specifying a Dezrann line into one of the six possible Dezrann line values.

    Accepts a number between -3 and 6 (as int or str), where 1-3 correspond to 'top.1' to 'top.3' and 4-6 to
    'bot.1' to 'bot.3'; -1, -2, -3 are aliases for 4, 5, 6; and 0 or None mean 'no line' and return None. Valid
    Dezrann line names such as 'bot.1' are returned as they are, so the function is idempotent.

    Raises:
        ValueError: If the argument is not valid.
    """
    if line is None:
        return None
    if isinstance(line, str):
        line = line.strip()
        if line in LINE_VALUES.values():
            return line
    try:
        assert not isinstance(line, float) or line.is_integer()
        line = int(line)
        assert line in DEZ_LINE_ARGS
    except (TypeError, ValueError, AssertionError):
        raise ValueError(
            f"{line!r} is not a valid Dezrann line. Pass an integer within [-3, 6] or one of "
            f"{', '.join(LINE_VALUES.values())}."
        )
    if line == 0:
        return None
    if line < 0:
        line = abs(line) + 3
    return LINE_VALUES[line]


def resolve_line_arguments(
    harmonies: Optional[DezrannLayer] = None,
    keys: Optional[DezrannLayer] = None,
    phrases: Optional[DezrannLayer] = None,
    raw: Optional[DezrannLayer] = None,
) -> Dict[str, Optional[DezrannLine]]:
    """Translates the line arguments for the four types of segment labels into Dezrann line values (or None)
    and makes sure that no line has been selected more than once.

    Raises:
        ValueError: If one of the arguments is invalid or if several arguments correspond to the same line.
    """
    original = dict(harmonies=harmonies, keys=keys, phrases=phrases, raw=raw)
    resolved = {arg: transform_line_argument(value) for arg, value in original.items()}
    selected = {arg: line for arg, line in resolved.items() if line is not None}
    if len(set(selected.values())) < len(selected):
        clashing = [
            line
            for line in set(selected.values())
            if list(selected.values()).count(line) > 1
        ]
        selected_args = {
            arg: f"{original[arg]!r} => {line}"
            for arg, line in selected.items()
            if line in clashing
        }
        raise ValueError(
            f"You selected the same annotation line more than once: {selected_args}."
        )
    return resolved


def safe_frac(s) -> Optional[Fraction]:
    """Converts a TSV cell such as '1/3' or '4.0' into a Fraction. Empty cells are returned as None.

    Raises:
        ValueError: If the value cannot be converted.
    """
    if s is None or (isinstance(s, float) and pd.isna(s)) or s is pd.NA:
        return None
    if isinstance(s, str):
        s = s.strip()
        if s == "":
            return None
    try:
        return Fraction(s)
    except (TypeError, ValueError, ZeroDivisionError):
        raise ValueError(f"Could not convert {s!r} to a fraction.")


def _to_float(series: pd.Series) -> pd.Series:
    """Converts a column containing Fractions, numbers, numerical strings, None, or NA to float (NA => NaN)."""
    return series.map(
        lambda x: float("nan") if pd.isna(x) else float(safe_frac(x))
    ).astype(float)


def get_volta_groups(mc2volta: pd.Series) -> List[List[int]]:
    """Takes a Series where the index has measure counts and values are NA for 'normal' measures and 1, 2... for
    measures belonging to a first, second... ending. Returns for each group a list of MCs each of which pertains
    to the first measure of an alternative ending. For example, two alternative two-bar endings in MC [15, 16][17, 18]
    would figure as [15, 17] in the result list.
    """
    volta_groups = []
    current_group = []
    previous_number = None
    for mc, number in mc2volta.sort_index().items():
        if pd.isna(number):
            # current group ends, if there is one
            if current_group:
                volta_groups.append(current_group)
                current_group = []
            previous_number = None
            continue
        number = int(number)
        if number == previous_number:
            # the current ending continues
            continue
        if current_group and previous_number is not None and number < previous_number:
            # a new group of endings follows immediately upon the previous one
            volta_groups.append(current_group)
            current_group = []
        current_group.append(mc)
        previous_number = number
    if current_group:
        # piece ends with an alternative ending
        volta_groups.append(current_group)
    return volta_groups


def has_voltas(measures: pd.DataFrame) -> bool:
    """Returns True if the measures table indicates first/second endings in the column 'volta'."""
    return "volta" in measures.columns and bool(measures.volta.notna().any())


def get_measure_offsets(measures: pd.DataFrame) -> pd.Series:
    """Returns a Series mapping each MC to its offset in quarterbeats on the measuring-stick timeline (i.e. with all
    endings laid out one after the other), as Fractions. Uses the column 'quarterbeats_all_endings' if present (pieces
    with voltas), otherwise 'quarterbeats'. If neither is available (or 'quarterbeats' is incomplete, as for first
    endings), the offsets are computed from the actual measure durations ('act_dur').
    """
    M = measures.set_index("mc").sort_index()
    for column in ("quarterbeats_all_endings", "quarterbeats"):
        if column in M.columns:
            offsets = M[column].map(safe_frac)
            if offsets.notna().all():
                return offsets.rename("quarterbeats")
    if "act_dur" not in M.columns:
        raise ValueError(
            "The measures table needs to have one of the columns 'quarterbeats_all_endings', 'quarterbeats', "
            "or 'act_dur'."
        )
    act_dur_qb = M.act_dur.map(safe_frac) * 4
    offsets = act_dur_qb.cumsum().shift(fill_value=Fraction(0))
    return offsets.rename("quarterbeats")


def get_end_of_score(measures: pd.DataFrame) -> float:
    """Returns the measuring-stick offset of the end of the score in quarterbeats."""
    offsets = get_measure_offsets(measures)
    last_mc = offsets.index[-1]
    last_act_dur = safe_frac(measures.set_index("mc").loc[last_mc, "act_dur"])
    return float(offsets.loc[last_mc] + last_act_dur * 4)


def get_label_quarterbeats(labels: pd.DataFrame, measures: pd.DataFrame) -> pd.Series:
    """Returns the measuring-stick positions of the labels as floats. If the labels come with the relevant column
    ('quarterbeats_all_endings' if the measures have it, 'quarterbeats' otherwise), it is used; missing values (e.g.
    in first endings) are computed as ``offset(mc) + 4 * mc_onset``.
    """
    if "quarterbeats_all_endings" in measures.columns:
        column = "quarterbeats_all_endings"
    elif not has_voltas(measures):
        column = "quarterbeats"
    else:
        # labels' 'quarterbeats' would not be on the measuring-stick timeline
        column = None
    if column is not None and column in labels.columns:
        quarterbeats = labels[column].map(safe_frac)
    else:
        quarterbeats = pd.Series(None, index=labels.index, dtype=object)
    missing = quarterbeats.isna()
    if missing.any():
        if not {"mc", "mc_onset"}.issubset(labels.columns):
            raise ValueError(
                f"The positions of {missing.sum()} labels cannot be computed because the labels lack the columns "
                f"'mc' and/or 'mc_onset'."
            )
        offsets = get_measure_offsets(measures)
        computed = (
            labels.loc[missing, "mc"].map(offsets)
            + labels.loc[missing, "mc_onset"].map(safe_frac) * 4
        )
        quarterbeats = quarterbeats.where(~missing, computed)
    return _to_float(quarterbeats).rename("quarterbeats")


def _strip_cadence_and_phrase(label: str) -> str:
    """Removes cadence and phrase annotations from a DCML label so it can be repeated at a volta beginning."""
    return _CADENCE_AND_PHRASE_REGEX.sub("", label)


def _repeat_labels_at_volta_beginnings(
    label_and_qb: pd.DataFrame,
    measures: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:
    """For each group of alternative endings, repeats the label active before the group at the beginning of the
    second, third... ending if no label exists at this position. Returns the labels sorted by position.
    """
    if not has_voltas(measures):
        return label_and_qb
    offsets = get_measure_offsets(measures)
    volta_groups = get_volta_groups(measures.set_index("mc").volta)
    existing_positions = {round(qb, 6) for qb in label_and_qb.quarterbeats}
    repeated_labels = []
    for group in volta_groups:
        group_beginning = float(offsets.loc[group[0]])
        labels_before_group = label_and_qb.loc[
            label_and_qb.quarterbeats < group_beginning, "label"
        ]
        if len(labels_before_group) == 0:
            logger.debug(
                f"No {label_column!r} label before the endings beginning in MC {group[0]}, nothing to repeat."
            )
            continue
        repeated_label = labels_before_group.iloc[-1]
        if label_column == "label":
            repeated_label = _strip_cadence_and_phrase(repeated_label)
            if repeated_label == "":
                continue
        for mc in group[1:]:
            volta_beginning_qb = float(offsets.loc[mc])
            if round(volta_beginning_qb, 6) in existing_positions:
                continue
            logger.debug(
                f"Repeating {label_column!r} label {repeated_label!r} at the beginning of the ending in MC {mc}."
            )
            repeated_labels.append(
                dict(label=repeated_label, quarterbeats=volta_beginning_qb)
            )
    if len(repeated_labels) == 0:
        return label_and_qb
    label_and_qb = pd.concat(
        [label_and_qb, pd.DataFrame(repeated_labels)], ignore_index=True
    )
    return label_and_qb.sort_values("quarterbeats", kind="stable", ignore_index=True)


def dcml_labels2dicts(
    labels: pd.DataFrame,
    measures: pd.DataFrame,
    label_column: str = "label",
) -> List[DcmlLabel]:
    """Turns one column of a label table into a list of labels with measuring-stick positions and durations.

    Parameters
    ----------
    labels:
        Dataframe as found in the 'harmonies' folder of a DCML corpus. Needs to have the columns
        {'mc': int, 'mc_onset': fractions.Fraction} and/or quarterbeat positions ('quarterbeats' and, for pieces with
        voltas, 'quarterbeats_all_endings'), plus the ``label_column``. Fraction columns may also contain strings
        such as '1/3'.
    measures:
        Dataframe as found in the 'measures' folder of a DCML corpus. Requires the columns {'mc': int,
        'act_dur': fractions.Fraction} and 'quarterbeats', or, for pieces with voltas, 'volta' and
        'quarterbeats_all_endings' (ms3 >= 1.0.0).
    label_column: {'label', 'chord', 'localkey', 'cadence', 'phraseend'}
        The column that is to be used as label string. Defaults to 'label'. For 'localkey', only key changes are
        included; for 'phraseend', only phrase beginnings; and labels from the 'cadence' column have duration 0.

    Returns
    -------
        List of dictionaries where each represents one label, sorted by position.
    """
    if label_column not in labels.columns:
        raise ValueError(f"The labels have no column {label_column!r}.")
    end_of_score = get_end_of_score(measures)
    quarterbeats = get_label_quarterbeats(labels, measures)
    label_and_qb = pd.concat(
        [labels[label_column].rename("label"), quarterbeats], axis=1
    )
    n_before = len(label_and_qb.index)
    label_and_qb = label_and_qb[
        label_and_qb.label.notna() & (label_and_qb.label.astype(str) != "")
    ]
    label_and_qb = label_and_qb.astype({"label": str})
    if label_and_qb.quarterbeats.isna().any():
        logger.warning(
            f"Dropping {label_and_qb.quarterbeats.isna().sum()} {label_column!r} labels whose position could not be "
            f"determined."
        )
        label_and_qb = label_and_qb[label_and_qb.quarterbeats.notna()]
    if label_column == "phraseend":
        label_and_qb = label_and_qb[label_and_qb.label.str.contains("{", regex=False)]
    label_and_qb = label_and_qb.sort_values(
        "quarterbeats", kind="stable", ignore_index=True
    )
    if label_column in REPEATED_AT_VOLTA_BEGINNINGS:
        label_and_qb = _repeat_labels_at_volta_beginnings(
            label_and_qb, measures, label_column
        )
    if label_column == "localkey":
        # keep only key changes
        label_and_qb = label_and_qb[label_and_qb.label != label_and_qb.label.shift()]
    logger.debug(
        f"Creating {len(label_and_qb.index)} {label_column!r} labels from {n_before} rows."
    )
    if label_column == "cadence":
        duration = pd.Series(0.0, dtype=float, index=label_and_qb.index)
    else:
        qb_column = label_and_qb.quarterbeats
        duration = qb_column.shift(-1).fillna(end_of_score) - qb_column
    transformed_df = pd.concat(
        [label_and_qb, duration.rename("duration").astype(float)], axis=1
    )
    return transformed_df.to_dict(orient="records")


def make_dezrann_label(
    label_type: str,
    quarterbeats: float,
    duration: float,
    label: str,
    origin: Union[str, Tuple[str, ...]],
) -> DezrannLabel:
    if isinstance(origin, str):
        layers = [origin]
    else:
        layers = list(origin)
    return DezrannLabel(
        type=label_type,
        start=float(quarterbeats),
        duration=float(duration),
        tag=str(label),
        layers=layers,
    )


def convert_dcml_list_to_dezrann_list(
    values_dict: List[DcmlLabel],
    label_type: str,
    origin: Union[str, Tuple[str, ...]] = "DCML",
) -> List[DezrannLabel]:
    return [
        make_dezrann_label(
            label_type=label_type,
            quarterbeats=e["quarterbeats"],
            duration=e["duration"],
            label=e["label"],
            origin=origin,
        )
        for e in values_dict
    ]


def make_layout(
    cadences: bool = False,
    harmonies: Optional[DezrannLayer] = None,
    keys: Optional[DezrannLayer] = None,
    phrases: Optional[DezrannLayer] = None,
    raw: Optional[DezrannLayer] = None,
) -> List[dict]:
    """
    Compile the line positions for target labels into Dezrann layout parameter. Line arguments are translated
    via :func:`transform_line_argument`.
    """
    lines = resolve_line_arguments(
        harmonies=harmonies, keys=keys, phrases=phrases, raw=raw
    )
    layout = []
    if cadences:
        layout.append(
            {"filter": {"type": LABEL_TYPES["cadence"]}, "style": {"line": "all"}}
        )
    for arg, label_column in (
        ("harmonies", "chord"),
        ("keys", "localkey"),
        ("phrases", "phraseend"),
        ("raw", "label"),
    ):
        if lines[arg] is not None:
            layout.append(
                {
                    "filter": {"type": LABEL_TYPES[label_column]},
                    "style": {"line": lines[arg]},
                }
            )
    return layout


def generate_dez_from_dfs(
    measures_df: pd.DataFrame,
    harmonies_df: pd.DataFrame,
    output_path: str,
    cadences: bool = False,
    harmonies: Optional[DezrannLayer] = None,
    keys: Optional[DezrannLayer] = None,
    phrases: Optional[DezrannLayer] = None,
    raw: Optional[DezrannLayer] = None,
    origin: Union[str, Tuple[str, ...]] = "DCML",
) -> bool:
    """Create a .dez file from a measures and a labels/expanded dataframe.

    Args:
        measures_df: Measures table of the piece.
        harmonies_df: Label table (expanded or not) of the piece.
        output_path: Path to the .dez file to be (over)written.
        cadences: If True, labels in the 'cadence' column (if present) are included as vertical lines.
        harmonies: Specify a DezrannLayer to include the labels from the 'chord' column.
        keys: Specify a DezrannLayer to include the labels from the 'localkey' column.
        phrases: Specify a DezrannLayer to include the labels from the 'phraseend' column.
        raw: Specify a DezrannLayer to include the labels from the 'label' column.
        origin: Value to show in Dezrann's "Layer" field. Defaults to 'DCML'.

    Returns:
        True if a .dez file was written.

    Raises:
        ValueError: If the line arguments are invalid.
    """
    lines = resolve_line_arguments(
        harmonies=harmonies, keys=keys, phrases=phrases, raw=raw
    )
    if not cadences and all(line is None for line in lines.values()):
        logger.warning("Nothing to do because no features have been selected.")
        return False
    dezrann_labels = []
    features = []
    if cadences:
        features.append(("cadence", True))
    features.extend(
        (label_column, lines[arg] is not None)
        for arg, label_column in (
            ("harmonies", "chord"),
            ("keys", "localkey"),
            ("phrases", "phraseend"),
            ("raw", "label"),
        )
    )
    for label_column, selected in features:
        if not selected:
            continue
        if label_column not in harmonies_df.columns:
            logger.warning(
                f"{output_path}: The labels have no column {label_column!r}."
            )
            continue
        dcml_labels = dcml_labels2dicts(
            labels=harmonies_df, measures=measures_df, label_column=label_column
        )
        dezrann_labels += convert_dcml_list_to_dezrann_list(
            dcml_labels, label_type=LABEL_TYPES[label_column], origin=origin
        )
    if len(dezrann_labels) == 0:
        selected_features = [
            label_column for label_column, selected in features if selected
        ]
        hint = ""
        if lines["raw"] is None and "label" in harmonies_df.columns:
            hint = " To convert the labels as they are, select a line for raw labels."
        logger.warning(
            f"{output_path} not written because the labels contain none of {selected_features}.{hint}"
        )
        return False
    layout = make_layout(cadences=cadences, **lines)
    dezrann_content = DezrannDict(labels=dezrann_labels, meta={"layout": layout})
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dezrann_content, f, indent=2, allow_nan=False)
    logger.debug(f"Wrote {len(dezrann_labels)} labels to {output_path}.")
    return True


def load_tsv(path: str, **kwargs) -> pd.DataFrame:
    """Loads a TSV file as written by ms3, reading columns that may contain fractions as Fractions (None for empty
    cells), 'volta' as Int64, and all other columns without interpreting strings such as 'NA' as missing values.
    """
    header = pd.read_csv(path, sep="\t", nrows=0).columns
    converters = {col: safe_frac for col in FRACTION_COLUMNS if col in header}
    dtype = {
        col: "Int64" for col in ("mc", "mn", "volta", "staff", "voice") if col in header
    }
    return pd.read_csv(
        path,
        sep="\t",
        converters=converters,
        dtype=dtype,
        keep_default_na=False,
        na_values=[""],
        **kwargs,
    )


def generate_dez(
    path_measures: str,
    path_labels: str,
    output_path: str,
    cadences: bool = False,
    harmonies: Optional[DezrannLayer] = None,
    keys: Optional[DezrannLayer] = None,
    phrases: Optional[DezrannLayer] = None,
    raw: Optional[DezrannLayer] = None,
    origin: Union[str, Tuple[str, ...]] = "DCML",
) -> bool:
    """Create a .dez file from a path to a measures TSV file and a path to a labels/expanded TSV file.

    Args:
        path_measures: Path to a DCML measures TSV file.
        path_labels: Path to a DCML labels/expanded TSV file.
        output_path: Path to the .dez file to be (over)written.
        cadences: If True, labels in the 'cadence' column (if present) are included as vertical lines.
        harmonies: Specify a DezrannLayer to include the labels from the 'chord' column.
        keys: Specify a DezrannLayer to include the labels from the 'localkey' column.
        phrases: Specify a DezrannLayer to include the labels from the 'phraseend' column.
        raw: Specify a DezrannLayer to include the labels from the 'label' column.
        origin: Value to show in Dezrann's "Layer" field. Defaults to 'DCML'.

    Returns:
        True if a .dez file was written.

    Raises:
        ValueError: If reading from one of the specified file paths fails.
    """
    try:
        harmonies_df = load_tsv(path_labels)
    except (ValueError, AssertionError, FileNotFoundError) as e:
        raise ValueError(
            f"{path_labels} could not be loaded as a label table because of the following error:\n'{e}'"
        )
    try:
        measures_df = load_tsv(path_measures)
        if "mc" not in measures_df.columns:
            raise ValueError("The measures table has no column 'mc'.")
    except (ValueError, AssertionError, FileNotFoundError) as e:
        raise ValueError(
            f"{path_measures} could not be loaded as a measure map because of the following error:\n'{e}'"
        )

    return generate_dez_from_dfs(
        measures_df,
        harmonies_df,
        output_path,
        cadences=cadences,
        harmonies=harmonies,
        keys=keys,
        phrases=phrases,
        raw=raw,
        origin=origin,
    )


def split_tsv_name(file_name: str) -> Tuple[str, Optional[str]]:
    """Splits a TSV file name into piece name and facet, e.g. 'K283-2.harmonies.tsv' => ('K283-2', 'harmonies')
    and 'K283-2.tsv' => ('K283-2', None). Suffixes that are no known facet are considered part of the piece name.
    """
    stem = file_name[: -len(".tsv")] if file_name.endswith(".tsv") else file_name
    if "." in stem:
        piece, suffix = stem.rsplit(".", 1)
        if suffix in LABEL_FACETS + OTHER_FACETS:
            return piece, suffix
    return stem, None


def find_measures_file(measures_dir: str, piece: str) -> Optional[str]:
    """Returns the path of the measures TSV for the given piece, trying '<piece>.measures.tsv' and '<piece>.tsv'."""
    for file_name in (f"{piece}.measures.tsv", f"{piece}.tsv"):
        path = os.path.join(measures_dir, file_name)
        if os.path.isfile(path):
            return path
    return None


def find_label_files(input_dir: str) -> Dict[str, str]:
    """Returns {piece name -> path} for the label tables found in ``input_dir``. TSV files with a facet suffix other
    than those in :const:`LABEL_FACETS` are ignored. If several label tables exist for a piece, the facet coming first
    in :const:`LABEL_FACETS` (or the un-suffixed file) is preferred.
    """
    candidates: Dict[str, List[Tuple[int, str]]] = {}
    for file_name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, file_name)
        if (
            not file_name.endswith(".tsv")
            or file_name.startswith(".")
            or not os.path.isfile(path)
        ):
            continue
        piece, facet = split_tsv_name(file_name)
        if facet is None:
            priority = -1
        elif facet in LABEL_FACETS:
            priority = LABEL_FACETS.index(facet)
        else:
            logger.debug(f"Ignoring {file_name} which is not a label table.")
            continue
        candidates.setdefault(piece, []).append((priority, path))
    result = {}
    for piece, paths in candidates.items():
        paths.sort()
        result[piece] = paths[0][1]
        for _, path in paths[1:]:
            logger.info(
                f"Ignoring {path} because {result[piece]} is used for piece {piece!r}."
            )
    return result


def main(
    input_dir: str,
    measures_dir: str,
    output_dir: str,
    cadences: bool = False,
    harmonies: Optional[DezrannLayer] = None,
    keys: Optional[DezrannLayer] = None,
    phrases: Optional[DezrannLayer] = None,
    raw: Optional[DezrannLayer] = None,
    origin: Union[str, Tuple[str, ...]] = "DCML",
) -> ConversionSummary:
    """Main function for using this module as a script. It gathers file paths and converts the detected DCML-style
    labels/expanded TSV file to .dez format.

    Args:
        input_dir: Path containing DCML-style labels/expanded TSV files.
        measures_dir: Path containing the corresponding DCML-style measures TSV files.
        output_dir: Path at which the resulting .dez files are to be (over)written as <piece>.dez.
        cadences: If True, labels in the 'cadence' column (if present) are included as vertical lines.
        harmonies: Specify a DezrannLayer to include the labels from the 'chord' column.
        keys: Specify a DezrannLayer to include the labels from the 'localkey' column.
        phrases: Specify a DezrannLayer to include the labels from the 'phraseend' column.
        raw: Specify a DezrannLayer to include the labels from the 'label' column.
        origin: Value to show in Dezrann's "Layer" field. Defaults to 'DCML'.

    Returns:
        The number of .dez files written, of label files skipped, and of conversions that failed.
    """
    lines = resolve_line_arguments(
        harmonies=harmonies, keys=keys, phrases=phrases, raw=raw
    )
    if not cadences and all(line is None for line in lines.values()):
        logger.warning("Nothing to do because no features have been selected.")
        return ConversionSummary()
    label_files = find_label_files(input_dir)
    if len(label_files) == 0:
        logger.warning(f"No label TSV files found in {input_dir}.")
        return ConversionSummary()
    written, skipped, failed = 0, 0, 0
    for piece, input_file in label_files.items():
        measures_file = find_measures_file(measures_dir, piece)
        if measures_file is None or os.path.samefile(measures_file, input_file):
            logger.warning(
                f"No measures TSV found for {piece!r} in {measures_dir}. Skipping {input_file}."
            )
            skipped += 1
            continue
        output_file_path = os.path.join(output_dir, f"{piece}.dez")
        try:
            success = generate_dez(
                path_labels=input_file,
                path_measures=measures_file,
                output_path=output_file_path,
                cadences=cadences,
                origin=origin,
                **lines,
            )
        except Exception as e:
            logger.error(
                f"Converting {input_file} failed with {e.__class__.__name__}: {e}"
            )
            logger.debug("Traceback:", exc_info=True)
            failed += 1
            continue
        if success:
            logger.info(f"{output_file_path} successfully written.")
            written += 1
        else:
            skipped += 1
    summary = ConversionSummary(written=written, skipped=skipped, failed=failed)
    logger.info(
        f"Done. Created {written} .dez files ({skipped} skipped, {failed} failed)."
    )
    return summary


def resolve_dir(d):
    """Resolves '~' to HOME directory and turns ``d`` into an absolute path."""
    if d is None:
        return None
    d = str(d)
    if "~" in d:
        d = os.path.expanduser(d)
    return os.path.abspath(d)


def process_arguments(args: argparse.Namespace) -> dict:
    """Transforms the user's input arguments into keyword arguments for :func:`main` or raises a ValueError."""
    input_dir = resolve_dir(args.dir)
    if not os.path.isdir(input_dir):
        raise ValueError(f"{args.dir} is not an existing directory.")
    if args.measures is None:
        measures_dir = os.path.abspath(os.path.join(input_dir, "..", "measures"))
        if not os.path.isdir(measures_dir):
            if any(name.endswith(".measures.tsv") for name in os.listdir(input_dir)):
                measures_dir = input_dir
            else:
                raise ValueError(
                    f"No directory with measure maps was specified, the default path {measures_dir} does not "
                    f"exist, and {input_dir} does not contain any *.measures.tsv files."
                )
    else:
        measures_dir = resolve_dir(args.measures)
        if not os.path.isdir(measures_dir):
            raise ValueError(f"{measures_dir} is not an existing directory.")
    if args.out is None:
        output_dir = input_dir
    else:
        output_dir = resolve_dir(args.out)
        if not os.path.isdir(output_dir):
            raise ValueError(f"{output_dir} is not an existing directory.")
    kwargs = dict(input_dir=input_dir, measures_dir=measures_dir, output_dir=output_dir)
    lines = resolve_line_arguments(
        harmonies=args.harmonies, keys=args.keys, phrases=args.phrases, raw=args.raw
    )
    kwargs.update(lines)
    kwargs["cadences"] = args.cadences
    logger.debug(f"Arguments: {kwargs}")
    return kwargs


def line_argument(value: str) -> Optional[str]:
    """argparse type for line arguments."""
    try:
        return transform_line_argument(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


LINE_HELP = """\
LINE can be 1-3 for the lines above the system (top.1-top.3), 4-6 for the lines below (bot.1-bot.3),
-1, -2, -3 as aliases for 4, 5, 6, a Dezrann line name such as bot.1, or 0 to omit the labels."""


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f"""\
-----------------------------
| DCML => Dezrann converter |
-----------------------------

This script converts DCML harmony annotations into the .dez JSON format used by the dezrann.net app. It is
standalone and does not require ms3 to be installed. Its only requirement is pandas.

Apart from that, the script requires that you have previously extracted both harmonies and measures from the
annotated scores or that you are converting a DCML corpus (https://github.com/DCMLab/dcml_corpora),
where both facets are provided by default. In order to (re-) extract the labels, use the command:

    ms3 extract -X -M

Or, if you want to convert other harmony or chord labels from your MuseScore files, use -L for labels and
convert them using --raw. ms3 extract -h will show you all options.

Label files <piece>.harmonies.tsv (or .expanded.tsv, .labels.tsv, or un-suffixed <piece>.tsv) are matched with
<piece>.measures.tsv (or <piece>.tsv) and converted into <piece>.dez.

{LINE_HELP}
""",
    )
    parser.add_argument(
        "dir",
        metavar="IN_DIR",
        help="Folder that will be scanned for TSV files to convert. Sub-directories are not taken into account.",
    )
    parser.add_argument(
        "-m",
        "--measures",
        metavar="MEASURES_DIR",
        help="Folder in which to look for the corresponding measure maps. By default, the script will try "
        "to find a sibling to the source dir called 'measures' or, if there is none, use IN_DIR itself.",
    )
    parser.add_argument(
        "-o",
        "--out",
        metavar="OUT_DIR",
        help="Output directory for .dez files. Defaults to the input directory.",
    )
    parser.add_argument(
        "-C",
        "--cadences",
        action="store_true",
        help="Pass this flag if you want to add time-point cadence labels to the .dez files.",
    )
    parser.add_argument(
        "-H",
        "--harmonies",
        metavar="LINE",
        default="4",
        type=line_argument,
        help="Line for the harmony annotations ('chord' column). Default: 4 (bot.1), the first line under the "
        "system. Pass 0 to not add harmonies.",
    )
    parser.add_argument(
        "-K",
        "--keys",
        metavar="LINE",
        default="5",
        type=line_argument,
        help="Line for the local key segments ('localkey' column). Default: 5 (bot.2), the second line under the "
        "system. Pass 0 to not add key segments. Note, however, that harmonies are underdetermined without their "
        "local key.",
    )
    parser.add_argument(
        "-P",
        "--phrases",
        metavar="LINE",
        default="6",
        type=line_argument,
        help="Line for the phrase annotations ('phraseend' column). Default: 6 (bot.3), the third line under the "
        "system. Pass 0 to not add phrases.",
    )
    parser.add_argument(
        "--raw",
        metavar="LINE",
        type=line_argument,
        help="Pass this argument to add a line with the 'raw' labels ('label' column), i.e. including local key, "
        "cadence and phrase annotations. Not included by default. Needs to be different from the lines used "
        "for the other label types.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show debugging information.",
    )
    return parser


class _StderrHandler(logging.StreamHandler):
    """Writes to whatever sys.stderr currently is (instead of the stream at the time of instantiation)."""

    def __init__(self):
        super().__init__(sys.stderr)

    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, value):
        pass


def configure_logging(verbose: bool = False) -> None:
    """Makes the module's logger print to stderr, with level DEBUG if ``verbose`` else INFO. Leaves the root logger
    untouched."""
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not any(isinstance(handler, _StderrHandler) for handler in logger.handlers):
        handler = _StderrHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)


def run(argv: Optional[List[str]] = None) -> int:
    """Command line entry point. Returns 0 if no conversion failed, 1 otherwise."""
    parser = make_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    try:
        kwargs = process_arguments(args)
    except ValueError as e:
        parser.error(str(e))
    summary = main(**kwargs)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(run())
