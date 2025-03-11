import io
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import warnings
from ast import literal_eval
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from fractions import Fraction
from functools import cache, reduce
from inspect import getfullargspec, stack
from itertools import chain, repeat, takewhile
from pathlib import Path
from shutil import which
from tempfile import NamedTemporaryFile as Temp
from typing import (
    IO,
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)
from zipfile import ZipFile as Zip

import bs4
import git
import numpy as np
import pandas as pd
import webcolors
from gitdb.exc import BadName
from ms3._typing import Facet, FileDataframeTupleMaybe, FileDict, ViewDict
from ms3.logger import LogCapturer, get_logger, update_cfg
from numpy.typing import NDArray
from pandas._typing import Dtype
from pandas.errors import EmptyDataError
from pathos import multiprocessing
from pytablewriter import MarkdownTableWriter
from tqdm import tqdm
from typing_extensions import Self

from .constants import (
    BOOLEAN_COLUMNS,
    COLLECTION_COLUMNS,
    CSS_COLORS,
    DCML_REGEX,
    DEFAULT_CREATOR_METADATA,
    FORM_LEVEL_CAPTURE_REGEX,
    FORM_LEVEL_REGEX,
    FORM_LEVEL_SPLIT_REGEX,
    FORM_TOKEN_ABBREVIATIONS,
    KEYSIG_DICT_ENTRY_REGEX,
    LEGACY_COLUMNS,
    METADATA_COLUMN_ORDER,
    MS3_HTML,
    MS3_RGB,
    SCORE_EXTENSIONS,
    SLICE_INTERVAL_REGEX,
    STANDARD_COLUMN_ORDER,
    STANDARD_NAMES,
    STANDARD_NAMES_OR_GIT,
    TIMESIG_DICT_ENTRY_REGEX,
    rgba,
)

module_logger = get_logger(__name__)


class map_dict(dict):
    """Such a dictionary can be mapped to a Series to replace its values but leaving the values absent from the dict
    keys intact."""

    def __missing__(self, key):
        return key


def assert_all_lines_equal(before, after, original, tmp_file):
    """Compares two multiline strings to test equality."""
    diff = [
        (i, bef, aft)
        for i, (bef, aft) in enumerate(zip(before.splitlines(), after.splitlines()), 1)
        if bef != aft
    ]
    if len(diff) > 0:
        line_n, left, _ = zip(*diff)
        ln = len(str(max(line_n)))  # length of the longest line number
        left_col = max(len(s) for s in left)  # length of the longest line
        folder, file = os.path.split(original)
        tmp_persist = os.path.join(folder, "..", file)
        shutil.copy(tmp_file.name, tmp_persist)
        diff = [("", original, tmp_persist)] + diff
    assert len(diff) == 0, "\n" + "\n".join(
        f"{a:{ln}}  {b:{left_col}}    {c}" for a, b, c in diff
    )


def assert_dfs_equal(old, new, exclude=[]):
    """Compares the common columns of two DataFrames to test equality. Uses: nan_eq()"""
    old_l, new_l = len(old), len(new)
    greater_length = max(old_l, new_l)
    if old_l != new_l:
        print(f"Old length: {old_l}, new length: {new_l}")
        old_is_shorter = new_l == greater_length
        shorter = old if old_is_shorter else new
        missing_rows = abs(old_l - new_l)
        shorter_cols = shorter.columns
        patch = pd.DataFrame(
            [["missing row"] * len(shorter_cols)] * missing_rows, columns=shorter_cols
        )
        shorter = pd.concat([shorter, patch], ignore_index=True)
        if old_is_shorter:
            old = shorter
        else:
            new = shorter
    # old.index.rename('old_ix', inplace=True)
    # new.index.rename('new_ix', inplace=True)
    cols = [
        col
        for col in set(old.columns).intersection(set(new.columns))
        if col not in exclude
    ]
    diff = [
        (i, j, ~nan_eq(o, n))
        for ((i, o), (j, n)) in zip(old[cols].iterrows(), new[cols].iterrows())
    ]
    old_bool = pd.DataFrame.from_dict(
        {ix: bool_series for ix, _, bool_series in diff}, orient="index"
    )
    new_bool = pd.DataFrame.from_dict(
        {ix: bool_series for _, ix, bool_series in diff}, orient="index"
    )
    diffs_per_col = old_bool.sum(axis=0)

    def show_diff():
        comp_str = []
        if "mc" in old.columns:
            position_col = "mc"
        elif "last_mc" in old.columns:
            position_col = "last_mc"
        else:
            position_col = None
        for col, n_diffs in diffs_per_col.items():
            if n_diffs > 0:
                if position_col is None:
                    columns = col
                else:
                    columns = [position_col, col]
                comparison = pd.concat(
                    [
                        old.loc[old_bool[col], columns]
                        .reset_index(drop=True)
                        .iloc[:20],
                        new.loc[new_bool[col], columns]
                        .iloc[:20]
                        .reset_index(drop=True),
                    ],
                    axis=1,
                    keys=["old", "new"],
                )
                comp_str.append(
                    f"{n_diffs}/{greater_length} ({n_diffs / greater_length * 100:.2f} %) rows are different for "
                    f"{col}{' (showing first 20)' if n_diffs > 20 else ''}:\n{comparison}\n"
                )
        return "\n".join(comp_str)

    assert diffs_per_col.sum() == 0, show_diff()


def ambitus2oneliner(ambitus):
    """Turns a ``metadata['parts'][staff_id]`` dictionary into a string."""
    if "min_midi" in ambitus:
        return f"{ambitus['min_midi']}-{ambitus['max_midi']} ({ambitus['min_name']}-{ambitus['max_name']})"
    if "max_midi" in ambitus:
        return f"{ambitus['max_midi']}-{ambitus['max_midi']} ({ambitus['max_name']}-{ambitus['max_name']})"
    return ""


def changes2list(changes, sort=True):
    """Splits a string of changes into a list of 4-tuples.

    Example
    -------
    >>> changes2list('+#7b5')
    [('+#7', '+', '#', '7'),
     ('b5',  '',  'b', '5')]
    """
    res = [t for t in re.findall(r"((\+|-|\^|v)?(#+|b+)?(1\d|\d))", changes)]
    return sorted(res, key=lambda x: int(x[3]), reverse=True) if sort else res


def changes2tpc(changes, numeral, minor=False, root_alterations=False, logger=None):
    """
    Given a numeral and changes, computes the intervals that the changes represent.
    Changes do not express absolute intervals but instead depend on the numeral and the mode.

    Uses: split_scale_degree(), changes2list()

    Parameters
    ----------
    changes : :obj:`str`
        A string of changes following the DCML harmony standard.
    numeral : :obj:`str`
        Roman numeral. If it is preceded by accidentals, it depends on the parameter
        `root_alterations` whether these are taken into account.
    minor : :obj:`bool`, optional
        Set to true if the `numeral` occurs in a minor context.
    root_alterations : :obj:`bool`, optional
        Set to True if accidentals of the root should change the result.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    root_alteration, num_degree = split_scale_degree(numeral, count=True, logger=logger)
    # build 2-octave diatonic scale on C major/minor
    root = ["I", "II", "III", "IV", "V", "VI", "VII"].index(num_degree.upper())
    tpcs = (
        2 * [i for i in (0, 2, -3, -1, 1, -4, -2)]
        if minor
        else 2 * [i for i in (0, 2, 4, -1, 1, 3, 5)]
    )
    tpcs = tpcs[root:] + tpcs[:root]  # starting the scale from chord root
    root = tpcs[0]
    if root_alterations:
        root += 7 * root_alteration
        tpcs[0] = root

    alts = changes2list(changes, sort=False)

    def acc2tpc(accidentals):
        return 7 * (accidentals.count("#") - accidentals.count("b"))

    return [
        (
            full,
            added,
            acc,
            chord_interval,
            (
                (tpcs[int(chord_interval) - 1] + acc2tpc(acc) - root)
                if chord_interval not in ["3", "5"]
                else None
            ),
        )
        for full, added, acc, chord_interval in alts
    ]


def check_labels(
    df,
    regex,
    column="label",
    split_regex=None,
    return_cols=["mc", "mc_onset", "staff", "voice"],
):
    """Checks the labels in ``column`` against ``regex`` and returns those that don't match.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        DataFrame containing a column with labels.
    regex : :obj:`str`
        Regular expression that incorrect labels don't match.
    column : :obj:`str`, optional
        Column name where the labels are. Defaults to 'label'
    split_regex : :obj:`str`, optional
        If you pass a regular expression (or simple string), it will be used to split the labels before checking the
        resulting column separately. Instead, pass True to use the default (a '-' that does not precede a scale degree).
    return_cols : :obj:`list`, optional
        Pass a list of the DataFrame columns that you want to be displayed for the wrong labels.

    Returns
    -------
    df : :obj:`pandas.DataFrame`
        DataFrame with wrong labels.

    """
    if split_regex is not None:
        if split_regex:
            check_this = split_alternatives(df, column=column, alternatives_only=True)
        else:
            check_this = split_alternatives(
                df, column=column, regex=split_regex, max=1000, alternatives_only=True
            )
    else:
        check_this = df[[column]]
    if regex.__class__ != re.compile("").__class__:
        regex = re.compile(regex, re.VERBOSE)
    not_matched = check_this.apply(lambda c: ~c.str.match(regex).fillna(True))
    cols = [c for c in return_cols if c in df.columns]
    select_wrong = not_matched.any(axis=1)
    res = check_this.where(not_matched, other=".")[select_wrong]
    res = res.apply(lambda c: c.str.replace("^/$", "empty_harmony", regex=True))
    return pd.concat([df.loc[select_wrong, cols], res], axis=1)


def check_phrase_annotations(df: pd.DataFrame, column: str, logger=None) -> bool:
    """"""
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    p_col = df[column]
    opening = p_col.fillna("").str.count("{")
    closing = p_col.fillna("").str.count("}")
    if "mn_playthrough" in df.columns:
        position_col = "mn_playthrough"
    else:
        logger.info(
            "Column 'mn_playthrough' is missing, so my assessment of the phrase annotations might be wrong."
        )
        position_col = "mn"
    columns = [position_col, column]
    if opening.sum() != closing.sum():
        o = df.loc[(opening > 0), columns]
        c = df.loc[(closing > 0), columns]
        compare = pd.concat(
            [o.reset_index(drop=True), c.reset_index(drop=True)], axis=1
        )
        if "mn" in compare:
            compare = compare.astype({"mn": "Int64"})
        logger.warning(
            f"Phrase beginning and endings don't match:\n{compare.to_string(index=False)}",
            extra={"message_id": (16,)},
        )
        return False
    return True


def color2rgba(c):
    """Pass a RGB or RGBA tuple, HTML color or name to convert it to RGBA"""
    if isinstance(c, tuple):
        if len(c) > 3:
            return rgba(*c[:4])
        if len(c) == 3:
            return rgba(*(c + (255,)))
        else:
            return rgba(*c)
    if c[0] == "#":
        return html_color2rgba(c)
    return color_name2rgba(c)


def color_name2format(n, format="rgb"):
    """Converts a single CSS3 name into one of 'HTML', 'rgb', or 'rgba'"""
    if pd.isnull(n):
        return n
    if n in CSS_COLORS:
        html = webcolors.name_to_hex(n)
    elif n in MS3_HTML.values():
        html = next(k for k, v in MS3_HTML.items() if v == n)
    elif n[0] == "#":
        html = n
    else:
        return n
    if format == "html":
        return html
    if format == "rgb":
        return webcolors.hex_to_rgb(html)
    if format == "rgba":
        rgb = webcolors.hex_to_rgb(html)
        return rgba(*(rgb + (255,)))
    return html


def color_name2html(n):
    """Converts a single CSS3 name into HTML"""
    return color_name2format(n, format="html")


def color_name2rgb(n):
    """Converts a single CSS3 name into RGB"""
    return color_name2format(n, format="rgb")


def color_name2rgba(n):
    """Converts a single CSS3 name into RGBA"""
    return color_name2format(n, format="rgba")


def color_params2rgba(
    color_name=None,
    color_html=None,
    color_r=None,
    color_g=None,
    color_b=None,
    color_a=None,
    logger=None,
):
    """For functions where the color can be specified in four different ways (HTML string, CSS name,
    RGB, or RGBA), convert the given parameters to RGBA.

    Parameters
    ----------
    color_name : :obj:`str`, optional
        As a name you can use CSS colors or MuseScore colors (see :py:attr:`.MS3_COLORS`).
    color_html : :obj:`str`, optional
        An HTML color needs to be string of length 6.
    color_r : :obj:`int`, optional
        If you specify the color as RGB(A), you also need to specify color_g and color_b.
    color_g : :obj:`int`, optional
        If you specify the color as RGB(A), you also need to specify color_r and color_b.
    color_b : :obj:`int`, optional
        If you specify the color as RGB(A), you also need to specify color_r and color_g.
    color_a : :obj:`int`, optional
        If you have specified an RGB color, the alpha value defaults to 255 unless specified otherwise.

    Returns
    -------
    :obj:`rgba`
        :obj:`namedtuple` with four integers.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if all(
        pd.isnull(param)
        for param in [color_name, color_html, color_r, color_g, color_b, color_a]
    ):
        logger.debug("None of the parameters have been specified. Returning None.")
        return None
    res = None
    if not pd.isnull(color_r):
        if pd.isnull(color_a):
            color_a = 255
        if pd.isnull(color_g) or pd.isnull(color_b):
            if pd.isnull(color_name) and pd.isnull(color_html):
                logger.warning(f"Not a valid RGB color: {(color_r, color_g, color_b)}")
        else:
            res = (color_r, color_g, color_b, color_a)
    if res is None and not pd.isnull(color_html):
        res = color2rgba(color_html)
    if res is None and not pd.isnull(color_name):
        res = color2rgba(color_name)
    return rgba(*res)


def allnamesequal(name):
    return all(n == name[0] for n in name[1:])


def commonprefix(paths, sep="/"):
    """Returns common prefix of a list of paths.
    Uses: allnamesequal(), itertools.takewhile()"""
    bydirectorylevels = zip(*[p.split(sep) for p in paths])
    return sep.join(x[0] for x in takewhile(allnamesequal, bydirectorylevels))


def compute_mn(measures: pd.DataFrame) -> pd.Series:
    """Compute measure number integers from a measures table.


    Args:
      measures: Measures table with columns ['mc', 'dont_count', 'numbering_offset'].

    Returns:

    """
    excluded = measures["dont_count"].fillna(0).astype(bool)
    offset = measures["numbering_offset"]
    mn = (~excluded).cumsum()
    if offset.notna().any():
        offset = offset.fillna(0).astype(int).cumsum()
        mn += offset
    return mn.rename("mn")


def compute_mn_playthrough(measures: pd.DataFrame, logger=None) -> pd.Series:
    """Compute measure number strings from an unfolded measures table, such that the first occurrence of a measure
    number ends on 'a', the second one on 'b' etc.

    The function requires the column 'dont_count' in order to correctly number the return of a completing MC after an
    incomplete MC with "endrepeat" sign. For example, if a repeated section begins with an upbeat that at first
    completes MN 16 it will have mn_playthrough '16a' the first time and '32a' the second time (assuming it completes
    the incomplete MN 32).

    Args:
      measures: Measures table with columns ['mc', 'mn', 'dont_count']

    Returns:
      'mn_playthrough' Series of disambiguated measure number strings. If no measure repeats, the result will be
      equivalent to converting column 'mn' to strings and appending 'a' to all of them.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    previous_occurrences = defaultdict(lambda: 0)

    def get_mn_playthrough(mn):
        repeat_char = chr(previous_occurrences[mn] + 96)
        return f"{mn}{repeat_char}"

    mn_playthrough_column = []
    previous_mc, previous_mn = 0, -1
    for mc, mn, dont_count in measures[["mc", "mn", "dont_count"]].itertuples(
        name=None, index=False
    ):
        if mn != previous_mn:
            previous_occurrences[mn] += 1
        if mc < previous_mc and not pd.isnull(dont_count):
            # an earlier MC completes a later one after a repeat
            mn_playthrough = get_mn_playthrough(previous_mn)
            logger.debug(
                f"MN {mn} < previous MN {previous_mn}; but since MC {mc} is excluded from barcount, "
                f"the repetition has mn_playthrough {mn_playthrough}."
            )
        else:
            mn_playthrough = get_mn_playthrough(mn)
        mn_playthrough_column.append(mn_playthrough)
        previous_mn = mn
        previous_mc = mc
    return pd.Series(mn_playthrough_column, index=measures.index, name="mn_playthrough")


@cache
def get_major_version_of_musescore_executable(ms) -> Optional[int]:
    try:
        result = subprocess.run([ms, "-v"], capture_output=True, text=True)
        version = result.stdout
        if version.startswith("MuseScore"):
            return int(version[9])
        return
    except Exception:
        return


def convert(
    old,
    new,
    MS="mscore",
    logger=None,
):
    """Calls "MS -fo new old", which converts old to new with the given MuseScore executable."""
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    version = get_major_version_of_musescore_executable(MS)
    if version == 4:
        convert_to_ms4(old=old, new=new, MS=MS, logger=logger)
        return
    process = [
        MS,
        "-fo",
        new,
        old,
    ]  # [MS, '--appimage-extract-and-run', "-fo", new, old] if MS.endswith('.AppImage') else [MS,
    # "-fo", new, old]
    if subprocess.run(process, capture_output=True, text=True):
        logger.info(f"Converted {old} to {new}")
    else:
        logger.warning("Error while converting " + old)


def convert_to_ms4(old, new, MS="mscore", logger=None):
    """
    Calls "MS -fo new old", which converts old to new with the given MuseScore executable. This function offers a
    workaround for MuseScore 4's half-baked commandline conversion.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    version = get_major_version_of_musescore_executable(MS)
    if not (version is None or version == 4):
        if version is None:
            msg = f"{MS} is not MuseScore 4.x.x."
        else:
            msg = f"{MS} is MuseScore {version}, not 4. Try convert()."
        raise RuntimeError(msg)
    old_path, old_file = os.path.split(old)
    old_piece_name, old_ext = os.path.splitext(old_file)
    if old_ext.lower() not in SCORE_EXTENSIONS:
        logger.warning(
            f"Source file '{old}' doesn't have one of the following extensions and is skipped: {SCORE_EXTENSIONS}."
        )
        return
    if os.path.isdir(new):
        target_folder = new
        target_file = old_file
    else:
        target_folder, target_file = os.path.split(new)
    target_piece_name, target_ext = os.path.splitext(target_file)
    tmp_needed = target_ext in (".mscx", ".mscz")
    if tmp_needed:
        tmp_dir = os.path.join(target_folder, ".ms3_tmp_" + old_file)
        try:
            os.makedirs(tmp_dir, exist_ok=False)
        except FileExistsError:
            logger.warning(
                f"The temporary directory {tmp_dir} exists already and would be overwritten. Delete it first."
            )
            return
        tmp_mscx = os.path.join(tmp_dir, target_piece_name + ".mscx")
        new = tmp_mscx
    process = [MS, "-fo", new, old]
    result = subprocess.run(process, capture_output=True, text=True)
    if result.returncode == 0:
        if tmp_needed:
            target_path_without_extension = os.path.join(
                target_folder, target_piece_name
            )
            if not os.path.isdir(tmp_dir):
                logger.warning(
                    f"Temporary directory '{tmp_dir}' did not exist after conversion."
                )
                return
            if not os.path.isfile(tmp_mscx):
                logger.warning(
                    f"Conversion of '{old}' did not yield a complete uncompressed MuseScore folder, an .mscx file was "
                    f"missing."
                )
                shutil.rmtree(tmp_dir)
                return
            if target_ext == ".mscz":
                zip_file_path = shutil.make_archive(
                    target_path_without_extension, "zip", tmp_dir
                )
                new = target_path_without_extension + ".mscz"
                shutil.move(zip_file_path, new)
            else:
                new = target_path_without_extension + ".mscx"
                shutil.move(tmp_mscx, new)
            shutil.rmtree(tmp_dir)
        logger.info(f"Successfully converted {old} => {new}")
    else:
        logger.error(f"Conversion to MuseScore 4 failed:\n{result.stderr}")
        if tmp_needed and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)


def _convert_kwargs(kwargs):
    """Auxiliary function allowing to use Pool.starmap() with keyword arguments (needed in order to
    pass the logger argument which is not part of the signature of convert() )."""
    return convert(**kwargs)


def convert_folder(
    directory=None,
    file_paths=None,
    target_dir=None,
    extensions=[],
    target_extension="mscx",
    regex=".*",
    suffix=None,
    recursive=True,
    ms="mscore",
    overwrite=False,
    parallel=False,
    logger=None,
):
    """Convert all files in `dir` that have one of the `extensions` to .mscx format using the executable `MS`.

    Parameters
    ----------
    directory : :obj:`str`
        Directory in which to look for files to convert.
    file_paths : :obj:`list` of `dir`
        List of file paths to convert. These are not filtered by any means.
    target_dir : :obj:`str`
        Directory where to store converted files. Defaults to ``directory``
    extensions : list, optional
        If you want to convert only certain formats, give those, e.g. ['mscz', 'xml']
    recursive : bool, optional
        Subdirectories as well.
    MS : str, optional
        Give the path to the MuseScore executable on your system. Need only if
        the command 'mscore' does not execute MuseScore on your system.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    MS = get_musescore(ms, logger=logger)
    assert MS is not None, f"MuseScore not found: {ms}"
    assert any(
        arg is not None for arg in (directory, file_paths)
    ), "Pass at least a directory or one path."
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    if target_extension[0] == ".":
        target_extension = target_extension[1:]
    conversion_params = []
    # logger.info(f"Traversing {dir} {'' if recursive else 'non-'}recursively...")
    if len(extensions) > 0:
        exclude_re = f"^(?:(?!({'|'.join(extensions)})).)*$"
    else:
        exclude_re = ""
    if target_dir is None:
        target_dir = directory
    resolved_target_dir = resolve_dir(target_dir)
    new_dirs = {}
    subdir_file_tuples = iter([])
    if directory is not None:
        subdir_file_tuples = chain(
            subdir_file_tuples,
            scan_directory(
                directory,
                file_re=regex,
                exclude_re=exclude_re,
                recursive=recursive,
                subdirs=True,
                exclude_files_only=True,
                logger=logger,
            ),
        )
    if file_paths is not None:
        subdir_file_tuples = chain(
            subdir_file_tuples,
            (os.path.split(resolve_dir(path)) for path in file_paths),
        )
    for subdir, file in subdir_file_tuples:
        if subdir in new_dirs:
            new_subdir = new_dirs[subdir]
        else:
            if os.path.isabs(target_dir):
                old_subdir = os.path.relpath(subdir, directory)
                new_subdir = (
                    os.path.join(target_dir, old_subdir)
                    if old_subdir != "."
                    else target_dir
                )
            else:
                new_subdir = resolved_target_dir
            os.makedirs(new_subdir, exist_ok=True)
            new_dirs[subdir] = new_subdir
        name, _ = os.path.splitext(file)
        if suffix is not None:
            piece_name = f"{name}{suffix}.{target_extension}"
        else:
            piece_name = f"{name}.{target_extension}"
        old = os.path.join(subdir, file)
        new = os.path.join(new_subdir, piece_name)
        if overwrite or not os.path.isfile(new):
            conversion_params.append([dict(old=old, new=new, MS=MS, logger=logger)])
        else:
            logger.debug(new, "exists already. Pass -o to overwrite.")

    if len(conversion_params) == 0:
        logger.info("No files to convert.")

    if parallel:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        pool.starmap(_convert_kwargs, conversion_params)
        pool.close()
        pool.join()
    else:
        for old, new, MS in conversion_params:
            convert(old=old, new=new, MS=MS, logger=logger)


def convert_from_metadata_tsv(
    directory=None,
    file_paths=None,
    target_dir=None,
    extensions=[],
    target_extension="mscx",
    regex=".*",
    suffix=None,
    recursive=True,
    ms="mscore",
    overwrite=False,
    parallel=False,
    logger=None,
):
    if file_paths is None:
        file_paths = []
    else:
        file_paths = list(file_paths)
    tsv_files = set(f for f in os.listdir(directory) if f.endswith(".tsv"))
    if "metadata.tsv" in tsv_files:
        metadata_fname = "metadata.tsv"
    else:
        for f in tsv_files:
            if f.endswith(".metadata.tsv"):
                metadata_fname = f
                break
        else:
            raise FileNotFoundError(
                f"No metadata TSV found in {directory}. Use --all/-a to detect files."
            )
    metadata_path = os.path.join(directory, metadata_fname)
    metadata = load_tsv(metadata_path)
    file_paths += [os.path.join(directory, rel_path) for rel_path in metadata.rel_path]
    convert_folder(
        directory=None,
        file_paths=file_paths,
        target_dir=target_dir,
        extensions=extensions,
        target_extension=target_extension,
        regex=regex,
        suffix=suffix,
        recursive=recursive,
        ms=ms,
        overwrite=overwrite,
        parallel=parallel,
        logger=logger,
    )


def decode_harmonies(
    df,
    label_col="label",
    keep_layer=True,
    return_series=False,
    alt_cols="alt_label",
    alt_separator="-",
    logger=None,
):
    """MuseScore stores types 2 (Nashville) and 3 (absolute chords) in several columns. This function returns a copy of
    the DataFrame ``Annotations.df`` where the label column contains the strings corresponding to these columns.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        DataFrame with encoded harmony labels as stored in an :obj:`Annotations` object.
    label_col : :obj:`str`, optional
        Column name where the main components (<name> tag) are stored, defaults to 'label'
    keep_layer : :obj:`bool`, optional
        Defaults to True, retaining the 'harmony_layer' column with original layers.
    return_series : :obj:`bool`, optional
        If set to True, only the decoded labels column is returned as a Series rather than a copy of ``df``.
    alt_cols : :obj:`str` or :obj:`list`, optional
        Column(s) with alternative labels that are joined with the label columns using ``alt_separator``. Defaults to
        'alt_label'. Suppress by passing None.
    alt_separator: :obj:`str`, optional
        Separator for joining ``alt_cols``.

    Returns
    -------
    :obj:`pandas.DataFrame` or :obj:`pandas.Series`
        Decoded harmony labels.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    df = df.copy()
    drop_cols, compose_label = [], []
    if "nashville" in df.columns:
        sel = df.nashville.notna()
        df.loc[sel, label_col] = df.loc[sel, "nashville"].astype(str) + df.loc[
            sel, label_col
        ].replace("/", "")
        drop_cols.append("nashville")
    if "leftParen" in df.columns:
        df.leftParen.replace("/", "(", inplace=True)
        compose_label.append("leftParen")
        drop_cols.append("leftParen")
    if "absolute_root" in df.columns and df.absolute_root.notna().any():
        sel = df.absolute_root.notna()
        root_as_note_name = fifths2name(
            df.loc[sel, "absolute_root"].to_list(), ms=True, logger=logger
        )
        df.absolute_root = df.absolute_root.astype("string")
        df.loc[sel, "absolute_root"] = root_as_note_name
        compose_label.append("absolute_root")
        drop_cols.append("absolute_root")
        if "rootCase" in df.columns:
            sel = df.rootCase.notna()
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
                df.loc[sel, "absolute_root"] = df.loc[sel, "absolute_root"].str.lower()
            drop_cols.append("rootCase")
    if label_col in df.columns:
        compose_label.append(label_col)
    if "absolute_base" in df.columns and df.absolute_base.notna().any():
        sel = df.absolute_base.notna()
        base_as_note_name = fifths2name(
            df.loc[sel, "absolute_base"].to_list(), ms=True, logger=logger
        )
        df.absolute_base = df.absolute_base.astype("string")
        df.loc[sel, "absolute_base"] = base_as_note_name
        df.absolute_base = "/" + df.absolute_base
        compose_label.append("absolute_base")
        drop_cols.append("absolute_base")
    if "rightParen" in df.columns:
        df.rightParen.replace("/", ")", inplace=True)
        compose_label.append("rightParen")
        drop_cols.append("rightParen")
    new_label_col = df[compose_label].fillna("").sum(axis=1).astype(str)
    new_label_col = new_label_col.str.replace(
        "^/$", "empty_harmony", regex=True
    ).replace("", pd.NA)

    if alt_cols is not None:
        if isinstance(alt_cols, str):
            alt_cols = [alt_cols]
        present = [c for c in alt_cols if c in df.columns]
        if len(present) > 0:
            alt_joined = pd.Series("", index=new_label_col.index)
            for c in present:
                alt_joined += (alt_separator + df[c]).fillna("")
            new_label_col += alt_joined

    if return_series:
        return new_label_col

    if "harmony_layer" in df.columns and not keep_layer:
        drop_cols.append("harmony_layer")
    df[label_col] = new_label_col
    df.drop(columns=drop_cols, inplace=True)
    return df


def df2md(df: pd.DataFrame, name: str = "Overview") -> MarkdownTableWriter:
    """Alias for :func:`dataframe2markdown`."""
    return dataframe2markdown(df, name=name)


def dataframe2markdown(
    df: pd.DataFrame, name: Optional[str] = None
) -> MarkdownTableWriter:
    """Turns a DataFrame into a MarkDown table. The returned writer can be converted into a string."""
    writer = MarkdownTableWriter()
    writer.table_name = name
    writer.from_dataframe(df)
    return writer


def dict2oneliner(d: dict) -> str:
    """Turns a dictionary into a single-line string without brackets."""
    if pd.isnull(d) or isinstance(d, str):
        return d
    return ", ".join(f"{k}: {v}" for k, v in d.items())


def resolve_form_abbreviations(
    token: str,
    abbreviations: dict,
    mc: Optional[Union[int, str]] = None,
    fallback_to_lowercase: bool = True,
    logger=None,
) -> str:
    """Checks for each consecutive substring of the token if it matches one of the given abbreviations and replaces
    it with the corresponding long name. Trailing numbers are separated by a space in this case.

    Args:
      token: Individual token after splitting alternative readings.
      abbreviations: {abbreviation -> long name} dict for string replacement.
      fallback_to_lowercase: By default, the substrings are checked against the dictionary keys and, if unsuccessful,
          again in lowercase. Pass False to use only the original string.

    Returns:

    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    mc_string = "" if mc is None else f"MC {mc}: "
    if "," in token:
        logger.warning(
            f"{mc_string}'{token}' contains a comma, which might result from a syntax error."
        )
    sub_component_regex = r"\W+"
    ends_on_numbers_regex = r"\d+$"
    resolved_substrings = []
    for original_substring in re.split(sub_component_regex, token):
        trailing_numbers_match = re.search(ends_on_numbers_regex, original_substring)
        if trailing_numbers_match is None:
            trailing_numbers = ""
            substring = original_substring
        else:
            trailing_numbers_position = trailing_numbers_match.start()
            if trailing_numbers_position == 0:
                # token is just a number
                resolved_substrings.append(original_substring)
                continue
            trailing_numbers = " " + trailing_numbers_match.group()
            substring = original_substring[:trailing_numbers_position]
        lowercase = substring.lower()
        check_lower = fallback_to_lowercase and (lowercase != substring)
        if substring in abbreviations:
            resolved = abbreviations[substring] + trailing_numbers
        elif check_lower and lowercase in abbreviations:
            resolved = abbreviations[lowercase] + trailing_numbers
        else:
            resolved = original_substring
        resolved_substrings.append(resolved)
    return "".join(
        substring + separator
        for substring, separator in zip(
            resolved_substrings, re.findall(sub_component_regex, token) + [""]
        )
    )


def distribute_tokens_over_levels(
    levels: Collection[str],
    tokens: Collection[str],
    mc: Optional[Union[int, str]] = None,
    abbreviations: dict = {},
    logger=None,
) -> Dict[Tuple[str, str], str]:
    """Takes the regex matches of one label and turns them into as many {layer -> token} pairs as the label contains
    tokens.

    Args:
      levels: Collection of strings indicating analytical layers.
      tokens: Collection of tokens coming along, same size as levels.
      mc: Pass the label's label's MC to display it in error messages.
      abbreviations: {abbrevation -> long name} mapping abbreviations to what they are to be replaced with

    Returns:
      A {(form_tree, level) -> token} dict where form_tree is either '' or a letter between a-h identifying one of
      several trees annotated in parallel.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    column2value = {}
    reading_regex = r"^((?:[ivx]+\&?)+):"
    mc_string = "" if mc is None else f"MC {mc}: "
    for level_str, token_str in zip(levels, tokens):
        split_level_info = pd.Series(level_str.split("&"))
        # turn layer indications into a DataFrame with the columns ['level', 'form_tree', and 'reading']
        analytical_layers = split_level_info.str.extract(FORM_LEVEL_REGEX)
        levels_include_reading = analytical_layers.reading.notna().any()
        # reading indications become part of the token and each will be followed by a colon
        analytical_layers.reading = (analytical_layers.reading + ": ").fillna("")
        # propagate information that has been omitted in the second and following indications,
        # e.g. 2a&b -> [2a:, 2b:]; 1aii&iii -> [1aii:, 1aiii:]; 1ai&b -> [1ai:, 1b] (i.e., readings are not propagated)
        analytical_layers = analytical_layers.ffill()
        analytical_layers.form_tree = analytical_layers.form_tree.fillna("")
        # split token into alternative components, replace special with normal white-space characters, and strip each
        # component from white space and separating commas
        token_alternatives = [
            re.sub(r"\s+", " ", t).strip(" \n,") for t in token_str.split(" - ")
        ]
        token_alternatives = [t for t in token_alternatives if t != ""]
        if len(abbreviations) > 0:
            token_alternatives = [
                resolve_form_abbreviations(token, abbreviations, mc=mc, logger=logger)
                for token in token_alternatives
            ]
        if len(token_alternatives) == 1:
            if levels_include_reading:
                analytical_layers.reading += token_alternatives[0]
                label = ""
            else:
                label = token_alternatives[0]
        else:
            # this section deals with cases where alternative readings are or are not identified by Roman numbers,
            # and deals with the given Roman numbers depending on whether the analytical layers include some as well
            token_includes_reading = any(
                re.match(reading_regex, t) is not None for t in token_alternatives
            )
            if token_includes_reading:
                reading_info = [re.match(reading_regex, t) for t in token_alternatives]
                reading2token = {}
                for readings_str, token_component in zip(
                    reading_info[1:], token_alternatives[1:]
                ):
                    if readings_str is None:
                        reading = ""
                        reading2token[reading] = token_component
                    else:
                        match_end = readings_str.end()
                        token_component = token_component[match_end:].strip(" ")
                        for roman in readings_str.group(1).split("&"):
                            reading = f"{roman}: "
                            if (
                                levels_include_reading
                                and reading in analytical_layers.reading.values
                            ):
                                column_empty = (analytical_layers == "").all()
                                show_layers = analytical_layers.loc[:, ~column_empty]
                                logger.warning(
                                    f"{mc_string}Alternative reading in '{token_str}' specifies Roman '{reading}' "
                                    f"which conflicts with one specified in the level:\n{show_layers}"
                                )
                            reading2token[reading] = token_component
                label = " - ".join(
                    reading + tkn for reading, tkn in reading2token.items()
                )
                if levels_include_reading:
                    if reading_info[0] is not None:
                        logger.warning(
                            f"{mc_string}'{token_str}': The first reading '{reading_info[0].group()}' specifies Roman "
                            f"number in addition to those specified in the level:\n{analytical_layers}"
                        )
                    analytical_layers.reading += token_alternatives[0]
                else:
                    label = token_alternatives[0] + " - " + label
            else:
                # token does not include any Roman numbers and is used as is for all layers
                analytical_layers.reading += " - ".join(t for t in token_alternatives)
                label = ""

        for (form_tree, level), df in analytical_layers.groupby(
            ["form_tree", "level"], dropna=False
        ):
            key = (
                form_tree,
                level,
            )  # form_tree becomes first level of the columns' MultiIndex, e.g. 'a' and 'b'
            if (df.reading == "").all():
                value = label
                if len(df) > 1:
                    logger.warning(
                        f"{mc_string}Duplication of level without specifying separate readings:\n{df}"
                    )
            else:
                value = " - ".join(reading for reading in df.reading.to_list())
                if label != "":
                    value += " - " + label
            if key in column2value:
                column2value[key] += " - " + value
                # logger.warning(f"{mc_string}The token '{column2value[key]}' for level {key} was overwritten with '{
                # value}':\nlevels: {level_str}, token: {token}")
            else:
                column2value[key] = value
    return column2value


def expand_single_form_label(
    label: str, default_abbreviations=True, **kwargs
) -> Dict[Tuple[str, str], str]:
    """Splits a form label and applies distribute_tokens_over_levels()

    Args:
      label: Complete form label including indications of analytical layer(s).
      default_abbreviations: By default, each token component is checked against a mapping from abbreviations to
          long names. Pass False to prevent that.
      **kwargs: Abbreviation='long name' mappings to resolve individual abbreviations

    Returns:
      A DataFrame with one column added per hierarchical layer of analysis, starting from level 0.
    """
    extracted_levels = re.findall(FORM_LEVEL_CAPTURE_REGEX, label)
    extracted_tokens = re.split(FORM_LEVEL_SPLIT_REGEX, label)[1:]
    abbreviations = FORM_TOKEN_ABBREVIATIONS if default_abbreviations else {}
    abbreviations.update(kwargs)
    return distribute_tokens_over_levels(
        extracted_levels, extracted_tokens, abbreviations=abbreviations
    )


def expand_form_labels(
    fl: pd.DataFrame,
    fill_mn_until: int = None,
    default_abbreviations=True,
    logger=None,
    **kwargs,
) -> pd.DataFrame:
    """Expands form labels into a hierarchical view of levels in a table.

    Args:
      fl: A DataFrame containing raw form labels as retrieved from :meth:`ms3.Score.mscx.form_labels()`.
      fill_mn_until: Pass the last measure number if you want every measure of the piece have a row in the tree view,
          even if it doesn't come with a form label. This may be desired for increased intuition of proportions,
          rather than seeing all form labels right below each other. In order to add the empty rows, even without
          knowing the number of measures, pass -1.
      default_abbreviations: By default, each token component is checked against a mapping from abbreviations to
          long names. Pass False to prevent that.
      **kwargs: Abbreviation='long name' mappings to resolve individual abbreviations

    Returns:
      A DataFrame with one column added per hierarchical layer of analysis, starting from level 0.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    form_labels = fl.form_label.str.replace("&amp;", "&", regex=False).str.replace(
        r"\s", " ", regex=True
    )
    extracted_levels = form_labels.str.extractall(FORM_LEVEL_CAPTURE_REGEX)
    extracted_levels = extracted_levels.unstack().reindex(fl.index)
    extracted_tokens = form_labels.str.split(FORM_LEVEL_SPLIT_REGEX, expand=True)
    # CHECKS:
    # extracted_tokens.apply(lambda S: S.str.contains(r'(?<![ixv]):').fillna(False)).any(axis=1)
    # extracted_tokens[extracted_tokens[0] != '']
    # fl[fl.form_label.str.contains(':&')]
    extracted_tokens = extracted_tokens.drop(columns=0)
    assert (extracted_tokens.index == extracted_levels.index).all(), (
        "Indices need to be identical after regex " "extraction."
    )
    result_dict: Dict[int, Dict[Tuple[str, str], str]] = {}
    """{ix -> {(form_tree, level) -> token}}"""
    abbreviations: Dict[str, str] = (
        FORM_TOKEN_ABBREVIATIONS if default_abbreviations else {}
    )
    abbreviations.update(kwargs)
    for mc, (i, lvls), (_, tkns) in zip(
        fl.mc, extracted_levels.iterrows(), extracted_tokens.iterrows()
    ):
        level_select, token_select = lvls.notna(), tkns.notna()
        present_levels, present_tokens = lvls[level_select], tkns[token_select]
        assert level_select.sum() == token_select.sum(), (
            f"MC {mc}: {level_select.sum()} levels\n"
            f"{present_levels}\nbut {token_select.sum()} tokens:\n"
            f"{present_tokens}"
        )
        result_dict[i] = distribute_tokens_over_levels(
            present_levels,
            present_tokens,
            mc=mc,
            abbreviations=abbreviations,
            logger=logger,
        )
    form_labels_without_positions = pd.DataFrame.from_dict(result_dict, orient="index")
    form_labels_without_positions.columns = pd.MultiIndex.from_tuples(
        form_labels_without_positions.columns
    )
    form_types = form_labels_without_positions.columns.levels[0]
    has_multiple_trees = len(form_types) > 1
    if has_multiple_trees:
        # expanded columns will be ordered by form trees, 'a_0', 'a_1', ..., 'b_0', 'b_1', ...
        if "" in form_types:
            # there are labels pertaining to all form_types
            forms = [f for f in form_types if f != ""]
            pertaining_to_all = form_labels_without_positions.loc[:, ""]
            distributed_to_all = pd.concat(
                [pertaining_to_all] * len(forms), keys=forms, axis=1
            )
            level_exists = distributed_to_all.columns.isin(
                form_labels_without_positions.columns
            )
            existing_level_names = distributed_to_all.columns[level_exists]
            form_labels_without_positions = pd.concat(
                [
                    form_labels_without_positions.loc[:, forms],
                    distributed_to_all.loc[:, ~level_exists],
                ],
                axis=1,
            )
            potentially_preexistent = distributed_to_all.loc[:, level_exists]
            check_double_attribution = (
                form_labels_without_positions[existing_level_names].notna()
                & potentially_preexistent.notna()
            )
            if check_double_attribution.any().any():
                logger.warning(
                    "Did not distribute levels to all form types because some had already been individually specified."
                )
            form_labels_without_positions.loc[:, existing_level_names] = (
                form_labels_without_positions[existing_level_names].fillna(
                    potentially_preexistent
                )
            )
        # fl_multiindex = pd.concat([fl], keys=[""], axis=1)
        # result = pd.concat([fl_multiindex, form_labels_without_positions.sort_index(axis=1)], axis=1)
    # else:
    # only a single form tree has been encoded
    # if form_types[0] == "":
    #     result = pd.concat([fl, form_labels_without_positions.droplevel(0, axis=1).sort_index(axis=1)], axis=1)
    # else:
    #     result = pd.concat([fl, form_labels_without_positions.sort_index(axis=1)], axis=1)
    #     logger.info(
    #         f"Syntax for several form types used for a single one: '{form_types[0]}'"
    #     )
    elif form_types[0] != "":
        logger.warning(
            f"Syntax for several form types used for a single one: '{form_types[0]}'"
        )

    def merge_level_names(level_names):
        return "_".join(name for name in level_names if name != "")

    form_labels_without_positions.columns = form_labels_without_positions.columns.map(
        merge_level_names
    )
    result = pd.concat([fl, form_labels_without_positions.sort_index(axis=1)], axis=1)

    if fill_mn_until is not None:
        mn_col, mn_onset = "mn", "mn_onset"
        first_mn = fl.mn.min()
        last_mn = fill_mn_until if fill_mn_until > -1 else fl.mn.max()
        all_mns = set(range(first_mn, last_mn + 1))
        missing = all_mns.difference(set(result[mn_col]))
        missing_mn = pd.DataFrame({mn_col: list(missing)}).reindex(
            result.columns, axis=1
        )
        result = (
            pd.concat([result, missing_mn], ignore_index=True)
            .sort_values([mn_col, mn_onset])
            .reset_index(drop=True)
        )
    return result


@overload
def add_collections(left: pd.Series, right: Collection, dtype: Dtype) -> pd.Series: ...


@overload
def add_collections(left: NDArray, right: Collection, dtype: Dtype) -> NDArray: ...


@overload
def add_collections(left: list, right: Collection, dtype: Dtype) -> list: ...


@overload
def add_collections(left: tuple, right: Collection, dtype: Dtype) -> tuple: ...


def add_collections(
    left: Union[pd.Series, NDArray, list, tuple],
    right: Collection,
    dtype: Dtype = "string",
) -> Union[pd.Series, NDArray, list, tuple]:
    """Zip-adds together the strings (by default) contained in two collections regardless of their types (think of
    adding
    two columns together element-wise). Pass another ``dtype`` if you want the values to be converted to another
    datatype before adding them together.
    """
    if isinstance(left, pd.Series) and isinstance(right, pd.Series):
        left.astype(dtype) + right.astype(dtype)
    result_series = pd.Series(left, dtype=dtype) + pd.Series(right, dtype=dtype)
    try:
        return left.__class__(result_series.to_list())
    except (TypeError, ValueError):
        return result_series.values


@overload
def cast2collection(coll: pd.Series, func: Callable, *args, **kwargs) -> pd.Series: ...


@overload
def cast2collection(coll: NDArray, func: Callable, *args, **kwargs) -> NDArray: ...


@overload
def cast2collection(coll: list, func: Callable, *args, **kwargs) -> list: ...


@overload
def cast2collection(coll: tuple, func: Callable, *args, **kwargs) -> tuple: ...


def cast2collection(
    coll: Union[pd.Series, NDArray, list, tuple], func: Callable, *args, **kwargs
) -> Union[pd.Series, NDArray, list, tuple]:
    if isinstance(coll, pd.Series):
        return transform(coll, func, *args, **kwargs)
    with warnings.catch_warnings():
        # pandas developers doing their most annoying thing >:(
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=(".*The default dtype for empty Series.*"),
        )
        result_series = func(pd.Series(coll), *args, **kwargs)
    try:
        return coll.__class__(result_series.to_list())
    except TypeError:
        return result_series.values


@overload
def fifths2acc(fifths: int) -> str: ...


@overload
def fifths2acc(fifths: pd.Series) -> pd.Series: ...


@overload
def fifths2acc(fifths: NDArray[int]) -> NDArray[str]: ...


@overload
def fifths2acc(fifths: List[int]) -> List[str]: ...


@overload
def fifths2acc(fifths: Tuple[int]) -> Tuple[str]: ...


def fifths2acc(
    fifths: Union[int, pd.Series, NDArray[int], List[int], Tuple[int]],
) -> Union[str, pd.Series, NDArray[str], List[str], Tuple[str]]:
    """Returns accidentals for a stack of fifths that can be combined with a
    basic representation of the seven steps."""
    try:
        fifths = int(float(fifths))
    except TypeError:
        return cast2collection(coll=fifths, func=fifths2acc)
    acc = (fifths + 1) // 7
    return abs(acc) * "b" if acc < 0 else acc * "#"


@overload
def fifths2iv(
    fifths: int,
    smallest: bool,
    perfect: str,
    major: str,
    minor: str,
    augmented: str,
    diminished: str,
) -> Optional[str]: ...


@overload
def fifths2iv(
    fifths: pd.Series,
    smallest: bool,
    perfect: str,
    major: str,
    minor: str,
    augmented: str,
    diminished: str,
) -> Optional[pd.Series]: ...


@overload
def fifths2iv(
    fifths: NDArray[int],
    smallest: bool,
    perfect: str,
    major: str,
    minor: str,
    augmented: str,
    diminished: str,
) -> Optional[NDArray[str]]: ...


@overload
def fifths2iv(
    fifths: List[int],
    smallest: bool,
    perfect: str,
    major: str,
    minor: str,
    augmented: str,
    diminished: str,
) -> Optional[List[str]]: ...


@overload
def fifths2iv(
    fifths: Tuple[int],
    smallest: bool,
    perfect: str,
    major: str,
    minor: str,
    augmented: str,
    diminished: str,
) -> Optional[Tuple[str]]: ...


def fifths2iv(
    fifths: Union[int, pd.Series, NDArray[int], List[int], Tuple[int]],
    smallest: bool = False,
    perfect: str = "P",
    major: str = "M",
    minor: str = "m",
    augmented: str = "a",
    diminished: str = "d",
) -> Optional[Union[str, pd.Series, NDArray[str], List[str], Tuple[str]]]:
    """Return interval name of a stack of fifths such that 0 = 'P1', -1 = 'P4', -2 = 'm7', 4 = 'M3' etc. If you pass
    ``smallest=True``, intervals of a fifth or greater will be inverted (e.g. 'm6' => '-M3' and 'D5' => '-A4').


    Args:
      fifths: Number of fifths representing the inveral
      smallest: Pass True if you want to wrap intervals of a fifths and larger to the downward counterpart.
      perfect: String representing the perfect interval quality, defaults to 'P'.
      major: String representing the major interval quality, defaults to 'M'.
      minor: String representing the minor interval quality, defaults to 'm'.
      augmented: String representing the augmented interval quality, defaults to 'a'.
      diminished: String representing the diminished interval quality, defaults to 'd'.

    Returns:
      Name of the interval as a string.
    """
    try:
        if pd.isnull(fifths):
            return fifths
    except ValueError:
        pass
    if isinstance(fifths, pd.Series):
        return cast2collection(
            coll=fifths,
            func=fifths2iv,
            smallest=smallest,
            perfect=perfect,
            major=major,
            minor=minor,
            augmented=augmented,
            diminished=diminished,
        )
    try:
        fifths = int(float(fifths))
    except TypeError:
        return cast2collection(
            coll=fifths,
            func=fifths2iv,
            smallest=smallest,
            perfect=perfect,
            major=major,
            minor=minor,
            augmented=augmented,
            diminished=diminished,
        )
    fifths_plus_one = fifths + 1  # making 0 = fourth, 1 = unison, 2 = fifth etc.
    int_num = ["4", "1", "5", "2", "6", "3", "7"][fifths_plus_one % 7]
    sharp_wise_quality = augmented
    flat_wise_quality = diminished
    qualities = (
        minor,
        minor,
        minor,
        minor,
        perfect,
        perfect,
        perfect,
        major,
        major,
        major,
        major,
    )
    quality = ""
    if smallest and int(int_num) > 4:
        int_num = str(9 - int(int_num))
        sharp_wise_quality, flat_wise_quality = flat_wise_quality, sharp_wise_quality
        qualities = tuple(reversed(qualities))
        quality = "-"
    if -5 <= fifths <= 5:
        quality += qualities[fifths + 5]
    elif fifths > 5:
        quality += sharp_wise_quality * (fifths_plus_one // 7)
    else:
        quality += flat_wise_quality * ((-fifths + 1) // 7)
    return quality + int_num


@overload
def tpc2name(tpc: int, ms: bool = False, minor: bool = False) -> Optional[str]: ...


@overload
def tpc2name(
    tpc: pd.Series, ms: bool = False, minor: bool = False
) -> Optional[pd.Series]: ...


@overload
def tpc2name(
    tpc: NDArray[int], ms: bool = False, minor: bool = False
) -> Optional[NDArray[str]]: ...


@overload
def tpc2name(
    tpc: List[int], ms: bool = False, minor: bool = False
) -> Optional[List[str]]: ...


@overload
def tpc2name(
    tpc: Tuple[int], ms: bool = False, minor: bool = False
) -> Optional[Tuple[str]]: ...


def tpc2name(
    tpc: Union[int, pd.Series, NDArray[int], List[int], Tuple[int]],
    ms: bool = False,
    minor: bool = False,
) -> Optional[Union[str, pd.Series, NDArray[str], List[str], Tuple[str]]]:
    """Turn a tonal pitch class (TPC) into a name or perform the operation on a collection of integers.

    Args:
      tpc: Tonal pitch class(es) to turn into a note name.
      ms: Pass True if ``tpc`` is a MuseScore TPC, i.e. C = 14
      minor: Pass True if the string is to be returned as lowercase.

    Returns:

    """
    try:
        if pd.isnull(tpc):
            return tpc
    except ValueError:
        pass
    if isinstance(tpc, pd.Series):
        return cast2collection(coll=tpc, func=tpc2name, ms=ms, minor=minor)
    try:
        tpc = int(float(tpc))
    except TypeError:
        return cast2collection(coll=tpc, func=tpc2name, ms=ms, minor=minor)
    note_names = (
        ("f", "c", "g", "d", "a", "e", "b")
        if minor
        else ("F", "C", "G", "D", "A", "E", "B")
    )
    if ms:
        tpc = tpc - 14
    acc, ix = divmod(tpc + 1, 7)
    acc_str = abs(acc) * "b" if acc < 0 else acc * "#"
    return f"{note_names[ix]}{acc_str}"


def tpc2scale_degree(
    tpc: Union[int, pd.Series, NDArray[int], List[int], Tuple[int]],
    localkey: str,
    globalkey: str,
) -> Optional[Union[str, pd.Series, NDArray[str], List[str], Tuple[str]]]:
    """For example, tonal pitch class 3 (fifths, i.e. "A") is scale degree '#3' in the localkey of 'iv' within 'c'
    minor.

    Args:
        fifths: Tonal pitch class(es) to turn into scale degree(s).
        localkey: Local key in which the pitch classes are situated, as Roman numeral (can include slash notation
        such as V/ii).
        globalkey: Global key as a note name. E.g. `Ab` for Ab major, or 'c#' for C# minor.

    Returns:
        The given tonal pitch class(es), expressed as scale degree(s).
    """
    try:
        if pd.isnull(tpc):
            return tpc
    except ValueError:
        pass
    if isinstance(tpc, pd.Series):
        return cast2collection(
            coll=tpc, func=tpc2scale_degree, localkey=localkey, globalkey=globalkey
        )
    try:
        tpc = int(float(tpc))
    except TypeError:
        return cast2collection(
            coll=tpc, func=tpc2scale_degree, localkey=localkey, globalkey=globalkey
        )
    global_minor = globalkey.islower()
    if "/" in localkey:
        localkey = resolve_relative_keys(localkey, global_minor)
    localkey_is_minor = localkey.islower()
    lk_fifths = roman_numeral2fifths(localkey, global_minor)
    gk_fifths = name2fifths(globalkey)
    transposed_tpc = transpose(tpc, -(gk_fifths + lk_fifths))
    return fifths2sd(transposed_tpc, minor=localkey_is_minor)


@overload
def fifths2name(
    fifths: int, midi: Optional[int], ms: bool, minor: bool
) -> Optional[str]: ...


@overload
def fifths2name(
    fifths: pd.Series, midi: Optional[pd.Series], ms: bool, minor: bool
) -> Optional[pd.Series]: ...


@overload
def fifths2name(
    fifths: NDArray[int], midi: Optional[NDArray[int]], ms: bool, minor: bool
) -> Optional[NDArray[str]]: ...


@overload
def fifths2name(
    fifths: List[int], midi: Optional[List[int]], ms: bool, minor: bool
) -> Optional[List[str]]: ...


@overload
def fifths2name(
    fifths: Tuple[int], midi: Optional[Tuple[int]], ms: bool, minor: bool
) -> Optional[Tuple[str]]: ...


def fifths2name(
    fifths: Union[int, pd.Series, NDArray[int], List[int], Tuple[int]],
    midi: Optional[Union[int, pd.Series, NDArray[int], List[int], Tuple[int]]] = None,
    ms: bool = False,
    minor: bool = False,
    logger=None,
) -> Optional[Union[str, pd.Series, NDArray[str], List[str], Tuple[str]]]:
    """Return note name of a stack of fifths such that
       0 = C, -1 = F, -2 = Bb, 1 = G etc. This is a wrapper of :func:`tpc2name`, that additionally accepts the argument
       ``midi`` which allows for adding octave information.

    Args:
      fifths: Tonal pitch class(es) to turn into a note name.
      midi: In order to include the octave into the note name, pass the corresponding MIDI pitch(es).
      ms: Pass True if ``fifths`` is a MuseScore TPC, i.e. C = 14
      minor: Pass True if the string is to be returned as lowercase.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    try:
        if pd.isnull(fifths):
            return fifths
    except ValueError:
        pass
    try:
        fifths = int(float(fifths))
    except TypeError:
        names = tpc2name(fifths, ms=ms, minor=minor)
        if midi is None:
            return names
        octaves = midi2octave(midi, fifths)
        return add_collections(names, octaves, dtype="string")
    name = tpc2name(fifths, ms=ms, minor=minor)
    if midi is None:
        return name
    octave = midi2octave(midi, fifths)
    return f"{name}{octave}"


def fifths2pc(fifths):
    """Turn a stack of fifths into a chromatic pitch class.
    Uses: map2elements()
    """
    try:
        fifths = int(float(fifths))
    except Exception:
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
    if isinstance(fifths, Iterable):
        return map2elements(fifths, fifths2rn, minor=minor)
    if pd.isnull(fifths):
        return fifths
    rn = (
        ["VI", "III", "VII", "IV", "I", "V", "II"]
        if minor
        else ["IV", "I", "V", "II", "VI", "III", "VII"]
    )
    sel = fifths + 3 if minor else fifths
    res = _fifths2str(sel, rn)
    if auto_key and is_minor_mode(fifths, minor):
        return res.lower()
    return res


def fifths2sd(fifths, minor=False):
    """Return scale degree of a stack of fifths such that
    0 = '1', -1 = '4', -2 = 'b7' in major, '7' in minor etc.
    Uses: map2elements(), fifths2str()
    """
    if isinstance(fifths, Iterable):
        return map2elements(fifths, fifths2sd, minor=minor)
    if pd.isnull(fifths):
        return fifths
    sd = (
        ["6", "3", "7", "4", "1", "5", "2"]
        if minor
        else ["4", "1", "5", "2", "6", "3", "7"]
    )
    if minor:
        fifths += 3
    return _fifths2str(fifths, sd)


def _fifths2str(fifths: int, steps: Collection[str], inverted: bool = False) -> str:
    """Boiler plate used by fifths2-functions.

    Args:
      fifths: Stack of fifths
      steps: Collection of seven names, scale degrees, intervals, etc.
      inverted: By default, return accidental + step. Pass True to get step + accidental instead.

    Returns:
      Accidentals + step from ``steps`` or, if inverted, step + accidentals.
    """
    acc = fifths2acc(fifths)
    fifths += 1
    if inverted:
        return steps[fifths % 7] + acc
    return acc + steps[fifths % 7]


def get_name_of_highest_version_tag(
    repo: git.Repo,
) -> Optional[str]:
    descending_tags = repo.git.tag(l=True, sort="-v:refname")
    latest_version = descending_tags.split("\n")[0]
    if latest_version:
        return latest_version


@cache
def get_git_commit(
    repo_path: str, git_revision: Optional[str], logger=None
) -> Optional[git.Commit]:
    """Returns the git commit object for the given revision.

    Args:
        repo_path:
        git_revision:
            Any specifier that git understands (branch, tag, commit hash, "HEAD", etc.). In addition,
            "LATEST_VERSION" can be passed to get the tag with the highest version number.
        logger:

    Returns:
        git.Commit object that corresponds to the given revision specifier.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    try:
        repo = git.Repo(repo_path, search_parent_directories=True)
    except Exception as e:
        logger.error(f"{repo_path} is not an existing git repository: {e}")
        return
    if git_revision == "LATEST_VERSION":
        git_revision = get_name_of_highest_version_tag(repo)
        if git_revision is None:
            logger.error(
                "Could not find the latest version tag, falling back to current HEAD."
            )
    try:
        return repo.commit(git_revision)
    except BadName:
        logger.error(
            f"{git_revision} does not resolve to a commit for repo {os.path.basename(repo_path)}."
        )


def get_git_repo(
    directory: str | Path,
    search_parent_directories: bool = True,
    logger: Optional[logging.Logger | str] = None,
) -> Optional[git.Repo]:
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    try:
        repo = git.Repo(directory, search_parent_directories=search_parent_directories)
        logger.debug(
            f"{directory!r} has been recognized to be (part of) the git repository "
            f"{repo.working_tree_dir}."
        )
    except Exception as e:
        repo = None
        logger.debug(
            f"{directory!r} is not (part of) an existing git repository: {e!r}"
        )
    return repo


def get_git_revision(
    repo: Optional[git.Repo] = None,
    repo_path: Optional[str] = None,
) -> str:
    if repo is None:
        repo = get_git_repo(repo_path)
        assert repo is not None, "No git repo to get the version from."
    else:
        assert repo_path is None, "Pass either repo or repo_path, not both."
    return repo.head.commit.hexsha


@overload
def get_git_tag(
    repo: Optional[git.Repo], repo_path: Optional[str], always: Literal[True]
) -> str: ...


@overload
def get_git_tag(
    repo: Optional[git.Repo], repo_path: Optional[str], always: Literal[False]
) -> Optional[str]: ...


def get_git_tag(
    repo: Optional[git.Repo] = None,
    repo_path: Optional[str] = None,
    always: bool = True,
) -> Optional[str]:
    """If always is set to True and no tags are found, the commit short hash is returned instead."""
    if repo is None:
        repo = get_git_repo(repo_path)
        assert repo is not None, "No git repo to get the version from."
    else:
        assert repo_path is None, "Pass either repo or repo_path, not both."
    if always:
        return repo.git.describe(tags=True, always=always)
    try:
        return repo.git.describe(tags=True)
    except Exception:
        return


def get_git_version_info(
    repo: Optional[git.Repo] = None,
    repo_path: Optional[str] = None,
    only_if_clean: bool = True,
):
    if repo is None:
        repo = get_git_repo(repo_path)
        assert repo is not None, "No git repo to get the version from."
    else:
        assert repo_path is None, "Pass either repo or repo_path, not both."
    if only_if_clean and repo.is_dirty():
        return {}
    return dict(
        git_revision=get_git_revision(repo=repo),
        git_tag=get_git_tag(repo=repo, always=True),
    )


def git_repo_is_clean(
    repo: Optional[git.Repo] = None,
    repo_path: Optional[str] = None,
) -> bool:
    if repo is None:
        repo = get_git_repo(repo_path)
        assert repo is not None, "No git repo to get the version from."
    else:
        assert repo_path is None, "Pass either repo or repo_path, not both."
    return not repo.is_dirty()


def get_ms_version(mscx_file):
    with open(mscx_file, encoding="utf-8") as file:
        for i, l in enumerate(file):
            if i < 2:
                pass
            if i == 2:
                m = re.search(r"<programVersion>(.*?)</programVersion>", l)
                if m is None:
                    return None
                else:
                    return m.group(1)


def get_musescore(
    MS: Union[str, Literal["auto", "win", "mac"]] = "auto", logger=None
) -> Optional[str]:
    """Tests whether a MuseScore executable can be found on the system.
    Uses: test_binary()


    Args:
      MS: A path to the executable, installed command, or one of the keywords {'auto', 'win', 'mac'}

    Returns:
      Path to the executable if found or None.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if MS is None:
        return MS
    if MS == "auto":
        mapping = {"Windows": "win", "Darwin": "mac", "Linux": "mscore"}
        system = platform.system()
        try:
            MS = mapping[system]
        except Exception:
            logger.warning(f"System could not be inferred: {system}")
            MS = "mscore"
    if MS == "win":
        program_files = os.environ["PROGRAMFILES"]
        MS = os.path.join(program_files, r"MuseScore 3\bin\MuseScore3.exe")
    elif MS == "mac":
        MS = "/Applications/MuseScore 3.app/Contents/MacOS/mscore"
    return test_binary(MS, logger=logger)


def get_path_component(path, after):
    """Returns only the path's subfolders below ``after``. If ``after`` is the last
    component, '.' is returned."""
    dir1, base1 = os.path.split(path)
    if dir1 in ("", ".", "/", "~"):
        if base1 == after:
            return "."
        return path
    dir2, base2 = os.path.split(dir1)
    if base2 == after:
        return base1
    higher_levels = get_path_component(dir1, after=after)
    return os.path.join(higher_levels, base1)


# def get_quarterbeats_length(measures: pd.DataFrame, decimals: int = 2) -> Tuple[float, Optional[float]]:
#     """ Returns the symbolic length and unfolded symbolic length of a piece in ♩.
#
#     Parameters
#     ----------
#     measures : :obj:`pandas.DataFrame`
#
#     Returns
#     -------
#     float, float
#         Length and unfolded length, both measured in ♩.
#     """
#     mc_durations = measures.set_index('mc').act_dur * 4.
#     length_qb = round(mc_durations.sum(), decimals)
#     try:
#         playthrough2mc = make_playthrough2mc(measures, logger=logger)
#         if len(playthrough2mc) == 0:
#             length_qb_unfolded = None
#         else:
#             length_qb_unfolded = round(mc_durations.loc[playthrough2mc.values].sum(), decimals)
#     except Exception:
#         length_qb_unfolded = None
#     return length_qb, length_qb_unfolded


def group_id_tuples(list_of_pairs):
    """Turns a list of (key, ix) into a {key: [ix]}"""
    d = defaultdict(list)
    for k, i in list_of_pairs:
        if k is not None:
            d[k].append(i)
    return dict(d)


def html2format(df, format="name", html_col="color_html"):
    """Converts the HTML column of a DataFrame into 'name', 'rgb , or 'rgba'."""
    if format == "name":
        return df[html_col].map(color_name2html)
    if format == "rgb":
        return df[html_col].map(color_name2rgb)
    if format == "rgba":
        return df[html_col].map(color_name2rgba)


def html_color2format(h, format="name"):
    """Converts a single HTML color into 'name', 'rgb', or  'rgba'."""
    if pd.isnull(h):
        return h
    if format == "name":
        try:
            return webcolors.hex_to_name(h)
        except Exception:
            try:
                return MS3_HTML[h]
            except Exception:
                return h
    if format == "rgb":
        return webcolors.hex_to_rgb(h)
    if format == "rgba":
        rgb = webcolors.hex_to_rgb(h)
        return rgba(*(rgb + (255,)))


def html_color2name(h):
    """Converts a HTML color into its CSS3 name or itself if there is none."""
    return html_color2format(h, "name")


def html_color2rgb(h):
    """Converts a HTML color into RGB."""
    return html_color2format(h, "rgb")


def html_color2rgba(h):
    """Converts a HTML color into RGBA."""
    return html_color2format(h, "rgba")


def interval_overlap(a, b, closed=None):
    """Returns the overlap of two pd.Intervals as a new pd.Interval.

    Parameters
    ----------
    a, b : :obj:`pandas.Interval`
        Intervals for which to compute the overlap.
    closed : {'left', 'right', 'both', 'neither'}, optional
        If no value is passed, the closure of the returned interval is inferred from ``a`` and ``b``.

    Returns
    -------
    :obj:`pandas.Interval`
    """
    if not a.overlaps(b):
        return None
    if b.left < a.left:
        # making a the leftmost interval
        a, b = b, a
    if closed is None:
        if a.right < b.right:
            right = a.right
            right_closed = a.closed in ("right", "both")
            other_iv = b
        else:
            right = b.right
            right_closed = b.closed in ("right", "both")
            other_iv = a
        if (
            right_closed
            and right == other_iv.right
            and other_iv.closed not in ("right", "both")
        ):
            right_closed = False
        left_closed = b.closed in ("left", "both")
        if left_closed and a.left == b.left and a.closed not in ("left", "both"):
            left_closed = False

        if left_closed and right_closed:
            closed = "both"
        elif left_closed:
            closed = "left"
        elif right_closed:
            closed = "right"
        else:
            closed = "neither"
    else:
        right = a.right if a.right < b.right else b.right
    return pd.Interval(b.left, right, closed=closed)


def interval_overlap_size(a, b, decimals=3):
    """Returns the size of the overlap of two pd.Intervals."""
    if not a.overlaps(b):
        return 0.0
    if b.left < a.left:
        # making a the leftmost interval
        a, b = b, a
    right = a.right if a.right < b.right else b.right
    result = right - b.left
    return round(result, decimals)


def is_any_row_equal(df1, df2):
    """Returns True if any two rows of the two DataFrames contain the same value tuples."""
    assert len(df1.columns) == len(
        df2.columns
    ), "Pass the same number of columns for both DataFrames"
    v1 = set(df1.itertuples(index=False, name=None))
    v2 = set(df2.itertuples(index=False, name=None))
    return v1.intersection(v2)


def is_minor_mode(fifths, minor=False):
    """Returns True if the scale degree `fifths` naturally has a minor third in the scale."""
    thirds = [-4, -3, -2, -1, 0, 1, 2] if minor else [3, 4, 5, -1, 0, 1, 2]
    third = thirds[(fifths + 1) % 7] - fifths
    return third == -3


def iter_nested(nested):
    """Iterate through any nested structure of lists and tuples from left to right."""
    for elem in nested:
        if isinstance(elem, list) or isinstance(elem, tuple):
            for lower in iter_nested(elem):
                yield lower
        else:
            yield elem


def iter_selection(collectio, selector=None, opposite=False):
    """Returns a generator of ``collectio``. ``selector`` can be a collection of index numbers to select or unselect
    elements -- depending on ``opposite``"""
    if selector is None:
        for e in collectio:
            yield e
    if opposite:
        for i, e in enumerate(collectio):
            if i not in selector:
                yield e
    else:
        for i, e in enumerate(collectio):
            if i in selector:
                yield e


def iterable2str(iterable):
    if isinstance(iterable, str):
        return iterable
    try:
        return ", ".join(str(s) for s in iterable)
    except Exception:
        return iterable


def contains_metadata(path):
    for _, _, files in os.walk(path):
        return any(f == "metadata.tsv" for f in files)


def first_level_subdirs(path):
    """Returns the directory names contained in path."""
    for _, subdirs, _ in os.walk(path):
        return subdirs


def first_level_files_and_subdirs(path):
    """Returns the directory names and filenames contained in path."""
    for _, subdirs, files in os.walk(path):
        return subdirs, files


def contains_corpus_indicator(path, logger=None):
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    for subdir in first_level_subdirs(path):
        if subdir in STANDARD_NAMES_OR_GIT:
            logger.debug(
                f"{path} contains a subdirectory called {subdir} and is assumed to be a corpus."
            )
            return True
    return False


def get_first_level_corpora(path: str, logger=None) -> List[str]:
    """Checks the first-level subdirectories of path for indicators of being a corpus. If one of them shows an
    indicator (presence of a 'metadata.tsv' file, or of a '.git' folder or any of the default folder names), returns
    a list of all subdirectories.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if path is None or not os.path.isdir(path):
        logger.info(f"{path} is not an existing directory.")
        return
    subpaths = [
        os.path.join(path, subdir)
        for subdir in first_level_subdirs(path)
        if subdir[0] != "."
    ]
    for subpath in subpaths:
        if contains_metadata(subpath):
            logger.debug(
                f"{subpath} recognized as corpus directory because it contains metadata."
            )
            return subpaths
        if contains_corpus_indicator(subpath, logger=logger):
            return subpaths
    return []


def join_tsvs(dfs, sort_cols=False, logger=None):
    """Performs outer join on the passed DataFrames based on 'mc' and 'mc_onset', if any.
    Uses: functools.reduce(), sort_cols(), sort_note_lists()

    Parameters
    ----------
    dfs : :obj:`Collection`
        Collection of DataFrames to join.
    sort_cols : :obj:`bool`, optional
        If you pass True, the columns after those defined in :py:attr:`STANDARD_COLUMN_ORDER`
        will be sorted alphabetically.

    Returns
    -------

    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if len(dfs) == 1:
        return dfs[0]
    zero, one, two = [], [], []
    for df in dfs:
        if "mc" in df.columns:
            if "mc_onset" in df.columns:
                two.append(df)
            else:
                one.append(df)
        else:
            zero.append(df)
    join_order = two + one
    if len(zero) > 0:
        logger.info(
            f"{len(zero)} DataFrames contain none of the columns 'mc' and 'mc_onset'."
        )

    pos_cols = ["mc", "mc_onset"]

    def join_tsv(a, b):
        join_cols = [c for c in pos_cols if c in a.columns and c in b.columns]
        res = pd.merge(
            a, b, how="outer", on=join_cols, suffixes=("", "_y")
        ).reset_index(drop=True)
        duplicates = [col for col in res.columns if col.endswith("_y")]
        for d in duplicates:
            left = d[:-2]
            if res[left].isna().any():
                res[left].fillna(res[d], inplace=True)
        return res.drop(columns=duplicates)

    res = reduce(join_tsv, join_order)
    if "midi" in res.columns:
        res = sort_note_list(res)
    elif len(two) > 0:
        res = res.sort_values(pos_cols)
    else:
        res = res.sort_values("mc")
    return column_order(res, sort=sort_cols).reset_index(drop=True)


def str2inttuple(tuple_string: str, strict: bool = True) -> Tuple[int]:
    tuple_string = tuple_string.strip("[](),")
    if tuple_string == "":
        return tuple()
    res = []
    for s in tuple_string.split(", "):
        try:
            res.append(int(s))
        except ValueError:
            if strict:
                print(
                    f"String value '{s}' could not be converted to an integer, "
                    f"'{tuple_string}' not to an integer tuple."
                )
                raise
            if s[0] == s[-1] and s[0] in ('"', "'"):
                s = s[1:-1]
            try:
                res.append(int(s))
            except ValueError:
                res.append(s)
    return tuple(res)


def str2keysig_dict(dict_content: str) -> Dict[int, int]:
    entry_tuples = re.findall(KEYSIG_DICT_ENTRY_REGEX, dict_content)
    return {int(mc): int(keysig) for mc, keysig in entry_tuples}


def str2timesig_dict(dict_content: str) -> Dict[int, str]:
    entry_tuples = re.findall(TIMESIG_DICT_ENTRY_REGEX, dict_content)
    return {int(mc): timesig for mc, timesig in entry_tuples}


# default values copied from marshmallow's fields.Boolean
TRUTHY_VALUES = {
    "t",
    "T",
    "true",
    "True",
    "TRUE",
    "on",
    "On",
    "ON",
    "y",
    "Y",
    "yes",
    "Yes",
    "YES",
    "1",
    1,
    True,
}
FALSY_VALUES = {
    "f",
    "F",
    "false",
    "False",
    "FALSE",
    "off",
    "Off",
    "OFF",
    "n",
    "N",
    "no",
    "No",
    "NO",
    "0",
    0,
    0.0,
    False,
}


def int2bool(s: str) -> Union[bool, str]:
    try:
        return bool(int(s))
    except Exception:
        return s


def value2bool(value: str | float | int | bool) -> bool | str | float | int:
    if value in TRUTHY_VALUES:
        return True
    if value in FALSY_VALUES:
        return False
    if isinstance(value, str):
        try:
            converted = float(value)
        except Exception:
            return value
        if converted in TRUTHY_VALUES:
            return True
        if converted in FALSY_VALUES:
            return False
    return value


def safe_frac(s: str) -> Union[Fraction, str]:
    try:
        return Fraction(s)
    except Exception:
        return s


def safe_int(s) -> Union[int, str]:
    try:
        return int(float(s))
    except Exception:
        return s


def tuple_to_list_recursive(t):
    if isinstance(t, (tuple, list)):
        return [tuple_to_list_recursive(e) for e in t]
    return t


def eval_string_to_nested_list(s):
    """Tries to parse a string encoding a nested list, returns the input if it fails."""
    try:
        nested_tuple_or_list = literal_eval(s)
        return tuple_to_list_recursive(nested_tuple_or_list)
    except Exception:
        return s


def parse_interval_index_column(df, column=None, closed="left"):
    """Turns a column of strings in the form '[0.0, 1.1)' into a :obj:`pandas.IntervalIndex`.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
    column : :obj:`str`, optional
        Name of the column containing strings. If not specified, use the index.
    closed : :obj:`str`, optional
        On whot side the intervals should be closed. Defaults to 'left'.

    Returns
    -------
    :obj:`pandas.IntervalIndex`
    """
    iv_regex = SLICE_INTERVAL_REGEX
    if column is None:
        iv_strings = df.index
    else:
        iv_strings = df[column]
    values = iv_strings.str.extract(iv_regex).astype(float)
    iix = pd.IntervalIndex.from_arrays(values[0], values[1], closed=closed)
    return iix


TSV_COLUMN_CONVERTERS = {
    "added_tones": str2inttuple,
    "act_dur": safe_frac,
    "composed_end": safe_int,
    "composed_start": safe_int,
    "chord_tones": str2inttuple,
    "globalkey_is_minor": value2bool,
    "KeySig": str2keysig_dict,
    "localkey_is_minor": value2bool,
    "mc_offset": safe_frac,
    "mc_onset": safe_frac,
    "mn_onset": safe_frac,
    "movementNumber": safe_int,
    "next": str2inttuple,
    "nominal_duration": safe_frac,
    "quarterbeats": safe_frac,
    "quarterbeats_playthrough": safe_frac,
    "quarterbeats_all_endings": safe_frac,
    "onset": safe_frac,
    "duration": safe_frac,
    "scalar": safe_frac,
    "TimeSig": str2timesig_dict,
    "volta_mcs": eval_string_to_nested_list,
}

TSV_COLUMN_TITLES = {
    "absolute_base": "Int64",
    "absolute_root": "Absolute Root",
    "act_dur": "Actual Length",
    "added_tones": "Added Tones",
    "all_notes_qb": "Summed Note Duration",
    "alt_label": "Alternative Label",
    "barline": "Barline",
    "base": "Base",
    "bass_note": "Bass Note",
    "breaks": "Breaks",
    "cadence": "Cadence",
    "cadences_id": "Cadence ID",
    "changes": "Changes",
    "chord": "Chord",
    "chord_id": "Chord ID",
    "chord_tones": "Chord Tones",
    "chord_type": "Chord Type",
    "color_a": "Color Alpha",
    "color_b": "Color Blue",
    "color_g": "Color Green",
    "color_html": "Color HTML",
    "color_name": "Color Name",
    "color_r": "Color Red",
    "composed_end": "Latest Composition Year",
    "composed_start": "Earliest Composition Year",
    "corpus": "Corpus",
    "dont_count": "Exclude from measure count",
    "duration": "Duration",
    "duration_qb": "Duration in ♩",
    "expanded_id": "Label ID",
    "figbass": "Chord Inversion",
    "fname": "Piece identifier",
    "form": "Chord Category",
    "form_label_count": "Number of Form Annotation Labels",
    "globalkey": "Global Key",
    "globalkey_is_minor": "Global Key is Minor",
    "gracenote": "Grace Note",
    "guitar_chord_count": "Number of Guitar/Jazz Chord Labels",
    "harmonies_id": "Label ID",
    "harmony_layer": "Harmony Encoding Layer",
    "i": "Index",
    "keysig": "Key Signature",
    "KeySig": "Key Signatures",
    "label": "Label",
    "label_count": "Number of DCML Labels",
    "label_type": "Label Type",
    "last_mc": "Highest Measure Count",
    "last_mc_unfolded": "Highest Measure Count (unfolded)",
    "last_mn": "Highest Measure Number",
    "last_mn_unfolded": "Highest Measure Number (unfolded)",
    "leftParen": "Left Parenthesis",
    "length_qb": "Quarter Length",
    "length_qb_unfolded": "Quarter Length (unfolded)",
    "localkey": "Local Key",
    "localkey_is_minor": "Local Key is Minor",
    "marker": "Marker",
    "mc": "Measure Count",
    "mc_offset": "Offset of Encoded Measure",
    "mc_onset": "Offset within Encoded Measure",
    "mc_playthrough": "Measure Count Unfolded",
    "midi": "MIDI pitch",
    "mn": "Measure Number",
    "mn_onset": "Offset within Logical Measure",
    "movementNumber": "Movement Number",
    "name": "Name",
    "nashville": "Nashville",
    "next": "Next Measure Counts",
    "n_onsets": "Number of Note Onsets",
    "n_onset_positions": "Number of Unique Onset Positions (slices)",
    "nominal_duration": "Nominal Duration",
    "notes_id": "Notes ID",
    "numbering_offset": "Numbering Offset",
    "numeral": "Roman Numeral",
    "octave": "Octave",
    "offset:x": "Horizontal Offset",
    "offset:y": "Vertical Offset",
    "offset_x": "Horizontal Offset",
    "offset_y": "Vertical Offset",
    # 'onset':
    "pedal": "Pedal Point",
    "phraseend": "Phrase Annotation",
    "piece": "Piece identifier",
    # 'playthrough':
    "quarterbeats": "Offset from Beginning (leaving out alternative endings)",
    "quarterbeats_playthrough": "Offset from the beginning, including all repeats (in unfolded tables)",
    "quarterbeats_all_endings": "Offset from Beginning (counting through alternative "
    "endings as if they were adjacent bars)",
    "regex_match": "Regular Expression Match",
    "relativeroot": "Relative Root",
    "repeats": "Repeats",
    "rightParen": "Right Parenthesis",
    "root": "Root",
    "rootCase": "Root Case",
    "scalar": "Scalar",
    "slur": "Slur",
    "special": "Special Label",
    "staff": "Staff",
    "tied": "Tied Note",
    "timesig": "Time Signature",
    "TimeSig": "Time Signatures",
    "tpc": "Tonal Pitch Class",
    "voice": "Notational Layer",
    # 'voices': 'Voices',
    "volta": "Volta",
    "volta_mcs": "Volta Measure Counts",
}

TSV_COLUMN_DESCRIPTIONS = {
    "absolute_base": "MuseScore encoding the bass pitch class of an absolute chord label, as MuseScore tonal pitch "
    "class such that C=14, G=15, etc.",
    "absolute_root": "MuseScore encoding the root pitch class of an absolute chord label, as MuseScore tonal pitch "
    "class such that C=14, G=15, etc.",
    "act_dur": "How long a measure actually lasts, which can deviate from the time signature. Relevant, for example, "
    "for pickup measures, split measures, cadenzas.",
    "added_tones": "Chord tones considered as added, expressed as fifth intervals relative to the local tonic.",
    "all_notes_qb": "The summed duration of all notes in a given piece, measured in ♩.",
    "alt_label": "Another interpretation of the same chord which the annotator finds equally or slightly less "
    "convincing.",
    "barline": "Name of non-default barline.",
    "base": "Base",
    "bass_note": "Loweste note designated by the chord label, expressed as fifth intervals relative to the local "
    "tonic.",
    "breaks": 'Can be "section" or "line".',
    "cadence": "PAC, IAC, EC, DC, PC, HC, and HC subtypes such as HC.SIM",
    "cadences_id": "Row in the cadence table.",
    "changes": "Alterations, suspensions, additions, and omissions, written within parentheses in the chord label.",
    "chord": "The part of an annotation label that corresponds to the actual chord label.",
    "chord_id": "Row in the chords table.",
    "chord_tones": "Tones expressed by the label, as fifth intervals relative to the local tonic.",
    "chord_type": "'M', 'm', 'o', '+', 'mm7', 'Mm7', 'MM7', 'mM7', 'o7', '%7', '+7', '+M7'",
    "color_a": "Value between 0 and 255",
    "color_b": "Value between 0 and 255",
    "color_g": "Value between 0 and 255",
    "color_html": "A hexadicmal RGB value, e.g. #FF0000",
    "color_name": "Can be a CSS name or one of the names for MuseScore colors stored in ms3.utils.MS3_HTML",
    "color_r": "Value between 0 and 255",
    "composed_end": "Latest possible year of composition.",
    "composed_start": "Earliest possible year of composition.",
    "corpus": "Name of a group of pieces",
    "dont_count": "Is 1 if this MC does not increase the measure number counter, according to the MuseScore setting.",
    "duration": "As fraction of a whole note.",
    "duration_qb": "A float corresponding to duration * 4",
    "expanded_id": "Row in the expanded table.",
    "figbass": "7, 65, 43, 2, 64, 6 or empty for root position.",
    "fname": "Name identifier (filename without suffixes) of a piece",
    "form": "%, o, +, M, +M",
    "form_label_count": "Number of form annotation labels in a piece.",
    "globalkey": "The key of the entire piece as note name, lowercase designating a minor key.",
    "globalkey_is_minor": "Boolean that is 1 if the piece is in minor and 0 if it is in major.",
    "gracenote": 'Name given to a type of grace note in the MuseScore encoding, e.g. "grace16"',
    "guitar_chord_count": "Number of guitar/jazz/absolute chord labels in a piece.",
    "harmonies_id": "Row in the expanded table.",
    "harmony_layer": "0: Simple string (does not begin with a note name, otherwise MS3 will turn it into type 3; "
    "prevent through leading dot);\n"
    "1: MuseScore's Roman Numeral Annotation format;\n"
    "2: MuseScore's Nashville Number format;\n"
    "3: Absolute chord encoded by MuseScore.",
    "i": "An integer serving as row ID",
    "keysig": "Positive integer for number of sharps, negative integer for number of flats.",
    "KeySig": "All key signatures in a piece and where they occur. Format: '<MC>: <keysig>, <MC>: <keysig>, ...'"
    "Corresponds to a dictionary without outer curly braces.",
    "label": "String corresponding to the entire annotation label.",
    "label_count": "Number of chord annotation labels that match the regex of the DCML annotation standard.",
    "label_type": 'Previous name of what now is called "regex_match".',
    "last_mc": "The highest Measure Count value in the piece. Corresponds to the number of <Measure> tags.",
    "last_mc_unfolded": "The amount of <Measure> tags of a full playthrough, i.e. when all repeats are unfolded. "
    "Equals 'last_mc' if piece has no repeat signs.",
    "last_mn": "The highest Measure Number in the piece. Corresponds roughly to the number of complete bars.",
    "last_mn_unfolded": "The last Measure Number when all repeats are unfolded. Equals 'last_mn' if piece has no "
    "repeat signs.",
    "leftParen": "Pertaining to MuseScore encoding.",
    "length_qb": "The length of a piece in quarter notes.",
    "length_qb_unfolded": "The length of full playthrough in quarter notes, i.e. when all repeats are unfolded. "
    "Equals 'length_qb' if piece has no repeat signs.",
    "localkey": "The key that a Roman numeral is relative to, expressed as a Roman numeral relative to the global key.",
    "localkey_is_minor": "Boolean that is 1 if the local key is minor and 0 if it is major.",
    "marker": "Pertaining to MuseScore encoding.",
    "mc": "Running count of encoded <Measure> tags which do not necessarily express a full measure (e.g. in case of "
    "an anacrusis).",
    "mc_offset": "Distance of a <Measure> from the beginning of the logical measure, expressed as fraction of a whole "
    "note. Relevant only for split and anacrusis measures, 0 otherwise.",
    "mc_onset": "Distance of an event from the beginning of the <Measure> tag.",
    "mc_playthrough": 'Has the function of the "mc" column in tables with unfolded repeats where "mc" is not unique.',
    "midi": "Value between 0 and 127 where 60 = C4, 61 = C#4/Db4, etc.",
    "mn": "Measure number as printed in the score, computed from mc, dont_count and numbering_offset.",
    "mn_onset": "Distance from the beginning of the logical measure. Relevant, for example, to compute the metric "
    "position of an event.",
    "movementNumber": "Metadata field for the number of a movement. Should be specified as integer, not as a Roman "
    "number.",
    "n_onsets": "Number of onsets in a piece. Different from the number of entries in the 'notes' table, because "
    "that one corresponds to the number of note heads; but notes being tied to do not represent onsets.",
    "n_onset_positions": "Number of unique onset positions in a piece. Corresponds to the number of slices when "
    "performing full expansion.",
    "name": "",
    "nashville": 'Numbering system that specifies the root of a chord as scale degree of the local key, e.g. "1", '
    '"b3", "#5", "b7".',
    "next": 'The "mc" values of all <Measure> tags that can follow this one, specified as a tuple of integers.',
    "nominal_duration": "The duration corresponding to a note/rest value without applying any dots or n-tuplets, "
    'as fraction of a whole note. Multiplied with "scalar" to yield the actual duration of the '
    "note/rest.",
    "notes_id": "Row in the notes table.",
    "numbering_offset": "An integer to be added to the measure number count, as specified in MuseScore.",
    "numeral": "The Roman numeral part of a DCML label, lowercase for all chords with a minor third.",
    "octave": "Octave number where 4 is the middle octave (' in Helmholtz notation).",
    "offset:x": "Pertaining to MuseScore encoding.",
    "offset:y": "Pertaining to MuseScore encoding.",
    "offset_x": "Pertaining to MuseScore encoding.",
    "offset_y": "Pertaining to MuseScore encoding.",
    # 'onset':
    "pedal": "Specified as Roman numeral.",
    "phraseend": "The phrase annotation part of a DCML label, can be {, }, or }{. In an older version, the only label "
    "was \\\\",
    "piece": "Name identifier (filename without suffixes) of a piece",
    # 'playthrough':
    "quarterbeats": "Distance from the piece's beginning. By default, only second endings are taken into "
    "account to reflect the proportions of a simply playthrough without repeats.",
    "quarterbeats_playthrough": "Distance from the piece's beginning, including all repeats. This column is present "
    "in unfolded representations that correspond to a full 'playthrough' including all repetitions.",
    "quarterbeats_all_endings": "Distance from the piece's beginning, taking all endings into account for "
    "addressability purposes.",
    "regex_match": "The name of the first registered regular expression matching a label. By default, these include "
    '"dcml" and "form_labels".',
    "relativeroot": "The Roman numeral following the / in applied chords. Can itself include a /. For example, "
    "V/V if the label pertains to the major scale on the second scale degree.",
    "repeats": 'Can be "start", "end", "firstMeasure", "lastMeasure"',
    "rightParen": "Pertaining to MuseScore encoding.",
    "root": "Pertaining to MuseScore encoding.",
    "rootCase": "Pertaining to MuseScore encoding.",
    "scalar": "Decimal value reflecting all dots and n-tuplets modifying the duration of a note/rest. Yields the "
    'actual duration when multiplied with "nominal_duration".',
    "slur": "IDs of active slurs that a chord falls under, as a tuple of integers.",
    "special": 'Column where special labels such as "Fr6" or "Ger65" are stored which are converted internally to '
    "Roman numerals.",
    "staff": "Number of the staff where an event occurs, 1 designating the top staff.",
    "tied": "1 if a note is tied to the following one, -1 if it is being tied to by the previous one, 0 if both.",
    "timesig": 'Given as string, e.g. "4/4".',
    "TimeSig": "All time signatures in a piece and where they occur. Format: '<MC>: <timesig>, <MC>: <timesig>, ...' "
    "Corresponds to a dictionary without outer curly braces.",
    "tpc": "Specified on the line of fifths such that 0 = C, 1 = G, -1 = F, etc.",
    "voice": "A number between 1-4 where 1 is MuseScore's default layer (blue), 2 the second layer in green with "
    "downward stems, etc.",
    # 'voices': 'Voices',
    "volta": "Number of an ending bracket, given as integer.",
    "volta_mcs": 'For each group of alternative endings, a nested list where each inner list contains the "mc" values '
    "of the <Measure> tags grouped into one ending.",
}

TSV_COLUMN_DTYPES = {
    "absolute_base": "Int64",
    "absolute_root": "Int64",
    "all_notes_qb": float,
    "alt_label": "string",
    "barline": "string",
    "base": "Int64",
    "bass_note": "Int64",
    "breaks": "string",
    "cadence": "string",
    "cadences_id": "Int64",
    "changes": "string",
    "chord": "string",
    "chord_id": "Int64",
    "chord_type": "string",
    "color_name": "string",
    "color_html": "string",
    "color_r": "Int64",
    "color_g": "Int64",
    "color_b": "Int64",
    "color_a": "Int64",
    "corpus": str,
    "dont_count": "Int64",
    "duration_qb": float,
    "expanded_id": "Int64",
    "figbass": "string",
    "fname": str,
    "form": "string",
    "form_label_count": int,
    "globalkey": str,
    "gracenote": "string",
    "guitar_chord_count": int,
    "harmonies_id": "Int64",
    "harmony_layer": "Int64",
    "i": int,
    "keysig": "Int64",
    "label": "string",
    "label_count": int,
    "label_type": "string",
    "last_mc": int,
    "last_mc_unfolded": "Int64",
    "last_mn": int,
    "last_mn_unfolded": "Int64",
    "leftParen": "string",
    "length_qb": float,
    "length_qb_unfolded": float,
    "localkey": str,
    "marker": "string",
    "mc": "Int64",
    "mc_playthrough": "Int64",
    "midi": "Int64",
    "mn": str,
    "name": "string",
    "offset:x": "string",
    "offset_x": "string",
    "offset:y": "string",
    "offset_y": "string",
    "n_onsets": int,
    "n_onset_positions": int,
    "nashville": "Int64",
    "notes_id": "Int64",
    "numbering_offset": "Int64",
    "numeral": "string",
    "octave": "Int64",
    "pedal": "string",
    "piece": str,
    "phraseend": "string",
    "playthrough": "Int64",
    "regex_match": "string",
    "relativeroot": "string",
    "repeats": "string",
    "rightParen": "string",
    "root": "Int64",
    "rootCase": "Int64",
    "slur": "string",
    "special": "string",
    "staff": "Int64",
    "tied": "Int64",
    "timesig": str,
    "tpc": "Int64",
    "voice": "Int64",
    # 'voices': 'Int64',
    "volta": "Int64",
}


def load_tsv(
    path, index_col=None, sep="\t", converters={}, dtype={}, stringtype=False, **kwargs
) -> Optional[pd.DataFrame]:
    """Loads the TSV file `path` while applying correct type conversion and parsing tuples.

    Parameters
    ----------
    path : :obj:`str`
        Path to a TSV file as output by format_data().
    index_col : :obj:`list`, optional
        By default, the first two columns are loaded as MultiIndex.
        The first level distinguishes pieces and the second level the elements within.
    converters, dtype : :obj:`dict`, optional
        Enhances or overwrites the mapping from column names to types included the constants.
    stringtype : :obj:`bool`, optional
        If you're using pandas >= 1.0.0 you might want to set this to True in order
        to be using the new `string` datatype that includes the new null type `pd.NA`.
    """

    global TSV_COLUMN_CONVERTERS, TSV_COLUMN_DTYPES

    if converters is None:
        conv = None
    else:
        conv = dict(TSV_COLUMN_CONVERTERS)
        conv.update(converters)

    if dtype is None:
        types = None
    elif isinstance(dtype, str):
        types = dtype
    else:
        types = dict(TSV_COLUMN_DTYPES)
        types.update(dtype)

    if stringtype:
        types = {col: "string" if typ == str else typ for col, typ in types.items()}
    try:
        df = pd.read_csv(
            path, sep=sep, index_col=index_col, dtype=types, converters=conv, **kwargs
        )
    except EmptyDataError:
        return
    if "mn" in df:
        mn_volta = mn2int(df.mn)
        df.mn = mn_volta.mn
        if mn_volta.volta.notna().any():
            if "volta" not in df.columns:
                df["volta"] = pd.Series(pd.NA, index=df.index).astype("Int64")
            df.volta.fillna(mn_volta.volta, inplace=True)
    if "interval" in df:
        try:
            iv_index = parse_interval_index_column(df, "interval")
            df.index = iv_index
            df = df.drop(columns="interval")
        except Exception:
            pass
    return df


@cache
def tsv_column2csvw_datatype() -> Dict[str, str | Dict[str, str]]:
    mapping = defaultdict(lambda: "string")
    mapping.update(
        {
            "Int64": "integer",
            str: "string",
            "string": "string",
            float: "float",
            int: "integer",
            int2bool: "boolean",
            value2bool: "boolean",
            safe_frac: {"base": "string", "format": r"-?\d+(?:\/\d+)?"},
            safe_int: "integer",
            str2inttuple: {
                "base": "string",
                "format": r"^[([]?(?:-?\d+\s*,?\s*)*[])]?$",
            },
        }
    )
    column2datatype = {
        col: mapping[dtype] for col, dtype in TSV_COLUMN_CONVERTERS.items()
    }
    column2datatype.update(
        {col: mapping[dtype] for col, dtype in TSV_COLUMN_DTYPES.items()}
    )
    return column2datatype


@cache
def tsv_column2description(col: str) -> Optional[str]:
    mapping = {
        "mc": "Measure count.",
        "mn": "Measure number.",
        "mc_onset": "An event's distance (fraction of a whole note) from the beginning of the MC.",
        "mn_onset": "An event's distance (fraction of a whole note) from the beginning of the MN.",
    }
    if col in mapping:
        return mapping[col]


@cache
def tsv_column2csvw_schema(col: str) -> dict:
    result = {
        "titles": col,
    }
    column2type = tsv_column2csvw_datatype()
    if col in column2type:
        result["datatype"] = column2type[col]
    description = tsv_column2description(col)
    if description is not None:
        result["dc:description"] = description
    return result


def make_csvw_jsonld(
    title: str,
    columns: Collection[str],
    urls: Union[str, Collection[str]],
    description: Optional[str] = None,
) -> dict:
    """W3C's CSV on the Web Primer: https://www.w3.org/TR/tabular-data-primer/"""
    result = {
        "@context": ["http://www.w3.org/ns/csvw#", {"@language": "en "}],
        "dc:title": title,
        "dialect": {
            "delimiter": "\t",
        },
    }
    if description is not None:
        result["dc:description"] = description
    result["dc:created"] = datetime.now().replace(microsecond=0).isoformat()

    result["dc:creator"] = [DEFAULT_CREATOR_METADATA]
    if isinstance(urls, str):
        result["url"] = (urls,)
    else:
        result["tables"] = [{"url": p} for p in urls]
    result["tableSchema"] = {
        "columns": [tsv_column2csvw_schema(col) for col in columns]
    }
    return result


def store_csvw_jsonld(
    corpus: str,
    folder: str,
    facet: str,
    columns: Collection[str],
    files: Union[str, Collection[str]],
) -> str:
    titles = {
        "expanded": "DCML harmony annotations",
        "measures": "Measure tables",
        "notes": "Note tables",
    }
    descriptions = {
        "expanded": "One feature matrix per score, containing one line per label. The first columns (until 'label') "
        "are the same as in extracted 'labels' tables with the difference that only those harmony labels "
        "that match the DCML harmony annotation standard (dcmlab.github.io/standards) are included. Since "
        "these follow a specific syntax, they can be split into their components (features) and "
        "transformed "
        "into scale degrees. For more information, please refer to the docs at "
        "https://ms3.readthedocs.io/columns",
        "measures": "One feature matrix per score, containing one line per stack of <Measure> tags in the score's XML "
        "tree. "
        "They are counted in the column 'mc' starting from 1, whereas the conventional measure numbers "
        "are shown "
        "in the column 'mn'. One MN is frequently composed in two (or more) MCs. Furthermore, "
        "these tables include "
        "special bar lines, repeat signs, first and second endings, irregular measure lengths, "
        "as well as the "
        "column 'next' which contains follow-up MCs for unfolding a score's repeat structure. For more "
        "information, "
        "please refer to the docs at https://ms3.readthedocs.io/columns",
        "notes": "One feature matrix per score, containing one row per note head. Not every row represents an "
        "onset because note heads may be tied together (see column 'tied'). "
        "For more information, please refer to the docs at https://ms3.readthedocs.io/columns",
    }
    title = titles[facet] if facet in titles else facet
    title += " for " + corpus
    description = descriptions[facet] if facet in descriptions else None
    jsonld = make_csvw_jsonld(
        title=title, columns=columns, urls=files, description=description
    )
    json_path = os.path.join(folder, "csv-metadata.json")
    with open(json_path, "w", encoding="utf-8") as f:
        print(json.dumps(jsonld, indent=2), file=f)
    return json_path


def make_continuous_offset_series(
    measures: pd.DataFrame,
    quarters: bool = True,
    negative_anacrusis: Optional[Fraction] = None,
    logger: Optional[logging.Logger | str] = None,
) -> pd.Series:
    """Accepts a measure table without 'quarterbeats' column and computes each MC's offset from the piece's beginning.
    Deal with voltas before passing the table.

    If you need an offset_dict and the measures already come with a 'quarterbeats' column, you can call
    :func:`make_offset_dict_from_measures`.

    Args:
        measures:
            A measures table with 'normal' RangeIndex containing the column 'act_durs' and one of
            'mc' or 'mc_playthrough' (if repeats were unfolded).
        quarters:
            By default, the continuous offsets are expressed in ♩. Pass false to leave them as fractions
            of a whole note.
        negative_anacrusis:
            By default, the first value is 0. If you pass a fraction here, the first value will be its negative and the
            second value will be 0.
        logger:

    Returns:
        Cumulative sum of the actual durations, shifted down by 1. Compared to the original DataFrame it has
        length + 2 because it adds the end value twice, once with the next index value, and once with the index 'end'.
        Otherwise the end value would be lost due to the shifting.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if "mc_playthrough" in measures.columns:
        act_durs = measures.set_index("mc_playthrough").act_dur
    elif "mc" in measures.columns:
        act_durs = measures.set_index("mc").act_dur
    else:
        logger.error(
            "Expected to have at least one column called 'mc' or 'mc_playthrough'."
        )
        return pd.Series()
    if quarters:
        act_durs = act_durs * 4
    res = act_durs.cumsum()
    last_val = res.iloc[-1]
    last_ix = res.index[-1] + 1
    res = res.shift(fill_value=0)
    ending = pd.Series([last_val, last_val], index=[last_ix, "end"])
    res = pd.concat([res, ending])
    if negative_anacrusis is not None:
        res -= abs(Fraction(negative_anacrusis))
    return res


def make_offset_dict_from_measures(
    measures: pd.DataFrame, all_endings: bool = False
) -> dict:
    """Turn a measure table that comes with a 'quarterbeats' column into a dictionary that maps MCs (measure counts)
    to their quarterbeat offset from the piece's beginning, used for computing quarterbeats for other facets.

    This function is used for the default case. If you need more options, e.g. an offset dict from unfolded
    measures or expressed in whole notes or with negative anacrusis, use
    :func:`make_continuous_offset_series` instead.

    Args:
      measures: Measures table containing a 'quarterbeats' column.
      all_endings: Uses the column 'quarterbeats_all_endings' of the measures table if it has one, otherwise
          falls back to the default 'quarterbeats'.

    Returns:
      {MC -> quarterbeat_offset}. Offsets are Fractions. If ``all_endings`` is not set to ``True``,
      values for MCs that are part of a first ending (or third or larger) are NA.
    """
    measures = measures.set_index("mc")
    if all_endings and "quarterbeats_all_endings" in measures.columns:
        col = "quarterbeats_all_endings"
    else:
        col = "quarterbeats"
    offset_dict = measures[col].to_dict()
    last_row = measures.iloc[-1]
    offset_dict["end"] = last_row[col] + 4 * last_row.act_dur
    return offset_dict


def make_id_tuples(key, n):
    """For a given key, this function returns index tuples in the form [(key, 0), ..., (key, n)]

    Returns
    -------
    list
        indices in the form [(key, 0), ..., (key, n)]

    """
    return list(zip(repeat(key), range(n)))


def make_interval_index_from_breaks(
    S, end_value=None, closed="left", name="interval", logger=None
):
    """Interpret a Series as interval breaks and make an IntervalIndex out of it.

    Parameters
    ----------
    S : :obj:`pandas.Series`
        Interval breaks. It is assumed that the breaks are sorted.
    end_value : numeric, optional
        Often you want to pass the right border of the last interval.
    closed : :obj:`str`, optional
        Defaults to 'left'. Argument passed to to :py:meth:`pandas.IntervalIndex.from_breaks`.
    name : :obj:`str`, optional
        Name of the created index. Defaults to 'interval'.

    Returns
    -------
    :obj:`pandas.IntervalIndex`

    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    breaks = S.to_list()
    if end_value is not None:
        last = breaks[-1]
        if end_value > last:
            breaks += [end_value]
        else:
            breaks += [last]
    try:
        iix = pd.IntervalIndex.from_breaks(breaks, closed=closed, name=name)
    except Exception as e:
        unsorted = [(a, b) for a, b in zip(breaks, breaks[1:]) if b < a]
        if len(unsorted) > 0:
            logger.error(f"Breaks are not sorted: {unsorted}")
        else:
            logger.error(
                f"Cannot create IntervalIndex from these breaks:\n{breaks}\nException: {e}"
            )
        raise
    return iix


def make_name_columns(df):
    """Relies on the columns ``localkey`` and ``globalkey`` to transform the columns ``root`` and ``bass_notes`` from
    scale degrees (expressed as fifths) to absolute note names, e.g. in C major: 0 => 'C', 7 => 'C#', -5 => 'Db'
    Uses: transform(), scale_degree2name"""
    new_cols = {}
    for col in ("root", "bass_note"):
        if col in df.columns:
            new_cols[f"{col}_name"] = transform(
                df, scale_degree2name, [col, "localkey", "globalkey"]
            )
    return pd.DataFrame(new_cols)


def make_playthrough2mc(measures: pd.DataFrame, logger=None) -> Optional[pd.Series]:
    """Turns the column 'next' into a mapping of playthrough_mc -> mc."""
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    ml = measures.set_index("mc")
    try:
        seq = next2sequence(ml.next, logger=logger)
        if seq is None:
            return
    except Exception as e:
        logger.warning(
            f"Computing unfolded sequence of MCs failed with:\n'{e}'",
            extra={"message_id": (26,)},
        )
        return
    mc_playthrough = pd.Series(seq, name="mc_playthrough", dtype="Int64")
    if len(mc_playthrough) == 0:
        pass
    elif seq[0] == 1:
        mc_playthrough.index += 1
    else:
        assert seq[0] == 0, f"The first mc should be 0 or 1, not {seq[0]}"
    return mc_playthrough


def make_playthrough_info(
    measures: pd.DataFrame, logger=None
) -> Optional[Union[pd.DataFrame, pd.Series]]:
    """Turns a measures table into a DataFrame or Series that can be passed as argument to :func:`unfold_repeats`.
    The return type is DataFrame if the unfolded measures table contains an 'mn_playthrough' column, otherwise it
    is equal to the result of :func:`make_playthrough2mc`. Hence, the purpose of the function is to add an
    'mn_playthrough' column to unfolded facets whenever possible.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    unfolded_measures = unfold_measures_table(measures, logger=logger)
    if unfolded_measures is None:
        return
    unfolded_measures = unfolded_measures.set_index("mc_playthrough")
    if "mn_playthrough" in unfolded_measures.columns:
        return unfolded_measures[["mc", "mn_playthrough"]]
    return unfolded_measures.mc


def map2elements(e, f, *args, **kwargs):
    """If `e` is an iterable, `f` is applied to all elements."""
    if isinstance(e, Iterable) and not isinstance(e, str):
        try:
            return e.__class__(map2elements(x, f, *args, **kwargs) for x in e)
        except TypeError:
            if isinstance(e, pd.Index):
                # e.g., if a numerical index is transformed to strings
                return pd.Index(map2elements(x, f, *args, **kwargs) for x in e)
    return f(e, *args, **kwargs)


def merge_ties(df, return_dropped=False, perform_checks=True, logger=None):
    """In a note list, merge tied notes to single events with accumulated durations.
        Input dataframe needs columns ['duration', 'tied', 'midi', 'staff']. This
        function does not handle correctly overlapping ties on the same pitch since
        it doesn't take into account the notational layers ('voice').


    Parameters
    ----------
    df
    return_dropped

    Returns
    -------

    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)

    def merge(df):
        vc = df.tied.value_counts()
        if vc[1] != 1 or vc[-1] != 1:
            logger.warning(f"More than one 1 or -1:\n{vc}")
        ix = df.iloc[0].name
        dur = df.duration.sum()
        drop = df.iloc[1:].index.to_list()
        return pd.Series({"ix": ix, "duration": dur, "dropped": drop})

    def merge_notes(staff_midi):
        staff_midi["chunks"] = (staff_midi.tied == 1).astype(int).cumsum()
        t = staff_midi.groupby("chunks", group_keys=False).apply(merge)
        return t.set_index("ix")

    if not df.tied.notna().any():
        return df
    df = df.copy()
    notna = df.loc[df.tied.notna(), ["duration", "tied", "midi", "staff"]]
    if perform_checks:
        before = notna.tied.value_counts()
    new_dur = (
        notna.groupby(["staff", "midi"], group_keys=False)
        .apply(merge_notes)
        .sort_index()
    )
    try:
        df.loc[new_dur.index, "duration"] = new_dur.duration
    except Exception:
        print(new_dur)
    if return_dropped:
        df.loc[new_dur.index, "dropped"] = new_dur.dropped
    df = df.drop(new_dur.dropped.sum())
    if perform_checks:
        after = df.tied.value_counts()
        assert (
            before[1] == after[1]
        ), f"Error while merging ties. Before:\n{before}\nAfter:\n{after}"
    return df


def merge_chords_and_notes(
    chords_table: pd.DataFrame, notes_table: pd.DataFrame
) -> pd.DataFrame:
    """Performs an outer join between a chords table and a notes table, based on the column 'chord_id'. If the chords
    come with an 'event' column, all chord events matched with at least one note will be renamed to 'Note'.
    Markup displayed in individual rows ('Dynamic', 'Spanner', 'StaffText', 'SystemText', 'Tempo', 'FiguredBass'),
    are/remain placed before the note(s) with the same onset.
    Markup showing up in a Chord event's row (e.g. a Spanner ID) will be duplicated for each note pertaining to that
                           chord,
    i.e., only for notes in the same staff and voice.

    Args:
      chords_table:
      notes_table:

    Returns:
      Merged DataFrame.
    """
    notes_columns = [
        "tied",
        "tpc",
        "midi",
        "name",
        "octave",
        "chord_id",
    ]  # 'gracenote', 'tremolo' would be contained in chords already
    present_columns = [col for col in notes_columns if col in notes_table]
    assert "chord_id" in present_columns, (
        f"Notes table does not come with a 'chord_id' column needed for merging: "
        f"{notes_table.columns}"
    )
    notes = notes_table[present_columns].astype({"chord_id": "Int64"})
    amend_events = "event" in chords_table
    merged = pd.merge(
        chords_table, notes, on="chord_id", how="outer", indicator=amend_events
    )
    merged = sort_note_list(merged)
    if amend_events:
        matches_mask = merged._merge == "both"
        merged.loc[matches_mask, "event"] = "Note"
        merged.drop(columns="_merge", inplace=True)
    return merged


def metadata2series(metadata: dict) -> pd.Series:
    """Turns a metadata dict into a pd.Series() (for storing in a DataFrame)
    Uses: ambitus2oneliner(), dict2oneliner(), parts_info()

    Returns
    -------
    :obj:`pandas.Series`
        A series allowing for storing metadata as a row of a DataFrame.
    """
    metadata = dict(metadata)
    metadata["TimeSig"] = dict2oneliner(metadata["TimeSig"])
    metadata["KeySig"] = dict2oneliner(metadata["KeySig"])
    if "ambitus" in metadata:
        metadata["ambitus"] = ambitus2oneliner(metadata["ambitus"])
    if "parts" in metadata:
        metadata.update(parts_info(metadata["parts"]))
        del metadata["parts"]
    s = pd.Series(metadata)
    return s


@overload
def midi_and_tpc2octave(midi: int, tpc: int) -> int: ...


@overload
def midi_and_tpc2octave(midi: pd.Series, tpc: pd.Series) -> pd.Series: ...


@overload
def midi_and_tpc2octave(midi: NDArray[int], tpc: NDArray[int]) -> NDArray[int]: ...


@overload
def midi_and_tpc2octave(midi: List[int], tpc: List[int]) -> List[int]: ...


@overload
def midi_and_tpc2octave(midi: Tuple[int], tpc: Tuple[int]) -> Tuple[int]: ...


def midi_and_tpc2octave(
    midi: Union[int, pd.Series, NDArray[int], List[int], Tuple[int]],
    tpc: Union[int, pd.Series, NDArray[int], List[int], Tuple[int]],
) -> Union[int, pd.Series, NDArray[int], List[int], Tuple[int]]:
    try:
        midi = int(float(midi))
    except TypeError:
        try:
            # if numpy array or Pandas Series, compute vectorized, otherwise iterate
            midi.dtype
        except AttributeError:
            return midi.__class__(midi_and_tpc2octave(m, t) for m, t in zip(midi, tpc))
    acc = tpc // 7
    return (midi - acc) // 12 - 1


@overload
def midi2octave(midi: int, fifths: Optional[int]) -> int: ...


@overload
def midi2octave(midi: pd.Series, fifths: Optional[pd.Series]) -> pd.Series: ...


@overload
def midi2octave(midi: NDArray[int], fifths: Optional[NDArray]) -> NDArray[int]: ...


@overload
def midi2octave(midi: List[int], fifths: Optional[List[int]]) -> List[int]: ...


@overload
def midi2octave(midi: Tuple[int], fifths: Optional[Tuple[int]]) -> Tuple[int]: ...


def midi2octave(
    midi: Union[int, pd.Series, NDArray[int], List[int], Tuple[int]],
    fifths: Optional[Union[int, pd.Series, NDArray[int], List[int], Tuple[int]]] = None,
) -> Union[int, pd.Series, NDArray[int], List[int], Tuple[int]]:
    """For a given MIDI pitch, calculate the octave. Middle octave = 4
        Uses: midi_and_tpc2octave(), map2elements()

    Parameters
    ----------
    midi : :obj:`int`
        MIDI pitch (positive integer)
    fifths : :obj:`int`, optional
        To be precise, for some Tonal Pitch Classes, the octave deviates
        from the simple formula ``MIDI // 12 - 1``, e.g. for B# or Cb.
    """
    if fifths is not None:
        return midi_and_tpc2octave(midi, fifths)
    try:
        midi = int(float(midi))
    except TypeError:
        try:
            # if numpy array or Pandas Series, compute vectorized, otherwise iterate
            midi.dtype
        except AttributeError:
            return map2elements(midi, midi2octave)
    return midi // 12 - 1


def midi2name(midi):
    try:
        midi = int(float(midi))
    except Exception:
        if isinstance(midi, pd.Series):
            return transform(midi, midi2name)
        if isinstance(midi, Iterable):
            return map2elements(midi, midi2name)
        return midi
    names = {
        0: "C",
        1: "C#/Db",
        2: "D",
        3: "D#/Eb",
        4: "E",
        5: "F",
        6: "F#/Gb",
        7: "G",
        8: "G#/Ab",
        9: "A",
        10: "A#/Bb",
        11: "B",
    }
    return names[midi % 12]


def mn2int(mn_series):
    """Turn a series of measure numbers parsed as strings into two integer columns 'mn' and 'volta'."""
    try:
        split = mn_series.fillna("").str.extract(r"(?P<mn>\d+)(?P<volta>[a-g])?")
    except Exception:
        mn_series = pd.DataFrame(mn_series, columns=["mn", "volta"])
        try:
            return mn_series.astype("Int64")
        except Exception:
            return mn_series
    split.mn = pd.to_numeric(split.mn)
    split.volta = pd.to_numeric(
        split.volta.map({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
    )
    return split.astype("Int64")


def name2format(df, format="html", name_col="color_name"):
    """Converts a column with CSS3 names into 'html', 'rgb', or  'rgba'."""
    if format == "html":
        return df[name_col].map(color_name2html)
    if format == "rgb":
        return df[name_col].map(color_name2rgb)
    if format == "rgba":
        return df[name_col].map(color_name2rgba)


def name2fifths(nn, logger=None):
    """Turn a note name such as `Ab` into a tonal pitch class, such that -1=F, 0=C, 1=G etc.
    Uses: split_note_name()
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if nn.__class__ == int or pd.isnull(nn):
        return nn
    name_tpcs = {"C": 0, "D": 2, "E": 4, "F": -1, "G": 1, "A": 3, "B": 5}
    accidentals, note_name = split_note_name(nn, count=True, logger=logger)
    if note_name is None:
        return None
    step_tpc = name_tpcs[note_name.upper()]
    return step_tpc + 7 * accidentals


def name2pc(nn, logger=None):
    """Turn a note name such as `Ab` into a tonal pitch class, such that -1=F, 0=C, 1=G etc.
    Uses: split_note_name()
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if nn.__class__ == int or pd.isnull(nn):
        logger.warning(f"'{nn}' is not a valid note name.")
        return nn
    name_tpcs = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    accidentals, note_name = split_note_name(nn, count=True, logger=logger)
    if note_name is None:
        return None
    step_pc = name_tpcs[note_name.upper()]
    return (step_pc + accidentals) % 12


def nan_eq(a, b):
    """Returns True if a and b are equal or both null. Works on two Series or two elements."""
    return (a == b).fillna(False) | (pd.isnull(a) & pd.isnull(b))


def next2sequence(next_col: pd.Series, logger=None) -> Optional[List[int]]:
    """Turns a 'next' column into the correct sequence of MCs corresponding to unfolded repetitions.
    Requires that the Series' index be the MCs as in ``measures.set_index('mc').next``.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    mc = next_col.index[0]
    last_mc = next_col.index[-1]
    max_iter = 10 * last_mc
    i = 0
    result = []
    nxt = next_col.to_dict()
    while mc != -1 and i < max_iter:
        if mc not in nxt:
            logger.error(
                f"Column 'next' contains MC {mc} which the pieces does not have.",
                extra={"message_id": (26,)},
            )
            return
        result.append(mc)
        new_mc, *rest = nxt[mc]
        if len(rest) > 0:
            nxt[mc] = rest
        mc = new_mc
        i += 1
    if i == max_iter:
        return []
    return result


def no_collections_no_booleans(
    df: pd.DataFrame,
    collection_columns: Optional[Collection[str]] = None,
    boolean_columns: Optional[Collection[str]] = None,
    logger=None,
):
    """
    Cleans the DataFrame columns ['next', 'chord_tones', 'added_tones', 'volta_mcs] from tuples and the columns
    ['globalkey_is_minor', 'localkey_is_minor'] from booleans, converting them all to integers

    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if df is None:
        return df
    collection_columns = (
        list(COLLECTION_COLUMNS)
        if collection_columns is None
        else list(collection_columns)
    )
    boolean_columns = (
        list(BOOLEAN_COLUMNS) if boolean_columns is None else list(boolean_columns)
    )
    try:
        coll_cols = [c for c in collection_columns if c in df.columns]
    except Exception:
        logger.error(f"df needs to be a DataFrame, not a {df.__class__}.")
        return df
    df = df.copy()
    for c in coll_cols:
        null_vals = df[c].isna()
        if null_vals.all():
            continue
        df.loc[null_vals, c] = pd.NA
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
            df.loc[:, c] = transform(df[c], iterable2str)
        logger.debug(f"Transformed iterables in the column {c} to strings.")
    # df.loc[:, cc] = transform(df[cc], iterable2str, column_wise=True)
    bool_cols = [c for c in boolean_columns if c in df.columns]
    for bc in bool_cols:
        null_vals = df[bc].isna()
        if null_vals.all():
            continue
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
            # try:
            numeric_column = pd.to_numeric(df[bc], errors="coerce")
            df[bc] = numeric_column.astype("Int64")
            # except TypeError:
            #     logger.warning(
            #         f"Could not convert column {bc} to boolean. It contains values other than True, False, and NaN:\n"
            #         f"{df[bc].unique()}"
            #     )
            #     raise
        logger.debug(f"Transformed booleans in the column {bc} to integers.")
    if "TimeSig" in df.columns:
        df.TimeSig = df.TimeSig.map(dict2oneliner)
    if "KeySig" in df.columns:
        df.KeySig = df.KeySig.map(dict2oneliner)
    return df


def ordinal_suffix(n):
    suffixes = {"1": "st", "2": "nd", "3": "rd"}
    last_digit = str(n)[-1]
    return suffixes.get(last_digit, "th")


def parts_info(d):
    """
    Turns a (nested) ``metadata['parts']`` dict into a flat dict based on staves.

    Example
    -------
    >>> d = s.mscx.metadata
    >>> parts_info(d['parts'])
    {'staff_1_instrument': 'Voice',
     'staff_1_ambitus': '66-76 (F#4-E5)',
     'staff_2_instrument': 'Voice',
     'staff_2_ambitus': '55-69 (G3-A4)',
     'staff_3_instrument': 'Voice',
     'staff_3_ambitus': '48-67 (C3-G4)',
     'staff_4_instrument': 'Voice',
     'staff_4_ambitus': '41-60 (F2-C4)'}
    """
    res = {}
    for part_dict in d.values():
        for id in part_dict["staves"]:
            name = f"staff_{id}"
            res[f"{name}_instrument"] = part_dict["instrument"]
            amb_name = name + "_ambitus"
            res[amb_name] = ambitus2oneliner(part_dict[amb_name])
    return res


def path2type(path, logger=None):
    """Determine a file's type by scanning its path for default components in the constant STANDARD_NAMES.

    Parameters
    ----------
    path

    Returns
    -------

    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    _, fext = os.path.splitext(path)
    if fext.lower() in SCORE_EXTENSIONS:
        logger.debug(
            f"Categorized {path} as score based on the file extension {fext!r}."
        )
        return "scores"
    component2type = path_component2file_type_map()

    def find_components(s):
        res = [comp for comp in component2type.keys() if comp in s]
        return res, len(res)

    if os.path.isfile(path):
        # give preference to folder names before file names
        directory, piece_name = os.path.split(path)
        if "metadata" in piece_name:
            logger.debug(
                f"Categorized {path} as metadata based on the filename {piece_name!r}."
            )
            return "metadata"
        found_components, n_found = find_components(directory)
        if n_found == 0:
            found_components, n_found = find_components(piece_name)
    else:
        found_components, n_found = find_components(path)
    if n_found == 0:
        logger.debug(
            f"Type could not be inferred from path '{path}'. Letting it default to 'labels'."
        )
        return "labels"
    if n_found == 1:
        typ = component2type[found_components[0]]
        logger.debug(
            f"Categorized {path} as {typ} based on the component {found_components[0]!r}."
        )
        return typ
    else:
        for path_component in reversed(directory.split(os.sep)):
            for comp in component2type.keys():
                if comp in path_component:
                    typ = component2type[comp]
                    logger.debug(
                        f"Multiple components ({', '.join(found_components)}) found in path '{path}'; opted for the "
                        f"last one: {typ}"
                    )
                    return typ
        logger.warning(
            f"Components {', '.join(found_components)} found in path '{path}', but not in one of its constituents."
        )
        return "other"


@cache
def path_component2file_type_map() -> dict:
    comp2type = {comp: comp for comp in STANDARD_NAMES}
    comp2type["MS3"] = "scores"
    comp2type["harmonies"] = "expanded"
    comp2type["output"] = "labels"
    comp2type["infer"] = "labels"
    return comp2type


def file_type2path_component_map() -> dict:
    comp2type = path_component2file_type_map()
    type2comps = defaultdict(list)
    for comp, typ in comp2type.items():
        type2comps[typ].append(comp)
    return dict(type2comps)


def pretty_dict(
    ugly_dict: dict, heading_key: str = None, heading_value: str = None
) -> str:
    """Turns a dictionary into a string where the keys are printed in a column, separated by '->'."""
    if heading_key is not None or heading_value is not None:
        head_key = "KEY" if heading_key is None else heading_key
        head_val = "" if heading_value is None else heading_value
        head_val_length = len(head_val) + 4
        d = {head_key: head_val}
        d.update(ugly_dict)
    else:
        head_val_length = -1
        try:
            d = dict(ugly_dict)
        except ValueError:
            print(f"Please pass a dictionary, not a {type(ugly_dict)}: {ugly_dict}")
            raise
    left = max(len(str(k)) for k in d.keys())
    res = []
    for k, v in d.items():
        ks = str(k)
        if isinstance(v, pd.DataFrame) or isinstance(v, pd.Series):
            vs = v.to_string()
        else:
            vs = str(v)
        if "\n" in vs:
            lines = vs.split("\n")
            res.extend(
                [f"{ks if i == 0 else '':{left}} -> {l}" for i, l in enumerate(lines)]
            )
        else:
            res.append(f"{ks:{left}} -> {vs}")
    if head_val_length > -1:
        res.insert(1, "-" * (left + head_val_length))
    return "\n".join(res)


def resolve_dir(d):
    """Resolves '~' to HOME directory and turns ``d`` into an absolute path."""
    if d is None:
        return None
    d = str(d)
    if "~" in d:
        return os.path.expanduser(d)
    return os.path.abspath(d)


def rgb2format(df, format="html", r_col="color_r", g_col="color_g", b_col="color_b"):
    """Converts three RGB columns into a color_html or color_name column."""
    cols = [r_col, g_col, b_col]
    if format == "html":
        html = list(map(rgb_tuple2html, df[cols].itertuples(index=False, name=None)))
        return pd.Series(html, index=df.index).rename("color_html")
    if format == "name":
        names = list(map(rgb_tuple2name, df[cols].itertuples(index=False, name=None)))
        return pd.Series(names, index=df.index).rename("color_name")


def rgb_tuple2format(t, format="html"):
    """Converts a single RGB tuple into 'HTML' or 'name'."""
    if pd.isnull(t):
        return t
    if pd.isnull(t[0]):
        return t[0]
    norm = webcolors.normalize_integer_triplet(tuple(int(i) for i in t))
    if format == "html":
        return webcolors.rgb_to_hex(norm)
    if format == "name":
        try:
            return webcolors.rgb_to_name(norm)
        except Exception:
            try:
                return MS3_RGB[norm]
            except Exception:
                return webcolors.rgb_to_hex(norm)


def rgb_tuple2html(t):
    """Converts a single RGB tuple into HTML."""
    return rgb_tuple2format(t, format="html")


def rgb_tuple2name(t):
    """Converts a single RGB tuple into its CSS3 name or to HTML if there is none."""
    return rgb_tuple2format(t, format="name")


def rgba2attrs(named_tuple):
    return {k: str(v) for k, v in named_tuple._asdict().items()}


def rgba2params(named_tuple):
    attrs = rgba2attrs(named_tuple)
    return {"color_" + k: v for k, v in attrs.items()}


def roman_numeral2fifths(rn, global_minor=False, logger=None):
    """Turn a Roman numeral into a TPC interval (e.g. for transposition purposes).
    Uses: split_scale_degree()
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if pd.isnull(rn):
        return rn
    if "/" in rn:
        resolved = resolve_relative_keys(rn, global_minor)
        mode = "minor" if global_minor else "major"
        logger.debug(f"Relative numeral {rn} in {mode} mode resolved to {resolved}.")
        rn = resolved
    rn_tpcs_maj = {"I": 0, "II": 2, "III": 4, "IV": -1, "V": 1, "VI": 3, "VII": 5}
    rn_tpcs_min = {"I": 0, "II": 2, "III": -3, "IV": -1, "V": 1, "VI": -4, "VII": -2}
    accidentals, rn_step = split_scale_degree(rn, count=True, logger=logger)
    if any(v is None for v in (accidentals, rn_step)):
        return None
    rn_step = rn_step.upper()
    step_tpc = rn_tpcs_min[rn_step] if global_minor else rn_tpcs_maj[rn_step]
    return step_tpc + 7 * accidentals


def roman_numeral2semitones(rn, global_minor=False, logger=None):
    """Turn a Roman numeral into a semitone distance from the root (0-11).
    Uses: split_scale_degree()
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if pd.isnull(rn):
        return rn
    if "/" in rn:
        resolved = resolve_relative_keys(rn, global_minor)
        mode = "minor" if global_minor else "major"
        logger.debug(f"Relative numeral {rn} in {mode} mode resolved to {resolved}.")
        rn = resolved
    rn_tpcs_maj = {"I": 0, "II": 2, "III": 4, "IV": 5, "V": 7, "VI": 9, "VII": 11}
    rn_tpcs_min = {"I": 0, "II": 2, "III": 3, "IV": 5, "V": 7, "VI": 8, "VII": 10}
    accidentals, rn_step = split_scale_degree(rn, count=True, logger=logger)
    if any(v is None for v in (accidentals, rn_step)):
        return None
    rn_step = rn_step.upper()
    step_tpc = rn_tpcs_min[rn_step] if global_minor else rn_tpcs_maj[rn_step]
    return step_tpc + accidentals


@overload
def scale_degree2name(fifths: int, localkey: str, globalkey: str) -> str: ...


@overload
def scale_degree2name(
    fifths: pd.Series, localkey: str, globalkey: str
) -> pd.Series: ...


@overload
def scale_degree2name(
    fifths: NDArray[int], localkey: str, globalkey: str
) -> NDArray[str]: ...


@overload
def scale_degree2name(
    fifths: List[int], localkey: str, globalkey: str
) -> List[str]: ...


@overload
def scale_degree2name(
    fifths: Tuple[int], localkey: str, globalkey: str
) -> Tuple[str]: ...


def scale_degree2name(
    fifths: Union[int, pd.Series, NDArray[int], List[int], Tuple[int]],
    localkey: str,
    globalkey: str,
) -> Union[str, pd.Series, NDArray[str], List[str], Tuple[str]]:
    """For example, scale degree -1 (fifths, i.e. the subdominant) of the localkey of 'VI' within 'e' minor is 'F'.

    Args:
        fifths: Scale degree expressed as distance from the tonic in fifths.
        localkey: Local key in which the scale degree is situated, as Roman numeral (can include slash notation such
        as V/ii).
        globalkey: Global key as a note name. E.g. `Ab` for Ab major, or 'c#' for C# minor.

    Returns:
        The given scale degree(s), expressed as a note name(s).
    """
    try:
        if any(pd.isnull(val) for val in (fifths, localkey, globalkey)):
            return fifths
    except ValueError:
        pass
    if isinstance(fifths, pd.Series):
        return cast2collection(
            coll=fifths, func=scale_degree2name, localkey=localkey, globalkey=globalkey
        )
    try:
        fifths = int(float(fifths))
    except TypeError:
        return cast2collection(
            coll=fifths, func=scale_degree2name, localkey=localkey, globalkey=globalkey
        )
    global_minor = globalkey.islower()
    if "/" in localkey:
        localkey = resolve_relative_keys(localkey, global_minor)
    lk_fifths = roman_numeral2fifths(localkey, global_minor)
    gk_fifths = name2fifths(globalkey)
    sd_transposed = transpose(fifths, lk_fifths + gk_fifths)
    return fifths2name(sd_transposed)


def scan_directory(
    directory: str,
    file_re: str = r".*",
    folder_re: str = r".*",
    exclude_re: str = r"^(\.|_)",
    recursive: bool = True,
    subdirs: bool = False,
    progress: bool = False,
    exclude_files_only: bool = False,
    return_metadata: bool = False,
    logger=None,
) -> Iterator[Union[str, Tuple[str, str]]]:
    """Generator of filtered file paths in ``directory``.

    Args:
      directory: Directory to be scanned for files.
      file_re, folder_re:
          Regular expressions for filtering certain file names or folder names.
          The regEx are checked with search(), not match(), allowing for fuzzy search.
      exclude_re:
          Exclude files and folders (unless ``exclude_files_only=True``) containing this regular expression.
      recursive: By default, sub-directories are recursively scanned. Pass False to scan only ``dir``.
      subdirs: By default, full file paths are returned. Pass True to return (path, name) tuples instead.
      progress: Pass True to display the progress (useful for large directories).
      exclude_files_only:
          By default, ``exclude_re`` excludes files and folder. Pass True to exclude only files matching the regEx.
      return_metadata:
          If set to True, 'metadata.tsv' are always yielded regardless of ``file_re``.

    Yields:
      Full file path or, if ``subdirs=True``, (path, file_name) pairs in random order.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if file_re is None:
        file_re = r".*"
    if folder_re is None:
        folder_re = r".*"

    def traverse(d):
        nonlocal counter

        def check_regex(reg, s, excl=exclude_re):
            try:
                res = re.search(reg, s) is not None and re.search(excl, s) is None
            except Exception:
                print(reg)
                raise
            return res

        for dir_entry in os.scandir(d):
            name = dir_entry.name
            path = os.path.join(d, name)
            if dir_entry.is_dir() and (recursive or folder_re != ".*"):
                for res in traverse(path):
                    yield res
            else:
                if pbar is not None:
                    pbar.update()
                if folder_re == ".*":
                    folder_passes = True
                else:
                    folder_path = os.path.dirname(path)
                    if recursive:
                        folder_passes = check_regex(
                            folder_re, folder_path, excl="^$"
                        )  # passes if the folder path matches the regex
                    else:
                        folder = os.path.basename(folder_path)
                        folder_passes = check_regex(
                            folder_re, folder, excl="^$"
                        )  # passes if the folder name itself matches the regex
                    if (
                        folder_passes and not exclude_files_only
                    ):  # True if the exclude_re should also exclude folder
                        # names
                        folder_passes = check_regex(
                            folder_re, folder_path
                        )  # is false if any part of the folder path matches
                        # exclude_re
                if (
                    dir_entry.is_file()
                    and folder_passes
                    and (
                        check_regex(file_re, name)
                        or (return_metadata and name == "metadata.tsv")
                    )
                ):
                    counter += 1
                    if pbar is not None:
                        pbar.set_postfix({"selected": counter})
                    if subdirs:
                        yield (d, name)
                    else:
                        yield path

    if exclude_re is None or exclude_re == "":
        exclude_re = "^$"
    directory = resolve_dir(directory)
    counter = 0
    if not os.path.isdir(directory):
        logger.error("Not an existing directory: " + directory)
        return iter([])
    pbar = tqdm(desc="Scanning files", unit=" files") if progress else None
    return traverse(directory)


def column_order(df, first_cols=None, sort=True):
    """Sort DataFrame columns so that they start with the order of ``first_cols``, followed by those not included."""
    if first_cols is None:
        first_cols = STANDARD_COLUMN_ORDER
    cols = df.columns
    remaining = [col for col in cols if col not in first_cols]
    if sort:
        # Problem: string sort orders staff_1 staff_10 staff_11 ... and only then staff_2
        remaining = sorted(remaining)
    column_order = [col for col in first_cols if col in cols] + remaining
    return df[column_order]


def sort_note_list(
    df, mc_col="mc", mc_onset_col="mc_onset", midi_col="midi", duration_col="duration"
):
    """Sort every measure (MC) by ['mc_onset', 'midi', 'duration'] while leaving gracenotes' order (duration=0) intact.

    Parameters
    ----------
    df
    mc_col
    mc_onset_col
    midi_col
    duration_col

    Returns
    -------

    """
    df = df.copy()
    is_grace = df[duration_col] == 0
    grace_ix = {
        k: v.to_numpy()
        for k, v in df[is_grace].groupby([mc_col, mc_onset_col]).groups.items()
    }
    has_nan = df[midi_col].isna().any()
    if has_nan:
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
            df.loc[:, midi_col] = df[midi_col].fillna(1000)
    normal_ix = (
        df.loc[~is_grace, [mc_col, mc_onset_col, midi_col, duration_col]]
        .groupby([mc_col, mc_onset_col])
        .apply(
            lambda gr: gr.index[
                np.lexsort((gr.values[:, 3], gr.values[:, 2]))
            ].to_numpy()
        )
    )
    sorted_ixs = [
        np.concatenate((grace_ix[onset], ix)) if onset in grace_ix else ix
        for onset, ix in normal_ix.items()
    ]
    df = df.reindex(np.concatenate(sorted_ixs)).reset_index(drop=True)
    if has_nan:
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
            df.loc[:, midi_col] = df[midi_col].replace({1000: pd.NA}).astype("Int64")
    return df


def sort_tpcs(tpcs, ascending=True, start=None):
    """Sort tonal pitch classes by order on the piano.
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


def split_alternatives(
    df,
    column="label",
    regex=r"-(?!(\d|b+\d|\#+\d))",
    max=2,
    inplace=False,
    alternatives_only=False,
    logger=None,
):
    """
    Splits labels that come with an alternative separated by '-' and adds
    a new column. Only one alternative is taken into account. `df` is
    mutated inplace.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        Dataframe where one column contains DCML chord labels.
    column : :obj:`str`, optional
        Name of the column that holds the harmony labels.
    regex : :obj:`str`, optional
        The regular expression (or simple string) that detects the character combination used to separate alternative
        annotations.
        By default, alternatives are separated by a '-' that does not precede a scale degree such as 'b6' or '3'.
    max : :obj:`int`, optional
        Maximum number of admitted alternatives, defaults to 2.
    inplace : :obj:`bool`, optional
        Pass True if you want to mutate ``df``.
    alternatives_only : :obj:`bool`, optional
        By default the alternatives are added to the original DataFrame (``inplace`` or not).
        Pass True if you just need the split alternatives.

    Example
    -------
    >>> import pandas as pd
    >>> labels = pd.read_csv('labels.csv')
    >>> split_alternatives(labels, inplace=True)
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if not inplace:
        df = df.copy()
    alternatives = df[column].str.split(regex, expand=True)
    alternatives.dropna(axis=1, how="all", inplace=True)
    alternatives.columns = range(alternatives.shape[1])
    if alternatives_only:
        columns = [column] + [
            f"alt_{column}" if i == 1 else f"alt{i}_{column}"
            for i in alternatives.columns[1:]
        ]
        alternatives.columns = columns
        return alternatives.iloc[:, :max]
    if len(alternatives.columns) > 1:
        logger.debug("Labels split into alternatives.")
        df.loc[:, column] = alternatives[0]
        position = df.columns.get_loc(column) + 1
        for i in alternatives.columns[1:]:
            if i == max:
                break
            alt_name = f"alt_{column}" if i == 1 else f"alt{i}_{column}"
            df.insert(
                position, alt_name, alternatives[i].fillna(pd.NA)
            )  # replace None by NA
            position += 1
        if len(alternatives.columns) > max:
            logger.warning(
                f"More than {max} alternatives are not taken into account:\n{alternatives[alternatives[2].notna()]}"
            )
    else:
        logger.debug("Contains no alternative labels.")
    if not inplace:
        return df


def split_note_name(nn, count=False, logger=None):
    """Splits a note name such as 'Ab' into accidentals and name.

    nn : :obj:`str`
        Note name.
    count : :obj:`bool`, optional
        Pass True to get the accidentals as integer rather than as string.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    m = re.match("^([A-G]|[a-g])(#*|b*)$", str(nn))
    if m is None:
        logger.error(nn + " is not a valid scale degree.")
        return None, None
    note_name, accidentals = m.group(1), m.group(2)
    if count:
        accidentals = accidentals.count("#") - accidentals.count("b")
    return accidentals, note_name


def split_scale_degree(
    sd, count=False, logger=None
) -> Tuple[Optional[int], Optional[str]]:
    """Splits a scale degree such as 'bbVI' or 'b6' into accidentals and numeral.

    sd : :obj:`str`
        Scale degree.
    count : :obj:`bool`, optional
        Pass True to get the accidentals as integer rather than as string.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    m = re.match(r"^(#*|b*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|\d)$", str(sd))
    if m is None:
        if "/" in sd:
            logger.error(
                f"{sd} needs to be resolved, which requires information about the mode of the local key. "
                f"You can use ms3.utils.resolve_relative_keys(scale_degree, is_minor_context)."
            )
        else:
            logger.error(f"{sd} is not a valid scale degree.")
        return None, None
    acc, num = m.group(1), m.group(2)
    if count:
        acc = acc.count("#") - acc.count("b")
    return acc, num


# def chunkstring(string, length=80):
#     """ Generate chunks of a given length """
#     string = str(string)
#     return (string[0 + i:length + i] for i in range(0, len(string), length))
#
#
# def string2lines(string, length=80):
#     """ Use chunkstring() and make chunks into lines. """
#     return '\n'.join(chunkstring(string, length))


def test_binary(command, logger=None):
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if command is None:
        return command
    if os.path.isfile(command):
        logger.debug(f"Found MuseScore binary: {command}")
        return command
    if which(command) is None:
        logger.warning(
            f"MuseScore binary not found and not an installed command: {command}"
        )
        return None
    else:
        logger.debug(f"Found MuseScore command: {command}")
        return command


def transform(df, func, param2col=None, column_wise=False, **kwargs):
    """Compute a function for every row of a DataFrame, using several cols as arguments.
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
                apply_cols = [
                    col for col in df.columns if col not in param2col.values()
                ]
                assert len(var_arg) < 2, (
                    f"Name only one variable keyword argument as which {apply_cols} are used "
                    f"{'argument': None}."
                )
                var_arg = (
                    var_arg[0] if len(var_arg) > 0 else getfullargspec(func).args[0]
                )
                param2col = {k: v for k, v in param2col.items() if v is not None}
                result_cols = {
                    col: transform(
                        df, func, dict({var_arg: col}, **param2col), **kwargs
                    )
                    for col in apply_cols
                }
            else:
                apply_cols = [col for col in df.columns if col not in param2col]
                result_cols = {
                    col: transform(df, func, [col] + param2col, **kwargs)
                    for col in apply_cols
                }
            return pd.DataFrame(result_cols, index=df.index)

    if param2col.__class__ == dict:
        param_tuples = list(df[param2col.values()].itertuples(index=False, name=None))
        result_dict = {
            t: func(**{a: b for a, b in zip(param2col.keys(), t)}, **kwargs)
            for t in set(param_tuples)
        }
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
                param_tuples = list(
                    df[list(param2col)].itertuples(index=False, name=None)
                )
            result_dict = {t: func(*t, **kwargs) for t in set(param_tuples)}
    with warnings.catch_warnings():
        # pandas developers doing their most annoying thing >:(
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=(".*The default dtype for empty Series.*"),
        )
        res = pd.Series([result_dict[t] for t in param_tuples], index=df.index)
    return res


def adjacency_groups(
    S: pd.Series,
    na_values: Optional[str] = "group",
    prevent_merge: bool = False,
    logger=None,
) -> Tuple[pd.Series, Dict[int, Any]]:
    """Turns a Series into a Series of ascending integers starting from 1 that reflect groups of successive
    equal values. There are several options of how to deal with NA values.

    Args:
      S: Series in which to group identical adjacent values with each other.
      na_values:
          | 'group' creates individual groups for NA values (default).
          | 'backfill' or 'bfill' groups NA values with the subsequent group
          | 'pad', 'ffill' groups NA values with the preceding group
          | Any other string works like 'group', with the difference that the groups will be named with this value.
          | Passing None means NA values & ranges are being ignored, i.e. they will also be present in the output and
          the
            subsequent value will be based on the preceding value.
      prevent_merge:
          By default, if you use the `na_values` argument to fill NA values, they might lead to two groups merging.
          Pass True to prevent this. For example, take the sequence ['a', NA, 'a'] with ``na_values='ffill'``: By
          default,
          it will be merged to one single group ``[1, 1, 1], {1: 'a'}``. However, passing ``prevent_merge=True`` will
          result in ``[1, 1, 2], {1: 'a', 2: 'a'}``.

    Returns:
      A series with increasing integers that can be used for grouping.
      A dictionary mapping the integers to the grouped values.

    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    reindex_flag = False
    # reindex is set to True in cases where NA values are being excluded from the operation and restored afterwards
    if prevent_merge:
        forced_beginnings = S.notna() & ~S.notna().shift().fillna(False)
    if na_values is None:
        if S.isna().any():
            s = S.dropna()
            reindex_flag = True
        else:
            s = S
    elif na_values == "group":
        s = S
    elif na_values in ("backfill", "bfill"):
        s = S.bfill()
    elif na_values in ("pad", "ffill"):
        s = S.ffill()
    else:
        s = S.fillna(value=na_values)

    if s.isna().any():
        if na_values == "group":
            shifted = s.shift()
            beginnings = ~nan_eq(s, shifted)
        else:
            logger.warning(
                f"After treating the Series '{S.name}' with na_values='{na_values}', "
                f"there were still {s.isna().sum()} NA values left."
            )
            s = s.dropna()
            beginnings = (s != s.shift()).fillna(False)
            reindex_flag = True
    else:
        beginnings = s != s.shift()
    beginnings.iat[0] = True
    if prevent_merge:
        beginnings |= forced_beginnings
    groups = beginnings.cumsum()
    names = dict(enumerate(s[beginnings], 1))
    if reindex_flag:
        groups = groups.reindex(S.index)
    try:
        return pd.to_numeric(groups).astype("Int64"), names
    except TypeError:
        logger.warning(f"Erroneous outcome while computing adjacency groups: {groups}")
        return groups, names


def unfold_measures_table(
    measures: pd.DataFrame, logger=None
) -> Optional[pd.DataFrame]:
    """Returns a copy of a measures table that corresponds through a succession of MCs when playing all repeats.
    To distinguish between repeated MCs and MNs, it adds the continues column 'mc_playthrough' (starting at 1) and
    'mn_playthrough' which contains the values of 'mn' as string with letters {'a', 'b', ...} appended.

    Args:
      measures: Measures table with columns ['mc', 'next', 'dont_count']

    Returns:

    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if "mc_playthrough" in measures:
        logger.info(
            "Received a dataframe with the column 'mc_playthrough' that is already unfolded. Returning as is."
        )
        return measures
    playthrough2mc = make_playthrough2mc(measures, logger=logger)
    if playthrough2mc is None or len(playthrough2mc) == 0:
        logger.warning(
            f"Error in the repeat structure: Did not reach the stopping value -1 in measures.next:\n"
            f"{measures.set_index('mc').next}"
        )
        return None
    else:
        logger.debug("Repeat structure successfully unfolded.")
    unfolded_measures = unfold_repeats(measures, playthrough2mc, logger=logger)
    try:
        mn_playthrough_col = compute_mn_playthrough(unfolded_measures, logger=logger)
        insert_position = unfolded_measures.columns.get_loc("mc_playthrough") + 1
        unfolded_measures.insert(insert_position, "mn_playthrough", mn_playthrough_col)
    except Exception as e:
        logger.warning(
            f"Adding the column 'mn_playthrough' to the unfolded measures table failed with:\n'{e}'"
        )
    logger.debug("Measures successfully unfolded.")
    return unfolded_measures


def unfold_repeats(
    df: pd.DataFrame, playthrough_info: Union[pd.Series, pd.DataFrame], logger=None
) -> pd.DataFrame:
    """Use a succesion of MCs to bring a DataFrame in this succession. MCs may repeat.

    Args:
      df: DataFrame needs to have the columns 'mc'. If 'mn' is present, the column 'mn' will be added, too.
      playthrough2mc:
          A Series of the format ``{mc_playthrough: mc}`` where ``mc_playthrough`` corresponds
          to continuous MC

    Returns:
      A copy of the dataframe with the columns 'mc_playthrough' and 'mn_playthrough' (if 'mn' is present) inserted.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    n_occurrences = df.mc.value_counts()
    result_df = df.set_index("mc")
    if isinstance(playthrough_info, pd.DataFrame):
        playthrough2mc = playthrough_info["mc"]
        playthrough2mn = playthrough_info["mn_playthrough"].to_dict()
    else:
        playthrough2mc = playthrough_info
        playthrough2mn = None
    playthrough2mc = playthrough2mc[playthrough2mc.isin(result_df.index)]
    mc_playthrough_col = pd.Series(
        sum(
            [
                [playthrough] * n_occurrences[mc]
                for playthrough, mc in playthrough2mc.items()
            ],
            [],
        )
    )
    result_df = result_df.loc[playthrough2mc.values].reset_index()
    if "mn" in result_df.columns:
        column_position = result_df.columns.get_loc("mn") + 1
        if playthrough2mn is not None:
            mn_playthrough_col = mc_playthrough_col.map(playthrough2mn)
            result_df.insert(column_position, "mn_playthrough", mn_playthrough_col)
    else:
        column_position = result_df.columns.get_loc("mc") + 1
    result_df.insert(column_position, "mc_playthrough", mc_playthrough_col)
    return result_df


@contextmanager
def unpack_mscz(mscz, tmp_dir=None, logger=None):
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if tmp_dir is None:
        tmp_dir = os.path.dirname(mscz)
    tmp_file = Temp(suffix=".mscx", prefix=".", dir=tmp_dir, delete=False)
    with Zip(mscz) as zip_file:
        mscx_files = [f for f in zip_file.namelist() if f.endswith(".mscx")]
        if len(mscx_files) > 1:
            logger.info(f"{mscz} contains several MSCX files. Picking the first one")
        mscx = mscx_files[0]
        with zip_file.open(mscx) as mscx_file:
            with tmp_file as tmp:
                for line in mscx_file:
                    tmp.write(line)
    try:
        yield tmp_file.name
    except Exception:
        logger.error(
            f"Error while dealing with the temporarily unpacked {os.path.basename(mscz)}"
        )
        raise
    finally:
        os.remove(tmp_file.name)


@contextmanager
def capture_parse_logs(
    logger_object: logging.Logger, level: Union[str, int] = "w", logger=None
) -> LogCapturer:
    """Within the context, the given logger will have an additional handler that captures all messages with level
    ``level`` or higher. At the end of the context, retrieve the message list via LogCapturer.content_list.

    Example:
        .. code-block:: python

            with capture_parse_logs(logger, level='d') as capturer:
                # do the stuff of which you want to capture the log messages of the given level (and above)
                all_messages = capturer.content_list
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    captured_warnings = LogCapturer(level=level)
    logger_object.addHandler(captured_warnings.log_handler)
    yield captured_warnings
    logger_object.removeHandler(captured_warnings.log_handler)


def update_labels_cfg(labels_cfg, logger=None):
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    keys = [
        "staff",
        "voice",
        "harmony_layer",
        "positioning",
        "decode",
        "column_name",
        "color_format",
    ]
    if "logger" in labels_cfg:
        del labels_cfg["logger"]
    updated, incorrect = update_cfg(cfg_dict=labels_cfg, admitted_keys=keys)
    if len(incorrect) > 0:
        last_5 = ", ".join(f"-{i}: {stack()[i].function}()" for i in range(1, 6))
        plural = "These options are" if len(incorrect) > 1 else "This option is"
        logger.warning(
            f"{plural} not valid to configure labels: {incorrect}\nLast 5 function calls leading here: {last_5}"
        )
    return updated


def write_metadata(
    metadata_df: pd.DataFrame, path: str, index=False, logger=None
) -> bool:
    """
    Write the DataFrame ``metadata_df`` to ``path``, updating an existing file rather than overwriting it.

    Args:
      metadata_df:
          DataFrame with one row per piece and an index of strings identifying pieces. The index is used for
          updating a potentially pre-existent file, from which the first column ∈ ('piece', 'fname', 'fnames', 'name',
          'names')
          will be used as index.
      path:
          If folder path, the filename 'metadata.tsv' will be appended; file_path will be used as is but a
          warning is thrown if the extension is not .tsv
      index: Pass True if you want the first column of the output to be a RangeIndex starting from 0.

    Returns:
      True if the metadata were successfully written, False otherwise.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    metadata_df = metadata_df.astype("string")
    metadata_df = enforce_piece_index_for_metadata(metadata_df)
    path = resolve_dir(path)
    if os.path.isdir(path):
        tsv_path = os.path.join(path, "metadata.tsv")
    else:
        tsv_path = path
    _, fext = os.path.splitext(tsv_path)
    if fext != ".tsv":
        logger.warning(
            f"The output format for metadata is Tab-Separated Values (.tsv) but the file extensions is {fext}."
        )
    if not os.path.isfile(tsv_path):
        output_df = metadata_df
        msg = "Created"
    else:
        # Trying to load an existing 'metadata.tsv' file to update existing rows
        previous = pd.read_csv(tsv_path, sep="\t", dtype="string")
        previous = enforce_piece_index_for_metadata(previous)
        for ix, what in zip(
            (previous.index, previous.columns, metadata_df.index, metadata_df.columns),
            (
                "index of the existing",
                "columns of the existing",
                "index of the updated",
                "columns of the updated",
            ),
        ):
            if not ix.is_unique:
                duplicated = ix[ix.duplicated()].to_list()
                logger.error(
                    f"The {what} metadata contains duplicates and no metadata were written.\nDuplicates: {duplicated}"
                )
                return False
        new_cols = metadata_df.columns.difference(previous.columns)
        shared_cols = metadata_df.columns.intersection(previous.columns)
        new_rows = metadata_df.index.difference(previous.index)
        with warnings.catch_warnings():
            # pandas 2.1.0: "FutureWarning: The behavior of array concatenation with empty entries is deprecated. In
            # a future version, this will no longer exclude empty items when determining the result dtype. To retain
            # the old behavior, exclude the empty entries before the concat operation."
            warnings.filterwarnings("ignore", category=FutureWarning)
            previous = pd.concat([previous, metadata_df.loc[new_rows, shared_cols]])
            previous = pd.concat([previous, metadata_df[new_cols]], axis=1)
        previous.update(metadata_df)
        legacy_columns = [c for c in LEGACY_COLUMNS if c in previous.columns]
        if len(legacy_columns) > 0:
            plural = (
                f"s {legacy_columns}"
                if len(legacy_columns) > 1
                else f" {legacy_columns[0]}"
            )
            previous = previous.drop(columns=legacy_columns)
            logger.info(f"Dropped legacy column{plural} .")
        output_df = previous.reset_index()
        msg = "Updated"
    output_df = prepare_metadata_for_writing(output_df)
    output_df.to_csv(tsv_path, sep="\t", index=index)
    logger.info(f"{msg} {tsv_path}")
    return True


def enforce_piece_index_for_metadata(
    metadata_df: pd.DataFrame, append=False, logger=None
) -> pd.DataFrame:
    """Returns a copy of the DataFrame that has an index level called 'piece'."""
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    possible_column_names = (
        "piece",
        "fname",
        "fnames",
        "filename",
        "name",
        "names",
    )
    if any(name in metadata_df.index.names for name in possible_column_names):
        return metadata_df.copy()
    try:
        piece_col = next(
            col for col in possible_column_names if col in metadata_df.columns
        )
    except StopIteration:
        raise ValueError(
            "Metadata is expected to come with a column or index level called 'piece' or ("
            "previously) 'fname'."
        )
    if piece_col != "piece":
        metadata_df = metadata_df.rename(columns={piece_col: "piece"})
        logger.info(f"Renamed column '{piece_col}' -> 'piece'")
    return metadata_df.set_index("piece", append=append)


def overwrite_overview_section_in_markdown_file(file_path, md_str, logger=None):
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if os.path.isfile(file_path):
        msg = "Updated"
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        msg = "Created"
        lines = []
    # in case the README.md exists, everything from the line including '# Overview' (or last line otherwise) is
    # overwritten
    with open(file_path, "w", encoding="utf-8") as f:
        for line in lines:
            if "# Overview" in line:
                break
            f.write(line)
        else:
            f.write("\n\n")
        f.write(md_str)
    logger.info(f"{msg} {file_path}")


def write_markdown(metadata_df: pd.DataFrame, file_path: str, logger=None) -> None:
    """
    Write a subset of the DataFrame ``metadata_df`` to ``path`` in markdown format. If the file exists, it will be
    scanned for a line containing the string '# Overview' and overwritten from that line onwards.

    Args:
      metadata_df: DataFrame containing metadata.
      file_path: Path of the markdown file.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    rename4markdown = {
        "piece": "file_name",
        "last_mn": "measures",
        "label_count": "labels",
        "harmony_version": "standard",
        "annotators": "annotators",
        "reviewers": "reviewers",
    }
    rename4markdown = {
        k: v for k, v in rename4markdown.items() if k in metadata_df.columns
    }
    drop_index = "fname" in metadata_df.columns
    md = metadata_df.reset_index(drop=drop_index)[list(rename4markdown.keys())].fillna(
        ""
    )
    md = md.rename(columns=rename4markdown)
    md_table = "#" + str(
        dataframe2markdown(md, name="Overview")
    )  # comes with a first-level heading which we turn into second-level
    md_table += "\n\n*Overview table automatically updated using [ms3](https://ms3.readthedocs.io/).*\n"

    overwrite_overview_section_in_markdown_file(file_path, md_table, logger)


def prepare_metadata_for_writing(metadata_df):
    # convert_to_str = {c: 'string' for c in ('length_qb', 'length_qb_unfolded', 'all_notes_qb') if c in metadata_df}
    # if len(convert_to_str) > 0:
    #     metadata_df = metadata_df.astype(convert_to_str, errors='ignore')
    metadata_df = metadata_df.astype("string", errors="ignore")
    metadata_df.sort_index(inplace=True)
    metadata_df = column_order(metadata_df, METADATA_COLUMN_ORDER, sort=False)
    # on Windows, make sure the paths are written with / separators
    path_cols = [col for col in ("subdir", "rel_path") if col in metadata_df.columns]
    for col in path_cols:
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
            metadata_df.loc[:, col] = metadata_df[col].str.replace(
                os.sep, "/", regex=False
            )
    staff_cols, other_cols = [], []
    for col in metadata_df.columns:
        if re.match(r"^staff_(\d+)", col):
            staff_cols.append(col)
        else:
            other_cols.append(col)
    staff_cols = sorted(staff_cols, key=lambda s: int(re.match(r"^staff_(\d+)", s)[1]))
    metadata_df = metadata_df[other_cols + staff_cols]
    if not isinstance(metadata_df.index, pd.RangeIndex):
        metadata_df = metadata_df.reset_index()
    return metadata_df


def ensure_correct_column_types(
    df: pd.DataFrame, exclude_columns: Optional[Collection[str]] = None
) -> pd.DataFrame:
    excluded_columns = (
        list(BOOLEAN_COLUMNS + COLLECTION_COLUMNS)
        if exclude_columns is None
        else exclude_columns
    )
    columns_to_convert = {
        col: TSV_COLUMN_DTYPES[col]
        for col in df.columns
        if col in TSV_COLUMN_DTYPES and col not in excluded_columns
    }
    df_converted = df.astype(
        columns_to_convert, errors="ignore"
    )  # without "ignore", pandas 2.1.0 complains about NA
    # values in the duration_qb (float!) column 🙄

    return df_converted


def write_tsv(
    df: pd.DataFrame, file_path: str, pre_process: bool = True, logger=None, **kwargs
):
    """Write a DataFrame to a TSV or CSV file based on the extension of 'file_path'. By default,
    the index is not included, unless you pass ``index=True`` as additional keyword argument.
    Uses: :py:func:`no_collections_no_booleans`

    Args:
        df: DataFrame to write to disk.
        file_path:
            File to create or overwrite. If the extension is .tsv, the argument 'sep' will be set to '\t', otherwise the
            extension is expected to be .csv and the default separator ',' will be used.
            Apart from that, the extension 'zip' is also allowed but you need to provide the kwargs yourself, especially
            something like ``compression = dict(method='zip', archive_name='innername.csv')``
        pre_process:
            By default, DataFrame cells containing lists and tuples will be transformed to strings and Booleans will be
            converted to 0 and 1 (otherwise they will be written out as True and False). Pass False to prevent.
        **kwargs:
            Additional keyword arguments will be passed on to :py:meth:`pandas.DataFrame.to_csv`.
            Defaults arguments are ``index=False`` and ``sep='\t'`` (assuming extension '.tsv', see above).
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    path, file = os.path.split(file_path)
    path = resolve_dir(path)
    os.path.join(path, file)
    piece, ext = os.path.splitext(file)
    if ext.lower() not in (".tsv", ".csv", ".zip"):
        logger.error(
            f"This function expects file_path to include the file name ending on .csv, .tsv, or .zip, not '{ext}'."
        )
        return
    os.makedirs(path, exist_ok=True)
    if ext.lower() == ".tsv":
        kwargs.update(dict(sep="\t"))
    if "index" not in kwargs:
        kwargs["index"] = False
    if pre_process:
        df = no_collections_no_booleans(df, logger=logger)
        df = ensure_correct_column_types(df)
    df.to_csv(file_path, **kwargs)
    logger.debug(f"{file_path} written with parameters {kwargs}.")
    return


def abs2rel_key(
    absolute: str, localkey: str, global_minor: bool = False, logger=None
) -> str:
    """
    Expresses a Roman numeral as scale degree relative to a given localkey.
    The result changes depending on whether Roman numeral and localkey are
    interpreted within a global major or minor key.

    Uses: :py:func:`split_scale_degree`


    Args:
      absolute: Absolute key expressed as Roman scale degree of the local key.
      localkey: The local key in terms of which ``absolute`` will be expressed.
      global_minor: Has to be set to True if `absolute` and `localkey` are scale degrees of a global minor key.

    Examples:
      In a minor context, the key of II would appear within the key of vii as #III.

          >>> abs2rel_key('iv', 'VI', global_minor=False)
          'bvi'       # F minor expressed with respect to A major
          >>> abs2rel_key('iv', 'vi', global_minor=False)
          'vi'      # F minor expressed with respect to A minor
          >>> abs2rel_key('iv', 'VI', global_minor=True)
          'vi'      # F minor expressed with respect to Ab major
          >>> abs2rel_key('iv', 'vi', global_minor=True)
          '#vi'       # F minor expressed with respect to Ab minor

          >>> abs2rel_key('VI', 'IV', global_minor=False)
          'III'       # A major expressed with respect to F major
          >>> abs2rel_key('VI', 'iv', global_minor=False)
          '#III'       # A major expressed with respect to F minor
          >>> abs2rel_key('VI', 'IV', global_minor=True)
          'bIII'       # Ab major expressed with respect to F major
          >>> abs2rel_key('VI', 'iv', global_minor=False)
          'III'       # Ab major expressed with respect to F minor
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if pd.isnull(absolute) or pd.isnull(localkey):
        return absolute
    absolute = resolve_relative_keys(absolute)
    localkey = resolve_relative_keys(localkey)
    maj_rn = ["I", "II", "III", "IV", "V", "VI", "VII"]
    min_rn = ["i", "ii", "iii", "iv", "v", "vi", "vii"]
    white_key_major_accidentals = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 1, 1, 0, 1, 1, 1],
        ]
    )
    abs_accidentals, absolute = split_scale_degree(absolute, count=True, logger=logger)
    localkey_accidentals, localkey = split_scale_degree(
        localkey, count=True, logger=logger
    )
    resulting_accidentals = abs_accidentals - localkey_accidentals
    numerals = maj_rn if absolute.isupper() else min_rn
    localkey_index = maj_rn.index(localkey.upper())
    result_index = (numerals.index(absolute) - localkey_index) % 7
    result_numeral = numerals[result_index]
    if localkey.islower() and result_index in [2, 5, 6]:
        resulting_accidentals += 1
    if global_minor:
        localkey_index = (localkey_index - 2) % 7
    resulting_accidentals -= white_key_major_accidentals[localkey_index][result_index]
    acc = (
        resulting_accidentals * "#"
        if resulting_accidentals > 0
        else -resulting_accidentals * "b"
    )
    return acc + result_numeral


def rel2abs_key(
    relative: str, localkey: str, global_minor: bool = False, logger=None
) -> Optional[str]:
    """Expresses a Roman numeral that is expressed relative to a localkey
    as scale degree of the global key. For local keys {III, iii, VI, vi, VII, vii}
    the result changes depending on whether the global key is major or minor.

    Uses: :py:func:`split_scale_degree`


    Args:
      relative: Relative key or chord expressed as Roman scale degree of the local key.
      localkey: The local key to which `rel` is relative.
      global_minor: Has to be set to True if `localkey` is a scale degree of a global minor key.

    Examples:
      If the label viio6/VI appears in the context of the local key VI or vi,
      the absolute key to which viio6 applies depends on the global key.
      The comments express the examples in relation to global C major or C minor.

          >>> rel2abs_key('vi', 'VI', global_minor=False)
          '#iv'       # vi of A major = F# minor
          >>> rel2abs_key('vi', 'vi', global_minor=False)
          'iv'      # vi of A minor = F minor
          >>> rel2abs_key('vi', 'VI', global_minor=True)
          'iv'      # vi of Ab major = F minor
          >>> rel2abs_key('vi', 'vi', global_minor=True)
          'biv'       # vi of Ab minor = Fb minor

      The same examples hold if you're expressing in terms of the global key
      the root of a VI-chord within the local keys VI or vi.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if pd.isnull(relative) or pd.isnull(localkey):
        return relative
    relative = resolve_relative_keys(relative)
    localkey = resolve_relative_keys(localkey)
    maj_rn = ["I", "II", "III", "IV", "V", "VI", "VII"]
    min_rn = ["i", "ii", "iii", "iv", "v", "vi", "vii"]
    white_key_major_accidentals = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 1, 1, 0, 1, 1, 1],
        ]
    )
    relative_accidentals, relative = split_scale_degree(
        relative, count=True, logger=logger
    )
    localkey_accidentals, localkey = split_scale_degree(
        localkey, count=True, logger=logger
    )
    if relative is None or localkey is None:
        return
    resulting_accidentals = relative_accidentals + localkey_accidentals
    numerals = maj_rn if relative.isupper() else min_rn
    rel_num = numerals.index(relative)
    localkey_index = maj_rn.index(localkey.upper())
    result_numeral = numerals[(rel_num + localkey_index) % 7]
    if localkey.islower() and rel_num in [2, 5, 6]:
        resulting_accidentals -= 1
    if global_minor:
        localkey_index = (localkey_index - 2) % 7
    resulting_accidentals += white_key_major_accidentals[rel_num][localkey_index]
    acc = (
        resulting_accidentals * "#"
        if resulting_accidentals > 0
        else -resulting_accidentals * "b"
    )
    return acc + result_numeral


def make_interval_index_from_durations(
    df,
    position_col="quarterbeats",
    duration_col="duration_qb",
    closed="left",
    round=None,
    name="interval",
    logger=None,
):
    """Given an annotations table with positions and durations, create an :obj:`pandas.IntervalIndex`.
    Returns None if any row is underspecified.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        Annotation table containing the columns of ``position_col`` (default: 'quarterbeats') and ``duration_col``
        default: 'duration_qb').
    position_col : :obj:`str`, optional
        Name of the column containing positions, used as left boundaries.
    duration_col : :obj:`str`, optional
        Name of the column containing durations which will be added to the positions to obtain right boundaries.
    closed : :obj:`str`, optional
        'left', 'right' or 'both' <- defining the interval boundaries
    round : :obj:`int`, optional
        To how many decimal places to round the intervals' boundary values.
    name : :obj:`str`, optional
        Name of the created index. Defaults to 'interval'.

    Returns
    -------
    :obj:`pandas.IntervalIndex`
        A copy of ``df`` with the original index replaced and underspecified rows removed (those where no interval
        could be coputed).
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if not all(c in df.columns for c in (position_col, duration_col)):
        missing = [c for c in (position_col, duration_col) if c not in df.columns]
        plural = "s" if len(missing) > 1 else ""
        logger.warning(f"Column{plural} not present in DataFrame: {', '.join(missing)}")
        return
    if df[position_col].isna().any() or df[duration_col].isna().any():
        missing = df[df[[position_col, duration_col]].isna().any(axis=1)]
        logger.warning(
            f"Could not make IntervalIndex because of missing values:\n{missing}"
        )
        return
    try:
        left = df[position_col].astype(float)
        right = (left + df[duration_col]).astype(float)
        if round is not None:
            left, right = left.round(round), right.round(round)
        return pd.IntervalIndex.from_arrays(
            left=left, right=right, closed=closed, name=name
        )
    except Exception as e:
        logger.warning(f"Creating IntervalIndex failed with exception {e}.")


def replace_index_by_intervals(
    df,
    position_col="quarterbeats",
    duration_col="duration_qb",
    closed="left",
    filter_zero_duration=False,
    round=None,
    name="interval",
    logger=None,
):
    """Given an annotations table with positions and durations, replaces its index with an :obj:`pandas.IntervalIndex`.
    Underspecified rows are removed.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        Annotation table containing the columns of ``position_col`` (default: 'quarterbeats') and ``duration_col``
        default: 'duration_qb').
    position_col : :obj:`str`, optional
        Name of the column containing positions.
    duration_col : :obj:`str`, optional
        Name of the column containing durations.
    closed : :obj:`str`, optional
        'left', 'right' or 'both' <- defining the interval boundaries
    filter_zero_duration : :obj:`bool`, optional
        Defaults to False, meaning that rows with zero durations are maintained. Pass True to remove them.
    round : :obj:`int`, optional
        To how many decimal places to round the intervals' boundary values.
    name : :obj:`str`, optional
        Name of the created index. Defaults to 'interval'.

    Returns
    -------
    :obj:`pandas.DataFrame`
        A copy of ``df`` with the original index replaced and underspecified rows removed (those where no interval
        could be computed).
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if not all(c in df.columns for c in (position_col, duration_col)):
        missing = [c for c in (position_col, duration_col) if c not in df.columns]
        plural = "s" if len(missing) > 1 else ""
        logger.warning(f"Column{plural} not present in DataFrame: {', '.join(missing)}")
        return df
    mask = (
        df[position_col].notna() & (df[position_col] != "") & df[duration_col].notna()
    )
    if not mask.any():
        logger.warning(
            f"Either position_col {position_col!r} or duration_col {duration_col!r} are empty."
        )
        return df
    n_dropped = (~mask).sum()
    if filter_zero_duration:
        mask &= df[duration_col] > 0
    elif n_dropped > 0:
        logger.info(
            f"Had to drop {n_dropped} rows for creating the IntervalIndex:\n{df[~mask]}"
        )
    df = df[mask].copy()
    iv_index = make_interval_index_from_durations(
        df,
        position_col=position_col,
        duration_col=duration_col,
        closed=closed,
        round=round,
        name=name,
        logger=logger,
    )
    if df[duration_col].dtype != float:
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
            df.loc[:, duration_col] = pd.to_numeric(df[duration_col])
    if iv_index is None:
        logger.warning("Creating IntervalIndex failed.")
        return df
    df.index = iv_index
    return df


def boolean_mode_col2strings(S) -> pd.Series:
    """Turn the boolean is_minor columns into string columns such that True => 'minor', False => 'major'."""
    return S.map({True: "minor", False: "major"})


def replace_boolean_mode_by_strings(df) -> pd.DataFrame:
    """Replaces boolean '_is_minor' columns with string columns renamed to '_mode'.
    Example: df['some_col', 'some_name_is_minor'] => df['some_col', 'some_name_mode']
    """
    bool_cols = [col for col in df.columns if col.endswith("_is_minor")]
    if len(bool_cols) == 0:
        return df
    df = df.copy()
    renaming = {}
    for col_name in bool_cols:
        numeral_name = col_name[: -len("_is_minor")]
        if col_name in df.columns:
            new_col_name = f"{numeral_name}_mode"
            new_col = boolean_mode_col2strings(df[col_name])
            df.loc[:, col_name] = new_col
            renaming[col_name] = new_col_name
    df.rename(columns=renaming, inplace=True)
    return df


def resolve_relative_keys(relativeroot, minor=False, logger=None):
    """Resolve nested relative keys, e.g. 'V/V/V' => 'VI' if minor is False (default) or '#VI' if True.

    Uses: :py:func:`rel2abs_key`, :py:func:`str_is_minor`

    relativeroot : :obj:`str`
        One or several relative keys, e.g. iv/v/VI (fourth scale degree of the fifth scale degree of the sixth scale
            degree)
    minor : :obj:`bool`, optional
        Pass True if the last of the relative keys is to be interpreted within a minor context, and therefore the
        result.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if pd.isnull(relativeroot):
        return relativeroot
    spl = relativeroot.split("/")
    if len(spl) < 2:
        return relativeroot
    if len(spl) == 2:
        applied, to = spl
        return rel2abs_key(applied, to, minor, logger=logger)
    previous, last = "/".join(spl[:-1]), spl[-1]
    return rel2abs_key(
        resolve_relative_keys(previous, str_is_minor(last, is_name=False)), last, minor
    )


def series_is_minor(S, is_name=True):
    """Returns boolean Series where every value in ``S`` representing a minor key/chord is True."""
    # regex = r'([A-Ga-g])[#|b]*' if is_name else '[#|b]*(\w+)'
    # return S.str.replace(regex, lambda m: m.group(1)).str.islower()
    return (
        S.str.islower()
    )  # as soon as one character is not lowercase, it should be major


def str_is_minor(tone, is_name=True):
    """Returns True if ``tone`` represents a minor key or chord."""
    # regex = r'([A-Ga-g])[#|b]*' if is_name else '[#|b]*(\w+)'
    # m = re.match(regex, tone)
    # if m is None:
    #     return m
    # return m.group(1).islower()
    return tone.islower()


def transpose_changes(
    changes, old_num, new_num, old_minor=False, new_minor=False, logger=None
):
    """
    Since the interval sizes expressed by the changes of the DCML harmony syntax
    depend on the numeral's position in the scale, these may change if the numeral
    is transposed. This function expresses the same changes for the new position.
    Chord tone alterations (of 3 and 5) stay untouched.

    Uses: :py:func:`changes2tpc`

    Parameters
    ----------
    changes : :obj:`str`
        A string of changes following the DCML harmony standard.
    old_num, new_num : :obj:`str`:
        Old numeral, new numeral.
    old_minor, new_minor : :obj:`bool`, optional
        For each numeral, pass True if it occurs in a minor context.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if pd.isnull(changes):
        return changes
    old = changes2tpc(changes, old_num, minor=old_minor, root_alterations=True)
    new = changes2tpc(changes, new_num, minor=new_minor, root_alterations=True)
    res = []

    def get_acc(n):
        return n * "#" if n > 0 else -n * "b"

    for (full, added, acc, chord_interval, iv1), (_, _, _, _, iv2) in zip(old, new):
        if iv1 is None or iv1 == iv2:
            res.append(full)
        else:
            d = iv2 - iv1
            if d % 7 > 0:
                logger.warning(
                    f"The difference between the intervals of {full} in {old_num} and {new_num} (in "
                    f"{'minor' if new_minor else 'major'}) don't differ by chromatic semitones."
                )
            n_acc = acc.count("#") - acc.count("b")
            new_acc = get_acc(n_acc - d // 7)
            res.append(added + new_acc + chord_interval)
    return "".join(res)


def features2tpcs(
    numeral,
    form=None,
    figbass=None,
    changes=None,
    relativeroot=None,
    key="C",
    minor=None,
    merge_tones=True,
    bass_only=False,
    mc=None,
    logger=None,
):
    """
    Given the features of a chord label, this function returns the chord tones
    in the order of the inversion, starting from the bass note. The tones are
    expressed as tonal pitch classes, where -1=F, 0=C, 1=G etc.

    Uses: :py:func:`~.utils.changes2list`, :py:func:`~.utils.name2fifths`, :py:func:`~.utils.resolve_relative_keys`,
    :py:func:`~.utils.roman_numeral2fifths`,
    :py:func:`~.utils.sort_tpcs`, :py:func:`~.utils.str_is_minor`

    Parameters
    ----------
    numeral: :obj:`str`
        Roman numeral of the chord's root
    form: {None, 'M', 'o', '+' '%'}, optional
        Indicates the chord type if not a major or minor triad (for which ``form`` is None).
        '%' and 'M' can only occur as tetrads, not as triads.
    figbass: {None, '6', '64', '7', '65', '43', '2'}, optional
        Indicates chord's inversion. Pass None for triad root position.
    changes: :obj:`str`, optional
        Added steps such as '+6' or suspensions such as '4' or any combination such as (9+64).
        Numbers need to be in descending order.
    relativeroot: :obj:`str`, optional
        Pass a Roman scale degree if `numeral` is to be applied to a different scale
        degree of the local key, as in 'V65/V'
    key : :obj:`str` or :obj:`int`, optional
        The local key expressed as the root's note name or a tonal pitch class.
        If it is a name and `minor` is `None`, uppercase means major and lowercase minor.
        If it is a tonal pitch class, `minor` needs to be specified.
    minor : :obj:`bool`, optional
        Pass True for minor and False for major. Can be omitted if `key` is a note name.
        This affects calculation of chords related to III, VI and VII.
    merge_tones : :obj:`bool`, optional
        Pass False if you want the function to return two tuples, one with (potentially suspended)
        chord tones and one with added notes.
    bass_only : :obj:`bool`, optional
        Return only the bass note instead of all chord tones.
    mc : int or str
        Pass measure count to display it in warnings.

    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if pd.isnull(numeral) or numeral == "@none":
        if bass_only or merge_tones:
            return pd.NA
        else:
            return {
                "chord_tones": pd.NA,
                "added_tones": pd.NA,
                "root": pd.NA,
            }
    form, figbass, changes, relativeroot = tuple(
        "" if pd.isnull(val) else val for val in (form, figbass, changes, relativeroot)
    )
    label = (
        f"{numeral}{form}{figbass}{'(' + changes + ')' if changes != '' else ''}"
        f"{'/' + relativeroot if relativeroot != '' else ''}"
    )
    MC = "" if mc is None else f"MC {mc}: "
    if minor is None:
        try:
            minor = str_is_minor(key, is_name=True)
            logger.debug(f"Mode inferred from {key}.")
        except Exception:
            raise ValueError(
                f"If parameter 'minor' is not specified, 'key' needs to be a string, not {key}"
            )

    key = name2fifths(key, logger=logger)

    if form in ["%", "M", "+M"]:
        assert figbass in ["7", "65", "43", "2"], (
            f"{MC}{label}: {form} requires figbass (7, 65, 43, or 2) since it specifies a "
            f"chord's seventh."
        )

    if relativeroot != "":
        resolved = resolve_relative_keys(relativeroot, minor, logger=logger)
        rel_minor = str_is_minor(resolved, is_name=False)
        transp = roman_numeral2fifths(resolved, minor, logger=logger)
        logger.debug(
            f"{MC}Chord applied to {relativeroot}. Therefore transposing it by {transp} fifths."
        )
        return features2tpcs(
            numeral=numeral,
            form=form,
            figbass=figbass,
            relativeroot=None,
            changes=changes,
            key=key + transp,
            minor=rel_minor,
            merge_tones=merge_tones,
            bass_only=bass_only,
            mc=mc,
            logger=logger,
        )

    if numeral.lower() == "#vii" and not minor:
        logger.warning(
            f"{MC}{numeral} in major context corrected to {numeral[1:]}.",
            extra={"message_id": (27, MC)},
        )
        numeral = numeral[1:]

    root_alteration, num_degree = split_scale_degree(numeral, count=True, logger=logger)

    # build 2-octave diatonic scale on C major/minor
    root = ["I", "II", "III", "IV", "V", "VI", "VII"].index(num_degree.upper())
    tpcs = (
        2 * [i + key for i in (0, 2, -3, -1, 1, -4, -2)]
        if minor
        else 2 * [i + key for i in (0, 2, 4, -1, 1, 3, 5)]
    )
    # starting the scale from chord root, i.e. root will be tpcs[0], the chord's seventh tpcs[6] etc.
    tpcs = tpcs[root:] + tpcs[:root]
    root = tpcs[0] + 7 * root_alteration
    tpcs[0] = root  # octave stays diatonic, is not altered

    # logger.debug(f"{num_degree}: The {'minor' if minor else 'major'} scale starting from the root: {tpcs}")

    def set_iv(chord_interval, interval_size):
        """Add to the interval of a given chord interval in `tpcs` (both viewed from the root note).

        Parameters
        ----------
        chord_interval : :obj:`int`
            Pass 0 for the root note, 2 for the third, 8 for the ninth etc.
        interval_size : :obj:`int`
            Stack-of-fifths interval.
        """
        nonlocal tpcs, root
        iv = root + interval_size
        tpcs[chord_interval] = iv
        tpcs[chord_interval + 7] = iv

    is_triad = figbass in ["", "6", "64"]
    is_seventh_chord = figbass in ["7", "65", "43", "2"]
    if not is_triad and not is_seventh_chord:
        raise ValueError(f"{MC}{figbass} is not a valid chord inversion.")

    if form == "o":
        set_iv(2, -3)
        set_iv(4, -6)
        if is_seventh_chord:
            set_iv(6, -9)
    elif form == "%":
        set_iv(2, -3)
        set_iv(4, -6)
        set_iv(6, -2)
    elif form == "+":
        set_iv(2, 4)
        set_iv(4, 8)
        if is_seventh_chord:
            set_iv(6, -2)
    elif form == "+M":
        set_iv(2, 4)
        set_iv(4, 8)
        set_iv(6, 5)
    else:  # triad with or without major or minor seven
        set_iv(4, 1)
        if num_degree.isupper():
            set_iv(2, 4)
        else:
            set_iv(2, -3)
        if form == "M":
            set_iv(6, 5)
        elif is_seventh_chord:
            set_iv(6, -2)

    tone_functions = (0, 2, 4, 6) if is_seventh_chord else (0, 2, 4)
    root_position = {i: [tpcs[i]] for i in tone_functions}
    replacements = {i: [] for i in tone_functions}

    def replace_chord_tone(which, by):
        nonlocal root_position, replacements
        if which in root_position:
            root_position[which] = []
            replacements[which].insert(0, by)
        else:
            logger.warning(f"Only chord tones [0,2,4,(6)] can be replaced, not {which}")

    # apply changes
    alts = changes2list(changes, sort=False)
    added_notes = []
    for full, add_remove, acc, chord_interval in alts:
        added = add_remove == "+"
        substracted = add_remove == "-"
        replacing_upper = add_remove == "^"
        replacing_lower = add_remove == "v"
        chord_interval = int(chord_interval) - 1
        # From here on, `chord_interval` is decremented, i.e. the root is 0, the seventh is 6 etc. (just like in
        # `tpcs`)
        if (chord_interval == 0 and not substracted) or chord_interval > 13:
            logger.warning(
                f"{MC}Change {full} is meaningless and ignored because it concerns chord tone {chord_interval + 1}."
            )
            continue
        next_octave = chord_interval > 7
        shift = 7 * (acc.count("#") - acc.count("b"))
        new_val = tpcs[chord_interval] + shift
        if substracted:
            if chord_interval not in tone_functions:
                logger.warning(
                    f"{MC}The change {full} has no effect because it concerns an interval which is not implied by "
                    f"{numeral}{form}{figbass}."
                )
            else:
                root_position[chord_interval] = []
        elif added:
            added_notes.append(new_val)
        elif next_octave:
            if any((replacing_lower, replacing_upper, substracted)):
                logger.info(
                    f"{MC}{full[0]} has no effect on tonal pitch classes computed for {full}  because the interval is "
                    f"larger than an octave."
                )
            added_notes.append(new_val)
        elif chord_interval in [
            1,
            3,
            5,
        ]:  # these are changes to scale degree 2, 4, 6 that replace the lower neighbour
            # unless they have a # or ^
            if "#" in acc or replacing_upper:
                if "#" in acc and replacing_upper:
                    logger.info(f"{MC}^ is redundant in {full}.")
                if (
                    chord_interval == 5 and is_triad
                ):  # leading tone to 7 but not in seventh chord
                    added_notes.append(new_val)
                else:
                    replace_chord_tone(chord_interval + 1, new_val)
            else:
                if replacing_lower:
                    logger.info(f"{MC}v is redundant in {full}.")
                replace_chord_tone(chord_interval - 1, new_val)
        else:  # chord tone alterations
            if replacing_lower:
                # TODO: This must be possible, e.g. V(6v5) where 5 is suspension of 4
                logger.info(
                    f"{MC}{full} -> chord tones cannot replace neighbours, use + instead."
                )
            elif chord_interval == 6 and figbass != "7":  # 7th are a special case:
                if figbass == "":  # in root position triads they are added
                    # TODO: The standard is lacking a distinction, because the root in root pos. can also be replaced
                    #  from below!
                    added_notes.append(new_val)
                elif (
                    figbass in ["6", "64"] or "#" in acc
                ):  # in inverted triads they replace the root, as does #7
                    replace_chord_tone(0, new_val)
                else:  # otherwise they are unclear
                    logger.warning(
                        f"{MC}In seventh chords, such as {label}, it is not clear whether the {full} alters the 7 or "
                        f"replaces the 8 and should not be used.",
                        extra={"message_id": (18,)},
                    )
            elif tpcs[chord_interval] == new_val:
                logger.info(
                    f"{MC}The change {full} has no effect in {numeral}{form}{figbass}"
                )
            else:
                root_position[chord_interval] = [new_val]

    figbass2bass = {"": 0, "7": 0, "6": 1, "65": 1, "64": 2, "43": 2, "2": 3}
    bass = figbass2bass[figbass]
    chord_tones = []
    tone_function_names = {
        0: "root",
        2: "3rd",
        4: "5th",
        6: "7th",
    }
    for tf in tone_functions[bass:] + tone_functions[:bass]:
        chord_tone, replacing_tones = root_position[tf], replacements[tf]
        if chord_tone == replacing_tones == []:
            logger.debug(
                f"{MC}{label} results in a chord without {tone_function_names[tf]}."
            )
        if chord_tone != []:
            chord_tones.append(chord_tone[0])
            if replacing_tones != []:
                logger.warning(
                    f"{MC}{label} results in a chord tone {tone_function_names[tf]} AND its replacement(s) (TPC "
                    f"{replacing_tones}). "
                    f"You might want to add a + to distinguish from a suspension, or add this warning to "
                    f"IGNORED_WARNINGS with a comment.",
                    extra={"message_id": (6, mc, label)},
                )
        chord_tones.extend(replacing_tones)

    bass_tpc = chord_tones[0]
    if bass_only:
        return bass_tpc
    elif merge_tones:
        return tuple(sort_tpcs(chord_tones + added_notes, start=bass_tpc))
    else:
        return {
            "chord_tones": tuple(chord_tones),
            "added_tones": tuple(added_notes),
            "root": root,
        }


def path2parent_corpus(path):
    """Walk up the path and return the name of the first superdirectory that is a git repository or contains a
    'metadata.tsv' file."""
    if path in ("", "/"):
        return None
    try:
        if os.path.isdir(path):
            listdir = os.listdir(path)
            if "metadata.tsv" in listdir or ".git" in listdir:
                return path
        return path2parent_corpus(os.path.dirname(path))
    except Exception:
        return None


def chord2tpcs(
    chord: str,
    regex: Optional[re.Pattern] = None,
    logger: Optional[logging.Logger] = None,
    **kwargs,
):
    """
    Split a chord label into its features and apply features2tpcs().

    Uses: features2tpcs()

    Args:
        chord:
            Chord label that can be split into the features ['numeral', 'form', 'figbass',
            'changes', 'relativeroot'].
        regex:
            Compiled regex with named groups for the five features. By default, the current
            version of the DCML harmony annotation standard is used.
        **kwargs:
            arguments for features2tpcs (pass mc=MC to show it in warnings!)
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if regex is None:
        regex = DCML_REGEX
    chord_features = re.match(regex, chord)
    assert chord_features is not None, f"{chord} does not match the regex."
    chord_features = chord_features.groupdict()
    numeral, form, figbass, changes, relativeroot = tuple(
        chord_features[f]
        for f in ("numeral", "form", "figbass", "changes", "relativeroot")
    )
    return features2tpcs(
        numeral=numeral,
        form=form,
        figbass=figbass,
        changes=changes,
        relativeroot=relativeroot,
        logger=logger,
        **kwargs,
    )


def transpose(e, n):
    """Add `n` to all elements `e` recursively."""
    return map2elements(e, lambda x: x + n)


def parse_ignored_warnings(
    messages: Collection[str],
) -> Iterator[Tuple[str, Tuple[int]]]:
    """Turns a list of log messages into an iterator of (logger_name, (message_info, ...)) pairs.
    Log messages consist of a header of the shape WARNING_ENUM_MEMBER (enum_value, [mc, more_info...]) ms3.(
    Parse|Corpus).corpus.piece [-- potentially more, irrelevant stuff].
    The header might be followed by several lines of comments, each beginning with a space or tab.
    """
    if isinstance(messages, str):
        yield from parse_ignored_warnings([messages])
    else:
        for message in messages:
            if "\n" in message:
                yield from parse_ignored_warnings(message.split("\n"))
            elif message == "":
                continue
            elif message[0] in (" ", "\t", "#"):
                # if several lines of a warning were copied, use only the first one
                continue
            else:
                try:
                    # if the annotator copied too much, cut off the redundant information at the end
                    redundant = message.index(" --")
                    message = message[:redundant]
                except ValueError:
                    pass
                message = message.strip()
                split_re = r"^(.*) (\S+)$"
                try:
                    msg, logger_name = re.match(split_re, message).groups()
                except AttributeError:
                    print(
                        f"The following message could not be split, apparently it does not end with the logger_name: "
                        f"{message}"
                    )
                    raise
                if msg[-1] != ")":
                    if any(
                        msg.startswith(level)
                        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
                    ):
                        # message_id has not (yet) specified for this log message and is ignored;
                        # a warning could be implemented at this point
                        continue
                    else:
                        raise ValueError(f"Unexpected log message format: {msg}")
                tuple_start = msg.index("(") + 1
                tuple_str = msg[tuple_start:-1]
                info = literal_eval(tuple_str)
                yield logger_name, info


def ignored_warnings2dict(messages: Collection[str]) -> Dict[str, List[Tuple[int]]]:
    """

    Args:
      messages:

    Returns:
      {logger_name -> [ignored_warnings]} dict.
    """
    ignored_warnings = defaultdict(list)
    for logger_name, info in parse_ignored_warnings(messages):
        ignored_warnings[logger_name].append(info)
    return dict(ignored_warnings)


def parse_ignored_warnings_file(path: str) -> Dict[str, List[Tuple[int, Tuple[int]]]]:
    """Parse file with log messages that have to be ignored to the dict.
    The expected structure of message: warning_type (warning_type_id, *integers) file
    Example of message: INCORRECT_VOLTA_MN_WARNING (2,
    94) ms3.Parse.mixed_files.Did03M-Son_regina-1762-Sarti.mscx.MeasureList

    Parameters
    ----------
    key : :obj:`str`
        | Path to IGNORED_WARNINGS

    Returns
    -------
    :obj: dict
        {logger_name: [(message_id, label_of_message), (message_id, label_of_message), ...]}.
    """
    path = resolve_dir(path)
    messages = open(path, "r", encoding="utf-8").readlines()
    return ignored_warnings2dict(messages)


def overlapping_chunk_per_interval(
    df: pd.DataFrame, intervals: List[pd.Interval], truncate: bool = True
) -> Dict[pd.Interval, pd.DataFrame]:
    """For each interval, create a chunk of the given DataFrame based on its IntervalIndex.
        This is an optimized algorithm compared to calling IntervalIndex.overlaps(interval) for each
        given interval, with the additional advantage that it will not discard rows where the
        interval is zero, such as [25.0, 25.0).

        Parameters
        ----------
        df : :obj:`pandas.DataFrame`
            The DataFrame is expected to come with an IntervalIndex and contain the columns 'quarterbeats' and
    'duration_qb'.
            Those can be obtained through ``Parse.get_lists(interval_index=True)`` or
            ``Parse.iter_transformed(interval_index=True)``.
        intervals : :obj:`list` of :obj:`pd.Interval`
            The intervals defining the chunks' dimensions. Expected to be non-overlapping and monotonically increasing.
        truncate : :obj:`bool`, optional
            Defaults to True, meaning that the interval index and the 'duration_qb' will be adapted for overlapping
            intervals.
            Pass False to get chunks with all overlapping intervals as they are.

        Returns
        -------
        :obj:`dict`
            {interval -> chunk}
    """
    lefts = (
        df.index.left.values
    )  # lefts and rights will get shorter (potentially) with every
    rights = (
        df.index.right.values
    )  # interval, in order to reduce the time for comparing values
    chunks = {}
    current_start_mask = np.ones(len(df.index), dtype=bool)  # length remains the same
    for iv in intervals:
        # assumes intervals are non-overlapping and monotonically increasing
        l, r = iv.left, iv.right
        # never again check events ending before the current interval's start
        not_ending_before_l = rights >= l
        lefts = lefts[not_ending_before_l]
        rights = rights[not_ending_before_l]
        current_start_mask[current_start_mask] = not_ending_before_l
        starting_before_r = r > lefts
        not_ending_on_l_except_empty = (rights != l) | (lefts == l)
        overlapping = starting_before_r & not_ending_on_l_except_empty
        bool_mask = current_start_mask.copy()
        bool_mask[current_start_mask] = overlapping
        chunk = df[bool_mask].copy()
        if truncate:
            new_lefts, new_rights = lefts[overlapping], rights[overlapping]
            starting_before_l, ending_after_r = (new_lefts < l), (new_rights > r)
            if starting_before_l.sum() > 0 or ending_after_r.sum() > 0:
                new_lefts[starting_before_l] = l
                new_rights[ending_after_r] = r
                chunk.index = pd.IntervalIndex.from_arrays(
                    new_lefts, new_rights, closed="left"
                )
                chunk.duration_qb = new_rights - new_lefts
                chunk.quarterbeats = new_lefts
                chunk.sort_values(["quarterbeats", "duration_qb"], inplace=True)
        chunks[iv] = chunk
    return chunks


def infer_tsv_type(df: pd.DataFrame) -> Optional[str]:
    """Infers the contents of a DataFrame from the presence of particular columns."""
    type2cols = {
        "notes": ["tpc", "midi"],
        "events": ["Chord/durationType", "Rest/durationType"],
        "chords": ["chord_id"],
        "rests": ["nominal_duration"],
        "expanded": ["numeral"],
        "labels": ["harmony_layer", "label", "label_type"],
        "measures": ["act_dur"],
        "cadences": ["cadence"],
        "metadata": ["piece"],
        "form_labels": ["form_label", "a"],
    }
    for t, columns in type2cols.items():
        if any(c in df.columns for c in columns):
            if t == "expanded":
                # check if it's cadences only
                if "cadence" in df.columns and all(df.cadence.notna()):
                    return "cadences"
                else:
                    return "expanded"
            if t == "notes":
                # check if it contains rests, too
                if "tpc" in df.columns and any(df.tpc.isna()):
                    return "notes_and_rests"
                else:
                    return "notes"
            return t
    if any(c in df.columns for c in ["mc", "mn"]):
        return "labels"
    return "unknown"


def reduce_dataframe_duration_to_first_row(df: pd.DataFrame) -> pd.DataFrame:
    """Reduces a DataFrame to its row and updates the duration_qb column to reflect the reduced duration.

    Args:
      df: Dataframe of which to keep only the first row. If it has an IntervalIndex, the interval is updated to
          reflect the whole duration.

    Returns:
      DataFrame with one row.
    """
    if len(df) == 1:
        return df
    idx = df.index
    first_loc = idx[0]
    row = df.iloc[[0]]
    # if isinstance(ix, pd.Interval) or (isinstance(ix, tuple) and isinstance(ix[-1], pd.Interval)):
    if isinstance(idx, pd.IntervalIndex):
        start = min(idx.left)
        end = max(idx.right)
        iv = pd.Interval(start, end, closed=idx.closed)
        row.index = pd.IntervalIndex([iv])
        row.loc[iv, "duration_qb"] = iv.length
    else:
        new_duration = df.duration_qb.sum()
        row.loc[first_loc, "duration_qb"] = new_duration
    return row


@dataclass
class File:
    """Storing path and file name information for one file."""

    ix: int
    """Index integer (ID)"""
    type: str
    """Recognized type ∈ :attr:`ms3._typing.Facet`"""
    file: str
    """File name including extension."""
    piece: str
    """piece excluding the suffix (after registering the file with a :obj:`Piece`)."""
    fext: str
    """File extensions."""
    subdir: str
    """Directory relative to the corpus path (e.g. './MS3')."""
    corpus_path: str
    """Absolute path of the file's parent directory that is considered as corpus directory."""
    rel_path: str
    """File path relative to the corpus path. Equivalent to <subdir>/<file>."""
    full_path: str
    """Absolute file path."""
    directory: str
    """Absolute folder path where the file is located."""
    suffix: str = ""
    """Upon registering the File with a :obj:`Piece`, if the current piece has a suffix compared to the Piece's piece,
    suffix is removed from the File object's piece field and added to the suffix field."""
    commit_sha: str = ""
    """The the file has been retrieved from a particular git revision, this is set to the revision's hash."""

    def __repr__(self):
        suffix = "" if self.suffix == "" else f", suffix: {self.suffix}."
        commit = "" if self.commit_sha == "" else f"@{self.commit_sha[:7]}"
        return f"{self.ix}: '{self.rel_path}'{commit}{suffix}"

    def replace_extension(self, new_extension: str, **kwargs) -> Self:
        if new_extension[0] != ".":
            new_extension = "." + new_extension
        old_ext_len = len(self.fext)
        new_vals = {}
        for field in ("file", "fext", "rel_path", "full_path"):
            old_val = getattr(self, field)
            new_vals[field] = old_val[:-old_ext_len] + new_extension
        return replace(self, **new_vals, **kwargs)

    @classmethod
    def from_corpus_path(
        cls,
        corpus_path: str,
        filename: str,
        ftype: Optional[str] = None,
        subdir=".",
        ix: int = -1,
    ):
        """Creates File object from individual components

        Args:
            corpus_path: Root directory of the file's corpus.
            filename: Full file name including suffixes and extensions.
            ftype: File type (used as default folder name for creating file_paths).
            subdir: relative directory appended to corpus_path, defaults to '.', i.e. no subfolder.
            ix: Arbitrary index number, defaults to -1.
        """
        full_path = os.path.realpath(os.path.join(corpus_path, subdir, filename))
        file_name, file_ext = os.path.splitext(filename)
        rel_path = os.path.join(subdir, filename)
        if ftype is None:
            file_type = path2type(full_path)
        else:
            file_type = ftype
        return cls(
            ix=ix,
            type=file_type,
            file=filename,
            piece=file_name,
            fext=file_ext,
            subdir=subdir,
            corpus_path=corpus_path,
            rel_path=rel_path,
            full_path=full_path,
            directory=os.path.dirname(full_path),
            suffix="",
        )


def automatically_choose_from_disambiguated_files(
    disambiguated_choices: Dict[str, File], piece: str, file_type: str, logger=None
) -> File:
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if len(disambiguated_choices) == 1:
        return list(disambiguated_choices.keys())[0]
    disamb_series = pd.Series(disambiguated_choices)
    files = list(disambiguated_choices.values())
    files_df = pd.DataFrame(files, index=disamb_series.index)
    choice_between_n = len(files)
    if file_type == "scores":
        # if a score is requested, check if there is only a single MSCX or otherwise MSCZ file and pick that
        fexts = files_df.fext.str.lower()
        fext_counts = fexts.value_counts()
        if ".mscx" in fext_counts:
            if fext_counts[".mscx"] == 1:
                selected_file = disamb_series[fexts == ".mscx"].iloc[0]
                logger.debug(
                    f"In order to pick one from the {choice_between_n} scores with piece '{piece}', "
                    f"'{selected_file.rel_path}' was selected because it is the only "
                    f"one in MSCX format."
                )
                return selected_file
        elif ".mscz" in fext_counts and fext_counts[".mscz"] == 1:
            selected_file = disamb_series[fexts == ".mscz"].iloc[0]
            logger.debug(
                f"In order to pick one from the {choice_between_n} scores with piece '{piece}', "
                f"'{selected_file.rel_path}' was selected because it is the only "
                f"one in MSCZ format."
            )
            return selected_file
    # as first disambiguation criterion, check if the shortest disambiguation string pertains to 1 file only and pick
    # that
    disamb_str_lengths = pd.Series(
        disamb_series.index.map(len), index=disamb_series.index
    )
    shortest_length_selector = disamb_str_lengths == disamb_str_lengths.min()
    n_have_shortest_length = shortest_length_selector.sum()
    if n_have_shortest_length == 1:
        ix = disamb_str_lengths.idxmin()
        selected_file = disamb_series.loc[ix]
        logger.debug(
            f"In order to pick one from the {choice_between_n} '{file_type}' with piece '{piece}', the one with the "
            f"shortest disambiguating string '{ix}' was selected."
        )
        return selected_file
    if file_type != "unknown":
        # otherwise, check if only one file is lying in a directory with default name
        subdirs = files_df.subdir
        default_components = file_type2path_component_map()[file_type]
        default_components_regex = "|".join(
            comp.replace(".", r"\.") for comp in default_components
        )
        default_selector = subdirs.str.contains(default_components_regex, regex=True)
        if default_selector.sum() == 1:
            subdir = subdirs[default_selector].iloc[0]
            selected_file = disamb_series[default_selector].iloc[0]
            logger.debug(
                f"In order to pick one from the {choice_between_n} '{file_type}' with piece '{piece}', the one in the "
                f"default subdir '{subdir}' was selected."
            )
            return selected_file
        # or if only one file contains a default name in its suffix
        suffixes = files_df.suffix
        default_selector = suffixes.str.contains(default_components_regex, regex=True)
        if default_selector.sum() == 1:
            suffix = suffixes[default_selector].iloc[0]
            selected_file = disamb_series[default_selector].iloc[0]
            logger.debug(
                f"In order to pick one from the {choice_between_n} '{file_type}' with piece '{piece}', the one in the "
                f"default suffix '{suffix}' was selected."
            )
            return selected_file
    # if no file was selected, try again with only those having the shortest disambiguation strings
    if shortest_length_selector.all():
        # if all disambiguation strings already have the shortest length, as a last resort
        # fall back to the lexigographically first
        sorted_disamb_series = disamb_series.sort_index()
        disamb = sorted_disamb_series.index[0]
        selected_file = sorted_disamb_series.iloc[0]
        logger.warning(
            f"Unable to automatically choose from the {choice_between_n} '{file_type}' with piece '{piece}'. I'm "
            f"picking '{selected_file.rel_path}' "
            f"because its disambiguation string '{disamb}' is the lexicographically first among "
            f"{sorted_disamb_series.index.to_list()}"
        )
        return selected_file
    only_shortest_disamb_str = disamb_series[shortest_length_selector].to_dict()
    logger.info(
        f"After the first unsuccessful attempt to choose from {choice_between_n} '{file_type}' with piece '{piece}', "
        f"trying again "
        f"after reducing the choices to the {shortest_length_selector.sum()} with the shortest disambiguation strings."
    )
    return automatically_choose_from_disambiguated_files(
        only_shortest_disamb_str, piece, file_type
    )


def ask_user_to_choose(query: str, choices: Collection[Any]) -> Optional[Any]:
    """Ask user to input an integer and return the nth choice selected by the user."""
    n_choices = len(choices)
    range_str = f"1-{n_choices}"
    while True:
        s = input(query)
        try:
            int_i = int(s)
        except Exception:
            print(f"Value '{s}' could not be converted to an integer.")
            continue
        if not (0 <= int_i <= n_choices):
            print(f"Value '{s}' is not within {range_str}.")
            continue
        if int_i == 0:
            return None
        return choices[int_i - 1]


def ask_user_to_choose_from_disambiguated_files(
    disambiguated_choices: Dict[str, File], piece: str, file_type: str = ""
) -> Optional[File]:
    sorted_keys = sorted(disambiguated_choices.keys(), key=lambda s: (len(s), s))
    disambiguated_choices = {k: disambiguated_choices[k] for k in sorted_keys}
    file_list = list(disambiguated_choices.values())
    disamb_strings = pd.Series(disambiguated_choices.keys(), name="disambiguation_str")
    choices_df = pd.concat(
        [disamb_strings, pd.DataFrame(file_list)[["rel_path", "type", "ix"]]], axis=1
    )
    choices_df.index = pd.Index(range(1, len(file_list) + 1), name="select:")
    range_str = f"1-{len(disambiguated_choices)}"
    query = f"Selection [{range_str}]: "
    print(f"Several '{file_type}' available for '{piece}':\n{choices_df.to_string()}")
    print(
        f"Please select one of the files by passing an integer between {range_str} (or 0 for none):"
    )
    return ask_user_to_choose(query, file_list)


def disambiguate_files(
    files: Collection[File],
    piece: str,
    file_type: str,
    choose: Literal["auto", "ask"] = "auto",
    logger=None,
) -> Optional[File]:
    """Receives a collection of :obj:`File` with the aim to pick one of them.
    First, a dictionary is created where the keys are disambiguation strings based on the files' paths and
    suffixes.

    Args:
      files:
      choose: If 'auto' (default), the file with the shortest disambiguation string is chosen. Set to True
          if you want to be asked to manually choose a file.

    Returns:
      The selected file.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    n_files = len(files)
    if n_files == 0:
        return
    files = tuple(files)
    if n_files == 1:
        return files[0]
    if choose not in ("auto", "ask"):
        logger.info(
            f"The value for choose needs to be 'auto' or 'ask', not {choose}. Setting to 'auto'."
        )
        choose = "auto"
    disambiguation_dict = files2disambiguation_dict(files, logger=logger)
    if choose == "ask":
        return ask_user_to_choose_from_disambiguated_files(
            disambiguation_dict, piece, file_type
        )
    return automatically_choose_from_disambiguated_files(
        disambiguation_dict, piece, file_type, logger=logger
    )


def files2disambiguation_dict(
    files: Collection[File], include_disambiguator: bool = False, logger=None
) -> FileDict:
    """Takes a list of :class:`File` returns a dictionary with disambiguating strings based on path components.
    of distinct strings to distinguish files pertaining to the same type."""
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    n_files = len(files)
    if n_files == 0:
        return {}
    files = tuple(files)
    if n_files == 1:
        f = files[0]
        return {f.type: f}
    disambiguation = [f.type for f in files]
    if len(set(disambiguation)) == n_files:
        # done disambiguating
        return dict(zip(disambiguation, files))
    if include_disambiguator and len(set(disambiguation)) > 1:
        logger.warning(
            f"Including the disambiguator removes the facet name, but the files pertain to "
            f"several facets: {set(disambiguation)}"
        )
    # first, try to disambiguate based on the files' sub-directories
    subdirs = []
    for f in files:
        file_type = f.type
        subdir = f.subdir.strip(r"\/.")
        if subdir.startswith(file_type):
            len_fext = len(file_type)
            subdir = subdir[len_fext:]
        if subdir.strip(r"\/") == "":
            subdir = ""
        subdirs.append(subdir)
    if len(set(subdirs)) > 1:
        # files can (partially) be disambiguated because they are in different sub-directories
        if include_disambiguator:
            disambiguation = [
                f"subdir: {'.' if subdir == '' else subdir}"
                for disamb, subdir in zip(disambiguation, subdirs)
            ]
        else:
            disambiguation = [
                os.path.join(disamb, subdir)
                for disamb, subdir in zip(disambiguation, subdirs)
            ]
    if len(set(disambiguation)) == n_files:
        # done disambiguating
        return dict(zip(disambiguation, files))
    # next, try adding detected suffixes
    for ix, f in enumerate(files):
        if f.suffix != "":
            if include_disambiguator:
                disambiguation[ix] += f", suffix: {f.suffix}"
            else:
                disambiguation[ix] += f"[{f.suffix}]"
    if len(set(disambiguation)) == n_files:
        # done disambiguating
        return dict(zip(disambiguation, files))
    # now, add file extensions to disambiguate further
    if len(set(f.fext for f in files)) > 1:
        for ix, f in enumerate(files):
            if include_disambiguator:
                disambiguation[ix] += f", fext: {f.fext}"
            else:
                disambiguation[ix] += f.fext
    if len(set(disambiguation)) == n_files:
        # done disambiguating
        return dict(zip(disambiguation, files))
    str_counts = Counter(disambiguation)
    duplicate_disambiguation_strings = [s for s, cnt in str_counts.items() if cnt > 1]
    ambiguate_files = {
        s: [f for disamb, f in zip(disambiguation, files) if disamb == s]
        for s in duplicate_disambiguation_strings
    }
    result = dict(zip(disambiguation, files))
    remaining_ones = {s: result[s] for s in duplicate_disambiguation_strings}
    logger.warning(
        f"The following files could not be ambiguated: {ambiguate_files}.\n"
        f"In the result, only these remain: {remaining_ones}."
    )
    return result


def literal_type2tuple(typ: TypeVar) -> Tuple[str]:
    """Turns the first Literal included in the TypeVar into a list of values. The first literal value
    needs to be a string, otherwise the function may lead to unexpected behaviour.
    """
    result = []
    for arg in typ.__args__:
        if isinstance(arg, str):
            result.append(arg)
        else:
            result.extend(literal_type2tuple(arg))
    return result


@cache
def argument_and_literal_type2list(
    argument: Union[str, Tuple[str], Literal[None]],
    typ: Optional[Union[TypeVar, Tuple[str]]] = None,
    none_means_all: bool = True,
    logger=None,
) -> Optional[List[str]]:
    """Makes sure that an input value is a list of strings and that all strings are valid w.r.t. to
    the type's expected literal values (strings).

    Args:
      argument:
          If string, wrapped in a list, otherwise expected to be a tuple of strings (passing a list will fail).
          If None, a list of all possible values according to the type is returned if none_means_all.
      typ:
          A typing.Literal declaration or a TypeVar where the first component is one, or a tuple of allowed values.
          All allowed values should be strings.
      none_means_all:
          By default, None values are replaced with all allowed values, if specified.
          Pass False to return None in this case.

    Returns:
      The list of accepted strings.
      The list of rejected strings.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if typ is None:
        allowed = None
    else:
        if isinstance(typ, tuple):
            allowed = typ
        else:
            allowed = literal_type2tuple(typ)
    if argument is None:
        if none_means_all and allowed is not None:
            return list(allowed)
        else:
            return
    if isinstance(argument, str):
        argument = [argument]
    if allowed is None:
        return argument
    else:
        singular_dict = {allwd[:-1]: allwd for allwd in allowed}
    accepted, rejected = [], []
    for arg in argument:
        if arg in allowed:
            accepted.append(arg)
        elif arg in singular_dict:
            accepted.append(singular_dict[arg])
        else:
            rejected.append(arg)
    n_rejected = len(rejected)
    if n_rejected > 0:
        if n_rejected == 1:
            logger.warning(
                f"This is not an accepted value: {rejected[0]}\n"
                f"Choose from {allowed}"
            )
        else:
            logger.warning(
                f"These are not accepted value, only: {rejected}"
                f"Choose from {allowed}"
            )
    if len(accepted) > 0:
        return accepted
    logger.warning(f"Pass at least one of {allowed}.")
    return


L = TypeVar("L")


def check_argument_against_literal_type(
    argument: str, typ: L, logger=None
) -> Optional[L]:
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if not isinstance(argument, str):
        logger.warning(f"Argument needs to be a string, not '{type(argument)}'")
        return None
    allowed = literal_type2tuple(typ)
    singular_dict = {allwd[:-1]: allwd for allwd in allowed}
    if argument not in allowed and argument not in singular_dict:
        logger.warning(f"Invalid argument '{argument}'. Pass one of {allowed}")
        return None
    if argument in singular_dict:
        return singular_dict[argument]
    return argument


def resolve_facets_param(
    facets, facet_type_var: TypeVar = Facet, none_means_all=True, logger=None
):
    """Like :func:`argument_and_literal_type2list`, but also resolves 'tsv' to all non-score facets."""
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if isinstance(facets, str) and facets in ("tsv", "tsvs"):
        selected_facets = list(literal_type2tuple(facet_type_var))
        if "scores" in selected_facets:
            selected_facets.remove("scores")
    else:
        if isinstance(facets, list):
            facets = tuple(facets)
        selected_facets = argument_and_literal_type2list(
            facets, facet_type_var, none_means_all=none_means_all, logger=logger
        )
    # logger.debug(f"Resolved argument '{facets}' to {selected_facets}.")
    return selected_facets


def bold_font(s):
    return f"\033[1m{s}\033[0;0m"


def available_views2str(views_dict: ViewDict, active_view_name: str = None) -> str:
    view_names = {
        key: view.name if key is None else key for key, view in views_dict.items()
    }
    current_view = view_names[active_view_name]
    view_list = [bold_font(current_view)] + [
        name for name in view_names.values() if name != current_view
    ]
    return f"[{'|'.join(view_list)}]\n"


def unpack_json_paths(paths: Collection[str], logger=None) -> None:
    """Mutates the list with paths by replacing .json files with the list (of paths) contained in them."""
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    json_ixs = [i for i, p in enumerate(paths) if p.endswith(".json")]
    if len(json_ixs) > 0:
        for i in reversed(json_ixs):
            try:
                with open(paths[i]) as f:
                    loaded_paths = json.load(f)
                paths.extend(loaded_paths)
                logger.info(
                    f"Unpacked the {len(loaded_paths)} paths found in {paths[i]}."
                )
                del paths[i]
            except Exception:
                logger.info(
                    f"Could not load paths from {paths[i]} because of the following error(s):\n{sys.exc_info()[1]}"
                )


def resolve_paths_argument(
    paths: Union[str, Collection[str]], files: bool = True, logger=None
) -> List[str]:
    """Makes sure that the given path(s) exists(s) and filters out those that don't.

    Args:
      paths: One or several paths given as strings.
      files: By default, only file paths are returned. Set to False to return only folders.

    Returns:

    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if isinstance(paths, str):
        paths = [paths]
    resolved_paths = [resolve_dir(p) for p in paths]
    if files:
        not_a_file = [p for p in resolved_paths if not os.path.isfile(p)]
        if len(not_a_file) > 0:
            if len(not_a_file) == 1:
                msg = f"No existing file at {not_a_file[0]}."
            else:
                msg = f"These are not paths of existing files: {not_a_file}"
            logger.warning(msg)
            resolved_paths = [p for p in resolved_paths if os.path.isfile(p)]
    else:
        not_a_folder = [p for p in resolved_paths if not os.path.isdir(p)]
        if len(not_a_folder) > 0:
            if len(not_a_folder) == 1:
                msg = f"{not_a_folder[0]} is not a path to an existing folder."
            else:
                msg = f"These are not paths of existing folders: {not_a_folder}"
            logger.warning(msg)
            resolved_paths = [p for p in resolved_paths if os.path.isdir(p)]
    return resolved_paths


def compute_path_from_file(
    file: File,
    root_dir: Optional[str] = None,
    folder: Optional[str] = None,
    logger=None,
) -> str:
    """
    Constructs a path based on the arguments.

    Args:
      file: This function uses the fields corpus_path, subdir, and type.
      root_dir:
          Defaults to None, meaning that the path is constructed based on the corpus_path.
          Pass a directory to construct the path relative to it instead. If ``folder`` is an absolute path,
          ``root_dir`` is ignored.
      folder:
          * If ``folder`` is None (default), the files' type will be appended to the ``root_dir``.
          * If ``folder`` is an absolute path, ``root_dir`` will be ignored.
          * If ``folder`` is a relative path starting with a dot ``.`` the relative path is appended to the file's
          subdir.
            For example, ``..\notes`` will resolve to a sibling directory of the one where the ``file`` is located.
          * If ``folder`` is a relative path that does not begin with a dot ``.``, it will be appended to the
            ``root_dir``.
          * If ``folder`` == '' (empty string), the result will be `root_dir`.

    Returns:
      The constructed directory path.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if folder is not None and (os.path.isabs(folder) or "~" in folder):
        folder = resolve_dir(folder)
        path = folder
    else:
        root = file.corpus_path if root_dir is None else resolve_dir(root_dir)
        if folder is None:
            path = os.path.join(root, file.type)
        elif folder == "":
            path = root
        elif folder[0] == ".":
            path = os.path.abspath(os.path.join(root, file.subdir, folder))
        else:
            path = os.path.abspath(os.path.join(root, folder))
    return path


def make_file_path(
    file: File, root_dir=None, folder: str = None, suffix: str = "", fext: str = ".tsv"
):
    """Constructs a file path based on the arguments.

    Args:
      file: This function uses the fields piece, corpus_path, subdir, and type.
      root_dir:
        Defaults to None, meaning that the path is constructed based on the corpus_path.
        Pass a directory to construct the path relative to it instead. If ``folder`` is an absolute path,
        ``root_dir`` is ignored.
      folder:
        Different behaviours are available. Note that only the third option ensures that file paths are distinct for
        files that have identical pieces but are located in different subdirectories of the same corpus.
        * If ``folder`` is None (default), the files' type will be appended to the ``root_dir``.
        * If ``folder`` is an absolute path, ``root_dir`` will be ignored.
        * If ``folder`` is a relative path starting with a dot ``.`` the relative path is appended to the file's subdir.
          For example, ``..\notes`` will resolve to a sibling directory of the one where the ``file`` is located.
        * If ``folder`` is a relative path that does not begin with a dot ``.``, it will be appended to the
          ``root_dir``.
      suffix: String to append to the file's piece.
      fext: File extension to append to the (piece+suffix). Defaults to ``.tsv``.

    Returns:
      The constructed file path.
    """
    assert fext is not None, ""
    path = compute_path_from_file(file, root_dir=root_dir, folder=folder)
    if suffix is None:
        suffix = ""
    piece = file.piece + suffix + fext
    return os.path.join(path, piece)


def string2identifier(s: str, remove_leading_underscore: bool = True) -> str:
    """Transform a string in a way that it can be used as identifier (variable or attribute name).
    Solution by Kenan Banks on https://stackoverflow.com/a/3303361
    """
    # Remove invalid characters
    s = re.sub("[^0-9a-zA-Z_]", "", s)

    # Remove leading characters until we find a letter or underscore
    regex = "^[^a-zA-Z]+" if remove_leading_underscore else "^[^a-zA-Z_]+"
    s = re.sub(regex, "", s)

    return s


@cache
def resolve_git_revision(
    repo_path: str, git_revision: Optional[str], logger=None
) -> Optional[str]:
    """Returns the commit hash for the given revision.

    Args:
        repo_path:
        git_revision:
            Any specifier that git understands (branch, tag, commit hash, "HEAD", etc.). In addition,
            "LATEST_VERSION" can be passed to get the tag with the highest version number. None defaults
            to "HEAD".
        logger:

    Returns:
        Hash of the commit that corresponds to the given revision specifier.
    """
    commit = get_git_commit(repo_path, git_revision, logger=logger)
    if commit is None:
        return None
    return commit.hexsha


def parse_tsv_file_at_git_revision(
    file: File, git_revision: str, repo_path: Optional[str] = None, logger=None
) -> FileDataframeTupleMaybe:
    """
       Pass a File object of a TSV file and an identifier for a git revision to retrieve the parsed TSV file at that
    commit.
       The file needs to have existed at the revision in question.

       Args:
         file:
         git_revision:
         repo_path:

       Returns:

    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if file.type == "scores":
        raise NotImplementedError(
            "Parsing older revisions of scores is not implemented. Checkout the revision yourself."
        )
    if repo_path is None:
        repo_path = file.corpus_path
    commit = get_git_commit(repo_path, git_revision, logger=logger)
    if commit is None:
        return None, None
    commit_sha = commit.hexsha
    short_sha = commit_sha[:7]
    commit_info = f"{short_sha} with message '{commit.message.strip()}'"
    if short_sha != git_revision:
        logger.debug(f"Resolved '{git_revision}' to '{short_sha}'.")
    rel_path = os.path.normpath(file.rel_path).replace("\\", "/")
    try:
        targetfile = commit.tree / rel_path
    except KeyError:
        # add logic here to find older path when the file has been moved or renamed
        logger.error(f"{rel_path} did not exist at commit {commit_info}.")
        return None, None
    try:
        with io.BytesIO(targetfile.data_stream.read()) as f:
            parsed = load_tsv(f)
    except Exception as e:
        logger.error(
            f"Parsing {rel_path} @ commit {commit_info} failed with the following exception:\n{e}"
        )
        return None, None
    new_file = replace(file, commit_sha=commit_sha)
    return new_file, parsed


def write_messages_to_file_or_remove(
    warnings_file: str, warnings: List[str], header: str, logger=None
) -> bool:
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    warnings_path = os.path.dirname(warnings_file)
    if len(warnings) > 0:
        if warnings_path:
            os.makedirs(warnings_path, exist_ok=True)
        header = f"{header}\n{'=' * len(header)}\n\n"
        with open(warnings_file, "w", encoding="utf-8") as f:
            f.write(header)
            f.write("\n".join(warnings))
        return True
    elif os.path.isfile(warnings_file):
        logger.info(f"Problems seem to be solved, removing {warnings_file}")
        os.remove(warnings_file)
    return False


def write_warnings_to_file(
    warnings_file: str, warnings: List[str], header: Optional[str] = None, logger=None
):
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if header is None:
        header = "Warnings encountered during the last execution of ms3 review"
    if write_messages_to_file_or_remove(warnings_file, warnings, header):
        logger.info(f"Written warnings to {warnings_file}.")


def write_validation_errors_to_file(
    errors_file: str, errors: List[str], header: Optional[str] = None, logger=None
):
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if header is None:
        header = "Validation error encountered after file creation"
    if write_messages_to_file_or_remove(errors_file, errors, header):
        logger.info(f"Written validation errors to {errors_file}.")


def replace_extension(filepath: str, new_extension: str) -> str:
    if new_extension[0] != ".":
        new_extension = "." + new_extension
    return os.path.splitext(filepath)[0] + new_extension


# region Functions for writing BeautifulSoup to MSCX file


def escape_string(s):
    return (
        str(s)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def opening_tag(node, closed=False):
    result = f"<{node.name}"
    attributes = node.attrs
    if len(attributes) > 0:
        result += " " + " ".join(
            f'{attr}="{escape_string(value)}"' for attr, value in attributes.items()
        )
    closing = "/" if closed else ""
    return f"{result}{closing}>"


def closing_tag(node_name):
    return f"</{node_name}>"


def make_oneliner(node):
    """Pass a tag of which the layout does not spread over several lines."""
    result = opening_tag(node)
    for c in node.children:
        if isinstance(c, bs4.element.Tag):
            result += make_oneliner(c)
        else:
            result += escape_string(c)
    result += closing_tag(node.name)
    return result


def format_node(node, indent):
    """Recursively format Beautifulsoup tag as in an MSCX file."""
    nxt_indent = indent + 2
    space = indent * " "
    node_name = node.name
    # The following tags are exceptionally not abbreviated when empty,
    # so for instance you get <metaTag></metaTag> and not <metaTag/>
    if node_name in [
        "continueAt",
        "continueText",
        "endText",
        "LayerTag",
        "metaTag",
        "name",
        "programRevision",
        "text",
        "trackName",
    ]:
        return f"{space}{make_oneliner(node)}\n"
    children = node.find_all(recursive=False)
    if len(children) > 0:
        result = f"{space}{opening_tag(node)}\n"
        result += "".join(format_node(child, nxt_indent) for child in children)
        result += f"{nxt_indent * ' '}{closing_tag(node_name)}\n"
        return result
    if node.string == "\n":
        return (
            f"{space}{opening_tag(node)}\n{nxt_indent * ' '}{closing_tag(node_name)}\n"
        )
    if node.string is None:
        return f"{space}{opening_tag(node, closed=True)}\n"
    return f"{space}{make_oneliner(node)}\n"


def bs4_to_mscx(soup: bs4.BeautifulSoup):
    """Turn the BeautifulSoup into a string representing an MSCX file"""
    assert soup is not None, "BeautifulSoup XML structure is None"
    initial_tag = """<?xml version="1.0" encoding="UTF-8"?>\n"""
    first_tag = soup.find()
    return initial_tag + format_node(first_tag, indent=0)


def write_score_to_handler(
    soup: bs4.BeautifulSoup,
    file_handler: IO,
    logger=None,
) -> bool:
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    try:
        mscx_string = bs4_to_mscx(soup)
    except Exception as e:
        logger.error(f"Couldn't output score because of the following error:\n{e}")
        return False
    file_handler.write(mscx_string)
    return True


def write_soup_to_mscx_file(
    soup: bs4.Tag,
    mscx_path: str,
    overwrite: bool = False,
    logger=None,
) -> bool:
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    if not mscx_path.endswith(".mscx"):
        logger.error(f"File {mscx_path} does not have the .mscx extension.")
        return False
    filepath = resolve_dir(mscx_path)
    if os.path.isfile(filepath) and not overwrite:
        logger.error(f"File {filepath} already exists and overwrite is set to False.")
        return False
    with open(filepath, "w", encoding="utf-8") as file_handler:
        return write_score_to_handler(soup, file_handler, logger=logger)


# endregion Functions for writing BeautifulSoup to MSCX file
# region concatenating sub-corpus metadata


def update_relative_paths_with_corpus_dirs(concatenated: pd.DataFrame) -> None:
    """Assumes that the first index level includes folder names and adds them to the relative paths.
    The first column to be updated is "subdirectory" (default name) or "rel_paths" (old name).
    The second column, if present, is "rel_path". The operation is performed in-place.
    """
    try:
        rel_path_col = next(
            col for col in ("subdirectory", "rel_paths") if col in concatenated.columns
        )
    except StopIteration:
        raise ValueError(
            "Metadata is expected to come with a column called 'subdirectory' or (previously) 'rel_paths'."
        )
    rel_paths = [
        os.path.join(corpus, rel_path)
        for corpus, rel_path in zip(
            concatenated.index.get_level_values(0), concatenated[rel_path_col].values
        )
    ]
    concatenated.loc[:, rel_path_col] = rel_paths
    if "rel_path" in concatenated.columns:
        rel_paths = [
            os.path.join(corpus, rel_path)
            for corpus, rel_path in zip(
                concatenated.index.get_level_values(0), concatenated.rel_path.values
            )
        ]
        concatenated.loc[:, "rel_path"] = rel_paths


def concat_metadata_dfs(corpus2metadata_df: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concats the dataframes corresponding to the metadata.tsv files of sub-corpora.
    The corpus names will be prepended as an additional index level and to the relative file paths in the column
    "subdirectory" (default name) or "rel_paths" (old name).

    Args:
        corpus2metadata_df: Dictionary mapping corpus names (i.e., folder names) to parsed metadata.tsv files.

    """
    try:
        concatenated = pd.concat(corpus2metadata_df)
    except AssertionError:
        info = "Levels: " + ", ".join(
            f"{key}: {df.index.nlevels} ({df.index.names})"
            for key, df in corpus2metadata_df.items()
        )
        print(f"Concatenation of DataFrames failed due to an alignment error. {info}")
        raise
    update_relative_paths_with_corpus_dirs(concatenated)
    concatenated = concatenated.droplevel(1)
    concatenated.index.rename("corpus", inplace=True)
    return concatenated


def concat_metadata_tsv_files_of_subdirs(path: str) -> pd.DataFrame:
    """Walk through the first level of subdirectories and concatenate their metadata.tsv files."""
    _, folders, _ = next(os.walk(path))
    corpus2tsv_path = {}
    for subdir in sorted(folders):
        potential = os.path.join(path, subdir, "metadata.tsv")
        if os.path.isfile(potential):
            corpus2tsv_path[subdir] = potential
    if len(corpus2tsv_path) == 0:
        return pd.DataFrame()
    corpus2metadata_df = {
        key: pd.read_csv(tsv_path, sep="\t", dtype="string")
        for key, tsv_path in corpus2tsv_path.items()
    }
    concatenated = concat_metadata_dfs(corpus2metadata_df)
    return concatenated


def concatenated_metadata2markdown(concatenated):
    try:
        fname_col = next(
            col for col in ("piece", "fname", "fnames") if col in concatenated.columns
        )
    except StopIteration:
        raise ValueError(
            "Metadata is expected to come with a column called 'piece' or (previously) 'fname' or 'fnames'."
        )
    rename4markdown = {
        fname_col: "file_name",
        "last_mn": "measures",
        "label_count": "labels",
        "harmony_version": "standard",
    }
    concatenated = concatenated.rename(columns=rename4markdown)
    existing_columns = [
        col for col in rename4markdown.values() if col in concatenated.columns
    ]
    result = "# Overview"
    for corpus_name, df in concatenated[existing_columns].groupby(level=0):
        heading = f"\n\n## {corpus_name}\n\n"
        md = str(dataframe2markdown(df.fillna("")))
        result += heading + md
    return result


def concat_metadata(
    meta_corpus_dir: str, out: str, tsv_name="concatenated_metadata.tsv", logger=None
):
    """Concatenate metadata.tsv files from the sub-corpora of a meta-corpus, adapt the file paths, update the README."""
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    concatenated = concat_metadata_tsv_files_of_subdirs(meta_corpus_dir)
    if len(concatenated) == 0:
        print(f"No metadata found in the child directories of {meta_corpus_dir}.")
        return
    tsv_path = os.path.join(out, tsv_name)
    write_tsv(concatenated, tsv_path)
    md_str = concatenated_metadata2markdown(concatenated)
    md_path = os.path.join(out, "README.md")
    overwrite_overview_section_in_markdown_file(
        file_path=md_path, md_str=md_str, logger=logger
    )


# endregion concatenating sub-corpus metadata
