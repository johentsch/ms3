import os, platform, re, shutil, subprocess
from collections import defaultdict, namedtuple
from collections.abc import Iterable
from contextlib import contextmanager
from fractions import Fraction as frac
from functools import reduce
from itertools import repeat, takewhile
from shutil import which
from tempfile import NamedTemporaryFile as Temp
from zipfile import ZipFile as Zip

import pandas as pd
import numpy as np
import webcolors
from pathos import multiprocessing
from tqdm import tqdm
from pytablewriter import MarkdownTableWriter

from .logger import function_logger, update_cfg

STANDARD_NAMES = ['notes', 'rests', 'notes_and_rests', 'measures', 'events', 'labels', 'chords', 'expanded',
                  'harmonies', 'cadences', 'form_labels', 'MS3', 'scores']
""":obj:`list`
Indicators for subcorpora: If a folder contains any file or folder beginning or ending on any of these names, it is 
considered to be a subcorpus by the function :py:func:`iterate_subcorpora`.
"""

DCML_REGEX = re.compile(r"""
^(\.?
    ((?P<globalkey>[a-gA-G](b*|\#*))\.)?
    ((?P<localkey>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)+)\.)?
    ((?P<pedal>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)+)\[)?
    (?P<chord>
        (?P<numeral>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))
        (?P<form>(%|o|\+|M|\+M))?
        (?P<figbass>(7|65|43|42|2|64|6))?
        (\((?P<changes>((\+|-|\^|v)?(b*|\#*)\d)+)\))?
        (/(?P<relativeroot>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)*))?
    )
    (?P<pedalend>\])?
)?
(\|(?P<cadence>((HC|PAC|IAC|DC|EC|PC)(\..+?)?)))?
(?P<phraseend>(\\\\|\}\{|\{|\}))?$
            """, re.VERBOSE)
""":obj:`str`
Constant with a regular expression that recognizes labels conforming to the DCML harmony annotation standard excluding those
consisting of two alternatives.
"""

DCML_DOUBLE_REGEX = re.compile(r"""
                                ^(?P<first>
                                  (\.?
                                    ((?P<globalkey>[a-gA-G](b*|\#*))\.)?
                                    ((?P<localkey>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)+)\.)?
                                    ((?P<pedal>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)+)\[)?
                                    (?P<chord>
                                        (?P<numeral>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))
                                        (?P<form>(%|o|\+|M|\+M))?
                                        (?P<figbass>(7|65|43|42|2|64|6))?
                                        (\((?P<changes>((\+|-|\^|v)?(b*|\#*)\d)+)\))?
                                        (/(?P<relativeroot>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)*))?
                                    )
                                    (?P<pedalend>\])?
                                  )?
                                  (\|(?P<cadence>((HC|PAC|IAC|DC|EC|PC)(\..+?)?)))?
                                  (?P<phraseend>(\\\\|\}\{|\{|\})
                                  )?
                                 )
                                 (-
                                  (?P<second>
                                    ((?P<globalkey2>[a-gA-G](b*|\#*))\.)?
                                    ((?P<localkey2>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)+)\.)?
                                    ((?P<pedal2>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)+)\[)?
                                    (?P<chord2>
                                        (?P<numeral2>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))
                                        (?P<form2>(%|o|\+|M|\+M))?
                                        (?P<figbass2>(7|65|43|42|2|64|6))?
                                        (\((?P<changes2>((\+|-|\^|v)?(b*|\#*)\d)+)\))?
                                        (/(?P<relativeroot2>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)*))?
                                    )
                                    (?P<pedalend2>\])?
                                  )?
                                  (\|(?P<cadence2>((HC|PAC|IAC|DC|EC|PC)(\..+?)?)))?
                                  (?P<phraseend2>(\\\\|\}\{|\{|\})
                                  )?
                                 )?
                                $
                                """,
                        re.VERBOSE)
""":obj:`str`
Constant with a regular expression that recognizes complete labels conforming to the DCML harmony annotation standard 
including those consisting of two alternatives, without having to split them. It is simply a doubled version of DCML_REGEX.
"""

FORM_DETECTION_REGEX = r"\d{1,2}(?:i+|\w)?:"
FORM_LEVEL_REGEX = r"(?P<levels>(?:(?:\d{1,2})(?:i+|\w)?[\&=]?)+):(?P<token>(?:\D|\d+(?!(?:$|i+|\w)?[\&=:]))+)"


MS3_HTML = {'#005500': 'ms3_darkgreen',
            '#aa0000': 'ms3_darkred',
            '#aa5500': 'ms3_sienna',
            '#00aa00': 'ms3_green',
            '#aaaa00': 'ms3_darkgoldenrod',
            '#aaff00': 'ms3_chartreuse',
            '#00007f': 'ms3_navy',
            '#aa007f': 'ms3_darkmagenta',
            '#00557f': 'ms3_teal',
            '#aa557f': 'ms3_indianred',
            '#00aa7f': 'ms3_darkcyan',
            '#aaaa7f': 'ms3_darkgray',
            '#aaff7f': 'ms3_palegreen',
            '#aa00ff': 'ms3_darkviolet',
            '#0055ff': 'ms3_dodgerblue',
            '#aa55ff': 'ms3_mediumorchid',
            '#00aaff': 'ms3_deepskyblue',
            '#aaaaff': 'ms3_lightsteelblue',
            '#aaffff': 'ms3_paleturquoise',
            '#550000': 'ms3_maroon',
            '#555500': 'ms3_darkolivegreen',
            '#ff5500': 'ms3_orangered',
            '#55aa00': 'ms3_olive',
            '#ffaa00': 'ms3_orange',
            '#55ff00': 'ms3_lawngreen',
            '#55007f': 'ms3_indigo',
            '#ff007f': 'ms3_deeppink',
            '#55557f': 'ms3_darkslateblue',
            '#ff557f': 'ms3_lightcoral',
            '#55aa7f': 'ms3_mediumseagreen',
            '#ffaa7f': 'ms3_lightsalmon',
            '#55ff7f': 'ms3_lightgreen',
            '#ffff7f': 'ms3_khaki',
            '#5500ff': 'ms3_blue',
            '#5555ff': 'ms3_royalblue',
            '#ff55ff': 'ms3_violet',
            '#55aaff': 'ms3_cornflowerblue',
            '#ffaaff': 'ms3_lightpink',
            '#55ffff': 'ms3_aquamarine'}

MS3_RGB = {(0, 85, 0): 'ms3_darkgreen',
           (170, 0, 0): 'ms3_darkred',
           (170, 85, 0): 'ms3_sienna',
           (0, 170, 0): 'ms3_green',
           (170, 170, 0): 'ms3_darkgoldenrod',
           (170, 255, 0): 'ms3_chartreuse',
           (0, 0, 127): 'ms3_navy',
           (170, 0, 127): 'ms3_darkmagenta',
           (0, 85, 127): 'ms3_teal',
           (170, 85, 127): 'ms3_indianred',
           (0, 170, 127): 'ms3_darkcyan',
           (170, 170, 127): 'ms3_darkgray',
           (170, 255, 127): 'ms3_palegreen',
           (170, 0, 255): 'ms3_darkviolet',
           (0, 85, 255): 'ms3_dodgerblue',
           (170, 85, 255): 'ms3_mediumorchid',
           (0, 170, 255): 'ms3_deepskyblue',
           (170, 170, 255): 'ms3_lightsteelblue',
           (170, 255, 255): 'ms3_paleturquoise',
           (85, 0, 0): 'ms3_maroon',
           (85, 85, 0): 'ms3_darkolivegreen',
           (255, 85, 0): 'ms3_orangered',
           (85, 170, 0): 'ms3_olive',
           (255, 170, 0): 'ms3_orange',
           (85, 255, 0): 'ms3_lawngreen',
           (85, 0, 127): 'ms3_indigo',
           (255, 0, 127): 'ms3_deeppink',
           (85, 85, 127): 'ms3_darkslateblue',
           (255, 85, 127): 'ms3_lightcoral',
           (85, 170, 127): 'ms3_mediumseagreen',
           (255, 170, 127): 'ms3_lightsalmon',
           (85, 255, 127): 'ms3_lightgreen',
           (255, 255, 127): 'ms3_khaki',
           (85, 0, 255): 'ms3_blue',
           (85, 85, 255): 'ms3_royalblue',
           (255, 85, 255): 'ms3_violet',
           (85, 170, 255): 'ms3_cornflowerblue',
           (255, 170, 255): 'ms3_lightpink',
           (85, 255, 255): 'ms3_aquamarine'}

CSS2MS3 = {c[4:]: c for c in MS3_HTML.values()}
CSS_COLORS = list(webcolors.CSS3_NAMES_TO_HEX.keys())
COLORS = sum([[c, CSS2MS3[c]] if c in CSS2MS3 else [c] for c in CSS_COLORS], [])
rgba = namedtuple('RGBA', ['r', 'g', 'b', 'a'])


class map_dict(dict):
    """Such a dictionary can be mapped to a Series to replace its values but leaving the values absent from the dict keys intact."""
    def __missing__(self, key):
        return key

@function_logger
def add_quarterbeats_col(df, offset_dict, insert_after='mc', interval_index=False):
    """ Insert a column measuring the distance of events from MC 1 in quarter notes. If no 'mc_onset' column is present,
        the column corresponds to the ``insert_after`` column's measure counts.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        DataFrame with an ``mc_playthrough`` and an ``mc_onset`` column.
    offset_dict : :obj:`pandas.Series` or :obj:`dict`
        If unfolded: {mc_playthrough -> offset}
        Otherwise: {mc -> offset}
        You can create the dict using the function :py:meth:`Parse.get_continuous_offsets()<ms3.parsed.Parse.get_continuous_offsets>`
    insert_after : :obj:`str`, optional
        Name of the column after which the new column will be inserted.
    interval_index : :obj:`bool`, optional
        Defaults to False. Pass True to replace the index with an :obj:`pandas.

    Returns
    -------

    """
    if offset_dict is None:
        logger.warning(f"No offset_dict was passed: Not adding quarterbeats.")
        return df
    if 'quarterbeats' not in df.columns:
        df = df.copy()
        quarterbeats = df[insert_after].map(offset_dict)
        if 'mc_onset' in df.columns:
            quarterbeats += df.mc_onset * 4
        insert_here = df.columns.get_loc(insert_after) + 1
        df.insert(insert_here, 'quarterbeats', quarterbeats)
        if 'duration_qb' not in df.columns:
            if 'duration' in df.columns:
                dur = (df.duration * 4).astype(float).round(3)
                df.insert(insert_here + 1, 'duration_qb', dur)
            elif 'end' in offset_dict:
                present_qb = df.quarterbeats.notna()
                try:
                    ivs = make_interval_index(df.loc[present_qb, 'quarterbeats'].astype(float).round(3),
                                              end_value=float(offset_dict['end']))
                    df.insert(insert_here + 1, 'duration_qb', pd.NA)
                    df.loc[present_qb, 'duration_qb'] = ivs.length
                except:
                    logger.warning("Error while creating durations from quarterbeats column. Check consistency (quarterbeats need to be monotically ascending; 'end' value in offset_dict needs to be larger than the last quarterbeat).")
            else:
                logger.warning("Column 'duration_qb' could not be created.")
    else:
        logger.debug("quarterbeats column was already present.")
    if interval_index and all(c in df.columns for c in ('quarterbeats', 'duration_qb')):
        df = replace_index_by_intervals(df)
    return df

def assert_all_lines_equal(before, after, original, tmp_file):
    """ Compares two multiline strings to test equality."""
    diff = [(i, bef, aft) for i, (bef, aft) in enumerate(zip(before.splitlines(), after.splitlines()), 1) if bef != aft]
    if len(diff) > 0:
        line_n, left, _ = zip(*diff)
        ln = len(str(max(line_n)))
        left_col = max(len(s) for s in left)
        folder, file = os.path.split(original)
        tmp_persist = os.path.join(folder, '..', file)
        shutil.copy(tmp_file.name, tmp_persist)
        diff = [('', original, tmp_persist)] + diff
    assert len(diff) == 0, '\n' + '\n'.join(
        f"{a:{ln}}  {b:{left_col}}    {c}" for a, b, c in diff)


def assert_dfs_equal(old, new, exclude=[]):
    """ Compares the common columns of two DataFrames to test equality."""
    old_l, new_l = len(old), len(new)
    greater_length = max(old_l, new_l)
    if old_l != new_l:
        print(f"Old length: {old_l}, new length: {new_l}")
        old_is_shorter = new_l == greater_length
        shorter = old if old_is_shorter else new
        missing_rows = abs(old_l - new_l)
        shorter_cols = shorter.columns
        patch = pd.DataFrame([['missing row'] * len(shorter_cols)] * missing_rows, columns=shorter_cols)
        shorter = pd.concat([shorter, patch], ignore_index=True)
        if old_is_shorter:
            old = shorter
        else:
            new = shorter
    old.index.rename('old_ix', inplace=True)
    new.index.rename('new_ix', inplace=True)
    cols = [col for col in set(old.columns).intersection(set(new.columns)) if col not in exclude]
    nan_eq = lambda a, b: (a == b) | pd.isnull(a) & pd.isnull(b)
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
                    f"{n_diffs}/{greater_length} ({n_diffs / greater_length * 100:.2f} %) rows are different for {col}{' (showing first 20)' if n_diffs > 20 else ''}:\n{comparison}\n")
        return '\n'.join(comp_str)

    assert diffs_per_col.sum() == 0, show_diff()



def ambitus2oneliner(ambitus):
    """ Turns a ``metadata['parts'][staff_id]`` dictionary into a string."""
    if 'min_midi' in ambitus:
        return f"{ambitus['min_midi']}-{ambitus['max_midi']} ({ambitus['min_name']}-{ambitus['max_name']})"
    if 'max_midi' in ambitus:
        return f"{ambitus['max_midi']}-{ambitus['max_midi']} ({ambitus['max_name']}-{ambitus['max_name']})"
    return ''



def changes2list(changes, sort=True):
    """ Splits a string of changes into a list of 4-tuples.

    Example
    -------
    >>> changes2list('+#7b5')
    [('+#7', '+', '#', '7'),
     ('b5',  '',  'b', '5')]
    """
    res = [t for t in re.findall(r"((\+|-|\^|v)?(#+|b+)?(1\d|\d))", changes)]
    return sorted(res, key=lambda x: int(x[3]), reverse=True) if sort else res



def changes2tpc(changes, numeral, minor=False, root_alterations=False):
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
    root_alteration, num_degree = split_scale_degree(numeral, count=True, logger=logger)
    # build 2-octave diatonic scale on C major/minor
    root = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'].index(num_degree.upper())
    tpcs = 2 * [i for i in (0, 2, -3, -1, 1, -4, -2)] if minor else 2 * [i for i in (0, 2, 4, -1, 1, 3, 5)]
    tpcs = tpcs[root:] + tpcs[:root]  # starting the scale from chord root
    root = tpcs[0]
    if root_alterations:
        root += 7 * root_alteration
        tpcs[0] = root

    alts = changes2list(changes, sort=False)
    acc2tpc = lambda accidentals: 7 * (accidentals.count('#') - accidentals.count('b'))
    return [(full, added, acc, chord_interval,
             (tpcs[int(chord_interval) - 1] + acc2tpc(acc) - root) if not chord_interval in ['3', '5'] else None) for
            full, added, acc, chord_interval in alts]



def check_labels(df, regex, column='label', split_regex=None, return_cols=['mc', 'mc_onset', 'staff', 'voice']):
    """  Checks the labels in ``column`` against ``regex`` and returns those that don't match.

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
        if split_regex == True:
            check_this = split_alternatives(df, column=column, alternatives_only=True)
        else:
            check_this = split_alternatives(df, column=column, regex=split_regex, max=1000, alternatives_only=True)
    else:
        check_this = df[[column]]
    if regex.__class__ != re.compile('').__class__:
        regex = re.compile(regex, re.VERBOSE)
    not_matched = check_this.apply(lambda c: ~c.str.match(regex).fillna(True))
    cols = [c for c in return_cols if c in df.columns]
    select_wrong = not_matched.any(axis=1)
    res = check_this.where(not_matched, other='.')[select_wrong]
    res = res.apply(lambda c: c.str.replace('^/$', 'empty_harmony', regex=True))
    return pd.concat([df.loc[select_wrong, cols], res], axis=1)





def color2rgba(c):
    """ Pass a RGB or RGBA tuple, HTML color or name to convert it to RGBA """
    if isinstance(c, tuple):
        if len(c) > 3:
            return rgba(*c[:4])
        if len(c) == 3:
            return rgba(*(c + (255,)))
        else:
            return rgba(*c)
    if c[0] == '#':
        return html_color2rgba(c)
    return color_name2rgba(c)



def color_name2format(n, format='rgb'):
    """ Converts a single CSS3 name into one of 'HTML', 'rgb', or 'rgba'"""
    if pd.isnull(n):
        return n
    if n in webcolors.CSS3_NAMES_TO_HEX:
        html = webcolors.name_to_hex(n)
    elif n in MS3_HTML.values():
        html = next(k for k, v in MS3_HTML.items() if v == n)
    elif n[0] == '#':
        html = n
    else:
        return n
    if format == 'html':
        return html
    if format == 'rgb':
        return webcolors.hex_to_rgb(html)
    if format == 'rgba':
        rgb = webcolors.hex_to_rgb(html)
        return rgba(*(rgb + (255,)))
    return html


def color_name2html(n):
    """ Converts a single CSS3 name into HTML"""
    return color_name2format(n, format='html')


def color_name2rgb(n):
    """ Converts a single CSS3 name into RGB"""
    return color_name2format(n, format='rgb')


def color_name2rgba(n):
    """ Converts a single CSS3 name into RGBA"""
    return color_name2format(n, format='rgba')


def color_params2rgba(color_name=None, color_html=None, color_r=None, color_g=None, color_b=None, color_a=None):
    if all(pd.isnull(param) for param in [color_name, color_html, color_r, color_g, color_b, color_a]):
        return None
    res = None
    if not pd.isnull(color_r):
        if pd.isnull(color_a):
            color_a = 255
        if pd.isnull(color_g) or pd.isnull(color_b):
            if pd.isnull(color_name) and pd.isnull(color_html):
                self.logger.warning(f"Not a valid RGB color: {(color_r, color_g, color_b)}")
        else:
            res = (color_r, color_g, color_b, color_a)
    if res is None and not pd.isnull(color_html):
        res = color2rgba(color_html)
    if res is None and not pd.isnull(color_name):
        res = color2rgba(color_name)
    return rgba(*res)


def allnamesequal(name):
    return all(n == name[0] for n in name[1:])

def commonprefix(paths, sep='/'):
    """ Returns common prefix of a list of paths.
    Uses: allnamesequal(), itertools.takewhile()"""
    bydirectorylevels = zip(*[p.split(sep) for p in paths])
    return sep.join(x[0] for x in takewhile(allnamesequal, bydirectorylevels))


def compute_mn(df):
    """ Compute measure numbers from a measure list with columns ['dont_count', 'numbering_offset']
    """
    excluded = df['dont_count'].fillna(0).astype(bool)
    offset = df['numbering_offset']
    mn = (~excluded).cumsum()
    if offset.notna().any():
        offset = offset.fillna(0).astype(int).cumsum()
        mn += offset
    return mn.rename('mn')




@function_logger
def convert(old, new, MS='mscore'):
    process = [MS, "-fo", new, old] #[MS, '--appimage-extract-and-run', "-fo", new, old] if MS.endswith('.AppImage') else [MS, "-fo", new, old]
    if subprocess.run(process):
        logger.info(f"Converted {old} to {new}")
    else:
        logger.warning("Error while converting " + old)

@function_logger
def convert_folder(directory, new_folder, extensions=[], target_extension='mscx', regex='.*', suffix=None, recursive=True,
                   ms='mscore', overwrite=False, parallel=False):
    """ Convert all files in `dir` that have one of the `extensions` to .mscx format using the executable `MS`.

    Parameters
    ----------
    directory, new_folder : str
        Directories
    extensions : list, optional
        If you want to convert only certain formats, give those, e.g. ['mscz', 'xml']
    recursive : bool, optional
        Subdirectories as well.
    MS : str, optional
        Give the path to the MuseScore executable on your system. Need only if
        the command 'mscore' does not execute MuseScore on your system.
    """
    MS = get_musescore(ms)
    assert MS is not None, f"MuseScore not found: {ms}"
    if target_extension[0] == '.':
        target_extension = target_extension[1:]
    conversion_params = []
    #logger.info(f"Traversing {dir} {'' if recursive else 'non-'}recursively...")
    if len(extensions) > 0:
        exclude_re = f"^(?:(?!({'|'.join(extensions)})).)*$"
    else:
        exclude_re = ''
    if new_folder is None:
        new_folder = directory
    new_dirs = {}
    try:
        for subdir, file in scan_directory(directory, file_re=regex, exclude_re=exclude_re, recursive=recursive, subdirs=True, exclude_files_only=True):
            if subdir in new_dirs:
                new_subdir = new_dirs[subdir]
            else:
                old_subdir = os.path.relpath(subdir, directory)
                new_subdir = os.path.join(new_folder, old_subdir) if old_subdir != '.' else new_folder
                os.makedirs(new_subdir, exist_ok=True)
                new_dirs[subdir] = new_subdir
            name, _ = os.path.splitext(file)
            if suffix is not None:
                fname = f"{name}{suffix}.{target_extension}"
            else:
                fname = f"{name}.{target_extension}"
            old = os.path.join(subdir, file)
            new = os.path.join(new_subdir, fname)
            if overwrite or not os.path.isfile(new):
                conversion_params.append((old, new, MS))
            else:
                logger.debug(new, 'exists already. Pass -o to overwrite.')

        if len(conversion_params) == 0:
            logger.info(f"No files to convert.")
    except:
        logger.error(f"Failed to scan directory {directory} because of the following error:")
        raise


    # TODO: pass filenames as 'logger' argument to convert()
    if parallel:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        pool.starmap(convert, conversion_params)
        pool.close()
        pool.join()
    else:
        for o, n, ms in conversion_params:
            convert(o, n, ms)


def decode_harmonies(df, label_col='label', keep_type=True, return_series=False, alt_cols='alt_label', alt_separator='-'):
    """MuseScore stores types 2 (Nashville) and 3 (absolute chords) in several columns. This function returns a copy of
    the DataFrame ``Annotations.df`` where the label column contains the strings corresponding to these columns.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        DataFrame with encoded harmony labels as stored in an :obj:`Annotations` object.
    label_col : :obj:`str`, optional
        Column name where the main components (<name> tag) are stored, defaults to 'label'
    keep_type : :obj:`bool`, optional
        Defaults to True, retaining the 'label_type' column and setting types 2 and 3 to 0.
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
    df = df.copy()
    drop_cols, compose_label = [], []
    if 'nashville' in df.columns:
        sel = df.nashville.notna()
        df.loc[sel, label_col] = df.loc[sel, 'nashville'].astype(str) + df.loc[sel, label_col].replace('/', '')
        drop_cols.append('nashville')
    if 'leftParen' in df.columns:
        df.leftParen.replace('/', '(', inplace=True)
        compose_label.append('leftParen')
        drop_cols.append('leftParen')
    if 'absolute_root' in df.columns:
        df.absolute_root = fifths2name(df.absolute_root, ms=True)
        compose_label.append('absolute_root')
        drop_cols.append('absolute_root')
        if 'rootCase' in df.columns:
            sel = df.rootCase.notna()
            df.loc[sel, 'absolute_root'] = df.loc[sel, 'absolute_root'].str.lower()
            drop_cols.append('rootCase')
    if label_col in df.columns:
        compose_label.append(label_col)
    if 'absolute_base' in df.columns:
        df.absolute_base = '/' + fifths2name(df.absolute_base, ms=True)
        compose_label.append('absolute_base')
        drop_cols.append('absolute_base')
    if 'rightParen' in df.columns:
        df.rightParen.replace('/', ')', inplace=True)
        compose_label.append('rightParen')
        drop_cols.append('rightParen')
    new_label_col = df[compose_label].fillna('').sum(axis=1).astype(str)
    new_label_col = new_label_col.str.replace('^/$', 'empty_harmony', regex=True).replace('', np.nan)

    if alt_cols is not None:
        if isinstance(alt_cols, str):
            alt_cols = [alt_cols]
        present = [c for c in alt_cols if c in df.columns]
        if len(present) > 0:
            alt_joined = pd.Series('', index=new_label_col.index)
            for c in present:
                alt_joined += (alt_separator + df[c]).fillna('')
            new_label_col += alt_joined

    if return_series:
        return new_label_col

    if 'label_type' in df.columns:
        if keep_type:
            df.loc[df.label_type.isin([2, 3, '2', '3']), 'label_type'] == 0
        else:
            drop_cols.append('label_type')
    df[label_col] = new_label_col
    df.drop(columns=drop_cols, inplace=True)
    return df


def df2md(df, name="Overview"):
    """ Turns a DataFrame into a MarkDown table. The returned writer can be converted into a string.
    """
    writer = MarkdownTableWriter()
    writer.table_name = name
    writer.header_list = list(df.columns.values)
    writer.value_matrix = df.values.tolist()
    return writer



def dict2oneliner(d):
    """ Turns a dictionary into a single-line string without brackets."""
    return ', '.join(f"{k}: {v}" for k, v in d.items())

@function_logger
def expand_form_labels(fl, fill_mn_until=None):
    """Expands form labels into a hierarchical view of levels.

    Parameters
    ----------
    fill_mn_until : :obj:`int`, optional
        Pass the last measure number in order to insert rows for every measure without a form label.
        If you pass -1, the measure number of the last form label will be used.
    """

    def distribute_levels(df, mc=None):
        """Takes the regex matches of one label and turns them into one row where
        the levels become column names. Pass label's MC to display it in error messages.
        """
        col2val = {}
        levels_re = r"(\d{1,2})(i+|\w)?[\&=]?"
        for levels, token in df.values:
            for level, hybrid in re.findall(levels_re, levels):
                key = (hybrid, level)  # hybrid becomes first index level, e.g. 'a' and 'b'
                if key in col2val:
                    mc_string = '' if mc is None else f"MC {mc}: "
                    logger.warning(
                        f"{mc_string}The token '{col2val[key]}' for level {key} was overwritten with '{token}':\n{df}")
                col2val[key] = token
        return col2val

    matches = fl.form_label.str.extractall(FORM_LEVEL_REGEX)
    matches.token = matches.token.str.strip('\n ,')
    mcs = fl.mc.to_dict()
    levels = list(range(matches.index.nlevels - 1))  # all index levels except the one for the matches
    res = {i: distribute_levels(df, mcs[i]) for i, df in matches.groupby(level=levels)}
    res = pd.DataFrame.from_dict(res, orient='index')
    res.columns = pd.MultiIndex.from_tuples(res.columns)
    form_types = res.columns.levels[0]
    if len(form_types) > 1:
        # columns will be MultiIndex
        if '' in form_types:
            # there are labels pertaining to all form_types
            forms = [f for f in form_types if f != '']
            pertaining_to_all = res.loc[:, '']
            distributed_to_all = pd.concat([pertaining_to_all] * len(forms), keys=forms, axis=1)
            level_exists = distributed_to_all.columns.isin(res.columns)
            existing_level_names = distributed_to_all.columns[level_exists]
            res = pd.concat([res.loc[:, forms], distributed_to_all.loc[:, ~level_exists]], axis=1)
            potentially_preexistent = distributed_to_all.loc[:, level_exists]
            check_double_attribution = res[existing_level_names].notna() & potentially_preexistent.notna()
            if check_double_attribution.any().any():
                logger.warning(
                    "Could not distribute levels to all form types because some had already been individually specified.")
            res.loc[:, existing_level_names] = res[existing_level_names].fillna(potentially_preexistent)
        fl_multiindex = pd.concat([fl], keys=[''], axis=1)
        res = pd.concat([fl_multiindex, res.sort_index(axis=1)], axis=1)
    else:
        if form_types[0] == '':
            res = pd.concat([fl, res.droplevel(0, axis=1).sort_index(axis=1)], axis=1)
        else:
            raise NotImplementedError(f"Syntax for several form types used for a single one: '{form_types[0]}'")

    if fill_mn_until is not None:
        if len(form_types) == 1:
            mn_col, mn_onset = 'mn', 'mn_onset'
        else:
            mn_col, mn_onset = ('', 'mn'), ('', 'mn_onset')
        first_mn = fl.mn.min()
        last_mn = fill_mn_until if fill_mn_until > -1 else fl.mn.max()
        all_mns = set(range(first_mn, last_mn + 1))
        missing = all_mns.difference(set(res[mn_col]))
        missing_mn = pd.DataFrame({mn_col: list(missing)}).reindex(res.columns, axis=1)
        res = pd.concat([res, missing_mn], ignore_index=True).sort_values([mn_col, mn_onset]).reset_index(drop=True)
    return res



def fifths2acc(fifths):
    """ Returns accidentals for a stack of fifths that can be combined with a
        basic representation of the seven steps."""
    return abs(fifths // 7) * 'b' if fifths < 0 else fifths // 7 * '#'



def fifths2iv(fifths, smallest=False):
    """ Return interval name of a stack of fifths such that
       0 = 'P1', -1 = 'P4', -2 = 'm7', 4 = 'M3' etc. If you pass ``smallest=True``, intervals of a fifth or greater
       will be inverted (e.g. 'm6' => '-M3' and 'D5' => '-A4').
       Uses: map2elements()
    """
    if isinstance(fifths, Iterable):
        return map2elements(fifths, fifths2iv)
    if pd.isnull(fifths):
        return fifths
    interval_qualities = {0: ['P', 'P', 'P', 'M', 'M', 'M', 'M'],
                          -1: ['D', 'D', 'D', 'm', 'm', 'm', 'm']}
    interval_qualities_inverted = {0: ['P', 'P', 'P', 'm', 'm', 'm', 'm'],
                                   -1: ['A', 'A', 'A', 'M', 'M', 'M', 'M']}
    fifths += 1  # making 0 = fourth, 1 = unison, 2 = fifth etc.
    pos = fifths % 7
    int_num = [4, 1, 5, 2, 6, 3, 7][pos]
    qual_region = fifths // 7
    if smallest and int_num > 4:
        int_num = 9 - int_num
        if qual_region in interval_qualities_inverted:
            int_qual = interval_qualities_inverted[qual_region][pos]
        elif qual_region < 0:
            int_qual = (abs(qual_region) - 1) * 'A'
        else:
            int_qual = qual_region * 'D'
        int_qual = '-' + int_qual
    else:
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
    if isinstance(fifths, Iterable):
        return map2elements(fifths, fifths2rn, minor=minor)
    if pd.isnull(fifths):
        return fifths
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
    if isinstance(fifths, Iterable):
        return map2elements(fifths, fifths2sd, minor=minor)
    if pd.isnull(fifths):
        return fifths
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


def get_ms_version(mscx_file):
    with open(mscx_file, encoding='utf-8') as file:
        for i, l in enumerate(file):
            if i < 2:
                pass
            if i == 2:
                m = re.search(r"<programVersion>(.*?)</programVersion>", l)
                if m is None:
                    return None
                else:
                    return m.group(1)


@function_logger
def get_musescore(MS):
    """ Tests whether a MuseScore executable can be found on the system.
    Uses: test_binary()

    Parameters
    ----------
    MS : :obj:`str`
        A path to the executable, installed command, or one of the keywords {'auto', 'win', 'mac'}

    Returns
    -------
    :obj:`str`
        Path to the executable if found or None.

    """
    if MS is None:
        return MS
    if MS == 'auto':
        mapping = {
            'Windows': 'win',
            'Darwin': 'mac',
            'Linux': 'mscore'
        }
        system = platform.system()
        try:
            MS = mapping[system]
        except:
            logger.warning(f"System could not be inferred: {system}")
            MS = 'mscore'
    if MS == 'win':
        program_files = os.environ['PROGRAMFILES']
        MS = os.path.join(program_files, r"MuseScore 3\bin\MuseScore3.exe")
    elif MS == 'mac':
        MS = "/Applications/MuseScore 3.app/Contents/MacOS/mscore"
    return test_binary(MS, logger=logger)


def get_path_component(path, after):
    """Returns only the path's subfolders below ``after``. If ``after`` is the last
    component, '.' is returned."""
    dir1, base1 = os.path.split(path)
    if dir1 in ('', '.', '/', '~'):
        if base1 == after:
            return '.'
        return path
    dir2, base2 = os.path.split(dir1)
    if base2 == after:
        return base1
    higher_levels = get_path_component(dir1, after=after)
    return os.path.join(higher_levels, base1)


def group_id_tuples(l):
    """ Turns a list of (key, ix) into a {key: [ix]}

    """
    d = defaultdict(list)
    for k, i in l:
        if k is not None:
            d[k].append(i)
    return dict(d)


def html2format(df, format='name', html_col='color_html'):
    """ Converts the HTML column of a DataFrame into 'name', 'rgb , or 'rgba'. """
    if format == 'name':
        return df[html_col].map(color_name2html)
    if format == 'rgb':
        return df[html_col].map(color_name2rgb)
    if format == 'rgba':
        return df[html_col].map(color_name2rgba)


def html_color2format(h, format='name'):
    """ Converts a single HTML color into 'name', 'rgb', or  'rgba'."""
    if pd.isnull(h):
        return h
    if format == 'name':
        try:
            return webcolors.hex_to_name(h)
        except:
            try:
                return MS3_HTML[h]
            except:
                return h
    if format == 'rgb':
        return webcolors.hex_to_rgb(h)
    if format == 'rgba':
        rgb = webcolors.hex_to_rgb(h)
        return rgba(*(rgb + (255,)))


def html_color2name(h):
    """ Converts a HTML color into its CSS3 name or itself if there is none."""
    return html_color2format(h, 'name')


def html_color2rgb(h):
    """ Converts a HTML color into RGB."""
    return html_color2format(h, 'rgb')


def html_color2rgba(h):
    """ Converts a HTML color into RGBA."""
    return html_color2format(h, 'rgba')


@function_logger
def is_any_row_equal(df1, df2):
    """ Returns True if any two rows of the two DataFrames contain the same value tuples. """
    assert len(df1.columns) == len(df2.columns), "Pass the same number of columns for both DataFrames"
    v1 = set(df1.itertuples(index=False, name=None))
    v2 = set(df2.itertuples(index=False, name=None))
    return v1.intersection(v2)


def is_minor_mode(fifths, minor=False):
    """ Returns True if the scale degree `fifths` naturally has a minor third in the scale.
    """
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
    """ Returns a generator of ``collectio``. ``selector`` can be a collection of index numbers to select or unselect
    elements -- depending on ``opposite`` """
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
    try:
        return ', '.join(str(s) for s in iterable)
    except:
        return iterable


def iterate_subcorpora(path: str,
                       prefixes: Iterable = None,
                       suffixes: Iterable = None,
                       ignore_case: bool = True):
    """ Recursively walk through subdirectory and files but stop and return path as soon as
    at least one file or at least one folder matches at least one prefix or at least one suffix.

    Parameters
    ----------
    path : :obj:`str`
        Directory to scan.
    prefixes : :obj:`collections.abc.Iterable`, optional
        Current directory is returned if at least one contained item starts with one of the prefixes.
        Defaults to ``['metadata']``
    suffixes : :obj:`collections.abc.Iterable`, optional
        Current directory is returned if at least one contained item ends with one of the suffixes.
        Files are tested against suffixes including file extensions.
        Defaults to ``['notes', 'rests', 'notes_and_rests', 'measures', 'events', 'labels', 'chords', 'expanded',
                    'harmonies', 'cadences', 'form_labels', 'MS3']``
    ignore_case : :obj:`bool`, optional
        Defaults to True, meaning that file and folder names match prefixes and suffixes independent
        of capitalization.

    Yields
    ------
    :obj:`str`
        Full path of the next subcorpus.

    """

    def check_fname(s):
        if ignore_case:
            return any(s.lower().startswith(p) for p in prefixes) or \
                   any(s.lower().endswith(suf) for suf in suffixes)
        return any(s.startswith(p) for p in prefixes) or \
               any(s.endswith(suf) for suf in suffixes)

    if prefixes is None:
        prefixes = []
    if suffixes is None:
        suffixes = STANDARD_NAMES

    if ignore_case:
        prefixes = [p.lower() for p in prefixes]
        suffixes = [s.lower() for s in suffixes]

    for d, subdirs, files in os.walk(path):
        subdirs[:] = sorted(subdirs)
        if any(check_fname(f) for f in files) or \
            any(check_fname(d) for d in subdirs):
            del(subdirs[:])
            yield d


@function_logger
def join_tsvs(dfs, sort_cols=False):
    """ Performs outer join on the passed DataFrames based on 'mc' and 'mc_onset', if any.
    Uses: functools.reduce(), sort_cols(), sort_note_lists()

    Parameters
    ----------
    dfs : :obj:`Collection`
        Collection of DataFrames to join.
    sort_cols : :obj:`bool`, optional
        If you pass True, the remaining columns (those that are not defined in the standard column order in the function
        sort_cols) will be sorted.

    Returns
    -------

    """
    if len(dfs) == 1:
        return dfs[0]
    zero, one, two = [], [], []
    for df in dfs:
        if 'mc' in df.columns:
            if 'mc_onset' in df.columns:
                two.append(df)
            else:
                one.append(df)
        else:
            zero.append(df)
    join_order = two + one
    if len(zero) > 0:
        logger.info(f"{len(zero)} DataFrames contain none of the columns 'mc' and 'mc_onset'.")

    pos_cols = ['mc', 'mc_onset']

    def join_tsv(a, b):
        join_cols = [c for c in pos_cols if c in a.columns and c in b.columns]
        res = pd.merge(a, b, how='outer', on=join_cols, suffixes=('', '_y')).reset_index(drop=True)
        duplicates = [col for col in res.columns if col.endswith('_y')]
        for d in duplicates:
            left = d[:-2]
            if res[left].isna().any():
                res[left].fillna(res[d], inplace=True)
        return res.drop(columns=duplicates)

    res = reduce(join_tsv, join_order)
    if 'midi' in res.columns:
        res = sort_note_list(res)
    elif len(two) > 0:
        res = res.sort_values(pos_cols)
    else:
        res = res.sort_values('mc')
    return column_order(res, sort=sort_cols).reset_index(drop=True)


def str2inttuple(l):
    if l == '':
        return tuple()
    res = []
    for s in l.split(', '):
        try:
            res.append(int(s))
        except:
            print(f"String value '{s}' could not be converted to a tuple.")
    return tuple(res)


def int2bool(s):
    try:
        return bool(int(s))
    except:
        return s


def safe_frac(s):
    try:
        return frac(s)
    except:
        return s


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


    CONVERTERS = {
        'added_tones': str2inttuple,
        'act_dur': safe_frac,
        'chord_tones': str2inttuple,
        'globalkey_is_minor': int2bool,
        'localkey_is_minor': int2bool,
        'mc_offset': safe_frac,
        'mc_onset': safe_frac,
        'mn_onset': safe_frac,
        'next': str2inttuple,
        'nominal_duration': safe_frac,
        'quarterbeats': safe_frac,
        'onset': safe_frac,
        'duration': safe_frac,
        'scalar': safe_frac, }

    DTYPES = {
        'absolute_base': 'Int64',
        'absolute_root': 'Int64',
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
        'color_name': str,
        'color_html': str,
        'color_r': 'Int64',
        'color_g': 'Int64',
        'color_b': 'Int64',
        'color_a': 'Int64',
        'dont_count': 'Int64',
        'figbass': str,
        'form': str,
        'globalkey': str,
        'gracenote': str,
        'harmonies_id': 'Int64',
        'keysig': 'Int64',
        'label': str,
        'label_type': object,
        'leftParen': str,
        'localkey': str,
        'mc': 'Int64',
        'mc_playthrough': 'Int64',
        'midi': 'Int64',
        'mn': str,
        'offset:x': str,
        'offset_x': str,
        'offset:y': str,
        'offset_y': str,
        'nashville': 'Int64',
        'notes_id': 'Int64',
        'numbering_offset': 'Int64',
        'numeral': str,
        'pedal': str,
        'playthrough': 'Int64',
        'phraseend': str,
        'relativeroot': str,
        'repeats': str,
        'rightParen': str,
        'root': 'Int64',
        'rootCase': 'Int64',
        'slur': str,
        'special': str,
        'staff': 'Int64',
        'tied': 'Int64',
        'timesig': str,
        'tpc': 'Int64',
        'voice': 'Int64',
        'voices': 'Int64',
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
    df = pd.read_csv(path, sep=sep, index_col=index_col,
                       dtype=types,
                       converters=conv, **kwargs)
    if 'mn' in df:
        mn_volta = mn2int(df.mn)
        df.mn = mn_volta.mn
        if mn_volta.volta.notna().any():
            if 'volta' not in df.columns:
                df['volta'] = pd.Series(pd.NA, index=df.index).astype('Int64')
            df.volta.fillna(mn_volta.volta, inplace=True)
    return df



def make_continuous_offset(act_durs, quarters=True, negative_anacrusis=None):
    """ In order to compute continuous offset, this function is required to compute each MC's offset from the
    piece's beginning.

    Parameters
    ----------
    act_durs : :obj:`pandas.Series`
        A series of actual measures durations as fractions of whole notes (might differ from time signature).
    quarters : :obj:`bool`, optional
        By default, the continuous offsets are expressed in quarter notes. Pass false to leave them as fractions
        of a whole note.
    negative_anacrusis : :obj:`fractions.Fraction`
        By default, the first value is 0. If you pass a fraction here, the first value will be its negative and the
        second value will be 0.

    Returns
    -------
    :obj:`pandas.Series`
        Cumulative sum of the values, shifted down by 1.

    """
    if quarters:
        act_durs = act_durs * 4
    res = act_durs.cumsum()
    last_val = res.iloc[-1]
    last_ix = res.index[-1] + 1
    res = res.shift(fill_value=0)
    res = res.append(pd.Series([last_val], index=[last_ix]))
    res = res.append(pd.Series([last_val], index=['end']))
    if negative_anacrusis is not None:
        res -= abs(frac(negative_anacrusis))
    return res



def make_id_tuples(key, n):
    """ For a given key, this function returns index tuples in the form [(key, 0), ..., (key, n)]

    Returns
    -------
    list
        indices in the form [(key, 0), ..., (key, n)]

    """
    return list(zip(repeat(key), range(n)))


def make_interval_index(S, end_value=None, closed='left', **kwargs):
    """ Interpret a Series as interval breaks and make an IntervalIndex out of it.

    Parameters
    ----------
    S : :obj:`pandas.Series`
        Interval breaks. It is assumed that the breaks are sorted.
    end_value : numeric
        Often you want to pass the right border of the last interval.
    closed : :obj:`str`
        Defaults to 'left'. This and the kwargs are fed to :py:meth:`pandas.IntervalIndex.from_breaks`.
    kwargs

    Returns
    -------

    """
    breaks = S.to_list()
    if end_value is not None:
        last = breaks[-1]
        if end_value > last:
            breaks += [end_value]
        else:
            breaks += [last]
    try:
        iix = pd.IntervalIndex.from_breaks(breaks, closed=closed, **kwargs)
    except:
        print(breaks)
        raise
    return iix


def make_name_columns(df):
    """Relies on the columns ``localkey`` and ``globalkey`` to transform the columns ``root`` and ``bass_notes`` from
    scale degrees (expressed as fifths) to absolute note names, e.g. in C major: 0 => 'C', 7 => 'C#', -5 => 'Db'
    Uses: transform(), scale_degree2name"""
    new_cols = {}
    for col in ('root', 'bass_note'):
        if col in df.columns:
            new_cols[f"{col}_name"] = transform(df, scale_degree2name, [col, 'localkey', 'globalkey'])
    return pd.DataFrame(new_cols)



def map2elements(e, f, *args, **kwargs):
    """ If `e` is an iterable, `f` is applied to all elements.
    """
    if isinstance(e, Iterable) and not isinstance(e, str):
        return e.__class__(map2elements(x, f, *args, **kwargs) for x in e)
    return f(e, *args, **kwargs)


@function_logger
def merge_ties(df, return_dropped=False, perform_checks=True):
    """ In a note list, merge tied notes to single events with accumulated durations.
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

    def merge(df):
        vc = df.tied.value_counts()
        if vc[1] != 1 or vc[-1] != 1:
            logger.warning(f"More than one 1 or -1:\n{vc}")
        ix = df.iloc[0].name
        dur = df.duration.sum()
        drop = df.iloc[1:].index.to_list()
        return pd.Series({'ix': ix, 'duration': dur, 'dropped': drop})

    def merge_notes(staff_midi):

        staff_midi['chunks'] = (staff_midi.tied == 1).astype(int).cumsum()
        t = staff_midi.groupby('chunks', group_keys=False).apply(merge)
        return t.set_index('ix')

    if not df.tied.notna().any():
        return df
    df = df.copy()
    notna = df.loc[df.tied.notna(), ['duration', 'tied', 'midi', 'staff']]
    if perform_checks:
        before = notna.tied.value_counts()
    new_dur = notna.groupby(['staff', 'midi'], group_keys=False).apply(merge_notes).sort_index()
    try:
        df.loc[new_dur.index, 'duration'] = new_dur.duration
    except:
        print(new_dur)
    if return_dropped:
        df.loc[new_dur.index, 'dropped'] = new_dur.dropped
    df = df.drop(new_dur.dropped.sum())
    if perform_checks:
        after = df.tied.value_counts()
        assert before[1] == after[1], f"Error while merging ties. Before:\n{before}\nAfter:\n{after}"
    return df



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
    if 'ambitus' in d:
        d['ambitus'] = ambitus2oneliner(d['ambitus'])
    if 'parts' in d:
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
    fifths : :obj:`int`, optional
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


def midi2name(midi):
    names = {0: 'C',
             1: 'C#/Db',
             2: 'D',
             3: 'D#/Eb',
             4: 'E',
             5: 'F',
             6: 'F#/Gb',
             7: 'G',
             8: 'G#/Ab',
             9: 'A',
             10: 'A#/Bb',
             11: 'B'}
    return names[midi % 12]


def mn2int(mn_series):
    """ Turn a series of measure numbers parsed as strings into two integer columns 'mn' and 'volta'. """
    try:
        split = mn_series.fillna('').str.extract(r"(?P<mn>\d+)(?P<volta>[a-g])?")
    except:
        mn_series = pd.DataFrame(mn_series, columns=['mn', 'volta'])
        try:
            return mn_series.astype('Int64')
        except:
            return mn_series
    split.mn = pd.to_numeric(split.mn)
    split.volta = pd.to_numeric(split.volta.map({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}))
    return split.astype('Int64')


def name2format(df, format='html', name_col='color_name'):
    """ Converts a column with CSS3 names into 'html', 'rgb', or  'rgba'."""
    if format == 'html':
        return df[name_col].map(color_name2html)
    if format == 'rgb':
        return df[name_col].map(color_name2rgb)
    if format == 'rgba':
        return df[name_col].map(color_name2rgba)


@function_logger
def name2fifths(nn):
    """ Turn a note name such as `Ab` into a tonal pitch class, such that -1=F, 0=C, 1=G etc.
        Uses: split_note_name()
    """
    if nn.__class__ == int or pd.isnull(nn):
        return nn
    name_tpcs = {'C': 0, 'D': 2, 'E': 4, 'F': -1, 'G': 1, 'A': 3, 'B': 5}
    accidentals, note_name = split_note_name(nn, count=True, logger=logger)
    if note_name is None:
        return None
    step_tpc = name_tpcs[note_name.upper()]
    return step_tpc + 7 * accidentals


@function_logger
def name2pc(nn):
    """ Turn a note name such as `Ab` into a tonal pitch class, such that -1=F, 0=C, 1=G etc.
        Uses: split_note_name()
    """
    if nn.__class__ == int or pd.isnull(nn):
        logger.warning(f"'{nn}' is not a valid note name.")
        return nn
    name_tpcs = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    accidentals, note_name = split_note_name(nn, count=True, logger=logger)
    if note_name is None:
        return None
    step_pc = name_tpcs[note_name.upper()]
    return (step_pc + accidentals) % 12



def next2sequence(nxt):
    """ Turns a 'next' column into the correct sequence of MCs corresponding to unfolded repetitions.
    Requires that the Series' index be the MCs as in ``measures.set_index('mc').next``.
    """
    mc = nxt.index[0]
    result = []
    nxt = nxt.to_dict()
    while mc != -1:
        result.append(mc)
        new_mc, *rest = nxt[mc]
        if len(rest) > 0:
            nxt[mc] = rest
        mc = new_mc
    return result


@function_logger
def no_collections_no_booleans(df, coll_columns=None, bool_columns=None):
    """
    Cleans the DataFrame columns ['next', 'chord_tones', 'added_tones'] from tuples and the columns
    ['globalkey_is_minor', 'localkey_is_minor'] from booleans, converting them all to integers

    """
    if df is None:
        return df
    collection_cols = ['next', 'chord_tones', 'added_tones']
    bool_cols = ['globalkey_is_minor', 'localkey_is_minor']
    if coll_columns is not None:
        collection_cols += list(coll_columns)
    if bool_columns is not None:
        bool_cols += list(bool_columns)
    try:
        cc = [c for c in collection_cols if c in df.columns]
    except:
        logger.error(f"df needs to be a DataFrame, not a {df.__class__}.")
        return df
    if len(cc) > 0:
        df = df.copy()
        df.loc[:, cc] = transform(df[cc], iterable2str, column_wise=True)
        logger.debug(f"Transformed iterables in the columns {cc} to strings.")
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
        for id in part_dict['staves']:
            name = f"staff_{id}"
            res[f"{name}_instrument"] = part_dict['instrument']
            amb_name = name + '_ambitus'
            res[amb_name] = ambitus2oneliner(part_dict[amb_name])
    return res


@function_logger
def path2type(path):
    """ Determine a file's type by scanning its path for default components in the constant STANDARD_NAMES.

    Parameters
    ----------
    path

    Returns
    -------

    """
    comp2type = {comp: comp for comp in STANDARD_NAMES}
    comp2type['MS3'] = 'scores'
    comp2type['harmonies'] = 'expanded'
    found_components = [comp for comp in comp2type.keys() if comp in path]
    n_found = len(found_components)
    if n_found == 0:
        score_extensions = ('.mscx', '.mscz', '.cap', '.capx', '.midi', '.mid', '.musicxml', '.mxl', '.xml')
        _, fext = os.path.splitext(path)
        if fext.lower() in score_extensions:
            logger.debug(f"Recognized file extension '{fext}' as score.")
            return 'scores'
        logger.debug(f"Type could not be inferred from path '{path}'.")
        return 'unknown'
    if n_found == 1:
        typ = comp2type[found_components[0]]
        logger.debug(f"Path '{path}' recognized as {typ}.")
        return typ
    else:
        shortened_path = path
        while shortened_path > 0:
            shortened_path, base = os.path.split(shortened_path)
            for comp in comp2type.keys():
                if comp in base:
                    typ = comp2type[comp]
                    logger.debug(f"Multiple components ({', '.join(found_components)}) found in path '{path}'. Chose the last one: {typ}")
                    return typ
        logger.warning(f"Components {', '.join(found_components)} found in path '{path}', but not in one of its constituents.")
        return 'unknown'


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



def resolve_dir(d):
    """ Resolves '~' to HOME directory and turns ``d`` into an absolute path.
    """
    if d is None:
        return None
    if '~' in d:
        return os.path.expanduser(d)
    return os.path.abspath(d)


def rgb2format(df, format='html', r_col='color_r', g_col='color_g', b_col='color_b'):
    """ Converts three RGB columns into a color_html or color_name column. """
    cols = [r_col, g_col, b_col]
    if format == 'html':
        html = list(map(rgb_tuple2html, df[cols].itertuples(index=False, name=None)))
        return pd.Series(html, index=df.index).rename('color_html')
    if format == 'name':
        names = list(map(rgb_tuple2name, df[cols].itertuples(index=False, name=None)))
        return pd.Series(names, index=df.index).rename('color_name')


def rgb_tuple2format(t, format='html'):
    """ Converts a single RGB tuple into 'HTML' or 'name'."""
    if pd.isnull(t):
        return t
    if pd.isnull(t[0]):
        return t[0]
    norm = webcolors.normalize_integer_triplet(tuple(int(i) for i in t))
    if format == 'html':
        return webcolors.rgb_to_hex(norm)
    if format == 'name':
        try:
            return webcolors.rgb_to_name(norm)
        except:
            try:
                return MS3_RGB[norm]
            except:
                return webcolors.rgb_to_hex(norm)


def rgb_tuple2html(t):
    """ Converts a single RGB tuple into HTML."""
    return rgb_tuple2format(t, format='html')


def rgb_tuple2name(t):
    """ Converts a single RGB tuple into its CSS3 name or to HTML if there is none."""
    return rgb_tuple2format(t, format='name')


def rgba2attrs(named_tuple):
    return {k: str(v) for k, v in named_tuple._asdict().items()}

def rgba2params(named_tuple):
    attrs = rgba2attrs(named_tuple)
    return {'color_'+k: v for k, v in attrs.items()}


def roman_numeral2fifths(rn, global_minor=False):
    """ Turn a Roman numeral into a TPC interval (e.g. for transposition purposes).
        Uses: split_scale_degree()
    """
    if pd.isnull(rn):
        return rn
    rn_tpcs_maj = {'I': 0, 'II': 2, 'III': 4, 'IV': -1, 'V': 1, 'VI': 3, 'VII': 5}
    rn_tpcs_min = {'I': 0, 'II': 2, 'III': -3, 'IV': -1, 'V': 1, 'VI': -4, 'VII': -2}
    accidentals, rn_step = split_scale_degree(rn, count=True)
    if any(v is None for v in (accidentals, rn_step)):
        return None
    rn_step = rn_step.upper()
    step_tpc = rn_tpcs_min[rn_step] if global_minor else rn_tpcs_maj[rn_step]
    return step_tpc + 7 * accidentals


def roman_numeral2semitones(rn, global_minor=False):
    """ Turn a Roman numeral into a semitone distance from the root (0-11).
        Uses: split_scale_degree()
    """
    if pd.isnull(rn):
        return rn
    rn_tpcs_maj = {'I': 0, 'II': 2, 'III': 4, 'IV': 5, 'V': 7, 'VI': 9, 'VII': 11}
    rn_tpcs_min = {'I': 0, 'II': 2, 'III': 3, 'IV': 5, 'V': 7, 'VI': 8, 'VII': 10}
    accidentals, rn_step = split_scale_degree(rn, count=True)
    if any(v is None for v in (accidentals, rn_step)):
        return None
    rn_step = rn_step.upper()
    step_tpc = rn_tpcs_min[rn_step] if global_minor else rn_tpcs_maj[rn_step]
    return step_tpc + accidentals


def scale_degree2name(sd, localkey, globalkey):
    """ For example, scale degree -1 (fifths, i.e. the subdominant) of the localkey of 'VI' within 'E' minor is 'F'.

    Parameters
    ----------
    sd : :obj:`int`
        Scale degree expressed as distance from the tonic in fifths.
    localkey : :obj:`str`
        Local key in which the scale degree is situated, as Roman numeral (can include slash notation such as V/ii).
    globalkey : :obj:`str`
        Global key as a note name. E.g. `Ab` for Ab major, or 'c#' for C# minor.

    Returns
    -------
    :obj:`str`
        The given scale degree, expressed as a note name.

    """
    if any(pd.isnull(val) for val in (sd, localkey, globalkey)):
        return pd.NA
    global_minor = globalkey.islower()
    if '/' in localkey:
        localkey = resolve_relative_keys(localkey, global_minor)
    lk_fifths = roman_numeral2fifths(localkey, global_minor)
    gk_fifths = name2fifths(globalkey)
    sd_transposed = sd + lk_fifths + gk_fifths
    return fifths2name(sd_transposed)


@function_logger
def scan_directory(directory, file_re=r".*", folder_re=r".*", exclude_re=r"^(\.|_)", recursive=True, subdirs=False, progress=False, exclude_files_only=False):
    """ Generator of file names in ``directory``.

    Parameters
    ----------
    dir : :obj:`str`
        Directory to be scanned for files.
    file_re, folder_re : :obj:`str` or :obj:`re.Pattern`, optional
        Regular expressions for filtering certain file names or folder names.
        The regEx are checked with search(), not match(), allowing for fuzzy search.
    recursive : :obj:`bool`, optional
        By default, sub-directories are recursively scanned. Pass False to scan only ``dir``.
    subdirs : :obj:`bool`, optional
        By default, full file paths are returned. Pass True to return (path, name) tuples instead.
    progress : :obj:`bool`, optional
        By default, the scanning process is shown. Pass False to prevent.
    exclude_files_only : :obj:`bool`, optional
        By default, ``exclude_re`` excludes files and folder. Pass True to exclude only files matching the regEx.


    Yields
    ------
    list
        List of full paths meeting the criteria.

    """
    def traverse(d):
        nonlocal counter

        def check_regex(reg, s, excl=exclude_re):
            try:
                res = re.search(reg, s) is not None and re.search(excl, s) is None
            except:
                print(reg)
                raise
            return res

        for dir_entry in os.scandir(d):
            name = dir_entry.name
            path = os.path.join(d, name)
            if dir_entry.is_dir() and (recursive or folder_re != '.*'):
                for res in traverse(path):
                    yield res
            else:
                if pbar is not None:
                    pbar.update()
                if folder_re == '.*':
                    folder_passes = True
                else:
                    folder_path = os.path.dirname(path)
                    if recursive:
                        folder_passes = check_regex(folder_re, folder_path, excl='^$')  # passes if the folder path matches the regex
                    else:
                        folder = os.path.basename(folder_path)
                        folder_passes = check_regex(folder_re, folder, excl='^$')  # passes if the folder name itself matches the regex
                    if folder_passes and not exclude_files_only: # True if the exclude_re should also exclude folder names
                        folder_passes = check_regex(folder_re, folder_path) # is false if any part of the folder path matches exclude_re
                if dir_entry.is_file() and folder_passes and check_regex(file_re, name):
                    counter += 1
                    if pbar is not None:
                        pbar.set_postfix({'selected': counter})
                    if subdirs:
                        yield (d, name)
                    else:
                        yield path

    if exclude_re is None or exclude_re == '':
        exclude_re = '^$'
    directory = resolve_dir(directory)
    counter = 0
    if not os.path.isdir(directory):
        logger.warning("Not an existing directory: " + directory)
    pbar = tqdm(desc='Scanning files', unit=' files') if progress else None
    return traverse(directory)


def column_order(df, first_cols=None, sort=True):
    """Sort DataFrame columns so that they start with the order of ``first_cols``, followed by those not included. """
    if first_cols is None:
        first_cols = [
            'mc', 'mc_playthrough', 'mn', 'mn_playthrough', 'quarterbeats', 'mc_onset', 'mn_onset', 'beat', 'event', 'timesig', 'staff', 'voice', 'duration', 'tied',
            'gracenote', 'nominal_duration', 'scalar', 'tpc', 'midi', 'volta', 'chord_id']
    cols = df.columns
    remaining = [col for col in cols if col not in first_cols]
    if sort:
        remaining = sorted(remaining)
    column_order = [col for col in first_cols if col in cols] + remaining
    return df[column_order]


def sort_note_list(df, mc_col='mc', mc_onset_col='mc_onset', midi_col='midi', duration_col='duration'):
    """ Sort every measure (MC) by ['mc_onset', 'midi', 'duration'] while leaving gracenotes' order (duration=0) intact.

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
    is_grace = df[duration_col] == 0
    grace_ix = {k: v.to_numpy() for k, v in df[is_grace].groupby([mc_col, mc_onset_col]).groups.items()}
    has_nan = df[midi_col].isna().any()
    if has_nan:
        df.loc[:, midi_col] = df[midi_col].fillna(1000)
    normal_ix = df.loc[~is_grace, [mc_col, mc_onset_col, midi_col, duration_col]].groupby([mc_col, mc_onset_col]).apply(
        lambda gr: gr.index[np.lexsort((gr.values[:, 3], gr.values[:, 2]))].to_numpy())
    sorted_ixs = [np.concatenate((grace_ix[onset], ix)) if onset in grace_ix else ix for onset, ix in
                  normal_ix.iteritems()]
    df = df.reindex(np.concatenate(sorted_ixs)).reset_index(drop=True)
    if has_nan:
        df.loc[:, midi_col] = df[midi_col].replace({1000: np.nan}).astype('Int64')
    return df


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
def split_alternatives(df, column='label', regex=r"-(?!(\d|b+\d|\#+\d))", max=2, inplace=False, alternatives_only=False):
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
        The regular expression (or simple string) that detects the character combination used to separate alternative annotations.
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
    if not inplace:
        df = df.copy()
    alternatives = df[column].str.split(regex, expand=True)
    alternatives.dropna(axis=1, how='all', inplace=True)
    alternatives.columns = range(alternatives.shape[1])
    if alternatives_only:
        columns = [column] + [f"alt_{column}" if i == 1 else f"alt{i}_{column}" for i in alternatives.columns[1:]]
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
            df.insert(position, alt_name, alternatives[i].fillna(np.nan))  # replace None by NaN
            position += 1
        if len(alternatives.columns) > max:
            logger.warning(
                f"More than {max} alternatives are not taken into account:\n{alternatives[alternatives[2].notna()]}")
    else:
        logger.debug("Contains no alternative labels.")
    if not inplace:
        return df



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


@function_logger
def split_scale_degree(sd, count=False):
    """ Splits a scale degree such as 'bbVI' or 'b6' into accidentals and numeral.

    sd : :obj:`str`
        Scale degree.
    count : :obj:`bool`, optional
        Pass True to get the accidentals as integer rather than as string.
    """
    m = re.match(r"^(#*|b*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|\d)$", str(sd))
    if m is None:
        logger.error(f"{sd} is not a valid scale degree.")
        return None, None
    acc, num = m.group(1), m.group(2)
    if count:
        acc = acc.count('#') - acc.count('b')
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


@function_logger
def test_binary(command):
    if command is None:
        return command
    if os.path.isfile(command):
        logger.debug(f"Found MuseScore binary: {command}")
        return command
    if which(command) is None:
        logger.warning(f"MuseScore binary not found and not an installed command: {command}")
        return None
    else:
        logger.debug(f"Found MuseScore command: {command}")
        return command


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


def unfold_repeats(df, mc_sequence):
    """ Use a succesion of MCs to bring a DataFrame in this succession. MCs may repeat.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        DataFrame needs to have the columns 'mc' and 'mn'.
    mc_sequence : :obj:`pandas.Series`
        A Series of the format ``{mc_playthrough: mc}`` where ``mc_playthrough`` corresponds
        to continuous MC
    """
    ############## < v0.5: playthrough <=> mn; >= v0.5: playthrough <=> mc
    vc = df.mc.value_counts()
    res = df.set_index('mc')
    seq = mc_sequence[mc_sequence.isin(res.index)]
    playthrough_col = sum([[playthrough] * vc[mc] for playthrough, mc in seq.items()], [])
    res = res.loc[seq.values].reset_index()
    res.insert(res.columns.get_loc('mc') + 1, 'mc_playthrough', playthrough_col)
    return res


@contextmanager
def unpack_mscz(mscz, tmp_dir=None):
    if tmp_dir is None:
        tmp_dir = os.path.dirname(mscz)
    tmp_file = Temp(suffix='.mscx', prefix='.', dir=tmp_dir, delete=False)
    with Zip(mscz) as zip_file:
        mscx_files = [f for f in zip_file.namelist() if f.endswith('.mscx')]
        if len(mscx_files) > 1:
            logger.info(f"{mscz} contains several MSCX files. Picking the first one")
        mscx = mscx_files[0]
        with zip_file.open(mscx) as mscx_file:
            with tmp_file as tmp:
                for line in mscx_file:
                    tmp.write(line)
    try:
        yield tmp_file.name
    except:
        logger.error(f"Error while dealing with the temporarily unpacked {os.path.basename(mscz)}")
        raise
    finally:
        os.remove(tmp_file.name)


@function_logger
def update_labels_cfg(labels_cfg):
    keys = ['staff', 'voice', 'label_type', 'positioning', 'decode', 'column_name', 'color_format']
    if 'logger' in labels_cfg:
        del(labels_cfg['logger'])
    updated = update_cfg(cfg_dict=labels_cfg, admitted_keys=keys, logger=logger)
    if 'logger' in updated:
        del(updated['logger'])
    return updated


@function_logger
def write_metadata(df, path, markdown=True, index=False):
    if os.path.isdir(path):
        path = os.path.join(path, 'metadata.tsv')
    if not os.path.isfile(path):
        write_this = df
        msg = 'Created'
    else:
        try:
            # Trying to load an existing 'metadata.tsv' file to update overlapping indices, assuming two index levels
            previous = pd.read_csv(path, sep='\t', dtype=str, index_col=[0, 1])
            ix_union = previous.index.union(df.index)
            col_union = previous.columns.union(df.columns)
            previous = previous.reindex(index=ix_union, columns=col_union)
            previous.loc[df.index, df.columns] = df
            write_this = previous
            msg = 'Updated'
        except:
            write_this = df
            msg = 'Replaced '
    first_cols = ['last_mc', 'last_mn', 'KeySig', 'TimeSig', 'label_count', 'harmony_version',
                  'annotated_key', 'annotators', 'reviewers', 'composer', 'workTitle', 'movementNumber',
                  'movementTitle',
                  'workNumber', 'poet', 'lyricist', 'arranger', 'copyright', 'creationDate',
                  'mscVersion', 'platform', 'source', 'translator', 'musescore', 'ambitus']
    write_this.sort_index(inplace=True)
    column_order(write_this, first_cols).to_csv(path, sep='\t', index=index)
    logger.info(f"{msg} {path}")
    if markdown:
        rename4markdown = {
            'fnames': 'file_name',
            'last_mn': 'measures',
            'label_count': 'labels',
            'harmony_version': 'standard',
            'annotators': 'annotators',
            'reviewers': 'reviewers',
        }
        drop_index = 'fnames' in write_this.columns
        md = write_this.reset_index(drop=drop_index).fillna('')
        for c in rename4markdown.keys():
            if c not in md.columns:
                md[c] = ''
        md = md.rename(columns=rename4markdown)[list(rename4markdown.values())]
        md_table = str(df2md(md))

        p = os.path.dirname(path)
        readme = os.path.join(p, 'README.md')
        if os.path.isfile(readme):
            msg = 'Updated'
            with open(readme, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        else:
            msg = 'Created'
            lines = []
        # in case the README exists, everything from the line including '# Overview' (or last line otherwise) is overwritten
        with open(readme, 'w', encoding='utf-8') as f:
            for line in lines:
                if '# Overview' in line:
                    break
                f.write(line)
            else:
                f.write('\n\n')
            f.write(md_table)
        logger.info(f"{msg} {readme}")


def abs2rel_key(absolute, localkey, global_minor=False):
    """
    Expresses a Roman numeral as scale degree relative to a given localkey.
    The result changes depending on whether Roman numeral and localkey are
    interpreted within a global major or minor key.

    Uses: :py:func:`split_scale_degree`

    Parameters
    ----------
    absolute : :obj:`str`
        Relative key expressed as Roman scale degree of the local key.
    localkey : :obj:`str`
        The local key in terms of which `absolute` will be expressed.
    global_minor : bool, optional
        Has to be set to True if `absolute` and `localkey` are scale degrees of a global minor key.

    Examples
    --------
    In a minor context, the key of II would appear within the key of vii as #III.

        >>> abs2rel_key('iv', 'VI', global_minor=False)
        'bvi'       # F minor expressed with respect to A major
        >>> abs2rel_key('iv', 'vi', global_minor=False)
        'vi'        # F minor expressed with respect to A minor
        >>> abs2rel_key('iv', 'VI', global_minor=True)
        'vi'        # F minor expressed with respect to Ab major
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
    if pd.isnull(absolute):
        return np.nan
    maj_rn = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
    min_rn = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii']
    shifts = np.array([[0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 1],
                       [0, 1, 1, 0, 0, 1, 1],
                       [0, 0, 0, -1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 1, 0, 0, 1, 1],
                       [0, 1, 1, 0, 1, 1, 1]])
    abs_acc, absolute = split_scale_degree(absolute, count=True, logger=logger)
    localkey_acc, localkey = split_scale_degree(localkey, count=True, logger=logger)
    shift = abs_acc - localkey_acc
    steps = maj_rn if absolute.isupper() else min_rn
    key_num = maj_rn.index(localkey.upper())
    abs_num = (steps.index(absolute) - key_num) % 7
    step = steps[abs_num]
    if localkey.islower() and abs_num in [2, 5, 6]:
        shift += 1
    if global_minor:
        key_num = (key_num - 2) % 7
    shift -= shifts[key_num][abs_num]
    acc = shift * '#' if shift > 0 else -shift * 'b'
    return acc + step


@function_logger
def rel2abs_key(rel, localkey, global_minor=False):
    """
    Expresses a Roman numeral that is expressed relative to a localkey
    as scale degree of the global key. For local keys {III, iii, VI, vi, VII, vii}
    the result changes depending on whether the global key is major or minor.

    Uses: :py:func:`split_scale_degree`

    Parameters
    ----------
    rel : :obj:`str`
        Relative key or chord expressed as Roman scale degree of the local key.
    localkey : :obj:`str`
        The local key to which `rel` is relative.
    global_minor : bool, optional
        Has to be set to True if `localkey` is a scale degree of a global minor key.

    Examples
    --------
    If the label viio6/VI appears in the context of the local key VI or vi,
    viio6 the absolute key to which viio6 applies depends on the global key.
    The comments express the examples in relation to global C major or C minor.

        >>> rel2abs_key('vi', 'VI', global_minor=False)
        '#iv'       # vi of A major = F# minor
        >>> rel2abs_key('vi', 'vi', global_minor=False)
        'iv'        # vi of A minor = F minor
        >>> rel2abs_key('vi', 'VI', global_minor=True)
        'iv'        # vi of Ab major = F minor
        >>> rel2abs_key('vi', 'vi', global_minor=True)
        'biv'       # vi of Ab minor = Fb minor

    The same examples hold if you're expressing in terms of the global key
    the root of a VI-chord within the local keys VI or vi.
    """
    if pd.isnull(rel) or pd.isnull(localkey):
        return rel
    maj_rn = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
    min_rn = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii']
    shifts = np.array([[0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 1],
                       [0, 1, 1, 0, 0, 1, 1],
                       [0, 0, 0, -1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 1, 0, 0, 1, 1],
                       [0, 1, 1, 0, 1, 1, 1]])
    rel_acc, rel = split_scale_degree(rel, count=True, logger=logger)
    localkey_acc, localkey = split_scale_degree(localkey, count=True, logger=logger)
    shift = rel_acc + localkey_acc
    steps = maj_rn if rel.isupper() else min_rn
    rel_num = steps.index(rel)
    key_num = maj_rn.index(localkey.upper())
    step = steps[(rel_num + key_num) % 7]
    if localkey.islower() and rel_num in [2, 5, 6]:
        shift -= 1
    if global_minor:
        key_num = (key_num - 2) % 7
    shift += shifts[rel_num][key_num]
    acc = shift * '#' if shift > 0 else -shift * 'b'
    return acc + step


@function_logger
def replace_index_by_intervals(df, position_col='quarterbeats', duration_col='duration_qb', closed='left',
                               filter_zero_duration=False, round=3):
    """Given an annotations table with positions and durations, replaces its index with an :obj:`pandas.IntervalIndex <pandas.IntervalIndex>`.
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

    Returns
    -------
    :obj:`pandas.DataFrame`

    """
    if not all(c in df.columns for c in (position_col, duration_col)):
        missing = [c for c in (position_col, duration_col) if c not in df.columns]
        plural = 's' if len(missing) > 1 else ''
        logger.warning(f"Column{plural} not present in DataFrame: {', '.join(missing)}")
        return df
    mask = df[position_col].notna()
    if filter_zero_duration:
        mask &= (df[duration_col] > 0)
    df = df[mask]
    left = df[position_col].astype(float)
    right = left + df[duration_col].astype(float)
    index_tuples = zip(left.round(round), right.round(round))
    df.index = pd.IntervalIndex.from_arrays(left=left.round(round), right=right.round(round), closed=closed)
    return df


def resolve_relative_keys(relativeroot, minor=False):
    """ Resolve nested relative keys, e.g. 'V/V/V' => 'VI'.

    Uses: :py:func:`rel2abs_key`, :py:func:`str_is_minor`

    relativeroot : :obj:`str`
        One or several relative keys, e.g. iv/v/VI (fourth scale degree of the fifth scale degree of the sixth scale degree)
    minor : :obj:`bool`, optional
        Pass True if the last of the relative keys is to be interpreted within a minor context.
    """
    if pd.isnull(relativeroot):
        return relativeroot
    spl = relativeroot.split('/')
    if len(spl) < 2:
        return relativeroot
    if len(spl) == 2:
        applied, to = spl
        return rel2abs_key(applied, to, minor)
    previous, last = '/'.join(spl[:-1]), spl[-1]
    return rel2abs_key(resolve_relative_keys(previous, str_is_minor(last, is_name=False)), last, minor)


def series_is_minor(S, is_name=True):
    """ Returns boolean Series where every value in ``S`` representing a minor key/chord is True."""
    # regex = r'([A-Ga-g])[#|b]*' if is_name else '[#|b]*(\w+)'
    # return S.str.replace(regex, lambda m: m.group(1)).str.islower()
    return S.str.islower()  # as soon as one character is not lowercase, it should be major


def str_is_minor(tone, is_name=True):
    """ Returns True if ``tone`` represents a minor key or chord."""
    # regex = r'([A-Ga-g])[#|b]*' if is_name else '[#|b]*(\w+)'
    # m = re.match(regex, tone)
    # if m is None:
    #     return m
    # return m.group(1).islower()
    return tone.islower()


@function_logger
def transpose_changes(changes, old_num, new_num, old_minor=False, new_minor=False):
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
    if pd.isnull(changes):
        return changes
    old = changes2tpc(changes, old_num, minor=old_minor, root_alterations=True)
    new = changes2tpc(changes, new_num, minor=new_minor, root_alterations=True)
    res = []
    get_acc = lambda n: n * '#' if n > 0 else -n * 'b'
    for (full, added, acc, chord_interval, iv1), (_, _, _, _, iv2) in zip(old, new):
        if iv1 is None or iv1 == iv2:
            res.append(full)
        else:
            d = iv2 - iv1
            if d % 7 > 0:
                logger.warning(
                    f"The difference between the intervals of {full} in {old_num} and {new_num} (in {'minor' if minor else 'major'}) don't differ by chromatic semitones.")
            n_acc = acc.count('#') - acc.count('b')
            new_acc = get_acc(n_acc - d // 7)
            res.append(added + new_acc + chord_interval)
    return ''.join(res)


@function_logger
def features2tpcs(numeral, form=None, figbass=None, changes=None, relativeroot=None, key='C', minor=None,
                  merge_tones=True, bass_only=False, mc=None):
    """
    Given the features of a chord label, this function returns the chord tones
    in the order of the inversion, starting from the bass note. The tones are
    expressed as tonal pitch classes, where -1=F, 0=C, 1=G etc.

    Uses: :py:func:`~.utils.changes2list`, :py:func:`~.utils.name2fifths`, :py:func:`~.utils.resolve_relative_keys`, :py:func:`~.utils.roman_numeral2fifths`,
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
    if pd.isnull(numeral) or numeral == '@none':
        if bass_only or merge_tones:
            return np.nan
        else:
            return {
                'chord_tones': np.nan,
                'added_tones': np.nan,
                'root': np.nan,
            }
    form, figbass, changes, relativeroot = tuple(
        '' if pd.isnull(val) else val for val in (form, figbass, changes, relativeroot))
    label = f"{numeral}{form}{figbass}{'(' + changes + ')' if changes != '' else ''}{'/' + relativeroot if relativeroot != '' else ''}"
    MC = '' if mc is None else f'MC {mc}: '
    if minor is None:
        try:
            minor = str_is_minor(key, is_name=True)
            logger.debug(f"Mode inferred from {key}.")
        except:
            raise ValueError(f"If parameter 'minor' is not specified, 'key' needs to be a string, not {key}")

    key = name2fifths(key)

    if form in ['%', 'M', '+M']:
        assert figbass in ['7', '65', '43',
                           '2'], f"{MC}{label}: {form} requires figbass (7, 65, 43, or 2) since it specifies a chord's seventh."

    if relativeroot != '':
        resolved = resolve_relative_keys(relativeroot, minor)
        rel_minor = str_is_minor(resolved, is_name=False)
        transp = roman_numeral2fifths(resolved, minor)
        logger.debug(
            f"{MC}Chord applied to {relativeroot}. Therefore transposing it by {transp} fifths.")
        return features2tpcs(numeral=numeral, form=form, figbass=figbass, relativeroot=None, changes=changes,
                             key=key + transp, minor=rel_minor, merge_tones=merge_tones, bass_only=bass_only, mc=mc,
                             logger=logger)

    if numeral.lower() == '#vii' and not minor:
        # logger.warning(
        #     f"{MC}{label} in a major context is most probably an annotation error.")
        logger.warning(f"{MC}{numeral} in major context corrected to {numeral[1:]}.")
        numeral = numeral[1:]

    root_alteration, num_degree = split_scale_degree(numeral, count=True, logger=logger)

    # build 2-octave diatonic scale on C major/minor
    root = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'].index(num_degree.upper())
    tpcs = 2 * [i + key for i in (0, 2, -3, -1, 1, -4, -2)] if minor else 2 * [i + key for i in (0, 2, 4, -1, 1, 3, 5)]
    # starting the scale from chord root, i.e. root will be tpcs[0], the chord's seventh tpcs[6] etc.
    tpcs = tpcs[root:] + tpcs[:root]
    root = tpcs[0] + 7 * root_alteration
    tpcs[0] = root  # octave stays diatonic, is not altered
    logger.debug(f"{num_degree}: The {'minor' if minor else 'major'} scale starting from the root: {tpcs}")

    def set_iv(chord_interval, interval_size):
        """ Add to the interval of a given chord interval in `tpcs` (both viewed from the root note).

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

    is_triad = figbass in ['', '6', '64']
    is_seventh_chord = figbass in ['7', '65', '43', '2']
    if not is_triad and not is_seventh_chord:
        raise ValueError(f"{MC}{figbass} is not a valid chord inversion.")

    if form == 'o':
        set_iv(2, -3)
        set_iv(4, -6)
        if is_seventh_chord:
            set_iv(6, -9)
    elif form == '%':
        set_iv(2, -3)
        set_iv(4, -6)
        set_iv(6, -2)
    elif form == '+':
        set_iv(2, 4)
        set_iv(4, 8)
        if is_seventh_chord:
            set_iv(6, -2)
    elif form == '+M':
        set_iv(2, 4)
        set_iv(4, 8)
        set_iv(6, 5)
    else:  # triad with or without major or minor seven
        set_iv(4, 1)
        if num_degree.isupper():
            set_iv(2, 4)
        else:
            set_iv(2, -3)
        if form == 'M':
            set_iv(6, 5)
        elif is_seventh_chord:
            set_iv(6, -2)

    tone_functions = (0, 2, 4, 6) if is_seventh_chord else (0, 2, 4)
    root_position = {i: [tpcs[i]] for i in tone_functions}
    replacements  = {i: [] for i in tone_functions}

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
        added = add_remove == '+'
        substracted = add_remove == '-'
        replacing_upper = add_remove == '^'
        replacing_lower = add_remove == 'v'
        chord_interval = int(chord_interval) - 1
        ### From here on, `chord_interval` is decremented, i.e. the root is 0, the seventh is 6 etc. (just like in `tpcs`)
        if (chord_interval == 0 and not substracted) or chord_interval > 13:
            logger.warning(
                f"{MC}Change {full} is meaningless and ignored because it concerns chord tone {chord_interval + 1}.")
            continue
        next_octave = chord_interval > 7
        shift = 7 * (acc.count('#') - acc.count('b'))
        new_val = tpcs[chord_interval] + shift
        if substracted:
            if chord_interval not in tone_functions:
                logger.warning(
                    f"{MC}The change {full} has no effect because it concerns an interval which is not implied by {numeral}{form}{figbass}.")
            else:
                root_position[chord_interval] = []
        elif added:
            added_notes.append(new_val)
        elif next_octave:
            if any((replacing_lower, replacing_upper, substracted)):
                logger.warning(f"{MC}{full[0]} has no effect in {full}  because the interval is larger than an octave.")
            added_notes.append(new_val)
        elif chord_interval in [1, 3, 5]:  # these are changes to scale degree 2, 4, 6 that replace the lower neighbour unless they have a # or ^
            if '#' in acc or replacing_upper:
                if '#' in acc and replacing_upper:
                    logger.warning(f"{MC}^ is redundant in {full}.")
                if chord_interval == 5 and is_triad:  # leading tone to 7 but not in seventh chord
                    added_notes.append(new_val)
                else:
                    replace_chord_tone(chord_interval + 1, new_val)
            else:
                if replacing_lower:
                    logger.warning(f"{MC}v is redundant in {full}.")
                replace_chord_tone(chord_interval - 1, new_val)
        else:  # chord tone alterations
            if replacing_lower:
                # TODO: This must be possible, e.g. V(6v5) where 5 is suspension of 4
                logger.warning(f"{MC}{full} -> chord tones cannot replace neighbours, use + instead.")
            elif chord_interval == 6 and figbass != '7':  # 7th are a special case:
                if figbass == '':  # in root position triads they are added
                    # TODO: The standard is lacking a distinction, because the root in root pos. can also be replaced from below!
                    added_notes.append(new_val)
                elif figbass in ['6', '64'] or '#' in acc:  # in inverted triads they replace the root, as does #7
                    replace_chord_tone(0, new_val)
                else:  # otherwise they are unclear
                    logger.warning(
                        f"{MC}In seventh chords, such as {label}, it is not clear whether the {full} alters the 7 or replaces the 8 and should not be used.")
            elif tpcs[chord_interval] == new_val:
                logger.warning(
                    f"{MC}The change {full} has no effect in {numeral}{form}{figbass}")
            else:
                root_position[chord_interval] = [new_val]

    figbass2bass = {
        '': 0,
        '7': 0,
        '6': 1,
        '65': 1,
        '64': 2,
        '43': 2,
        '2': 3
    }
    bass = figbass2bass[figbass]
    chord_tones = []
    for tf in tone_functions[bass:] + tone_functions[:bass]:
        chord_tone, replacing_tones = root_position[tf], replacements[tf]
        if chord_tone == replacing_tones == []:
            logger.debug(f"{MC}{label} results in a chord without {tf + 1}.")
        if chord_tone != []:
            chord_tones.append(chord_tone[0])
            if replacing_tones != []:
                logger.warning(f"{MC}{label} results in a chord tone {tf + 1} AND its replacement(s) {replacing_tones}.")
        chord_tones.extend(replacing_tones)

    bass_tpc = chord_tones[0]
    if bass_only:
        return bass_tpc
    elif merge_tones:
        return tuple(sort_tpcs(chord_tones + added_notes, start=bass_tpc))
    else:
        return {
            'chord_tones': tuple(chord_tones),
            'added_tones': tuple(added_notes),
            'root': root,
        }