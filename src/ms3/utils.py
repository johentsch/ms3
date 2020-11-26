import os, platform, re, shutil, subprocess
from collections import defaultdict, namedtuple
from collections.abc import Iterable
from contextlib import contextmanager
from fractions import Fraction as frac
from itertools import repeat, takewhile
from shutil import which
from tempfile import NamedTemporaryFile as Temp
from zipfile import ZipFile as Zip

import pandas as pd
import numpy as np
import webcolors

from .logger import function_logger, update_cfg

DCML_REGEX = re.compile(r"""
            ^(\.?
                ((?P<globalkey>[a-gA-G](b*|\#*))\.)?
                ((?P<localkey>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\.)?
                ((?P<pedal>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\[)?
                (?P<chord>
                    (?P<numeral>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))
                    (?P<form>(%|o|\+|M|\+M))?
                    (?P<figbass>(7|65|43|42|2|64|6))?
                    (\((?P<changes>((\+|-|\^|v)?(b*|\#*)\d)+)\))?
                    (/(?P<relativeroot>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)*))?
                )
                (?P<pedalend>\])?
            )?
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
                                    ((?P<localkey>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\.)?
                                    ((?P<pedal>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\[)?
                                    (?P<chord>
                                        (?P<numeral>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))
                                        (?P<form>(%|o|\+|M|\+M))?
                                        (?P<figbass>(7|65|43|42|2|64|6))?
                                        (\((?P<changes>((\+|-|\^|v)?(b*|\#*)\d)+)\))?
                                        (/(?P<relativeroot>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)*))?
                                    )
                                    (?P<pedalend>\])?
                                  )?
                                  (?P<phraseend>(\\\\|\}\{|\{|\})
                                  )?
                                 )
                                 (-
                                  (?P<second>
                                    ((?P<globalkey2>[a-gA-G](b*|\#*))\.)?
                                    ((?P<localkey2>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\.)?
                                    ((?P<pedal2>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\[)?
                                    (?P<chord2>
                                        (?P<numeral2>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))
                                        (?P<form2>(%|o|\+|M|\+M))?
                                        (?P<figbass2>(7|65|43|42|2|64|6))?
                                        (\((?P<changes2>((\+|-|\^|v)?(b*|\#*)\d)+)\))?
                                        (/(?P<relativeroot2>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)*))?
                                    )
                                    (?P<pedalend2>\])?
                                  )?
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
    def __missing__(self, key):
        return key


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
    return f"{ambitus['min_midi']}-{ambitus['max_midi']} ({ambitus['min_name']}-{ambitus['max_name']})"


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
    res = res.apply(lambda c: c.str.replace('^/$', 'empty_harmony'))
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


def decode_harmonies(df, label_col='label', keep_type=True, return_series=False):
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
    new_label_col = new_label_col.str.replace('^/$', 'empty_harmony').replace('', np.nan)

    if return_series:
        return new_label_col

    if 'label_type' in df.columns:
        if keep_type:
            df.loc[df.label_type.isin([1, 2, 3, '1', '2', '3']), 'label_type'] == 0
        else:
            drop_cols.append('label_type')
    df[label_col] = new_label_col
    df.drop(columns=drop_cols, inplace=True)
    return df


@function_logger
def convert(old, new, MS='mscore'):
    process = [MS, '--appimage-extract-and-run', "-o", new, old] if MS.endswith('.AppImage') else [MS, "-o", new, old]
    if subprocess.run(process):
        logger.info(f"Converted {old} to {new}")
    else:
        logger.warning("Error while converting " + old)


def convert_folder(dir, new_folder, extensions=[], target_extension='mscx', regex='.*', suffix=None, recursive=True,
                   ms='mscore', overwrite=False, parallel=False):
    """ Convert all files in `dir` that have one of the `extensions` to .mscx format using the executable `MS`.

    Parameters
    ----------
    dir, new_folder : str
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

    conversion_params = []
    for subdir, dirs, files in os.walk(dir):
        if not recursive:
            dirs[:] = []
        else:
            dirs.sort()
        old_subdir = os.path.relpath(subdir, dir)
        new_subdir = os.path.join(new_folder, old_subdir) if old_subdir != '.' else new_folder

        for file in files:
            name, ext = os.path.splitext(file)
            ext = ext[1:]
            if re.search(regex, file) and (ext in extensions or extensions == []):
                if not os.path.isdir(new_subdir):
                    os.makedirs(new_subdir)
                if target_extension[0] == '.':
                    target_extension = target_extension[1:]
                if suffix is not None:
                    neu = '%s%s.%s' % (name, suffix, target_extension)
                else:
                    neu = '%s.%s' % (name, target_extension)
                old = os.path.join(subdir, file)
                new = os.path.join(new_subdir, neu)
                if overwrite or not os.path.isfile(new):
                    conversion_params.append((old, new, MS))
                else:
                    print(new, 'exists already. Pass -o to overwrite.')

    # TODO: pass filenames as 'logger' argument to convert()
    if parallel:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        pool.starmap(convert, conversion_params)
        pool.close()
        pool.join()
    else:
        for o, n, ms in conversion_params:
            convert(o, n, ms)



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


def get_ms_version(mscx_file):
    with open(mscx_file) as file:
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


def iterable2str(iterable):
    try:
        return ', '.join(str(s) for s in iterable)
    except:
        return iterable


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

    def safe_frac(s):
        try:
            return frac(s)
        except:
            return s

    CONVERTERS = {
        'added_tones': str2inttuple,
        'act_dur': safe_frac,
        'chord_tones': str2inttuple,
        'globalkey_is_minor': int2bool,
        'localkey_is_minor': int2bool,
        'next': str2inttuple,
        'nominal_duration': safe_frac,
        'mc_offset': safe_frac,
        'mc_onset': safe_frac,
        'mn_onset': safe_frac,
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


@function_logger
def scan_directory(dir, file_re=r".*", folder_re=r".*", exclude_re=r"^(\.|_)", recursive=True):
    """ Get a list of files.

    Parameters
    ----------
    dir : :obj:`str`
        Directory to be scanned for files.
    file_re, folder_re : :obj:`str` or :obj:`re.Pattern`, optional
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

    if not os.path.isdir(dir):
        logger.warning("Not an existing directory: " + dir)
    res = []
    for subdir, dirs, files in os.walk(dir):
        _, current_folder = os.path.split(subdir)
        if recursive and check_regex('', current_folder):
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
def split_alternatives(df, column='label', regex=r"-(?!(\d|b+\d|\#))", max=2, inplace=False, alternatives_only=False):
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


def chunkstring(string, length=80):
    """ Generate chunks of a given length """
    string = str(string)
    return (string[0 + i:length + i] for i in range(0, len(string), length))


def string2lines(string, length=80):
    """ Use chunkstring() and make chunks into lines. """
    return '\n'.join(chunkstring(string, length))


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





@contextmanager
def unpack_mscz(mscz, dir=None):
    if dir is None:
        dir = os.path.dirname(mscz)
    tmp_file = Temp(suffix='.mscx', prefix='.', dir=dir, delete=False)
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