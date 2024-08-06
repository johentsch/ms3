import re
from collections import namedtuple

import webcolors
from ms3._version import __version__

LATEST_MUSESCORE_VERSION = "3.6.2"
COLLECTION_COLUMNS = ["next", "chord_tones", "added_tones"]
BOOLEAN_COLUMNS = ["globalkey_is_minor", "localkey_is_minor", "has_drumset"]
COMPUTED_METADATA_COLUMNS = [
    "TimeSig",
    "KeySig",
    "last_mc",
    "last_mn",
    "length_qb",
    "last_mc_unfolded",
    "last_mn_unfolded",
    "length_qb_unfolded",
    "volta_mcs",
    "all_notes_qb",
    "n_onsets",
    "n_onset_positions",
    "guitar_chord_count",
    "form_label_count",
    "label_count",
    "annotated_key",
]
"""Automatically computed columns"""

DCML_METADATA_COLUMNS = [
    "harmony_version",
    "annotators",
    "reviewers",
    "score_integrity",
    "composed_start",
    "composed_end",
    "composed_source",
]
"""Arbitrary column names used in the DCML corpus initiative"""

MUSESCORE_METADATA_FIELDS = [
    "composer",
    "workTitle",
    "movementNumber",
    "movementTitle",
    "workNumber",
    "poet",
    "lyricist",
    "arranger",
    "copyright",
    "creationDate",
    "mscVersion",
    "platform",
    "source",
    "translator",
]
"""Default fields available in the File -> Score Properties... menu."""

VERSION_COLUMNS = [
    "musescore",
    "ms3_version",
]
"""Software versions"""

INSTRUMENT_RELATED_COLUMNS = ["has_drumset", "ambitus"]

MUSESCORE_HEADER_FIELDS = [
    "title_text",
    "subtitle_text",
    "lyricist_text",
    "composer_text",
    "part_name_text",
]
"""Default text fields in MuseScore"""

OTHER_COLUMNS = [
    "subdirectory",
    "rel_path",
]

LEGACY_COLUMNS = [
    "fname",
    "fnames",
    "rel_paths",
    "md5",
]

AUTOMATIC_COLUMNS = (
    COMPUTED_METADATA_COLUMNS
    + VERSION_COLUMNS
    + INSTRUMENT_RELATED_COLUMNS
    + OTHER_COLUMNS
)
"""This combination of column names is excluded when updating metadata fields in MuseScore files via ms3 metadata."""

METADATA_COLUMN_ORDER = (
    ["piece"]
    + COMPUTED_METADATA_COLUMNS
    + DCML_METADATA_COLUMNS
    + MUSESCORE_METADATA_FIELDS
    + MUSESCORE_HEADER_FIELDS
    + VERSION_COLUMNS
    + OTHER_COLUMNS
    + INSTRUMENT_RELATED_COLUMNS
)
"""The default order in which columns of metadata.tsv files are to be sorted."""

SCORE_EXTENSIONS = (
    ".mscx",
    ".mscz",
    ".cap",
    ".capx",
    ".midi",
    ".mid",
    ".musicxml",
    ".mxl",
    ".xml",
)

STANDARD_COLUMN_ORDER = [
    "mc",
    "mc_playthrough",
    "mn",
    "mn_playthrough",
    "quarterbeats",
    "mc_onset",
    "mn_onset",
    "beat",
    "event",
    "timesig",
    "staff",
    "voice",
    "duration",
    "tied",
    "gracenote",
    "nominal_duration",
    "scalar",
    "tpc",
    "midi",
    "volta",
    "chord_id",
]

STANDARD_NAMES = [
    "notes_and_rests",
    "rests",
    "notes",
    "measures",
    "events",
    "labels",
    "chords",
    "expanded",
    "harmonies",
    "cadences",
    "form_labels",
    "MS3",
    "scores",
]
""":obj:`list`
Indicators for corpora: If a folder contains any file or folder beginning or ending on any of these names, it is
considered to be a corpus by the function :py:func:`iterate_corpora`.
"""

STANDARD_NAMES_OR_GIT = STANDARD_NAMES + [".git"]

DCML_REGEX = re.compile(
    r"""
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
            """,
    re.VERBOSE,
)
""":obj:`str`
Constant with a regular expression that recognizes labels conforming to the DCML harmony annotation standard excluding
those consisting of two alternatives.
"""

DCML_DOUBLE_REGEX = re.compile(
    r"""
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
                                        (?P<numeral2>(b*|\#*)(
                                        VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))
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
    re.VERBOSE,
)
""":obj:`str`
Constant with a regular expression that recognizes complete labels conforming to the DCML harmony annotation standard
including those consisting of two alternatives, without having to split them. It is simply a doubled version of
DCML_REGEX.
"""

FORM_DETECTION_REGEX = r"^\d{1,2}.*?:"
""":obj:`str`
Following Gotham & Ireland (@ISMIR 20 (2019): "Taking Form: A Representation Standard, Conversion Code, and Example
Corpus for Recording, Visualizing, and Studying Analyses of Musical Form"),
detects form labels as those strings that start with indicating a hierarchical level (one or two digits) followed by
a colon.
By extension (Llorens et al., forthcoming), allows one or more 'i' characters or any other alphabet character to
further specify the level.
"""
FORM_LEVEL_REGEX = r"^(?P<level>\d{1,2})?(?P<form_tree>[a-h])?(?P<reading>[ivx]+)?:?$"  # (?P<token>(?:\D|\d+(?!(
# ?:$|i+|\w)?[\&=:]))+)"
FORM_LEVEL_FORMAT = r"\d{1,2}[a-h]?[ivx]*(?:\&\d{0,2}[a-h]?[ivx]*)*"
FORM_LEVEL_SPLIT_REGEX = FORM_LEVEL_FORMAT + ":"
FORM_LEVEL_CAPTURE_REGEX = f"(?P<levels>{FORM_LEVEL_FORMAT}):"
FORM_TOKEN_ABBREVIATIONS = {
    "s": "satz",
    "x": "unit",
    "ah": "anhang",
    "bi": "basic idea",
    "br": "bridge",
    "ci": "contrasting idea",
    "cr": "crux",
    "eg": "eingang",
    "hp": "hauptperiode",
    "hs": "hauptsatz",
    "ip": "interpolation",
    "li": "lead-in",
    "pc": "pre-core",
    "pd": "period",
    "si": "secondary idea",
    "ti": "thematic introduction",
    "tr": "transition",
    "1st": "first theme",
    "2nd": "second theme",
    "3rd": "third theme",
    "ans": "answer",
    "ant": "antecedent",
    "ate": "after-the-end",
    "beg": "beginning",
    "btb": "before-the-beginning",
    "cad": "cadential idea",
    "cbi": "compound basic idea",
    "chr": "chorus",
    "cls": "closing theme",
    "dev": "development section",
    "epi": "episode",
    "exp": "exposition",
    "liq": "liquidation",
    "mid": "middle",
    "mod": "model",
    "mvt": "movement",
    "rec": "recapitulation",
    "rep": "repetition",
    "rit": "ritornello",
    "rtr": "retransition",
    "seq": "sequence",
    "sub": "subject",
    "th0": "theme0",
    "var": "variation",
    "ver": "verse",
    "cdza": "cadenza",
    "coda": "coda",
    "cons": "consequent",
    "cont": "continuation",
    "conti": "continuation idea",
    "ctta": "codetta",
    "depi": "display episode",
    "diss": "dissolution",
    "expa": "expansion",
    "frag": "fragmentation",
    "pcad": "postcadential",
    "pres": "presentation",
    "schl": "schlusssatz",
    "sent": "sentence",
    "intro": "introduction",
    "prchr": "pre-chorus",
    "ptchr": "post-chorus",
}

KEYSIG_DICT_ENTRY_REGEX = r"(\d+): (-?\d+)(?:, )?"
TIMESIG_DICT_ENTRY_REGEX = r"(\d+): (\d+\/\d+)(?:, )?"
SLICE_INTERVAL_REGEX = (
    r"[\[\)]((?:\d+\.?\d*)|(?:\.\d+)), ((?:\d+\.?\d*)|(?:\.\d+))[\)\]]"
)
"""Regular expression for slice interval in open/closed notation and any flavour of floating point numbers,
e.g. [0, 1.5) or (.5, 2.]"""

MS3_HTML = {
    "#005500": "ms3_darkgreen",
    "#aa0000": "ms3_darkred",
    "#aa5500": "ms3_sienna",
    "#00aa00": "ms3_green",
    "#aaaa00": "ms3_darkgoldenrod",
    "#aaff00": "ms3_chartreuse",
    "#00007f": "ms3_navy",
    "#aa007f": "ms3_darkmagenta",
    "#00557f": "ms3_teal",
    "#aa557f": "ms3_indianred",
    "#00aa7f": "ms3_darkcyan",
    "#aaaa7f": "ms3_darkgray",
    "#aaff7f": "ms3_palegreen",
    "#aa00ff": "ms3_darkviolet",
    "#0055ff": "ms3_dodgerblue",
    "#aa55ff": "ms3_mediumorchid",
    "#00aaff": "ms3_deepskyblue",
    "#aaaaff": "ms3_lightsteelblue",
    "#aaffff": "ms3_paleturquoise",
    "#550000": "ms3_maroon",
    "#555500": "ms3_darkolivegreen",
    "#ff5500": "ms3_orangered",
    "#55aa00": "ms3_olive",
    "#ffaa00": "ms3_orange",
    "#55ff00": "ms3_lawngreen",
    "#55007f": "ms3_indigo",
    "#ff007f": "ms3_deeppink",
    "#55557f": "ms3_darkslateblue",
    "#ff557f": "ms3_lightcoral",
    "#55aa7f": "ms3_mediumseagreen",
    "#ffaa7f": "ms3_lightsalmon",
    "#55ff7f": "ms3_lightgreen",
    "#ffff7f": "ms3_khaki",
    "#5500ff": "ms3_blue",
    "#5555ff": "ms3_royalblue",
    "#ff55ff": "ms3_violet",
    "#55aaff": "ms3_cornflowerblue",
    "#ffaaff": "ms3_lightpink",
    "#55ffff": "ms3_aquamarine",
}

MS3_RGB = {
    (0, 85, 0): "ms3_darkgreen",
    (170, 0, 0): "ms3_darkred",
    (170, 85, 0): "ms3_sienna",
    (0, 170, 0): "ms3_green",
    (170, 170, 0): "ms3_darkgoldenrod",
    (170, 255, 0): "ms3_chartreuse",
    (0, 0, 127): "ms3_navy",
    (170, 0, 127): "ms3_darkmagenta",
    (0, 85, 127): "ms3_teal",
    (170, 85, 127): "ms3_indianred",
    (0, 170, 127): "ms3_darkcyan",
    (170, 170, 127): "ms3_darkgray",
    (170, 255, 127): "ms3_palegreen",
    (170, 0, 255): "ms3_darkviolet",
    (0, 85, 255): "ms3_dodgerblue",
    (170, 85, 255): "ms3_mediumorchid",
    (0, 170, 255): "ms3_deepskyblue",
    (170, 170, 255): "ms3_lightsteelblue",
    (170, 255, 255): "ms3_paleturquoise",
    (85, 0, 0): "ms3_maroon",
    (85, 85, 0): "ms3_darkolivegreen",
    (255, 85, 0): "ms3_orangered",
    (85, 170, 0): "ms3_olive",
    (255, 170, 0): "ms3_orange",
    (85, 255, 0): "ms3_lawngreen",
    (85, 0, 127): "ms3_indigo",
    (255, 0, 127): "ms3_deeppink",
    (85, 85, 127): "ms3_darkslateblue",
    (255, 85, 127): "ms3_lightcoral",
    (85, 170, 127): "ms3_mediumseagreen",
    (255, 170, 127): "ms3_lightsalmon",
    (85, 255, 127): "ms3_lightgreen",
    (255, 255, 127): "ms3_khaki",
    (85, 0, 255): "ms3_blue",
    (85, 85, 255): "ms3_royalblue",
    (255, 85, 255): "ms3_violet",
    (85, 170, 255): "ms3_cornflowerblue",
    (255, 170, 255): "ms3_lightpink",
    (85, 255, 255): "ms3_aquamarine",
}

MS3_COLORS = list(MS3_HTML.values())
CSS2MS3 = {c[4:]: c for c in MS3_COLORS}
CSS_COLORS = [ # copied from webcolors.CSS3_NAMES_TO_HEX.keys()        (<=1.13)
 'aliceblue',  #             webcolors._definitions._CSS3_NAMES_TO_HEX (> 1.13)
 'antiquewhite',
 'aqua',
 'aquamarine',
 'azure',
 'beige',
 'bisque',
 'black',
 'blanchedalmond',
 'blue',
 'blueviolet',
 'brown',
 'burlywood',
 'cadetblue',
 'chartreuse',
 'chocolate',
 'coral',
 'cornflowerblue',
 'cornsilk',
 'crimson',
 'cyan',
 'darkblue',
 'darkcyan',
 'darkgoldenrod',
 'darkgray',
 'darkgrey',
 'darkgreen',
 'darkkhaki',
 'darkmagenta',
 'darkolivegreen',
 'darkorange',
 'darkorchid',
 'darkred',
 'darksalmon',
 'darkseagreen',
 'darkslateblue',
 'darkslategray',
 'darkslategrey',
 'darkturquoise',
 'darkviolet',
 'deeppink',
 'deepskyblue',
 'dimgray',
 'dimgrey',
 'dodgerblue',
 'firebrick',
 'floralwhite',
 'forestgreen',
 'fuchsia',
 'gainsboro',
 'ghostwhite',
 'gold',
 'goldenrod',
 'gray',
 'grey',
 'green',
 'greenyellow',
 'honeydew',
 'hotpink',
 'indianred',
 'indigo',
 'ivory',
 'khaki',
 'lavender',
 'lavenderblush',
 'lawngreen',
 'lemonchiffon',
 'lightblue',
 'lightcoral',
 'lightcyan',
 'lightgoldenrodyellow',
 'lightgray',
 'lightgrey',
 'lightgreen',
 'lightpink',
 'lightsalmon',
 'lightseagreen',
 'lightskyblue',
 'lightslategray',
 'lightslategrey',
 'lightsteelblue',
 'lightyellow',
 'lime',
 'limegreen',
 'linen',
 'magenta',
 'maroon',
 'mediumaquamarine',
 'mediumblue',
 'mediumorchid',
 'mediumpurple',
 'mediumseagreen',
 'mediumslateblue',
 'mediumspringgreen',
 'mediumturquoise',
 'mediumvioletred',
 'midnightblue',
 'mintcream',
 'mistyrose',
 'moccasin',
 'navajowhite',
 'navy',
 'oldlace',
 'olive',
 'olivedrab',
 'orange',
 'orangered',
 'orchid',
 'palegoldenrod',
 'palegreen',
 'paleturquoise',
 'palevioletred',
 'papayawhip',
 'peachpuff',
 'peru',
 'pink',
 'plum',
 'powderblue',
 'purple',
 'red',
 'rosybrown',
 'royalblue',
 'saddlebrown',
 'salmon',
 'sandybrown',
 'seagreen',
 'seashell',
 'sienna',
 'silver',
 'skyblue',
 'slateblue',
 'slategray',
 'slategrey',
 'snow',
 'springgreen',
 'steelblue',
 'tan',
 'teal',
 'thistle',
 'tomato',
 'turquoise',
 'violet',
 'wheat',
 'white',
 'whitesmoke',
 'yellow',
 'yellowgreen'
]
COLORS = sum([[c, CSS2MS3[c]] if c in CSS2MS3 else [c] for c in CSS_COLORS], [])
rgba = namedtuple("RGBA", ["r", "g", "b", "a"])
DEFAULT_CREATOR_METADATA = {
    "@context": "https://schema.org/",
    "@type": "SoftwareApplication",
    "@id": "https://pypi.org/project/ms3/",
    "name": "ms3",
    "description": "A parser for MuseScore 3 files and data factory for annotated music corpora.",
    "author": {
        "name": "Johannes Hentschel",
        "@id": "https://orcid.org/0000-0002-1986-9545",
    },
    "softwareVersion": __version__,
}
