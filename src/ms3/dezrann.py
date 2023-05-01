#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
DCML to Dezrann
===============

Script to convert contiguous score annotations from a tabular format (one line per label) into
the JSON-LD format .dez, used by the Dezrann annotation tool developed at the Algomus group.

# Intro

The script presents a first application of what is to become a formal standard of a "measure map";
see first discussion points at 

* https://gitlab.com/algomus.fr/dezrann/dezrann/-/issues/1030#note_1122509147)
* https://github.com/MarkGotham/bar-measure/

As an early proxy of a measure map, the current version uses the measure tables that each
DCML corpus provides in its `measures` folder. This is beneficial in the current context because:

1. The files are required for correct, actionable quarter-note positions without having to re-parse 
  the entire score.
2. The files play an essential role for validating the conversion output.
3. They help avoiding the confusion that necessarily arises when several addressing schemes are 
  at play.

In detail:

## 1. Quarterbeats

From a technical perspective, offsets in the sense of "distance from the origin" represent the
primary mechanism of referencing positions in a text (character counts being the default in NLP).
Music scores are typically aligned with a time line of "musical time", an alignment which is 
frequently expressed as float values representing an event's distance from the score's beginning, 
measured in quarter notes, here referred to as quarterbeats. The fundamental problem, however, is
ensuring that quarterbeat positions refer to the same time line. The commonplace
score encoding formats do not indicate quarterbeat positions. Instead, they structure
musical time in a sequence of containers, generally called "measures", each of which represents 
a time line starting from 0. Counting measure units (of some kind) therefore represents the second 
prevalent way of indicating positions in a score, together with an event onset indicating an
event's distance from the container's beginning. To avoid terminological confusion, we call 
the distance from the beginning of a measure container "onset".

Looking at a single score, there is an unambiguous mapping between the two types of positions:
`event_offset = measure_offset + event_onset`. Problems arise, however when information from one
score is to be set into relation with timed information from another source. This is a wide-spread
problem in the context of music research and musical corpus studies where data from different
sources with different ways of expressing timestamps frequently needs to be aligned, often in 
absence of the original score that one of the source is aligned to. Currently, there is no
standardized way of storing such alignments for later re-use. Hence the idea of a central
mapping file for storing alignments between positions given as quarterbeats, measure+onset,
recording timestamps in seconds, IDs, and other data relevant for score addressability.

**Different types of quarterbeats**

All TSV files issued by the DCML come with the column `quarterbeats` indicating every event's
offset from the score's beginning (position 0). With the caveat that, in the case of first/second endings 
("voltas"), the indicated values do not take into account any but the second ending, with the
rationale that they should represent the temporal proportion of a single playthrough without any
repetitions. For correct conversion, therefore, using a strict, measuring-stick-based variant
of `quarterbeats` will probably be useful. This means that the default `quarterbeats` should be 
ignored (unless first endings are to be categorically excluded) in favour of a
`quarterbeats_all_endings` column. Since the DCML measure maps already come with columns of both 
names, the simple formula mentioned above `quarterbeats = quarterbeats(measure) + event_onset` 
has its analogue `quarterbeats_all_measures = quarterbeats_all_measures(measure) + event_onset`.

Input: DataFrame containing DCML harmony labels as output via the command `ms3 extract -X` 
(X for 'expanded'), stored by default in a folder called 'harmonies'. Using these TSV files
ensures using only valid DCML labels but in principle this script can be used for converting
labels of all kinds as long as they come in the specified tabular format.

## 2. Validating the output

Going from a `DcmlLabel` dictionary to a `DezrannLabel` dictionary is straightforward because
they exchange positions as quarterbeats. Validation, on the other hand, requires relating
the output .dez format with the converted score which it is layed over in Dezrann. In the 
interface, positions are shown to the user in terms of `measure_count + event_onset`. Extracting
this information and comparing it to the one in the original TSVs will 

Columns:

* `mc`: measure count (XML measures, always starting from 1)
* 




Output:
JSON Dezrann file (.dez) containing all the harmony labels, aligned with the score. 
Here is an example of Dezrann file structure:
'''
{
  "labels": [
    {"type": "Harmony", "start": 0, "duration": 4, "line": "top.3", "tag": "I{"},
    {"type": "Harmony", "start": 4, "duration": 4, "line": "top.3", "tag": "V(64)"},
    {"type": "Harmony", "start": 8, "duration": 4, "line": "top.3", "tag": "V}"},
    ...
}
'''
"""
import argparse
import json
import os
from typing import Dict, List, TypedDict, Union, Tuple, Optional

from fractions import Fraction
import pandas as pd



def safe_frac(s: str) -> Union[Fraction, str]:
    try:
        return Fraction(s)
    except Exception:
        return s

class DezrannLabel(TypedDict):
    """Represents one label in a .dez file."""
    type: str
    start: float
    duration: float
    #line: str # Determined by the meta-layout
    tag: str
    layers: List[str]

class DezrannDict(TypedDict):
    """Represents one .dez file."""
    labels: List[DezrannLabel]
    meta: Dict

class DcmlLabel(TypedDict):
    """Represents one label from a TSV annotation file"""
    quarterbeats: float
    duration: float
    label: str


def get_volta_groups(mc2volta: pd.Series) -> List[List[int]]:
    """Takes a Series where the index has measure counts and values are NA for 'normal' measures and 1, 2... for
    measures belonging to a first, second... ending. Returns for each group a list of MCs each of which pertains
    to the first measure of an alternative ending. For example, two alternative two-bar endings in MC [15, 16][17, 18]
    would figure as [15, 17] in the result list.
    """
    volta_groups = []
    filled_volta_col = mc2volta.fillna(-1).astype(int)
    volta_segmentation = (filled_volta_col != filled_volta_col.shift()).fillna(True).cumsum()
    current_groups_first_mcs = []
    for i, segment in filled_volta_col.groupby(volta_segmentation):
        volta_number = segment.iloc[0]
        if volta_number == -1:
            # current group ends, if there is one
            if i == 1:
                continue
            elif len(current_groups_first_mcs) == 0:
                raise RuntimeError(f"Mistake in the algorithm when processing column {filled_volta_col.volta}")
            else:
                volta_groups.append(current_groups_first_mcs)
                current_groups_first_mcs = []
        else:
            first_mc = segment.index[0]
            current_groups_first_mcs.append(first_mc)
    return volta_groups

def transform_df(labels: pd.DataFrame,
                 measures: pd.DataFrame,
                 label_column: str = 'label',
                 ) -> List[DcmlLabel]:
    """

    Parameters
    ----------
    labels:
        Dataframe as found in the 'harmonies'  folder of a DCML corpus. Needs to have columns with
        the correct dtypes {'mc': int,
                            'mc_onset': fractions.Fraction,
                            'duration_qb': float,
                            'quarterbeats': fraction.Fraction,
                            'label': str,
                            'chord': str,
                            'localkey': str,
                            'cadence': str,
                            'phraseend': str}
        and no missing values.
    measures:
        (optional) Dataframe as found in the 'measures' folder of a DCML corpus for computing quarterbeats for pieces with
        voltas. Requires the columns {'mc': int, 'quarterbeats_all_endings': fractions.Fraction} (ms3 >= 1.0.0).
    label_column: {'label', 'chord', 'cadence', 'phraseend'}
        The column that is to be used as label string. Defaults to 'label'.

    Returns
    -------
        List of dictionaries where each represents one row of the input labels.
    """
    score_has_voltas = "quarterbeats_all_endings" in measures.columns
    last_mc_row = measures.iloc[-1]
    end_of_score = float(last_mc_row.act_dur) * 4.0
    if not score_has_voltas:
        assert "quarterbeats" in labels.columns, f"Labels are lacking 'quarterbeats' column: {labels.columns}"
        quarterbeats = labels["quarterbeats"]
        end_of_score += float(last_mc_row.quarterbeats)
    else:
        # the column 'quarterbeats_all_endings' is present, meaning the piece has first and second endings and the
        # quarterbeats, which normally leave out first endings, need to be recomputed
        end_of_score += float(last_mc_row.quarterbeats_all_endings)
        M = measures.set_index("mc")
        offset_dict = M["quarterbeats_all_endings"]
        quarterbeats = labels['mc'].map(offset_dict)
        quarterbeats = quarterbeats + (labels.mc_onset * 4.0)
        quarterbeats.rename('quarterbeats', inplace=True)
        # also, the first beat of each volta needs to have a label for computing correct durations
        volta_groups = get_volta_groups(M.volta)
    label_and_qb = pd.concat([labels[label_column].rename('label'), quarterbeats.astype(float)], axis=1)
    n_before = len(labels.index)
    if label_column == 'phraseend':
        label_and_qb = label_and_qb[label_and_qb.label == '{']
    if label_column == 'localkey':
        label_and_qb = label_and_qb[label_and_qb.label != label_and_qb.label.shift().fillna(True)]
    else: # {'chord', 'cadence', 'label'}
        label_and_qb = label_and_qb[label_and_qb.label.notna()]
    n_after = len(label_and_qb.index)
    print(f"Creating labels for {n_after} {label_column} labels out of {n_before} rows.")
    if label_column == 'cadence':
        duration = pd.Series(0.0, dtype=float, index=label_and_qb.index, name='duration')
    else:
        if score_has_voltas:
            for group in volta_groups:
                volta_beginnings_quarterbeats = [M.loc[mc, 'quarterbeats_all_endings'] for mc in group]
                labels_before_group = label_and_qb.loc[label_and_qb.quarterbeats < volta_beginnings_quarterbeats[0], 'label']
                for volta_beginning_qb in volta_beginnings_quarterbeats:
                    if volta_beginning_qb in label_and_qb.quarterbeats.values:
                        continue
                    repeated_label = pd.DataFrame([[labels_before_group.iloc[-1], float(volta_beginning_qb)]],
                                                  columns=['label', 'quarterbeats'])
                    label_and_qb = pd.concat([label_and_qb, repeated_label], ignore_index=True)
            label_and_qb = label_and_qb.sort_values('quarterbeats')
        qb_column = label_and_qb.quarterbeats
        duration = qb_column.shift(-1).fillna(end_of_score) - qb_column
        duration = duration.rename('duration').astype(float)
    transformed_df = pd.concat([label_and_qb, duration], axis=1)
    return transformed_df.to_dict(orient='records')

def make_dezrann_label(
            label_type: str,
            quarterbeats: float,
            duration: float,
            label: str,
            origin: Union[str, Tuple[str]]) -> DezrannLabel:
    if isinstance(origin, str):
        layers = [origin]
    else:
        layers = list(origin)
    return DezrannLabel(
        type=label_type,
        start=quarterbeats,
        duration=duration,
        tag=label,
        layers=layers
    )

def convert_dcml_list_to_dezrann_list(values_dict: List[DcmlLabel],
                                      label_type: str,
                                      origin: Union[str, Tuple[str]] = "DCML") -> DezrannDict:
    dezrann_label_list = []
    for e in values_dict:
        dezrann_label_list.append(
            make_dezrann_label(
                label_type=label_type,
                quarterbeats=e["quarterbeats"],
                duration=e["duration"],
                label=e["label"],
                origin=origin
            )
        )

    return dezrann_label_list

def make_layout(
               cadences: bool = False,
               harmonies: Optional[str] = None,
               keys: Optional[str] = None,
               phrases: Optional[str] = None,
               raw: Optional[str] = None):
    """
    Compile the line positions for target labels into Dezrann layout parameter.
    """
    layout = []
    if cadences:
        layout.append({"filter": {"type": "Cadence"}, "style": {"line": "all"}})
    if harmonies:
        layout.append({"filter": {"type": "Harmony"}, "style": {"line": harmonies}})
    if keys:
        layout.append({"filter": {"type": "Local Key"}, "style": {"line": keys}})
    if phrases:
        layout.append({"filter": {"type": "Phrase"}, "style": {"line": phrases}})
    if raw:
        layout.append({"filter": {"type": "Harmony"}, "style": {"line": raw}})

    return layout
    
def generate_dez(path_measures: str,
                 path_labels: str,
                 output_path: str = "labels.dez",
                 cadences: bool = False,
                 harmonies: Optional[str] = None,
                 keys: Optional[str] = None,
                 phrases: Optional[str] = None,
                 raw: Optional[str] = None,
                 origin: Union[str, Tuple[str]] = "DCML"):
    """
    path_measures : :obj:`str`
        Path to a TSV file as output by format_data().
    path_labels : :obj:`str`
        Path to a TSV file as output by format_data().
    output_labels : :obj:`str`
        Path to a TSV file as output by format_data().
    origin : :obj:`tuple`
        Tuple of source(s) from which the labels originate. Defaults to "DCML".
    """
    try:
        harmonies_df = pd.read_csv(
            path_labels, sep='\t',
            converters={'mc': int,
                        'mc_onset': safe_frac,
                        'quarterbeats': safe_frac,
                        }
        )
    except (ValueError, AssertionError, FileNotFoundError) as e:
        raise ValueError(f"{path_labels} could not be loaded as a measure map because of the following error:\n'{e}'")
    try:
        measures_df = pd.read_csv(
            path_measures, sep='\t',
            dtype={'mc': int, 'volta': 'Int64'},
            converters={'quarterbeats_all_endings': safe_frac,
                        'quarterbeats': safe_frac,
                        'act_dur': safe_frac}
        )
    except (ValueError, AssertionError, FileNotFoundError) as e:
        raise ValueError(f"{path_measures} could not be loaded as a measure map because of the following error:\n'{e}'")

    dezrann_labels = []
    if cadences:
        dcml_labels = transform_df(labels=harmonies_df, measures=measures_df, label_column='cadence')
        dezrann_labels += convert_dcml_list_to_dezrann_list(dcml_labels, label_type="Cadence", origin=origin)
    for arg, label_column, label_type in ((harmonies, "chord", "Harmony"), #Third argument
                                          (keys, "localkey", "Local Key"),
                                          (phrases, "phraseend", "Phrase"),
                                          (raw, "label", "Harmony")):
        if arg is not None:
            dcml_labels = transform_df(labels=harmonies_df, measures=measures_df, label_column=label_column)
            dezrann_labels += convert_dcml_list_to_dezrann_list(
                dcml_labels,
                label_type=label_type,
                origin=origin
            )
    
    layout = make_layout(
        cadences=cadences,
        harmonies=harmonies,
        keys=keys,
        phrases=phrases,
        raw=raw
    )
    dezrann_content = DezrannDict(labels=dezrann_labels, meta={"layout": layout})

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dezrann_content, f, indent=2)


# Test
MOZART_SONATAS = [
    'K279-1', 'K279-2', 'K279-3',
    'K280-1', 'K280-2', 'K280-3',
    'K283-1', 'K283-2', 'K283-3',
]
MEASURE_DIR = os.path.join("src", "ms3") #to be updated
HARMONY_DIR = os.path.join("src", "ms3") #to be updated
MEASURE_PATHS = [
    os.path.join(MEASURE_DIR, f"{movement}_measures.tsv")
    for movement in MOZART_SONATAS
]
HARMONY_PATHS = [
    os.path.join(HARMONY_DIR, f"{movement}_harmonies.tsv")
    for movement in MOZART_SONATAS
]

OUTPUT_DIR = "." #to be updated
def generate_all_dez(output_dir=OUTPUT_DIR):
    for i_piece, piece in enumerate(MOZART_SONATAS):
        generate_dez(MEASURE_PATHS[i_piece], HARMONY_PATHS[i_piece])

def main(input_dir: str,
         measures_dir: str,
         output_dir: str,
         cadences: bool = False,
         harmonies: Optional[str] = None,
         keys: Optional[str] = None,
         phrases: Optional[str] = None,
         raw: Optional[str] = None):
    if not cadences and all(arg is None for arg in (harmonies, keys, phrases, raw)):
        print(f"Nothing to do because no features have been selected.")
        return
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.tsv')]
    # measures_files = glob.glob(f"{measures_dir}/*.tsv")
    harmony_measure_matches = []
    for tsv_name in input_files:
        measures_file_path = os.path.join(measures_dir, tsv_name)
        if not os.path.isfile(measures_file_path):
            # could be a directory
            continue
        if os.path.isfile(measures_file_path):
            harmonies_file_path = os.path.join(input_dir, tsv_name)
            harmony_measure_matches.append((harmonies_file_path, measures_file_path))
        else:
            print(f"No measure map found for {tsv_name}. Skipping.")
            continue
    if len(harmony_measure_matches) == 0:
        print(f"No matching measure maps found for any of these files: {input_files}")
        return
    for input_file, measure_file in harmony_measure_matches:
        if output_dir == input_dir:
            output_file_path = input_file.replace(".tsv", ".dez")
        else:
            dez_file = os.path.basename(measure_file).replace(".tsv", ".dez")
            output_file_path = os.path.join(output_dir, dez_file)
        try:
            generate_dez(
                path_labels=input_file,
                path_measures=measure_file,
                output_path=output_file_path,
                cadences=cadences,
                harmonies=harmonies,
                keys=keys,
                phrases=phrases,
                raw=raw
            )
            print(f"{output_file_path} successfully written.")
        except Exception as e:
            print(f"Converting {input_file} failed with '{e}'")

LINE_VALUES = {
    1: "top.1",
    2: "top.2",
    3: "top.3",
    4: "bot.1",
    5: "bot.2",
    6: "bot.3"
}

def transform_line_argument(line: Optional[Union[int, str]]) -> Optional[str]:
    """Takes a number bet"""
    if line is None:
        return
    try:
        line = int(line)
        assert line in [1,2,3,4,5,6, 0 -1, -2, -3]
    except (TypeError, ValueError, AssertionError):
        raise ValueError(f"{line} is not a valid argument, should be within [0, 6].")
    if line == 0:
        return None
    if line < 0:
        line = abs(line) + 3
    return LINE_VALUES[line]

def resolve_dir(d):
    """ Resolves '~' to HOME directory and turns ``d`` into an absolute path.
    """
    if d is None:
        return None
    d = str(d)
    if '~' in d:
        return os.path.expanduser(d)
    return os.path.abspath(d)
    

def process_arguments(args: argparse.Namespace) -> dict:
    """Transforms the user's input arguments into keyword arguments for :func:`main` or raises a ValueError."""
    input_dir = resolve_dir(args.dir)
    assert os.path.isdir(input_dir), f"{args.dir} is not an existing directory."
    if args.measures is None:
        measures_dir = os.path.abspath(os.path.join(input_dir, '..', 'measures'))
        if not os.path.isdir(measures_dir):
            raise ValueError(f"No directory with measure maps was specified and the default path "
            f"{measures_dir} does not exist.")
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
    kwargs = dict(
        input_dir=input_dir,
        measures_dir=measures_dir,
        output_dir=output_dir
    )
    line_args = ('harmonies', 'keys', 'phrases', 'raw')
    transformed_line_args = {}
    for arg in line_args:
        arg_val = getattr(args, arg)
        if arg_val is None:
            continue
        line_arg = transform_line_argument(arg_val)
        if line_arg is None:
            continue
        transformed_line_args[arg] = line_arg
    if len(set(transformed_line_args.values())) < len(transformed_line_args.values()):
        selected_args = {arg: f"'{getattr(args, arg)}' => {arg_val}" for arg, arg_val in transformed_line_args.items()}
        raise ValueError(f"You selected the same annotation layer more than once: {selected_args}.")
    kwargs.update(transformed_line_args)
    if args.cadences:
        kwargs['cadences'] = True
    print(kwargs)
    return kwargs


def run():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''\
-----------------------------
| DCML => Dezrann converter |
-----------------------------

This script converts DCML harmony annotations into the .dez JSON format used by the dezrann.net app. It is 
standalone and does not require ms3 to be installed. Its only requirement is pandas.

Apart from that, the script requires that you have previously extracted both harmonies and measures from the 
annotated scores or that you are converting a DCML corpus (https://github.com/DCMLab/dcml_corpora), 
where both facets are provided by default. In order to (re-) extract the labels, use the command:

    ms3 extract -X -M

Or, if you want to convert other harmony or chord labels from your MuseScore files, use -L for labels.
ms3 extract -h will show you all options.
''')
    parser.add_argument("dir", metavar='IN_DIR',
                        help='Folder that will be scanned for TSV files to convert. Defaults to current working directory. '
                             'Sub-directories are not taken into account.')
    parser.add_argument('-m', '--measures', metavar='MEASURES_DIR',
                        help="Folder in which to look for the corrsponding measure maps. By default, the script will try "
                             "to find a sibling to the source dir called 'measures'.")
    parser.add_argument('-o', '--out', metavar='OUT_DIR',
                        help='Output directory for .dez files. Defaults to the input directory.')
    parser.add_argument('-C', 
                        '--cadences', 
                        action="store_true",
                        help="Pass this flag if you want to add time-point cadence labels to the .dez files."
                        )
    possible_line_arguments = ("0", "1", "2", "3", "4", "5", "6", "-1", "-2", "-3")
    parser.add_argument('-H',
                        '--harmonies',
                        metavar="{0-6}, default: 4",
                        default="4",
                        choices=possible_line_arguments,
                        help="By default, harmony annotations will be set on the first line under the system (layer "
                             "4 out of 6). Pick another layer or pass 0 to not add harmonies."
                        )
    parser.add_argument('-K', 
                        '--keys', 
                        metavar="{0-6}, default: 5",
                        default="5",
                        choices=possible_line_arguments,
                        help="By default, local key segments will be set on the second line under the system (layer "
                             "5 out of 6). Pick another layer or pass 0 to not add key segments. Note, however, "
                             "that harmonies are underdetermined without their local key.")
    parser.add_argument('-P', 
                        '--phrases', 
                        metavar="{0-6}, default: 6",
                        default="6", 
                        choices=possible_line_arguments,
                        help="By default, phrase annotations will be set on the third line under the system (layer "
                             "6 out of 6). Pick another layer or pass 0 to not add phrases.")
    parser.add_argument('--raw', 
                        metavar="{1-6}",
                        choices=possible_line_arguments,
                        help="Pass this argument to add a layer with the 'raw' labels, i.e. including local key, "
                             "cadence and phrase annotations.")
    args = parser.parse_args()
    kwargs = process_arguments(args)
    main(**kwargs)

if __name__ == "__main__":
    run()


    # import ms3
    # measures = ms3.load_tsv('K283-2_measures.tsv')
    # harmonies = ms3.load_tsv('K283-2_harmonies.tsv')
    # transformed = transform_df(labels=harmonies, measures=measures)

    #dez = generate_dez('K283-2_measures.tsv', 'K283-2_harmonies.tsv', cadences=True, harmonies="bot.4", keys="bot.5", phrases="bot.6", raw="top.3")
    #generate_all_dez()