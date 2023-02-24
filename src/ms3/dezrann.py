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
    harmony: str
    key: str
    phrase: str
    cadence: str


def transform_df(labels: pd.DataFrame,
                measures: Optional[pd.DataFrame],
                label_column: str = 'label') -> List[DcmlLabel]:
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
                            'cadence': str,
                            'phraseend': str}
        and no missing values.
    measures:
        (optional) Dataframe as found in the 'measures' folder of a DCML corpus for computing quarterbeats for pieces with
        voltas. Requires the columns {'mc': int, 'quarterbeats_all_endings': fractions.Fraction} (ms3 >= 1.0.0).
    label_column: str, optional
        The column that is to be used as label string. Defaults to 'label'.

    Returns
    -------
        List of dictionaries where each represents one row of the input labels.
    """

    if measures is None or "quarterbeats_all_endings" not in measures.columns:
        assert "quarterbeats" in labels.columns, f"Labels are lacking 'quarterbeats': {labels.columns}"
        quarterbeats = labels["quarterbeats"]
    else:
        offset_dict = measures.set_index("mc")["quarterbeats_all_endings"]
        quarterbeats = labels['mc'].map(offset_dict)
        quarterbeats = quarterbeats.astype('float') + (labels.mc_onset * 4.0)
        quarterbeats.rename('quarterbeats', inplace=True)
    transformed_df = pd.concat([quarterbeats, labels.duration_qb.rename('duration'), labels[label_column].rename('label')], axis=1)
    return transformed_df.to_dict(orient='records')
    
def make_dezrann_label(
            quarterbeats: float, duration: float, label: str, origin: Union[str, Tuple[str]]) -> DezrannLabel:
    if isinstance(origin, str):
        layers = [origin]
    else:
        layers = list(origin)
    return DezrannLabel(
        type="Harmony", #TODO: adapt type to current label 
        start=quarterbeats,
        duration=duration,
        tag=label,
        layers=layers
    )

def convert_dcml_list_to_dezrann_list(values_dict: List[DcmlLabel],
                                      cadences: bool = False,
                                      harmony_line: Optional[str] = None,
                                      keys_line: Optional[str] = None,
                                      phrases_line: Optional[str] = None,
                                      raw_line: Optional[str] = None,
                                      origin: Union[str, Tuple[str]] = "DCML") -> DezrannDict:
    label_list = []
    for e in values_dict:
        label_list.append(
            make_dezrann_label(
                quarterbeats=e["quarterbeats"],
                duration=e["duration"],
                label=e["label"],
                origin=origin
            )
        )
    layout = []
    if cadences:
        layout.append({"filter": {"type": "Cadence"}, "style": {"line": "all"}})
    if harmony_line:
        layout.append({"filter": {"type": "Harmony"}, "style": {"line": harmony_line}})
    if keys_line:
        layout.append({"filter": {"type": "Localkey"}, "style": {"line": keys_line}})
    if phrases_line:
        layout.append({"filter": {"type": "Phrase"}, "style": {"line": phrases_line}})
    if raw_line:
        layout.append({"filter": {"type": "Harmony"}, "style": {"line": raw_line}})

    return DezrannDict(labels=label_list, meta={"layout": layout})
    

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
    harmonies_df = pd.read_csv(
        path_labels, sep='\t',
        usecols=['mc', 'mc_onset', 'duration_qb', 'quarterbeats', 'label', 'chord', 'cadence', 'phraseend'],
        converters={'mc_onset': safe_frac}
    )
    try:
        measures_df = pd.read_csv(
            path_measures, sep='\t',
            usecols=['mc', 'quarterbeats_all_endings'],
            converters={'quarterbeats_all_endings': safe_frac}
        )
    except (ValueError, AssertionError) as e:
        measures_df = None
        # raise ValueError(f"{path_measures} could not be loaded as a measure map because of the following error:\n'{e}'")
    try:
        dcml_labels = transform_df(labels=harmonies_df, measures=measures_df)
    except Exception as e:
        raise ValueError(f"Converting {path_labels} failed with the exception '{e}'.")
    dezrann_content = convert_dcml_list_to_dezrann_list(
        dcml_labels,
        cadences=cadences,
        harmony_line=harmonies,
        keys_line=keys,
        phrases_line=phrases,
        raw_line=raw,
        origin=origin
    )
    
    # Manual post-processing  #TODO: improve these cases
    # 1) Avoid NaN values in "duration" (happens in second endings)
    # optional : in the transform_df : transformed_df = transformed_df.replace('NaN', 0) ?
    for label in dezrann_content['labels']:
        if pd.isnull(label['duration']):
            print(f"WARNING: NaN duration detected in label {label}.")
            label['duration'] = 0
    # 2) Remove "start" value in the first label ?
    if dezrann_content['labels'][0]['start'] == 0.:
        del dezrann_content['labels'][0]['start']

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
    # print(transformed)

    #dez = generate_dez('K283-2_measures.tsv', 'K283-2_harmonies.tsv')
    #generate_all_dez()