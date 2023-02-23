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
from typing import Dict, List, TypedDict, Union, Tuple

from fractions import Fraction
import pandas as pd



def safe_frac(s: str) -> Union[Fraction, str]:
    try:
        return Fraction(s)
    except Exception:
        return s

class DezrannLabel(TypedDict):
    type: str #= "Harmony" # Default value ?
    start: float
    duration: float
    line: str #= "top.3" #Literal?
    tag: str
    layers: List[str]

class DezrannDict(TypedDict):
    """Represents one .dez file."""
    labels: List[DezrannLabel]
    meta: Dict

class DcmlLabel(TypedDict):
    quarterbeats: float
    duration: float
    label: str


def transform_df(labels: pd.DataFrame, 
                measures: pd.DataFrame, 
                label_column: str = 'label') -> List[DcmlLabel]:
    """

    Parameters
    ----------
    labels:
        Dataframe as found in the 'harmonies'  folder of a DCML corpus. Needs to have columns with
        the correct dtypes {'mc': int, 'mc_onset': fractions.Fraction} and no missing values.
    measures:
        Dataframe as found in the 'measures' folder of a DCML corpus. Requires the columns
        {'mc': int, 'quarterbeats_all_endings': fractions.Fraction}
    label_column: str, optional
        The column that is to be used as label string. Defaults to 'label'.

    Returns
    -------
        List of dictionaries where each represents one row of the input labels.
    """
    offset_dict = measures.set_index("mc")["quarterbeats_all_endings"]
    quarterbeats = labels['mc'].map(offset_dict)
    quarterbeats = quarterbeats.astype('float') + (labels.mc_onset * 4.0)
    transformed_df = pd.concat([quarterbeats.rename('quarterbeats'), labels.duration_qb.rename('duration'), labels[label_column].rename('label')], axis=1)
    return transformed_df.to_dict(orient='records')
    
def make_dezrann_label(
            quarterbeats: float, duration: float, label: str, origin: Union[str, Tuple[str]]) -> DezrannLabel:
    if isinstance(origin, str):
        layers = [origin]
    else:
        layers = list(origin)
    return DezrannLabel(
        type="Harmony",
        start=quarterbeats,
        duration=duration,
        line="top.3",
        tag=label,
        layers=layers
    )

def convert_dcml_list_to_dezrann_list(values_dict: List[DcmlLabel],
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
    return DezrannDict(labels=label_list, meta={"layout": []})
    

def generate_dez(path_measures: str,
                 path_labels: str,
                 output_path: str = "labels.dez",
                 origin: Union[str, Tuple[str]] = "DCML"):
    """
    path_measures : :obj:`str`
        Path to a TSV file as output by format_data().
    path_labels : :obj:`str`
        Path to a TSV file as output by format_data().
    output_labels : :obj:`str`
        Path to a TSV file as output by format_data().
    origin : :obj:`list`
        List of source(s) from which the labels originate. Defaults to ["DCML"].
    """
    harmonies = pd.read_csv(
        path_labels, sep='\t',
        usecols=['mc', 'mc_onset', 'duration_qb', 'label'], #'chord'
        converters={'mc_onset': safe_frac}
    )
    measures = pd.read_csv(
        path_measures, sep='\t',
        usecols=['mc', 'quarterbeats_all_endings'],
        converters={'quarterbeats_all_endings': safe_frac}
    )
    dcml_labels = transform_df(labels=harmonies, measures=measures)
    dezrann_content = convert_dcml_list_to_dezrann_list(dcml_labels, origin=origin)
    
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
         harmony_layer: int,
         keys_layer:int,
         phrases_layer: int,
         cadences_layer: int,
         raw_layer: int):
    pass

def process_arguments(args) -> dict:
    pass


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
    parser.add_argument("dir", metavar='DIR',
                        help='Folder that will be scanned for TSV files to convert. Defaults to current working directory.')
    parser.add_argument('-m', '--measures', metavar='DIR',
                        help='Folder(s) that will be scanned for TSV files to convert. Defaults to current working directory.')
    parser.add_argument('-o', '--out', metavar='OUT_DIR',
                        help='Output directory for .dez files. Defaults to the input directory.')
    parser.add_argument('-C', 
                        '--cadences', 
                        action="store_true",
                        )
    parser.add_argument('-H', 
                        '--harmonies', 
                        metavar="{1-6}, default: 4",
                        default=4,
                        choices=[1, 2, 3, 4, 5, 6], 
                        )
    parser.add_argument('-K', 
                        '--keys', 
                        metavar="{1-6}, default: 5",
                        default=5,
                        choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument('-P', 
                        '--phrases', 
                        metavar="{1-6}, default: 6",
                        default=6, 
                        choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--raw', 
                        metavar="{1-6}",
                        choices=[1, 2, 3, 4, 5, 6])
    args = parser.parse_args()
    kwargs = process_arguments(args)
    main(**kwargs)

if __name__ == "__main__":
    run()



    #measures = ms3.load_tsv('src/ms3/K283-2_measures.tsv')
    #harmonies = ms3.load_tsv('src/ms3/K283-2_harmonies.tsv')
    #transformed = transform_df(labels=harmonies, measures=measures)
    #print(transformed)
    
    #dez = generate_dez('K283-2_measures.tsv', 'K283-2_harmonies.tsv')
    #generate_all_dez()