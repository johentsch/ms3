#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
DCML to Measure Map
===================

Script to convert measure map from a tabular format (one line per measure) into 
JSON format.

Output: 
JSON file (.json) containing the measure-map labels, aligned with the score. 
Here is an example of measure map file structure, in compressed form:
'''
{
  "meter": [
    {
       "onset": 0,
       "time-signature": "4/4",
       "actual-duration": "1/8",
       "bar-number": "0"
    },
    {"onset": 24.5, "bar-number": "7a"},
    {"onset": 28.5, "bar-number": "7b"},
    {"onset": 32.5, "time-signature": "3/4", "bar-number": "8"}
  ]
}
'''

Keywords in measure-map labels:
* `onset`: float. Value indicating the time, in quarter length, between the score's 
beginning (position 0), and the beginning of current measure. Required in each label.
* `bar-number`: str. Nominal numbering for the measure, including repeated measure 
(e.g. "6", "7a", "7b"). Required in each label.
    * TODO: dedicated standard notation for splitted measures (distinct from 'a', 'b')
* `time-signature`: str, optional. Fraction corresponding to the time signature in the 
measure. By default, the time signature is equal to the last previously labelled time 
signature. Indicating a new time signature overrides this value.
* `actual-duration`: str, optional. Fraction corresponding to the actual duration in
the measure, expressed in fraction of a whole note. By default, the actual duration is 
equal to the duration of a full measure with current time-signature
    * For example, a anacrusis with time-signature "4/4" which actually last only
one quarter note will have an `actual-duration` of "1/4".


Used columns in TSV (Tab-Separated Values) DCML measures files:
* `mc`: measure count (XML measures, always starting from 1)
* ...

See also: 
* https://gitlab.com/algomus.fr/dezrann/dezrann/-/issues/1030#note_1122509147
* https://github.com/MarkGotham/bar-measure/

"""
#import argparse
import json
import os
import re
from typing import Dict, List, TypedDict, Union, Tuple, Optional

from fractions import Fraction
import pandas as pd



def safe_frac(s: str) -> Union[Fraction, str]:
    try:
        return Fraction(s)
    except Exception:
        return s

def get_int(mn: str) -> int:
    """
    Return the integer part of a measure number.
    For example, `get_int("7a")` returns 7.
    """
    if type(mn) == int:
        return mn
    regex_mn = "(\d+)([a-z]?)"
    regex_match = re.match(regex_mn, str(mn))
    if not regex_match:
        raise ValueError(f"'{mn}' is not a valid measure number.")
    return int(regex_match.group(1))



def generate_measure_map(file: str) -> List[dict]:
    """
    file: :obj:`str`
        Path to a '_measures.tsv' file in the 'measure' folder of a DCML corpus.
        Requires the columns {'mc': int, 'quarterbeats_all_endings': fractions.Fraction} (ms3 >= 1.0.0).
    """
    try:
        measures_df = pd.read_csv(
            file, sep='\t',
            dtype={'mc': int, 'volta': 'Int64'},
            converters={'quarterbeats_all_endings': safe_frac,
                        'quarterbeats': safe_frac,
                        'act_dur': safe_frac}
        )
    except (ValueError, AssertionError) as e:
        raise ValueError(f"{file} could not be loaded because of the following error:\n'{e}'")
    score_has_voltas = "quarterbeats_all_endings" in measures_df.columns

    measure_map = [] # Measure map, list

    previous_measure_dict = {"bar-number": "0"}
    current_time_sig = None
    for i_measure, measure in measures_df.iterrows():
        measure_mn = str(measure.mn)
        measure_dict = {}
        display_measure = False

        # Time signature, time signature upbeat
        if Fraction(measure.timesig) != Fraction(measure.act_dur):
            # Partial measure
            display_measure = True
            measure_dict["actual-duration"] = str(measure.act_dur)
        if measure.timesig != current_time_sig:
            # New time signature
            # (always the case for first measures: always displayed)
            display_measure = True
            measure_dict["time-signature"] = measure.timesig
            current_time_sig = measure.timesig

        # Measure number
        have_same_number = (get_int(previous_measure_dict["bar-number"]) == get_int(measure_mn))
        if i_measure > 0 and have_same_number:
            # Not the next numbered measure
            if measure_map[-1]["bar-number"] != measure_mn:
                # Add previous measure, which is needed (because not displayed yet)
                measure_map.append(previous_measure_dict)
            measure_map[-1]["bar-number"] += "a" # Add letter to previous measure
            measure_dict["bar-number"] = measure_mn + "b" # And to current measure
        else:
            measure_dict["bar-number"] = measure_mn

        have_consecutive_number = (get_int(previous_measure_dict["bar-number"])+1 == get_int(measure_mn))
        if not have_consecutive_number:
            # Display the new numbering (e.g. if measure numbers were skipped or reinitialized)
            display_measure = True

        # Onsets
        if not score_has_voltas:
            measure_dict["onset"] = float(measure.quarterbeats)
        else:
            measure_dict["onset"] = float(measure.quarterbeats_all_endings)

        if display_measure:
            # i.e. the measure need to be in the compressed version
            measure_map.append(measure_dict)        
        previous_measure_dict = measure_dict

    return measure_map

def save_measure_map(file: str, output_path: str):
    """
    Generate a JSON file containing the measure map converted from
    given TSV (Tab-Separated Value) 'measure' file in DCML format.
    """
    json_content = {"meter": generate_measure_map(file)}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_content, f, indent=2)



if __name__ == "__main__":
    # Test to be run in '<root-folder>/ms3/src/ms3/' as working directory
    # With a clone of Annotated Mozart Sonatas dataset in the <root-folder>
    test_file = os.path.join(
        "..", "..", "..",
        "mozart_piano_sonatas", "measures",
        "K279-3.tsv"
    )
    output_path = os.path.join(".", "K279-3_measuremap.json")
    
    save_measure_map(
        file=test_file,
        output_path=output_path
    )
