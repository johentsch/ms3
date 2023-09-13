"""
DCML to Measure Map
===================

Script to generate a 'measure map' in JSON format from DCML measure
descriptions in TSV format.

Output:
JSON file (.json) containing the measure-map labels, aligned with the score.
Here is an example of measure map file structure, in compressed form:
'''
{
  "meter": [
    {
       "qstamp": 0,
       "name": "0",
       "count": 1,
       "time_signature": "4/4",
       "actual_duration": 0.5,
    },
    {"qstamp": 24.5, "name": "7a", "count": 8},
    {"qstamp": 28.5, "name": "7b", "count": 9},
    {"qstamp": 32.5, "number": "8", "count": 10, "time-signature": "3/4"}
  ]
}
'''

Keywords in measure-map labels:
* `qstamp`: float, required. Value indicating the duration, in quarter length, between
the score's beginning (position 0), and the beginning of current measure. Required in all labels.
* `name`: str, required. Nominal numbering for the measure, including repeated measure
(e.g. "6", "7a", "7b"). Required in each label.
* `number`: int. A number assigned to the measure - usually included in the "name".
* `count`: int. Ordinal position of the 'printed' measure in the piece, so that the
first has counter 1 and the last has the value of the total number of printed measures
in the piece. This value is unique for each measure of the piece.
* `id`: str. Unique identifier of the measure in the piece. It defaults to a string
a the `count` value.
* `time_signature`: str. Fraction corresponding to the time signature in the
measure. By default, the time signature is equal to the last previously labelled time
signature. Indicating a new time signature overrides this value.
* `nominal_length`: float. Standard length, in quarter values, for current time signature.
* `actual_length`: float. Actual length, in quarter values, of the measure. By default,
the actual duration is equal to the nominal duration.
    * For example, a anacrusis with time signature "4/4" which actually last only
one quarter note will have an `actual_length` of 1.0.
* `start-repeat`: bool. If True, indicates a start repeat at the beginning of the measure.
Defaults to False.
* `end-repeat`: bool. If True, indicates an end repeat at the end of the measure. Defaults to False.
* `next`: list of int. By default, the list contains only one value: the following measure,
referred by its following `counter` value. Other values can be added in the case of repeats,
or second endings, for example.


Used columns in TSV (Tab-Separated Values) DCML measures files:
* `mc`: measure count (XML measures, always starting from 1)
* ...

See also:
* https://gitlab.com/algomus.fr/dezrann/dezrann/-/issues/1030#note_1122509147
* https://github.com/MarkGotham/bar-measure/

"""
import json
import os
import re
from fractions import Fraction
from pprint import pprint
from typing import List, Union

import pandas as pd

####
# Utils
####


def safe_frac(s: str) -> Union[Fraction, str]:
    try:
        return Fraction(s)
    except Exception:
        return


def get_int(mn: Union[str, int]) -> int:
    """
    Return the integer part of a string measure name.
    For example, `get_int("7a")` returns 7.
    """
    if isinstance(mn, int):
        return mn
    regex_mn = r"(\d+)([a-z]?)"
    regex_match = re.match(regex_mn, str(mn))
    if not regex_match:
        raise ValueError(f"'{mn}' is not a valid measure number.")
    return int(regex_match.group(1))


####
# Main function
####


def generate_measure_map(file: str, output_file=None, compressed=True) -> List[dict]:
    """
    Generate a measure map in JSON from

    Args:
        file: Path to a '_measures.tsv' file in the 'measure' folder of a DCML corpus.
        Requires the columns {'mc': int, 'mn': int, 'quarterbeats_all_endings': fractions.Fraction} (ms3 >= 1.0.0).
        output_file: TODO
        compressed: TODO

    Returns:
        Measure map: list of measure description.
    """
    try:
        measures_df = pd.read_csv(
            file,
            sep="\t",
            dtype={"mc": int, "volta": "Int64"},
            converters={
                "quarterbeats_all_endings": safe_frac,
                "quarterbeats": safe_frac,
                "act_dur": safe_frac,
            },
        )
    except (ValueError, AssertionError) as e:
        raise ValueError(
            f"{file} could not be loaded because of the following error:\n'{e}'"
        )
    score_has_voltas = "quarterbeats_all_endings" in measures_df.columns

    measure_map = []  # Measure map, list

    previous_measure_dict = {"name": "0"}
    current_time_sig = None
    for i_measure, measure in measures_df.iterrows():
        measure_mn = int(measure.mn)
        measure_mc = int(measure.mc)
        measure_dict = {}
        display_measure = not compressed  # default value

        # Time signature, time signature upbeat
        if Fraction(measure.timesig) != Fraction(measure.act_dur):
            # Partial measure
            display_measure = True
            measure_dict["actual_duration"] = float(
                measure.duration_qb
            )  # str(measure.act_dur)
        if measure.timesig != current_time_sig:
            # New time signature
            # (always the case for first measures: always displayed)
            display_measure = True
            measure_dict["time_signature"] = measure.timesig
            current_time_sig = measure.timesig
            # TODO: nominal_duration

        # Measure number
        have_same_number = get_int(previous_measure_dict["name"]) == measure_mn
        if i_measure > 0 and have_same_number:
            # Not the next numbered measure
            if measure_map[-1]["name"] != str(measure_mn):
                # Add previous measure, which is needed (because not displayed yet)
                measure_map.append(previous_measure_dict)
            measure_map[-1]["name"] += "a"  # Add letter to previous measure
            measure_dict["name"] = str(measure_mn) + "b"  # And to current measure
        else:
            measure_dict["name"] = str(measure_mn)
        # measure_dict["number"] = measure_mn # if needed
        measure_dict["count"] = measure_mc

        have_consecutive_number = (
            get_int(previous_measure_dict["name"]) + 1 == measure_mn
        )
        if not have_consecutive_number:
            # Display the new numbering (e.g. if measure numbers were skipped or reinitialized)
            display_measure = True

        # Onsets / qstamp
        if not score_has_voltas:
            measure_dict["qstamp"] = float(measure.quarterbeats)
        else:
            measure_dict["qstamp"] = float(measure.quarterbeats_all_endings)

        if display_measure:
            # i.e. the measure need to be in the compressed version
            measure_map.append(measure_dict)
        previous_measure_dict = measure_dict
    # TODO: always add last measure

    json_str = {"meter": measure_map}
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_str, f, indent=2)

    return measure_map


####
# Main and tests
####


def main_generate_mm(
    pieces, output_dir=".", verbose=False, compressed=True, stops=False
):
    for i, piece in enumerate(pieces):
        output_path = os.path.join(
            output_dir, os.path.basename(piece).replace(".tsv", "_mm.json")
        )
        measure_map = generate_measure_map(piece, output_path, compressed=compressed)
        if verbose:
            print(os.path.basename(piece))
            pprint(measure_map)
        if stops and i < len(pieces) - 1:
            input("Press enter to continue...")


if __name__ == "__main__":
    # Path to local Annotated Mozart Sonatas repository for testing
    INPUT_DIR = os.path.join(  # from ~ms3/src/ms3 as working directory
        "..", "..", "..", "mozart_piano_sonatas", "measures"
    )
    TEST_PIECES = ["K284-3"]
    TEST_INPUT_PATHS = [
        os.path.join(INPUT_DIR, f"{piece}.tsv") for piece in TEST_PIECES
    ]
    OUTPUT_DIR = "."

    main_generate_mm(
        pieces=TEST_INPUT_PATHS,
        output_dir=OUTPUT_DIR,
        verbose=True,
        stops=True,
        compressed=True,
    )