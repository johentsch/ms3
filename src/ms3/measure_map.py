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
pass
