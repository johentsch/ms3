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
from typing import Dict, List, TypedDict, Union, Tuple, Optional

from fractions import Fraction
import pandas as pd



def safe_frac(s: str) -> Union[Fraction, str]:
    try:
        return Fraction(s)
    except Exception:
        return s

