#!/usr/bin/env python
"""Tests for `ms3` package."""

import pytest
import os


from ms3 import Score
from ms3.utils import next2sequence



@pytest.mark.parametrize("mscx_file, expected_mc_sequence", [
    ('repeats0.mscx', [1, 2, 3, 6, 2, 4, 1, 2, 5]),
    ('repeats1.mscx', [1, 2, 3, 1, 2, 4, 2, 5, 6]),
    ('repeats2.mscx', [1, 2, 3, 1, 2, 4, 2, 5, 1, 2, 6]),])
def test_repeats(mscx_file, expected_mc_sequence):
    test_folder, _ = os.path.split(os.path.realpath(__file__))
    mscx_path = os.path.realpath(os.path.join(test_folder, 'repeat_dummies', mscx_file))
    s = Score(mscx_path, parser='bs4')
    res = next2sequence(s.mscx.measures.set_index('mc').next)
    assert res == expected_mc_sequence
