#!/usr/bin/env python
"""Tests for `ms3` package."""
import os
import shutil
import tempfile

import pytest
import ms3
from ms3 import assert_dfs_equal


def assert_stored_mscx_identical(sc_obj, suffix):
    original_mscx = sc_obj.full_paths['mscx']
    original_path = sc_obj.paths['mscx']
    original_fname = sc_obj.fnames['mscx']
    fname = f"{original_fname}{suffix}.mscx"
    tmp_persist = os.path.join(original_path, fname)
    try:
        tmp_file = tempfile.NamedTemporaryFile(mode='r', suffix='.mscx', dir=original_path, encoding='utf-8', delete=True)
        sc_obj.store_mscx(tmp_file.name)
        tmp_lines = open(tmp_file.name, encoding='utf-8').readlines()
        expected_lines = open(original_mscx, encoding='utf-8').readlines()
        try:
            assert tmp_lines == expected_lines
        except AssertionError:
            if sorted(tmp_lines) == sorted(expected_lines):
                print(f"The lines in {tmp_persist} come in a different order than in {original_path}.")
                ### uncomment the following line to inspect tmp_persist
                # raise
            else:
                raise
        if os.path.isfile(tmp_persist):
            os.remove(tmp_persist)
    except AssertionError:
        # store the erroneous tmp_file
        shutil.copy(tmp_file.name, tmp_persist)
        raise
    finally:
        tmp_file.close()


class TestBasic:

    def test_init(self):
        s = ms3.Score()
        assert isinstance(s, ms3.score.Score)
        with pytest.raises(LookupError):
            s.mscx


class TestScore:

    test_folder = os.path.dirname(os.path.realpath(__file__))

    def test_store_mscx(self, directory, score_object,):
        assert_stored_mscx_identical(score_object, '_rewrite')

    def test_removing_and_reinserting_labels(self, directory, score_object,):
        if not score_object.mscx.has_annotations:
            return
        before = score_object.annotations.df
        assert len(before) == len(score_object.mscx.parsed.soup.find_all('Harmony')), "Not all <Harmony> tags appear in _.annotations.df !"
        score_object.detach_labels('labels')
        after = score_object.labels.df
        assert_dfs_equal(before, after)
        score_object.attach_labels('labels')
        assert_stored_mscx_identical(score_object, '_label_reinsertion')




