#!/usr/bin/env python
"""Tests for `ms3` package."""
import os
import shutil
import tempfile

import pytest
import ms3


def assert_stored_mscx_identical(sc_obj, suffix):
    original_mscx = sc_obj.full_paths['mscx']
    original_path = sc_obj.paths['mscx']
    original_fname = sc_obj.fnames['mscx']
    try:
        tmp_file = tempfile.NamedTemporaryFile(mode='r', suffix='.mscx', dir=original_path, encoding='utf-8', delete=True)
        sc_obj.store_mscx(tmp_file.name)
        assert tmp_file.read() == open(original_mscx, encoding='utf-8').read()
    except AssertionError:
        # store the erroneous tmp_file
        fname = f"{original_fname}{suffix}.mscx"
        tmp_persist = os.path.join(original_path, fname)
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
        score_object.detach_labels('labels')
        score_object.attach_labels('labels')
        assert_stored_mscx_identical(score_object, '_label_reinsertion')




