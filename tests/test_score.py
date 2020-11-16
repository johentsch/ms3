#!/usr/bin/env python
"""Tests for `ms3` package."""

import pytest
import os
import tempfile

import ms3
from ms3.utils import assert_all_lines_equal, assert_dfs_equal, decode_harmonies, load_tsv

@pytest.fixture(
    params=['Did03M-Son_regina-1762-Sarti.mscx', 'D973deutscher01.mscx', '05_symph_fant.mscx', 'BWV_0815.mscx', 'K281-3.mscx', '76CASM34A33UM.mscx'],
ids=['sarti', "schubert", "berlioz", 'bach', 'mozart', 'monty'])
def score_object(request):
    test_folder, _ = os.path.split(os.path.realpath(__file__))
    mscx_path = os.path.join(test_folder, 'MS3', request.param)
    s = ms3.Score(mscx_path, parser='bs4')
    return s

class TestBasic:

    def test_init(self):
        s = ms3.Score()
        assert isinstance(s, ms3.score.Score)
        with pytest.raises(LookupError):
            s.mscx


class TestParser:

    test_folder, _ = os.path.split(os.path.realpath(__file__))

    def test_parse_and_write_back(self, score_object):
        original_mscx = score_object.full_paths['mscx']
        tmp_file = tempfile.NamedTemporaryFile(mode='r', suffix='.mscx', dir=self.test_folder)
        if score_object.mscx.has_annotations:
            score_object.detach_labels('labels')
            score_object.attach_labels('labels')
        score_object.store_mscx(tmp_file.name)
        original = open(original_mscx).read()
        after_parsing = tmp_file.read()
        assert_all_lines_equal(original, after_parsing, original=original_mscx, tmp_file=tmp_file)


    def test_store_and_load_labels(self, score_object):
        if score_object.mscx.has_annotations:
            fname = score_object.fnames['mscx'] + '_labels.tsv'
            score_object.store_annotations(tsv_path=fname)
            score_object.load_annotations(fname, key='tsv')
            score_object.detach_labels('labels')
            score_object.attach_labels('tsv')
            tmp_file = tempfile.NamedTemporaryFile(mode='r', suffix='.tsv', dir=self.test_folder)
            score_object.store_mscx(tmp_file.name)
            original_mscx = score_object.full_paths['mscx']
            before = open(original_mscx).read()
            after = tmp_file.read()
            assert_all_lines_equal(before, after, original=original_mscx, tmp_file=tmp_file)

    def test_expanded_labels(self, score_object):
        if score_object.mscx.has_annotations:
            fname = score_object.fnames['mscx']
            fpath = os.path.join(score_object.paths['mscx'], '..')
            new_path = os.path.join(fpath, fname + '_labels.tsv')
            old_path = os.path.join(fpath, 'harmonies', fname + '.tsv')
            new_expanded = load_tsv(new_path)
            old_expanded = load_tsv(old_path)
            assert_dfs_equal(old_expanded, decode_harmonies(new_expanded))

    def test_parse_to_measurelist(self, score_object):
        fname = score_object.fnames['mscx']
        fpath = os.path.join(score_object.paths['mscx'], '..')
        old_path = os.path.join(fpath, 'measures', fname + '.tsv')
        old_measurelist = load_tsv(old_path, index_col=None)
        new_measurelist = score_object.mscx.measures
        # Exclude 'repeat' column because the old parser used startRepeat, endRepeat and newSection
        # Exclude 'offset' and 'next' because the new parser does them more correctly
        excl = ['repeats', 'offset', 'next']
        new_measurelist.next = new_measurelist.next.map(lambda l: ', '.join(str(s) for s in l))
        new_measurelist.to_csv(fname + '_measures.tsv', sep='\t', index=False)
        assert_dfs_equal(old_measurelist, new_measurelist, exclude=excl)

    def test_parse_to_notelist(self, score_object):
        fname = score_object.fnames['mscx']
        fpath = os.path.join(score_object.paths['mscx'], '..')
        old_path = os.path.join(fpath, 'notes', fname + '.tsv')
        old_notelist = load_tsv(old_path, index_col=None)
        new_notelist = score_object.mscx.notes
        # Exclude 'onset' because the new parser computes 'offset' (measure list) more correctly
        excl = ['onset']
        new_notelist.to_csv(fname + '_notes.tsv', sep='\t', index=False)
        assert_dfs_equal(old_notelist, new_notelist, exclude=excl)

    def test_parse_to_eventlist(self, score_object):
        fname = score_object.fnames['mscx']
        score_object.mscx.events.to_csv(fname + '_events.tsv', sep='\t', index=False)




