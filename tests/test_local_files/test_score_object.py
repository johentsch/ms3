#!/usr/bin/env python
"""Tests for `ms3` package."""

import pytest
import os
import tempfile

import ms3
from ms3.utils import assert_all_lines_equal, assert_dfs_equal, decode_harmonies, load_tsv, write_tsv, no_collections_no_booleans


@pytest.fixture(
    params=['Did03M-Son_regina-1762-Sarti.mscx', 'D973deutscher01.mscx', '05_symph_fant.mscx', 'BWV_0815.mscx', 'K281-3.mscx', '76CASM34A33UM.mscx', 'stabat_03_coloured.mscx'],
ids=['sarti', "schubert", "berlioz", 'bach', 'mozart', 'monty', 'pergolesi'])
def score_object(request):
    test_folder = os.path.dirname(os.path.realpath(__file__))
    mscx_path = os.path.join(test_folder, 'MS3', request.param)
    s = ms3.Score(mscx_path)
    return s

class TestBasic:

    def test_init(self):
        s = ms3.Score()
        assert isinstance(s, ms3.score.Score)
        with pytest.raises(LookupError):
            s.mscx


class TestScore:

    test_folder = os.path.dirname(os.path.realpath(__file__))
    test_results = os.path.join(test_folder, 'test_results')

    def test_parse_and_write_back(self, score_object):
        original_mscx = score_object.full_paths['mscx']
        try:
            tmp_file = tempfile.NamedTemporaryFile(mode='r', suffix='.mscx', dir=self.test_folder, encoding='utf-8', delete=False)
            if score_object.mscx.has_annotations:
                score_object.detach_labels('labels')
                score_object.attach_labels('labels')
            score_object.store_score(tmp_file.name)
            original = open(original_mscx, encoding='utf-8').read()
            after_parsing = tmp_file.read()
            assert_all_lines_equal(original, after_parsing, original=original_mscx, tmp_file=tmp_file)
        finally:
            tmp_file.close()
            os.remove(tmp_file.name)



    def test_store_and_load_labels(self, score_object):
        if score_object.mscx.has_annotations:
            fname = score_object.fnames['mscx'] + '_labels.tsv'
            labels_path = os.path.join(self.test_results, fname)
            score_object.load_annotations(labels_path, key='tsv')
            score_object.detach_labels('labels')
            score_object.attach_labels('tsv')
            try:
                tmp_file = tempfile.NamedTemporaryFile(mode='r', suffix='.tsv', dir=self.test_folder, encoding='utf-8', delete=False)
                score_object.store_score(tmp_file.name)
                original_mscx = score_object.full_paths['mscx']
                before = open(original_mscx, encoding='utf-8').read()
                after = tmp_file.read()
                assert_all_lines_equal(before, after, original=original_mscx, tmp_file=tmp_file)
            finally:
                tmp_file.close()
                os.remove(tmp_file.name)

    def test_expanded_labels(self, score_object):
        if score_object.mscx.has_annotations:
            fname = score_object.fnames['mscx'] + '_labels.tsv'
            old_path = os.path.join(self.test_results, fname)
            old_labels = decode_harmonies(load_tsv(old_path))
            try:
                extracted_labels = no_collections_no_booleans(score_object.mscx.labels())
                with tempfile.NamedTemporaryFile(mode='r+', suffix='.tsv', dir=self.test_folder, encoding='utf-8', delete=False) as tmp_file:
                    extracted_labels.to_csv(tmp_file, sep='\t', index=False)
                    new_path = tmp_file.name
                new_labels = load_tsv(new_path)
                assert len(new_labels) > 0
                assert_dfs_equal(old_labels, new_labels)
            finally:
                os.remove(tmp_file.name)

    def test_parse_to_measurelist(self, score_object):
        fname = score_object.fnames['mscx'] + '_measures.tsv'
        old_path = os.path.join(self.test_results, fname)
        old_measurelist = load_tsv(old_path)
        try:
            extracted_measurelist = no_collections_no_booleans(score_object.mscx.measures())
            with tempfile.NamedTemporaryFile(mode='r+', suffix='.tsv', dir=self.test_folder, encoding='utf-8', delete=False) as tmp_file:
                extracted_measurelist.to_csv(tmp_file, sep='\t', index=False)
                new_path = tmp_file.name
            new_measurelist = load_tsv(new_path)
            assert len(new_measurelist) > 0
            assert_dfs_equal(old_measurelist, new_measurelist)
        finally:
            os.remove(tmp_file.name)

    def test_parse_to_notelist(self, score_object):
        fname = score_object.fnames['mscx'] + '_notes.tsv'
        old_path = os.path.join(self.test_results, fname)
        old_notelist = load_tsv(old_path)
        try:
            extracted_notelist = no_collections_no_booleans(score_object.mscx.notes())
            with tempfile.NamedTemporaryFile(mode='r+', suffix='.tsv', dir=self.test_folder, encoding='utf-8', delete=False) as tmp_file:
                extracted_notelist.to_csv(tmp_file, sep='\t', index=False)
                new_path = tmp_file.name
            new_notelist = load_tsv(new_path)
            assert len(new_notelist) > 0
            assert_dfs_equal(old_notelist, new_notelist)
        finally:
            os.remove(tmp_file.name)




