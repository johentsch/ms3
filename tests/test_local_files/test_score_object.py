#!/usr/bin/env python
"""Tests for `ms3` package."""

import os
import tempfile

import ms3
import pytest
from ms3.bs4_measures import MeasureList, make_offset_col
from ms3.utils import (
    assert_all_lines_equal,
    assert_dfs_equal,
    decode_harmonies,
    load_tsv,
    no_collections_no_booleans,
)


@pytest.fixture(
    params=[
        "Did03M-Son_regina-1762-Sarti.mscx",
        "D973deutscher01.mscx",
        "05_symph_fant.mscx",
        "BWV_0815.mscx",
        "K281-3.mscx",
        "76CASM34A33UM.mscx",
        "stabat_03_coloured.mscx",
    ],
    ids=["sarti", "schubert", "berlioz", "bach", "mozart", "monty", "pergolesi"],
)
def score_object(request):
    test_folder = os.path.dirname(os.path.realpath(__file__))
    mscx_path = os.path.join(test_folder, "MS3", request.param)
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
    test_results = os.path.join(test_folder, "test_results")

    @pytest.fixture()
    def measure_list_object(self, score_object):
        ml = MeasureList(
            score_object.mscx.parsed._measures,
            sections=True,
            secure=True,
            reset_index=True,
            logger_cfg=dict(score_object.logger_cfg),
        )
        ml.make_ml()
        return ml

    @pytest.fixture()
    def target_measures_table(self, score_object):
        piece_name = score_object.fnames["mscx"] + "_measures.tsv"
        target_path = os.path.join(self.test_results, piece_name)
        return load_tsv(target_path)

    def test_parse_and_write_back(self, score_object):
        original_mscx = score_object.full_paths["mscx"]
        try:
            tmp_file = tempfile.NamedTemporaryFile(
                mode="r",
                suffix=".mscx",
                dir=self.test_folder,
                encoding="utf-8",
                delete=False,
            )
            if score_object.mscx.has_annotations:
                score_object.detach_labels("labels")
                score_object.attach_labels("labels")
            score_object.store_score(tmp_file.name)
            original = open(original_mscx, encoding="utf-8").read()
            after_parsing = tmp_file.read()
            assert_all_lines_equal(
                original, after_parsing, original=original_mscx, tmp_file=tmp_file
            )
        finally:
            tmp_file.close()
            os.remove(tmp_file.name)

    def test_store_and_load_labels(self, score_object):
        if score_object.mscx.has_annotations:
            piece_name = score_object.fnames["mscx"] + "_labels.tsv"
            labels_path = os.path.join(self.test_results, piece_name)
            score_object.load_annotations(labels_path, key="tsv")
            score_object.detach_labels("labels")
            score_object.attach_labels("tsv")
            try:
                tmp_file = tempfile.NamedTemporaryFile(
                    mode="r",
                    suffix=".tsv",
                    dir=self.test_folder,
                    encoding="utf-8",
                    delete=False,
                )
                score_object.store_score(tmp_file.name)
                original_mscx = score_object.full_paths["mscx"]
                before = open(original_mscx, encoding="utf-8").read()
                after = tmp_file.read()
                assert_all_lines_equal(
                    before, after, original=original_mscx, tmp_file=tmp_file
                )
            finally:
                tmp_file.close()
                os.remove(tmp_file.name)

    def test_expanded_labels(self, score_object):
        if score_object.mscx.has_annotations:
            piece_name = score_object.fnames["mscx"] + "_labels.tsv"
            target_path = os.path.join(self.test_results, piece_name)
            target_labels = decode_harmonies(load_tsv(target_path))
            try:
                extracted_labels = no_collections_no_booleans(
                    score_object.mscx.labels()
                )
                with tempfile.NamedTemporaryFile(
                    mode="r+",
                    suffix=".tsv",
                    dir=self.test_folder,
                    encoding="utf-8",
                    delete=False,
                ) as tmp_file:
                    extracted_labels.to_csv(tmp_file, sep="\t", index=False)
                    new_path = tmp_file.name
                new_labels = load_tsv(new_path)
                assert len(new_labels) > 0
                assert_dfs_equal(target_labels, new_labels)
            finally:
                os.remove(tmp_file.name)

    def test_mc_offset(self, score_object, target_measures_table):
        target_mc_offset = target_measures_table["mc_offset"]
        new_mc_offset = make_offset_col(target_measures_table, section_breaks="breaks")
        assert (target_mc_offset == new_mc_offset).all()

    def test_parse_to_measures_table(
        self, score_object, target_measures_table, tmp_path
    ):
        extracted_measurelist = no_collections_no_booleans(score_object.mscx.measures())

        tmp_file = tmp_path / (score_object.fnames["mscx"] + "_measures.tsv")
        extracted_measurelist.to_csv(tmp_file, sep="\t", index=False)
        new_measurelist = load_tsv(tmp_file)
        assert len(new_measurelist) > 0
        assert_dfs_equal(target_measures_table, new_measurelist)

    def test_parse_to_notelist(self, score_object):
        piece_name = score_object.fnames["mscx"] + "_notes.tsv"
        target_path = os.path.join(self.test_results, piece_name)
        target_notelist = load_tsv(target_path)
        try:
            extracted_notelist = no_collections_no_booleans(score_object.mscx.notes())
            with tempfile.NamedTemporaryFile(
                mode="r+",
                suffix=".tsv",
                dir=self.test_folder,
                encoding="utf-8",
                delete=False,
            ) as tmp_file:
                extracted_notelist.to_csv(tmp_file, sep="\t", index=False)
                new_path = tmp_file.name
            new_notelist = load_tsv(new_path)
            assert len(new_notelist) > 0
            assert_dfs_equal(target_notelist, new_notelist)
        finally:
            os.remove(tmp_file.name)

    def test_excerpt(self, score_object, tmp_path):
        for start, end in ((1, 3), (2, 2), (3, None)):
            score_object.mscx.store_excerpt(
                start_mc=start,
                end_mc=end,
                directory=tmp_path,
            )
