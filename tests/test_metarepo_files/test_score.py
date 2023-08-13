#!/usr/bin/env python
"""Tests for `ms3` package."""
import os
from typing import Optional

import ms3
import pytest
from ms3 import Score, assert_dfs_equal


def assert_store_scores_identical(
    score_object: Score,
    tmp_dir: Optional[str] = None,
    suffix: str = "_tmp",
):
    original_mscx = score_object.full_paths["mscx"]
    original_path = score_object.paths["mscx"]
    original_piece_name = score_object.fnames["mscx"]
    piece_name = f"{original_piece_name}{suffix}.mscx"
    if tmp_dir is None:
        tmp_dir = original_path
    tmp_filepath = os.path.join(tmp_dir, piece_name)
    score_object.store_score(tmp_filepath)
    tmp_lines = open(tmp_filepath, encoding="utf-8").readlines()
    expected_lines = open(original_mscx, encoding="utf-8").readlines()
    try:
        assert tmp_lines == expected_lines
    except AssertionError:
        if sorted(tmp_lines) == sorted(expected_lines):
            print(
                f"The lines in {tmp_filepath} come in a different order than in {original_path}."
            )
        else:
            raise


class TestBasic:
    def test_init(self):
        s = ms3.Score()
        assert isinstance(s, ms3.score.Score)
        with pytest.raises(LookupError):
            s.mscx


class TestScore:
    test_folder = os.path.dirname(os.path.realpath(__file__))

    def test_store_scores(self, score_object, tmp_path):
        assert_store_scores_identical(
            score_object, tmp_dir=str(tmp_path), suffix="_rewrite"
        )

    def test_removing_and_reinserting_labels(self, score_object, tmp_path):
        if not score_object.mscx.has_annotations:
            return
        before = score_object.get_labels()
        assert len(before) == len(
            score_object.mscx.parsed.soup.find_all("Harmony")
        ), "Not all <Harmony> tags appear in _.annotations.df !"
        score_object.detach_labels("labels")
        after = score_object.get_labels("labels")
        assert_dfs_equal(before, after)
        score_object.attach_labels("labels")
        assert_store_scores_identical(
            score_object, tmp_dir=str(tmp_path), suffix="_label_reinsertion"
        )

    def test_measures(self, score_object):
        mscx = score_object.mscx.parsed
        raw_measures = mscx.ml()
        effective_measures = score_object.mscx.measures()
        for col in ("quarterbeats", "duration_qb"):
            assert col in effective_measures
            assert col not in raw_measures
            assert effective_measures[col].notna().any()
        if raw_measures.volta.notna().any():
            assert "quarterbeats_all_endings" in effective_measures
            assert "quarterbeats_all_endings" not in raw_measures
            assert effective_measures.quarterbeats.isna().any()
        else:
            assert effective_measures.quarterbeats.notna().all()
        assert_dfs_equal(raw_measures, effective_measures)
        print(score_object.mscx.offset_dict())

    def test_notes(self, score_object):
        mscx = score_object.mscx.parsed
        notelist = mscx.notes()
        print(notelist)

    def test_form_labels(self, score_object):
        form = score_object.mscx.form_labels()
        if form is not None:
            for col in ("quarterbeats", "duration_qb"):
                print(form)
                assert (col in form) or (("", col) in form)

    # def test_partially_removing_and_reinserting_labels(self, directory):
    #     path = os.path.join(directory, "mixed_files", "stabat_03_coloured.mscx")
    #     sc_obj = Score(path)
    #     # TODO
