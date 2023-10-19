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
    check_phrase_annotations,
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
        "K284-3_section_breaks.mscx",
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
        piece_name = score_object.fnames["mscx"] + ".measures.tsv"
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
            piece_name = score_object.fnames["mscx"] + ".labels.tsv"
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

    def test_expanded_labels(
        self,
        score_object,
        tmp_path,
    ):
        if score_object.mscx.has_annotations:
            piece_name = score_object.fnames["mscx"] + ".labels.tsv"
            target_path = os.path.join(self.test_results, piece_name)
            tmp_filepath = str(tmp_path / piece_name)
            target_labels = decode_harmonies(load_tsv(target_path))
            extracted_labels = no_collections_no_booleans(score_object.mscx.labels())
            extracted_labels.to_csv(tmp_filepath, sep="\t", index=False)
            new_labels = load_tsv(tmp_filepath)
            assert len(new_labels) > 0
            assert_dfs_equal(target_labels, new_labels)

    def test_mc_offset(self, score_object, target_measures_table):
        target_mc_offset = target_measures_table["mc_offset"]
        new_mc_offset = make_offset_col(target_measures_table, section_breaks="breaks")
        assert (target_mc_offset == new_mc_offset).all()

    def test_parse_to_measures_table(
        self, score_object, target_measures_table, tmp_path
    ):
        extracted_measurelist = no_collections_no_booleans(score_object.mscx.measures())

        tmp_file = tmp_path / (score_object.fnames["mscx"] + ".measures.tsv")
        extracted_measurelist.to_csv(tmp_file, sep="\t", index=False)
        new_measurelist = load_tsv(tmp_file)
        assert len(new_measurelist) > 0
        assert_dfs_equal(target_measures_table, new_measurelist)

    def test_parse_to_notelist(self, score_object):
        piece_name = score_object.fnames["mscx"] + ".notes.tsv"
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
        print(f"CREATING EXCERPTS IN {tmp_path}")
        for start, end in ((1, 3), (2, 2), (3, None)):
            score_object.mscx.store_excerpt(
                start_mc=start,
                end_mc=end,
                directory=str(tmp_path),
            )
        assert len(os.listdir(tmp_path)) == 3

    def test_phrase_excerpts(self, score_object, tmp_path):
        skip_if_no_annotation_labels(score_object)
        print(f"CREATING PHRASE EXCERPTS IN {tmp_path}")
        score_object.mscx.store_phrase_excerpts(
            directory=str(tmp_path),
        )
        if score_object.mscx.has_annotations:
            assert len(os.listdir(tmp_path)) > 0

    def test_random_excerpts(self, score_object, tmp_path):
        print(f"CREATING RANDOM EXCERPTS IN {tmp_path}")
        score_object.mscx.store_random_excerpts(
            n_excerpts=3,
            directory=str(tmp_path),
        )
        assert len(os.listdir(tmp_path)) == 3

    def test_storing_all_excerpts(self, score_object, tmp_path):
        print(f"CREATING RANDOM EXCERPTS IN {tmp_path}")
        last_mc = score_object.mscx.measures().mc.max()
        mn_length = last_mc - 2
        score_object.mscx.store_random_excerpts(
            mc_length=int(mn_length),
            directory=str(tmp_path),
        )
        assert len(os.listdir(tmp_path)) == 3

    def test_store_measures(self, score_object, tmp_path):
        print(f"CREATING PHRASE EXCERPTS IN {tmp_path}")
        score_object.mscx.store_measures(
            included_mcs=(1, 2),
            directory=str(tmp_path),
        )
        assert len(os.listdir(tmp_path)) == 1

    def test_within_phrase_excerpts(self, score_object, tmp_path):
        skip_if_no_annotation_labels(score_object)
        print(f"CREATING WITHIN PHRASE EXCERPTS IN {tmp_path}")
        score_object.mscx.store_within_phrase_excerpts(directory=str(tmp_path))
        assert len(os.listdir(tmp_path)) > 0

    def test_phrase_endings(self, score_object, tmp_path):
        skip_if_no_annotation_labels(score_object)
        print(f"CREATING PHRASE ENDING EXCERPTS IN {tmp_path}")
        score_object.mscx.store_phrase_endings(directory=str(tmp_path))
        assert len(os.listdir(tmp_path)) > 0


def skip_if_no_annotation_labels(score_object):
    dcml_labels = score_object.mscx.expanded(unfold=True)
    if dcml_labels is None or len(dcml_labels) == 0:
        pytest.skip("No labels to extract phrase endings from.")
    if not check_phrase_annotations(dcml_labels, "phraseend"):
        pytest.skip("Incongruent phrase annotations.")
