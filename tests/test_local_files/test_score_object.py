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

_SCORE_STEMS = [
    ("sarti", "Did03M-Son_regina-1762-Sarti"),
    ("schubert", "D973deutscher01"),
    ("berlioz", "05_symph_fant"),
    ("bach", "BWV_0815"),
    ("mozart", "K284-3_section_breaks"),
    ("monty", "76CASM34A33UM"),
    ("pergolesi", "stabat_03_coloured"),
]
_SCORE_VARIANTS = [("MS3", ".mscx"), ("MS4", ".mscz")]

_score_params = [
    pytest.param(
        (subdir, stem + ext),
        id=f"{nick}-{subdir.lower()}",
        # all tests using the same score run on the same xdist worker so that it is parsed only once
        marks=pytest.mark.xdist_group(f"local-{nick}-{subdir.lower()}"),
    )
    for nick, stem in _SCORE_STEMS
    for subdir, ext in _SCORE_VARIANTS
]


def _select_params(*ids):
    """Selects a subset of the score params (keeping their xdist_group so that each score is parsed only once)."""
    selected = [param for param in _score_params if param.id in ids]
    assert len(selected) == len(ids), f"Unknown score IDs among {ids}"
    return selected


# Excerpts are a minor feature and expensive to test (each excerpt is a complete MSCX file), so the excerpt tests
# only use 3 pieces in MS3 format and 3 other pieces in MS4 format.
_EXCERPT_PARAMS = _select_params(
    "pergolesi-ms3",
    "schubert-ms3",
    "sarti-ms3",
    "mozart-ms4",
    "bach-ms4",
    "monty-ms4",
)
# the only two of the above that come with phrase annotations
_PHRASE_PARAMS = _select_params("pergolesi-ms3", "mozart-ms4")

MAX_STORED_EXCERPTS = 2
"""Phrase-based excerpt methods create one excerpt per phrase, window etc. (up to several hundred per score). The
tests compute all of them but only store the first few."""


@pytest.fixture(scope="module", params=_score_params)
def score_path(request):
    test_folder = os.path.dirname(os.path.realpath(__file__))
    subdir, filename = request.param
    return os.path.join(test_folder, subdir, filename)


@pytest.fixture(scope="module")
def parsed_scores() -> dict:
    """{path -> Score} cache, needed because the excerpt tests parametrize score_path separately, which makes pytest
    re-create the score_object fixture."""
    return {}


@pytest.fixture(scope="module")
def score_object(score_path, parsed_scores):
    """Parsed once per module and shared between tests. Tests that modify the score need to create their own
    ms3.Score(score_path)."""
    if score_path not in parsed_scores:
        parsed_scores[score_path] = ms3.Score(score_path)
    return parsed_scores[score_path]


@pytest.fixture()
def stored_excerpts(score_object, monkeypatch):
    """Lets MSCX.store_measures(), which all excerpt methods use, store only the first MAX_STORED_EXCERPTS excerpts
    and returns the list of all calls' kwargs."""
    mscx = score_object.mscx
    store_measures = mscx.store_measures
    calls = []

    def capped_store_measures(**kwargs):
        calls.append(kwargs)
        if len(calls) <= MAX_STORED_EXCERPTS:
            store_measures(**kwargs)

    monkeypatch.setattr(mscx, "store_measures", capped_store_measures)
    return calls


def _skip_if_ms4(score_object):
    """Round-trip tests compare the on-disk source against ms3's serialization.
    Since ms3 only writes MS3-format XML, MS4 sources cannot round-trip."""
    if score_object.mscx.parsed.version.startswith("4"):
        pytest.skip("Round-trip to source format unsupported for MS4 input.")


def _piece_name(score_object):
    """Return the bare piece stem, regardless of source format (.mscx vs .mscz)."""
    for key in ("mscx", "mscz"):
        if key in score_object.fnames:
            return score_object.fnames[key]
    raise KeyError(f"Score has no MuseScore source: {score_object.fnames}")


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
        piece_name = _piece_name(score_object) + ".measures.tsv"
        target_path = os.path.join(self.test_results, piece_name)
        return load_tsv(target_path)

    def test_parse_and_write_back(self, score_object, score_path):
        _skip_if_ms4(score_object)
        # this test modifies the score, so it must not use the shared one
        score_object = ms3.Score(score_path)
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

    def test_store_and_load_labels(self, score_object, score_path):
        _skip_if_ms4(score_object)
        if score_object.mscx.has_annotations:
            # this test modifies the score, so it must not use the shared one
            score_object = ms3.Score(score_path)
            piece_name = _piece_name(score_object) + ".labels.tsv"
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
            piece_name = _piece_name(score_object) + ".labels.tsv"
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

        tmp_file = tmp_path / (_piece_name(score_object) + ".measures.tsv")
        extracted_measurelist.to_csv(tmp_file, sep="\t", index=False)
        new_measurelist = load_tsv(tmp_file)
        assert len(new_measurelist) > 0
        assert_dfs_equal(target_measures_table, new_measurelist)

    def test_parse_to_notelist(self, score_object):
        piece_name = _piece_name(score_object) + ".notes.tsv"
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


@pytest.mark.parametrize("score_path", _EXCERPT_PARAMS, indirect=True, scope="module")
class TestExcerpts:
    def test_excerpt(self, score_object, tmp_path):
        for start, end in ((1, 3), (2, 2), (3, None)):
            score_object.mscx.store_excerpt(
                start_mc=start,
                end_mc=end,
                directory=str(tmp_path),
            )
        assert len(os.listdir(tmp_path)) == 3

    def test_store_measures(self, score_object, tmp_path):
        score_object.mscx.store_measures(
            included_mcs=(1, 2),
            directory=str(tmp_path),
        )
        assert len(os.listdir(tmp_path)) == 1

    def test_random_excerpts(self, score_object, tmp_path):
        score_object.mscx.store_random_excerpts(
            n_excerpts=2,
            directory=str(tmp_path),
        )
        assert len(os.listdir(tmp_path)) == 2


@pytest.mark.parametrize("score_path", _PHRASE_PARAMS, indirect=True, scope="module")
class TestPhraseExcerpts:
    @pytest.fixture(autouse=True)
    def assert_phrase_annotations(self, score_object):
        dcml_labels = score_object.mscx.expanded(unfold=True)
        assert check_phrase_annotations(dcml_labels, "phraseend")

    def test_phrase_excerpts(self, score_object, stored_excerpts, tmp_path):
        score_object.mscx.store_phrase_excerpts(
            directory=str(tmp_path),
        )
        n_phrases = len(score_object.mscx.get_phrase_boundaries())
        assert len(stored_excerpts) == n_phrases > 0
        assert len(os.listdir(tmp_path)) == min(n_phrases, MAX_STORED_EXCERPTS)

    def test_within_phrase_excerpts(self, score_object, stored_excerpts, tmp_path):
        score_object.mscx.store_within_phrase_excerpts(directory=str(tmp_path))
        assert len(stored_excerpts) > 0
        assert len(os.listdir(tmp_path)) == min(
            len(stored_excerpts), MAX_STORED_EXCERPTS
        )

    def test_phrase_endings(self, score_object, stored_excerpts, tmp_path):
        score_object.mscx.store_phrase_endings(directory=str(tmp_path))
        assert len(stored_excerpts) > 0
        assert len(os.listdir(tmp_path)) == min(
            len(stored_excerpts), MAX_STORED_EXCERPTS
        )


@pytest.mark.parametrize(
    "score_path", _select_params("schubert-ms3"), indirect=True, scope="module"
)
def test_storing_all_excerpts(score_object, tmp_path):
    """Without n_excerpts, all possible excerpts (one per window of 2 MCs) are stored. Tested on the shortest score."""
    score_object.mscx.store_random_excerpts(directory=str(tmp_path))
    last_mc = score_object.mscx.measures().mc.max()
    assert len(os.listdir(tmp_path)) == last_mc - 1
