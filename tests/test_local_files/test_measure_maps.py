"""Tests for measure maps (MeasureMap format) as stored by ``ms3 extract -MM``."""

import json
import os
import shutil

import pytest
from ms3 import Corpus, Score
from ms3.cli import extract_cmd, get_arg_parser
from ms3.transformations import MEASURE_MAP_FIELDS, measures2measure_map
from ms3.utils import load_tsv

MS3_DIR = os.path.join(os.path.dirname(__file__), "MS3")
PIECE = "D973deutscher01"  # small score with a first and a second ending
SCORE_PATH = os.path.join(MS3_DIR, f"{PIECE}.mscx")


@pytest.fixture(scope="module")
def score():
    return Score(SCORE_PATH, level="c")


@pytest.fixture(scope="module")
def measure_map(score):
    return measures2measure_map(score.mscx.measures())


@pytest.fixture()
def corpus_path(tmp_path):
    """A corpus without metadata.tsv containing a copy of the score in the subfolder MS3."""
    score_dir = tmp_path / "corpus" / "MS3"
    score_dir.mkdir(parents=True)
    shutil.copy(SCORE_PATH, score_dir / f"{PIECE}.mscx")
    return tmp_path / "corpus"


def check_measure_map_entries(entries):
    """Checks the content of D973deutscher01's measure map (as list of dicts)."""
    assert len(entries) == 21
    assert all(tuple(entry.keys()) == MEASURE_MAP_FIELDS for entry in entries)
    assert [entry["count"] for entry in entries] == list(range(1, 22))
    assert [entry["ID"] for entry in entries] == [str(i) for i in range(1, 22)]
    assert all(
        isinstance(entry["qstamp"], float)
        and isinstance(entry["nominal_length"], float)
        and isinstance(entry["actual_length"], float)
        and isinstance(entry["start_repeat"], bool)
        and isinstance(entry["end_repeat"], bool)
        and isinstance(entry["next"], list)
        and all(isinstance(mc, int) for mc in entry["next"])
        for entry in entries
    )
    # qstamps include all endings: each one is the sum of the preceding actual lengths
    qstamp = 0.0
    for entry in entries:
        assert entry["qstamp"] == qstamp
        qstamp += entry["actual_length"]
    by_count = {entry["count"]: entry for entry in entries}
    assert by_count[1]["number"] == 0  # anacrusis
    assert by_count[1]["actual_length"] == 1.25
    assert all(entry["time_signature"] == "3/4" for entry in entries)
    assert all(entry["nominal_length"] == 3.0 for entry in entries)
    # names: measures in the first and second ending get letters
    assert [by_count[mc]["name"] for mc in (18, 19, 20, 21)] == [
        "16",
        "17a",
        "17b",
        "18b",
    ]
    # repeats
    assert [entry["count"] for entry in entries if entry["start_repeat"]] == [10]
    assert [entry["count"] for entry in entries if entry["end_repeat"]] == [9, 19]
    # next
    assert by_count[1]["next"] == [2]
    assert by_count[9]["next"] == [1, 10]
    assert by_count[18]["next"] == [19, 20]
    assert by_count[19]["next"] == [10]
    assert by_count[21]["next"] == [-1]


def test_measures2measure_map(measure_map):
    assert tuple(measure_map.columns) == MEASURE_MAP_FIELDS
    entries = measure_map.to_dict(orient="records")
    check_measure_map_entries(entries)
    # JSON serialisable without loss
    assert json.loads(json.dumps(entries)) == entries


def test_measures2measure_map_without_voltas():
    score = Score(os.path.join(MS3_DIR, "stabat_03_coloured.mscx"), level="c")
    measures = score.mscx.measures()
    assert "volta" not in measures.columns
    entries = measures2measure_map(measures).to_dict(orient="records")
    json.dumps(entries)
    assert [entry["name"] for entry in entries] == [str(mn) for mn in measures.mn]
    assert entries[-1]["next"] == [-1]


def test_measures2measure_map_rejects_unfolded_measures(score):
    with pytest.raises(ValueError, match="unfolded"):
        measures2measure_map(score.mscx.measures(unfold=True))


def test_extract_cli_stores_measure_maps(corpus_path):
    """-MM writes <piece>.mm.json into ../measure_maps, separate from the measures TSVs, and measure maps
    are not affected by -u."""
    args = get_arg_parser().parse_args(
        ["extract", "-d", str(corpus_path), "-a", "-M", "-MM", "-u", "--iterative"]
    )
    extract_cmd(args)
    mm_path = corpus_path / "measure_maps" / f"{PIECE}.mm.json"
    assert mm_path.is_file()
    assert sorted(os.listdir(corpus_path / "measure_maps")) == [f"{PIECE}.mm.json"]
    assert (corpus_path / "measures" / f"{PIECE}_unfolded.measures.tsv").is_file()
    check_measure_map_entries(json.loads(mm_path.read_text()))


def test_store_extracted_facets_returns_measure_map_paths(corpus_path, tmp_path):
    corpus = Corpus(str(corpus_path), only_metadata_pieces=False)
    corpus.parse_scores(parallel=False)
    expected_path = os.path.join(str(corpus_path), "mm", f"{PIECE}.mm.json")

    paths = corpus.store_extracted_facets(measure_maps_folder="../mm", simulate=True)
    assert paths == [expected_path]
    assert not os.path.exists(expected_path)

    out_dir = tmp_path / "output"
    paths = corpus.store_extracted_facets(
        root_dir=str(out_dir), measure_maps_folder="mm", unfold=True
    )
    expected_path = os.path.join(str(out_dir), "mm", f"{PIECE}.mm.json")
    assert paths == [expected_path]
    with open(expected_path) as f:
        check_measure_map_entries(json.load(f))


def test_measure_map_from_measures_tsv(corpus_path, measure_map):
    """A measures TSV stored by ms3 and loaded again yields the same measure map as the parsed score."""
    corpus = Corpus(str(corpus_path), only_metadata_pieces=False)
    corpus.parse_scores(parallel=False)
    corpus.store_extracted_facets(measures_folder="../measures", frictionless=False)
    tsv_path = corpus_path / "measures" / f"{PIECE}.measures.tsv"
    measures = load_tsv(str(tsv_path))
    from_tsv = measures2measure_map(measures).to_dict(orient="records")
    assert from_tsv == measure_map.to_dict(orient="records")
