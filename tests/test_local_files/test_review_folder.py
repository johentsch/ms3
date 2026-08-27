"""Regression tests for corpora that contain a 'reviewed' folder produced by a previous ``ms3 review`` run."""

import os
import shutil

import pytest
from ms3 import Corpus

SCORE_PATH = os.path.join(os.path.dirname(__file__), "MS3", "BWV_0815.mscx")
PIECE = "BWV_0815"


@pytest.fixture()
def corpus_path(tmp_path):
    """A corpus without metadata.tsv containing one score and the 'reviewed' copy of a previous run."""
    shutil.copy(SCORE_PATH, os.path.join(tmp_path, f"{PIECE}.mscx"))
    reviewed_dir = os.path.join(tmp_path, "reviewed")
    os.makedirs(reviewed_dir)
    shutil.copy(SCORE_PATH, os.path.join(reviewed_dir, f"{PIECE}_reviewed.mscx"))
    return str(tmp_path)


def test_review_copy_does_not_create_its_own_piece(corpus_path):
    """A score in the 'reviewed' folder is a copy of a regular score and must be registered with the latter's
    Piece rather than spawning a Piece of its own."""
    corpus = Corpus(corpus_path)
    assert corpus.pnames == [PIECE]
    assert set(corpus.ix2pname.values()) == {PIECE}
    assert corpus.ix2orphan_file == {}
    piece_obj = corpus.get_piece(PIECE)
    assert len(piece_obj.ix2file) == 2


def test_review_score_without_counterpart_creates_a_piece(tmp_path):
    """Review scores that do not correspond to any regular score remain the only representatives of their piece."""
    reviewed_dir = os.path.join(tmp_path, "reviewed")
    os.makedirs(reviewed_dir)
    shutil.copy(SCORE_PATH, os.path.join(reviewed_dir, f"{PIECE}_reviewed.mscx"))
    corpus = Corpus(str(tmp_path))
    assert corpus.pnames == [f"{PIECE}_reviewed"]


def test_storing_over_existing_review_copy(corpus_path):
    """Storing a parsed score to a path that is already registered with the corpus used to raise an
    AssertionError in Piece.add_parsed_score()."""
    corpus = Corpus(corpus_path, only_metadata_pieces=False)
    corpus.parse_scores()
    paths = corpus.store_parsed_scores(
        only_changed=False, folder="reviewed", suffix="_reviewed", overwrite=True
    )
    assert len(paths) == 1
    assert os.path.normpath(paths[0]) == os.path.normpath(
        os.path.join(corpus_path, "reviewed", f"{PIECE}_reviewed.mscx")
    )
    piece_obj = corpus.get_piece(PIECE)
    stored_file = corpus.get_file_from_path(paths[0])
    assert stored_file.ix in piece_obj.ix2parsed_score
