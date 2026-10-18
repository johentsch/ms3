import logging
import os
import sys

import pytest
from ms3 import Corpus
from ms3.utils import functions


@pytest.fixture
def without_gitpython(monkeypatch):
    """Makes ``import git`` raise an ImportError."""
    monkeypatch.setitem(sys.modules, "git", None)
    functions._warn_about_missing_gitpython.cache_clear()


def test_get_git_repo_warns_inside_git_work_tree(without_gitpython, tmp_path, caplog):
    (tmp_path / ".git").mkdir()
    logger = logging.getLogger("test_optional_git")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        assert functions.get_git_repo(tmp_path / "subdir", logger=logger) is None
    assert "pip install ms3[git]" in caplog.text


def test_get_git_repo_silent_outside_git_work_tree(without_gitpython, tmp_path, caplog):
    logger = logging.getLogger("test_optional_git")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        assert (
            functions.get_git_repo(
                tmp_path, search_parent_directories=False, logger=logger
            )
            is None
        )
    assert caplog.text == ""


def test_git_revision_requires_gitpython(without_gitpython, tmp_path):
    with pytest.raises(ImportError, match=r"pip install ms3\[git\]"):
        functions.get_git_commit(str(tmp_path), "HEAD")


def test_corpus_without_gitpython(without_gitpython):
    corpus = Corpus(os.path.dirname(__file__))
    assert corpus.repo is None
    assert corpus.get_version_info() == {}
