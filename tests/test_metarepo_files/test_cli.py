import os
import re
import shutil
from typing import Optional

import pytest
from git import Repo
from ms3.cli import compare_cmd, extract_cmd, get_arg_parser, review_cmd


def get_repo_changes(repo, name: Optional[str] = None) -> str:
    """Produce a git status-like summary of changes and  untracked files."""
    if name is None:
        result = "\n"
    else:
        result = "\n" + name + "\n" + len(name) * "=" + "\n"
    changes = repo.index.diff(None)
    no_changes = len(changes) == 0
    untracked = repo.untracked_files
    no_untracked = len(untracked) == 0
    if no_changes and no_untracked:
        result += "Clean."
        return result
    if no_changes:
        result += "DIFF: no changes\n"
    else:
        changes_str = "\n".join("\t" + str(change) for change in changes)
        result += f"DIFF:\n{changes_str}\n"
    if no_untracked:
        result += "UNTRACKED FILES: none"
    else:
        untracked_str = "\n".join("\t" + file for file in untracked)
        result += f"UNTRACKED FILES:\n{untracked_str}"
    return result


def assert_test_repo_unchanged(path_to_git_repo):
    repo = Repo(path_to_git_repo)
    is_clean = {}
    repo_name = os.path.basename(repo.git.rev_parse("--show-toplevel"))
    print(get_repo_changes(repo, repo_name))
    is_clean[repo_name] = not repo.is_dirty(untracked_files=True)
    for sm in repo.iter_submodules():
        smm = sm.module()
        print(get_repo_changes(smm, sm.name))
        is_clean[sm.name] = not smm.is_dirty(untracked_files=True)
    assert all(is_clean.values())


# ms3 converts .mscz files to temporary files named like '.k3j_x9a2.mscx' next to them
MS3_TMP_FILE = re.compile(r"^\.[a-z0-9_]{8}\.mscx$")


def ignore_ms3_tmp_files(_, names):
    return [name for name in names if MS3_TMP_FILE.match(name)]


@pytest.fixture()
def corpus_copy(directory, tmp_path):
    """A private copy of the metacorpus (including the git metadata of its submodules) so that the commands, which
    write into the corpus, neither change the clone nor disturb other tests reading it in parallel. Skips the
    temporary files that tests parsing .mscz files of the clone in parallel may have created.
    """
    path = os.path.join(tmp_path, os.path.basename(directory))
    shutil.copytree(directory, path, symlinks=True, ignore=ignore_ms3_tmp_files)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_review_cmd(corpus_copy):
    """Review (= check + extract + compare) parses serially, hence it is run on a single, small corpus."""
    parser = get_arg_parser()
    directory = os.path.join(corpus_copy, "wagner_overtures")
    args = parser.parse_args(["review", "-d", directory])
    review_cmd(args)
    # make sure the repo has not changed
    # currently deactivated because the .warning files change with every new ms3 version
    # assert_test_repo_unchanged(corpus_copy)


def test_compare_cmd(corpus_copy):
    parser = get_arg_parser()
    directory = os.path.join(corpus_copy, "sweelinck_keyboard")
    args = parser.parse_args(["compare", "-d", directory])
    compare_cmd(args)
    new_files = Repo(directory).untracked_files
    assert len(new_files) == 0


# def test_review_cmd_with_git():
#     parser = get_arg_parser()
#     ABC = os.path.expanduser("~/ABC")
#     args = parser.parse_args(["review", "-d", ABC, "-M", "-N", "-X", "-c=HEAD", "-r=n04op18-4_03"])
#     review_cmd(args)


def test_extract_cmd(corpus_copy):
    parser = get_arg_parser()
    args = parser.parse_args(["extract", "-d", corpus_copy, "-M", "-N", "-X"])
    extract_cmd(args)
    repo = Repo(corpus_copy)
    new_files = repo.untracked_files
    if len(new_files) > 0:
        print(
            f"ms3 extract added these files that had not been there before: {new_files}"
        )
    modified_files = [item.a_path for item in repo.index.diff(None)]
    n_modified = len(modified_files)
    if n_modified != 3:
        print(
            f"expected only the 3 CSV files to change during extraction but: {modified_files}"
        )
    assert n_modified == 3
    assert len(new_files) == 0
