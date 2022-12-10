import os
from git import Repo
from ms3.cli import get_arg_parser, review_cmd, extract_cmd, compare_cmd


def test_review_cmd(directory):
    parser = get_arg_parser()
    args = parser.parse_args(["review", "-d", directory])
    review_cmd(args)

def test_compare_cmd():
    parser = get_arg_parser()
    directory = os.path.expanduser("~/debussy_suite_bergamasque")
    args = parser.parse_args(["compare", "-d", directory])
    compare_cmd(args)
    repo = Repo(directory)
    new_files = repo.untracked_files
    repo.git.clean('-fdx')
    assert len(new_files) == 0

# def test_review_cmd_with_git():
#     parser = get_arg_parser()
#     ABC = os.path.expanduser("~/ABC")
#     args = parser.parse_args(["review", "-d", ABC, "-M", "-N", "-X", "-c=HEAD", "-r=n04op18-4_03"])
#     review_cmd(args)

def test_extract_cmd(directory):
    parser = get_arg_parser()
    args = parser.parse_args(["extract", "-d", directory, "-M", "-N", "-X"])
    extract_cmd(args)
    repo = Repo(directory)
    new_files = repo.untracked_files
    if len(new_files) > 0:
        repo.git.clean('-fdx')
        print(f"ms3 extract added these files that had not been there before: {new_files}")
    modified_files = [item.a_path for item in repo.index.diff(None)]
    n_modified = len(modified_files)
    repo.git.submodule('foreach', 'git', 'reset', '--hard')
    if n_modified != 3:
        print(f"expected only the 3 CSV files to change during extraction but: {modified_files}")
    assert n_modified == 3
    assert len(new_files) == 0
