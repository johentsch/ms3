import multiprocessing
import os
import re
from copy import deepcopy

import pytest
from git import Repo
from ms3 import Parse, Score
from ms3.logger import WarningFilter, get_logger, iter_ms3_loggers
from ms3.utils import capture_parse_logs, scan_directory

CORPUS_DIR = "~/git/meta_repositories/"  # Directory holding your clone of DCMLab/unittest_metacorpus
UNITTEST_METACORPUS = os.path.join(
    os.path.expanduser(CORPUS_DIR), "unittest_metacorpus"
)
TEST_COMMIT = (
    "03b35dbe"  # commit of DCMLab/unittest_metacorpus for which the tests should pass
)
MS3_DIR = os.path.normpath(os.path.join(os.path.realpath(__file__), "..", ".."))
DOCS_DIR = os.path.join(MS3_DIR, "docs")
DOCS_EXAMPLES_DIR = os.path.join(DOCS_DIR, "examples")


def pytest_configure(config):
    """When running on several xdist workers, shrink the process pools that ms3 creates for parsing
    (mp.Pool(mp.cpu_count())) so that the workers together don't spawn n_workers * n_cores processes.
    """
    n_workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", 1))
    if n_workers > 1:
        n_processes = max(1, os.cpu_count() // n_workers)
        multiprocessing.cpu_count = lambda: n_processes


@pytest.fixture(scope="session")
def directory():
    """Compose the path for the test corpus."""
    path = UNITTEST_METACORPUS
    if not os.path.isdir(path):
        print(
            f"Directory does not exist: {path} Clone DCMLab/unittest_metacorpus, checkout ms3_tests branch, "
            f"and specify CORPUS_DIR above."
        )
    assert os.path.isdir(path)
    repo = Repo(path)
    commit = repo.commit("HEAD")
    sha = commit.hexsha[: len(TEST_COMMIT)]
    assert sha == TEST_COMMIT
    assert repo.git.diff() == ""
    return path


@pytest.fixture(scope="session")
def small_directory(directory):
    path = os.path.join(directory, "ravel_piano")
    return path


@pytest.fixture(
    scope="module",
    params=[
        "regex",
        "everything",
        # "regular_dirs_at_once",
        # "file_re_without_key",
        # "without_metadata",
        # "redundant",
        # "regular_dirs",
        # "chaotic_dirs",
        # "hidden_dirs",
        # "files_without_key",
        # "files_with_inferred_key",
        # "files_with_wrong_key",
        # "files_correct_without_metadata",
        # "files_with_correct_key",
    ],
)
def parse_obj(directory, request) -> Parse:
    logger = get_logger("ms3.tests")
    if request.param == "regex":
        return Parse(directory=directory, file_re="WWV", folder_re="MS3")
    if request.param == "everything":
        return Parse(directory=directory).all
    if request.param == "file_re_without_key":
        p = Parse(directory=directory, file_re="SwWV")
        return p
    if request.param == "without_metadata":
        add_path = os.path.join(directory, "mixed_files", "orchestral")
        return Parse(add_path)
    if request.param == "redundant":
        add_path = os.path.join(directory, "mixed_files", "keyboard", "classic")
        return Parse(add_path)
    if request.param == "regular_dirs_at_once":
        os.chdir(directory)
        regular_dirs = ["ravel_piano", "sweelinck_keyboard", "wagner_overtures"]
        return Parse(regular_dirs)
    p = Parse()
    if request.param == "regular_dirs":
        for subdir in ["ravel_piano", "sweelinck_keyboard", "wagner_overtures"]:
            add_path = os.path.join(directory, subdir)
            p.add_dir(add_path)
    if request.param == "chaotic_dirs":
        for subdir in ["mixed_files", "outputs"]:
            add_path = os.path.join(directory, subdir)
            p.add_dir(add_path)
    if request.param == "hidden_dirs":
        for subdir in [".git", ".github"]:
            add_path = os.path.join(directory, subdir)
            p.add_dir(add_path)
    if request.param.startswith("files_"):
        add_path = os.path.join(directory, "sweelinck_keyboard")
        files = list(scan_directory(add_path, logger=logger))
        files_with_inferrable_metadata = [
            f for f in files if os.path.basename(f) != "metadata.tsv"
        ]
        files_without_inferrable_metadata = list(
            scan_directory(
                os.path.join(directory, "mixed_files", "orchestral"), logger=logger
            )
        )
        if request.param == "files_without_key":
            p.add_files(files_without_inferrable_metadata)
        if request.param == "files_with_inferred_key":
            p.add_files(files_with_inferrable_metadata)
        if request.param == "files_with_wrong_key":
            p.add_files(files_with_inferrable_metadata)
            p.add_files(files_without_inferrable_metadata)
        if request.param == "files_correct_without_metadata":
            key = "frankenstein"
            p.add_files(files_with_inferrable_metadata, corpus_name=key)
            for path in scan_directory(
                os.path.join(directory, "outputs"), logger=logger
            ):
                p.add_files(path, corpus_name=key)
        if request.param == "files_with_correct_key":
            p.add_dir(os.path.join(directory, "outputs"))
            for path in files:
                p.add_files(path, corpus_name="sweelinck_keyboard")
    return p


@pytest.fixture(
    scope="module",
    params=[
        0,
        2,
    ],
    ids=[
        "parsed_tsv",
        "parsed_all",
    ],
)
def parsed_parse_obj(parse_obj, request) -> Parse:
    """Parsing only the scores is not tested separately because 'parsed_all' covers it."""
    p = deepcopy(parse_obj)
    if request.param == 0:
        p.parse_tsv()
    elif request.param == 2:
        p.parse()
    else:
        assert False
    return p


@pytest.fixture(scope="class")
def parse_objects(parse_obj: Parse, request):
    request.cls.parse_obj = parse_obj


@pytest.fixture(scope="class")
def parsed_parse_objects(parsed_parse_obj, request):
    request.cls.parsed_parse_obj = parsed_parse_obj


# ## Creating path tuples for score_object():
# for folder, subdirs, files in os.walk('.'):
#     subdirs[:] = [s for s in subdirs if not s.startswith('.')]
#     fldrs = tuple(['mixed_files'] + folder.split('/')[1:])
#     for f in files:
#         if f.endswith('.mscx'):
#             print(f"{fldrs + (f,)},")


_SCORE_OBJECT_PATHS = [
    ("monty[tremolo]", ("mixed_files", "76CASM34A33UM")),
    ("pergolesi[form]", ("mixed_files", "stabat_03_coloured")),
    ("berlioz[tremolo]", ("mixed_files", "orchestral", "05_symph_fant")),
    ("sarti[endings]", ("mixed_files", "orchestral", "Did03M-Son_regina-1762-Sarti")),
    ("caldara[form]", ("mixed_files", "orchestral", "caldara_form")),
    ("bach[endings]", ("mixed_files", "keyboard", "baroque", "BWV_0815")),
    (
        "frescobaldi",
        (
            "mixed_files",
            "keyboard",
            "ancient",
            "12.16_Toccata_cromaticha_per_l’elevatione_phrygian",
        ),
    ),
    (
        "schubert[endings][tremolo]",
        ("mixed_files", "keyboard", "nineteenth", "D973deutscher01"),
    ),
    ("mozart", ("mixed_files", "keyboard", "classic", "K281-3")),
]
_SCORE_OBJECT_VARIANTS = [("ms3", ".mscx"), ("ms4", ".mscz")]


def _score_object_param(parts, ext):
    return parts[:-1] + (parts[-1] + ext,)


def score_params(paths_and_ids):
    """Turns (path, id) pairs into fixture params pinned to one xdist worker per score, so that the module-scoped
    Score object is parsed only once."""
    # xdist ignores group names containing ']' because they would be confused with parameter IDs
    return [
        pytest.param(
            path,
            id=id,
            marks=pytest.mark.xdist_group("score-" + re.sub(r"[\[\]]", "_", id)),
        )
        for path, id in paths_and_ids
    ]


_score_object_params = score_params(
    (_score_object_param(parts, ext), f"{nick}-{ver}")
    for nick, parts in _SCORE_OBJECT_PATHS
    for ver, ext in _SCORE_OBJECT_VARIANTS
)


@pytest.fixture(scope="module", params=_score_object_params)
def score_path(directory, request):
    return os.path.join(directory, *request.param)


@pytest.fixture(scope="module")
def score_object(score_path):
    """Parsed once per module and shared between tests. Tests that modify the score need to create their own
    Score(score_path)."""
    return Score(score_path)


def is_ms4_score(score_object):
    """True for Score objects backed by an MS4 source file."""
    return score_object.mscx.parsed.version.startswith("4")


@pytest.fixture(scope="session")
def get_all_warnings(directory):
    p = Parse(directory)
    with capture_parse_logs(p.logger) as captured_warnings:
        p.parse()
        _ = p.extract_facets("expanded")
    return captured_warnings.content_list


@pytest.fixture(scope="session")
def parse_with_all_warnings_ignored(directory, get_all_warnings) -> dict:
    """Parses the whole corpus with all warnings ignored and returns the captured messages
    {'warnings': [<messages that were not ignored>], 'ignored': [<ignored messages without header line>]}.

    The WarningFilters remain attached to the (global) loggers, which is why this fixture depends on
    get_all_warnings (which would otherwise capture nothing) and removes them afterwards.
    """
    ignored_warnings_file = os.path.join(
        directory, "mixed_files", "ALL_WARNINGS_IGNORED"
    )
    p = Parse(directory, level="d")
    p.load_ignored_warnings(ignored_warnings_file)
    with capture_parse_logs(p.logger, level="d") as captured_msgs, capture_parse_logs(
        p.logger, level="w"
    ) as captured_warnings:
        p.parse()
        _ = p.get_dataframes(expanded=True)
    for _, logger in iter_ms3_loggers():
        for warning_filter in [
            f for f in logger.filters if isinstance(f, WarningFilter)
        ]:
            logger.removeFilter(warning_filter)
    ignored = [
        "\n".join(msg.split("\n\t")[1:])
        for msg in captured_msgs.content_list
        if msg.startswith("IGNORED")
    ]
    return dict(warnings=captured_warnings.content_list, ignored=ignored)
