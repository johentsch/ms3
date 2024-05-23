import os
from copy import deepcopy

import pytest
from git import Repo
from ms3 import Parse, Score
from ms3.logger import get_logger
from ms3.utils import capture_parse_logs, ignored_warnings2dict, scan_directory

CORPUS_DIR = "~/git/meta_repositories/"  # Directory holding your clone of DCMLab/unittest_metacorpus
UNITTEST_METACORPUS = os.path.join(
    os.path.expanduser(CORPUS_DIR), "unittest_metacorpus"
)
TEST_COMMIT = (
    "aeebac1"  # commit of DCMLab/unittest_metacorpus for which the tests should pass
)
MS3_DIR = os.path.normpath(os.path.join(os.path.realpath(__file__), "..", ".."))
DOCS_DIR = os.path.join(MS3_DIR, "docs")
DOCS_EXAMPLES_DIR = os.path.join(DOCS_DIR, "examples")


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
    scope="session",
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
    scope="session",
    params=[
        0,
        1,
        2,
    ],
    ids=[
        "parsed_tsv",
        "parse_scores",
        "parsed_all",
    ],
)
def parsed_parse_obj(parse_obj, request) -> Parse:
    p = deepcopy(parse_obj)
    if request.param == 0:
        p.parse_tsv()
    elif request.param == 1:
        p.parse_scores()
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


@pytest.fixture(
    params=[
        ("mixed_files", "76CASM34A33UM.mscx"),
        ("mixed_files", "stabat_03_coloured.mscx"),
        ("mixed_files", "orchestral", "05_symph_fant.mscx"),
        ("mixed_files", "orchestral", "Did03M-Son_regina-1762-Sarti.mscx"),
        ("mixed_files", "orchestral", "caldara_form.mscx"),
        ("mixed_files", "keyboard", "baroque", "BWV_0815.mscx"),
        (
            "mixed_files",
            "keyboard",
            "ancient",
            "12.16_Toccata_cromaticha_per_l’elevatione_phrygian.mscx",
        ),
        ("mixed_files", "keyboard", "nineteenth", "D973deutscher01.mscx"),
        ("mixed_files", "keyboard", "classic", "K281-3.mscx"),
    ],
    ids=[
        "monty[tremolo]",
        "pergolesi[form]",
        "berlioz[tremolo]",
        "sarti[endings]",
        "caldara[form]",
        "bach[endings]",
        "frescobaldi",
        "schubert[endings][tremolo]",
        "mozart",
    ],
)
def score_object(directory, request):
    mscx_path = os.path.join(directory, *request.param)
    s = Score(mscx_path)
    return s


@pytest.fixture(scope="session")
def get_all_warnings(directory):
    p = Parse(directory)
    with capture_parse_logs(p.logger) as captured_warnings:
        p.parse()
        _ = p.extract_facets("expanded")
    return captured_warnings.content_list


@pytest.fixture(scope="session")
def get_all_warnings_parsed(get_all_warnings):
    return ignored_warnings2dict(get_all_warnings)


@pytest.fixture(scope="session")
def get_all_supressed_warnings(directory):
    ignored_warnings_file = os.path.join(
        directory, "mixed_files", "ALL_WARNINGS_IGNORED"
    )
    p = Parse(directory, level="d")
    p.load_ignored_warnings(ignored_warnings_file)
    with capture_parse_logs(p.logger, level="d") as captured_msgs:
        p.parse()
        _ = p.get_dataframes(expanded=True)
        all_msgs = captured_msgs.content_list
    return [
        "\n".join(msg.split("\n\t")[1:])
        for msg in all_msgs
        if msg.startswith("IGNORED")
    ]
