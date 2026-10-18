"""Tests for the standalone DCML => Dezrann converter ms3/dezrann.py.

The fixtures in ``dezrann/`` are copies of the harmonies and measures TSV files of two movements from the DCML corpus
https://github.com/DCMLab/mozart_piano_sonatas (CC BY-NC-SA 4.0): K282-1 (without voltas) and K283-2 (with first and
second endings).
"""

import ast
import json
import logging
import os
import shutil
import subprocess
import sys
from collections import Counter

import pandas as pd
import pytest
from ms3 import dezrann

FIXTURES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dezrann")
HARMONIES_DIR = os.path.join(FIXTURES, "harmonies")
MEASURES_DIR = os.path.join(FIXTURES, "measures")
PIECES = {"K282-1": False, "K283-2": True}  # piece => has voltas
END_OF_SCORE = {
    "K282-1": 144.0,
    "K283-2": 164.0,
}  # measuring-stick length in quarter notes
SETTINGS = dict(cadences=True, harmonies=4, keys=5, phrases=6, raw=1)
EXPECTED_LAYOUT = [
    {"filter": {"type": "Cadence"}, "style": {"line": "all"}},
    {"filter": {"type": "Harmony"}, "style": {"line": "bot.1"}},
    {"filter": {"type": "Local Key"}, "style": {"line": "bot.2"}},
    {"filter": {"type": "Phrase"}, "style": {"line": "bot.3"}},
    {"filter": {"type": "Raw label"}, "style": {"line": "top.1"}},
]
SEGMENT_TYPES = ("Harmony", "Local Key", "Phrase", "Raw label")


def harmonies_path(piece: str) -> str:
    return os.path.join(HARMONIES_DIR, f"{piece}.harmonies.tsv")


def measures_path(piece: str) -> str:
    return os.path.join(MEASURES_DIR, f"{piece}.measures.tsv")


def expected_counts(harmonies: pd.DataFrame) -> dict:
    localkey = harmonies["localkey"]
    return {
        "Harmony": harmonies["chord"].notna().sum(),
        "Cadence": harmonies["cadence"].notna().sum(),
        "Phrase": harmonies["phraseend"]
        .fillna("")
        .str.contains("{", regex=False)
        .sum(),
        "Local Key": (localkey != localkey.shift()).sum(),
        "Raw label": harmonies["label"].notna().sum(),
    }


def labels_by_type(dez: dict) -> dict:
    result = {}
    for label in dez["labels"]:
        result.setdefault(label["type"], []).append(label)
    return result


@pytest.fixture(params=list(PIECES))
def piece(request) -> str:
    return request.param


@pytest.fixture()
def dez(piece, tmp_path) -> dict:
    output_path = tmp_path / f"{piece}.dez"
    assert dezrann.generate_dez(
        path_measures=measures_path(piece),
        path_labels=harmonies_path(piece),
        output_path=str(output_path),
        **SETTINGS,
    )
    with open(output_path, encoding="utf-8") as f:
        return json.load(f)


def test_fixtures_have_expected_voltas(piece):
    measures = dezrann.load_tsv(measures_path(piece))
    assert dezrann.has_voltas(measures) == PIECES[piece]


def test_label_counts(piece, dez):
    """The fixtures have labels at the beginning of every ending, so no labels are repeated."""
    harmonies = pd.read_csv(harmonies_path(piece), sep="\t")
    written = dict(Counter(label["type"] for label in dez["labels"]))
    assert written == expected_counts(harmonies)


def test_durations(piece, dez):
    end_of_score = END_OF_SCORE[piece]
    for label_type, labels in labels_by_type(dez).items():
        assert all(label["duration"] >= 0 for label in labels), label_type
        starts = [label["start"] for label in labels]
        assert starts == sorted(starts), label_type
        if label_type == "Cadence":
            assert all(label["duration"] == 0 for label in labels)
            continue
        # segments are contiguous and end with the score
        for previous, following in zip(labels, labels[1:]):
            assert previous["start"] + previous["duration"] == pytest.approx(
                following["start"]
            )
        last = labels[-1]
        assert last["start"] + last["duration"] == pytest.approx(end_of_score)
        assert sum(label["duration"] for label in labels) == pytest.approx(
            end_of_score - labels[0]["start"]
        )


def test_layout(dez):
    assert dez["meta"]["layout"] == EXPECTED_LAYOUT
    valid_lines = set(dezrann.LINE_VALUES.values()) | {"all"}
    assert all(rule["style"]["line"] in valid_lines for rule in dez["meta"]["layout"])


def test_labels_in_first_endings(dez, piece):
    """Labels in first endings have no 'quarterbeats' value and are positioned on the measuring-stick timeline."""
    if not PIECES[piece]:
        return
    harmonies = pd.read_csv(harmonies_path(piece), sep="\t")
    in_first_endings = harmonies[harmonies.quarterbeats.isna()]
    assert len(in_first_endings) > 0
    harmony_starts = [label["start"] for label in labels_by_type(dez)["Harmony"]]
    assert sorted(in_first_endings.quarterbeats_all_endings.astype(float)) == [
        start
        for start in harmony_starts
        if start in set(in_first_endings.quarterbeats_all_endings.astype(float))
    ]


def test_repeat_label_at_second_ending(tmp_path):
    """If a second ending does not begin with a label, the label active before the first ending is repeated there,
    except for phrase beginnings and cadences."""
    measures = dezrann.load_tsv(measures_path("K283-2"))
    harmonies = dezrann.load_tsv(harmonies_path("K283-2"))
    second_ending_mc = 15
    second_ending_qb = 56.0
    first_ending_qb = 52.0
    is_second_ending_beginning = (harmonies.mc == second_ending_mc) & (
        harmonies.mc_onset == 0
    )
    assert is_second_ending_beginning.sum() == 1
    harmonies = harmonies[~is_second_ending_beginning]
    before_group = harmonies[
        harmonies.quarterbeats_all_endings.map(float) < first_ending_qb
    ].iloc[-1]
    output_path = str(tmp_path / "K283-2.dez")
    assert dezrann.generate_dez_from_dfs(measures, harmonies, output_path, **SETTINGS)
    with open(output_path, encoding="utf-8") as f:
        labels = labels_by_type(json.load(f))
    harmony = [
        label for label in labels["Harmony"] if label["start"] == second_ending_qb
    ]
    assert [label["tag"] for label in harmony] == [before_group.chord]
    raw = [label for label in labels["Raw label"] if label["start"] == second_ending_qb]
    assert len(raw) == 1
    assert raw[0]["tag"] == dezrann._strip_cadence_and_phrase(before_group.label)
    assert (
        "|" not in raw[0]["tag"]
        and "{" not in raw[0]["tag"]
        and "}" not in raw[0]["tag"]
    )
    # the first ending's last harmony ends where the second ending begins
    in_first_ending = [
        label
        for label in labels["Harmony"]
        if first_ending_qb <= label["start"] < second_ending_qb
    ]
    assert (
        in_first_ending[-1]["start"] + in_first_ending[-1]["duration"]
        == second_ending_qb
    )
    for label_type in ("Phrase", "Cadence"):
        assert not any(
            label["start"] == second_ending_qb for label in labels[label_type]
        )
    keys = [
        label for label in labels["Local Key"] if label["start"] == second_ending_qb
    ]
    assert [label["tag"] for label in keys] == [before_group.localkey]


def read_dez(path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_positions_computed_from_mc_onset(piece, dez, tmp_path):
    """Without quarterbeats columns in the labels and without 'quarterbeats_all_endings' in the measures, positions
    are computed from mc + mc_onset and the measure durations, yielding the same result.
    """
    measures = dezrann.load_tsv(measures_path(piece)).drop(
        columns=["quarterbeats", "quarterbeats_all_endings"], errors="ignore"
    )
    harmonies = dezrann.load_tsv(harmonies_path(piece)).drop(
        columns=["quarterbeats", "quarterbeats_all_endings"]
    )
    output_path = tmp_path / "computed.dez"
    assert dezrann.generate_dez_from_dfs(
        measures, harmonies, str(output_path), **SETTINGS
    )
    assert json.loads(read_dez(output_path)) == dez


def test_dataframes_loaded_by_ms3(piece, dez, tmp_path):
    """DataFrames loaded by ms3 (with Fractions and pd.NA) yield the same result as the TSV files."""
    from ms3 import load_tsv

    output_path = tmp_path / "from_dfs.dez"
    assert dezrann.generate_dez_from_dfs(
        load_tsv(measures_path(piece)),
        load_tsv(harmonies_path(piece)),
        str(output_path),
        **SETTINGS,
    )
    assert json.loads(read_dez(output_path)) == dez


@pytest.mark.parametrize(
    "argument, expected",
    [
        (None, None),
        (0, None),
        ("0", None),
        (1, "top.1"),
        ("3", "top.3"),
        (4, "bot.1"),
        (-1, "bot.1"),
        ("-3", "bot.3"),
        ("bot.2", "bot.2"),
        ("top.1", "top.1"),
    ],
)
def test_transform_line_argument(argument, expected):
    assert dezrann.transform_line_argument(argument) == expected
    # idempotent
    assert (
        dezrann.transform_line_argument(dezrann.transform_line_argument(argument))
        == expected
    )


@pytest.mark.parametrize("argument", [7, -4, "bot.4", "x", 1.5])
def test_invalid_line_argument(argument):
    with pytest.raises(ValueError):
        dezrann.transform_line_argument(argument)


def test_same_line_twice():
    with pytest.raises(ValueError, match="more than once"):
        dezrann.resolve_line_arguments(harmonies=4, keys="bot.1")


def test_layout_values_are_translated():
    layout = dezrann.make_layout(harmonies=4, keys=-2, phrases="bot.3", raw=0)
    assert layout == [
        {"filter": {"type": "Harmony"}, "style": {"line": "bot.1"}},
        {"filter": {"type": "Local Key"}, "style": {"line": "bot.2"}},
        {"filter": {"type": "Phrase"}, "style": {"line": "bot.3"}},
    ]


def test_get_volta_groups():
    mc2volta = pd.Series(
        [pd.NA, 1, 1, 2, 2, pd.NA, 1, 2, 3, 1, 2],
        index=range(1, 12),
        dtype="Int64",
    )
    assert dezrann.get_volta_groups(mc2volta) == [[2, 4], [7, 8, 9], [10, 11]]


def test_split_tsv_name():
    assert dezrann.split_tsv_name("K283-2.harmonies.tsv") == ("K283-2", "harmonies")
    assert dezrann.split_tsv_name("K283-2.tsv") == ("K283-2", None)
    assert dezrann.split_tsv_name("op.27.measures.tsv") == ("op.27", "measures")
    assert dezrann.split_tsv_name("op.27.tsv") == ("op.27", None)


def test_cli_with_default_arguments(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    exit_code = dezrann.run([HARMONIES_DIR, "-o", str(output_dir)])
    assert exit_code == 0
    assert sorted(os.listdir(output_dir)) == ["K282-1.dez", "K283-2.dez"]
    with open(output_dir / "K283-2.dez", encoding="utf-8") as f:
        dez = json.load(f)
    assert {label["type"] for label in dez["labels"]} == {
        "Harmony",
        "Local Key",
        "Phrase",
    }
    assert dez["meta"]["layout"] == EXPECTED_LAYOUT[1:4]


def test_main_with_unsuffixed_file_names(tmp_path):
    """Older DCML corpora have harmonies/<piece>.tsv and measures/<piece>.tsv. Output defaults to the input dir."""
    for folder in ("harmonies", "measures"):
        (tmp_path / folder).mkdir()
    shutil.copy(harmonies_path("K283-2"), tmp_path / "harmonies" / "K283-2.tsv")
    shutil.copy(measures_path("K283-2"), tmp_path / "measures" / "K283-2.tsv")
    shutil.copy(
        harmonies_path("K282-1"), tmp_path / "harmonies" / "K282-1.tsv"
    )  # no measures => skipped
    summary = dezrann.main(
        input_dir=str(tmp_path / "harmonies"),
        measures_dir=str(tmp_path / "measures"),
        output_dir=str(tmp_path / "harmonies"),
        harmonies=4,
    )
    assert summary == dezrann.ConversionSummary(written=1, skipped=1, failed=0)
    assert os.path.isfile(tmp_path / "harmonies" / "K283-2.dez")


def test_main_reports_failures(tmp_path, caplog):
    # set the level explicitly because, depending on which tests ran before on the same worker, ms3 may have set
    # pytest's capturing handler (which pytest attaches also to ms3's non-propagating loggers) to WARNING
    caplog.set_level(logging.INFO, logger=dezrann.logger.name)
    for folder in ("harmonies", "measures"):
        (tmp_path / folder).mkdir()
    shutil.copy(measures_path("K283-2"), tmp_path / "measures")
    (tmp_path / "harmonies" / "K283-2.harmonies.tsv").write_text(
        "mc\tchord\nnot_a_number\tI\n"
    )
    exit_code = dezrann.run([str(tmp_path / "harmonies")])
    assert exit_code == 1
    assert not os.path.isfile(tmp_path / "harmonies" / "K283-2.dez")
    assert "successfully written" not in caplog.text
    assert "1 failed" in caplog.text


def test_script_has_no_ms3_imports():
    with open(dezrann.__file__, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import"
            imported.add(node.module.split(".")[0])
    stdlib = set(sys.stdlib_module_names)
    assert imported - stdlib == {"pandas"}


def test_standalone_script(tmp_path):
    """The script runs from a copy outside the package, in isolated mode, without importing ms3."""
    script = tmp_path / "dezrann.py"
    shutil.copy(dezrann.__file__, script)
    for folder in ("harmonies", "measures"):
        shutil.copytree(os.path.join(FIXTURES, folder), tmp_path / folder)
    check_ms3_not_imported = (
        "import runpy, sys; sys.argv = ['dezrann.py', 'harmonies', '-C']; "
        "sys.modules['ms3'] = None; "  # any attempt to import ms3 raises ImportError
        "\ntry:\n    runpy.run_path('dezrann.py', run_name='__main__')\n"
        "except SystemExit as e:\n    sys.exit(e.code)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", check_ms3_not_imported],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Created 2 .dez files" in result.stderr
    assert sorted(
        f for f in os.listdir(tmp_path / "harmonies") if f.endswith(".dez")
    ) == [
        "K282-1.dez",
        "K283-2.dez",
    ]
