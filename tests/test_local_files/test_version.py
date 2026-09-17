from importlib.metadata import version
from pathlib import Path

import pytest
from ms3 import Score
from ms3.cli import get_arg_parser


def test_cli_and_score_metadata_use_installed_version(capsys):
    installed_version = version("ms3")
    with pytest.raises(SystemExit) as exit_info:
        get_arg_parser().parse_args(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == installed_version

    score_path = Path(__file__).parent / "MS3" / "D973deutscher01.mscx"
    score = Score(str(score_path), parser="bs4")
    assert score.mscx.metadata["ms3_version"] == installed_version
