"""Extracted facets and metadata must not depend on the Python process that produced them.

Iterating over a set of strings yields a different order in every process because of hash randomization, which used
to reorder spanner columns (e.g. ``crescendo_hairpin`` vs. ``pedal``) and the items of ``composer_text`` & co.
between runs, causing meaningless diffs (and new schema identifiers) in corpus repositories.
"""

import os
import subprocess
import sys

import pandas as pd
import pytest
from ms3.utils.functions import prepare_metadata_for_writing

MS3_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "MS3")
SCORES = ["stabat_03_coloured.mscx", "05_symph_fant.mscx", "76CASM34A33UM.mscx"]

SNIPPET = """
import sys, warnings
warnings.simplefilter("ignore")
import ms3
score = ms3.Score(sys.argv[1], level="c")
print("chords columns:", list(score.mscx.chords().columns))
print("metadata:", sorted((k, str(v)) for k, v in score.mscx.metadata.items() if k.endswith("_text")))
"""


def run_with_hash_seed(path: str, seed: int) -> str:
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    result = subprocess.run(
        [sys.executable, "-c", SNIPPET, path],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


@pytest.mark.parametrize("score", SCORES)
def test_output_independent_of_hash_seed(score):
    path = os.path.join(MS3_DIR, score)
    outputs = {run_with_hash_seed(path, seed) for seed in (1, 2, 3, 4)}
    assert len(outputs) == 1, f"Output differs between processes:\n{outputs}"


def test_metadata_subdirectory_written_with_forward_slashes(monkeypatch):
    monkeypatch.setattr(os, "sep", "\\")
    metadata = pd.DataFrame(
        {
            "subdirectory": ["bach_solo\\MS3"],
            "rel_path": ["bach_solo\\MS3\\BWV1007_01_Prelude.mscx"],
        },
        index=pd.Index(["BWV1007_01_Prelude"], name="piece"),
    )
    result = prepare_metadata_for_writing(metadata).set_index("piece")
    assert result.loc["BWV1007_01_Prelude", "subdirectory"] == "bach_solo/MS3"
    assert (
        result.loc["BWV1007_01_Prelude", "rel_path"]
        == "bach_solo/MS3/BWV1007_01_Prelude.mscx"
    )
