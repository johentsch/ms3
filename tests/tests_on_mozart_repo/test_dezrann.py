import json
import os
from collections import Counter

import pytest
from git import Repo
from ms3 import (
    Parse,
    assert_all_lines_equal,
    get_value_profile_mask,
    load_tsv,
    resolve_dir,
)
from ms3.dezrann import generate_dez, generate_dez_from_dfs

MOZART_PIANO_SONATAS = "~/all_subcorpora/mozart_piano_sonatas"


@pytest.fixture(scope="session")
def mozart_piano_sonatas() -> str:
    """Get the path to local clone of DCMLab/mozart_piano_sonatas"""
    path = resolve_dir(MOZART_PIANO_SONATAS)
    if not os.path.isdir(path):
        print(
            f"Directory does not exist: {path} Clone DCMLab/mozart_piano_sonatas into the CORPUS_DIR specified above."
        )
    assert os.path.isdir(path)
    repo = Repo(path)
    yield path
    repo.git.clean("-fdx")  # removes new files potentially generated during test


MOZART_MOVEMENTS = [
    "K279-1",
    "K279-2",
    "K279-3",
    "K280-1",
    "K280-2",
    "K280-3",
    "K283-1",
    "K283-2",
    "K283-3",
]
SETTINGS = dict(cadences=True, harmonies=4, keys=5, phrases=6)


@pytest.fixture(params=MOZART_MOVEMENTS)
def movement(request) -> str:
    return request.param


def test_dcml2dez(mozart_piano_sonatas, movement):
    """
    This test creates Dezrann files from DCML annotations and compares the number of written labels with the target.
    """
    # first, create .dez file
    measures_path = os.path.join(mozart_piano_sonatas, "measures", f"{movement}.tsv")
    harmonies_path = os.path.join(mozart_piano_sonatas, "harmonies", f"{movement}.tsv")
    out_path = os.path.join(mozart_piano_sonatas, f"{movement}.dez")
    generate_dez(
        path_measures=measures_path,
        path_labels=harmonies_path,
        output_path=out_path,
        **SETTINGS,
    )
    # then, count the contained labels and compare with the target number (except if score contains voltas because then,
    # the .dez file might contain additional, repeated labels at the beginning of each ending).
    expanded = load_tsv(harmonies_path)
    if "volta" in expanded and expanded.volta.notna().any():
        return
    with open(out_path, "r", encoding="utf-8") as f:
        dezrann_file = json.load(f)
    type2column = {
        "Harmony": "chord",
        "Cadence": "cadence",
        "Phrase": "phraseend",
        "Local Key": "localkey",
    }
    written_labels = dict(
        Counter(type2column[label["type"]] for label in dezrann_file["labels"])
    )
    expected_counts = dict(
        chord=expanded["chord"].notna().sum(),
        cadence=expanded["cadence"].notna().sum(),
        phraseend=expanded["phraseend"].str.contains("{").sum(),
        localkey=get_value_profile_mask(expanded["localkey"]).sum(),
    )
    assert written_labels == expected_counts


def test_parse2dez(mozart_piano_sonatas):
    """This test creates two .dez files per piece and checks if they are identical. One is created from the DataFrames
    as parsed by the ms3.Parse() object, and the other is created directly from the TSV files.
    """
    file_re = "|".join(MOZART_MOVEMENTS)
    p = Parse(mozart_piano_sonatas, file_re=file_re)
    p.view.include("facets", "measures", "expanded")
    p.view.fnames_with_incomplete_facets = False
    p.parse_tsv()
    facet_dataframes = p.get_facets(
        ["expanded", "measures"], concatenate=False, choose="auto"
    )
    for (corpus, fname), facet2file_df_pair in facet_dataframes.items():
        measures_file, measures_df = facet2file_df_pair["measures"][0]
        harmonies_file, harmonies_df = facet2file_df_pair["expanded"][0]
        output_from_tsv = os.path.join(mozart_piano_sonatas, f"{fname}_from_tsv.dez")
        output_from_dfs = os.path.join(mozart_piano_sonatas, f"{fname}_from_df.dez")
        generate_dez(
            path_measures=measures_file.full_path,
            path_labels=harmonies_file.full_path,
            output_path=output_from_tsv,
            **SETTINGS,
        )
        generate_dez_from_dfs(
            measures_df=measures_df,
            harmonies_df=harmonies_df,
            output_path=output_from_dfs,
            **SETTINGS,
        )
        dez_from_tsv = open(output_from_tsv, "r", encoding="utf-8").read()
        dez_from_dfs = open(output_from_dfs, "r", encoding="utf-8").read()
        assert_all_lines_equal(
            dez_from_tsv, dez_from_dfs, output_from_tsv, output_from_dfs
        )
