import json
from zipfile import ZipFile

import pandas as pd
from ms3.cli import get_arg_parser, transform_cmd


def test_transform_unfolds_only_pieces_with_expanded_labels(tmp_path):
    (tmp_path / "harmonies").mkdir()
    (tmp_path / "measures").mkdir()
    output = tmp_path / "out"
    output.mkdir()
    (tmp_path / "metadata.tsv").write_text("piece\nwith_labels\nwithout_labels\n")
    (tmp_path / "measures" / "with_labels.measures.tsv").write_text(
        "mc\tmn\tquarterbeats\tquarterbeats_all_endings\tduration_qb\tact_dur\tdont_count\tnext\n"
        "1\t1\t0\t0\t4\t1\t\t2\n"
        "2\t2\t4\t4\t4\t1\t\t1, 3\n"
        "3\t3\t8\t8\t4\t1\t\t-1\n"
    )
    (tmp_path / "measures" / "without_labels.measures.tsv").write_text(
        "mc\tmn\tquarterbeats\tduration_qb\tact_dur\tdont_count\tnext\n"
        "1\t1\t0\t4\t1\t\t-1\n"
    )
    (tmp_path / "harmonies" / "with_labels.harmonies.tsv").write_text(
        "mc\tmn\tquarterbeats\tquarterbeats_all_endings\tduration_qb\tmc_onset\tlabel\tnumeral\n"
        "1\t1\t0\t0\t4\t0\tI\tI\n"
        "2\t2\t4\t4\t4\t0\tV\tV\n"
        "3\t3\t8\t8\t4\t0\tI\tI\n"
    )

    args = get_arg_parser().parse_args(
        ["transform", "-d", str(tmp_path), "-o", str(output), "-X", "-u", "--resources"]
    )
    transform_cmd(args)

    prefix = tmp_path.name
    descriptor = json.loads((output / f"{prefix}.expanded.resource.json").read_text())
    assert descriptor["path"] == f"{prefix}.zip"
    assert descriptor["innerpath"] == f"{prefix}.expanded.tsv"
    with ZipFile(output / f"{prefix}.zip") as archive:
        assert archive.namelist() == [f"{prefix}.expanded.tsv"]
        with archive.open(archive.namelist()[0]) as tsv:
            unfolded = pd.read_csv(tsv, sep="\t")
    assert unfolded.piece.unique().tolist() == ["with_labels"]
    assert unfolded.mc.tolist() == [1, 2, 1, 2, 3]
    assert unfolded.mc_playthrough.tolist() == [1, 2, 3, 4, 5]
    assert unfolded.columns.tolist().count("quarterbeats_all_endings") == 1
