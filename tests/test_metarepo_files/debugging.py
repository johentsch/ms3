#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script contains functions for the mere purpose of triggering a particular action to debug it in you favourite
debugger. Feel free to add functions and to hardcode paths to your system since this is an auxiliary file where,
the moment something is considered, it is considered obsolete.
"""
import os.path

from ms3 import Parse, Score
from ms3.logger import get_logger
from ms3.operations import transform_to_resources

CORPUS_PATH = "~/unittest_metacorpus/wagner_overtures"


def ignoring_warning():
    p = Parse("~/unittest_metacorpus/mixed_files")
    p.parse_scores()
    t = get_logger("ms3.Parse.mixed_files.Did03M-Son_regina-1762-Sarti.mscx")
    filt = t.filters[0]
    print("IGNORED_WARNINGS")
    print(filt.ignored_warnings)
    t.warning("This should be a DEBUG message.", extra={"message_id": (2, 94)})
    _ = p.get_dataframes(expanded=True)


def parse_object() -> Parse:
    p = Parse(
        CORPUS_PATH,
        recursive=True,
        only_metadata_pieces=True,
        include_convertible=False,
        exclude_review=True,
        file_re=None,
        folder_re=None,
        exclude_re=None,
        file_paths=None,
        labels_cfg={"positioning": False, "decode": True},
        ms=None,
        **{"level": "i", "path": None}
    )
    p.info()
    return p


def extraction():
    """Created by executing an ms3 command and coping the object initializing from the output."""
    p = parse_object()
    p.parse_scores()
    p.store_extracted_facets(expanded_folder="..")


def transform_cmd():
    p = parse_object()
    p.parse_tsv()
    output_folder, filename = os.path.split(CORPUS_PATH)
    transform_to_resources(
        ms3_object=p,
        facets="metadata",
        filename=filename,
        output_folder=output_folder,
    )


def single_score():
    path = "~/distant_listening_corpus/ABC/MS3/n13op130_06.mscx"
    return Score(path)


if __name__ == "__main__":
    score = single_score()
    score.mscx.store_excerpt(
        start_mc=62,
        start_mc_onset=0,
        end_mc=102,
        end_mc_onset=0,
        exclude_end=True,
        directory="/home/laser/Documents/phd/phrase_excerpts/231220_distant_listening_corpus",
        suffix="_phrase776",
    )
