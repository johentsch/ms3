#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script contains functions for the mere purpose of triggering a particular action to debug it in you favourite
debugger. Feel free to add functions and to hardcode paths to your system since this is an auxiliary file where,
the moment something is considered, it is considered obsolete.
"""
import os.path
from argparse import Namespace

from ms3 import Parse, Score
from ms3.cli import review_cmd
from ms3.logger import get_logger
from ms3.operations import transform_to_resources

CORPUS_PATH = "~/git/389_chorale_settings/original_complete/"


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
        file_re="B378",
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
    path = "/home/laser/git/389_chorale_settings/original_complete/MS3/B378.mscx"
    return Score(path)


if __name__ == "__main__":
    args = Namespace(
        action="review",
        ignore_scores=False,
        ignore_labels=False,
        fail=True,
        ignore_metronome=False,
        ask=False,
        use="expanded",
        flip=False,
        safe=True,
        force=False,
        measures="../measures",
        notes="../notes",
        rests=None,
        labels=None,
        expanded="../harmonies",
        form_labels="../form_labels",
        events=None,
        chords="../chords",
        joined_chords=None,
        metadata="",
        positioning=False,
        raw=True,
        unfold=False,
        interval_index=False,
        corpuswise=False,
        dir="/home/laser/git/389_chorale_settings",
        out=None,
        nonrecursive=False,
        all=False,
        include=None,
        exclude=None,
        folders=None,
        musescore=None,
        reviewed=False,
        files=["/home/laser/git/389_chorale_settings/original_complete/MS3/B378.mscx"],
        iterative=False,
        level="i",
        log=None,
        test=False,
        verbose=False,
        compare="LATEST_VERSION",
        threshold=0.6,
        # func=review_cmd
    )
    review_cmd(args)
