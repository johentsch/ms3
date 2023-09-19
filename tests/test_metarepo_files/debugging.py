#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script contains functions for the mere purpose of triggering a particular action to debug it in you favourite
debugger. Feel free to add functions and to hardcode paths to your system since this is an auxiliary file where,
the moment something is considered, it is considered obsolete.
"""
import os.path

from ms3 import Parse
from ms3.cli import get_arg_parser, review_cmd
from ms3.logger import get_logger
from ms3.operations import transform_to_resources
from ms3.utils import resolve_dir

CORPUS_PATH = "~/all_subcorpora/wagner_overtures"


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
    """Created by executing an ms3 command and coping the object initializing from the output."""
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
    p = parse_object()
    p.parse_scores()
    _ = p.get_facet("expanded")
    # p.store_extracted_facets(
    #     root_dir=root_dir,
    #     notes_folder=notes_folder,
    #     rests_folder=rests_folder,
    #     notes_and_rests_folder=notes_and_rests_folder,
    #     measures_folder=measures_folder,
    #     events_folder=events_folder,
    #     labels_folder=labels_folder,
    #     chords_folder=chords_folder,
    #     expanded_folder=expanded_folder,
    #     cadences_folder=cadences_folder,
    #     form_labels_folder=form_labels_folder,
    #     metadata_suffix=metadata_suffix,
    #     markdown=markdown,
    #     simulate=simulate,
    #     unfold=unfold,
    #     interval_index=interval_index,
    # )
    return


def execute_review_cmd():
    parser = get_arg_parser()
    args = parser.parse_args(["review", "-d", resolve_dir(CORPUS_PATH)])
    review_cmd(args)


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


if __name__ == "__main__":
    execute_review_cmd()
