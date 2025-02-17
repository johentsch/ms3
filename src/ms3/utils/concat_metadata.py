#!/usr/bin/env python
# coding: utf-8
import argparse
import os

from ms3.cli import check_and_create, check_dir
from ms3.utils import concat_metadata


def run(args):
    """Unpack the arguments and run the main function."""
    concat_metadata(
        meta_corpus_dir=args.dir,
        out=args.out,
    )


################################################################################
#                           COMMANDLINE INTERFACE
################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
-------------------------------------------------------------------
| Script for generating metadata and README for meta repositories |
-------------------------------------------------------------------

""",
    )
    parser.add_argument(
        "-d",
        "--dir",
        metavar="DIR",
        type=check_dir,
        help="Pass the root of the repository clone to gather metadata.tsv files from its child directories. "
        "Defaults to current working directory.",
    )
    parser.add_argument(
        "-o",
        "--out",
        metavar="OUT_DIR",
        type=check_and_create,
        help="""Output directory for TSV and MD file. Defaults to current working directory.""",
    )
    args = parser.parse_args()
    if args.dir is None:
        args.dir = os.getcwd()
    if args.out is None:
        args.out = os.getcwd()
    run(args)
