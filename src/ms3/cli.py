#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command line interface for ms3. This module includes the argument parsers for the various commands and has one function
<command>_cmd() per <command> which takes the parsed arguments namespace and dispatches them to the appropriate function
in the ms3.operations module, using a couple of helper functions to that aim.
"""

import argparse
import logging
import os
import re
from collections import defaultdict
from typing import List, Literal, Optional, overload

from ms3 import (
    Parse,
    compute_path_from_file,
    convert_from_metadata_tsv,
    get_git_repo,
    get_git_version_info,
    make_coloring_reports_and_warnings,
)
from ms3._version import __version__
from ms3.logger import get_logger, inspect_loggers
from ms3.operations import (
    check,
    compare,
    extract,
    insert_labels_into_score,
    store_scores,
    transform_to_package,
    transform_to_resources,
    update,
)
from ms3.utils import (
    capture_parse_logs,
    convert_folder,
    resolve_dir,
    write_warnings_to_file,
)

__author__ = "johentsch"
__copyright__ = "École Polytechnique Fédérale de Lausanne"
__license__ = "gpl3"


def gather_extract_params(args) -> List[str]:
    params = [
        name
        for name, arg in zip(
            (
                "measures",
                "notes",
                "rests",
                "labels",
                "expanded",
                "form_labels",
                "events",
                "chords",
            ),
            (
                args.measures,
                args.notes,
                args.rests,
                args.labels,
                args.expanded,
                args.form_labels,
                args.events,
                args.chords,
            ),
        )
        if arg
    ]
    if args.metadata is not None and not (args.metadata == False):  # noqa
        params.append("metadata")
    return params


def add_cmd(args, parse_obj: Optional[Parse] = None) -> Parse:
    logger = get_logger("ms3.add", level=args.level)
    if parse_obj is None:
        p = make_parse_obj(args)
    else:
        p = parse_obj

    insert_labels_into_score(
        p,
        facet=args.use,
        ask_for_input=args.ask,
        replace=args.replace,
    )
    corpus2paths = store_scores(
        p, root_dir=args.out, folder=".", simulate=args.test, suffix=args.suffix
    )
    changed = sum(map(len, corpus2paths.values()))
    logger.info(
        f"Operation resulted in {changed} annotated score{'s' if changed != 1 else ''}."
    )


def check_cmd(args, parse_obj: Optional[Parse] = None) -> Parse:
    if parse_obj is None:
        p = make_parse_obj(args)
    else:
        p = parse_obj
    _ = check(
        p,
        ignore_labels=args.ignore_labels,
        ignore_scores=args.ignore_scores,
        assertion=args.fail,
        ignore_metronome=args.ignore_metronome,
    )
    return p


def compare_cmd(args, parse_obj=None, output: bool = True, logger=None) -> None:
    if logger is None:
        logger = get_logger("ms3.compare", level=args.level)
    if parse_obj is None:
        p = make_parse_obj(args)
    else:
        p = parse_obj
    revision = None if args.compare == "" else args.compare
    try:
        n_changed, n_unchanged = compare(
            p,
            facet=args.use,
            ask=args.ask,
            revision_specifier=revision,
            flip=args.flip,
            force=args.force,
            logger=p.logger,
        )
    except TypeError:
        logger.error(
            "No files to compare. Try adding the flag -a to parse files even if they are not included in a "
            "metdata.tsv file."
        )
        return
    logger.debug(
        f"{n_changed} files changed labels during comparison, {n_unchanged} didn't."
    )
    if output:
        corpus2paths = store_scores(
            p,
            only_changed=not args.force,
            root_dir=args.out,
            simulate=args.test,
            suffix=args.suffix,
        )
        changed = sum(map(len, corpus2paths.values()))
        logger.info(
            f"Operation resulted in {changed} comparison file{'s' if changed != 1 else ''}."
        )
    else:
        logger.info(
            f"Operation resulted in colored labels within {n_changed} score{'s' if n_changed != 1 else ''}."
        )


def convert_cmd(args):
    # assert target[:len(
    #    dir)] != dir, "TARGET_DIR cannot be identical with nor a subfolder of DIR.\nDIR:        " + dir +
    #    '\nTARGET_DIR: ' + target
    update_logger = get_logger("ms3.convert", level=args.level)
    for argument_name, argument, default in (
        ("-e/--exclude", args.exclude, None),
        ("-f/--folders", args.folders, None),
        ("--reviewed", args.reviewed, False),
        ("-t/--test", args.test, False),
        ("-v/--verbose", args.verbose, False),
    ):
        if argument != default:
            update_logger.info(
                f"Argument '{argument_name}' is currently being ignored."
            )
    out_dir = "." if args.out is None else args.out
    ms = "auto" if args.musescore is None else args.musescore
    if args.all:
        convert_folder(
            directory=args.dir,
            file_paths=args.files,
            target_dir=out_dir,
            extensions=args.extensions,
            target_extension=args.format,
            regex=args.include,
            suffix=args.suffix,
            recursive=not args.nonrecursive,
            ms=ms,
            overwrite=args.safe,
            parallel=not args.iterative,
            logger=update_logger,
        )
    else:
        convert_from_metadata_tsv(
            directory=args.dir,
            file_paths=args.files,
            target_dir=out_dir,
            extensions=args.extensions,
            target_extension=args.format,
            regex=args.include,
            suffix=args.suffix,
            recursive=not args.nonrecursive,
            ms=ms,
            overwrite=args.safe,
            parallel=not args.iterative,
            logger=update_logger,
        )


def empty(args, parse_obj: Optional[Parse] = None):
    if parse_obj is None:
        p = make_parse_obj(args, parse_scores=True)
    else:
        p = parse_obj
    p.detach_labels()
    p.store_parsed_scores(
        only_changed=True,
        root_dir=args.out,
        suffix=args.suffix,
        overwrite=True,
    )


def extract_cmd(args, parse_obj: Optional[Parse] = None):
    if parse_obj is None:
        p = make_parse_obj(args)
    else:
        p = parse_obj
    params = gather_extract_params(args)
    if len(params) == 0:
        print(
            "In order to extract DataFrames, pass at least one of the following arguments: -M (measures), -N (notes), "
            "-R (rests), -L (labels), -X (expanded), -F (form_labels), -E (events), -C (chords), -D (metadata)"
        )
        return
    extract(
        p,
        root_dir=args.out,
        notes_folder=args.notes,
        labels_folder=args.labels,
        measures_folder=args.measures,
        rests_folder=args.rests,
        events_folder=args.events,
        chords_folder=args.chords,
        expanded_folder=args.expanded,
        form_labels_folder=args.form_labels,
        metadata_suffix=args.metadata,
        simulate=args.test,
        parallel=not args.iterative,
        unfold=args.unfold,
        interval_index=args.interval_index,
        corpuswise=args.corpuswise,
    )


def metadata(args, parse_obj: Optional[Parse] = None):
    """Update MSCX files with changes made in metadata.tsv (created via ms3 extract -D). In particular,
    add the values from (new?) columns to the corresponding fields in the MuseScore files' "Score info".
    """
    if parse_obj is None:
        p = make_parse_obj(args)
    else:
        p = parse_obj
    print("PARSING SCORES ONE AFTER THE OTHER...")
    p.parse_scores(parallel=False)
    modified_score_files = p.update_score_metadata_from_tsv(
        write_empty_values=args.empty,
        remove_unused_fields=args.remove,
        write_text_fields=args.prelims,
        update_instrumentation=args.instrumentation,
    )
    if len(modified_score_files) == 0:
        print("Nothing to update.")
        return
    corpus2paths = store_scores(
        p,
        only_changed=True,
        root_dir=args.out,
        folder=".",
        simulate=args.test,
        suffix=args.suffix,
    )
    changed = sum(map(len, corpus2paths.values()))
    print(
        f"Operation resulted in {changed} updated score{'s' if changed != 1 else ''}."
    )


# def repair(args):
#     print("Sorry, the command has not been implemented yet.")
#     print(args.dir)


def transform_cmd(args):
    params = gather_extract_params(args)
    if len(params) == 0:
        print(
            "Pass at least one of the following arguments: -M (measures), -N (notes), -R (rests), -L (labels), "
            "-X (expanded), -F (form_labels), -E (events), -C (chords), -D (metadata)"
        )
        return

    repo = get_git_repo(args.dir)
    if repo is None:
        version_info = None
    elif repo.is_dirty():
        if args.dirty:
            version_info = None
        else:
            print(
                "The repository is dirty. Please commit or stash your changes before running ms3 transform. This is "
                "important because the version information in the JSON descriptor(s) needs to be consistent with the "
                "repository state. To ignore this warning, add the --dirty flag to the command."
            )
            return
    else:
        version_info = get_git_version_info(repo=repo)
    parse_obj = make_parse_obj(args, parse_tsv=True, facets=params)
    filename = os.path.basename(args.dir)
    func = transform_to_resources if args.resources else transform_to_package
    # noinspection PyTypeChecker
    func(
        ms3_object=parse_obj,
        facets=params,
        filename=filename,
        output_folder=args.out,
        choose="auto",
        interval_index=args.interval_index,
        unfold=args.unfold,
        test=args.test,
        zipped=not args.uncompressed,
        overwrite=args.safe,
        log_level=args.level,
        custom_metadata=version_info,
    )


def update_cmd(args, parse_obj: Optional[Parse] = None):
    if parse_obj is None:
        p = make_parse_obj(args, parse_scores=True)
    else:
        p = parse_obj
    changed_paths = update(
        p,
        root_dir=args.out,
        suffix="" if args.suffix is None else args.suffix,
        overwrite=True,
        staff=int(args.staff),
        harmony_layer=int(args.type),
        above=args.above,
        safe=args.safe,
    )
    if len(changed_paths) > 0:
        print(f"Paths of scores with updated labels:\n{changed_paths}")
    else:
        print("Nothing to do.")


def check_and_create(d, resolve=True):
    """Turn input into an existing, absolute directory path."""
    if not os.path.isdir(d):
        d_resolved = resolve_dir(os.path.join(os.getcwd(), d))
        if not os.path.isdir(d_resolved):
            if input(d_resolved + " does not exist. Create? (y|n)") == "y":
                os.mkdir(d_resolved)
            else:
                raise argparse.ArgumentTypeError(
                    d + " needs to be an existing directory"
                )
    return d_resolved if resolve else d


def check_and_create_unresolved(d):
    return check_and_create(d, resolve=False)


def check_dir(d):
    if not os.path.isdir(d):
        d = resolve_dir(os.path.join(os.getcwd(), d))
        if not os.path.isdir(d):
            raise argparse.ArgumentTypeError(d + " needs to be an existing directory")
    return resolve_dir(d)


def precommit_cmd(
    args,
    parse_obj: Optional[Parse] = None,
) -> bool:
    """Like ms3 review but also adds the resulting files to the Git index."""
    logger = get_logger("ms3.precommit", level=args.level)
    if args.files:
        raise NotImplementedError(
            "The --files argument should not be used for ms3 precommit which takes files as "
            "positional arguments."
        )
    args.files = args.positional_args  # in the future, maybe use args.include instead
    test_passes = review_cmd(args, parse_obj=parse_obj, wrapped_by_precommit=True)
    repo = get_git_repo(args.dir)
    if repo is not None:
        repo.git.add(all=True)
    deliver_test_result(test_passes, args.fail, logger)


@overload
def review_cmd(args, parse_obj, wrapped_by_precommit: Literal[False]) -> None: ...


@overload
def review_cmd(args, parse_obj, wrapped_by_precommit: Literal[True]) -> bool: ...


def review_cmd(
    args,
    parse_obj: Optional[Parse] = None,
    wrapped_by_precommit: bool = False,
) -> Optional[bool]:
    """This command combines the functionalities check-extract-compare and additionally colors non-chord tones."""
    logger = get_logger("ms3.review", level=args.level)
    if parse_obj is None:
        p = make_parse_obj(args)
    else:
        p = parse_obj
    accumulated_warnings = []

    # call ms3 check, collecting all warnings issued while parsing
    if args.ignore_scores + args.ignore_labels < 2:
        what = "" if args.ignore_scores else "SCORES"
        if args.ignore_labels:
            what += "LABELS" if what == "" else "AND LABELS"
        print(f"CHECKING {what}...")
        warning_strings = check(
            p,
            ignore_labels=args.ignore_labels,
            ignore_scores=args.ignore_scores,
            assertion=False,
            parallel=False,
            ignore_metronome=args.ignore_metronome,
        )
        accumulated_warnings.extend(warning_strings)
        test_passes = len(warning_strings) == 0
        p.parse_tsv()
    else:
        test_passes = True
        p.parse(parallel=False)
    if p.n_parsed_scores == 0:
        msg = "NO SCORES PARSED, NOTHING TO DO."
        if not args.all:
            msg += (
                "\nI was disregarding all files whose file names are not listed in the column 'piece' of a "
                "'metadata.tsv' file. "
                "Add -a if you want me to include all scores."
            )
        print(msg)
        return

    # call ms3 extract
    params = gather_extract_params(args)
    if len(params) > 0:
        print("EXTRACTING FACETS...")
        extract_cmd(args, p)

    # color out-of-label tones
    print("COLORING OUT-OF-LABEL TONES...")
    with capture_parse_logs(p.logger) as captured_warnings:
        test_passes = (
            make_coloring_reports_and_warnings(
                p, out_dir=args.out, threshold=args.threshold
            )
            and test_passes
        )
        accumulated_warnings.extend(captured_warnings.content_list)

    # attribute warnings to pieces based on logger names
    logger2pieceID = {
        corpus.logger_names[piece]: (c, piece)
        for c, corpus in p.corpus_objects.items()
        for piece in corpus.keys()
    }
    piece2warnings = defaultdict(list)
    for warning_strings in accumulated_warnings:
        warning_lines = warning_strings.splitlines()
        first_line = warning_lines[0]
        match = re.search(r"(ms3\.Parse\..+) --", first_line)
        if match is None:
            logger.warning(
                f"This warning contains no ms3 logger name, skipping: {warning_strings}"
            )
            continue
        warning_lines[0] = first_line[
            : match.end(1)
        ]  # cut off the warning's header everything following the logger
        # name because paths to source code are system-dependent
        pieceID = logger2pieceID[match.group(1)]
        piece2warnings[pieceID].append("\n".join(warning_lines))

    # write all warnings to piece-specific warnings files and remove existing files where no warnings were captured
    for pieceID, score_files in p.get_files(
        "scores", unparsed=False, flat=True, include_empty=False
    ).items():
        file = score_files[0]
        warnings_path = compute_path_from_file(
            file, root_dir=args.out, folder="reviewed"
        )
        warnings_file = os.path.join(
            warnings_path, file.piece + file.suffix + ".warnings"
        )
        write_warnings_to_file(
            warnings_file=warnings_file, warnings=piece2warnings[pieceID], logger=logger
        )

    # call ms3 compare
    if args.compare is not None:
        revision = None if args.compare == "" else args.compare
        commit_str = "" if revision is None else f" @ {revision}"
        expanded = "EXPANDED " if args.use == "expanded" else ""
        print(
            f"COMPARING CURRENT {expanded}LABELS WITH PREVIOUS ONES FROM THE TSV FILES{commit_str}..."
        )
        n_changed, n_unchanged = compare(
            p,
            facet=args.use,
            ask=args.ask,
            revision_specifier=revision,
            flip=args.flip,
            force=True,  # metadata be updated even if no diffs found because reviewed scores are written either way
            logger=logger,
        )
        logger.debug(
            f"{n_changed} files changed labels during comparison, {n_unchanged} didn't."
        )
    corpus2paths = store_scores(
        p, only_changed=not args.force, root_dir=args.out, simulate=args.test
    )
    changed = sum(map(len, corpus2paths.values()))
    logger.info(
        f"Operation resulted in {changed} review file{'s' if changed != 1 else ''}."
    )
    out_dir = "." if args.out is None else args.out
    warnings_file = os.path.join(out_dir, "warnings.log")
    if os.path.isfile(warnings_file):
        logger.info(f"Removing deprecated warnings file: {warnings_file}")
        os.remove(warnings_file)
    if wrapped_by_precommit:
        return test_passes
    deliver_test_result(test_passes, args.fail, logger)


def deliver_test_result(
    test_passes: bool, raise_on_fail: bool, logger: Optional[logging.Logger]
):
    def output_text(msg):
        if logger is not None:
            print(msg)
        else:
            logger.info(msg)

    if test_passes:
        output_text("Parsed scores passed all tests.")
    else:
        msg = "Not all tests have passed. Please check the relevant warnings in the generated .warnings files."
        if raise_on_fail:
            assert test_passes, msg
        else:
            output_text(msg)


def make_parse_obj(args, parse_scores=False, parse_tsv=False, facets=None):
    labels_cfg = {}
    if hasattr(args, "positioning"):
        labels_cfg["positioning"] = args.positioning
    if hasattr(args, "raw"):
        labels_cfg["decode"] = args.raw

    logger_cfg = {
        "level": args.level,
        "path": args.log,
    }
    ms = args.musescore if hasattr(args, "musescore") else None
    file_re_str = None if args.include is None else f"'{args.include}'"
    folder_re_str = None if args.folders is None else f"'{args.folders}'"
    exclude_re_str = None if args.exclude is None else f"'{args.exclude}'"
    ms_str = None if ms is None else f"'{ms}'"
    print(
        f"""CREATING PARSE OBJECT WITH THE FOLLOWING PARAMETERS:
Parse('{args.dir}',
     recursive={not args.nonrecursive},
     only_metadata_pieces={not args.all},
     include_convertible={ms is not None},
     exclude_review={not args.reviewed},
     file_re={file_re_str},
     folder_re={folder_re_str},
     exclude_re={exclude_re_str},
     file_paths={args.files},
     labels_cfg={labels_cfg},
     ms={ms_str},
     **{logger_cfg})
"""
    )
    parse_obj = Parse(
        args.dir,
        recursive=not args.nonrecursive,
        only_metadata_pieces=not args.all,
        include_convertible=ms is not None,
        exclude_review=not args.reviewed,
        file_re=args.include,
        folder_re=args.folders,
        exclude_re=args.exclude,
        file_paths=args.files,
        labels_cfg=labels_cfg,
        ms=ms,
        **logger_cfg,
    )
    if facets is not None:
        facets = [f"{f}$" for f in facets if f != "metadata"]
        if len(facets) > 0:
            parse_obj.view.include("facets", *facets)
    if parse_scores:
        mode = "ONE AFTER THE OTHER" if args.iterative else "IN PARALLEL"
        print(f"PARSING SCORES {mode}...")
        parse_obj.parse_scores(parallel=not args.iterative)
    if parse_tsv and facets != []:
        print("PARSING TSV FILES...")
        parse_obj.parse_tsv()
    info_str = parse_obj.info(show_discarded=args.verbose, return_str=True).replace(
        "\n", "\n\t"
    )
    print(f"RESULTING PARSE OBJECT:\n\t{info_str}")
    if args.verbose:
        print(inspect_loggers())
    return parse_obj


def get_arg_parser():
    # reusable argument sets
    parse_args = argparse.ArgumentParser(add_help=False)
    parse_args.add_argument(
        "-d",
        "--dir",
        metavar="DIR",
        default=os.getcwd(),
        type=check_dir,
        help="Folder(s) that will be scanned for input files. Defaults to current working directory if no individual "
        "files are passed via -f.",
    )
    parse_args.add_argument(
        "-o",
        "--out",
        metavar="OUT_DIR",
        type=check_and_create_unresolved,
        help="Output directory. For conversion, an absolute path will result in a copy of the original sub-folder "
        "structure, whereas a relative path will contain all converted files next to each other.",
    )
    parse_args.add_argument(
        "-n",
        "--nonrecursive",
        action="store_true",
        help="Treat DIR as single corpus even if it contains corpus directories itself.",
    )
    parse_args.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="By default, only files listed in the 'piece' column of a 'metadata.tsv' file are parsed. With "
        "this option, all files will be parsed.",
    )
    parse_args.add_argument(
        "-i",
        "--include",
        metavar="REGEX",
        help="Select only files whose names include this string or regular expression.",
    )
    parse_args.add_argument(
        "-e",
        "--exclude",
        metavar="REGEX",
        help="Any files or folders (and their subfolders) including this regex will be disregarded."
        "By default, files including '_reviewed' or starting with . or _ or 'concatenated' are excluded.",
    )
    parse_args.add_argument(
        "-f",
        "--folders",
        metavar="REGEX",
        help="Select only folders whose names include this string or regular expression.",
    )
    parse_args.add_argument(
        "-m",
        "--musescore",
        metavar="PATH",
        nargs="?",
        const="auto",
        help="""Command or path of your MuseScore 3 executable. -m by itself will set 'auto' (attempt to use standard
        path for your system).
        Other shortcuts are -m win, -m mac, and -m mscore (for Linux).""",
    )
    parse_args.add_argument(
        "--reviewed",
        action="store_true",
        help="By default, review files and folder are excluded from parsing. With this option, "
        "they will be included, too.",
    )
    parse_args.add_argument(
        "--files",
        metavar="PATHs",
        nargs="+",
        help="(Deprecated) The paths are expected to be within DIR. They will be converted into a view "
        "that includes only the indicated files. This is equivalent to specifying the file names as "
        "a regex via --include (assuming that file names are unique amongst corpora.",
    )
    parse_args.add_argument(
        "--iterative",
        action="store_true",
        help="Do not use all available CPU cores in parallel to speed up batch jobs.",
    )
    parse_args.add_argument(
        "-l",
        "--level",
        metavar="{c, e, w, i, d}",
        default="i",
        help="Choose how many log messages you want to see: c (none), e, w, i, d (maximum)",
    )
    parse_args.add_argument(
        "--log",
        nargs="?",
        const=".",
        help="Can be a file path or directory path. Relative paths are interpreted relative to the current directory.",
    )
    parse_args.add_argument(
        "-t", "--test", action="store_true", help="No data is written to disk."
    )
    parse_args.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show more output such as files discarded from parsing.",
    )

    check_args = argparse.ArgumentParser(add_help=False)
    check_args.add_argument(
        "--ignore_scores",
        action="store_true",
        help="Don't check scores for encoding errors.",
    )
    check_args.add_argument(
        "--ignore_labels",
        action="store_true",
        help="Don't check DCML labels for syntactic correctness.",
    )
    check_args.add_argument(
        "--fail",
        action="store_true",
        help="If you pass this argument the process will deliberately fail with an AssertionError "
        "when there are any mistakes.",
    )
    check_args.add_argument(
        "--ignore_metronome",
        action="store_true",
        help="Pass this flag if you want the check to pass (not fail) even if there is a warning about a missing "
        "metronome mark in the first bar of the score.",
    )

    compare_args = argparse.ArgumentParser(add_help=False)
    compare_args.add_argument(
        "--flip",
        action="store_true",
        help="Pass this flag to treat the annotation tables as if updating the scores instead of the other way around, "
        "effectively resulting in a swap of the colors in the output files.",
    )
    compare_args.add_argument(
        "--safe", action="store_false", help="Don't overwrite existing files."
    )
    compare_args.add_argument(
        "--force",
        action="store_true",
        help="Output comparison files even when no differences are found.",
    )

    extract_args = argparse.ArgumentParser(add_help=False)
    extract_args.add_argument(
        "-M",
        "--measures",
        metavar="folder",
        nargs="?",
        const="../measures",
        help="Folder where to store TSV files with measure information needed for tasks such as unfolding repetitions.",
    )
    extract_args.add_argument(
        "-N",
        "--notes",
        metavar="folder",
        nargs="?",
        const="../notes",
        help="Folder where to store TSV files with information on all notes.",
    )
    extract_args.add_argument(
        "-R",
        "--rests",
        metavar="folder",
        nargs="?",
        const="../rests",
        help="Folder where to store TSV files with information on all rests.",
    )
    extract_args.add_argument(
        "-L",
        "--labels",
        metavar="folder",
        nargs="?",
        const="../labels",
        help="Folder where to store TSV files with information on all annotation labels.",
    )
    extract_args.add_argument(
        "-X",
        "--expanded",
        metavar="folder",
        nargs="?",
        const="../harmonies",
        help="Folder where to store TSV files with expanded DCML labels.",
    )
    extract_args.add_argument(
        "-F",
        "--form_labels",
        metavar="folder",
        nargs="?",
        const="../form_labels",
        help="Folder where to store TSV files with all form labels.",
    )
    extract_args.add_argument(
        "-E",
        "--events",
        metavar="folder",
        nargs="?",
        const="../events",
        help="Folder where to store TSV files with all events (chords, rests, articulation, etc.) without further "
        "processing.",
    )
    extract_args.add_argument(
        "-C",
        "--chords",
        metavar="folder",
        nargs="?",
        const="../chords",
        help="Folder where to store TSV files with <chord> tags, "
        "i.e. groups of notes in the same voice with identical onset and duration. "
        "The tables include lyrics, dynamics, articulation, staff- and system texts, tempo marking, spanners, "
        "and thoroughbass figures.",
    )
    extract_args.add_argument(
        "-J",
        "--joined_chords",
        metavar="folder",
        nargs="?",
        const="../chords",
        help="Like -C except that all Chords are substituted with the actual Notes they contain. This is useful, "
        "for example, "
        "for relating slurs to the notes they group, or bass figures to their bass notes. ",
    )
    extract_args.add_argument(
        "-D",
        "--metadata",
        metavar="suffix",
        nargs="?",
        const="",
        help="Set -D to update the 'metadata.tsv' files of the respective corpora with the parsed scores. "
        "Add a suffix if you want to update 'metadata{suffix}.tsv' instead.",
    )
    extract_args.add_argument(
        "-p",
        "--positioning",
        action="store_true",
        help="When extracting labels, include manually shifted position coordinates in order to restore them when "
        "re-inserting.",
    )
    extract_args.add_argument(
        "--raw",
        action="store_false",
        help="When extracting labels, leave chord symbols encoded instead of turning them into a single column of "
        "strings.",
    )
    extract_args.add_argument(
        "-u",
        "--unfold",
        action="store_true",
        help="Unfold the repeats for all stored DataFrames.",
    )
    extract_args.add_argument(
        "--interval_index",
        action="store_true",
        help="Prepend a column with [start, end) intervals to the TSV files.",
    )
    extract_args.add_argument(
        "--corpuswise",
        action="store_true",
        help="Parse one corpus after the other rather than all at once.",
    )

    select_facet_args = argparse.ArgumentParser(add_help=False)
    select_facet_args.add_argument(
        "--ask",
        action="store_true",
        help="If several files are available for the selected facet (default: 'expanded', see --use), I will pick one "
        "automatically. Add --ask if you want me to have you select which ones to compare with the scores.",
    )
    select_facet_args.add_argument(
        "--use",
        default="expanded",
        choices=["expanded", "labels"],
        help="Which type of labels you want to compare with the ones in the score. Defaults to 'expanded', "
        "i.e., DCML labels. Set --use labels to use other labels available as TSV and set --ask if several "
        "sets of labels are available that you want to choose from.",
    )

    review_args = argparse.ArgumentParser(
        add_help=False,
        parents=[check_args, select_facet_args, compare_args, extract_args, parse_args],
    )
    review_args.add_argument(
        "-c",
        "--compare",
        nargs="?",
        metavar="GIT_REVISION",
        const="",
        help="Pass -c if you want the _reviewed file to display removed labels in red and added labels in green, "
        "compared to the version currently represented in the present TSV files, if any. If instead you want a "
        "comparison with the TSV files from another Git commit, additionally pass its specifier, e.g. 'HEAD~3', "
        "<branch-name>, <commit SHA> etc. LATEST_VERSION is accepted as a revision specifier and will result in "
        "a comparison with the TSV files at the tag with the highest version number (falling back to HEAD if no "
        "tags  have been assigned to the repository.",
    )
    review_args.add_argument(
        "--threshold",
        default=0.6,
        type=float,
        help="Harmony segments where the ratio of non-chord tones vs. chord tones lies above this threshold "
        "will be printed in a warning and will cause the check to fail if the --fail flag is set. Defaults to "
        "0.6 (3:2).",
    )

    # main argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
--------------------------
| Welcome to ms3 parsing |
--------------------------

The library offers you the following commands. Add the flag -h to one of them to learn about its parameters.
""",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(
        help="The action that you want to perform.", dest="action"
    )

    add_parser = subparsers.add_parser(
        "add",
        help="Add labels from annotation tables to scores.",
        parents=[select_facet_args, parse_args],
    )
    # TODO: staff parameter needs to accept one to several integers including negative
    # add_parser.add_argument('-s', '--staff',
    #                            help="Remove labels from selected staves only. 1=upper staff; -1=lowest staff (
    #                            default)")
    # add_parser.add_argument('--type', default=1,
    #                            help="Only remove particular types of harmony labels.")
    add_parser.add_argument(
        "-s",
        "--suffix",
        metavar="SUFFIX",
        default="_annotated",
        help="Suffix of the new scores with inserted labels. Defaults to _annotated.",
    )
    add_parser.add_argument(
        "--replace",
        action="store_true",
        help="Remove existing labels from the scores prior to adding. Like calling ms3 empty first.",
    )
    add_parser.set_defaults(func=add_cmd)

    check_parser = subparsers.add_parser(
        "check",
        help="""Parse MSCX files and look for errors.
In particular, check DCML harmony labels for syntactic correctness.""",
        parents=[check_args, parse_args],
    )
    check_parser.set_defaults(func=check_cmd)

    compare_parser = subparsers.add_parser(
        "compare",
        help="For MSCX files for which annotation tables exist, create another MSCX file with a coloured label "
        "comparison if differences are found.",
        parents=[select_facet_args, compare_args, parse_args],
    )
    compare_parser.add_argument(
        "-c",
        "--compare",
        metavar="GIT_REVISION",
        default="",
        help="By default, the _reviewed file displays removed labels in red and added labels in green, compared to "
        "the version currently represented in the present TSV files, if any. If instead you want a "
        "comparison with the TSV files from another Git commit, additionally pass its specifier, e.g. 'HEAD~3', "
        "<branch-name>, <commit SHA> etc. LATEST_VERSION is accepted as a revision specifier and will result in "
        "a comparison with the TSV files at the tag with the highest version number (falling back to HEAD if no "
        "tags  have been assigned to the repository.",
    )
    compare_parser.add_argument(
        "-s",
        "--suffix",
        metavar="SUFFIX",
        default="_compared",
        help="Suffix of the newly created comparison files. Defaults to _compared",
    )
    compare_parser.set_defaults(func=compare_cmd)

    convert_parser = subparsers.add_parser(
        "convert",
        help="Use your local install of MuseScore to convert MuseScore files.",
        parents=[parse_args],
    )
    convert_parser.add_argument(
        "-s",
        "--suffix",
        metavar="SUFFIX",
        default="",
        help="Suffix of the converted files. Defaults to " ".",
    )
    convert_parser.add_argument(
        "--format",
        default="mscx",
        help="Output format of converted files. Defaults to mscx. Other options are "
        "{png, svg, pdf, mscz, wav, mp3, flac, ogg, musicxml, mxl, mid}",
    )
    convert_parser.add_argument(
        "--extensions",
        nargs="+",
        default=["mscx", "mscz"],
        help="Those file extensions that you want to be converted, separated by spaces. Defaults to mscx mscz",
    )
    convert_parser.add_argument(
        "--safe", action="store_false", help="Don't overwrite existing files."
    )
    convert_parser.set_defaults(func=convert_cmd)

    empty_parser = subparsers.add_parser(
        "empty",
        help="Remove harmony annotations and store the MuseScore files without them.",
        parents=[parse_args],
    )
    # TODO: staff parameter needs to accept one to several integers including negative
    # empty_parser.add_argument('-s', '--staff',
    #                            help="Remove labels from selected staves only. 1=upper staff; -1=lowest staff (
    #                            default)")
    # empty_parser.add_argument('--type', default=1,
    #                            help="Only remove particular types of harmony labels.")
    empty_parser.add_argument(
        "-s",
        "--suffix",
        metavar="SUFFIX",
        default="_clean",
        help="Suffix of the new scores with removed labels. Defaults to _clean.",
    )
    empty_parser.set_defaults(func=empty)

    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract selected information from MuseScore files and store it in TSV files.",
        parents=[extract_args, parse_args],
    )
    extract_parser.set_defaults(func=extract_cmd)

    metadata_parser = subparsers.add_parser(
        "metadata",
        help="Update MSCX files with changes made to metadata.tsv (created via ms3 extract -D [-a]).",
        parents=[parse_args],
    )
    metadata_parser.add_argument(
        "-s",
        "--suffix",
        metavar="SUFFIX",
        help="Suffix of the new scores with updated metadata fields.",
    )
    metadata_parser.add_argument(
        "-p",
        "--prelims",
        action="store_true",
        help="Pass this flag if, in addition to updating metadata fields, you also want score headers to be updated "
        "from the columns title_text, subtitle_text, composer_text, lyricist_text, part_name_text.",
    )
    metadata_parser.add_argument(
        "--instrumentation",
        action="store_true",
        help="Pass this flag to update the score's instrumentation based on changed values from "
        "'staff_<i>_instrument' columns.",
    )
    metadata_parser.add_argument(
        "--empty",
        action="store_true",
        help="Set this flag to also allow empty values to be used for overwriting existing ones.",
    )
    metadata_parser.add_argument(
        "--remove",
        action="store_true",
        help="Set this flag to remove non-default metadata fields that are not columns in the metadata.tsv file "
        "anymore.",
    )
    metadata_parser.set_defaults(func=metadata)
    #
    # repair_parser = subparsers.add_parser('repair',
    #                                       help="Apply automatic repairs to your uncompressed MuseScore files.",
    #                                       parents=[parse_args])
    # repair_parser.set_defaults(func=repair)

    review_parser = subparsers.add_parser(
        "review",
        help="Extract facets, check labels, and create _reviewed files.",
        parents=[review_args],
    )
    review_parser.set_defaults(func=review_cmd)

    transform_parser = subparsers.add_parser(
        "transform",
        help="Concatenate and transform TSV data from one or several corpora. "
        "Available transformations are unfolding repeats and adding an interval index.",
        parents=[parse_args],
    )
    transform_parser.add_argument(
        "-M",
        "--measures",
        action="store_true",
        help="Concatenate measures TSVs for all selected pieces.",
    )
    transform_parser.add_argument(
        "-N",
        "--notes",
        action="store_true",
        help="Concatenate notes TSVs for all selected pieces.",
    )
    transform_parser.add_argument(
        "-R",
        "--rests",
        action="store_true",
        help="Concatenate rests TSVs for all selected pieces (use ms3 extract -R to create those).",
    )
    transform_parser.add_argument(
        "-L",
        "--labels",
        action="store_true",
        help="Concatenate raw harmony label TSVs for all selected pieces (use ms3 extract -L to create those).",
    )
    transform_parser.add_argument(
        "-X",
        "--expanded",
        action="store_true",
        help="Concatenate expanded DCML label TSVs for all selected pieces.",
    )
    transform_parser.add_argument(
        "-F",
        "--form_labels",
        metavar="folder",
        nargs="?",
        const="../form_labels",
        help="Concatenate form label TSVs for all selected pieces.",
    )
    transform_parser.add_argument(
        "-E",
        "--events",
        action="store_true",
        help="Concatenate events TSVs (notes, rests, articulation, etc.) for all selected pieces (use ms3 extract -E "
        "to create those).",
    )
    transform_parser.add_argument(
        "-C",
        "--chords",
        action="store_true",
        help="Concatenate chords TSVs (<chord> tags group notes in the same voice with identical onset and duration) "
        "including "
        "lyrics, dynamics, articulation, staff- and system texts, tempo marking, spanners, and thoroughbass "
        "figures, "
        "for all selected pieces (use ms3 extract -C to create those).",
    )
    transform_parser.add_argument(
        "-D",
        "--metadata",
        action="store_true",
        help="Output 'concatenated_metadata.tsv' with one row per selected piece.",
    )
    transform_parser.add_argument(
        "-s",
        "--suffix",
        nargs="*",
        metavar="SUFFIX",
        help="Pass -s to use standard suffixes or -s SUFFIX to choose your own. In the latter case they will be "
        "assigned to the extracted aspects in the order "
        "in which they are listed above (capital letter arguments).",
    )
    transform_parser.add_argument(
        "-u",
        "--unfold",
        action="store_true",
        help="Unfold the repeats for all concatenated DataFrames.",
    )
    transform_parser.add_argument(
        "--interval_index",
        action="store_true",
        help="Prepend a column with [start, end) intervals to the TSV files.",
    )
    transform_parser.add_argument(
        "--resources",
        action="store_true",
        help="Store the concatenated DataFrames as TSV files with resource descriptors rather than in a ZIP with a "
        "package descriptor.",
    )
    transform_parser.add_argument(
        "--safe", action="store_false", help="Don't overwrite existing files."
    )
    transform_parser.add_argument(
        "--uncompressed",
        action="store_true",
        help="Store the transformed files as uncompressed TSVs rather than writing them into a ZIP file.",
    )
    transform_parser.add_argument(
        "--dirty",
        action="store_true",
        help="Allows to override the 'This repository is dirty' blocker.",
    )
    transform_parser.set_defaults(func=transform_cmd)

    update_parser = subparsers.add_parser(
        "update",
        help="Convert MSCX files to the latest MuseScore version and move all chord annotations "
        "to the Roman Numeral Analysis layer. This command overwrites existing files!!!",
        parents=[parse_args],
    )
    # update_parser.add_argument('-a', '--annotations', metavar='PATH', default='../harmonies',
    #                             help='Path relative to the score file(s) where to look for existing annotation
    #                             tables.')
    update_parser.add_argument(
        "-s",
        "--suffix",
        metavar="SUFFIX",
        help="Add this suffix to the filename of every new file.",
    )
    update_parser.add_argument(
        "--above", action="store_true", help="Display Roman Numerals above the system."
    )
    update_parser.add_argument(
        "--safe",
        action="store_true",
        help="Only moves labels if their temporal positions stay intact.",
    )
    update_parser.add_argument(
        "--staff",
        default=-1,
        help="Which staff you want to move the annotations to. 1=upper staff; -1=lowest staff (default)",
    )
    update_parser.add_argument(
        "--type",
        default=1,
        help="defaults to 1, i.e. moves labels to Roman Numeral layer. Other types have not been tested!",
    )
    update_parser.set_defaults(func=update_cmd)

    precommit_parser = subparsers.add_parser(
        "precommit",
        help="Like ms3 review but also adds the resulting files to the Git index.",
        parents=[review_args],
    )
    precommit_parser.add_argument(
        "positional_args",
        metavar="FILE",
        help="Shadows the --files argument because pre-commit passes files as positional arguments.",
        nargs="+",
    )
    precommit_parser.set_defaults(func=precommit_cmd)
    return parser


def run():
    parser = get_arg_parser()
    args = parser.parse_args()
    if "func" not in args:
        parser.print_help()
        return
    if args.files is not None:
        args.files = [resolve_dir(path) for path in args.files]
    args.func(args)


if __name__ == "__main__":
    run()
