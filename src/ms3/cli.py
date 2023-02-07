#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command line interface for ms3.
"""

import argparse, os
from datetime import datetime
from typing import Optional

import pandas as pd

from ms3 import Parse
from ms3.operations import extract, check, compare, update, store_scores, insert_labels_into_score
from ms3.utils import convert_folder, resolve_dir, write_tsv, MS3_VERSION
from ms3.logger import get_logger, inspect_loggers

__author__ = "johentsch"
__copyright__ = "École Polytechnique Fédérale de Lausanne"
__license__ = "gpl3"

def gather_extract_params(args):
    params = [name for name, arg in zip(
        ('measures', 'notes', 'rests', 'labels', 'expanded', 'form_labels', 'events', 'chords'),
        (args.measures, args.notes, args.rests, args.labels, args.expanded, args.form_labels, args.events, args.chords))
              if arg]
    if args.metadata is not None:
        params.append('metadata')
    return params




def make_suffixes(args, include_metadata: bool = False):
    params = gather_extract_params(args)
    if args.suffix is None:
        suffixes = {}
    else:
        l_suff = len(args.suffix)
        if l_suff == 0:
            suffixes = {f"{p}_suffix": f"_{p}" for p in params}
        elif l_suff == 1:
            suffixes = {f"{p}_suffix": args.suffix[0] for p in params}
        else:
            suffixes = {f"{p}_suffix": args.suffix[i] if i < l_suff else f"_{p}" for i, p in enumerate(params)}
    if not include_metadata and "metadata_suffix" in suffixes:
        del (suffixes["metadata_suffix"])
    return suffixes


def add_cmd(args,
            parse_obj: Optional[Parse] = None) -> Parse:
    logger = get_logger('ms3.add', level=args.level)
    if parse_obj is None:
        p = make_parse_obj(args)
    else:
        p = parse_obj

    insert_labels_into_score(p,
                             facet=args.use,
                             ask_for_input=args.ask,
                             replace=args.replace,
                             )
    corpus2paths = store_scores(p,
                                root_dir=args.out,
                                folder='.',
                                simulate=args.test,
                                suffix=args.suffix)
    changed = sum(map(len, corpus2paths.values()))
    logger.info(f"Operation resulted in {changed} annotated score{'s' if changed != 1 else ''}.")

def check_cmd(args,
              parse_obj: Optional[Parse] = None) -> Parse:
    if parse_obj is None:
        p = make_parse_obj(args)
    else:
        p = parse_obj
    _ = check(p,
              ignore_labels=args.ignore_labels,
              ignore_scores=args.ignore_scores,
              assertion=args.fail)
    return p


def compare_cmd(args,
                parse_obj=None,
                output: bool = True,
                logger = None) -> None:
    if logger is None:
        logger = get_logger('ms3.compare', level=args.level)
    if parse_obj is None:
        p = make_parse_obj(args)
    else:
        p = parse_obj
    n_changed, n_unchanged = compare(p,
                                 facet=args.use,
                                 ask=args.ask,
                                 revision_specifier=args.compare,
                                 flip=args.flip,
                                 logger=p.logger
                                 )
    logger.debug(f"{n_changed} files changed labels during comparison, {n_unchanged} didn't.")
    if output:
        corpus2paths = store_scores(p,
                                    root_dir=args.out,
                                    simulate=args.test,
                                    suffix=args.suffix)
        changed = sum(map(len, corpus2paths.values()))
        logger.info(f"Operation resulted in {changed} comparison file{'s' if changed != 1 else ''}.")
    else:
        logger.info(f"Operation resulted in colored labels within {n_changed} score{'s' if n_changed != 1 else ''}.")



def convert_cmd(args):
    # assert target[:len(
    #    dir)] != dir, "TARGET_DIR cannot be identical with nor a subfolder of DIR.\nDIR:        " + dir + '\nTARGET_DIR: ' + target
    update_logger = get_logger("ms3.convert", level=args.level)
    for argument_name, argument, default in (('-a/--all', args.all, False),
                                             ('-e/--exclude', args.exclude, None),
                                             ('-f/--folders', args.folders, None),
                                             ('--reviewed', args.reviewed, False),
                                             ('-t/--test', args.test, False),
                                             ('-v/--verbose', args.verbose, False),
                                             ):
        if argument != default:
            update_logger.info(f"Argument '{argument_name}' is currently being ignored.")
    out_dir = os.getcwd() if args.out is None else resolve_dir(args.out)
    ms = 'auto' if args.musescore is None else args.musescore
    convert_folder(directory=args.dir,
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
                   logger=update_logger)

def empty(args):
    logger_cfg = {
        'level': args.level,
        'path': args.log,
    }
    p = Parse(args.dir, recursive=not args.nonrecursive, file_re=args.regex, exclude_re=args.exclude, file_paths=args.file, **logger_cfg)
    p.parse_scores(parallel=False)
    p.detach_labels()
    p.logger.info(f"Overview of the removed labels:\n{p.count_annotation_layers(which='detached').to_string()}")
    ids = [id for id, score in p._parsed_mscx.items() if score.mscx.changed]
    if args.out is not None:
        p.store_scores(ids=ids, root_dir=args.out, overwrite=True)
    else:
        p.store_scores(ids=ids, overwrite=True)


def extract_cmd(args, parse_obj: Optional[Parse] = None):
    if parse_obj is None:
        p = make_parse_obj(args, parse_scores=True)
    else:
        p = parse_obj
    params = gather_extract_params(args)
    if len(params) == 0:
        print("In order to extract DataFrames, pass at least one of the following arguments: -M (measures), -N (notes), -R (rests), -L (labels), -X (expanded), -F (form_labels), -E (events), -C (chords), -D (metadata)")
        return
    suffixes = make_suffixes(args)
    silence_label_warnings = args.silence_label_warnings if hasattr(args, 'silence_label_warnings') else False
    extract(p,
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
            unfold=args.unfold,
            interval_index=args.interval_index,
            silence_label_warnings=silence_label_warnings,
            **suffixes)

def metadata(args, parse_obj: Optional[Parse] = None):
    """ Update MSCX files with changes made in metadata.tsv (created via ms3 extract -D). In particular,
        add the values from (new?) columns to the corresponding fields in the MuseScore files' "Score info".
    """
    if parse_obj is None:
        p = make_parse_obj(args)
    else:
        p = parse_obj
    print(f"PARSING SCORES ONE AFTER THE OTHER...")
    p.parse_scores(parallel=False)
    modified_score_files = p.update_score_metadata_from_tsv(write_empty_values=args.empty,
                                                            remove_unused_fields=args.remove,
                                                            write_text_fields=args.prelims)
    if len(modified_score_files) == 0:
        print("Nothing to update.")
        return
    corpus2paths = store_scores(p,
                                root_dir=args.out,
                                folder='.',
                                simulate=args.test,
                                suffix=args.suffix)
    changed = sum(map(len, corpus2paths.values()))
    print(f"Operation resulted in {changed} updated score{'s' if changed != 1 else ''}.")


# def repair(args):
#     print("Sorry, the command has not been implemented yet.")
#     print(args.dir)


def transform(args, parse_obj: Optional[Parse] = None):
    suffixes = make_suffixes(args, include_metadata=True)
    if parse_obj is None:
        p = make_parse_obj(args, parse_tsv=True)
    else:
        p = parse_obj
    params = gather_extract_params(args)
    print(params)
    if len(params) == 0:
        print(
            "Pass at least one of the following arguments: -M (measures), -N (notes), -R (rests), -L (labels), -X (expanded), -F (form_labels), -E (events), -C (chords), -D (metadata)")
        return

    for param in params:
        sfx = suffixes.get(f"{param}_suffix", '')
        tsv_name = f"concatenated_{param}{sfx}.tsv"
        if args.out is None:
            path = tsv_name
        else:
            path = os.path.join(args.out, tsv_name)
        if param == 'metadata':
            df = p.metadata()
        else:
            df = p.get_facets(param, flat=True, interval_index=args.interval_index, unfold=args.unfold)
        df = df.reset_index(drop=False)
        if args.test:
            p.logger.info(f"Would have written {path}.")
        else:
            write_tsv(df, path)
            p.logger.info(f"{path} written.")


def update_cmd(args, parse_obj: Optional[Parse] = None):
    if parse_obj is None:
        p = make_parse_obj(args, parse_scores=True)
    else:
        p = parse_obj
    changed_paths = update(p,
                           root_dir=args.out,
                           suffix='' if args.suffix is None else args.suffix,
                           overwrite=True,
                           staff=args.staff,
                           harmony_layer=args.type,
                           above=args.above,
                           safe=args.safe)
    if len(changed_paths) > 0:
        print(f"Paths of scores with updated labels:\n{changed_paths}")
    else:
        print(f"Nothing to do.")

def check_and_create(d):
    """ Turn input into an existing, absolute directory path.
    """
    if not os.path.isdir(d):
        d = resolve_dir(os.path.join(os.getcwd(), d))
        if not os.path.isdir(d):
            if input(d + ' does not exist. Create? (y|n)') == "y":
                os.mkdir(d)
            else:
                raise argparse.ArgumentTypeError(d + ' needs to be an existing directory')
    return resolve_dir(d)

def check_dir(d):
    if not os.path.isdir(d):
        d = resolve_dir(os.path.join(os.getcwd(), d))
        if not os.path.isdir(d):
            raise argparse.ArgumentTypeError(d + ' needs to be an existing directory')
    return resolve_dir(d)




def review_cmd(args,
               parse_obj: Optional[Parse] = None,
               ) -> None:
    """This command combines the functionalities check-extract-compare and additionally colors non-chord tones."""
    logger = get_logger('ms3.review', level=args.level)
    if parse_obj is None:
        p = make_parse_obj(args)
    else:
        p = parse_obj

    # create file for warnings
    out_dir = '.' if args.out is None else args.out
    warnings_file = os.path.join(out_dir, 'warnings.log')
    header = f"Warnings encountered during the last execution of ms3 review (v{MS3_VERSION})"
    header = f"{header}\n{'=' * len(header)}\n\n"
    with open(warnings_file, 'w', encoding='utf-8') as f:
        f.write(header)

    # call ms3 check
    if args.ignore_scores + args.ignore_labels < 2:
        what = '' if args.ignore_scores else 'SCORES'
        if args.ignore_labels:
            what += 'LABELS' if what == '' else 'AND LABELS'
        print(f"CHECKING {what}...")
        test_passes = check(p,
                            ignore_labels=args.ignore_labels,
                            ignore_scores=args.ignore_scores,
                            assertion=False,
                            parallel=False,
                            warnings_file=warnings_file)
        p.parse_tsv()
    else:
        p.parse(parallel=False)
    if p.n_parsed_scores == 0:
        msg = "NO SCORES PARSED, NOTHING TO DO."
        if not args.all:
            msg += "\nI was disregarding all files whose file names are not listed in the column 'fname' of a 'metadata.tsv' file. " \
                   "Add -a if you want me to include all scores."
        print(msg)
        os.remove(warnings_file)
        return

    # call ms3 extract
    params = gather_extract_params(args)
    if len(params) > 0:
        print("EXTRACTING FACETS...")
        extract_cmd(args, p)

    # color out-of-label tones
    print("COLORING OUT-OF-LABEL TONES...")
    review_reports = p.color_non_chord_tones()
    if len(review_reports) > 0:
        dataframes, keys = [], []
        for (corpus_name, fname), tuples in review_reports.items():
            for file, df in tuples:
                dataframes.append(df)
                keys.append((corpus_name, file.rel_path))
                report_path = os.path.join(file.corpus_path, 'reviewed', file.fname + '_reviewed.tsv')
                write_tsv(df, report_path)
        report = pd.concat(dataframes, keys=keys)
        warning_selection = (report.count_ratio > args.threshold) & report.chord_tones.notna()
        if warning_selection.sum() > 0:
            test_passes = False
            filtered_report = report[warning_selection].droplevel(-1)
            pretty_report = filtered_report.to_string(columns=['mc', 'mn', 'mc_onset', 'label', 'chord_tones', 'added_tones', 'n_colored', 'n_untouched', 'count_ratio'])
            logger.warning(pretty_report,
                           extra={'message_id': (19,)})
            with open(warnings_file, 'a', encoding='utf-8') as f:
                f.write(f"Chord labels where the ratio of colored notes vs. all notes in a segment lies above {args.threshold}:\n")
                f.write(pretty_report + "\n")
            logger.info(f"Added the {warning_selection.sum()} labels to {warnings_file}.")

    # call ms3 compare
    if args.compare is not None:
        revision = None if args.compare == '' else args.compare
        commit_str = '' if revision is None else f" @ {revision}"
        expanded = 'EXPANDED ' if args.use == 'expanded' else ''
        print(f"COMPARING CURRENT {expanded}LABELS WITH PREVIOUS ONES FROM THE TSV FILES{commit_str}...")
        n_changed, n_unchanged = compare(p,
                                         facet=args.use,
                                         ask=args.ask,
                                         revision_specifier=revision,
                                         flip=args.flip,
                                         logger=logger
                                         )
        logger.debug(f"{n_changed} files changed labels during comparison, {n_unchanged} didn't.")
    corpus2paths = store_scores(p,
                                root_dir=args.out,
                                simulate=args.test)
    changed = sum(map(len, corpus2paths.values()))
    logger.info(f"Operation resulted in {changed} review file{'s' if changed != 1 else ''}.")
    if test_passes:
        logger.info(f"Parsed scores passed all tests. Removed {warnings_file}")
        os.remove(warnings_file)
    else:
        msg = f"Not all tests have passed. Please check the relevant warnings in {warnings_file}"
        if args.fail:
            assert test_passes, msg
        else:
            logger.info(msg)




def make_parse_obj(args, parse_scores=False, parse_tsv=False):
    labels_cfg = {}
    if hasattr(args, 'positioning'):
        labels_cfg['positioning'] = args.positioning
    if hasattr(args, 'raw'):
        labels_cfg['decode'] = args.raw

    logger_cfg = {
        'level': args.level,
        'path': args.log,
    }
    ms = args.musescore if hasattr(args, 'musescore') else None
    file_re_str = None if args.include is None else f"'{args.include}'"
    folder_re_str = None if args.folders is None else f"'{args.folders}'"
    exclude_re_str = None if args.exclude is None else f"'{args.exclude}'"
    ms_str = None if ms is None else f"'{ms}'"
    print(f"""CREATING PARSE OBJECT WITH THE FOLLOWING PARAMETERS:
Parse('{args.dir}',
     recursive={not args.nonrecursive},
     only_metadata_fnames={not args.all},
     include_convertible={ms is not None},
     exclude_review={not args.reviewed},
     file_re={file_re_str},
     folder_re={folder_re_str},
     exclude_re={exclude_re_str},
     file_paths={args.files},
     labels_cfg={labels_cfg},
     ms={ms_str},
     **{logger_cfg})
""")
    parse_obj = Parse(args.dir,
                      recursive=not args.nonrecursive,
                      only_metadata_fnames=not args.all,
                      include_convertible=ms is not None,
                      exclude_review=not args.reviewed,
                      file_re=args.include,
                      folder_re=args.folders,
                      exclude_re=args.exclude,
                      file_paths=args.files,
                      labels_cfg=labels_cfg,
                      ms=ms,
                      **logger_cfg)
    if parse_scores:
        mode = "ONE AFTER THE OTHER" if args.iterative else "IN PARALLEL"
        print(f"PARSING SCORES {mode}...")
        parse_obj.parse_scores(parallel=not args.iterative)
    if parse_tsv:
        print(f"PARSING TSV FILES...")
        parse_obj.parse_tsv()
    info_str = parse_obj.info(show_discarded=args.verbose, return_str=True).replace('\n', '\n\t')
    print(f"RESULTING PARSE OBJECT:\n\t{info_str}")
    if args.verbose:
        print(inspect_loggers())
    return parse_obj




def get_arg_parser():
    # reusable argument sets
    parse_args = argparse.ArgumentParser(add_help=False)
    parse_args.add_argument('-d', '--dir', metavar='DIR', default=os.getcwd(), type=check_dir,
                                help='Folder(s) that will be scanned for input files. Defaults to current working directory if no individual files are passed via -f.')
    parse_args.add_argument('-o', '--out', metavar='OUT_DIR', type=check_and_create,
                                help='Output directory.')
    parse_args.add_argument('-n', '--nonrecursive', action='store_true',
                            help='Treat DIR as single corpus even if it contains corpus directories itself.')
    parse_args.add_argument('-a', '--all', action='store_true',
                            help="By default, only files listed in the 'fname' column of a 'metadata.tsv' file are parsed. With "
                                 "this option, all files will be parsed.")
    parse_args.add_argument('-i', '--include', metavar="REGEX",
                                help="Select only files whose names include this string or regular expression.")
    parse_args.add_argument('-e', '--exclude', metavar="REGEX",
                                help="Any files or folders (and their subfolders) including this regex will be disregarded."
                                     "By default, files including '_reviewed' or starting with . or _ or 'concatenated' are excluded.")
    parse_args.add_argument('-f', '--folders', metavar="REGEX",
                                help="Select only folders whose names include this string or regular expression.")
    parse_args.add_argument('-m', '--musescore', metavar="PATH", nargs='?', const='auto', help="""Command or path of your MuseScore 3 executable. -m by itself will set 'auto' (attempt to use standard path for your system).
        Other shortcuts are -m win, -m mac, and -m mscore (for Linux).""")
    parse_args.add_argument('--reviewed', action='store_true',
                            help='By default, review files and folder are excluded from parsing. With this option, '
                                 'they will be included, too.')
    parse_args.add_argument('--files', metavar='PATHs', nargs='+',
                            help='(Deprecated) The paths are expected to be within DIR. They will be converted into a view '
                                 'that includes only the indicated files. This is equivalent to specifying the file names as '
                                 'a regex via --include (assuming that file names are unique amongst corpora.')
    parse_args.add_argument('--iterative', action='store_true',
                                help="Do not use all available CPU cores in parallel to speed up batch jobs.")
    parse_args.add_argument('-l', '--level', metavar='{c, e, w, i, d}', default='i',
                                help="Choose how many log messages you want to see: c (none), e, w, i, d (maximum)")
    parse_args.add_argument('--log', nargs='?', const='.', help='Can be a file path or directory path. Relative paths are interpreted relative to the current directory.')
    parse_args.add_argument('-t', '--test', action='store_true', help="No data is written to disk.")
    parse_args.add_argument('-v', '--verbose', action='store_true', help="Show more output such as files discarded from parsing.")

    check_args = argparse.ArgumentParser(add_help=False)
    check_args.add_argument('--ignore_scores', action='store_true',
                              help="Don't check scores for encoding errors.")
    check_args.add_argument('--ignore_labels', action='store_true',
                              help="Don't check DCML labels for syntactic correctness.")
    check_args.add_argument('--fail', action='store_true', help="If you pass this argument the process will deliberately fail with an AssertionError "
                                                                "when there are any mistakes.")

    compare_args = argparse.ArgumentParser(add_help=False)
    compare_args.add_argument('--flip', action='store_true',
                                help="Pass this flag to treat the annotation tables as if updating the scores instead of the other way around, "
                                     "effectively resulting in a swap of the colors in the output files.")
    compare_args.add_argument('--safe', action='store_false', help="Don't overwrite existing files.")

    extract_args = argparse.ArgumentParser(add_help=False)
    extract_args.add_argument('-M', '--measures', metavar='folder', nargs='?',
                                const='../measures',
                                help="Folder where to store TSV files with measure information needed for tasks such as unfolding repetitions.")
    extract_args.add_argument('-N', '--notes', metavar='folder', nargs='?', const='../notes',
                                help="Folder where to store TSV files with information on all notes.")
    extract_args.add_argument('-R', '--rests', metavar='folder', nargs='?', const='../rests',
                                help="Folder where to store TSV files with information on all rests.")
    extract_args.add_argument('-L', '--labels', metavar='folder', nargs='?', const='../labels',
                                help="Folder where to store TSV files with information on all annotation labels.")
    extract_args.add_argument('-X', '--expanded', metavar='folder', nargs='?', const='../harmonies',
                                help="Folder where to store TSV files with expanded DCML labels.")
    extract_args.add_argument('-F', '--form_labels', metavar='folder', nargs='?', const='../form_labels',
                                help="Folder where to store TSV files with all form labels.")
    extract_args.add_argument('-E', '--events', metavar='folder', nargs='?', const='../events',
                                help="Folder where to store TSV files with all events (chords, rests, articulation, etc.) without further processing.")
    extract_args.add_argument('-C', '--chords', metavar='folder', nargs='?', const='../chords',
                                help="Folder where to store TSV files with <chord> tags, "
                                     "i.e. groups of notes in the same voice with identical onset and duration. "
                                     "The tables include lyrics, dynamics, articulation, staff- and system texts, tempo marking, spanners, and thoroughbass figures.")
    extract_args.add_argument('-J', '--joined_chords', metavar='folder', nargs='?', const='../chords',
                              help="Like -C except that all Chords are substituted with the actual Notes they contain. This is useful, for example, "
                                   "for relating slurs to the notes they group, or bass figures to their bass notes. ")
    extract_args.add_argument('-D', '--metadata', metavar='suffix', nargs='?', const='',
                                help="Set -D to update the 'metadata.tsv' files of the respective corpora with the parsed scores. "
                                     "Add a suffix if you want to update 'metadata{suffix}.tsv' instead.")
    extract_args.add_argument('-s', '--suffix', nargs='*', metavar='SUFFIX',
                                help="Pass -s to use standard suffixes or -s SUFFIX to choose your own. In the latter case they will be assigned to the extracted aspects in the order "
                                     "in which they are listed above (capital letter arguments).")
    extract_args.add_argument('-p', '--positioning', action='store_true',
                                help="When extracting labels, include manually shifted position coordinates in order to restore them when re-inserting.")
    extract_args.add_argument('--raw', action='store_false',
                                help="When extracting labels, leave chord symbols encoded instead of turning them into a single column of strings.")
    extract_args.add_argument('-u', '--unfold', action='store_true',
                                help="Unfold the repeats for all stored DataFrames.")
    extract_args.add_argument('--interval_index', action='store_true',
                                help="Prepend a column with [start, end) intervals to the TSV files.")

    select_facet_args = argparse.ArgumentParser(add_help=False)
    select_facet_args.add_argument('--ask', action='store_true',
                              help="If several files are available for the selected facet (default: 'expanded', see --use), I will pick one "
                                   "automatically. Add --ask if you want me to have you select which ones to compare with the scores.")
    select_facet_args.add_argument('--use', default='expanded', choices=['expanded', 'labels'],
                              help="Which type of labels you want to compare with the ones in the score. Defaults to 'expanded', "
                                   "i.e., DCML labels. Set --use labels to use other labels available as TSV and set --ask if several "
                                   "sets of labels are available that you want to choose from.")

    # main argument parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='''\
--------------------------
| Welcome to ms3 parsing |
--------------------------

The library offers you the following commands. Add the flag -h to one of them to learn about its parameters. 
''')
    parser.add_argument('--version', action='version', version=MS3_VERSION)
    subparsers = parser.add_subparsers(help='The action that you want to perform.', dest='action')



    add_parser = subparsers.add_parser('add',
                                         help="Add labels from annotation tables to scores.",
                                         parents=[select_facet_args, parse_args])
    # TODO: staff parameter needs to accept one to several integers including negative
    # add_parser.add_argument('-s', '--staff',
    #                            help="Remove labels from selected staves only. 1=upper staff; -1=lowest staff (default)")
    # add_parser.add_argument('--type', default=1,
    #                            help="Only remove particular types of harmony labels.")
    add_parser.add_argument('-s', '--suffix', metavar='SUFFIX', default='_annotated',
                                help='Suffix of the new scores with inserted labels. Defaults to _annotated.')
    add_parser.add_argument('--replace', action='store_true',
                               help="Remove existing labels from the scores prior to adding. Like calling ms3 empty first.")
    add_parser.set_defaults(func=add_cmd)



    check_parser = subparsers.add_parser('check', help="""Parse MSCX files and look for errors.
In particular, check DCML harmony labels for syntactic correctness.""", parents=[check_args, parse_args])
    check_parser.set_defaults(func=check_cmd)



    compare_parser = subparsers.add_parser('compare',
        help="For MSCX files for which annotation tables exist, create another MSCX file with a coloured label comparison if differences are found.",
        parents = [select_facet_args, compare_args, parse_args])
    compare_parser.add_argument('-c', '--compare', metavar='GIT_REVISION', default='',
                              help="By default, the _reviewed file displays removed labels in red and added labels in green, compared to the version currently "
                                   "represented in the present TSV files, if any. If instead you want a comparison with the TSV files from another Git commit, "
                                   "pass its specifier, e.g. 'HEAD~3', <branch-name>, <commit SHA> etc.")
    compare_parser.add_argument('-s', '--suffix', metavar='SUFFIX', default='_compared',
                              help='Suffix of the newly created comparison files. Defaults to _compared')
    compare_parser.set_defaults(func=compare_cmd)



    convert_parser = subparsers.add_parser('convert',
                                           help="Use your local install of MuseScore to convert MuseScore files.",
                                           parents=[parse_args])
    convert_parser.add_argument('-s', '--suffix', metavar='SUFFIX', default='',
                              help='Suffix of the converted files. Defaults to ''.')
    convert_parser.add_argument('--format', default='mscx',
                                help="Output format of converted files. Defaults to mscx. Other options are "
                                     "{png, svg, pdf, mscz, wav, mp3, flac, ogg, xml, mxl, mid}")
    convert_parser.add_argument('--extensions', nargs='+', default=['mscx', 'mscz'],
                                help="Those file extensions that you want to be converted, separated by spaces. Defaults to mscx mscz")
    convert_parser.add_argument('--safe', action='store_false', help="Don't overwrite existing files.")
    convert_parser.set_defaults(func=convert_cmd)



    empty_parser = subparsers.add_parser('empty',
                                         help="Remove harmony annotations and store the MuseScore files without them.",
                                         parents=[parse_args])
    # TODO: staff parameter needs to accept one to several integers including negative
    # empty_parser.add_argument('-s', '--staff',
    #                            help="Remove labels from selected staves only. 1=upper staff; -1=lowest staff (default)")
    # empty_parser.add_argument('--type', default=1,
    #                            help="Only remove particular types of harmony labels.")
    empty_parser.set_defaults(func=empty)

    extract_parser = subparsers.add_parser('extract',
                                           help="Extract selected information from MuseScore files and store it in TSV files.",
                                           parents=[extract_args, parse_args])
    extract_parser.set_defaults(func=extract_cmd)



    metadata_parser = subparsers.add_parser('metadata',
                                            help="Update MSCX files with changes made to metadata.tsv (created via ms3 extract -D [-a]).",
                                            parents=[parse_args])
    metadata_parser.add_argument('-s', '--suffix', metavar='SUFFIX',
                            help='Suffix of the new scores with updated metadata fields.')
    metadata_parser.add_argument('-p', '--prelims', action='store_true',
                                 help="Pass this flag if, in addition to updating metadata fields, you also want score headers to be updated "
                                      "from the columns title_text, subtitle_text, composer_text, lyricist_text, part_name_text.")
    metadata_parser.add_argument('--empty', action='store_true',
                                  help="Set this flag to also allow empty values to be used for overwriting existing ones.")
    metadata_parser.add_argument('--remove', action='store_true',
                                  help="Set this flag to remove non-default metadata fields that are not columns in the metadata.tsv file anymore.")
    metadata_parser.set_defaults(func=metadata)
    #
    # repair_parser = subparsers.add_parser('repair',
    #                                       help="Apply automatic repairs to your uncompressed MuseScore files.",
    #                                       parents=[parse_args])
    # repair_parser.set_defaults(func=repair)

    review_parser = subparsers.add_parser('review',
                                          help="Extract facets, check labels, and create _reviewed files.",
                                          parents=[check_args, select_facet_args, compare_args, extract_args, parse_args])
    review_parser.add_argument('-c', '--compare', nargs="?", metavar='GIT_REVISION', const='',
                              help="Pass -c if you want the _reviewed file to display removed labels in red and added labels in green, compared to the version currently "
                                   "represented in the present TSV files, if any. If instead you want a comparison with the TSV files from another Git commit, additionally "
                                   "pass its specifier, e.g. 'HEAD~3', <branch-name>, <commit SHA> etc.")
    review_parser.add_argument('--threshold', default=0.6,
                                  help="Harmony segments where the ratio of non-chord tones vs. chord tones lies above this threshold "
                                       "will be printed in a warning and will cause the check to fail if the --fail flag is set. Defaults to 0.6 (3:2).")
    review_parser.set_defaults(func=review_cmd)

    transform_parser = subparsers.add_parser('transform',
                                          help="Concatenate and transform TSV data from one or several corpora. "
                                               "Available transformations are unfolding repeats and adding an interval index.",
                                          parents=[parse_args])
    transform_parser.add_argument('-M', '--measures', action='store_true',
                                help="Concatenate measures TSVs for all selected pieces.")
    transform_parser.add_argument('-N', '--notes', action='store_true',
                                help="Concatenate notes TSVs for all selected pieces.")
    transform_parser.add_argument('-R', '--rests', action='store_true',
                                help="Concatenate rests TSVs for all selected pieces (use ms3 extract -R to create those).")
    transform_parser.add_argument('-L', '--labels', action='store_true',
                                help="Concatenate raw harmony label TSVs for all selected pieces (use ms3 extract -L to create those).")
    transform_parser.add_argument('-X', '--expanded', action='store_true',
                                help="Concatenate expanded DCML label TSVs for all selected pieces.")
    transform_parser.add_argument('-F', '--form_labels', metavar='folder', nargs='?', const='../form_labels',
                              help="Concatenate form label TSVs for all selected pieces.")
    transform_parser.add_argument('-E', '--events', action='store_true',
                                help="Concatenate events TSVs (notes, rests, articulation, etc.) for all selected pieces (use ms3 extract -E to create those).")
    transform_parser.add_argument('-C', '--chords', action='store_true',
                                help="Concatenate chords TSVs (<chord> tags group notes in the same voice with identical onset and duration) including "
                                     "lyrics, dynamics, articulation, staff- and system texts, tempo marking, spanners, and thoroughbass figures, "
                                     "for all selected pieces (use ms3 extract -C to create those).")
    transform_parser.add_argument('-D', '--metadata', action='store_true',
                                help="Output 'concatenated_metadata.tsv' with one row per selected piece.")
    transform_parser.add_argument('-s', '--suffix', nargs='*', metavar='SUFFIX',
                                help="Pass -s to use standard suffixes or -s SUFFIX to choose your own. In the latter case they will be assigned to the extracted aspects in the order "
                                     "in which they are listed above (capital letter arguments).")
    transform_parser.add_argument('-u', '--unfold', action='store_true',
                                help="Unfold the repeats for all concatenated DataFrames.")
    transform_parser.add_argument('--interval_index', action='store_true',
                              help="Prepend a column with [start, end) intervals to the TSV files.")
    transform_parser.set_defaults(func=transform)



    update_parser = subparsers.add_parser('update',
                                           help="Convert MSCX files to the latest MuseScore version and move all chord annotations "
                                                "to the Roman Numeral Analysis layer. This command overwrites existing files!!!",
                                           parents=[parse_args])
    # update_parser.add_argument('-a', '--annotations', metavar='PATH', default='../harmonies',
    #                             help='Path relative to the score file(s) where to look for existing annotation tables.')
    update_parser.add_argument('-s', '--suffix', metavar='SUFFIX', help='Add this suffix to the filename of every new file.')
    update_parser.add_argument('--above', action='store_true', help="Display Roman Numerals above the system.")
    update_parser.add_argument('--safe', action='store_true', help="Only moves labels if their temporal positions stay intact.")
    update_parser.add_argument('--staff', default=-1, help="Which staff you want to move the annotations to. 1=upper staff; -1=lowest staff (default)")
    update_parser.add_argument('--type', default=1, help="defaults to 1, i.e. moves labels to Roman Numeral layer. Other types have not been tested!")
    update_parser.set_defaults(func=update_cmd)



    return parser


def run():
    parser = get_arg_parser()
    args = parser.parse_args()
    if 'func' not in args:
        parser.print_help()
        return
    if args.files is not None:
        args.files = [resolve_dir(path) for path in args.file]
    args.func(args)





if __name__ == "__main__":
    run()
