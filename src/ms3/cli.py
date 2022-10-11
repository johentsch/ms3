#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command line interface for ms3.
"""

import argparse, os, sys

import pandas as pd

from ms3 import Score, Parse
from ms3.operations import extract, check, compare
from ms3.utils import assert_dfs_equal, convert, convert_folder, get_musescore, resolve_dir, scan_directory, write_tsv
from ms3.logger import get_logger

__author__ = "johentsch"
__copyright__ = "Êcole Polytechnique Fédérale de Lausanne"
__license__ = "gpl3"

def gather_extract_params(args):
    params = [name for name, arg in zip(
        ('measures', 'notes', 'rests', 'labels', 'expanded', 'events', 'chords', 'metadata', 'form_labels'),
        (args.measures, args.notes, args.rests, args.labels, args.expanded, args.events, args.chords, args.metadata, args.form_labels))
              if arg is not None]
    return params


def make_parse_obj(args):
    labels_cfg = {}
    if hasattr(args, 'positioning'):
        labels_cfg['positioning'] = args.positioning
    if hasattr(args, 'raw'):
        labels_cfg['decode'] = args.raw

    logger_cfg = {
        'level': args.level,
        'path': args.log,
    }
    simulate = args.test if hasattr(args, 'test') else False
    ms = args.musescore if hasattr(args, 'musescore') else None
    return Parse(args.dir, paths=args.file, file_re=args.regex, exclude_re=args.exclude, recursive=args.nonrecursive, labels_cfg=labels_cfg,
              logger_cfg=logger_cfg, simulate=simulate, ms=ms)


def make_suffixes(args):
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
    if "metadata_suffix" in suffixes:
        del (suffixes["metadata_suffix"])
    return suffixes


def add(args):
    logger_cfg = {
        'level': args.level,
        'path': args.log,
    }
    p = Parse(args.dir, paths=args.file, file_re=args.regex, exclude_re=args.exclude,
              recursive=args.nonrecursive, logger_cfg=logger_cfg)
    p.parse(parallel=False)
    if args.replace:
        p.detach_labels()
        p.logger.info(
            f"Overview of the removed labels:\n{p.count_annotation_layers(which='detached').to_string()}")
    p.add_labels(use=args.use)
    ids = [id for id, score in p._parsed_mscx.items() if score.mscx.changed]
    if args.out is not None:
        p.output_mscx(ids=ids, root_dir=args.out, overwrite=True)
    else:
        p.output_mscx(ids=ids, overwrite=True)


def check_cmd(args, parse_obj=None):
    if parse_obj is None:
        args.raw = True
        if args.regex is None:
            args.regex = r'\.mscx$'
        p = make_parse_obj(args)
    else:
        p = parse_obj

    _ = check(p,
                 scores_only=args.scores_only,
                 labels_only=args.labels_only,
                 assertion=args.assertion)
    return p


def compare_cmd(args, parse_obj=None):
    if parse_obj is None:
        args.raw = True
        if args.regex is None:
            args.regex = r'\.mscx$'
        p = make_parse_obj(args)
    else:
        p = parse_obj
    comparison_results = compare(p, use=args.use, revision_specifier=args.commit, flip=args.flip)
    modified_ids = [id for id, res in comparison_results.items() if res > (0, 0)]
    if args.out is None:
        folder_args = {}
    elif os.path.isabs(args.out):
        folder_args = dict(root_dir=args.out)
    else:
        folder_args = dict(folder=args.out)
    p.output_mscx(ids=modified_ids, suffix=args.suffix, overwrite=args.safe, **folder_args)



def convert_cmd(args):
    # assert target[:len(
    #    dir)] != dir, "TARGET_DIR cannot be identical with nor a subfolder of DIR.\nDIR:        " + dir + '\nTARGET_DIR: ' + target
    update_logger = get_logger("ms3.convert", level=args.level)
    out_dir = os.getcwd() if args.out is None else resolve_dir(args.out)
    convert_folder(directory=resolve_dir(args.dir),
                   paths=args.file,
                   target_dir=out_dir,
                   # extensions=args.extensions,
                   target_extension=args.target_format,
                   regex=args.regex,
                   suffix=args.suffix,
                   recursive=args.nonrecursive,
                   ms=args.musescore,
                   overwrite=args.safe,
                   parallel=args.nonparallel,
                   logger=update_logger)

def empty(args):
    logger_cfg = {
        'level': args.level,
        'path': args.log,
    }
    p = Parse(args.dir, paths=args.file, file_re=args.regex, exclude_re=args.exclude,
              recursive=args.nonrecursive, logger_cfg=logger_cfg)
    p.parse_mscx(parallel=False)
    p.detach_labels()
    p.logger.info(f"Overview of the removed labels:\n{p.count_annotation_layers(which='detached').to_string()}")
    ids = [id for id, score in p._parsed_mscx.items() if score.mscx.changed]
    if args.out is not None:
        p.output_mscx(ids=ids, root_dir=args.out, overwrite=True)
    else:
        p.output_mscx(ids=ids, overwrite=True)


def extract_cmd(args, parse_obj=None):
    if parse_obj is None:
        p = make_parse_obj(args)
    else:
        p = parse_obj
    params = gather_extract_params(args)
    if len(params) == 0:
        print("In order to extract DataFrames, pass at least one of the following arguments: -M (measures), -N (notes), -R (rests), -L (labels), -X (expanded), -E (events), -C (chords), -D (metadata) -F (form_labels)")
        return
    suffixes = make_suffixes(args)
    silence_label_warnings = args.silence_label_warnings if hasattr(args, 'silence_label_warnings') else False
    extract(p, root_dir=args.out,
                notes_folder=args.notes,
                labels_folder=args.labels,
                measures_folder=args.measures,
                rests_folder=args.rests,
                events_folder=args.events,
                chords_folder=args.chords,
                expanded_folder=args.expanded,
                form_labels_folder=args.form_labels,
                metadata_path=resolve_dir(args.metadata),
                simulate=args.test,
                unfold=args.unfold,
                quarterbeats=args.quarterbeats,
                silence_label_warnings=silence_label_warnings,
                **suffixes)

def metadata(args):
    """ Update MSCX files with changes made in metadata.tsv (created via ms3 extract -D). In particular,
        add the values from (new?) columns to the corresponding fields in the MuseScore files' "Score info".
    """
    logger_cfg = {
        'level': args.level,
        'path': args.log,
    }

    regex = r'(metadata\.tsv|\.mscx)$' if args.regex == '(\.mscx|\.mscz|\.tsv)$' else args.regex

    p = Parse(args.dir, paths=args.file, file_re=regex, exclude_re=args.exclude, recursive=args.nonrecursive,
              logger_cfg=logger_cfg)
    if not any('metadata' in fnames for fnames in p.fnames.values()):
        p.logger.info("metadata.tsv not found.")
        return
    p.parse(parallel=False)
    if len(p._metadata) == 0:
        p.logger.info("No suitable metadata recognized.")
        return
    ids = p.update_metadata() # Writes info to parsed MuseScore files
    if len(ids) == 0:
        p.logger.debug("Nothing to update.")
        return
    if args.out is not None:
        p.output_mscx(ids=ids, root_dir=args.out, overwrite=True)
    else:
        p.output_mscx(ids=ids, overwrite=True)
    if args.out is not None:
        p.output_dataframes(metadata_path=args.out)
    elif args.dir is not None:
        p.output_dataframes(metadata_path=args.dir)


def repair(args):
    print("Sorry, the command has not been implemented yet.")
    print(args.dir)


def transform(args):
    if args.out is None:
        args.out = os.getcwd()
    params = [name for name, arg in zip(
        ('measures', 'notes', 'rests', 'labels', 'expanded', 'events', 'chords', 'metadata'),
        (args.measures, args.notes, args.rests, args.labels, args.expanded, args.events, args.chords, args.metadata))
              if arg]
    if len(params) == 0:
        print(
            "Pass at least one of the following arguments: -M (measures), -N (notes), -R (rests), -L (labels), -X (expanded), -E (events), -C (chords), -D (metadata)")
        return
    if args.suffix is None:
        suffixes = {f"{p}_suffix": '' for p in params}
    else:
        l_suff = len(args.suffix)
        if l_suff == 0:
            suffixes = {f"{p}_suffix": f"_{p}" for p in params}
        elif l_suff == 1:
            suffixes = {f"{p}_suffix": args.suffix[0] for p in params}
        else:
            suffixes = {f"{p}_suffix": args.suffix[i] if i < l_suff else f"_{p}" for i, p in enumerate(params)}

    logger_cfg = {
        'level': args.level,
        'path': args.log,
    }

    p = Parse(args.dir, paths=args.file, file_re=args.regex, exclude_re=args.exclude, recursive=args.nonrecursive,
              logger_cfg=logger_cfg, simulate=args.test)
    p.parse_tsv()
    for param in params:
        if param == 'metadata':
            continue
        sfx = suffixes[f"{param}_suffix"]
        tsv_name = f"concatenated_{param}{sfx}.tsv"
        path = os.path.join(args.out, tsv_name)
        if args.test:
            p.logger.info(f"Would have written {path}.")
        else:
            df = p.__getattribute__(param)(quarterbeats=args.quarterbeats, unfold=args.unfold)
            df = df.reset_index(drop=False)
            write_tsv(df, path)
            p.logger.info(f"{path} written.")


def update(args):
    update_logger = get_logger("ms3.update", level=args.level)
    MS = get_musescore(args.musescore, logger=update_logger)
    assert MS is not None, f"MuseScore not found: {ms}"
    logger_cfg = {
        'level': args.level,
        'path': args.log,
    }
    if args.dir is None:
        paths = args.file
    else:
        paths = scan_directory(args.dir, file_re=args.regex, exclude_re=args.exclude, recursive=args.nonrecursive,
                               subdirs=False,
                               exclude_files_only=True,
                               logger=update_logger)

    for old in paths:
        path, name = os.path.split(old)
        fname, fext = os.path.splitext(name)
        if fext not in ('.mscx', '.mscz'):
            continue
        logger_cfg = dict(logger_cfg, name=f"ms3.{fname}")
        if args.suffix is not None:
            fname = f"{fname}{args.suffix}.mscx"
        else:
            fname = fname + '.mscx'
        if args.out is None:
            new = os.path.join(path, fname)
        else:
            new = os.path.join(args.out, fname)
        convert(old, new, MS, logger=name)
        s = Score(new, logger_cfg=logger_cfg)
        if s.mscx.has_annotations:
            s.mscx.style['romanNumeralPlacement'] = 0 if args.above else 1
            before = s.annotations.df
            harmony_layers = before.harmony_layer.astype(str).str[0].unique()
            if len(harmony_layers) > 1 or harmony_layers[0] != str(args.type):
                # If all labels have the target type already, nothing is changed, even if the staves don't meet the
                # target staff: For that one would have to transform the default target -1 into the last staff number
                s.detach_labels('old')
                if 'old' not in s._detached_annotations:
                    continue
                s.old.remove_initial_dots()
                s.attach_labels('old', staff=int(args.staff), voice=1, harmony_layer=int(args.type))
                if args.safe:
                    after = s.annotations.df
                    try:
                        assert_dfs_equal(before, after, exclude=['staff', 'voice', 'label', 'harmony_layer'])
                        s.output_mscx(new)
                    except:
                        s.logger.error(f"File was not updated because of the following error:\n{sys.exc_info()[1]}")
                        continue
                else:
                    s.output_mscx(new)
            else:
                s.logger.info(f"All labels are already of type {harmony_layers[0]}; no labels changed")
                s.output_mscx(new)
        else:
            s.logger.debug(f"File has no labels to update.")
            s.output_mscx(new)

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




def review_cmd(args, parse_obj=None):
    review_logger = get_logger('ms3.review')
    if parse_obj is None:
        args.raw = True
        if args.regex is None:
            args.regex = r'\.mscx$'
        p = make_parse_obj(args)
    else:
        p = parse_obj
    test_passes = True
    scores_ok = check(p, scores_only=True, parallel=False)
    if not args.ignore_score_warnings:
        test_passes = scores_ok
    labels_ok = check(p, labels_only=True)
    test_passes = test_passes and labels_ok
    extract_cmd(args, p)
    review_report = p.color_non_chord_tones()
    filtered_report = pd.concat(review_report.values(), keys=[(key, p.files[key][i]) for (key, i) in review_report.keys()])
    #filtered_report.to_csv('/home/hentsche/unittest_metacorpus/sweelinck_keyboard/MS3/report.tsv', sep='\t')
    threshold = 0.5
    print(filtered_report[filtered_report.dur_ratio > threshold])
    comparison = compare(p, use=args.use, revision_specifier=args.commit)
    modified_ids = [id for id, score in p._parsed_mscx.items() if score.mscx.changed]
    p.output_mscx(ids=modified_ids, folder="../reviewed", suffix='_reviewed', overwrite=args.safe, root_dir=args.out)
    if test_passes:
        review_logger.info(f"Parsed scores passed all tests.")
    else:
        msg = "Not all tests have passed."
        if args.assertion:
            assert test_passes, msg
        else:
            review_logger.info(msg)







def get_arg_parser():
    # reusable argument sets
    input_args = argparse.ArgumentParser(add_help=False)
    input_args.add_argument('-d', '--dir', metavar='DIR', nargs='+', type=check_dir,
                                help='Folder(s) that will be scanned for input files. Defaults to current working directory if no individual files are passed via -f.')
    input_args.add_argument('-n', '--nonrecursive', action='store_false',
                            help="Don't scan folders recursively, i.e. parse only files in DIR.")
    input_args.add_argument('-f', '--file', metavar='PATHs', nargs='+',
                            help='Add path(s) of individual file(s) to be checked.')
    input_args.add_argument('-o', '--out', metavar='OUT_DIR', type=check_and_create,
                                help="""Output directory. Subfolder trees are retained.""")
    input_args.add_argument('-r', '--regex', metavar="REGEX", default=r'(\.mscx|\.mscz|\.tsv)$',
                                help="Select only file names including this string or regular expression. Defaults to MSCX, MSCZ and TSV files only.")
    input_args.add_argument('-e', '--exclude', metavar="REGEX",
                                help="Any files or folders (and their subfolders) including this regex will be disregarded."
                                     "By default, files including '_reviewed' or starting with . or _ or 'concatenated' are excluded.")
    input_args.add_argument('-l', '--level', metavar='{c, e, w, i, d}', default='i',
                                help="Choose how many log messages you want to see: c (none), e, w, i, d (maximum)")
    input_args.add_argument('--log', nargs='?', const='.', help='Can be a file path or directory path. Relative paths are interpreted relative to the current directory.')

    # main argument parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='''\
--------------------------
| Welcome to ms3 parsing |
--------------------------

The library offers you the following commands. Add the flag -h to one of them to learn about its parameters. 
''')
    subparsers = parser.add_subparsers(help='The action that you want to perform.', dest='action')



    add_parser = subparsers.add_parser('add',
                                         help="Add labels from annotation tables to scores.",
                                         parents=[input_args])
    # TODO: staff parameter needs to accept one to several integers including negative
    # add_parser.add_argument('-s', '--staff',
    #                            help="Remove labels from selected staves only. 1=upper staff; -1=lowest staff (default)")
    # add_parser.add_argument('--type', default=1,
    #                            help="Only remove particular types of harmony labels.")
    add_parser.add_argument('--replace', action='store_true',
                               help="Remove existing labels from the scores prior to adding. Like calling ms3 empty first.")
    add_parser.add_argument('--use',
                            help="""In case there are several annotation labels present, set this value to expanded or to labels. 
If you don't, you will be asked for every ambiguous corpus to specify in which order the columns that 
show detected files are to be used.""")
    add_parser.set_defaults(func=add)



    check_parser = subparsers.add_parser('check', help="""Parse MSCX files and look for errors.
In particular, check DCML harmony labels for syntactic correctness.""", parents=[input_args])
    check_parser.add_argument('--labels_only', action='store_true',
                              help="Don't check scores for encoding errors.")
    check_parser.add_argument('--scores_only', action='store_true',
                              help="Don't check DCML labels for syntactic correctness.")
    check_parser.add_argument('--assertion', action='store_true', help="If you pass this argument, an error will be thrown if there are any mistakes.")
    check_parser.set_defaults(func=check_cmd)



    compare_parser = subparsers.add_parser('compare',
        help="For MSCX files for which annotation tables exist, create another MSCX file with a coloured label comparison.",
        parents = [input_args])
    compare_parser.add_argument('--use', nargs='?', const='any', metavar="{labels, expanded}",
                                help="""By default, if several sets of annotation files are found, the user is asked which one(s) to use.
To prevent the interaction, set this flag to use the first annotation table that comes along for every score. Alternatively, you can add the string
'expanded' or 'labels' to use only annotation tables that have the respective type.""")

    compare_parser.add_argument('-s', '--suffix', metavar='SUFFIX', default='_reviewed',
                                help='Suffix of the newly created comparison files. Defaults to _reviewed')
    compare_parser.add_argument('-c', '--commit', metavar='SPECIFIER',
                                help="If you want to compare labels against a TSV file from a particular git revision, pass its SHA (short or long), tag, branch name, or relative specifier such as 'HEAD~1'.")
    compare_parser.add_argument('--flip', action='store_true',
                                help="Pass this flag to treat the annotation tables as if updating the scores instead of the other way around, "
                                     "effectively resulting in a swap of the colors in the output files.")
    compare_parser.add_argument('--safe', action='store_false', help="Don't overwrite existing files.")
    compare_parser.set_defaults(func=compare_cmd)



    convert_parser = subparsers.add_parser('convert',
                                           help="Use your local install of MuseScore to convert MuseScore files.",
                                           parents=[input_args])
    # convert_parser.add_argument('-x', '--extensions', nargs='+', default=['mscx', 'mscz'],
    #                             help="List, separated by spaces, the file extensions that you want to convert. Defaults to mscx mscz")
    convert_parser.add_argument('-t', '--target_format', default='mscx',
                                help="You may choose one out of {png, svg, pdf, mscz, mscx, wav, mp3, flac, ogg, xml, mxl, mid}")
    convert_parser.add_argument('-m', '--musescore', default='mscore', help="""Path to MuseScore executable. Defaults to the command 'mscore' (standard on *nix systems).
    To use standard paths on commercial systems, try -m win, or -m mac.""")
    convert_parser.add_argument('-p', '--nonparallel', action='store_false',
                                help="Do not use all available CPU cores in parallel to speed up batch jobs.")
    convert_parser.add_argument('-s', '--suffix', metavar='SUFFIX', help='Add this suffix to the filename of every new file.')
    convert_parser.add_argument('--safe', action='store_false', help="Don't overwrite existing files.")
    convert_parser.set_defaults(func=convert_cmd)



    empty_parser = subparsers.add_parser('empty',
                                         help="Remove harmony annotations and store the MuseScore files without them.",
                                         parents=[input_args])
    # TODO: staff parameter needs to accept one to several integers including negative
    # empty_parser.add_argument('-s', '--staff',
    #                            help="Remove labels from selected staves only. 1=upper staff; -1=lowest staff (default)")
    # empty_parser.add_argument('--type', default=1,
    #                            help="Only remove particular types of harmony labels.")
    empty_parser.set_defaults(func=empty)



    extract_parser = subparsers.add_parser('extract',
                                           help="Extract selected information from MuseScore files and store it in TSV files.",
                                           parents=[input_args])
    extract_parser.add_argument('-M', '--measures', metavar='folder', nargs='?',
                                const='../measures',
                                help="Folder where to store TSV files with measure information needed for tasks such as unfolding repetitions.")
    extract_parser.add_argument('-N', '--notes', metavar='folder', nargs='?', const='../notes',
                                help="Folder where to store TSV files with information on all notes.")
    extract_parser.add_argument('-R', '--rests', metavar='folder', nargs='?', const='../rests',
                                help="Folder where to store TSV files with information on all rests.")
    extract_parser.add_argument('-L', '--labels', metavar='folder', nargs='?', const='../labels',
                                help="Folder where to store TSV files with information on all annotation labels.")
    extract_parser.add_argument('-X', '--expanded', metavar='folder', nargs='?', const='../harmonies',
                                help="Folder where to store TSV files with expanded DCML labels.")
    extract_parser.add_argument('-E', '--events', metavar='folder', nargs='?', const='../events',
                                help="Folder where to store TSV files with all events (notes, rests, articulation, etc.) without further processing.")
    extract_parser.add_argument('-C', '--chords', metavar='folder', nargs='?', const='../chords',
                                help="Folder where to store TSV files with <chord> tags, i.e. groups of notes in the same voice with identical onset and duration. The tables include lyrics, slurs, and other markup.")
    extract_parser.add_argument('-D', '--metadata', metavar='path', nargs='?', const='.',
                                help="Directory or full path for storing one TSV file with metadata. If no filename is included in the path, it is called metadata.tsv")
    extract_parser.add_argument('-F', '--form_labels', metavar='folder', nargs='?', const='../form_labels',
                                help="Folder where to store TSV files with all form labels.")
    extract_parser.add_argument('-s', '--suffix', nargs='*', metavar='SUFFIX',
                                help="Pass -s to use standard suffixes or -s SUFFIX to choose your own. In the latter case they will be assigned to the extracted aspects in the order "
                                     "in which they are listed above (capital letter arguments).")
    extract_parser.add_argument('-m', '--musescore', default='auto', help="""Command or path of MuseScore executable. Defaults to 'auto' (attempt to use standard path for your system).
    Other standard options are -m win, -m mac, and -m mscore (for Linux).""")
    extract_parser.add_argument('-t', '--test', action='store_true',
                                help="No data is written to disk.")
    extract_parser.add_argument('-p', '--positioning', action='store_true',
                                help="When extracting labels, include manually shifted position coordinates in order to restore them when re-inserting.")
    extract_parser.add_argument('--raw', action='store_false',
                                help="When extracting labels, leave chord symbols encoded instead of turning them into a single column of strings.")
    extract_parser.add_argument('-u', '--unfold', action='store_true',
                                help="Unfold the repeats for all stored DataFrames.")
    extract_parser.add_argument('-q', '--quarterbeats', action='store_true',
                                help="Add a column with continuous quarterbeat positions. If a score has first and second endings, the behaviour depends on "
                                     "the parameter --unfold: If it is not set, repetitions are not unfolded and only last endings are included in the continuous "
                                     "positions. If repetitions are being unfolded, all endings are taken into account.")
    extract_parser.set_defaults(func=extract_cmd)



    metadata_parser = subparsers.add_parser('metadata',
                                            help="Update MSCX files with changes made in metadata.tsv (created via ms3 extract -D).",
                                            parents=[input_args])
    metadata_parser.set_defaults(func=metadata)

    repair_parser = subparsers.add_parser('repair',
                                          help="Apply automatic repairs to your uncompressed MuseScore files.",
                                          parents=[input_args])
    repair_parser.set_defaults(func=repair)



    transform_parser = subparsers.add_parser('transform',
                                          help="Concatenate and transform TSV data from one or several corpora.",
                                          parents=[input_args])
    transform_parser.add_argument('-M', '--measures', action='store_true',
                                help="Folder where to store TSV files with measure information needed for tasks such as unfolding repetitions.")
    transform_parser.add_argument('-N', '--notes', action='store_true',
                                help="Folder where to store TSV files with information on all notes.")
    transform_parser.add_argument('-R', '--rests', action='store_true',
                                help="Folder where to store TSV files with information on all rests.")
    transform_parser.add_argument('-L', '--labels', action='store_true',
                                help="Folder where to store TSV files with information on all annotation labels.")
    transform_parser.add_argument('-X', '--expanded', action='store_true',
                                help="Folder where to store TSV files with expanded DCML labels.")
    transform_parser.add_argument('-E', '--events', action='store_true',
                                help="Folder where to store TSV files with all events (notes, rests, articulation, etc.) without further processing.")
    transform_parser.add_argument('-C', '--chords', action='store_true',
                                help="Folder where to store TSV files with <chord> tags, i.e. groups of notes in the same voice with identical onset and duration. The tables include lyrics, slurs, and other markup.")
    transform_parser.add_argument('-D', '--metadata', action='store_true',
                                help="Directory or full path for storing one TSV file with metadata. If no filename is included in the path, it is called metadata.tsv")
    transform_parser.add_argument('-s', '--suffix', nargs='*', metavar='SUFFIX',
                                help="Pass -s to use standard suffixes or -s SUFFIX to choose your own. In the latter case they will be assigned to the extracted aspects in the order "
                                     "in which they are listed above (capital letter arguments).")
    transform_parser.add_argument('-t', '--test', action='store_true', help="No data is written to disk.")
    transform_parser.add_argument('-u', '--unfold', action='store_true',
                                help="Unfold the repeats for all stored DataFrames.")
    transform_parser.add_argument('-q', '--quarterbeats', action='store_true',
                                help="Add a column with continuous quarterbeat positions. If a score has first and second endings, the behaviour depends on "
                                     "the parameter --unfold: If it is not set, repetitions are not unfolded and only last endings are included in the continuous "
                                     "positions. If repetitions are being unfolded, all endings are taken into account.")
    transform_parser.set_defaults(func=transform)



    update_parser = subparsers.add_parser('update',
                                           help="Convert MSCX files to the latest MuseScore version and move all chord annotations "
                                                "to the Roman Numeral Analysis layer. This command overwrites existing files!!!",
                                           parents=[input_args])
    # update_parser.add_argument('-a', '--annotations', metavar='PATH', default='../harmonies',
    #                             help='Path relative to the score file(s) where to look for existing annotation tables.')
    update_parser.add_argument('-s', '--suffix', metavar='SUFFIX', help='Add this suffix to the filename of every new file.')
    update_parser.add_argument('-m', '--musescore', default='mscore', help="""Path to MuseScore executable. Defaults to the command 'mscore' (standard on *nix systems).
        To try standard paths on commercial systems, try -m win, or -m mac.""")
    update_parser.add_argument('--above', action='store_true', help="Display Roman Numerals above the system.")
    update_parser.add_argument('--safe', action='store_true', help="Only moves labels if their temporal positions stay intact.")
    update_parser.add_argument('--staff', default=-1, help="Which staff you want to move the annotations to. 1=upper staff; -1=lowest staff (default)")
    update_parser.add_argument('--type', default=1, help="defaults to 1, i.e. moves labels to Roman Numeral layer. Other types have not been tested!")
    update_parser.set_defaults(func=update)

    review_parser = subparsers.add_parser('review',
                                         help="Extract facets, check labels, and create _reviewed files.",
                                         parents=[input_args])
    review_parser.add_argument('-M', '--measures', metavar='folder', nargs='?',
                                const='../measures',
                                help="Folder where to store TSV files with measure information needed for tasks such as unfolding repetitions.")
    review_parser.add_argument('-N', '--notes', metavar='folder', nargs='?', const='../notes',
                                help="Folder where to store TSV files with information on all notes.")
    review_parser.add_argument('-R', '--rests', metavar='folder', nargs='?', const='../rests',
                                help="Folder where to store TSV files with information on all rests.")
    review_parser.add_argument('-L', '--labels', metavar='folder', nargs='?',
                                const='../labels',
                                help="Folder where to store TSV files with information on all annotation labels.")
    review_parser.add_argument('-X', '--expanded', metavar='folder', nargs='?',
                                const='../harmonies',
                                help="Folder where to store TSV files with expanded DCML labels.")
    review_parser.add_argument('-E', '--events', metavar='folder', nargs='?', const='../events',
                                help="Folder where to store TSV files with all events (notes, rests, articulation, etc.) without further processing.")
    review_parser.add_argument('-C', '--chords', metavar='folder', nargs='?',
                                const='../chord_events',
                                help="Folder where to store TSV files with <chord> tags, i.e. groups of notes in the same voice with identical onset and duration. The tables include lyrics, slurs, and other markup.")
    review_parser.add_argument('-D', '--metadata', metavar='path', nargs='?', const='.',
                                help="Directory or full path for storing one TSV file with metadata. If no filename is included in the path, it is called metadata.tsv")
    review_parser.add_argument('-s', '--suffix', nargs='*', metavar='SUFFIX',
                                help="Pass -s to use standard suffixes or -s SUFFIX to choose your own. In the latter case they will be assigned to the extracted aspects in the order "
                                     "in which they are listed above (capital letter arguments).")
    review_parser.add_argument('-m', '--musescore', default='auto', help="""Command or path of MuseScore executable. Defaults to 'auto' (attempt to use standard path for your system).
        Other standard options are -m win, -m mac, and -m mscore (for Linux).""")
    review_parser.add_argument('-i', '--ignore_score_warnings', action='store_true',
                                help="By default, tests also fail upon erroneous score encodings. Pass -i to prevent this.")
    review_parser.add_argument('-t', '--test', action='store_true',
                                help="No data is written to disk.")
    review_parser.add_argument('-p', '--positioning', action='store_true',
                                help="When extracting labels, include manually shifted position coordinates in order to restore them when re-inserting.")
    review_parser.add_argument('--raw', action='store_false',
                                help="When extracting labels, leave chord symbols encoded instead of turning them into a single column of strings.")
    review_parser.add_argument('-u', '--unfold', action='store_true',
                                help="Unfold the repeats for all stored DataFrames.")
    review_parser.add_argument('-q', '--quarterbeats', action='store_true',
                                help="Add a column with continuous quarterbeat positions. If a score has first and second endings, the behaviour depends on "
                                     "the parameter --unfold: If it is not set, repetitions are not unfolded and only last endings are included in the continuous "
                                     "positions. If repetitions are being unfolded, all endings are taken into account.")
    review_parser.add_argument('-c', '--commit', metavar='SPECIFIER',
                                help="If you want to compare labels against a TSV file from a particular git revision, pass its SHA (short or long), tag, branch name, or relative specifier such as 'HEAD~1'.")
    review_parser.add_argument('--use', nargs='?', const='any', metavar="{labels, expanded}",
                                help="""By default, if several sets of annotation files are found, the user is asked which one(s) to use.
To prevent the interaction, set this flag to use the first annotation table that comes along for every score. Alternatively, you can add the string
'expanded' or 'labels' to use only annotation tables that have the respective type.""")

    review_parser.add_argument('--assertion', action='store_true', help="If you pass this argument, an error will be thrown if there are any mistakes.")
    review_parser.add_argument('--safe', action='store_false', help="Don't overwrite existing files.")
    review_parser.set_defaults(func=review_cmd)

    return parser


def run():
    parser = get_arg_parser()
    args = parser.parse_args()
    if 'func' not in args:
        parser.print_help()
        return
    if args.file is None:
        if args.dir is None:
            args.dir = os.getcwd()
    else:
        args.file = [resolve_dir(path) for path in args.file]
    args.func(args)





if __name__ == "__main__":
    run()
