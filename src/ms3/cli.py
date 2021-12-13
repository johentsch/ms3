#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command line interface for ms3.
"""

import argparse, os, sys

from ms3 import Score, Parse
from ms3.utils import assert_dfs_equal, convert, convert_folder, get_musescore, resolve_dir, scan_directory

__author__ = "johentsch"
__copyright__ = "Êcole Polytechnique Fédérale de Lausanne"
__license__ = "gpl3"



def check(args):
    labels_cfg = {'decode': True}
    log = args.log
    if log is not None:
        log = os.path.expanduser(log)
        if not os.path.isabs(log):
            log = os.path.join(os.getcwd(), log)
    logger_cfg = {
        'level': args.level,
        'file': log,
    }
    if args.regex is None:
        args.regex = r'\.mscx$'
    p = Parse(args.dir, paths=args.file, file_re=args.regex, exclude_re=args.exclude, recursive=args.nonrecursive,
              labels_cfg=labels_cfg, logger_cfg=logger_cfg)
    if '.mscx' not in p.count_extensions():
        p.logger.warning("No MSCX files to check.")
        return
    p.parse_mscx()
    res = True
    if not args.scores_only:
        wrong = p.check_labels()
        if wrong is None:
            res = None
        if len(wrong) == 0:
            p.logger.info("No syntactical errors.")
        else:
            if not args.assertion:
                p.logger.warning(f"The following labels don't match the regular expression:\n{wrong.to_string()}")
            res = False
    if args.assertion:
        assert res, "Contains syntactical errors:\n" + wrong.to_string()
    return res


def compare(args):
    logger_cfg = {
        'level': args.level,
    }
    if args.regex is None:
        args.regex = r'\.mscx$'
    p = Parse(args.dir, paths=args.file, file_re=args.regex, exclude_re=args.exclude, recursive=args.nonrecursive,
                  key='compare', logger_cfg=logger_cfg)
    if len(p._score_ids()) == 0:
        p.logger.warning(f"Your selection does not include any scores.")
        return
    p.add_rel_dir(args.annotations, suffix=args.suffix, score_extensions=args.extensions, new_key='old')
    p.parse_mscx()
    p.add_detached_annotations()
    p.compare_labels('old', store_with_suffix='_reviewed')


def convert_cmd(args):
    # assert target[:len(
    #    dir)] != dir, "TARGET_DIR cannot be identical with nor a subfolder of DIR.\nDIR:        " + dir + '\nTARGET_DIR: ' + target
    out_dir = os.getcwd() if args.out is None else resolve_dir(args.out)
    convert_folder(resolve_dir(args.dir), out_dir,
                   # extensions=args.extensions,
                   target_extension=args.target_format,
                   regex=args.regex,
                   suffix=args.suffix,
                   recursive=args.nonrecursive,
                   ms=args.musescore,
                   overwrite=args.safe,
                   parallel=args.nonparallel)


def extract(args):
    labels_cfg = {
        'positioning': args.positioning,    # default=False
        'decode': args.raw,                 # default=True
    }
    if sum([True for arg in [args.notes, args.labels, args.measures, args.rests, args.events, args.chords, args.expanded, args.metadata] if arg is not None]) == 0:
        print("Pass at least one of the following arguments: -N (notes), -L (labels), -M (measures), -R (rests), -E (events), -C (chords), -X (expanded)")
        return
    if args.suffix is not None:
        l_suff = len(args.suffix)
        params = ['notes', 'labels', 'measures', 'rests', 'events', 'chords', 'expanded']
        if l_suff == 0:
            suffixes = {f"{p}_suffix": f"_{p}" for p in params}
        elif l_suff == 1:
            suffixes = {f"{p}_suffix": args.suffix[0] for p in params}
        else:
            suffixes = {f"{p}_suffix": args.suffix[i] if i < l_suff else f"_{p}" for i, p in enumerate(params)}
    else:
        suffixes = {}

    logger_cfg = {
        'level': args.level,
        'file': args.logfile,
        'path': args.logpath,
    }

    p = Parse(args.dir, paths=args.file, file_re=args.regex, exclude_re=args.exclude, recursive=args.nonrecursive, labels_cfg=labels_cfg,
              logger_cfg=logger_cfg, simulate=args.test, ms=args.musescore)
    p.parse_mscx(simulate=args.test)
    p.store_lists(root_dir=args.out,
                  notes_folder=args.notes,
                  labels_folder=args.labels,
                  measures_folder=args.measures,
                  rests_folder=args.rests,
                  events_folder=args.events,
                  chords_folder=args.chords,
                  expanded_folder=args.expanded,
                  metadata_path=resolve_dir(args.metadata),
                  simulate=args.test,
                  unfold=args.unfold,
                  quarterbeats=args.quarterbeats,
                  **suffixes)


def metadata(args):
    """ Update MSCX files with changes made in metadata.tsv (created via ms3 extract -D). In particular,
        add the values from (new?) columns to the corresponding fields in the MuseScore files' "Score info".
    """
    logger_cfg = {
        'level': args.level,
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
        p.logger.info("Nothing to update.")
        return
    if args.out is not None:
        p.store_mscx(ids=ids, root_dir=args.out, overwrite=True)
    else:
        p.store_mscx(ids=ids, overwrite=True)
    if args.out is not None:
        p.store_lists(metadata_path=args.out)
    elif args.dir is not None:
        p.store_lists(metadata_path=args.dir)


def repair(args):
    print(args.dir)


def update(args):
    MS = get_musescore(args.musescore)
    assert MS is not None, f"MuseScore not found: {ms}"
    logger_cfg = {
        'level': args.level,
    }
    if args.dir is None:
        paths = args.file
    else:
        paths = scan_directory(args.dir, file_re=args.regex, exclude_re=args.exclude, recursive=args.nonrecursive,
                               subdirs=False,
                               exclude_files_only=True)

    for old in paths:
        path, name = os.path.split(old)
        fname, fext = os.path.splitext(name)
        if fext not in ('.mscx', '.mscz'):
            continue
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
            label_types = before.label_type.str[0].unique()
            if len(label_types) > 1 or label_types[0] != str(args.type):
                # If all labels have the target type already, nothing is changed, even if the staves don't meet the
                # target staff: For that one would have to transform the default target -1 into the last staff number
                s.detach_labels('old')
                if 'old' not in s._detached_annotations:
                    continue
                s.old.remove_initial_dots()
                s.attach_labels('old', staff=int(args.staff), voice=1,  label_type=int(args.type))
                if args.safe:
                    after = s.annotations.df
                    try:
                        assert_dfs_equal(before, after, exclude=['staff', 'voice', 'label', 'label_type'])
                        s.store_mscx(new)
                    except:
                        s.logger.error(f"File was not updated because of the following error:\n{sys.exc_info()[1]}")
                        continue
                else:
                    s.store_mscx(new)
            else:
                s.logger.info(f"All labels are already of type {label_types[0]}; no labels changed")
                s.store_mscx(new)
        else:
            s.logger.debug(f"File has no labels to update.")
            s.store_mscx(new)

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
    input_args.add_argument('-e', '--exclude', metavar="regex", default=r'(^(\.|_)|_reviewed)',
                                help="Any files or folders (and their subfolders) including this regex will be disregarded."
                                     "By default, files including '_reviewed' or starting with . or _ are excluded.")
    input_args.add_argument('-l', '--level', metavar='{c, e, w, i, d}', default='i',
                                help="Choose how many log messages you want to see: c (none), e, w, i, d (maximum)")

    # main argument parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='''\
--------------------------
| Welcome to ms3 parsing |
--------------------------

The library offers you the following commands. Add the flag -h to one of them to learn about its parameters. 
''')
    subparsers = parser.add_subparsers(help='The action that you want to perform.', dest='action')

    extract_parser = subparsers.add_parser('extract', help="Extract selected information from MuseScore files and store it in TSV files.",
                                           parents=[input_args])
    extract_parser.add_argument('-M', '--measures', metavar='folder', nargs='?', const='../measures',
                                help="Folder where to store TSV files with measure information needed for tasks such as unfolding repetitions.")
    extract_parser.add_argument('-N', '--notes', metavar='folder', nargs='?', const='../notes', help="Folder where to store TSV files with information on all notes.")
    extract_parser.add_argument('-R', '--rests', metavar='folder', nargs='?', const='../rests',
                                help="Folder where to store TSV files with information on all rests.")
    extract_parser.add_argument('-L', '--labels', metavar='folder', nargs='?', const='../annotations', help="Folder where to store TSV files with information on all annotation labels.")
    extract_parser.add_argument('-X', '--expanded', metavar='folder', nargs='?', const='../harmonies',
                                help="Folder where to store TSV files with expanded DCML labels.")
    extract_parser.add_argument('-E', '--events', metavar='folder', nargs='?', const='../events', help="Folder where to store TSV files with all events (notes, rests, articulation, etc.) without further processing.")
    extract_parser.add_argument('-C', '--chords', metavar='folder', nargs='?', const='../chord_events', help="Folder where to store TSV files with <chord> tags, i.e. groups of notes in the same voice with identical onset and duration. The tables include lyrics, slurs, and other markup.")
    extract_parser.add_argument('-D', '--metadata', metavar='path', nargs='?', const='.',
                                help="Directory or full path for storing one TSV file with metadata. If no filename is included in the path, it is called metadata.tsv")
    extract_parser.add_argument('-s', '--suffix', nargs='*',  metavar='SUFFIX',
                        help="Pass -s to use standard suffixes or -s SUFFIX to choose your own.")
    extract_parser.add_argument('-m', '--musescore', default='auto', help="""Command or path of MuseScore executable. Defaults to 'auto' (attempt to use standard path for your system).
Other standard options are -m win, -m mac, and -m mscore (for Linux).""")
    extract_parser.add_argument('-t', '--test', action='store_true', help="No data is written to disk.")
    extract_parser.add_argument('-p', '--positioning', action='store_true', help="When extracting labels, include manually shifted position coordinates in order to restore them when re-inserting.")
    extract_parser.add_argument('--raw', action='store_false', help="When extracting labels, leave chord symbols encoded instead of turning them into a single column of strings.")
    extract_parser.add_argument('-u', '--unfold', action='store_true', help="Unfold the repeats for all stored DataFrames.")
    extract_parser.add_argument('-q', '--quarterbeats', action='store_true',
                                help="Add a column with continuous quarterbeat positions. If a score as first and second endings, the behaviour depends on"
                                     "the parameter --unfold: If it is not set, repetitions are not unfolded and only last endings are included in the continuous"
                                     "count. If repetitions are being unfolded, all endings are taken into account.")
    extract_parser.add_argument('--logfile', metavar='file path or file name', help="""Either pass an absolute file path to store all logging data in that particular file
or pass just a file name and the argument --logpath to create several log files of the same name in a replicated folder structure.
In the former case, --logpath will be disregarded.""")
    extract_parser.add_argument('--logpath', type=check_and_create, nargs='?', const='.', help="""If you define a path for storing log files, the original folder structure of the parsed
MuseScore files is recreated there. Additionally, you can pass a filename to --logfile to combine logging data for each 
subdirectory; otherwise, an individual log file is automatically created for each MuseScore file. Pass without value to use current working directory.""")
    extract_parser.set_defaults(func=extract)



    check_parser = subparsers.add_parser('check', help="""Parse MSCX files and look for errors.
In particular, check DCML harmony labels for syntactic correctness.""", parents=[input_args])
    check_parser.add_argument('-s', '--scores_only', action='store_true',
                              help="Don't check DCML labels for syntactic correctness.")
    check_parser.add_argument('--assertion', action='store_true', help="If you pass this argument, an error will be thrown if there are any mistakes.")
    check_parser.add_argument('--log', metavar='NAME', help='Can be a an absolute file path or relative to the current directory.')
    check_parser.set_defaults(func=check)



    compare_parser = subparsers.add_parser('compare',
        help="For MSCX files for which annotation tables exist, create another MSCX file with a coloured label comparison.",
        parents = [input_args])
    compare_parser.add_argument('-a', '--annotations', metavar='PATH', default='../harmonies',
                                help='Path relative to the score file(s) where to look for existing annotation tables. Defaults to ../harmonies')
    compare_parser.add_argument('-s', '--suffix', metavar='SUFFIX', default='',
                                help='If existing annotation tables have a particular suffix, pass this suffix.')
    compare_parser.add_argument('-x', '--extensions', metavar='EXT', nargs='+',
                                help='If you only want to compare scores with particular extensions, pass these extensions.')
    compare_parser.set_defaults(func=compare)



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
    convert_parser.add_argument('--safe', action='store_false',
                                help="Don't overwrite existing files.")
    convert_parser.set_defaults(func=convert_cmd)



    metadata_parser = subparsers.add_parser('metadata',
                                            help="Update MSCX files with changes made in metadata.tsv (created via ms3 extract -D).",
                                            parents=[input_args])
    metadata_parser.set_defaults(func=metadata)

    repair_parser = subparsers.add_parser('repair',
                                          help="Apply automatic repairs to your uncompressed MuseScore files.",
                                          parents=[input_args])
    repair_parser.set_defaults(func=repair)



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

    return parser


def run():
    parser = get_arg_parser()
    args = parser.parse_args()
    if args.file is None and args.dir is None:
        args.dir = os.getcwd()
    if 'func' in args:
        args.func(args)
    else:
        parser.print_help()





if __name__ == "__main__":
    run()
