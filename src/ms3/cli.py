# -*- coding: utf-8 -*-
"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
         fibonacci = ms3.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
"""

import argparse, os

from ms3 import Parse
from ms3.convert import convert_folder
from ms3.utils import resolve_dir

__author__ = "johentsch"
__copyright__ = "Êcole Polytechnique Fédérale de Lausanne"
__license__ = "gpl3"





def extract(args):
    labels_cfg = {
        'positioning': args.positioning,
        'decode': args.raw,
    }
    if sum([True for arg in [args.notes, args.labels, args.measures, args.rests, args.events, args.chords, args.expanded] if arg is not None]) == 0:
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
    p = Parse(args.mscx_dir, file_re=args.file, exclude_re=args.exclude, recursive=args.nonrecursive, labels_cfg=labels_cfg, logger_cfg=logger_cfg, simulate=args.test)
    p.parse_mscx(simulate=args.test)
    p.store_lists(root_dir=args.out,
                  notes_folder=args.notes,
                  labels_folder=args.labels,
                  measures_folder=args.measures,
                  rests_folder=args.rests,
                  events_folder=args.events,
                  chords_folder=args.chords,
                  expanded_folder=args.expanded,
                  simulate=args.test,
                  **suffixes)


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


def convert(args):
    dir, target = resolve_dir(args.dir), resolve_dir(args.target)
    #assert target[:len(
    #    dir)] != dir, "TARGET_DIR cannot be identical with nor a subfolder of DIR.\nDIR:        " + dir + '\nTARGET_DIR: ' + target
    convert_folder(dir, target,
                   extensions=args.extensions,
                   target_extension=args.format,
                   regex=args.regex,
                   suffix=args.suffix,
                   recursive=args.nonrecursive,
                   MS=args.musescore,
                   overwrite=args.overwrite,
                   parallel=args.nonparallel)

def run():
    """Entry point for console_scripts
    """
    parser = argparse.ArgumentParser(
        prog='ms3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
--------------------------
| Welcome to ms3 Parsing |
--------------------------

The library offers you the following commands. Add the flag -h to one of them to learn about its parameters. 
''')


    subparsers = parser.add_subparsers(help='The action that you want to perform.')

    extract_parser = subparsers.add_parser('extract', help="Extract selected information from MuseScore files and store it in TSV files.")
    extract_parser.add_argument('mscx_dir', metavar='MSCX_DIR', nargs='?', type=check_dir,
                                default=os.getcwd(),
                                help='Folder that will be scanned for MuseScore files (.mscx).')
    extract_parser.add_argument('-N', '--notes', metavar='folder', help="Folder where to store TSV files with notes.")
    extract_parser.add_argument('-L', '--labels', metavar='folder', help="Folder where to store TSV files with annotation labels.")
    extract_parser.add_argument('-M', '--measures', metavar='folder', help="Folder where to store TSV files with measure information.")
    extract_parser.add_argument('-R', '--rests', metavar='folder', help="Folder where to store TSV files with rests.")
    extract_parser.add_argument('-E', '--events', metavar='folder', help="Folder where to store TSV files with events (notes, rests, articulation, etc.).")
    extract_parser.add_argument('-C', '--chords', metavar='folder', help="Folder where to store TSV files with chords, including lyrics, slurs, and other markup.")
    extract_parser.add_argument('-X', '--expanded', metavar='folder', help="Folder where to store TSV files with expanded DCML labels.")
    extract_parser.add_argument('-s', '--suffix', nargs='*',  metavar='SUFFIX',
                        help="Pass -s to use standard suffixes or -s SUFFIX to choose your own.")
    extract_parser.add_argument('-o', '--out', metavar='ROOT_DIR', type=check_and_create,
                                help="""Make all relative folder paths relative to ROOT_DIR rather than to MSCX_DIR.
This setting has no effect on absolute folder paths.""")
    extract_parser.add_argument('-f', '--file', metavar="regex", default=r'\.mscx$',
                                help="Select only file names including this regular expression.")
    extract_parser.add_argument('-e', '--exclude', metavar="regex", default=r'^(\.|_)',
                                help="Any files or folders (and their subfolders) including this regex will be disregarded.")
    extract_parser.add_argument('-l', '--level', metavar='LOG_LEVEL', default='i',
                                help="Choose how many log messages you want to see: d (maximum), i, w, e, c (none)")
    extract_parser.add_argument('-t', '--test', action='store_true', help="No data is written to disk.")
    extract_parser.add_argument('-p', '--positioning', action='store_true', help="When extracting labels, include manually shifted position coordinates in order to restore them when re-inserting.")
    extract_parser.add_argument('-r', '--raw', action='store_false', help="When extracting labels, leave chord symbols encoded instead of turning them into strings.")
    extract_parser.add_argument('-n', '--nonrecursive', action='store_false', help="Don't scan folders recursively, i.e. parse only files in MSCX_DIR.")
    extract_parser.add_argument('--logfile', metavar='file path or file name', help="""Either pass an absolute file path to store all logging data in that particular file
or pass just a file name and the argument --logpath to create several log files of the same name in a replicated folder structure.
In the former case, --logpath will be disregarded.""")
    extract_parser.add_argument('--logpath', type=check_and_create, help="""If you define a path for storing log files, the original folder structure of the parsed
MuseScore files is recreated there. Additionally, you can pass a filename to --logfile to combine logging data for each 
subdirectory; otherwise, an individual log file is automatically created for each MuseScore file.""")
    extract_parser.set_defaults(func=extract)

    check_parser = subparsers.add_parser('check', help="Check DCML harmony labels.")
    check_parser.add_argument('root_dir', metavar='MSCX_DIR', nargs='?', type=check_dir,
                              default=os.getcwd(),
                              help='Folder that will be scanned for MuseScore files (.mscx).')

    convert_parser = subparsers.add_parser('convert',
                                           help="Use your local install of MuseScore to convert MuseScore files.")
    convert_parser.add_argument('dir', metavar='DIR', type=check_dir,
                                help='path to folder with files to convert; can be relative to the folder where the script is located.')
    convert_parser.add_argument('target', metavar='TARGET_DIR', nargs='?', type=check_and_create, default=os.getcwd(),
                                help='Path to folder for converted files. Defaults to current working directory.')
    convert_parser.add_argument('-e', '--extensions', nargs='+', default=['mscx', 'mscz'],
                                help="List, separated by spaces, the file extensions that you want to convert. Defaults to mscx mscz")
    convert_parser.add_argument('-f', '--format', default='mscx',
                                help="You may choose one out of {png, svg, pdf, mscz, mscx, wav, mp3, flac, ogg, xml, mxl, mid}")
    convert_parser.add_argument('-m', '--musescore', default='mscore', help="""Path to MuseScore executable. Defaults to the command 'mscore' (standard on *nix systems).
    To use standard paths on commercial systems, try -m win, or -m mac.""")
    convert_parser.add_argument('-r', '--regex', default=r'.*',
                                help="Convert only files containing this regular expression (or a simple search string).")
    convert_parser.add_argument('-n', '--nonrecursive', action='store_false',
                                help="Don't scan folders recursively, i.e. parse only files in DIR.")
    convert_parser.add_argument('-o', '--overwrite', action='store_true',
                                help="Set true if existing files are to be overwritten.")
    convert_parser.add_argument('-p', '--nonparallel', action='store_false',
                                help="Do not use all available CPU cores in parallel to speed up batch jobs.")
    convert_parser.add_argument('-s', '--suffix', metavar='SUFFIX', help='Add this suffix to the filename of every new file.')
    convert_parser.set_defaults(func=convert)

    args = parser.parse_args()
    if 'func' in args:
        args.func(args)
    else:
        parser.print_help()




if __name__ == "__main__":
    run()
