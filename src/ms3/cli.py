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

import argparse, os, sys, logging

from ms3 import Parse, __version__
from ms3.utils import resolve_dir

__author__ = "johentsch"
__copyright__ = "johentsch"
__license__ = "gpl3"

_logger = logging.getLogger(__name__)


def fib(n):
    """Fibonacci example function

    Args:
      n (int): integer

    Returns:
      int: n-th Fibonacci number
    """
    assert n > 0
    a, b = 1, 1
    for i in range(n-1):
        a, b = b, a+b
    return a


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Just a Fibonacci demonstration")
    parser.add_argument(
        "--version",
        action="version",
        version="ms3 {ver}".format(ver=__version__))
    parser.add_argument(
        dest="n",
        help="n-th Fibonacci number",
        type=int,
        metavar="INT")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG)
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")
    print("The {}-th Fibonacci number is {}".format(args.n, fib(args.n)))
    _logger.info("Script ends here")


def extract(args):
    labels_cfg = {
        'positioning': args.positioning,
        'decode': args.raw,
    }
    p = Parse(args.mscx_dir, file_re=args.file, exclude_re=args.exclude, recursive=args.nonrecursive, labels_cfg=labels_cfg, level=args.level, simulate=args.test)
    p.parse_mscx()
    params = ['notes', 'rests', 'measures', 'events', 'labels', 'chords', 'expanded']
    suffixes = {f"{p}_suffix": f"_{p}" if args.suffix is None else args.suffix for p in params if p in args}
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


def run():
    """Entry point for console_scripts
    """
    parser = argparse.ArgumentParser(
        prog='ms3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
------------------------------------
| Format Mozart data to your needs |
------------------------------------

This script needs to remain at the top level of (your local copy of) the repo  mozart_piano_sonatas.
Run with parameter -t to see all file names.
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

    extract_parser.add_argument('-s', '--suffix', nargs='?', default='', const=None, metavar='SUFFIX',
                        help="Pass -s to use standard suffixes or -s SUFFIX to choose your own.")
    extract_parser.add_argument('-o', '--out', metavar='ROOT_DIR', type=check_and_create,
                                help="""Make all relative folder paths relative to ROOT_DIR rather than to MSCX_DIR.
This setting has no effect on absolute folder paths.""")
    extract_parser.add_argument('-f', '--file', metavar="regex", default=r'\.mscx$',
                                help="Select only file names including this regular expression.")
    extract_parser.add_argument('-e', '--exclude', metavar="regex", default=r'^(\.|_)',
                                help="Any files or folders (and their subfolders) including this regex will be disregarded.")
    extract_parser.add_argument('-l', '--level', metavar='LEVEL', default='i',
                                help="Choose how many log messages you want to see: d (maximum), i, w, e, c (none)")
    extract_parser.add_argument('-t', '--test', action='store_true', help="No data is written to disk.")
    extract_parser.add_argument('-p', '--positioning', action='store_true', help="When extracting labels, include their spacial positionings in order to restore them when re-inserting.")
    extract_parser.add_argument('-r', '--raw', action='store_false', help="When extracting labels, leave chord symbols encoded instead of turning them into strings.")
    extract_parser.add_argument('--nonrecursive', action='store_false', help="Don't scan folders recursively, i.e. parse only files in MSCX_DIR.")
    extract_parser.set_defaults(func=extract)

    check_parser = subparsers.add_parser('check', help="Check DCML harmony labels.")
    check_parser.add_argument('root_dir', metavar='MSCX_DIR', nargs='?', type=check_dir,
                              default=os.getcwd(),
                              help='Folder that will be scanned for MuseScore files (.mscx).')
    args = parser.parse_args()
    if 'func' in args:
        args.func(args)
    else:
        parser.print_help()




if __name__ == "__main__":
    print('YIP')
    run()
