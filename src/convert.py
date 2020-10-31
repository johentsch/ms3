import os, re, subprocess, argparse, multiprocessing
from itertools import product


def convert(old, new, MS='mscore'):
    process = [MS, '--appimage-extract-and-run',"-o",new,old] if MS.endswith('.AppImage') else [MS,"-o",new,old]
    if subprocess.run(process):
        print(f"Converted {old} to {new}")
    else:
        print("Error while converting " + old)




def convert_folder(dir, new_folder, extensions=[], target_extension='mscx', regex='.*', suffix=None, recursive=True, MS='mscore', overwrite=False, parallel=False):
    """ Convert all files in `dir` that have one of the `extensions` to .mscx format using the executable `MS`.

    Parameters
    ----------
    dir, new_folder : str
        Directories
    extensions : list, optional
        If you want to convert only certain formats, give those, e.g. ['mscz', 'xml']
    recursive : bool, optional
        Subdirectories as well.
    MS : str, optional
        Give the path to the MuseScore executable on your system. Need only if
        the command 'mscore' does not execute MuseScore on your system.
    """
    if MS == "win":
        MS = r"C:\Program files\MuseScore 3\bin\MuseScore3.exe"
        if not os.path.isfile(MS):
            MS = r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe"
    if MS == "mac":
        MS = "/Applications/MuseScore 3.app/Contents/MacOS/mscore"
    assert os.path.isfile(MS), "MuseScore executable not found under the standard path " + MS

    conversion_params = []
    for subdir, dirs, files in os.walk(dir):
        if not recursive:
            dirs[:] = []
        else:
            dirs.sort()
        old_subdir = os.path.relpath(subdir, dir)
        new_subdir = os.path.join(new_folder, old_subdir) if old_subdir != '.' else new_folder

        for file in files:
            name, ext = os.path.splitext(file)
            ext = ext[1:]
            if re.search(regex, file) and (ext in extensions or extensions == []):
                if not os.path.isdir(new_subdir):
                    os.makedirs(new_subdir)
                if target_extension[0] == '.':
                    target_extension = target_extension[1:]
                if suffix is not None:
                    neu = '%s%s.%s' % (name, suffix, target_extension)
                else:
                    neu = '%s.%s' % (name, target_extension)
                old = os.path.join(subdir, file)
                new = os.path.join(new_subdir, neu)
                if overwrite or not os.path.isfile(new):
                    conversion_params.append((old, new, MS))
                else:
                    print(new, 'exists already. Pass -o to overwrite.')


    if  parallel:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        pool.starmap(convert, conversion_params)
        pool.close()
        pool.join()
    else:
        for o, n, ms in conversion_params:
            convert(o, n, ms)







def check_dir(d):
    if not os.path.isdir(d):
        d = os.path.join(os.getcwd(),d)
        if not os.path.isdir(d):
            if input(d + ' does not exist. Create? (y|n)') == "y":
                os.mkdir(d)
            else:
                raise argparse.ArgumentTypeError(d + ' needs to be an existing directory')
    if not os.path.isabs(d):
        d = os.path.abspath(d)
    return d

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """Tries to convert all files in DIR that have given extension(s) using a MuseScore executable.

By default, this is done recursively on subfolders and the folder structure is mirrored in TARGET_DIR,
assuming that the standard use case is conversion from an older to a newer MuseScore version.""",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dir',metavar='DIR',type=check_dir,help='path to folder with files to convert; can be relative to the folder where the script is located.')
    parser.add_argument('target',metavar='TARGET_DIR',nargs='?',type=check_dir,default=os.getcwd(),help='path to folder for converted files.')
    parser.add_argument('-e','--extensions', nargs='+', default=['mscx', 'mscz'],help="List, separated by spaces, the file extensions that you want to convert. Defaults to mscx mscz")
    parser.add_argument('-f', '--format', default='mscx', help="You may choose one out of {png, svg, pdf, mscz, mscx, wav, mp3, flac, ogg, xml, mxl, mid}")
    parser.add_argument('-m','--musescore', default='mscore',help="""Path to MuseScore executable. Defaults to 'mscore' (standard on *nix systems).
To use standard paths on commercial systems, try -m win, or -m mac.""")
    parser.add_argument('-r','--regex', default=r'.*', help="Convert only files containing this regular expression (or a simple search string).")
    parser.add_argument('-n','--non_recursive',action='store_true',help="Don't recurse through sub-directories.")
    parser.add_argument('-o','--overwrite',action='store_true',help="Set true if existing files are to be overwritten.")
    parser.add_argument('-p','--parallel',action='store_true',help="Use all available CPU cores in parallel to speed up batch jobs.")
    parser.add_argument('--suffix', metavar='SUFFIX', help='Add this suffix to the filename of every new file.')
    args = parser.parse_args()
    dir, target = os.path.realpath(args.dir), os.path.realpath(args.target)
    assert target[:len(dir)] != dir, "TARGET_DIR cannot be identical with nor a subfolder of DIR.\nDIR:        " + dir + '\nTARGET_DIR: ' + target
    convert_folder(dir, target,
                    extensions=args.extensions,
                    target_extension=args.format,
                    regex=args.regex,
                    suffix=args.suffix,
                    recursive=not args.non_recursive,
                    MS=args.musescore,
                    overwrite=args.overwrite,
                    parallel=args.parallel)
