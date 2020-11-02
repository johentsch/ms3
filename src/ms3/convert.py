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
    if MS != 'mscore':
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
