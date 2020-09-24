import os
import traceback
import pathos.multiprocessing as mp
from collections import Counter, defaultdict

import pandas as pd

from .logger import get_logger
from .score import Score
from .utils import scan_directory

class Parse:

    def __init__(self, dir=None, key=None, file_re='.*', folder_re='.*', exclude_re=r"^(\.|__)", recursive=True, logger_name='Parse', level=None):
        self.logger = get_logger(logger_name, level)
        self.full_paths, self.rel_paths, self.paths, self.files, self.fnames, self.fexts = defaultdict(
            list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        self._parsed, self._notelists, self._measurelists, self._eventlists = {}, {}, {}, {}
        self.matches = pd.DataFrame()
        self.last_scanned_dir = dir
        if dir is not None:
            self.add_dir(dir=dir, key=key, file_re=file_re, folder_re=folder_re, exclude_re=exclude_re, recursive=recursive)

    def add_dir(self, dir, key=None, file_re='.*', folder_re='.*', exclude_re=r"^(\.|__)", recursive=True):
        self.last_scanned_dir = os.path.abspath(dir)
        res = scan_directory(dir, file_re=file_re, folder_re=folder_re, exclude_re=exclude_re, recursive=recursive)
        ix = [self.handle_path(p, key) for p in res]
        added = sum(True for i in ix if i[0] is not None)
        if added > 0:
            grouped_ix = group_index_tuples(ix)
            exts = {k: self.count_extensions(k, i) for k, i in grouped_ix.items()}
            self.logger.debug(f"{added} paths stored.\n{pretty_extensions(exts)}")
        else:
            self.logger.debug("No files added.")
        self.match_filenames()

    def handle_path(self, full_path, key=None):
        full_path = os.path.abspath(full_path)
        if os.path.isfile(full_path):
            file_path, file = os.path.split(full_path)
            file_name, file_ext = os.path.splitext(file)
            rel_path = os.path.relpath(file_path, self.last_scanned_dir)
            if key is None:
                key = rel_path
            if file in self.files[key]:
                self.logger.error(f"""The file {file} is already registered for key '{key}'.
Load one of the identically named files with a different key using add_dir(key='KEY').""")
                return (None, None)

            self.rel_paths[key].append(rel_path)
            self.full_paths[key].append(full_path)
            self.paths[key].append(file_path)
            self.files[key].append(file)
            self.fnames[key].append(file_name)
            self.fexts[key].append(file_ext)
            return key, len(self.paths[key]) - 1
        else:
            self.logger.error("No file found at this path: " + full_path)
            return (None, None)


    def _parse(self, key, ix, path=None, read_only=False, level=None):
        if path is None:
            path = self.full_paths[key][ix]
        file = self.files[key][ix]
        fname = self.fnames[key][ix]
        prev_logger = self.logger.name
        self.logger = get_logger(f"{fname}")
        self.logger.debug(f"Attempting to parse {file}")
        try:
            score = Score(path, read_only=read_only, level=level)
            self._parsed[(key, ix)] = score
            self.logger.info(f"Done parsing {file}")
            return score
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.logger.exception(traceback.format_exc())
        finally:
            self.logger = get_logger(prev_logger)


    def parse_mscx(self, keys=None, read_only=False, level=None, parallel=True):
        keys = self._treat_key_param(keys)
        parse_this = [(key, ix, path, read_only, level) for key in keys for ix, path in enumerate(self.full_paths[key]) if path.endswith('.mscx')]
        if parallel:
            pool = mp.Pool(mp.cpu_count())
            res = pool.starmap(self._parse, parse_this)
            pool.close()
            pool.join()
            indices = [t[:2] for t in parse_this]
            self._parsed.update({i: score for i, score in zip(indices, res)})
        else:
            for params in parse_this:
                self._parsed[params[:2]] = self._parse(*params)


    def _treat_key_param(self, keys):
        if keys is None:
            keys = list(self.full_paths.keys())
        elif isinstance(keys, str):
            keys = [keys]
        return keys

    def get_lists(self, keys=None, notes=False, measures=False, events=False, labels=False, chords=False):
        if len(self._parsed) == 0:
            self.logger.error("No scores have been parsed. Use parse_mscx()")
            return []
        keys = self._treat_key_param(keys)
        scores = [k for k in self._parsed.keys() if k[0] in keys]
        for ix in scores:
            score = self._parsed[ix]
            if notes:
                self._notelists[ix] = score.mscx.notes
            if measures:
                self._measurelists[ix] = score.mscx.measures
            if events:
                self._eventlists[ix] = score.mscx.events







    @property
    def parsed(self):
        res = {}
        for k, score in self._parsed.items():
            info = score.full_paths['mscx']
            if 'annotations' in score._annotations:
                info += f" -> {score.annotations.n_labels()} labels"
            res[k] = info
        return res

    def match_filenames(self):
        pass


    def count_extensions(self, key=None, ix=None):
        if key is None:
            c = {}
            for key, l in self.fexts.items():
                c[key] = dict(Counter(l))
        elif ix is None:
            c = Counter(self.fexts[key])
        else:
            c = Counter(self.fexts[key][i] for i in ix if i is not None)
        return dict(c)



    def __repr__(self):
        msg = f"{sum(True for l in self.paths.values() for i in l)} files.\n"
        msg += pretty_extensions(self.count_extensions())
        return msg



def group_index_tuples(l):
    d = defaultdict(list)
    for k, i in l:
        if k is not None:
            d[k].append(i)
    return dict(d)


def pretty_extensions(key2ext):
    exts = dict(KEY='EXTENSIONS', **key2ext)
    left = max(len(k) for k in exts.keys())
    return '\n'.join(f"{k:{left}} -> {c}" for k, c in exts.items())