import os
import traceback
import multiprocessing as mp
from collections import Counter, defaultdict

import pandas as pd

from .logger import get_logger, function_logger
from .score import Score
from .utils import iterable2str, scan_directory, transform

class Parse:

    def __init__(self, dir=None, key=None, file_re='.*', folder_re='.*', exclude_re=r"^(\.|__)", recursive=True, logger_name='Parse', level=None):
        self.logger = get_logger(logger_name, level)
        self.full_paths, self.rel_paths, self.scan_paths, self.paths, self.files, self.fnames, self.fexts = defaultdict(
            list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        self._parsed, self._notelists, self._restlists, self._noterestlists, self._measurelists = {}, {}, {}, {}, {}
        self._eventlists, self._labellists, self._chordlists, self._expandedlists = {}, {}, {}, {}
        self._lists = {
            'notes': self._notelists,
            'rests': self._restlists,
            'notes_and_rests': self._noterestlists,
            'measures': self._measurelists,
            'events': self._eventlists,
            'labels': self._labellists,
            'chords': self._chordlists,
            'expanded': self._expandedlists,
        }
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

            self.scan_paths[key].append(self.last_scanned_dir)
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
            return None
        finally:
            self.logger = get_logger(prev_logger)


    def parse_mscx(self, keys=None, read_only=True, level=None, parallel=True):
        keys = self._treat_key_param(keys)
        if parallel and not read_only:
            read_only = True
            self.logger.info("When pieces are parsed in parallel, the resulting objects are always in read_only mode.")
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

    def get_lists(self, keys=None, notes=False, rests=False, notes_and_rests=False, measures=False, events=False, labels=False, chords=False, expanded=False):
        if len(self._parsed) == 0:
            self.logger.error("No scores have been parsed. Use parse_mscx()")
            return
        keys = self._treat_key_param(keys)
        scores = {ix: score for ix, score in self._parsed.items() if ix[0] in keys}
        ix = list(scores.keys())
        bool_params = list(self._lists.keys())
        l = locals()
        params = {p: l[p] for p in bool_params}

        res = {}
        for i, score in scores.items():
            for param, li in self._lists.items():
                if params[param]:
                    if i not in li:
                        li[i] = score.mscx.__getattribute__(param)
                    res[i + (param,)] = li[i]
        if expanded:
            res.update(self.expand_labels())
        return res


    def store_lists(self, keys=None, root_dir=None, notes_folder=None, notes_suffix='',
                                                    rests_folder=None, rests_suffix='',
                                                    notes_and_rests_folder=None, notes_and_rests_suffix='',
                                                    measures_folder=None, measures_suffix='',
                                                    events_folder=None, events_suffix='',
                                                    labels_folder=None, labels_suffix='',
                                                    chords_folder=None, chords_suffix='',
                                                    expanded_folder=None, expanded_suffix='',
                                                    simulate=False):
        keys = self._treat_key_param(keys)
        l = locals()
        list_types = list(self._lists)
        folder_vars = [t + '_folder' for t in list_types]
        suffix_vars = [t + '_suffix' for t in list_types]
        folder_params = {t: l[p] for t, p in zip(list_types, folder_vars) if l[p] is not None}
        suffix_params = {t: l[p] for t, p in zip(list_types, suffix_vars) if t in folder_params}
        list_params = {p: True for p in folder_params.keys()}
        lists = self.get_lists(keys, **list_params)
        paths = []
        for (key, ix, what), li in lists.items():
            paths.append(self._store(df=li, key=key, ix=ix, folder=folder_params[what], suffix=suffix_params[what], root_dir=root_dir, what=what, simulate=simulate))
        if simulate:
            return paths


    def _store(self, df, key, ix, folder, suffix='', root_dir=None, what='DataFrame', simulate=False):
        """

        Parameters
        ----------
        df
        key
        ix
        folder
        suffix
        root_dir : :obj:`str`
        what
        simulate

        Returns
        -------

        """
        prev_logger = self.logger
        self.logger = get_logger(self.fnames[key][ix])
        if os.path.isabs(folder):
            path = folder
        else:
            root = os.path.abspath(self.scan_paths[key][ix]) if root_dir is None else os.path.abspath(root_dir)
            if folder[0] == '.':
                path = os.path.abspath(os.path.join(root, self.rel_paths[key][ix], folder))
            else:
                path = os.path.abspath(os.path.join(root, folder, self.rel_paths[key][ix]))
            if path[:len(root)] != root:
                self.logger.error(f"Not allowed to store files above the root {root}.\nErroneous path: {path}")
                return None


        fname = self.fnames[key][ix] + suffix + '.tsv'
        file_path = os.path.join(path, fname)
        if simulate:
            self.logger.debug(f"Would have written {what} to {file_path}.")
        else:
            os.makedirs(path, exist_ok=True)
            no_collections_no_booleans(df, logger=self.logger).to_csv(file_path, sep='\t', index=False)
            self.logger.debug(f"{what} written to {file_path}.")
        self.logger = prev_logger
        return file_path



    def expand_labels(self, keys=None, how='dcml'):
        keys = self._treat_key_param(keys)
        scores = {ix: score for ix, score in self._parsed.items() if ix[0] in keys}
        res = {}
        for ix, score in scores.items():
            if score.mscx._annotations is not None:
                exp = score.annotations.expanded
                self._expandedlists[ix] = exp
                res[ix + ('expanded',)] = exp
        return res


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


@function_logger
def no_collections_no_booleans(df):
    collection_cols = ['next', 'chord_tones', 'added_tones']
    try:
        cc = [c for c in collection_cols if c in df.columns]
    except:
        print(f"df: {df}, class: {df.__class__}")
        raise
    if len(cc) > 0:
        df = df.copy()
        df.loc[:, cc] = transform(df[cc], iterable2str, column_wise=True, logger=logger)
        logger.debug(f"Transformed iterables in the columns {cc} to strings.")
    bool_cols = ['globalkey_is_minor', 'localkey_is_minor']
    bc = [c for c in bool_cols if c in df.columns]
    if len(bc) > 0:
        conv = {c: int for c in bc}
        df = df.astype(conv)
    return df


def pretty_extensions(key2ext):
    exts = dict(KEY='EXTENSIONS', **key2ext)
    left = max(len(k) for k in exts.keys())
    return '\n'.join(f"{k:{left}} -> {c}" for k, c in exts.items())