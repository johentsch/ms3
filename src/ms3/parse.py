import os
import traceback
import pathos.multiprocessing as mp
from collections import Counter, defaultdict

import pandas as pd

from .logger import get_logger, function_logger
from .score import Score
from .utils import group_ix_tuples, iterable2str, make_ix_tuples, metadata2series, resolve_dir, scan_directory, transform

class Parse:
    """
    Class for storing and manipulating the information from multiple parses.
    """

    def __init__(self, dir=None, key=None, file_re=r'\.mscx$', folder_re='.*', exclude_re=r"^(\.|__)", recursive=True, logger_name='Parse', level=None):
        self.logger = get_logger(logger_name, level)
        self.full_paths, self.rel_paths, self.scan_paths, self.paths, self.files, self.fnames, self.fexts = defaultdict(
            list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        self._parsed, self._annotations, self._notelists, self._restlists, self._noterestlists, self._measurelists = {}, {}, {}, {}, {}, {}
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

    def add_dir(self, dir, key=None, file_re=r'\.mscx$', folder_re='.*', exclude_re=r"^(\.|__)", recursive=True):
        dir = resolve_dir(dir)
        self.last_scanned_dir = dir
        res = scan_directory(dir, file_re=file_re, folder_re=folder_re, exclude_re=exclude_re, recursive=recursive)
        ix = [self.handle_path(p, key) for p in res]
        added = sum(True for i in ix if i[0] is not None)
        if added > 0:
            grouped_ix = group_ix_tuples(ix)
            exts = {k: self.count_extensions(k, i) for k, i in grouped_ix.items()}
            self.logger.debug(f"{added} paths stored.\n{pretty_dict(exts, 'EXTENSIONS')}")
        else:
            self.logger.debug("No files added.")
        self.match_filenames()



    def handle_path(self, full_path, key=None):
        full_path = resolve_dir(full_path)
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
            score = Score(path, read_only=read_only, logger_name=self.logger.name, level=level)
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


    def parse_mscx(self, keys=None, read_only=True, level=None, parallel=True, only_new=True):
        keys = self._treat_key_param(keys)
        if parallel and not read_only:
            read_only = True
            self.logger.info("When pieces are parsed in parallel, the resulting objects are always in read_only mode.")
        if only_new:
            parse_this = [(key, ix, path, read_only, level) for key in keys
                                                            for ix, path in enumerate(self.full_paths[key])
                                                            if path.endswith('.mscx') and (key, ix) not in self._parsed]
        else:
            parse_this = [(key, ix, path, read_only, level) for key in keys
                                                            for ix, path in enumerate(self.full_paths[key])
                                                            if path.endswith('.mscx')]

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
        self._annotations.update({ix: score.annotations for ix, score in self._parsed.items()})


    def _treat_key_param(self, keys):
        if keys is None:
            keys = list(self.full_paths.keys())
        elif isinstance(keys, str):
            keys = [keys]
        return keys


    def _treat_label_type_param(self, label_type):
        if label_type is None:
            return None
        all_types = {k: str(k) for k in self.count_label_types().keys()}
        if isinstance(label_type, int) or isinstance(label_type, str):
            label_type = [label_type]
        lt = [str(t) for t in label_type]
        not_found = [t for t in lt if t not in all_types]
        if len(not_found) > 0:
            plural = len(not_found) > 1
            plural_s = 's' if plural else ''
            self.logger.warning(
                f"No labels found with {'these' if plural else 'this'} label{plural_s} label_type{plural_s}: {', '.join(not_found)}")
        return [all_types[t] for t in lt if t in all_types]



    def collect_lists(self, keys=None, notes=False, rests=False, notes_and_rests=False, measures=False, events=False,
                      labels=False, chords=False, expanded=False, only_new=True):
        """ Extracts DataFrames from the parsed scores in ``keys`` and stores them in dictionaries.

        Parameters
        ----------
        keys
        notes
        rests
        notes_and_rests
        measures
        events
        labels
        chords
        expanded
        only_new : :obj:`bool`, optional
            Set to True to also retrieve lists that have already been retrieved.

        Returns
        -------
        None
        """
        if len(self._parsed) == 0:
            self.logger.error("No scores have been parsed. Use parse_mscx()")
            return
        keys = self._treat_key_param(keys)
        scores = {ix: score for ix, score in self._parsed.items() if ix[0] in keys}
        ix = list(scores.keys())
        bool_params = list(self._lists.keys())
        l = locals()
        params = {p: l[p] for p in bool_params}

        for i, score in scores.items():
            for param, li in self._lists.items():
                if params[param] and (i not in li or not only_new):
                    df = score.mscx.__getattribute__(param)
                    if df is not None:
                        li[i] = df


    def get_lists(self, keys=None, notes=False, rests=False, notes_and_rests=False, measures=False, events=False,
                  labels=False, chords=False, expanded=False):
        if len(self._parsed) == 0:
            self.logger.error("No scores have been parsed. Use parse_mscx()")
            return
        keys = self._treat_key_param(keys)
        bool_params = list(self._lists.keys())
        l = locals()
        params = {p: l[p] for p in bool_params}
        self.collect_lists(keys, only_new=True, **params)
        ixs = list(self._iterix(keys))
        res = {}
        for param, li in self._lists.items():
            if params[param]:
                for ix in ixs:
                    if ix in li:
                        res[ix + (param,)] = li[ix]
        return res







    def metadata(self, keys=None):
        idx, meta_series = zip(*[(ix, metadata2series(self._parsed[ix].mscx.metadata)) for ix in self._iterix(keys) if
                          ix in self._parsed])
        return pd.DataFrame(meta_series, index=idx)



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


    def _iterix(self, keys=None):
        """ Iterator through index tuples for a given set of keys.

        Yields
        ------
        :obj:`tuple`
            (str, int)
        """
        keys = self._treat_key_param(keys)
        for key in sorted(keys):
            for ix in make_ix_tuples(key, len(self.fnames[key])):
                yield ix


    def _store(self, df, key, ix, folder, suffix='', root_dir=None, what='DataFrame', simulate=False, **kwargs):
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
        **kwargs: Arguments for :py:meth:`pandas.DataFrame.to_csv`

        Returns
        -------

        """
        def restore_logger(val):
            nonlocal prev_logger
            self.logger = prev_logger
            return val

        prev_logger = self.logger
        self.logger = get_logger(self.fnames[key][ix] + f":{what}")
        if df is None:
            self.logger.debug(f"No DataFrame for {what}.")
            return restore_logger(None)
        if os.path.isabs(folder) or '~' in folder:
            folder = resolve_dir(folder)
            path = folder
        else:
            root = self.scan_paths[key][ix] if root_dir is None else resolve_dir(root_dir)
            if folder[0] == '.':
                path = os.path.abspath(os.path.join(root, self.rel_paths[key][ix], folder))
            else:
                path = os.path.abspath(os.path.join(root, folder, self.rel_paths[key][ix]))
            base, _ = os.path.split(root)
            if path[:len(base)] != base:
                self.logger.error(f"Not allowed to store files above the level of root {root}.\nErroneous path: {path}")
                return restore_logger(None)

        if 'sep' not in kwargs:
            kwargs['sep'] = '\t'
        if 'index' not in kwargs:
            kwargs['index'] = False
        ext = '.tsv' if kwargs['sep'] == '\t' else '.csv'

        fname = self.fnames[key][ix] + suffix + ext
        file_path = os.path.join(path, fname)
        if simulate:
            self.logger.debug(f"Would have written {what} to {file_path}.")
        else:
            os.makedirs(path, exist_ok=True)

            no_collections_no_booleans(df, logger=self.logger).to_csv(file_path, **kwargs)
            self.logger.debug(f"{what} written to {file_path}.")

        return file_path



    # def expand_labels(self, keys=None, how='dcml'):
    #     keys = self._treat_key_param(keys)
    #     scores = {ix: score for ix, score in self._parsed.items() if ix[0] in keys}
    #     res = {}
    #     for ix, score in scores.items():
    #         if score.mscx._annotations is not None:
    #             exp = score.annotations.expanded
    #             self._expandedlists[ix] = exp
    #             res[ix + ('expanded',)] = exp
    #     return res


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


    def count_label_types(self, keys=None, per_key=False):
        keys = self._treat_key_param(keys)
        annotated = [ix for ix in self._iterix(keys) if ix in self._annotations]
        res_dict = defaultdict(Counter)
        for key, ix in annotated:
            res_dict[key].update(self._annotations[(key, ix)].label_types())
        if per_key:
            return {k: dict(v) for k, v in res_dict.items()}
        return dict(sum(res_dict.values(), Counter()))





    def count_extensions(self, keys=None, per_key=False):
        """ Returns a dict {key: Counter} or just a Counter.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`Collection`, defaults to None
            Key(s) for which to count file extensions.
        per_key : :obj:`bool`, optional
            If set to True, the results are returned as a dict {key: Counter},
            otherwise the counts are summed up in one Counter.

        Returns
        -------
        :obj:`dict` or :obj:`collections.Counter`

        """
        keys = self._treat_key_param(keys)
        res_dict = {}
        for key in keys:
            res_dict[key] = Counter(self.fexts[key])
        if per_key:
            return {k: dict(v) for k, v in res_dict.items()}
        return dict(sum(res_dict.values(), Counter()))


    def get_labels(self, keys=None, staff=None, voice=None, label_type=None, positioning=True, decode=False):
        keys = self._treat_key_param(keys)
        label_type = self._treat_label_type_param(label_type)
        self.collect_lists(labels=True, only_new=True)
        l = locals()
        params = {p: l[p] for p in ['staff', 'voice', 'label_type', 'positioning', 'decode']}
        ixs = [ix for ix in self._iterix(keys) if ix in self._annotations]
        annotation_tables = [self._annotations[ix].get_labels(**params, warnings=False) for ix in ixs]
        return pd.concat(annotation_tables, keys=ixs, names=['key', 'ix', 'i'])






    def info(self, keys=None, return_str=False):
        ixs = list(self._iterix(keys))
        info = f"{len(ixs)} files.\n"
        exts = self.count_extensions(keys, per_key=True)
        info += pretty_dict(exts, heading='EXTENSIONS')
        parsed = sum(True for ix in ixs if ix in self._parsed)
        if parsed > 0:
            mscx = self.count_extensions(keys, per_key=False)['.mscx']
            if parsed == mscx:
                info += f"\n\nAll {mscx} MSCX files have been parsed."
            else:
                info += f"\n\n{parsed}/{mscx} MSCX files have been parsed."
            annotated = sum(True for ix in ixs if ix in self._annotations)
            info += f"\n{annotated} of them have annotations attached."
            if annotated > 0:
                l_types = self.count_label_types(keys, per_key=True)
                info += f"\n{pretty_dict(l_types, heading='LABEL_TYPES')}"

        else:
            info += f"\n\nNo mscx files have been parsed."
        if return_str:
            return info
        print(info)



    def __repr__(self):
        return self.info(return_str=True)





@function_logger
def no_collections_no_booleans(df):
    """
    Cleans the DataFrame columns ['next', 'chord_tones', 'added_tones'] from tuples and the columns
    ['globalkey_is_minor', 'localkey_is_minor'] from booleans, converting them all to integers

    """
    if df is None:
        return df
    collection_cols = ['next', 'chord_tones', 'added_tones']
    try:
        cc = [c for c in collection_cols if c in df.columns]
    except:
        logger.error(f"df needs to be a DataFrame, not a {df.__class__}.")
        return df
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


def pretty_dict(d, heading=None):
    if heading:
        d = dict(KEY=heading, **d)
    left = max(len(str(k)) for k in d.keys())
    return '\n'.join(f"{k:{left}} -> {c}" for k, c in d.items())