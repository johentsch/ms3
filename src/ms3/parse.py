import sys, os
import traceback
import pathos.multiprocessing as mp
from collections import Counter, defaultdict
from collections.abc import Collection

import pandas as pd

from .logger import get_logger
from .score import Score
from .utils import group_id_tuples, load_tsv, make_id_tuples, metadata2series, no_collections_no_booleans, pretty_dict, resolve_dir, scan_directory

class Parse:
    """
    Class for storing and manipulating the information from multiple parses (i.e. :py:class:`~ms3.score.Score` objects).
    """

    def __init__(self, dir=None, key=None, index=None, file_re=r"\.mscx$", folder_re='.*', exclude_re=r"^(\.|__)", recursive=True, logger_name='Parse', level=None):
        """

        Parameters
        ----------
        dir, key, index, file_re, folder_re, exclude_re, recursive : optional
            Arguments for the method :py:meth:`~ms3.parse.add_folder`.
            If ``dir`` is not passed, no files are added to the new object.
        logger_name : :obj:`str`, optional
            If you have defined a logger, pass its name. Otherwise, the logger 'Parse' is used.
        level : {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}, optional
            Pass a level name for which (and above which) you want to see log records.
        """
        self.logger = get_logger(logger_name, level)

        # defaultdicts with keys as keys, each holding a list with file information (therefore accessed via [key][i] )
        self.full_paths, self.rel_paths, self.scan_paths, self.paths, self.files, self.fnames, self.fexts = defaultdict(
            list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)

        # dicts that have IDs as keys and are therefor accessed via [(key, i)]
        self._parsed_mscx, self._annotations, self._notelists, self._restlists, self._noterestlists = {}, {}, {}, {}, {}
        self._eventlists, self._labellists, self._chordlists, self._expandedlists, self._index = {}, {}, {}, {}, {}
        self._measurelists, self._parsed_tsv = {}, {}

        # dict with keys as keys, holding the names of the index levels (which have to be the same for all corresponding IDs)
        self._levelnames = {}
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
            self.add_dir(dir=dir, key=key, index=index, file_re=file_re, folder_re=folder_re, exclude_re=exclude_re, recursive=recursive)

    @property
    def parsed(self):
        """ Returns an overview of the MSCX files that have already been parsed."""
        return {k: score.full_paths['mscx'] for k, score in self._parsed_mscx.items()}
        # res = {}
        # for k, score in self._parsed.items():
        #     info = score.full_paths['mscx']
        #     if 'annotations' in score._annotations:
        #         info += f" -> {score.annotations.n_labels()} labels"
        #     res[k] = info
        # return res


    def add_detached_annotations(self, mscx_key, tsv_key, match_dict=None):
        pass



    def add_dir(self, dir, key=None, index=None, file_re=r'\.mscx$', folder_re='.*', exclude_re=r"^(\.|__)", recursive=True):
        """
        This function scans the directory ``dir`` for files matching the criteria.

        Parameters
        ----------
        dir : :obj:`str`
            Directory to scan for files.
        key : :obj:`str`, optional
            | Pass a string to identify the loaded files.
            | By default, the relative sub-directories of ``dir`` are used as keys. For example, for files within ``dir``
              itself, the key would be ``'.'``, for files in the subfolder ``scores`` it would be ``'scores'``, etc.
        index : element or :obj:`Collection` of {'key', 'fname', 'i', :obj:`Collection`}
            | Change this parameter if you want to create particular indices for multi-piece DataFrames.
            | The resulting index must be unique (for identification) and have as many elements as added files.
            | Every single element or Collection of elements ∈ {'key', 'fname', 'i', :obj:`Collection`} stands for an index level.
            | In other words, a single level will result in a single index and a collection of levels will result in a
              :obj:`~pandas.core.indexes.multi.MultiIndex`.
            | If you pass a Collection that does not start with one of {'key', 'fname', 'i'}, it is interpreted as an
              index level itself and needs to have at least as many elements as the number of added files.
            | The default ``None`` is equivalent to passing ``(key, i)``, i.e. a MultiIndex of IDs.
            | 'fname' evokes an index level made from file names.
        file_re
        folder_re
        exclude_re
        recursive

        Returns
        -------

        """
        dir = resolve_dir(dir)
        self.last_scanned_dir = dir
        if file_re in ['tsv', 'csv']:
            file_re = r"\." + file_re + '$'
        res = scan_directory(dir, file_re=file_re, folder_re=folder_re, exclude_re=exclude_re, recursive=recursive)
        ids = [self.handle_path(p, key) for p in res]
        selector, added_ids = zip(*[(i, x) for i, x in enumerate(ids) if x[0] is not None])
        if len(added_ids) > 0:
            grouped_ids = group_id_tuples(ids)
            exts = {k: self.count_extensions(k, i) for k, i in grouped_ids.items()}
            self.logger.debug(f"{len(added_ids)} paths stored.\n{pretty_dict(exts, 'EXTENSIONS')}")
            new_index, level_names = self._treat_index_param(index, ids=added_ids, selector=selector)
            self._index.update(new_index)
            for k in grouped_ids.keys():
                if k in self._levelnames:
                    previous = self._levelnames[k]
                    if previous != level_names:
                        replacement_ids = [(k, i) for i in grouped_ids.values()]
                        if None in previous:
                            new_levels = [level for level in previous if level is not None]
                            if len(new_levels) == 0:
                                new_levels = None
                            replacement_ix, new_levels = self._treat_index_param(new_levels, ids=replacement_ids)
                            self.logger.warning(f"""The created index has different levels ({level_names}) than the index that already exists for key '{k}': {previous}.
Since None stands for a custom level, an alternative index with levels {new_levels} has been created.""")
                        else:
                            replacement_ix, _ = self._treat_index_param(previous, ids=replacement_ids)
                            self.logger.info(f"""The created index has different levels ({level_names}) than the index that already exists for key '{k}': {previous}.
Therefore, the index for this key has been adapted.""")
                        self._index.update(replacement_ix)
                    else:
                        self.logger.debug(f"Index level names match the existing ones for key '{k}.'")
                else:
                    self._levelnames[k] = level_names
        else:
            self.logger.debug("No files added.")



    def collect_annotations_objects_references(self, keys=None, ids=None):
        if ids is None:
            ids = list(self._iterids(keys))
        updated = {}
        for id in ids:
            if id in self._parsed_mscx:
                score = self._parsed_mscx[id]
                if 'annotations' in score._annotations:
                    updated[id] = score.annotations
                elif id in self._annotations:
                    del (self._annotations[id])
        self._annotations.update(updated)



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
        if len(self._parsed_mscx) == 0:
            self.logger.error("No scores have been parsed so far. Use parse_mscx()")
            return
        keys = self._treat_key_param(keys)
        scores = {id: score for id, score in self._parsed_mscx.items() if id[0] in keys}
        ids = list(scores.keys())
        bool_params = list(self._lists.keys())
        l = locals()
        params = {p: l[p] for p in bool_params}

        for i, score in scores.items():
            for param, li in self._lists.items():
                if params[param] and (i not in li or not only_new):
                    df = score.mscx.__getattribute__(param)
                    if df is not None:
                        li[i] = df


    def count_annotation_layers(self, keys=None, per_key=False, detached=False):
        """ Returns a dict {key: Counter} or just a Counter.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`Collection`, defaults to None
            Key(s) for which to count annotation layers.
        per_key : :obj:`bool`, optional
            If set to True, the results are returned as a dict {key: Counter},
            otherwise the counts are summed up in one Counter.
        detached : :obj:`bool`, optional
            Set to True in order to count layers in annotations that are currently not attached to a score.

        Returns
        -------
        :obj:`dict` or :obj:`collections.Counter`

        """
        res_dict = defaultdict(Counter)

        if detached:
            for id in self._iterids(keys):
                if id in self._parsed_mscx:
                    for key, annotations in self._parsed_mscx[id]._annotations.items():
                        if key != 'annotations':
                            _, layers = annotations.annotation_layers
                            res_dict[key].update(layers.to_dict())
        else:
            for key, i in self._iterids(keys):
                if (key, i) in self._annotations:
                    _, layers = self._annotations[(key, i)].annotation_layers
                    res_dict[key].update(layers.to_dict())

        def make_series(counts):
            if len(counts) == 0:
                return pd.Series()
            data = counts.values()
            ix = pd.Index(counts.keys(), names=['staff', 'voice', 'label_type'])
            return pd.Series(data, ix)

        if per_key:
            res = {k: make_series(v) for k, v in res_dict.items()}
        else:
            res = make_series(sum(res_dict.values(), Counter()))
        if len(res) == 0:
            self.logger.info("No annotations found. Maybe no scores have been parsed using parse_mscx()?")
        return res


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



    def count_label_types(self, keys=None, per_key=False):
        keys = self._treat_key_param(keys)
        annotated = [id for id in self._iterids(keys) if id in self._annotations]
        res_dict = defaultdict(Counter)
        for key, i in annotated:
            res_dict[key].update(self._annotations[(key, i)].label_types)
        if len(res_dict) == 0:
            if len(self._parsed_mscx) == 0:
                self.logger.error("No scores have been parsed so far. Use parse_mscx().")
            else:
                self.logger.info("None of the scores contain annotations.")
        if per_key:
            return {k: dict(v) for k, v in res_dict.items()}
        return dict(sum(res_dict.values(), Counter()))



    def detach_labels(self, keys=None, annotation_key='detached', staff=None, voice=None, label_type=None, delete=True):
        assert annotation_key != 'annotations', "The key 'annotations' is reserved, please choose a different one."
        ids = [id for id in self._iterids(keys) if id in self._annotations]
        prev_logger = self.logger
        for id in ids:
            score = self._parsed_mscx[id]
            self.logger = score.logger
            try:
                score.detach_labels(key=annotation_key, staff=staff, voice=voice, label_type=label_type, delete=delete)
            except:
                self.logger.error(f"Detaching labels failed with the following error:\n{sys.exc_info()[1]}")
            finally:
                self.logger = prev_logger
        self.collect_annotations_objects_references(ids=ids)




    def get_labels(self, keys=None, staff=None, voice=None, label_type=None, positioning=True, decode=False):
        if len(self._parsed_mscx) == 0:
            self.logger.error("No scores have been parsed so far. Use parse_mscx()")
            return pd.DataFrame()
        keys = self._treat_key_param(keys)
        label_type = self._treat_label_type_param(label_type)
        self.collect_lists(labels=True, only_new=True)
        l = locals()
        params = {p: l[p] for p in ['staff', 'voice', 'label_type', 'positioning', 'decode']}
        ids = [id for id in self._iterids(keys) if id in self._annotations]
        annotation_tables = [self._annotations[id].get_labels(**params, warnings=False) for id in ids]
        idx, names = self.ids2idx(ids)
        names += tuple(annotation_tables[0].index.names)
        return pd.concat(annotation_tables, keys=idx, names=names)




    def get_lists(self, keys=None, notes=False, rests=False, notes_and_rests=False, measures=False, events=False,
                  labels=False, chords=False, expanded=False):
        if len(self._parsed_mscx) == 0:
            self.logger.error("No scores have been parsed so far. Use parse_mscx()")
            return {}
        keys = self._treat_key_param(keys)
        bool_params = list(self._lists.keys())
        l = locals()
        params = {p: l[p] for p in bool_params}
        self.collect_lists(keys, only_new=True, **params)
        ids = list(self._iterids(keys))
        res = {}
        for param, li in self._lists.items():
            if params[param]:
                for id in ids:
                    if id in li:
                        res[id + (param,)] = li[id]
        return res



    def handle_path(self, full_path, key=None):
        full_path = resolve_dir(full_path)
        if os.path.isfile(full_path):
            file_path, file = os.path.split(full_path)
            file_name, file_ext = os.path.splitext(file)
            rel_path = os.path.relpath(file_path, self.last_scanned_dir)
            if key is None:
                key = rel_path
            if file in self.files[key]:
                same_name = [i for i, f in enumerate(self.files[key]) if f == file]
                if any(True for i in same_name if self.rel_paths[key][i] == rel_path):
                    self.logger.error(f"""The file name {file} is already registered for key '{key}' and both files have the relative path {rel_path}.
Load one of the identically named files with a different key using add_dir(key='KEY').""")
                    return (None, None)
                self.logger.debug(f"The file {file} is already registered for key '{key}' but can be distinguished via the relative path.")

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


    def ids2idx(self, ids, pandas_index=False):
        idx = [self._index[id] for id in ids]
        levels = [len(ix) for ix in idx]
        error = False
        if not all(l == levels[0] for l in levels[1:]):
            self.logger.warning(
                f"Could not create index because the index values have different numbers of levels: {set(levels)}")
            idx = ids
            error = True

        if  error:
            names = ['key', 'i']
        else:
            grouped_ids = group_id_tuples(ids)
            level_names = {k: self._levelnames[k] for k in grouped_ids}
            if len(set(level_names.keys())) > 1:
                self.logger.warning(
                    f"Could not set level names because they differ for the different keys:\n{pretty_dict(level_names, 'LEVEL_NAMES')}")
                names = None
            else:
                names = tuple(level_names.values())[0]

        if pandas_index:
            idx = pd.Index(idx, names=names)
            return idx

        return idx, names



    def index(self, keys=None, per_key=False):
        if per_key:
            keys = self._treat_key_param(keys)
            return {k: self.index(k) for k in keys}
        return [self._index[id] for id in self._iterids(keys)]




    def info(self, keys=None, return_str=False):
        ids = list(self._iterids(keys))
        info = f"{len(ids)} files.\n"
        exts = self.count_extensions(keys, per_key=True)
        info += pretty_dict(exts, heading='EXTENSIONS')
        parsed_ids = [id for id in ids if id in self._parsed_mscx]
        parsed = len(parsed_ids)
        if parsed > 0:
            mscx = self.count_extensions(keys, per_key=False)['.mscx']
            if parsed == mscx:
                info += f"\n\nAll {mscx} MSCX files have been parsed."
            else:
                info += f"\n\n{parsed}/{mscx} MSCX files have been parsed."
            annotated = sum(True for id in ids if id in self._annotations)
            info += f"\n{annotated} of them have annotations attached."
            if annotated > 0:
                layers = self.count_annotation_layers(keys, per_key=True)
                info += f"\n{pretty_dict(layers, heading='ANNOTATION LAYERS')}"

            detached = sum(True for id in parsed_ids if self._parsed_mscx[id].has_detached_annotations)
            if detached > 0:
                info += f"\n{detached} of them have detached annotations:"
                layers = self.count_annotation_layers(keys, per_key=True, detached=True)
                info += f"\n{pretty_dict(layers, heading='ANNOTATION LAYERS')}"

        else:
            info += f"\n\nNo mscx files have been parsed."
        if return_str:
            return info
        print(info)



    def metadata(self, keys=None):
        parsed_ids = [id for id in self._iterids(keys) if id in self._parsed_mscx]
        if len(parsed_ids) > 0:
            ids, meta_series = zip(*[(id, metadata2series(self._parsed_mscx[id].mscx.metadata)) for id in parsed_ids])
            idx = self.ids2idx(ids, pandas_index=True)
            return pd.DataFrame(meta_series, index=idx)
        if len(self._parsed_mscx) == 0:
            self.logger.info("No scores have been parsed so far. Use parse_mscx()")
        return pd.DataFrame()


    def parse(self, keys=None, read_only=True, level=None, parallel=True, only_new=True, fexts=None, **kwargs):
        """ Shorthand for executing parse_mscx and parse_tsv at a time."""
        self.parse_mscx(keys=keys, read_only=read_only, level=level, parallel=parallel, only_new=only_new)
        self.parse_tsv(keys=keys, fexts=fexts, **kwargs)



    def parse_mscx(self, keys=None, read_only=True, level=None, parallel=True, only_new=True):

        if parallel and not read_only:
            read_only = True
            self.logger.info("When pieces are parsed in parallel, the resulting objects are always in read_only mode.")

        if only_new:
            paths = [(key, i, self.full_paths[key][i]) for key, i in self._iterids(keys) if
                     self.fexts[key][i] == '.mscx' and (key, i) not in self._parsed_mscx]
        else:
            paths = [(key, i, self.full_paths[key][i]) for key, i in self._iterids(keys) if
                     self.fexts[key][i] == '.mscx']

        parse_this = [(key, i, path, read_only, level) for key, i, path in paths]
        ids = [t[:2] for t in parse_this]
        if parallel:
            pool = mp.Pool(mp.cpu_count())
            res = pool.starmap(self._parse, parse_this)
            pool.close()
            pool.join()
            self._parsed_mscx.update({id: score for id, score in zip(ids, res)})
        else:
            for params in parse_this:
                self._parsed_mscx[params[:2]] = self._parse(*params)
        self.collect_annotations_objects_references(ids=ids)


    def parse_tsv(self, keys=None, fexts=None, **kwargs):
        if fexts is None:
            ids = [(key, i) for key, i in self._iterids(keys) if self.fexts[key][i] != '.mscx']
        else:
            if isinstance(fexts, str):
                fexts = [fexts]
            fexts = [ext if ext[0] == '.' else f".{ext}" for ext in fexts]
            ids = [(key, i) for key, i in self._iterids(keys) if self.fexts[key][i] in fexts]
        for key, i in ids:
            self._parsed_tsv[(key, i)] = load_tsv(self.full_paths[key][i], **kwargs)








    def store_lists(self, keys=None, root_dir=None, notes_folder=None, notes_suffix='',
                                                    rests_folder=None, rests_suffix='',
                                                    notes_and_rests_folder=None, notes_and_rests_suffix='',
                                                    measures_folder=None, measures_suffix='',
                                                    events_folder=None, events_suffix='',
                                                    labels_folder=None, labels_suffix='',
                                                    chords_folder=None, chords_suffix='',
                                                    expanded_folder=None, expanded_suffix='',
                                                    simulate=False):
        l = locals()
        list_types = list(self._lists)
        folder_vars = [t + '_folder' for t in list_types]
        suffix_vars = [t + '_suffix' for t in list_types]
        folder_params = {t: l[p] for t, p in zip(list_types, folder_vars) if l[p] is not None}
        suffix_params = {t: l[p] for t, p in zip(list_types, suffix_vars) if t in folder_params}
        list_params = {p: True for p in folder_params.keys()}
        lists = self.get_lists(keys, **list_params)
        paths = {}
        for (key, i, what), li in lists.items():
            new_path = self._store_tsv(df=li, key=key, i=i, folder=folder_params[what], suffix=suffix_params[what], root_dir=root_dir, what=what, simulate=simulate)
            if new_path in paths:
                modus = 'would ' if simulate else ''
                self.logger.warning(f"The {paths[new_path]} at {new_path} {modus}have been overwritten with {what}.")
            paths[new_path] = what
        if simulate:
            return list(set(paths.keys()))



    def store_mscx(self, keys=None, root_dir=None, folder='.', suffix='', simulate=False):
        """ Stores the parsed MuseScore files in their current state, e.g. after detaching or attaching annotations.
        """
        ids = [id for id in self._iterids(keys) if id in self._parsed_mscx]
        paths = []
        for key, i in ids:
            new_path = self._store_mscx(key=key, i=i, folder=folder, suffix=suffix, root_dir=root_dir, simulate=simulate)
            if new_path in paths:
                modus = 'would ' if simulate else ''
                self.logger.warning(f"The score at {new_path} {modus}have been overwritten.")
            else:
                paths.append(new_path)
        if simulate:
            return list(set(paths))




    def _calculate_path(self, key, i, root_dir, folder):
        """ Constructs a path and file name from a loaded file based on the arguments.

        Parameters
        ----------
        key, i : (:obj:`str`, :obj:`int`)
            ID from which to construct the new path and filename.
        folder : :obj:`str`
            Where to store the file. Can be relative to ``root_dir`` or absolute, in which case ``root_dir`` is ignored.
            If ``folder`` is relative, the behaviour depends on whether it starts with a dot ``.`` or not: If it does,
            the folder is created at every end point of the relative tree structure under ``root_dir``. If it doesn't,
            it is created only once, relative to ``root_dir``, and the relative tree structure is build below.
        root_dir : :obj:`str`, optional
            Defaults to None, meaning that the original root directory is used that was added to the Parse object.
            Otherwise, pass a directory to rebuild the original substructure. If ``folder`` is an absolute path,
            ``root_dir`` is ignored.
        """
        if os.path.isabs(folder) or '~' in folder:
            folder = resolve_dir(folder)
            path = folder
        else:
            root = self.scan_paths[key][i] if root_dir is None else resolve_dir(root_dir)
            if folder[0] == '.':
                path = os.path.abspath(os.path.join(root, self.rel_paths[key][i], folder))
            else:
                path = os.path.abspath(os.path.join(root, folder, self.rel_paths[key][i]))
            base, _ = os.path.split(root)
            if path[:len(base)] != base:
                self.logger.error(f"Not allowed to store files above the level of root {root}.\nErroneous path: {path}")
                return None
        return path



    def _iterids(self, keys=None):
        """ Iterator through IDs for a given set of keys.

        Yields
        ------
        :obj:`tuple`
            (str, int)
        """
        keys = self._treat_key_param(keys)
        for key in sorted(keys):
            for id in make_id_tuples(key, len(self.fnames[key])):
                yield id


    def _itersel(self, collectio, selector, opposite=False):
        if selector is None:
            for i, e in enumerate(collectio):
                yield e
        if opposite:
            for i, e in enumerate(collectio):
                if i not in selector:
                    yield e
        else:
            for i, e in enumerate(collectio):
                if i in selector:
                    yield e

    def _parse(self, key, i, path=None, read_only=False, level=None):
        if path is None:
            path = self.full_paths[key][i]
        file = self.files[key][i]
        fname = self.fnames[key][i]
        prev_logger = self.logger.name
        self.logger = get_logger(f"{fname}")
        self.logger.debug(f"Attempting to parse {file}")
        try:
            score = Score(path, read_only=read_only, logger_name=self.logger.name, level=level)
            self._parsed_mscx[(key, i)] = score
            self.logger.info(f"Done parsing {file}")
            return score
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.logger.exception(traceback.format_exc())
            return None
        finally:
            self.logger = get_logger(prev_logger)




    def _store_mscx(self, key, i, folder, suffix='', root_dir=None, simulate=False):
        """ Creates a MuseScore 3 file from the Score object at the given ID (key, i).

        Parameters
        ----------
        key, i : (:obj:`str`, :obj:`int`)
            ID from which to construct the new path and filename.
        folder, root_dir : :obj:`str`
            Parameters passed to :py:meth:`_calculate_path`.
        suffix : :obj:`str`, optional
            Suffix to append to the original file name.
        simulate : :obj:`bool`, optional
            Set to True if no files are to be written.

        Returns
        -------
        :obj:`str`
            Path of the stored file.

        """

        def restore_logger(val):
            nonlocal prev_logger
            self.logger = prev_logger
            return val

        prev_logger = self.logger
        fname = self.fnames[key][i]
        self.logger = get_logger(fname)
        id = (key, i)
        if id not in self._parsed_mscx:
            self.logger.error(f"No Score object found. Call parse_mscx() first.")
            return restore_logger(None)
        path = self._calculate_path(key=key, i=i, root_dir=root_dir, folder=folder)
        if path is None:
            return restore_logger(None)

        fname = fname + suffix + '.mscx'
        file_path = os.path.join(path, fname)
        if simulate:
            self.logger.debug(f"Would have written score to {file_path}.")
        else:
            os.makedirs(path, exist_ok=True)
            self._parsed_mscx[id].store_mscx(file_path)
            self.logger.debug(f"Score written to {file_path}.")

        return restore_logger(file_path)

    def _store_tsv(self, df, key, i, folder, suffix='', root_dir=None, what='DataFrame', simulate=False, **kwargs):
        """ Stores a given DataFrame by constructing path and file name from a loaded file based on the arguments.

        Parameters
        ----------
        df : :obj:`pandas.DataFrame`
            DataFrame to store as a TSV.
        key, i : (:obj:`str`, :obj:`int`)
            ID from which to construct the new path and filename.
        folder, root_dir : :obj:`str`
            Parameters passed to :py:meth:`_calculate_path`.
        suffix : :obj:`str`, optional
            Suffix to append to the original file name.
        what : :obj:`str`, optional
            Descriptor, what the DataFrame contains for more informative log message.
        simulate : :obj:`bool`, optional
            Set to True if no files are to be written.
        **kwargs: Arguments for :py:meth:`pandas.DataFrame.to_csv`. Defaults to ``{'sep': '\t', 'index': False}``.
            If 'sep' is changed to a different separator, the file extension(s) will be changed to '.csv' rather than '.tsv'.

        Returns
        -------
        :obj:`str`
            Path of the stored file.

        """
        def restore_logger(val):
            nonlocal prev_logger
            self.logger = prev_logger
            return val

        prev_logger = self.logger
        fname = self.fnames[key][i]
        self.logger = get_logger(fname + f":{what}")
        if df is None:
            self.logger.debug(f"No DataFrame for {what}.")
            return restore_logger(None)
        path = self._calculate_path(key=key, i=i, root_dir=root_dir, folder=folder)
        if path is None:
            return restore_logger(None)

        if 'sep' not in kwargs:
            kwargs['sep'] = '\t'
        if 'index' not in kwargs:
            kwargs['index'] = False
        ext = '.tsv' if kwargs['sep'] == '\t' else '.csv'

        fname = fname + suffix + ext
        file_path = os.path.join(path, fname)
        if simulate:
            self.logger.debug(f"Would have written {what} to {file_path}.")
        else:
            os.makedirs(path, exist_ok=True)

            no_collections_no_booleans(df, logger=self.logger).to_csv(file_path, **kwargs)
            self.logger.debug(f"{what} written to {file_path}.")

        return restore_logger(file_path)


    def _treat_index_param(self, index_param, ids, selector=None):
        if index_param is None:
            names = ('key', 'i')
            return {id: id for id in ids}, names
        if isinstance(index_param, str):
            index_param = [index_param]
        index_levels = []
        is_index_level=False
        names = []
        for i, level in enumerate(index_param):
            if isinstance(level, str):
                if level in ['key', 'fname', 'i']:
                    new_level = self._make_index_level(level, ids=ids, selector=selector)
                    index_levels.append(new_level)
                    names.append(level)
                    self.logger.debug(f"Level '{level}' generated: {new_level}")
                else:
                    assert len(index_levels) == 0, f"Failed to create index level '{level}', because it is neither a keyword nor a Collection."
                    is_index_level = True
                    break
            elif isinstance(level, Collection):
                new_level = self._make_index_level(level, ids=ids, selector=selector)
                if len(new_level) > 0:
                    index_levels.append(new_level)
                    names.append(None)
            else:
                assert len(index_levels) == 0, f"Failed to create index level '{level}', because it is neither a keyword nor a Collection."
                is_index_level = True
                break
        if is_index_level:
            self.logger.debug(f"index_param is interpreted as a single index level rather than a collection of levels.")
            new_level = self._make_index_level(index_param, ids=ids, selector=selector)
            if len(new_level) > 0:
                index_levels.append(new_level)
                names = [None]
        if len(index_levels) == 0:
            self.logger.error(f"No index could be created.")
        new_index = {id: ix for id, ix in zip(ids, zip(*[tuple(v.values()) for v in index_levels]))}
        existing = [ix for ix in new_index if ix in self._index.keys()]
        counts = {k: v for k, v in Counter(new_index.values()).items() if v > 1}
        l_counts, l_existing = len(counts), len(existing)
        if l_counts > 0 or l_existing > 0:
            new_index = self._treat_index_param(None, ids=ids)
            if l_counts > 0:
                plural_phrase = "These values occur" if l_counts > 1 else "This value occurs"
                self.logger.error(f"The generated index is not unique and has been replaced by the standard index (IDs).\n{plural_phrase} several times:\n{pretty_dict(counts)}")
            if l_existing > 0:
                plural_phrase = "s are" if l_existing > 1 else " is"
                self.logger.error(f"The generated index cannot be used because the following element{plural_phrase} already in use:\n{existing}")
        return new_index, tuple(names)



    def _make_index_level(self, level, ids, selector=None):
        if level == 'key':
            return {id: id[0] for id in ids}
        if level == 'i':
            return {id: id[1] for id in ids}
        if level == 'fname':
            return {(key, i): self.fnames[key][i] for key, i in ids}
        ll, li = len(level), len(ids)
        ls = 0 if selector is None else len(selector)
        if ll < li:
            self.logger.error(f"Index level (length {ll}) has not enough values for {li} ids.")
            return {}
        if ll > li:
            if ls == 0:
                res = {i: l for i, l in self._itersel(zip(ids, level), tuple(range(li)))}
                discarded = [l for l in self._itersel(level, tuple(range(li, ll)))]
                self.logger.warning(f"""Index level (length {ll}) has more values than needed for {li} ids and no selector has been passed.
Using the first {li} elements, discarding {discarded}""")
            elif ls != li:
                self.logger.error(f"The selector for picking elements from the overlong index level (length {ll}) should have length {li}, not {ls},")
                res = {}
            else:
                if ls != ll:
                    discarded = [l for l in self._itersel(level, selector, opposite=True)]
                    plural_s = 's' if len(discarded) > 1 else ''
                    self.logger.debug(f"Selector {selector} was applied, leaving out the index value{plural_s} {discarded}")
                res = {i: l for i, l in zip(ids, self._itersel(level, selector))}
        else:
            res = {i: l for i, l in zip(ids, level)}
        return res



    def _treat_key_param(self, keys):
        if keys is None:
            keys = list(self.full_paths.keys())
        elif isinstance(keys, str):
            keys = [keys]
        return keys

    def _treat_label_type_param(self, label_type):
        if label_type is None:
            return None
        all_types = {str(k): k for k in self.count_label_types().keys()}
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

    # def expand_labels(self, keys=None, how='dcml'):
    #     keys = self._treat_key_param(keys)
    #     scores = {id: score for id, score in self._parsed.items() if id[0] in keys}
    #     res = {}
    #     for id, score in scores.items():
    #         if score.mscx._annotations is not None:
    #             exp = score.annotations.expanded
    #             self._expandedlists[id] = exp
    #             res[id + ('expanded',)] = exp
    #     return res


    def __getitem__(self, item):
        if item in self.fexts:
            fexts = self.fexts[item]
            res = {}
            for i, ext in enumerate(fexts):
                id = (item, i)
                ix = str(self._index[id])
                if ext == '.mscx':
                    if id in self._parsed_mscx:
                        ix += " (parsed)"
                        val = str(self._parsed_mscx[id])
                    else:
                        ix += " (not parsed)"
                        val = self.full_paths[item][i]
                else:
                    if id in self._parsed_tsv:
                        ix += " (parsed)"
                        val = self._parsed_tsv[id].head(5).to_string()
                    else:
                        ix += " (not parsed)"
                        val = self.full_paths[item][i]
                ix += f"\n{'-' * len(ix)}\n\n"
                print(ix, val, '\n')

    def __repr__(self):
        return self.info(return_str=True)







