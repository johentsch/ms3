import sys, os, re
import json
import traceback
import pathos.multiprocessing as mp
from collections import Counter, defaultdict
from collections.abc import Collection

import pandas as pd
import numpy as np

from .annotations import Annotations
from .logger import LoggedClass, get_logger
from .score import Score
from .utils import add_quarterbeats_col, DCML_DOUBLE_REGEX, get_musescore, group_id_tuples, join_tsvs, load_tsv, \
    make_continuous_offset, make_id_tuples, metadata2series, next2sequence, no_collections_no_booleans, pretty_dict,\
    resolve_dir, scan_directory, column_order, string2lines, unfold_repeats, update_labels_cfg, write_metadata


class Parse(LoggedClass):
    """
    Class for storing and manipulating the information from multiple parses (i.e. :obj:`~ms3.score.Score` objects).
    """

    def __init__(self, directory=None, paths=None, key=None, index=['rel_paths', 'fnames'], file_re=None, folder_re='.*', exclude_re=r"^(\.|_)",
                 recursive=True, simulate=False, labels_cfg={}, logger_cfg={}, ms=None):
        """

        Parameters
        ----------
        directory, key, index, file_re, folder_re, exclude_re, recursive : optional
            Arguments for the method :py:meth:`~ms3.parse.add_folder`.
            If ``dir`` is not passed, no files are added to the new object except if you pass ``paths``
        paths : :obj:`~collections.abc.Collection` or :obj:`str`, optional
            List of file paths you want to add. If ``dir`` is also passed, all files will be combined in the same object.
            WARNING: If you want to use a custom index, don't use both arguments simultaneously.
        index : element or :obj:`~collections.abc.Collection` of {'key', 'i', :obj:`~collections.abc.Collection`, 'full_paths', 'rel_paths', 'scan_paths', 'paths', 'files', 'fnames', 'fexts'}
            | Change this parameter if you want to create particular indices for output DataFrames.
            | The resulting index must be unique (for identification) and have as many elements as added files.
            | Every single element or Collection of elements ∈
              {'key', 'i', :obj:`~collections.abc.Collection`, 'full_paths', 'rel_paths', 'scan_paths', 'paths', 'files', 'fnames', 'fexts'}
              stands for an index level in the :obj:`~pandas.core.indexes.multi.MultiIndex`.
            | If you pass a Collection that does not start with one of the defined keywords, it is interpreted as an
              index level itself and needs to have at least as many elements as the number of added files.
            | The default ``None`` is equivalent to passing ``(key, i)``, i.e. a MultiIndex of IDs which is always unique.
            | The keywords correspond to the dictionaries of Parse object that contain the constituents of the file paths.
        simulate : :obj:`bool`, optional
            Pass True if no parsing is actually to be done.
        logger_cfg : :obj:`dict`, optional
            | The following options are available:
            | 'name': LOGGER_NAME -> by default the logger name is based on the parsed file(s)
            | 'level': {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}
            | 'path': Directory in which log files are stored. If 'file' is relative, this path is used as root, otherwise, it is ignored.
            | 'file': PATH_TO_LOGFILE Pass absolute path to store all log messages in a single log file.
              If PATH_TO_LOGFILE is relative, multiple log files are created dynamically, relative to the original MSCX files' paths.
              If 'path' is set, the corresponding subdirectory structure is created there.
        ms : :obj:`str`, optional
            If you want to parse musicXML files or MuseScore 2 files by temporarily converting them, pass the path or command
            of your local MuseScore 3 installation. If you're using the standard path, you may try 'auto', or 'win' for
            Windows, 'mac' for MacOS, or 'mscore' for Linux.
        """
        if 'file' in logger_cfg and logger_cfg['file'] is not None and not os.path.isabs(logger_cfg['file']) and ('path' not in logger_cfg or logger_cfg['path'] is None):
            # if the log 'file' is relative but 'path' is not defined, Parse.log will be stored under `dir`;
            # if `dir` is also None, Parse.log will not be created and a warning will be shown.
            logger_cfg['path'] = directory
        super().__init__(subclass='Parse', logger_cfg=logger_cfg)
        self.simulate=simulate
        # defaultdicts with keys as keys, each holding a list with file information (therefore accessed via [key][i] )
        self.full_paths = defaultdict(list)
        """:obj:`collections.defaultdict`
        ``{key: [full_path]}`` dictionary of the full paths of all detected files.
        """

        self.rel_paths = defaultdict(list)
        """:obj:`collections.defaultdict`
        ``{key: [rel_path]}`` dictionary of the relative (to :obj:`.scan_paths`) paths of all detected files.
        """

        self.scan_paths = defaultdict(list)
        """:obj:`collections.defaultdict`
        ``{key: [scan_path]}`` dictionary of the scan_paths from which each file was detected.
        """

        self.paths = defaultdict(list)
        """:obj:`collections.defaultdict`
        ``{key: [path]}`` dictionary of the paths of all detected files (without file name).
        """

        self.files = defaultdict(list)
        """:obj:`collections.defaultdict`
        ``{key: [file]}`` dictionary of file names with extensions of all detected files.
        """

        self.fnames = defaultdict(list)
        """:obj:`collections.defaultdict`
        ``{key: [fname]}`` dictionary of file names without extensions of all detected files.
        """

        self.fexts = defaultdict(list)
        """:obj:`collections.defaultdict`
        ``{key: [fext]}`` dictionary of file extensions of all detected files.
        """

        self.logger_names = {'root': 'Parse'}
        """:obj:`dict`
        ``{(key, i): :obj:`str`}`` dictionary of logger names.
        """

        self._ms = get_musescore(ms, logger=self.logger)
        """:obj:`str`
        Path or command of the local MuseScore 3 installation if specified by the user."""


        self._parsed_mscx = {}
        """:obj:`dict`
        ``{(key, i): :obj:`~ms3.score.Score`}`` dictionary of parsed scores.
        """

        self._annotations = {}
        """:obj:`dict`
        {(key, i): :obj:`~ms3.annotations.Annotations`} dictionary of parsed sets of annotations.
        """

        self._notelists = {}
        """:obj:`dict`
        {(key, i): :obj:`pandas.DataFrame`} dictionary of DataFrames holding :obj:`~ms3.score.Score.notes` tables.
        """

        self._restlists = {}
        """:obj:`dict`
        {(key, i): :obj:`pandas.DataFrame`} dictionary of DataFrames holding :obj:`~ms3.score.Score.rests` tables 
        """

        self._noterestlists = {}
        """:obj:`dict`
        {(key, i): :obj:`pandas.DataFrame`} dictionary of DataFrames holding :obj:`~ms3.score.Score.notes_and_rests` tables
        """

        self._eventlists = {}
        """:obj:`dict`
        {(key, i): :obj:`pandas.DataFrame`} dictionary of DataFrames holding :obj:`~ms3.score.Score.events` tables.
        """

        self._labellists = {}
        """:obj:`dict`
        {(key, i): :obj:`pandas.DataFrame`} dictionary of DataFrames holding :obj:`~ms3.score.Score.labels` tables.
        """

        self._chordlists = {}
        """:obj:`dict`
        {(key, i): :obj:`pandas.DataFrame`} dictionary of DataFrames holding :obj:`~ms3.score.Score.chords` tables.
        """

        self._expandedlists = {}
        """:obj:`dict`
        {(key, i): :obj:`pandas.DataFrame`} dictionary of DataFrames holding :obj:`~ms3.score.Score.expanded` tables.
        """

        self._cadencelists = {}
        """:obj:`dict`
        {(key, i): :obj:`pandas.DataFrame`} dictionary of DataFrames holding :obj:`~ms3.score.Score.cadences` tables.
        """

        self._index = {}
        """:obj:`dict`
        {(key, i): :obj:`tuple`} dictionary of index tuples where every element represents one index level.
        """

        self._levelnames = {}
        """:obj:`dict`
        {(key, i): :obj:`tuple`} dictionary of index names for the corresponding index tuple.
        """

        self._measurelists = {}
        """:obj:`dict`
        {(key, i): :obj:`pandas.DataFrame`} dictionary of DataFrames holding :obj:`~ms3.score.Score.measures` tables.
        """

        self._metadata = pd.DataFrame()
        """:obj:`pandas.DataFrame`
        Concatenation of all parsed metadata TSVs.
        """

        self._parsed_tsv = {}
        """:obj:`dict`
        {(key, i): :obj:`pandas.DataFrame`} dictionary of all parsed (i.e. loaded as DataFrame) TSV files.
        """

        self._tsv_types = {}
        """:obj:`dict`
        {(key, i): :obj:`str`} dictionary of TSV types as inferred by :py:meth:`._infer_tsv_type`, i.e. one of
        ``None, 'notes', 'events', 'chords', 'rests', 'measures', 'labels'}``
        """

        self._unfolded_mcs = {}
        """:obj:`dict`
        {(key, i): :obj:`pandas.Series`} dictionary of a parsed score's MC succession after 'unfolding' all repeats.
        """

        self._quarter_offsets = {True: {}, False: {}}
        """:obj:`dict`
        { unfolded? -> {(key, i) -> {mc_playthrough -> quarter_offset}} } dictionary with keys True and false.
        True: For every mc_playthrough (i.e., after 'unfolding' all repeats) the total sum of preceding quarter beats, measured from m. 1, b. 0. 
        False: For every mc the total sum of preceding quarter beats after deleting all but second endings.
        """

        self.labels_cfg = {
            'staff': None,
            'voice': None,
            'label_type': None,
            'positioning': True,
            'decode': False,
            'column_name': 'label',
        }
        """:obj:`dict`
        Configuration dictionary to determine the output format of :obj:`~ms3.score.Score.labels` and
        :obj:`~ms3.score.Score.expanded` tables. The dictonary is passed to :obj:`~ms3.score.Score` upon parsing.
        """
        self.labels_cfg.update(update_labels_cfg(labels_cfg, logger=self.logger))

        self._lists = {
            'notes': self._notelists,
            'rests': self._restlists,
            'notes_and_rests': self._noterestlists,
            'measures': self._measurelists,
            'events': self._eventlists,
            'labels': self._labellists,
            'chords': self._chordlists,
            'expanded': self._expandedlists,
            'cadences': self._cadencelists,
        }
        """:obj:`dict`
        Dictionary exposing the different :obj:`dicts<dict>` of :obj:`DataFrames<pandas.DataFrame>`.
        """

        self._possible_levels = {
            'full_paths': self.full_paths,
            'rel_paths': self.rel_paths,
            'scan_paths': self.scan_paths,
            'paths': self.paths,
            'files': self.files,
            'fnames': self.fnames,
            'fexts': self.fexts,
        }
        """:obj:`dict`
        Dictionary including the various constituents of the file paths which can be used as index levels.
        """

        self._matches = pd.DataFrame(columns=['scores']+list(self._lists.keys()))
        """:obj:`pandas.DataFrame`
        Dataframe that holds the (file name) matches between MuseScore and TSV files.
        """



        self.last_scanned_dir = directory
        """:obj:`str`
        The directory that was scanned for files last.
        """
        if directory is not None:
            if isinstance(directory, str):
                directory = [directory]
            for d in directory:
                self.add_dir(directory=d, key=key, index=index, file_re=file_re, folder_re=folder_re, exclude_re=exclude_re, recursive=recursive)
        if paths is not None:
            if isinstance(paths, str):
                paths = [paths]
            if len(paths) == 1 and paths[0].endswith('.json'):
                # TODO: allow for any number of JSON files
                with open(paths[0]) as f:
                    paths = json.load(f)
            _ = self.add_files(paths, key=key, index=index, exclude_re=exclude_re)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END of __init__() %%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


    def _concat_lists(self, which, keys=None, ids=None, quarterbeats=False, unfold=False):
        """ Boiler plate for concatenating DataFrames with the same type of information.

        Parameters
        ----------
        which : {'cadences', 'chords', 'events', 'expanded', 'labels', 'measures', 'notes_and_rests', 'notes', 'rests'}
        keys
        ids

        Returns
        -------

        """
        # if quarterbeats and not unfold:
        #     self.logger.info('Adding quarterbeats without unfolding repeats has not been implemented yet, sorry.')
        #     quarterbeats = False
        d = self.get_lists(keys, ids, flat=False, quarterbeats=quarterbeats, unfold=unfold, **{which: True})
        d = d[which] if which in d else {}
        msg = {
            'cadences': 'cadence lists',
            'chords': '<chord> tables',
            'events': 'event tables',
            'expanded': 'expandable annotation tables',
            'labels': 'annotation tables',
            'measures': 'measure lists',
            'notes': 'note lists',
            'notes_and_rests': 'note and rest lists',
            'rests': 'rest lists',
        }
        if len(d) == 0:
            if keys is None and ids is None:
                self.logger.info(f'This Parse object does not include any {msg[which]}.')
            else:
                self.logger.info(f'keys={keys}, ids={ids}, does not yield any {msg[which]}.')
            return pd.DataFrame()
        return pd.concat(d.values(), keys=d.keys())

    def cadences(self, keys=None, ids=None, quarterbeats=False, unfold=False):
        return self._concat_lists('cadences', keys, ids, quarterbeats=quarterbeats, unfold=unfold)

    def chords(self, keys=None, ids=None, quarterbeats=False, unfold=False):
        return self._concat_lists('chords', keys, ids, quarterbeats=quarterbeats, unfold=unfold)

    def events(self, keys=None, ids=None, quarterbeats=False, unfold=False):
        return self._concat_lists('events', keys, ids, quarterbeats=quarterbeats, unfold=unfold)

    def expanded(self, keys=None, ids=None, quarterbeats=False, unfold=False):
        return self._concat_lists('expanded', keys, ids, quarterbeats=quarterbeats, unfold=unfold)

    def labels(self, keys=None, ids=None, quarterbeats=False, unfold=False):
        return self._concat_lists('labels', keys, ids, quarterbeats=quarterbeats, unfold=unfold)

    def measures(self, keys=None, ids=None, quarterbeats=False, unfold=False):
        return self._concat_lists('measures', keys, ids, quarterbeats=quarterbeats, unfold=unfold)

    def notes(self, keys=None, ids=None, quarterbeats=False, unfold=False):
        return self._concat_lists('notes', keys, ids, quarterbeats=quarterbeats, unfold=unfold)

    def notes_and_rests(self, keys=None, ids=None, quarterbeats=False, unfold=False):
        return self._concat_lists('notes_and_rests', keys, ids, quarterbeats=quarterbeats, unfold=unfold)

    def rests(self, keys=None, ids=None, quarterbeats=False, unfold=False):
        return self._concat_lists('rests', keys, ids, quarterbeats=quarterbeats, unfold=unfold)

    @property
    def ids(self):
        d = self._index
        return pd.DataFrame({'id': d.keys()}, index=d.values()).sort_index()


    @property
    def ms(self):
        return self._ms

    @ms.setter
    def ms(self, ms):
        self._ms = get_musescore(ms)


    @property
    def parsed_mscx(self):
        """:obj:`pandas.DataFrame`
        Returns an overview of the parsed scores."""
        if len(self._parsed_mscx) == 0:
            self.logger.info("No scores have been parsed yet. Use parse() or parse_mscx()")
            return None
        ids = list(self._iterids(only_parsed_mscx=True))
        ix = self.ids2idx(ids, pandas_index=True)
        paths = pd.Series([os.path.join(self.rel_paths[k][i], self.files[k][i]) for k, i in ids], index=ix, name='paths')
        attached = pd.Series([len(self._parsed_mscx[id].annotations.df) if self._parsed_mscx[id].annotations is not None else 0 for id in ids],
                             index=ix, name='labels')
        detached_keys = [', '.join(self._parsed_mscx[id]._detached_annotations.keys()) if len(
            self._parsed_mscx[id]._detached_annotations) > 0 else None for id in ids]
        if all(k is None for k in detached_keys):
            res = pd.concat([paths, attached], axis=1)
        else:
            detached_keys = pd.Series(detached_keys, index=ix,
                                  name='detached_annotations')
            res = pd.concat([paths, attached, detached_keys], axis=1)
        return res.sort_index()

    @property
    def parsed_tsv(self):
        """:obj:`pandas.DataFrame`
        Returns an overview of the parsed TSV files."""
        if len(self._parsed_tsv) == 0:
            self.logger.info("No TSV files have been parsed yet. Use parse() or parse_tsv()")
            return None
        ids = list(self._iterids(only_parsed_tsv=True))
        ix = self.ids2idx(ids, pandas_index=True)
        paths = pd.Series([os.path.join(self.rel_paths[k][i], self.files[k][i]) for k, i in ids], index=ix, name='paths')
        types = pd.Series([self._tsv_types[id] for id in ids], index=ix, name='types')
        res = pd.concat([paths, types], axis=1)
        return res.sort_index()




    def add_detached_annotations(self, score_key=None, tsv_key=None, new_key=None, match_dict=None):
        """ Add :obj:`~ms3.annotations.Annotations` objects generated from TSV files to the :obj:`~ms3.score.Score`
        objects to which they are being matched based on their filenames or on ``match_dict``.

        Parameters
        ----------
        score_key : :obj:`str`, optional
            A key under which parsed MuseScore files are stored.
            If one of ``score_key`` and ``tsv_key`` is None, no matching is performed and already matched files are used.
        tsv_key : :obj:`str`, optional
            A key under which parsed TSV files are stored of which the type has been inferred as 'labels'.
            If one of ``score_key`` and ``tsv_key`` is None, no matching is performed and already matched files are used.
        new_key : :obj:`str`, optional
            The key under which the :obj:`~ms3.annotations.Annotations` objects will be available after attaching
            them to the :obj:`~ms3.score.Score` objects (``Parsed.parsed_mscx[ID].key``). By default, ``tsv_key``
            is used.
        match_dict : :obj:`dict`, optional
            Dictionary mapping IDs of parsed :obj:`~ms3.score.Score` objects to IDs of parsed :obj:`~ms3.annotations.Annotations`
            objects.
        """
        if new_key is None:
            new_key = tsv_key
        if match_dict is None:
            if score_key is not None and tsv_key is not None:
                matches = self.match_files(keys=[score_key, tsv_key])
            else:
                matches = self._matches[self._matches.labels.notna() | self._matches.expanded.notna()]
            matches.labels.fillna(matches.expanded, inplace=True)
            match_dict = dict(matches[['scores', 'labels']].values)
        if len(match_dict) == 0:
            self.logger.info(f"No files could be matched based on file names, have you added the folder containing annotation tables?"
                    f"Instead, you could pass the match_dict argument with a mapping of Score IDs to Annotations IDs.")
            return
        for score_id, labels_id in match_dict.items():
            if score_id in self._parsed_mscx and not pd.isnull(labels_id):
                if labels_id in self._annotations:
                    k = labels_id[0] if pd.isnull(new_key) else new_key
                    try:
                        self._parsed_mscx[score_id].load_annotations(anno_obj=self._annotations[labels_id], key=k)
                    except:
                        print(f"score_id: {score_id}, labels_id: {labels_id}")
                        raise
                else:
                    k, i = labels_id
                    self.logger.warning(f"""The TSV {labels_id} has not yet been parsed as Annotations object.
Use parse_tsv(key='{k}') and specify cols={{'label': label_col}}.""")
            elif score_id not in self._parsed_mscx:
                self.logger.info(f"{self._index[score_id]} has not been parsed yet.")
            else:
                self.logger.debug(f"Nothing to add to {score_id}. Make sure that its counterpart has been recognized as tsv_type 'labels' or 'expanded'.")





    def add_dir(self, directory, key=None, index=['rel_paths', 'fnames'], file_re=None, folder_re='.*', exclude_re=r"^(\.|__)", recursive=True):
        """
        This method scans the directory ``dir`` for files matching the criteria and adds them (i.e. paths and file names)
        to the Parse object without looking at them. It is recommended to add different types of files with different keys,
        e.g. 'mscx' for score, 'harmonies' for chord labels, and 'form' for form labels.

        Parameters
        ----------
        directory : :obj:`str`
            Directory to scan for files.
        key : :obj:`str`, optional
            | Pass a string to identify the loaded files.
            | By default, the relative sub-directories of ``dir`` are used as keys. For example, for files within ``dir``
              itself, the key would be ``'.'``, for files in the subfolder ``scores`` it would be ``'scores'``, etc.
        index : element or :obj:`~collections.abc.Collection` of {'key', 'i', :obj:`~collections.abc.Collection`, 'full_paths', 'rel_paths', 'scan_paths', 'paths', 'files', 'fnames', 'fexts'}
            | Change this parameter if you want to create particular indices for output DataFrames.
            | The resulting index must be unique (for identification) and have as many elements as added files.
            | Every single element or Collection of elements ∈
              {'key', 'i', :obj:`~collections.abc.Collection`, 'full_paths', 'rel_paths', 'scan_paths', 'paths', 'files', 'fnames', 'fexts'}
              stands for an index level in the :obj:`~pandas.core.indexes.multi.MultiIndex`.
            | If you pass a Collection that does not start with one of the defined keywords, it is interpreted as an
              index level itself and needs to have at least as many elements as the number of added files.
            | The default ``None`` is equivalent to passing ``(key, i)``, i.e. a MultiIndex of IDs which is always unique.
            | The keywords correspond to the dictionaries of Parse object that contain the constituents of the file paths.
        directory : :obj:`str`
            Directory to be scanned for files.
        file_re : :obj:`str`, optional
            Regular expression for filtering certain file names. By default, all parseable score files and TSV files are detected.
            The regEx is checked with search(), not match(), allowing for fuzzy search.
        folder_re : :obj:`str`, optional
            Regular expression for filtering certain folder names.
            The regEx is checked with search(), not match(), allowing for fuzzy search.
        recursive : :obj:`bool`, optional
            By default, sub-directories are recursively scanned. Pass False to scan only ``dir``.
        """
        directory = resolve_dir(directory)
        self.last_scanned_dir = directory
        if file_re is None:
            file_re = Score._make_extension_regex(tsv=True)
        paths = tuple(scan_directory(directory, file_re=file_re, folder_re=folder_re, exclude_re=exclude_re, recursive=recursive, logger=self.logger))
        _ = self.add_files(paths=paths, key=key, index=index)


    def add_files(self, paths, key, index=None, exclude_re=None):
        """

        Parameters
        ----------
        paths : :obj:`~collections.abc.Collection`
            The paths of the files you want to add to the object.
        key : :obj:`str`
            | Pass a string to identify the loaded files.
            | If None is passed, paths relative to :py:attr:`last_scanned_dir` are used as keys. If :py:meth:`add_dir`
              hasn't been used before, the longest common prefix of all paths is used.
        index : element or :obj:`~collections.abc.Collection` of {'key', 'i', :obj:`~collections.abc.Collection`, 'full_paths', 'rel_paths', 'scan_paths', 'paths', 'files', 'fnames', 'fexts'}
            | Change this parameter if you want to create particular indices for output DataFrames.
            | The resulting index must be unique (for identification) and have as many elements as added files.
            | Every single element or Collection of elements ∈
              {'key', 'i', :obj:`~collections.abc.Collection`, 'full_paths', 'rel_paths', 'scan_paths', 'paths', 'files', 'fnames', 'fexts'}
              stands for an index level in the :obj:`~pandas.core.indexes.multi.MultiIndex`.
            | If you pass a Collection that does not start with one of the defined keywords, it is interpreted as an
              index level itself and needs to have at least as many elements as the number of added files.
            | The default ``None`` is equivalent to passing ``(key, i)``, i.e. a MultiIndex of IDs which is always unique.
            | The keywords correspond to the dictionaries of Parse object that contain the constituents of the file paths.

        Returns
        -------
        :obj:`list`
            The IDs of the added files.
        """
        if paths is None or len(paths) == 0:
            self.logger.debug(f"add_files() was called with paths = '{paths}'.")
            return []
        if isinstance(paths, str):
            paths = [paths]
        if exclude_re is not None:
            paths = [p for p in paths if re.search(exclude_re, p) is None]
        if self.last_scanned_dir is None:
            # if len(paths) > 1:
            #     self.last_scanned_dir = commonprefix(paths, os.path.sep)
            # else:
            #     self.last_scanned_dir = os.path.dirname(paths[0])
            self.last_scanned_dir = os.getcwd()

        ids = [self._handle_path(p, key) for p in paths]
        if sum(True for x in ids if x[0] is not None) > 0:
            selector, added_ids = zip(*[(i, x) for i, x in enumerate(ids) if x[0] is not None])
            grouped_ids = group_id_tuples(ids)
            exts = self.count_extensions(ids=ids, per_key=True)
            self.logger.debug(f"{len(added_ids)} paths stored:\n{pretty_dict(exts, 'EXTENSIONS')}")
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
            return added_ids
        else:
            self.logger.info("No files added.")
            return []


    def add_rel_dir(self, rel_dir, suffix='', score_extensions=None, keys=None, new_key=None, index=None):
        """
        This method can be used for adding particular TSV files belonging to already loaded score files. This is useful,
        for example, to add annotation tables for comparison.

        Parameters
        ----------
        rel_dir : :obj:`str`
            Path where the files to be added can be found, relative to each loaded MSCX file. They are expected to have
            the same file name, maybe with an added ``suffix``.
        suffix : :obj:`str`. optional
            If the files to be loaded can be identified by adding a suffix to the filename, pass this suffix, e.g. '_labels'.
        score_extensions : :obj:`~collections.abc.Collection`, optional
            If you want to match only scores with particular extensions, pass a Collection of these extensions.
        keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
            Key(s) under which score files are stored. By default, all keys are selected.
        new_key : :obj:`str`, optional
            Pass a string to identify the loaded files. By default, the keys of the score files are being used.
        index : element or :obj:`~collections.abc.Collection` of {'key', 'i', :obj:`~collections.abc.Collection`, 'full_paths', 'rel_paths', 'scan_paths', 'paths', 'files', 'fnames', 'fexts'}
            | Change this parameter if you want to create particular indices for output DataFrames.
            | The resulting index must be unique (for identification) and have as many elements as added files.
            | Every single element or Collection of elements ∈
              {'key', 'i', :obj:`~collections.abc.Collection`, 'full_paths', 'rel_paths', 'scan_paths', 'paths', 'files', 'fnames', 'fexts'}
              stands for an index level in the :obj:`~pandas.core.indexes.multi.MultiIndex`.
            | If you pass a Collection that does not start with one of the defined keywords, it is interpreted as an
              index level itself and needs to have at least as many elements as the number of added files.
            | The default ``None`` is equivalent to passing ``(key, i)``, i.e. a MultiIndex of IDs which is always unique.
            | The keywords correspond to the dictionaries of Parse object that contain the constituents of the file paths.
        """
        ids = self._score_ids(keys, score_extensions)
        grouped_ids = group_id_tuples(ids)
        self.logger.debug(f"{len(ids)} scores match the criteria.")
        expected_paths = {(k, i): os.path.join(self.paths[k][i], rel_dir, self.fnames[k][i] + suffix + '.tsv') for k, i in ids}
        existing = {k: [] for k in grouped_ids.keys()}
        for (k, i), path in expected_paths.items():
            if os.path.isfile(path):
                existing[k].append(path)
            else:
                ids.remove((k, i))
        existing = {k: v for k, v in existing.items() if len(v) > 0}
        self.logger.debug(f"{sum(len(paths) for paths in existing.values())} paths found for rel_dir {rel_dir}.")
        if index is None:
            if any(any(n is None for n in self._levelnames[k]) for k in existing.keys()):
                if new_key is None:
                    raise ValueError(f"There are custom index levels and this function cannot extend them. Pass the 'new_key' argument.")
                else:
                    index_levels = {k: None for k in existing.keys()}
            else:
                index_levels = {k: self._levelnames[k] for k in existing.keys()}
        else:
            index_levels = {k: index for k in existing.keys()}
        new_ids = []
        for k, paths in existing.items():
            key_param = k if new_key is None else new_key
            new_ids.extend(self.add_files(paths, key_param, index_levels[k]))
        self.parse_tsv(ids=new_ids)
        for score_id, tsv_id in zip(ids, new_ids):
            ix = self._index[score_id]
            tsv_type = self._tsv_types[tsv_id]
            if ix in self._matches.index:
                self._matches.loc[ix, tsv_type] = tsv_id
            else:
                row = pd.DataFrame.from_dict({ix: {'scores': score_id, tsv_type: tsv_id}}, orient='index')
                self._matches = pd.concat([self._matches, row])









    def attach_labels(self, keys=None, annotation_key=None, staff=None, voice=None, label_type=None, check_for_clashes=True):
        """ Attach all :obj:`~ms3.annotations.Annotations` objects that are reachable via ``Score.annotation_key`` to their
        respective :obj:`~ms3.score.Score`, changing their current XML. Calling :py:meth:`.store_mscx` will output
        MuseScore files where the annotations show in the score.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
            Key(s) under which parsed MuseScore files are stored. By default, all keys are selected.
        annotation_key : :obj:`str` or :obj:`list` or :obj:`tuple`, optional
            Key(s) under which the :obj:`~ms3.annotations.Annotations` objects to be attached are stored in the
            :obj:`~ms3.score.Score` objects. By default, all keys are selected.
        staff : :obj:`int`, optional
            If you pass a staff ID, the labels will be attached to that staff where 1 is the upper stuff.
            By default, the staves indicated in the 'staff' column of :obj:`ms3.annotations.Annotations.df`
            will be used.
        voice : {1, 2, 3, 4}, optional
            If you pass the ID of a notational layer (where 1 is the upper voice, blue in MuseScore),
            the labels will be attached to that one.
            By default, the notational layers indicated in the 'voice' column of
            :obj:`ms3.annotations.Annotations.df` will be used.
        check_for_clashes : :obj:`bool`, optional
            By default, warnings are thrown when there already exists a label at a position (and in a notational
            layer) where a new one is attached. Pass False to deactivate these warnings.
        """
        layers = self.count_annotation_layers(keys, which='detached', per_key=True)
        if len(layers) == 0:
            ks = '' if keys is None else ' under the key(s) ' + keys
            self.logger.warning(f"No detached annotations found{ks}.")
            return
        if annotation_key is None:
            annotation_key = list(layers.keys())
        elif isinstance(annotation_key, str):
            annotation_key = [annotation_key]
        if any(True for k in annotation_key if k not in layers):
            wrong = [k for k in annotation_key if k not in layers]
            annotation_key = [k for k in annotation_key if k in layers]
            if len(annotation_key) == 0:
                self.logger.error(
f"""'{wrong}' are currently not keys for sets of detached labels that have been added to parsed scores.
Currently available annotation keys are {list(layers.keys())}""")
                return
            else:
                self.logger.warning(
f"""'{wrong}' are currently not keys for sets of detached labels that have been added to parsed scores.
Continuing with {annotation_key}.""")

        ids = list(self._iterids(keys, only_detached_annotations=True))
        reached, goal = 0, 0
        for id in ids:
            for anno_key in annotation_key:
                if anno_key in self._parsed_mscx[id]:
                    r, g = self._parsed_mscx[id].attach_labels(anno_key, staff=staff, voice=voice, label_type=label_type, check_for_clashes=check_for_clashes)
                    self.logger.info(f"{r}/{g} labels successfully added to {self.files[id[0]][id[1]]}")
                    reached += r
                    goal += g
        self.logger.info(f"{reached}/{goal} labels successfully added to {len(ids)} files.")
        self._collect_annotations_objects_references(ids=ids)


    def change_labels_cfg(self, labels_cfg={}, staff=None, voice=None, label_type=None, positioning=None, decode=None):
        """ Update :obj:`Parse.labels_cfg` and retrieve new 'labels' tables accordingly.

        Parameters
        ----------
        labels_cfg : :obj:`dict`
            Using an entire dictionary or, to change only particular options, choose from:
        staff, voice, label_type, positioning, decode
            Arguments as they will be passed to :py:meth:`~ms3.annotations.Annotations.get_labels`
        """
        for k in self.labels_cfg.keys():
            val = locals()[k]
            if val is not None:
                labels_cfg[k] = val
        self.labels_cfg.update(update_labels_cfg(labels_cfg, logger=self.logger))
        ids = list(self._labellists.keys())
        if len(ids) > 0:
            self.collect_lists(ids=ids, labels=True)


    def check_labels(self, keys=None, ids=None):
        if len(self._parsed_mscx) == 0:
            self.logger.info("No scores have been parsed so far. Use parse_mscx()")
            return
        if ids is None:
            ids = list(self._iterids(keys, only_parsed_mscx=True))
        checks = {id: self._parsed_mscx[id].check_labels() for id in ids}
        checks = {k: v for k, v in checks.items() if v is not None and len(v) > 0}
        if len(checks) > 0:
            idx = self.ids2idx(checks.keys(), pandas_index=True)
            return pd.concat(checks.values(), keys=idx, names=idx.names)
        return pd.DataFrame()



    def collect_lists(self, keys=None, ids=None, notes=False, rests=False, notes_and_rests=False, measures=False, events=False,
                      labels=False, chords=False, expanded=False, cadences=False, only_new=True):
        """ Extracts DataFrames from the parsed scores in ``keys`` and stores them in dictionaries.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
            Key(s) under which parsed MuseScore files are stored. By default, all keys are selected.
        ids : :obj:`~collections.abc.Collection`
            If you pass a collection of IDs, ``keys`` is ignored and ``only_new`` is set to False.
        notes, rests, notes_and_rests, measures, events, labels, chords, expanded, cadences : :obj:`bool`, optional
        only_new : :obj:`bool`, optional
            Set to False to also retrieve lists that had already been retrieved.
        """
        if len(self._parsed_mscx) == 0 and len(self._parsed_tsv) == 0:
            self.logger.debug("Nothing has been parsed so far.")
            return
        if ids is None:
            only_new = False
            ids = list(self._iterids(keys, only_parsed_mscx=True))
        scores = {id: self._parsed_mscx[id] for id in ids if id in self._parsed_mscx}
        bool_params = list(self._lists.keys())
        l = locals()
        params = {p: l[p] for p in bool_params}

        for i, score in scores.items():
            for param, li in self._lists.items():
                if params[param] and (i not in li or not only_new):
                    if self.simulate:
                        df = pd.DataFrame()
                    else:
                        df = score.mscx.__getattribute__(param)
                    if df is not None:
                        li[i] = df


    def compare_labels(self, detached_key, new_color='ms3_darkgreen', old_color='ms3_darkred',
                       detached_is_newer=False, store_with_suffix=None):
        """ Compare detached labels ``key`` to the ones attached to the Score.
        By default, the attached labels are considered as the reviewed version and changes are colored in green;
        Changes with respect to the detached labels are attached to the Score in red.

        Parameters
        ----------
        detached_key : :obj:`str`
            Key under which the detached labels that you want to compare have been added to the scores.
        new_color, old_color : :obj:`str` or :obj:`tuple`, optional
            The colors by which new and old labels are differentiated. Identical labels remain unchanged.
        detached_is_newer : :obj:`bool`, optional
            Pass True if the detached labels are to be added with ``new_color`` whereas the attached changed labels
            will turn ``old_color``, as opposed to the default.
        store_with_suffix : :obj:`str`, optional
            If you pass a suffix, the comparison MSCX files are stored with this suffix next to the originals.
        """
        assert detached_key != 'annotations', "Pass a key of detached labels, not 'annotations'."
        ids = list(self._iterids(None, only_detached_annotations=True))
        if len(ids) == 0:
            if len(self._parsed_mscx) == 0:
                self.logger.warning("No scores have been parsed so far.")
                return
            self.logger.warning("None of the parsed score include detached labels to compare.")
            return
        available_keys = set(k for id in ids for k in self._parsed_mscx[id]._detached_annotations)
        if detached_key not in available_keys:
            self.logger.warning(f"""None of the parsed score include detached labels with the key '{detached_key}'.
Available keys: {available_keys}""")
            return
        ids = [id for id in ids if detached_key in self._parsed_mscx[id]._detached_annotations]
        self.logger.info(f"{len(ids)} parsed scores include detached labels with the key '{detached_key}'.")
        for id in ids:
            res = self._parsed_mscx[id].compare_labels(detached_key=detached_key, new_color=new_color, old_color=old_color,
                                                 detached_is_newer=detached_is_newer)
        if res and store_with_suffix is not None:
            self.store_mscx(ids=ids, suffix=store_with_suffix, overwrite=True, simulate=self.simulate)


    def count_annotation_layers(self, keys=None, which='attached', per_key=False):
        """ Counts the labels for each annotation layer defined as (staff, voice, label_type).
        By default, only labels attached to a score are counted.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
            Key(s) for which to count annotation layers.  By default, all keys are selected.
        which : {'attached', 'detached', 'tsv'}, optional
            'attached': Counts layers from annotations attached to a score.
            'detached': Counts layers from annotations that are in a Score object, but detached from the score.
            'tsv': Counts layers from Annotation objects that have been loaded from or into annotation tables.
        per_key : :obj:`bool`, optional
            If set to True, the results are returned as a dict {key: Counter}, otherwise the counts are summed up in one Counter.
            If ``which='detached'``, the keys are keys from Score objects, otherwise they are keys from this Parse object.

        Returns
        -------
        :obj:`dict` or :obj:`collections.Counter`
            By default, the function returns a Counter of labels for every annotation layer (staff, voice, label_type)
            If ``per_key`` is set to True, a dictionary {key: Counter} is returned, separating the counts.
        """
        res_dict = defaultdict(Counter)

        if which == 'detached':
            for id in self._iterids(keys, only_detached_annotations=True):
                for key, annotations in self._parsed_mscx[id]._detached_annotations.items():
                    if key != 'annotations':
                        _, layers = annotations.annotation_layers
                        layers_dict = {tuple(None if pd.isnull(e) else e for e in t): count for t, count in
                                       layers.to_dict().items()}
                        res_dict[key].update(layers_dict)
        elif which in ['attached', 'tsv']:
            for key, i in self._iterids(keys):
                if (key, i) in self._annotations:
                    ext = self.fexts[key][i]
                    if (which == 'attached' and ext == '.mscx') or (which == 'tsv' and ext != '.mscx'):
                        _, layers = self._annotations[(key, i)].annotation_layers
                        layers_dict = {tuple(None if pd.isnull(e) else e for e in t): count for t, count in
                                       layers.to_dict().items()}
                        res_dict[key].update(layers_dict)
        else:
            self.logger.error(f"Parameter 'which' needs to be one of {{'attached', 'detached', 'tsv'}}, not {which}.")
            return {} if per_key else pd.Series()


        def make_series(counts):
            if len(counts) == 0:
                return pd.Series()
            data = counts.values()
            ks = list(counts.keys())
            #levels = len(ks[0])
            names = ['staff', 'voice', 'label_type', 'color'] #<[:levels]
            try:
                ix = pd.MultiIndex.from_tuples(ks, names=names)
            except:
                cs = {k: v for k, v in counts.items() if len(k) != levels}
                print(f"names: {names}, counts:\n{cs}")
                raise
            return pd.Series(data, ix)

        if per_key:
            res = {k: make_series(v) for k, v in res_dict.items()}
        else:
            res = make_series(sum(res_dict.values(), Counter()))
        if len(res) == 0:
            self.logger.info("No annotations found. Maybe no scores have been parsed using parse_mscx()?")
        return res


    def count_extensions(self, keys=None, ids=None, per_key=False):
        """ Count file extensions.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
            Key(s) for which to count file extensions.  By default, all keys are selected.
        ids : :obj:`~collections.abc.Collection`
            If you pass a collection of IDs, ``keys`` is ignored and only the selected extensions are counted.
        per_key : :obj:`bool`, optional
            If set to True, the results are returned as a dict {key: Counter},
            otherwise the counts are summed up in one Counter.

        Returns
        -------
        :obj:`dict` or :obj:`collections.Counter`
            By default, the function returns a Counter of file extensions.
            If ``per_key`` is set to True, a dictionary {key: Counter} is returned, separating the counts.
        """

        res_dict = {}
        if ids is not None:
            grouped_ids = group_id_tuples(ids)
            for k, ixs in grouped_ids.items():
                res_dict[k] = Counter(self._itersel(self.fexts[k], ixs))
        else:
            keys = self._treat_key_param(keys)
            for key in keys:
                res_dict[key] = Counter(self.fexts[key])
        if per_key:
            return {k: dict(v) for k, v in res_dict.items()}
        return dict(sum(res_dict.values(), Counter()))



    def count_label_types(self, keys=None, per_key=False):
        """ Count label types.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
            Key(s) for which to count label types.  By default, all keys are selected.
        per_key : :obj:`bool`, optional
            If set to True, the results are returned as a dict {key: Counter},
            otherwise the counts are summed up in one Counter.

        Returns
        -------
        :obj:`dict` or :obj:`collections.Counter`
            By default, the function returns a Counter of label types.
            If ``per_key`` is set to True, a dictionary {key: Counter} is returned, separating the counts.
        """
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



    def count_tsv_types(self, keys=None, per_key=False):
        """ Count inferred TSV types.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
            Key(s) for which to count inferred TSV types.  By default, all keys are selected.
        per_key : :obj:`bool`, optional
            If set to True, the results are returned as a dict {key: Counter},
            otherwise the counts are summed up in one Counter.

        Returns
        -------
        :obj:`dict` or :obj:`collections.Counter`
            By default, the function returns a Counter of inferred TSV types.
            If ``per_key`` is set to True, a dictionary {key: Counter} is returned, separating the counts.
        """
        res_dict = defaultdict(Counter)
        for key, i in self._iterids(keys, only_parsed_mscx=True):
            t = self._tsv_types[(key, i)] if (key, i) in self._tsv_types else None
            res_dict[key].update([t])
        if per_key:
            return {k: dict(v) for k, v in res_dict.items()}
        return dict(sum(res_dict.values(), Counter()))



    def detach_labels(self, keys=None, annotation_key='detached', staff=None, voice=None, label_type=None, delete=True):
        """ Calls :py:meth:`Score.detach_labels<ms3.score.Score.detach_labels` on every parsed score with key ``key``.
        """
        assert annotation_key != 'annotations', "The key 'annotations' is reserved, please choose a different one."
        ids = list(self._iterids(keys, only_attached_annotations=True))
        if len(ids) == 0:
            self.logger.info(f"Selection did not contain scores with labels: keys = '{keys}'")
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
        self._collect_annotations_objects_references(ids=ids)




    def get_labels(self, keys=None, staff=None, voice=None, label_type=None, positioning=True, decode=False, column_name=None, concat=True):
        """ This function does not take into account self.labels_cfg """
        if len(self._annotations) == 0:
            self.logger.error("No labels available so far. Add files using add_dir() and parse them using parse().")
            return pd.DataFrame()
        keys = self._treat_key_param(keys)
        label_type = self._treat_label_type_param(label_type)
        self.collect_lists(labels=True, only_new=True)
        l = locals()
        params = {p: l[p] for p in self.labels_cfg.keys()}
        ids = [id for id in self._iterids(keys) if id in self._annotations]
        if len(ids) == 0:
            self.logger.info(f"No labels match the criteria.")
            return pd.DataFrame()
        annotation_tables = [self._annotations[id].get_labels(**params, warnings=False) for id in ids]
        idx, names = self.ids2idx(ids)
        if names is None:
            names = (None,) * len(idx[0])
        names += tuple(annotation_tables[0].index.names)
        if concat:
            return pd.concat(annotation_tables, keys=idx, names=names)
        return annotation_tables




    def get_lists(self, keys=None, ids=None, notes=False, rests=False, notes_and_rests=False, measures=False, events=False,
                  labels=False, chords=False, expanded=False, cadences=False, simulate=False, flat=True, unfold=False,
                  quarterbeats=False):
        """ Retrieve a dictionary with the selected feature matrices.

        Parameters
        ----------
        keys
        ids
        notes
        rests
        notes_and_rests
        measures
        events
        labels
        chords
        expanded
        cadences
        simulate
        flat : :obj:`bool`, optional
            By default, you get a dictionary {(id, list_type) -> list}.
            By passing False you get a nested dictionary {list_type -> {index -> list}}
        unfold : :obj:`bool`, optional
            Pass True if lists should reflect repetitions and voltas to create a correct playthrough.
            Defaults to False, meaning that all measures including second endings are included, unless ``quarterbeats``
            is set to True.
        quarterbeats : :obj:`bool`, optional
            Pass True to add a `quarterbeats` column with a continuous offset from the piece's beginning. If ``unfold``
            is False, this option will remove second endings and the ``volta`` column.

        Returns
        -------

        """
        if ids is None:
            ids = list(self._iterids(keys))
        if len(self._parsed_mscx) == 0 and len(self._parsed_tsv) == 0:
            self.logger.error("No scores or TSV files have been parsed so far.")
            return {}
        bool_params = list(self._lists.keys())
        l = locals()
        params = {p: l[p] for p in bool_params}
        self.collect_lists(ids=ids, only_new=True, **params)
        res = {}
        if unfold:
            mc_sequences = self.get_unfolded_mcs(ids=ids)
        if unfold or quarterbeats:
            _ = self.match_files(ids=ids)
        for param, li in self._lists.items():
            if params[param]:
                if not flat:
                    res[param] = {}
                for id in (i for i in ids if i in li):
                    key, i = id
                    df = li[id]
                    if unfold:
                        if id in mc_sequences:
                            df = unfold_repeats(df, mc_sequences[id])
                            if quarterbeats:
                                offset_dict = self.get_continuous_offsets(key, i, unfold=True)
                                df = add_quarterbeats_col(df, offset_dict, insert_after='mc_playthrough')
                        else:
                            self.logger.info(f"Cannot unfold {id} without measure information.")
                    elif quarterbeats:
                        if 'volta' in df.columns:
                            self.logger.debug("Only second voltas were included when computing quarterbeats.")
                        offset_dict = self.get_continuous_offsets(key, i, unfold=False)
                        df = add_quarterbeats_col(df, offset_dict)
                    if flat:
                        res[id + (param,)] = df
                    else:
                        res[param][self._index[id]] = df
        return res


    def get_tsvs(self, keys=None, types=None):
        if len(self._tsv_types) == 0:
            self.info(f"No TSV files have been parsed, use method parse_tsv().")
            return pd.DataFrame()
        if types is None:
            potential = self._tsv_types
            types = sorted(set(self._tsv_types.values()))
        elif isinstance(types, str):
            types = [types]
            potential = {id: typ for id, typ in self._tsv_types.items() if typ in types}
        plural = f"type {types[0]}" if len(types) == 1 else f"one of the types {types}"

        if len(potential) == 0:
            self.info(f"None of the parsed TSV files have been recognized as {plural}.")
            return pd.DataFrame()
        ids = [id for id in self._iterids(keys, only_parsed_tsv=True) if id in potential]
        if len(ids) == 0:
            self.info(f"None of the parsed TSV files with key {keys} have been recognized as {plural}.")
            return pd.DataFrame()
        keys = self.ids2idx(ids, pandas_index=True)
        return pd.concat([self._parsed_tsv[id] for id in ids], keys=keys)


    def get_unfolded_mcs(self, keys=None, ids=None):
        if ids is None:
            ids = list(self._iterids(keys))
        _ = self.match_files(ids=ids)
        res = {}
        for key, i in ids:
            unf_mcs = self._get_unfolded_mcs(key, i)
            if unf_mcs is not None:
                res[(key, i)] = unf_mcs
        return res

    def _get_measure_list(self, key, i, unfold=False):
        """ Tries to retrieve the corresponding measure list, e.g. for unfolding repeats. Preference is given to
        parsed MSCX files, then checks for parsed TSVs.

        Parameters
        ----------
        key, i
            ID
        unfold : :obj:`bool` or ``'raw'``
            Defaults to False, meaning that all voltas except second ones are dropped.
            If set to True, the measure list is unfolded.
            Pass the string 'raw' to leave the measure list as it is, with all voltas.

        Returns
        -------

        """
        id = (key, i)
        res = None
        if id in self._measurelists:
            res = self._measurelists[id]
        elif id in self._parsed_mscx:
            self.collect_lists(ids=[id], measures=True)
            res = self._measurelists[id]
        else:
            # trying to find a matched file to retrieve the measure list from
            ix = self._index[id]
            if ix not in self._matches.index:
                self.logger.debug(f"The index {ix} corresponding to ID {id} was not found in self._matches.")
                return
            matched_row = self._matches.loc[ix]
            if not pd.isnull(matched_row['measures']):
                matched_id = matched_row['measures']
                res = self._measurelists[matched_id]
            elif not pd.isnull(matched_row['scores']):
                matched_id = matched_row['scores']
                if matched_id in self._measurelists:
                    res = self._measurelists[matched_id]
                else:
                    self.collect_lists(ids=[matched_id], measures=True)
                    res = self._measurelists[matched_id]
            else:
                if matched_row.notna().sum() > 1:
                    self.logger.info(f"No measure list found for ID {id}. The matched IDs are:\n{matched_row}.")
                else:
                    self.logger.info(f"No matches found for ID {id} and therefore no measure list.")
        if res is not None:
            res = res.copy()
            if unfold == 'raw':
                return res
            if unfold:
                mc_sequence = self._get_unfolded_mcs(key, i)
                res = unfold_repeats(res, mc_sequence)
            elif 'volta' in res.columns:
                if 3 in res.volta.values:
                    self.logger.warning(f"Piece contains third endings, note that only second endings are taken into account.")
                res = res.drop(index=res[res.volta.fillna(2) != 2].index, columns='volta')
            return res



    def _get_unfolded_mcs(self, key, i):
        id = (key, i)
        if id in self._unfolded_mcs:
            return self._unfolded_mcs[id]
        if not id in self._measurelists:
            self.collect_lists(ids=[id], measures=True)
        ml = self._get_measure_list(key, i, unfold='raw')
        if ml is None:
            return
        ml = ml.set_index('mc')
        seq = next2sequence(ml.next)
        ############## < v0.5: playthrough <=> mn; >= v0.5: playthrough <=> mc
        # playthrough = compute_mn(ml[['dont_count', 'numbering_offset']].loc[seq]).rename('playthrough')
        mc_playthrough = pd.Series(seq, name='mc_playthrough')
        if seq[0] == 1:
            mc_playthrough.index += 1
        else:
            assert seq[0] == 0, f"The first mc should be 0 or 1, not {seq[0]}"
        # res = pd.Series(seq, index=playthrough)
        self._unfolded_mcs[id] = mc_playthrough
        return mc_playthrough


    def get_continuous_offsets(self, key, i, unfold):
        """ Using a corresponding measure list, return a dictionary mapping MCs to their absolute distance from MC 1,
            measured in quarter notes.

        Parameters
        ----------
        key, i:
            ID
        unfold : :obj:`bool`
            If True, return ``{mc_playthrough -> offset}``, otherwise ``{mc -> offset}``, keeping only second endings.

        Returns
        -------

        """
        id = (key, i)
        if id in self._quarter_offsets[unfold]:
            return self._quarter_offsets[unfold][id]
        ml = self._get_measure_list(key, i, unfold=unfold)
        if ml is None:
            self.logger.warning(f"Could not find measure list for key {id}.")
            return None
        if unfold:
            act_durs = ml.set_index('mc_playthrough').act_dur
        else:
            act_durs = ml.set_index('mc').act_dur
        offset_col = make_continuous_offset(act_durs, quarters=True)
        offsets = offset_col.to_dict()
        self._quarter_offsets[unfold][id] = offsets
        return offsets



    def ids2idx(self, ids=None, pandas_index=False):
        """ Receives a list of IDs and returns a list of index tuples or a pandas index created from it.

        Parameters
        ----------
        ids
        pandas_index

        Returns
        -------
        :obj:`pandas.Index` or :obj:`pandas.MultiIndex` or ( list(tuple()), tuple() )
        """
        if ids is None:
            ids = list(self._iterids())
        elif ids == []:
            if pandas_index:
                return pd.Index([])
            return list(), tuple()
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
            if len(set(level_names.values())) > 1:
                self.logger.warning(
                    f"Could not set level names because they differ for the different keys:\n{pretty_dict(level_names, 'LEVEL_NAMES')}")
                names = None
            else:
                names = tuple(level_names.values())[0]

        if pandas_index:
            idx = pd.MultiIndex.from_tuples(idx, names=names)
            return idx

        return idx, names


    def idx2id(self, full_path=None, rel_path=None, scan_path=None, path=None, file=None, fname=None, fext=None):
        """ Turn the respective value combinations of an index into an ID.

        Parameters
        ----------
        full_paths
        rel_paths
        scan_paths
        paths
        files
        fnames
        fexts

        Returns
        -------
        :obj:`tuple`
            the (key, i) pair corresponding to the index
        """
        levels = {name: str(l) for name, l in zip(('full_paths', 'rel_paths', 'scan_paths', 'paths', 'files', 'fnames', 'fexts'),
                                  (full_path, rel_path, scan_path, path, file, fname, fext))
                            if l is not None}
        sets = []
        for name, l in levels.items():
            ids = set((k, i) for k, vals in self._possible_levels[name].items() for i, val in enumerate(vals) if val == l)
            if len(ids) > 0:
                sets.append(ids)
        res = set.intersection(*sets)
        l = len(res)
        if l == 0:
            self.logger.warning(f"No parsed file matches these values: {levels}")
            return None
        if l > 1:
            self.logger.warning(f"The selection is ambiguous: {levels}")
            return None
        return list(res)[0]





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
        parsed_mscx_ids = [id for id in ids if id in self._parsed_mscx]
        parsed_mscx = len(parsed_mscx_ids)
        ext_counts = self.count_extensions(keys, per_key=False)
        others = len(self._score_ids(opposite=True))
        mscx = len(self._score_ids())
        by_conversion = len(self._score_ids(native=False))
        if parsed_mscx > 0:

            if parsed_mscx == mscx:
                info += f"\n\nAll {mscx} MSCX files have been parsed."
            else:
                info += f"\n\n{parsed_mscx}/{mscx} MSCX files have been parsed."
            annotated = sum(True for id in parsed_mscx_ids if id in self._annotations)
            if annotated == mscx:
                info += f"\n\nThey all have annotations attached."
            else:
                info += f"\n\n{annotated} of them have annotations attached."
            if annotated > 0:
                layers = self.count_annotation_layers(keys, which='attached', per_key=True)
                info += f"\n{pretty_dict(layers, heading='ANNOTATION LAYERS')}"

            detached = sum(True for id in parsed_mscx_ids if self._parsed_mscx[id].has_detached_annotations)
            if detached > 0:
                info += f"\n\n{detached} of them have detached annotations:"
                layers = self.count_annotation_layers(keys, which='detached', per_key=True)
                try:
                    info += f"\n{pretty_dict(layers, heading='ANNOTATION LAYERS')}"
                except:
                    print(layers)
                    raise
        elif '.mscx' in ext_counts:
            if mscx > 0:
                info += f"\n\nNone of the {mscx} score files have been parsed."
                if by_conversion > 0 and self.ms is None:
                    info += f"\n{by_conversion} files would need to be converted, for which you need to set the 'ms' property to your MuseScore 3 executable."
        if self.ms is not None:
            info += "\n\nMuseScore 3 executable has been found."


        parsed_tsv_ids = [id for id in ids if id in self._parsed_tsv]
        parsed_tsv = len(parsed_tsv_ids)
        if parsed_tsv > 0:
            annotations = sum(True for id in parsed_tsv_ids if id in self._annotations)
            if parsed_tsv == others:
                info += f"\n\nAll {others} tabular files have been parsed, {annotations} of them as Annotations object(s)."
            else:
                info += f"\n\n{parsed_tsv}/{others} tabular files have been parsed, {annotations} of them as Annotations object(s)."
            if annotations > 0:
                layers = self.count_annotation_layers(keys, which='tsv', per_key=True)
                info += f"\n{pretty_dict(layers, heading='ANNOTATION LAYERS')}"

        if return_str:
            return info
        print(info)


    def join(self, keys=None, ids=None, what=None, use_index=True):
        if what is not None:
            what = [w for w in what if w != 'scores']
        matches = self.match_files(keys=keys, ids=ids, what=what)
        join_this = set(map(tuple, matches.values))
        key_ids, *_ = zip(*join_this)
        if use_index:
            key_ids = self.ids2idx(key_ids, pandas_index=True)
        join_this = [[self._parsed_tsv[id] for id in ids if not pd.isnull(id)] for ids in join_this]
        joined = [join_tsvs(dfs) for dfs in join_this]
        return pd.concat(joined, keys=key_ids)



    def keys(self):
        return list(self.files.keys())

    def match_files(self, keys=None, ids=None, what=None, only_new=True):
        """ Match files based on their file names and return the matches for the requested keys or ids.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
            Which key(s) to return after matching matching files.
        what : :obj:`list` or ∈ {'scores', 'notes', 'rests', 'notes_and_rests', 'measures', 'events', 'labels', 'chords', 'expanded', 'cadences'}
            If you pass only one element, the corresponding files will be matched to all other types.
            If you pass several elements the first type will be matched to the following types.
        only_new : :obj:`bool`, optional
            Try matching only where matches are still missing.

        Returns
        -------
        :obj:`pandas.DataFrame`
            Those files that were matched. This is a subsection of self._matches
        """
        lists = {'scores': self._parsed_mscx}
        lists.update(dict(self._lists))
        if what is None:
            what = list(lists.keys())
        elif isinstance(what, str):
            what = [what]
        if len(what) == 1:
            what.extend([wh for wh in lists.keys() if wh != what[0]])
        assert all(True for wh in what if
                   wh in lists), f"Unknown matching parameter(s) for 'what': {[wh for wh in what if wh not in lists]}"
        for wh in what:
            if wh not in self._matches.columns:
                self._matches[wh] = np.nan

        matching_candidates = {wh: {id: self.fnames[id[0]][id[1]] for id in lists[wh].keys()} for wh in what}
        remove = []
        for i, wh in enumerate(what):
            if len(matching_candidates[wh]) == 0:
                self.logger.debug(f"There are no candidates for '{wh}' in the selected IDs.")
                remove.append(i)
        for i in reversed(remove):
            del (what[i])

        def get_row(ix):
            if ix in self._matches.index:
                return self._matches.loc[ix].copy()
            return pd.Series(np.nan, index=lists.keys(), name=ix)

        def update_row(ix, row):
            if ix in self._matches.index:
                self._matches.loc[ix, :] = row
            else:
                self._matches = self._matches.append(row)
                if len(self._matches) == 1:
                    self._matches.index = pd.MultiIndex.from_tuples(self._matches.index)

        #res_ix = set()
        for j, wh in enumerate(what):
            for id, fname in matching_candidates[wh].items():
                ix = self._index[id]
                row = get_row(ix)
                row[wh] = id
                for wha in what[j + 1:]:
                    if not pd.isnull(row[wha]) and only_new:
                        self.logger.debug(f"{ix} had already been matched to {wha} {row[wha]}")
                    else:
                        row[wha] = np.nan
                        key, i = id
                        file = self.files[key][i]
                        matches = {id: os.path.commonprefix([fname, c]) for id, c in matching_candidates[wha].items()}
                        lengths = {id: len(prefix) for id, prefix in matches.items()}
                        max_length = max(lengths.values())
                        if max_length == 0:
                            self.logger.debug(f"No matches for {id}")
                            break
                        longest = {id: prefix for id, prefix in matches.items() if lengths[id] == max_length}

                        if len(longest) == 0:
                            self.logger.info(
                                f"No match found for {file} among the candidates\n{pretty_dict(matching_candidates[wh])}")
                        elif len(longest) > 1:
                            ambiguity = {f"{key}: {self.full_paths[key][i]}": prefix for (key, i), prefix in
                                         longest.items()}
                            self.logger.info(
                                f"Matching {file} is ambiguous. Disambiguate using keys:\n{pretty_dict(ambiguity)}")
                        else:
                            match_id = list(longest.keys())[0]
                            row[wha] = match_id
                            match_ix = self._index[match_id]
                            match_row = get_row(match_ix)
                            match_row[wh] = id
                            update_row(match_ix, match_row)
                            #res_ix.add(match_ix)
                            match_file = self.files[match_id[0]][match_id[1]]
                            self.logger.debug(f"Matched {file} to {match_file} based on the prefix {longest[match_id]}")

                update_row(ix, row)
                #res_ix.add(ix)

        if ids is None:
            ids = list(self._iterids(keys))
        ids = [i for i in ids if any(i in lists[w] for w in what)]
        res_ix = self.ids2idx(ids, pandas_index=True)
        return self._matches.loc[res_ix, what].sort_index()


    def metadata(self, keys=None, include_tsv=False):
        parsed_ids = [id for id in self._iterids(keys) if id in self._parsed_mscx]
        df = pd.DataFrame()
        first_cols = ['last_mc', 'last_mn', 'KeySig', 'TimeSig', 'label_count',
                      'annotated_key', 'annotators', 'reviewers', 'composer', 'workTitle', 'movementNumber',
                      'movementTitle',
                      'workNumber', 'poet', 'lyricist', 'arranger', 'copyright', 'creationDate',
                      'mscVersion', 'platform', 'source', 'translator', 'musescore', 'ambitus']
        if len(parsed_ids) > 0:
            ids, meta_series = zip(*[(id, metadata2series(self._parsed_mscx[id].mscx.metadata)) for id in parsed_ids])
            idx = self.ids2idx(ids, pandas_index=True)
            df = pd.DataFrame(meta_series, index=idx)
        if include_tsv and len(self._parsed_tsv) > 0:
            tsv_df = self.get_tsvs(keys, types='metadata')
            if len(tsv_df) > 0:
                df = pd.concat([df, tsv_df])
        if len(df) > 0:
            return column_order(df, first_cols).sort_index()
        else:
            self.logger.info("No scores or metadata TSVs have been parsed so far.")
            return pd.DataFrame()


    def parse(self, keys=None, read_only=True, level=None, parallel=True, only_new=True, labels_cfg={}, fexts=None,
              cols={}, infer_types={'dcml': DCML_DOUBLE_REGEX}, simulate=None, **kwargs):
        """ Shorthand for executing parse_mscx and parse_tsv at a time."""
        if simulate is not None:
            self.simulate = simulate
        self.parse_mscx(keys=keys, read_only=read_only, level=level, parallel=parallel, only_new=only_new, labels_cfg=labels_cfg)
        self.parse_tsv(keys=keys, fexts=fexts, cols=cols, infer_types=infer_types, level=level, **kwargs)



    def parse_mscx(self, keys=None, ids=None, read_only=True, level=None, parallel=True, only_new=True, labels_cfg={}, simulate=False):
        """ Parse uncompressed MuseScore 3 files (MSCX) and store the resulting read-only Score objects. If they need
        to be writeable, e.g. for removing or adding labels, pass ``parallel=False`` which takes longer but prevents
        having to re-parse at a later point.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
            For which key(s) to parse all MSCX files.
        ids : :obj:`~collections.abc.Collection`
            To parse only particular files, pass their IDs. ``keys`` and ``fexts`` are ignored in this case.
        read_only : :obj:`bool`, optional
            If ``parallel=False``, you can increase speed and lower memory requirements by passing ``read_only=True``.
        level : {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}, optional
            Pass a level name for which (and above which) you want to see log records.
        parallel : :obj:`bool`, optional
            Defaults to True, meaning that all CPU cores are used simultaneously to speed up the parsing. It implies
            that the resulting Score objects are in read-only mode and that you might not be able to use the computer
            during parsing. Set to False to parse one score after the other.
        only_new : :obj:`bool`, optional
            By default, score which already have been parsed, are not parsed again. Pass False to parse them, too.

        Returns
        -------
        None

        """
        if simulate is not None:
            self.simulate = simulate
        self.labels_cfg.update(update_labels_cfg(labels_cfg, logger=self.logger))
        if parallel and not read_only:
            read_only = True
            self.logger.info("When pieces are parsed in parallel, the resulting objects are always in read_only mode.")

        if ids is not None:
            pass
        elif only_new:
            ids = [id for id in self._score_ids(keys) if id not in self._parsed_mscx]
        else:
            ids = self._score_ids(keys)

        exts = self.count_extensions(ids=ids)

        if any(ext[1:].lower() in Score.convertible_formats for ext in exts.keys()) and parallel:
            msg = f"The file extensions [{', '.join(ext[1:] for ext in exts.keys() if ext[1:].lower() in Score.convertible_formats )}] " \
                  f"require temporary conversion with your local MuseScore, which is not possible with parallel " \
                  f"processing. Parse with parallel=False or exclude these files from parsing."
            if self.ms is None:
                msg += "\n\nIn case you want to temporarily convert these files, you will also have to set the" \
                       "property ms of this object to the path of your MuseScore 3 executable."
            self.logger.error(msg)
            return

        if len(ids) == 0:
            reason = 'in this Parse object' if keys is None else f"for '{keys}'"
            self.logger.debug(f"No parseable scores found {reason}.")
            return
        if level is None:
            level = self.logger.logger.level
        cfg = {'level': level}

        ### If log files are going to be created, compute their paths and configure loggers for individual parses
        if self.logger_cfg['file'] is not None or self.logger_cfg['path'] is not None:
            file = None if self.logger_cfg['file'] is None else os.path.expanduser(self.logger_cfg['file'])
            path = None if self.logger_cfg['path'] is None else os.path.expanduser(self.logger_cfg['path'])
            if file is not None:
                file_path, file_name = os.path.split(file)
                if file_path == '':
                    if file_name in ['.', '..']:
                        file_path = file_name
                        file_name = None
                    else:
                        file_path = None
            else:
                file_path, file_name = None, None

            if file_path is not None and os.path.isabs(file_path):
                if os.path.isdir(file):
                    self.logger.error(f"You have passed the directory {file} as parameter 'file' which needs to be a relative dir or a (relative or absolute) file path.")
                    configs = [cfg for i in range(len(ids))]
                else:
                    cfg['file'] = file
                    configs = [cfg for i in range(len(ids))]
            elif not (file_path is None and file_name is None):
                root_dir = None if path is None else path
                if file_name is None:
                    log_paths = [os.path.abspath(os.path.join(self._calculate_path(k, i, root_dir, file_path),
                                                              f"{self.logger_names[(k, i)]}.log")) for k, i in ids]
                else:
                    log_paths = {(k, i): os.path.abspath(os.path.join(self._calculate_path(k, i, root_dir, file_path),
                                                             file_name)) for k, i in ids}
                    are_dirs = [p for p in set(log_paths.values()) if os.path.isdir(p)]
                    if len(are_dirs) > 0:
                        NL = '\n'
                        self.logger.info(
                        f"""The following file paths are actually existing directories, individual log files are created:
                        {NL.join(are_dirs)}""")
                        log_paths = {id: os.path.join(p, self.logger_names[id]) if os.path.isdir(p) else p for id, p in log_paths.items()}
                    log_paths = list(log_paths.values())
                configs = [dict(cfg, file=p) for p in log_paths]
            elif path is not None:
                configs = [dict(cfg, file=os.path.abspath(
                                            os.path.join(path, f"{self.logger_names[(k, i)]}.log")
                                          )) for k, i in ids]
            else:
                configs = [cfg for i in range(len(ids))]
        else:
            if self.logger.logger.file_handler is not None:
                cfg['file'] = self.logger.logger.file_handler.baseFilename
            configs = [cfg for i in range(len(ids))]

        ### collect argument tuples for calling self._parse
        parse_this = [t + (c, self.labels_cfg, read_only) for t, c in zip(ids, configs)]
        target = len(parse_this)
        successful = 0
        modus = 'would ' if self.simulate else ''
        try:
            ids = [t[:2] for t in parse_this]
            if self.simulate:
                logger_cfg = {'level': level}
                for key, i, _, _, read_only in parse_this:
                    logger_cfg['name'] = self.logger_names[(key, i)]
                    path = self.full_paths[key][i]
                    try:
                        score_object = Score(path, read_only=read_only, logger_cfg=logger_cfg)
                    except:
                        self.logger.exception(traceback.format_exc())
                        score_object = None
                    if score_object is not None:
                        self._parsed_mscx[(key, i)] = score_object
                        successful += 1
                        self.logger.debug(f"Successfully parsed {path}")
                    else:
                        self.logger.debug(f"Errors while parsing {path}")
            elif parallel:
                pool = mp.Pool(mp.cpu_count())
                res = pool.starmap(self._parse, parse_this)
                pool.close()
                pool.join()
                successful_results = {id: score for id, score in zip(ids, res) if score is not None}
                self._parsed_mscx.update(successful_results)
                successful = len(successful_results)
            else:
                for params in parse_this:
                    score_object = self._parse(*params)
                    if score_object is not None:
                        self._parsed_mscx[params[:2]] = score_object
                        successful += 1
            if successful > 0:
                if successful == target:
                    self.logger.info(f"All {target} files {modus}have been parsed successfully.")
                else:
                    self.logger.info(f"Only {successful} of the {target} files {modus}have been parsed successfully.")
            else:
                self.logger.info(f"None of the {target} files {modus}have been parsed successfully.")
        except KeyboardInterrupt:
            self.logger.info("Parsing interrupted by user.")
        finally:
            self._collect_annotations_objects_references(ids=ids)


    def parse_tsv(self, keys=None, ids=None, fexts=None, cols={}, infer_types={'dcml': DCML_DOUBLE_REGEX}, level=None, **kwargs):
        """ Parse TSV files (or other value-separated files such as CSV) to be able to do something with them.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
            Key(s) for which to parse all non-MSCX files.  By default, all keys are selected.
        ids : :obj:`~collections.abc.Collection`
            To parse only particular files, pass there IDs. ``keys`` and ``fexts`` are ignored in this case.
        fexts :  :obj:`str` or :obj:`~collections.abc.Collection`, optional
            If you want to parse only files with one or several particular file extension(s), pass the extension(s)
        annotations : :obj:`str` or :obj:`~collections.abc.Collection`, optional
            By default, if a column called ``'label'`` is found, the TSV is treated as an annotation table and turned into
            an Annotations object. Pass one or several column name(s) to treat *them* as label columns instead. If you
            pass ``None`` or no label column is found, the TSV is parsed as a "normal" table, i.e. a DataFrame.
        infer_types : :obj:`dict`, optional
            To recognize one or several custom label type(s), pass ``{name: regEx}``.
        level : {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}, optional
            Pass a level name for which (and above which) you want to see log records.
        **kwargs:
            Arguments for :py:meth:`pandas.DataFrame.to_csv`. Defaults to ``{'sep': '\t', 'index': False}``. In particular,
            you might want to update the default dictionaries for ``dtypes`` and ``converters`` used in :py:func:`load_tsv`.

        Returns
        -------
        None
        """
        if self.simulate:
            return
        if ids is not None:
            pass
        elif fexts is None:
            ids = [(key, i) for key, i in self._iterids(keys) if self.fexts[key][i][1:] not in Score.parseable_formats]
        else:
            if isinstance(fexts, str):
                fexts = [fexts]
            fexts = [ext if ext[0] == '.' else f".{ext}" for ext in fexts]
            ids = [(key, i) for key, i in self._iterids(keys) if self.fexts[key][i] in fexts]

        for key, i in ids:
            #rel_path = os.path.join(self.rel_paths[key][i], self.files[key][i])
            path = self.full_paths[key][i]
            try:
                df = load_tsv(path, **kwargs)
            except:
                self.logger.info(f"Couldn't be loaded, probably no tabular format or you need to specify 'sep', the delimiter."
                                 f"\n{path}\nError: {sys.exc_info()[1]}")
                continue
            label_col = cols['label'] if 'label' in cols else 'label'
            id = (key, i)
            try:
                self._parsed_tsv[id] = df
                if 'label' in cols and label_col in df.columns:
                    tsv_type = 'labels'
                else:
                    tsv_type = self._infer_tsv_type(df)

                if tsv_type is None:
                    self.logger.warning(
                        f"No label column '{label_col}' was found in {self.files[key][i]} and its content could not be inferred. Columns: {df.columns.to_list()}")
                else:
                    self._tsv_types[id] = tsv_type
                    if tsv_type == 'metadata':
                        self._metadata = pd.concat([self._metadata, self._parsed_tsv[id]])
                    else:
                        self._lists[tsv_type][id] = self._parsed_tsv[id]
                        if tsv_type in ['labels', 'expanded']:
                            if label_col in df.columns:
                                logger_name = self.files[key][i]
                                self._annotations[id] = Annotations(df=df, cols=cols, infer_types=infer_types,
                                                                          logger_cfg={'name': logger_name}, level=level)
                                self.logger.debug(
                                    f"{self.files[key][i]} parsed as a list of labels and an Annotations object was created.")
                            else:
                                self.logger.info(
    f"""The file {self.files[key][i]} was recognized to contain labels but no label column '{label_col}' was found in {df.columns.to_list()}
    Specify parse_tsv(key='{key}', cols={{'label'=label_column_name}}).""")
                        else:
                            self.logger.debug(f"{self.files[key][i]} parsed as a list of {tsv_type}.")

            except:
                self.logger.error(f"Parsing {self.files[key][i]} failed with the following error:\n{sys.exc_info()[1]}")



    def store_lists(self, keys=None, root_dir=None, notes_folder=None, notes_suffix='',
                                                    rests_folder=None, rests_suffix='',
                                                    notes_and_rests_folder=None, notes_and_rests_suffix='',
                                                    measures_folder=None, measures_suffix='',
                                                    events_folder=None, events_suffix='',
                                                    labels_folder=None, labels_suffix='',
                                                    chords_folder=None, chords_suffix='',
                                                    expanded_folder=None, expanded_suffix='',
                                                    cadences_folder=None, cadences_suffix='',
                                                    metadata_path=None, markdown=True,
                                                    simulate=None, unfold=False, quarterbeats=False):
        """ Store score information as TSV files.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
            Key(s) under which score files are stored. By default, all keys are selected.
        root_dir : :obj:`str`, optional
            Defaults to None, meaning that the original root directory is used that was added to the Parse object.
            Otherwise, pass a directory to rebuild the original directory structure. For ``_folder`` parameters describing
            absolute paths, ``root_dir`` is ignored.
        notes_folder, rests_folder, notes_and_rests_folder, measures_folder, events_folder, labels_folder, chords_folder, expanded_folder, cadences_folder : str, optional
            Specify directory where to store the corresponding TSV files.
        notes_suffix, rests_suffix, notes_and_rests_suffix, measures_suffix, events_suffix, labels_suffix, chords_suffix, expanded_suffix, cadences_suffix : str, optional
            Optionally specify suffixes appended to the TSVs' file names.
        metadata_path : str, optional
            Where to store an overview file with the MuseScore files' metadata.
            If no file name is specified, the file will be named ``metadata.tsv``.
        markdown : bool, optional
            By default, when ``metadata_path`` is specified, a markdown file called ``README.md`` containing
            the columns [file_name, measures, labels, standard, annotators, reviewers] is created. If it exists already,
            this table will be appended or overwritten after the heading ``# Overview``.
        simulate : bool, optional
            Defaults to ``None``. Pass a value to set the object attribute :py:attr:`~ms3.parse.Parse.simulate`.
        unfold : bool, optional
            By default, repetitions are not unfolded. Pass True to duplicate values so that they correspond to a full
            playthrough, including correct positioning of first and second endings.
        quarterbeats : bool, optional
            By default, no ``quarterbeats`` column is added with distances from the piece's beginning measured in quarter notes.
            Pass True to add the columns ``quarterbeats`` and ``durations_quarterbeats``. If a score has first and second endings,
            the behaviour depends on ``unfold``: If False, repetitions are not unfolded and only last endings are included in the
            continuous count. If repetitions are being unfolded, all endings are taken into account.

        Returns
        -------

        """
        if simulate is None:
            simulate = self.simulate
        else:
            self.simulate = simulate
        l = locals()
        list_types = list(self._lists)
        folder_vars = [t + '_folder' for t in list_types]
        suffix_vars = [t + '_suffix' for t in list_types]
        folder_params = {t: l[p] for t, p in zip(list_types, folder_vars) if l[p] is not None}
        if len(folder_params) == 0 and metadata_path is None:
            self.logger.warning("Pass at least one parameter to store files.")
            return [] if simulate else None
        suffix_params = {t: '_unfolded' if l[p] is None and unfold else l[p] for t, p in zip(list_types, suffix_vars) if t in folder_params}
        list_params = {p: True for p in folder_params.keys()}
        lists = self.get_lists(keys, unfold=unfold, quarterbeats=quarterbeats, **list_params)
        modus = 'would ' if simulate else ''
        if len(lists) == 0 and metadata_path is None:
            self.logger.info(f"No files {modus}have been written.")
            return [] if simulate else None
        paths = {}
        warnings, infos = [], []
        prev_logger = self.logger.name
        for (key, i, what), li in lists.items():
            self.update_logger_cfg(name=self.logger_names[(key, i)])
            new_path = self._store_tsv(df=li, key=key, i=i, folder=folder_params[what], suffix=suffix_params[what], root_dir=root_dir, what=what, simulate=simulate)
            if new_path in paths:
                warnings.append(f"The {paths[new_path]} at {new_path} {modus}have been overwritten with {what}.")
            else:
                infos.append(f"{what} {modus}have been stored as {new_path}.")
            paths[new_path] = what
        self.update_logger_cfg(name=prev_logger)
        if len(warnings) > 0:
            self.logger.warning('\n'.join(warnings))
        l_infos = len(infos)
        l_target = len(lists)
        if l_target > 0:
            if l_infos == 0:
                self.logger.info(f"\n\nNone of the {l_target} {modus}have been written.")
            elif l_infos < l_target:
                msg = f"\n\nOnly {l_infos} out of {l_target} files {modus}have been stored."
            else:
                msg = f"\n\nAll {l_infos} {modus}have been written."
            self.logger.info('\n'.join(infos) + msg)
        #self.logger = prev_logger
        if metadata_path is not None:
            md = self.metadata()
            if len(md.index) > 0:
                fname, ext = os.path.splitext(metadata_path)
                if ext != '':
                    path, file = os.path.split(metadata_path)
                else:
                    path = metadata_path
                    file = 'metadata.tsv'
                path = resolve_dir(path)
                if not os.path.isdir(path):
                    os.makedirs(path)
                full_path = os.path.join(path, file)
                write_metadata(self.metadata(), full_path, markdown=markdown, logger=self.logger)
            else:
                self.logger.debug(f"\n\nNo metadata to write.")

        if simulate:
            return list(set(paths.keys()))



    def store_mscx(self, keys=None, ids=None, root_dir=None, folder='.', suffix='', overwrite=False, simulate=False):
        """ Stores the parsed MuseScore files in their current state, e.g. after detaching or attaching annotations.
        """
        if ids is None:
            ids = [id for id in self._iterids(keys) if id in self._parsed_mscx]
        paths = []
        for key, i in ids:
            new_path = self._store_mscx(key=key, i=i, folder=folder, suffix=suffix, root_dir=root_dir, overwrite=overwrite, simulate=simulate)
            if new_path is not None:
                if new_path in paths:
                    modus = 'would have' if simulate else 'has'
                    self.logger.info(f"The score at {new_path} {modus} been overwritten.")
                else:
                    paths.append(new_path)
        if simulate:
            return list(set(paths))




    def _calculate_path(self, key, i, root_dir, folder, enforce_below_root=False):
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
        enforce_below_root : :obj:`bool`, optional
            If True is passed, the computed paths are checked to be within ``root_dir`` or ``folder`` respectively.
        """
        if folder is not None and (os.path.isabs(folder) or '~' in folder):
            folder = resolve_dir(folder)
            path = folder
        else:
            root = self.scan_paths[key][i] if root_dir is None else resolve_dir(root_dir)
            if folder is None:
                path = root
            elif folder[0] == '.':
                path = os.path.abspath(os.path.join(root, self.rel_paths[key][i], folder))
            else:
                path = os.path.abspath(os.path.join(root, folder, self.rel_paths[key][i]))
            base = os.path.basename(root)
            if enforce_below_root and path[:len(base)] != base:
                self.logger.error(f"Not allowed to store files above the level of root {root}.\nErroneous path: {path}")
                return None
        return path



    def _collect_annotations_objects_references(self, keys=None, ids=None):
        """ Updates the dictionary self._annotations with all parsed Scores that have labels attached (or not any more). """
        if ids is None:
            ids = list(self._iterids(keys, only_parsed_mscx=True))
        updated = {}
        for id in ids:
            if id in self._parsed_mscx:
                score = self._parsed_mscx[id]
                if score is not None:
                    if 'annotations' in score:
                        updated[id] = score.annotations
                    elif id in self._annotations:
                        del (self._annotations[id])
                else:
                    del (self._parsed_mscx[id])
        self._annotations.update(updated)

    def _handle_path(self, full_path, key=None):
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
                    self.logger.error(
                        f"""The file name {file} is already registered for key '{key}' and both files have the relative path {rel_path}.
Load one of the identically named files with a different key using add_dir(key='KEY').""")
                    return (None, None)
                self.logger.debug(
                    f"The file {file} is already registered for key '{key}' but can be distinguished via the relative path.")

            i = len(self.full_paths[key])
            self.full_paths[key].append(full_path)
            self.scan_paths[key].append(self.last_scanned_dir)
            self.rel_paths[key].append(rel_path)
            self.paths[key].append(file_path)
            self.files[key].append(file)
            self.logger_names[(key, i)] = file_name.replace('.', '')
            self.fnames[key].append(file_name)
            self.fexts[key].append(file_ext)
            return key, len(self.paths[key]) - 1
        else:
            self.logger.error("No file found at this path: " + full_path)
            return (None, None)


    def _infer_tsv_type(self, df):
        type2cols = {
            'notes': ['tpc', 'midi'],
            'events': ['event'],
            'chords': ['chord_id'],
            'rests': ['nominal_duration'],
            'measures': ['act_dur'],
            'expanded': ['numeral'],
            'labels': ['label_type'],
            'cadences': ['cadence'],
            'metadata': ['last_mn', 'md5'],
        }
        for t, columns in type2cols.items():
            if any(True for c in columns if c in df.columns):
                return t
        if any(True for c in ['mc', 'mn'] if c in df.columns):
            return 'labels'
        return

    def _iterids(self, keys=None, only_parsed_mscx=False, only_parsed_tsv=False, only_attached_annotations=False, only_detached_annotations=False):
        """Iterator through IDs for a given set of keys.

        Parameters
        ----------
        keys
        only_parsed_mscx
        only_attached_annotations
        only_detached_annotations

        Yields
        ------
        :obj:`tuple`
            (str, int)

        """
        keys = self._treat_key_param(keys)
        for key in sorted(keys):
            for id in make_id_tuples(key, len(self.fnames[key])):
                if only_parsed_mscx  or only_attached_annotations or only_detached_annotations:
                    if id not in self._parsed_mscx:
                        continue
                    if only_attached_annotations:
                        if 'annotations' in self._parsed_mscx[id]:
                            pass
                        else:
                            continue
                    elif only_detached_annotations:
                        if self._parsed_mscx[id].has_detached_annotations:
                            pass
                        else:
                            continue
                elif only_parsed_tsv:
                    if id in self._parsed_tsv:
                        pass
                    else:
                        continue

                yield id



    def _itersel(self, collectio, selector=None, opposite=False):
        """ Returns a generator of ``collectio``. ``selector`` can be a collection of index numbers to select or unselect
        elements -- depending on ``opposite`` """
        if selector is None:
            for e in collectio:
                yield e
        if opposite:
            for i, e in enumerate(collectio):
                if i not in selector:
                    yield e
        else:
            for i, e in enumerate(collectio):
                if i in selector:
                    yield e

    def _make_index_level(self, level, ids, selector=None):
        if isinstance(level, str) or len(level) == 1:
            if not isinstance(level, str):
                level = level[0]
            if level == 'key':
                return {id: id[0] for id in ids}
            if level == 'i':
                return {id: id[1] for id in ids}
            if level in self._possible_levels:
                return {(key, i): self._possible_levels[level][key][i] for key, i in ids}
        ll, li = len(level), len(ids)
        ls = 0 if selector is None else len(selector)
        if ll < li:
            self.logger.error(f"Index level (length {ll}) has not enough values for {li} ids.")
            return {}
        if ll > li:
            if ls == 0:
                res = {i: l for i, l in self._itersel(zip(ids, level), tuple(range(li)))}
                discarded = [l for l in self._itersel(level, tuple(range(li, ll)))]
                self.logger.warning(
                    f"""Index level (length {ll}) has more values than needed for {li} ids and no selector has been passed.
Using the first {li} elements, discarding {discarded}""")
            elif ls != li:
                self.logger.error(
                    f"The selector for picking elements from the overlong index level (length {ll}) should have length {li}, not {ls},")
                res = {}
            else:
                if ls != ll:
                    discarded = [l for l in self._itersel(level, selector, opposite=True)]
                    plural_s = 's' if len(discarded) > 1 else ''
                    self.logger.debug(
                        f"Selector {selector} was applied, leaving out the index value{plural_s} {discarded}")
                res = {i: l for i, l in zip(ids, self._itersel(level, selector))}
        else:
            res = {i: l for i, l in zip(ids, level)}
        return res


    def _parse(self, key, i, logger_cfg={}, labels_cfg={}, read_only=False):
        """Performs a single parse and returns the resulting Score object or None."""
        path = self.full_paths[key][i]
        file = self.files[key][i]
        self.logger.debug(f"Attempting to parse {file}")
        try:
            score = Score(path, read_only=read_only, labels_cfg=labels_cfg, logger_cfg=logger_cfg, ms=self.ms)
            if score is None:
                self.logger.debug(f"Encountered errors when parsing {file}")
            else:
                self.logger.debug(f"Successfully parsed {file}")
                # self._parsed_mscx[(key, i)] = score
            return score
        except (KeyboardInterrupt, SystemExit):
            self.logger.info("Process aborted.")
            raise
        except:
            self.logger.error(traceback.format_exc())
            return None


    def _score_ids(self, keys=None, score_extensions=None, native=True, convertible=True, opposite=False):
        if score_extensions is None:
            score_extensions = []
            if native:
                score_extensions.extend(Score.native_formats)
            if convertible:
                score_extensions.extend(Score.convertible_formats)
        if opposite:
            return [(k, i) for k, i in self._iterids(keys) if self.fexts[k][i][1:].lower() not in score_extensions]
        return [(k, i) for k, i in self._iterids(keys) if self.fexts[k][i][1:].lower() in score_extensions]



    def _store_mscx(self, key, i, folder, suffix='', root_dir=None, overwrite=False, simulate=False):
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

        id = (key, i)
        logger = get_logger(self.logger_names[id])
        fname = self.fnames[key][i]

        if id not in self._parsed_mscx:
            logger.error(f"No Score object found. Call parse_mscx() first.")
            return
        path = self._calculate_path(key=key, i=i, root_dir=root_dir, folder=folder)
        if path is None:
            return

        fname = fname + suffix + '.mscx'
        file_path = os.path.join(path, fname)
        if os.path.isfile(file_path):
            if simulate:
                if overwrite:
                    logger.warning(f"Would have overwritten {file_path}.")
                    return
                logger.warning(f"Would have skipped {file_path}.")
                return
            elif not overwrite:
                logger.warning(f"Skipped {file_path}.")
                return
        if simulate:
            logger.debug(f"Would have written score to {file_path}.")
        else:
            os.makedirs(path, exist_ok=True)
            self._parsed_mscx[id].store_mscx(file_path)
            logger.debug(f"Score written to {file_path}.")

        return file_path


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
        # make sure all subloggers store their information into Parse.log if it is being used
        # file = None if self.logger.logger.file_handler is None else self.logger.logger.file_handler.baseFilename
        # self.update_logger_cfg(name=self.logger_names[(key, i)] + f":{what}", file=file)
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
        """ Turns an index parameter (string or collection) and turns each elemeht into an index level.

        Parameters
        ----------
        index_param
        ids
        selector : :obj:`~collections.abc.Collection`
            Pass a collection of list indices to create index tuples for only those.


        Returns
        -------
        :obj:`pandas.core.indexes.multi.MultiIndex` or :obj:`pandas.core.indexes.base.Index`
            Newly created index.
        :obj:`tuple`
            Names of the index levels.

        """
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
                if level in self._possible_levels or level in ('key', 'i'):
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
            new_index, names = self._treat_index_param(None, ids=ids)
            if l_counts > 0:
                plural_phrase = "These index values occur" if l_counts > 1 else "This index value occurs"
                self.logger.error(f"""The generated index is not unique and has been replaced by the standard index (IDs).
To avoid the problem, define sufficient distinguishing index levels, e.g. ['fname', 'key', 'i'].\n{plural_phrase} several times:\n{pretty_dict(counts)}""")
            if l_existing > 0:
                plural_phrase = "s are" if l_existing > 1 else " is"
                self.logger.error(f"The generated index cannot be used because the following element{plural_phrase} already in use:\n{existing}")
        return new_index, tuple(names)


    def _treat_key_param(self, keys):
        if keys is None:
            keys = list(self.full_paths.keys())
        elif isinstance(keys, str):
            keys = [keys]
        return [k for k in set(keys) if k in self.files]


    def _treat_label_type_param(self, label_type):
        if label_type is None:
            return None
        all_types = {str(k): k for k in self.count_label_types().keys()}
        if isinstance(label_type, int) or isinstance(label_type, str):
            label_type = [label_type]
        lt = [str(t) for t in label_type]
        def matches_any_type(user_input):
            return any(True for t in all_types if user_input in t)
        def get_matches(user_input):
            return [t for t in all_types if user_input in t]

        not_found = [t for t in lt if not matches_any_type(t)]
        if len(not_found) > 0:
            plural = len(not_found) > 1
            plural_s = 's' if plural else ''
            self.logger.warning(
                f"No labels found with {'these' if plural else 'this'} label{plural_s} label_type{plural_s}: {', '.join(not_found)}")
        return [all_types[t] for user_input in lt for t in get_matches(user_input)]

    def update_metadata(self):
        """Uses all parsed metadata TSVs to update the information in the corresponding parsed MSCX files and returns
        the IDs of those that have been changed."""
        if len(self._metadata) == 0:
            self.logger.debug("No parsed metadata found.")
            return
        old = self._metadata.set_index(['rel_paths', 'fnames'])
        new = self.metadata()
        excluded_cols = ['ambitus', 'annotated_key', 'KeySig', 'label_count', 'last_mc', 'last_mn', 'musescore',
                         'TimeSig']
        old_cols = sorted([c for c in old.columns if c not in excluded_cols and c[:5] != 'staff'])

        parsed = old.index.map(lambda i: i in new.index)
        relevant = old.loc[parsed, old_cols]
        updates = defaultdict(dict)
        for i, row in relevant.iterrows():
            new_row = new.loc[i]
            for j, val in row[row.notna()].iteritems():
                val = str(val)
                if j not in new_row or str(new_row[j]) != val:
                    updates[i][j] = val

        l = len(updates)
        ids = []
        if l > 0:
            for (rel_path, fname), new_dict in updates.items():
                id = self.idx2id(rel_path=rel_path, fname=fname)
                if id not in self._parsed_mscx:
                    continue
                tags = self._parsed_mscx[id].mscx.parsed.metatags
                for name, val in new_dict.items():
                    tags[name] = val
                self._parsed_mscx[id].mscx.parsed.update_metadata()
                get_logger(self.logger_names[id]).debug(f"Updated with {new_dict}")
                ids.append(id)

            self.logger.info(f"{l} files updated.")
        else:
            self.logger.info("Nothing to update.")
        return ids


    def __getstate__(self):
        """ Override the method of superclass """
        return self.__dict__

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


    # def __getattr__(self, item):
    #     if item in self.fexts: # is an existing key
    #         fexts = self.fexts[item]
    #         res = {}
    #         for i, ext in enumerate(fexts):
    #             id = (item, i)
    #             ix = str(self._index[id])
    #             if ext == '.mscx':
    #                 if id in self._parsed_mscx:
    #                     ix += " (parsed)"
    #                     val = str(self._parsed_mscx[id])
    #                 else:
    #                     ix += " (not parsed)"
    #                     val = self.full_paths[item][i]
    #             else:
    #                 if id in self._parsed_tsv:
    #                     df = self._parsed_tsv[id]
    #                     if isinstance(df, Annotations):
    #                         ix += " (parsed annotations)"
    #                         val = str(df)
    #                     else:
    #                         t = self._tsv_types[id] if id in self._tsv_types else 'unrecognized DataFrame'
    #                         ix += f" (parsed {t}, length {len(df)})"
    #                         val = df.head(5).to_string()
    #                 else:
    #                     ix += " (not parsed)"
    #                     val = self.full_paths[item][i]
    #             ix += f"\n{'-' * len(ix)}\n"
    #             if ext != '.mscx':
    #                 ix += f"{self.full_paths[item][i]}\n"
    #             print(f"{ix}{val}\n")
    #     else:
    #         raise AttributeError(item)

    # def __getattr__(self, item):
    #     ext = f".{item}"
    #     ids = [(k, i) for k, i in self._iterids() if self.fexts[k][i] == ext]
    #     if len(ids) == 0:
    #         self.logger.info(f"Includes no files with the extension {ext}")
    #     return ids



    def __getitem__(self, item):
        if item in self._index:
            id = item
        elif item in self._index.values():
            id = next(k for k, v in self._index.items() if v == item)
        else:
            if item in self.files:
                self.logger.info(f"{item} is neither an ID nor an index, but a key with the following IDs:\n" + string2lines(make_id_tuples(item, len(self.files[item]))))
            else:
                self.logger.warning(f"{item} is neither an ID nor an index.")
            return
        if id in self._parsed_mscx:
            return self._parsed_mscx[id]
        if id in self._annotations:
            return self._annotations[id]
        if id in self._parsed_tsv:
            return self._parsed_tsv[id]
        else:
            self.logger.warning(f"{self.full_paths[id]} has or could not be parsed.")


    def __repr__(self):
        return self.info(return_str=True)







