
from typing import Literal, Collection, Generator, Tuple, Union, Dict, Optional

import io
import sys, os, re
from functools import lru_cache
import json
import traceback
from collections import Counter, defaultdict, namedtuple

import pandas as pd
import numpy as np
from git import Repo, InvalidGitRepositoryError
from gitdb.exc import BadName

from .corpus import Corpus
from .annotations import Annotations
from .logger import LoggedClass, get_logger, function_logger
from .piece import Piece
from .score import Score
from ._typing import FileDict, FileList, CorpusFnameTuple
from .utils import column_order, get_musescore, group_id_tuples, infer_tsv_type, \
    iter_selection, get_first_level_corpora, join_tsvs, load_tsv, make_continuous_offset_series, \
    make_id_tuples, make_playthrough2mc, METADATA_COLUMN_ORDER, metadata2series, parse_ignored_warnings_file, pretty_dict, resolve_dir, \
    update_labels_cfg, write_tsv, path2parent_corpus, available_views2str
from .transformations import dfs2quarterbeats
from .view import View, DefaultView

@function_logger
def unpack_json_paths(paths: Collection[str]) -> None:
    """Mutates the list with paths by replacing .json files with the list (of paths) contained in them."""
    json_ixs = [i for i, p in enumerate(paths) if p.endswith('.json')]
    if len(json_ixs) > 0:
        for i in reversed(json_ixs):
            try:
                with open(paths[i]) as f:
                    loaded_paths = json.load(f)
                paths.extend(loaded_paths)
                logger.info(f"Unpacked the {len(loaded_paths)} paths found in {paths[i]}.")
                del (paths[i])
            except Exception:
                logger.info(f"Could not load paths from {paths[i]} because of the following error(s):\n{sys.exc_info()[1]}")



def legacy_params2view(paths=None, file_re=None, folder_re=None, exclude_re=None) -> View:
    if all(param is None for param in (paths, file_re, folder_re, exclude_re)):
        return DefaultView()
    view = View("Gerhardt")
    if file_re is not None:
        view.include('files', file_re)
    if folder_re is not None and folder_re != '.*':
        view.include('folders', folder_re)
    if exclude_re is not None:
        view.exclude(('files', 'folders'), exclude_re)
    if paths is not None:
        if isinstance(paths, str):
            paths = [paths]
        unpack_json_paths(paths)
        regexes = [re.escape(os.path.basename(p)) for p in paths]
        view.include('files', *regexes)
    return view


class Parse(LoggedClass):
    """
    Class for storing and manipulating the information from multiple parses (i.e. :py:attr:`~.score.Score` objects).
    """

    def __init__(self, directory=None, paths=None, file_re=None, folder_re=None, exclude_re=None,
                 recursive=True, simulate=False, labels_cfg={}, ms=None, level=None, **logger_cfg):
        """

        Parameters
        ----------
        directory, key, index, file_re, folder_re, exclude_re, recursive : optional
            Arguments for the method :py:meth:`~ms3.parse.add_dir`.
            If ``dir`` is not passed, no files are added to the new object except if you pass ``paths``
        paths : :obj:`~collections.abc.Collection` or :obj:`str`, optional
            List of file paths you want to add. If ``directory`` is also passed, all files will be combined in the same object.
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
            If you pass the path to your local MuseScore 3 installation, ms3 will attempt to parse musicXML, MuseScore 2,
            and other formats by temporarily converting them. If you're using the standard path, you may try 'auto', or 'win' for
            Windows, 'mac' for MacOS, or 'mscore' for Linux. In case you do not pass the 'file_re' and the MuseScore executable is
            detected, all convertible files are automatically selected, otherwise only those that can be parsed without conversion.
        level: :obj:`str`
            Shorthand for setting logger_cfg['level'] to one of {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}.
        """
        if level is not None:
            logger_cfg['level'] = level
        if 'level' not in logger_cfg or (logger_cfg['level'] is None):
            logger_cfg['level'] = 'w'
        super().__init__(subclass='Parse', logger_cfg=logger_cfg)
        self.simulate=simulate

        self.corpus_paths = {}
        """obj:`dict`
        {corpus_name -> path} dictionary with each corpus's base directory.
        """

        self.corpus_objects = {}
        """obj:`dict`
        {corpus_name -> :obj:`Corpus`} dictionary with one object per Corpus.
        """

        self._ms = get_musescore(ms, logger=self.logger)
        """:obj:`str`
        Path or command of the local MuseScore 3 installation if specified by the user."""
        #
        #
        # self._parsed_mscx = {}
        # """:obj:`dict`
        # ``{(key, i): :py:attr:`~.score.Score`}`` dictionary of parsed scores.
        # """
        #
        # self._annotations = {}
        # """:obj:`dict`
        # {(key, i): :py:attr:`~.annotations.Annotations`} dictionary of parsed sets of annotations.
        # """
        #
        # self._fl_lists = {}
        # """:obj:`dict`
        # {(key, i): :obj:`pandas.DataFrame`} dictionary of DataFrames holding :py:attr:`~.score.Score.form_labels` tables.
        # """
        #
        # self._notelists = {}
        # """:obj:`dict`
        # {(key, i): :obj:`pandas.DataFrame`} dictionary of DataFrames holding :py:attr:`~.score.Score.notes` tables.
        # """
        #
        # self._restlists = {}
        # """:obj:`dict`
        # {(key, i): :obj:`pandas.DataFrame`} dictionary of DataFrames holding :py:attr:`~.score.Score.rests` tables
        # """
        #
        # self._noterestlists = {}
        # """:obj:`dict`
        # {(key, i): :obj:`pandas.DataFrame`} dictionary of DataFrames holding :py:attr:`~.score.Score.notes_and_rests` tables
        # """
        #
        # self._eventlists = {}
        # """:obj:`dict`
        # {(key, i): :obj:`pandas.DataFrame`} dictionary of DataFrames holding :py:attr:`~.score.Score.events` tables.
        # """
        #
        # self._labellists = {}
        # """:obj:`dict`
        # {(key, i): :obj:`pandas.DataFrame`} dictionary of DataFrames holding :py:attr:`~.score.Score.labels` tables.
        # """
        #
        # self._chordlists = {}
        # """:obj:`dict`
        # {(key, i): :obj:`pandas.DataFrame`} dictionary of DataFrames holding :py:attr:`~.score.Score.chords` tables.
        # """
        #
        # self._expandedlists = {}
        # """:obj:`dict`
        # {(key, i): :obj:`pandas.DataFrame`} dictionary of DataFrames holding :py:attr:`~.score.Score.expanded` tables.
        # """
        #
        # self._cadencelists = {}
        # """:obj:`dict`
        # {(key, i): :obj:`pandas.DataFrame`} dictionary of DataFrames holding :py:attr:`~.score.Score.cadences` tables.
        # """
        #
        # self._measurelists = {}
        # """:obj:`dict`
        # {(key, i): :obj:`pandas.DataFrame`} dictionary of DataFrames holding :py:attr:`~.score.Score.measures` tables.
        # """
        #
        # self._metadata = pd.DataFrame()
        # """:obj:`pandas.DataFrame`
        # Concatenation of all parsed metadata TSVs.
        # """
        #
        # self._parsed_tsv = {}
        # """:obj:`dict`
        # {(key, i): :obj:`pandas.DataFrame`} dictionary of all parsed (i.e. loaded as DataFrame) TSV files.
        # """
        #
        # self._tsv_types = {}
        # """:obj:`dict`
        # {(key, i): :obj:`str`} dictionary of TSV types as inferred by :py:meth:`._infer_tsv_type`, i.e. one of
        # ``None, 'notes', 'events', 'chords', 'rests', 'measures', 'labels'}``
        # """
        #
        # self._playthrough2mc = {}
        # """:obj:`dict`
        # {(key, i): :obj:`pandas.Series`} dictionary of a parsed score's MC succession after 'unfolding' all repeats.
        # """
        #
        # self._quarter_offsets = {True: {}, False: {}}
        # """:obj:`dict`
        # { unfolded? -> {(key, i) -> {mc_playthrough -> quarter_offset}} } dictionary with keys True and false.
        # True: For every mc_playthrough (i.e., after 'unfolding' all repeats) the total sum of preceding quarter beats, measured from m. 1, b. 0.
        # False: For every mc the total sum of preceding quarter beats after deleting all but second endings.
        # """
        #
        self._views: dict = {}
        self._views[None] = legacy_params2view(paths=paths, file_re=file_re, folder_re=folder_re, exclude_re=exclude_re)
        self._views['all'] = View('all')
        #
        # self._ignored_warnings = defaultdict(list)
        # """:obj:`collections.defaultdict`
        # {'logger_name' -> [(message_id), ...]} This dictionary stores the warnings to be ignored
        # upon loading them from an IGNORED_WARNINGS file.
        # """
        #
        self.labels_cfg = {
            'staff': None,
            'voice': None,
            'harmony_layer': None,
            'positioning': False,
            'decode': True,
            'column_name': 'label',
            'color_format': None,
        }
        """:obj:`dict`
        Configuration dictionary to determine the output format of :py:attr:`~.score.Score.labels` and
        :py:attr:`~.score.Score.expanded` tables. The dictonary is passed to :py:attr:`~.score.Score` upon parsing.
        """
        self.labels_cfg.update(update_labels_cfg(labels_cfg, logger=self.logger))
        #
        # self._dataframes = {
        #     'notes': self._notelists,
        #     'rests': self._restlists,
        #     'notes_and_rests': self._noterestlists,
        #     'measures': self._measurelists,
        #     'events': self._eventlists,
        #     'labels': self._labellists,
        #     'chords': self._chordlists,
        #     'expanded': self._expandedlists,
        #     'cadences': self._cadencelists,
        #     'form_labels': self._fl_lists,
        # }
        # """:obj:`dict`
        # Dictionary exposing the different :obj:`dicts<dict>` of :obj:`DataFrames<pandas.DataFrame>`.
        # """
        #
        #
        # self._matches = pd.DataFrame(columns=['scores']+list(Score.dataframe_types))
        # """:obj:`pandas.DataFrame`
        # Dataframe that holds the (file name) matches between MuseScore and TSV files.
        # """
        #
        # self.corpus2fname2score = defaultdict(dict)
        # """:obj:`collections.defaultdict`
        # {key -> {fname -> score_id}} For each corpus: the list of names identifying pieces.
        # """
        #
        # self._pieces = defaultdict(dict)
        # """:obj:`collections.defaultdict`
        # {key -> {fname -> :class:`.Piece`}} For each corpus: the list of names identifying pieces.
        # """
        #
        # self.id2piece_id = {}
        # """:obj:`dict`
        # {(key, i) -> (key, fname)} dict for associating parsed files with their piece.
        # """
        #
        # self.last_scanned_dir = directory
        # """:obj:`str`
        # The directory that was scanned for files last.
        # """
        if directory is not None:
            if isinstance(directory, str):
                directory = [directory]
            for d in directory:
                self.add_dir(directory=d, recursive=recursive)
        if paths is not None:
            _ = self.add_files(paths)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END of __init__() %%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    def set_view(self, current: View = None, **views: View):
        """Register one or several view_name=View pairs."""
        if current is not None:
            self._views[None] = current
        for view_name, view in views.items():
            if view.name is None:
                view.name = view_name
            self._views[view_name] = view
        for corpus_name, corpus in self:
            if current is not None and current.check_token('corpus', corpus_name):
                corpus.set_view(current)
            for view_name, view in views.items():
                if view.check_token('corpus', corpus_name):
                    corpus.set_view(**{view_name: view})


    def get_view(self,
                 view_name: Optional[str] = None,
                 **config
                 ) -> View:
        """Retrieve an existing or create a new View object, potentially while updating the config."""
        if view_name in self._views:
            view = self._views[view_name]
        elif view_name is not None and self._views[None].name == view_name:
            view = self._views[None]
        else:
            view = self.get_view().copy(new_name=view_name)
            self._views[view_name] = view
            self.logger.info(f"New view '{view_name}' created.")
        if len(config) > 0:
            view.update_config(**config)
        return view

    @property
    def views(self):
        print(pretty_dict({"[active]" if k is None else k: v for k, v in self._views.items()}, "view_name", "Description"))

    @property
    def view_names(self):
        return {view.name if name is None else name for name, view in self._views.items()}

    def switch_view(self, view_name: str,
                    show_info: bool = True,
                    propagate = True,
                    ) -> None:
        if view_name is None:
            return
        new_view = self.get_view(view_name)
        old_view = self.get_view()
        self._views[old_view.name] = old_view
        self._views[None] = new_view
        new_name = new_view.name
        if new_name in self._views:
            del(self._views[new_name])
        if propagate:
            for corpus_name, corpus in self:
                active_view = corpus.get_view()
                if active_view.name != new_name or active_view != new_view:
                    corpus.set_view(new_view)
        if show_info:
            self.info()


    def __getattr__(self, view_name) -> View:
        if view_name in self.view_names:
            self.switch_view(view_name, show_info=False)
            return self
        else:
            raise AttributeError(f"'{view_name}' is not an existing view. Use _.get_view('{view_name}') to create it.")

    def _add_annotations_by_ids(self, list_of_pairs, staff=None, voice=None, harmony_layer=1,
                                check_for_clashes=False):
        """ For each pair, adds the labels at tsv_id to the score at score_id.

        Parameters
        ----------
        list_of_pairs : list of (score_id, tsv_id)
            IDs of parsed files.
        staff : :obj:`int`, optional
            By default, labels are added to staves as specified in the TSV or to -1 (lowest).
            Pass an integer to specify a staff.
        voice : :obj:`int`, optional
            By default, labels are added to voices (notational layers) as specified in the TSV or to 1 (main voice).
            Pass an integer to specify a voice.
        harmony_layer : :obj:`int`, optional
            | By default, the labels are written into the staff's layer for Roman Numeral Analysis.
            | To change the behaviour pass
            | * None to instead attach them as absolute ('guitar') chords, meaning that when opened next time,
            |   MuseScore will split and encode those beginning with a note name ( resulting in ms3-internal harmony_layer 3).
            | * 2 to have MuseScore interpret them as Nashville Numbers
        check_for_clashes : :obj:`bool`, optional
            Defaults to True, meaning that the positions where the labels will be inserted will be checked for existing
            labels.
        """
        self._add_detached_annotations_by_ids(list_of_pairs, new_key='labels_to_attach')
        for score_id, tsv_id in list_of_pairs:
            score = self[score_id]
            score.attach_labels('labels_to_attach', staff=staff, voice=voice, harmony_layer=harmony_layer,
                                check_for_clashes=check_for_clashes,
                                remove_detached=True)



    def _add_detached_annotations_by_ids(self, list_of_pairs, new_key):
        """ For each pair, adds the labels at tsv_id to the score at score_id as a detached
        :py:class:`~ms3.annotations.Annotations` object.

        Parameters
        ----------
        list_of_pairs : list of (score_id, tsv_id)
            IDs of parsed files.
        new_key : :obj:`str`
            The key under which the detached annotations can be addressed using Score[new_key].
        """
        assert list_of_pairs is not None, "list_of_pairs cannot be None"
        for score_id, tsv_id in list_of_pairs:
            if pd.isnull(score_id):
                self.logger.info(f"No score found for annotation table {tsv_id}")
                continue
            if pd.isnull(tsv_id):
                self.logger.info(f"No labels found for score {score_id}")
                continue
            if score_id not in self._parsed_mscx:
                self.logger.info(f"{score_id} has not been parsed yet.")
                continue
            if tsv_id in self._annotations:
                k = tsv_id[0] if pd.isnull(new_key) else new_key
                try:
                    self._parsed_mscx[score_id].load_annotations(anno_obj=self._annotations[tsv_id], key=k)
                except:
                    print(f"score_id: {score_id}, labels_id: {tsv_id}")
                    raise
            elif tsv_id in self._parsed_tsv:
                k, i = tsv_id
                self.logger.warning(f"""The TSV {tsv_id} has not yet been parsed as Annotations object. Use parse_tsv(key='{k}') and specify cols={{'label': label_col}}.""")
            else:
                self.logger.debug(
                    f"Nothing to add to {score_id}. Make sure that its counterpart has been recognized as tsv_type 'labels' or 'expanded'.")


    def _make_grouped_ids(self, keys=None, ids=None):
        if ids is not None:
            grouped_ids = group_id_tuples(ids)
        else:
            grouped_ids = {k: list(range(len(self.piece_names[k]))) for k in self._treat_key_param(keys)}
        return grouped_ids

    def _concat_id_df_dict(self, dict_of_dataframes, id_index=False, third_level_name=None):
        """Concatenate DataFrames contained in a {ID -> df} dictionary.

        Parameters
        ----------
        dict_of_dataframes : :obj:`dict`
            {ID -> DataFrame}
        id_index : :obj:`bool`, optional
            By default, the concatenated data will be differentiated through a three-level MultiIndex with the levels
            'rel_paths', 'fnames', and '{which}_id'. Pass True if instead you want the first two levels to correspond to the file's IDs.
        third_level_name : :obj:`str`, optional

        Returns
        -------
        :obj:`pandas.DataFrame`
        """
        d = {k: v for k, v in dict_of_dataframes.items() if v.shape[0] > 0}
        if len(d) == 0:
            self.logger.info(f"Nothing to concatenate:\n{dict_of_dataframes}")
            return
        if id_index:
            result = pd.concat(d.values(), keys=d.keys())
            result.index.names = ['key', 'i', third_level_name]
        else:
            levels = [(self.rel_paths[key][i], self.fnames[key][i]) for key, i in d.keys()]
            result = pd.concat(d.values(), keys=levels)
            result.index.names = ['rel_paths', 'fnames', third_level_name]
        return result


    def _concat_lists(self, which, id_index=False, keys=None, ids=None, quarterbeats=False, unfold=False, interval_index=False):
        """ Boiler plate for concatenating DataFrames with the same type of information.

        Parameters
        ----------
        which : {'cadences', 'chords', 'events', 'expanded', 'labels', 'measures', 'notes_and_rests', 'notes', 'rests', 'form_labels'}
        id_index : :obj:`bool`, optional
            By default, the concatenated data will be differentiated through a three-level MultiIndex with the levels
            'rel_paths', 'fnames', and '{which}_id'. Pass True if instead you want the first two levels to correspond to the file's IDs.
        keys
        ids

        Returns
        -------

        """
        d = self.get_dataframes(keys, ids, flat=False, quarterbeats=quarterbeats, unfold=unfold, interval_index=interval_index, **{which: True})
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
            'form_labels': 'form label tables'
        }
        if len(d) == 0:
            if keys is None and ids is None:
                self.logger.info(f'This Parse object does not include any {msg[which]}.')
            else:
                self.logger.info(f'keys={keys}, ids={ids}, does not yield any {msg[which]}.')
            return pd.DataFrame()
        return self._concat_id_df_dict(d, id_index=id_index, third_level_name=f"{which}_id")

    def cadences(self, keys=None, ids=None, quarterbeats=False, unfold=False, interval_index=False):
        return self._concat_lists('cadences', keys=keys, ids=ids, quarterbeats=quarterbeats, unfold=unfold,
                                  interval_index=interval_index)

    def chords(self, keys=None, ids=None, quarterbeats=False, unfold=False, interval_index=False):
        return self._concat_lists('chords', keys=keys, ids=ids, quarterbeats=quarterbeats, unfold=unfold,
                                  interval_index=interval_index)

    def events(self, keys=None, ids=None, quarterbeats=False, unfold=False, interval_index=False):
        return self._concat_lists('events', keys=keys, ids=ids, quarterbeats=quarterbeats, unfold=unfold,
                                  interval_index=interval_index)

    def expanded(self, keys=None, ids=None, quarterbeats=False, unfold=False, interval_index=False):
        return self._concat_lists('expanded', keys=keys, ids=ids, quarterbeats=quarterbeats, unfold=unfold,
                                  interval_index=interval_index)

    def form_labels(self, keys=None, ids=None, quarterbeats=False, unfold=False, interval_index=False):
        return self._concat_lists('form_labels', keys=keys, ids=ids, quarterbeats=quarterbeats, unfold=unfold,
                                  interval_index=interval_index)

    def labels(self, keys=None, ids=None, quarterbeats=False, unfold=False, interval_index=False):
        return self._concat_lists('labels', keys=keys, ids=ids, quarterbeats=quarterbeats, unfold=unfold,
                                  interval_index=interval_index)

    def measures(self, keys=None, ids=None, quarterbeats=False, unfold=False, interval_index=False):
        return self._concat_lists('measures', keys=keys, ids=ids, quarterbeats=quarterbeats, unfold=unfold,
                                  interval_index=interval_index)

    def notes(self, keys=None, ids=None, quarterbeats=False, unfold=False, interval_index=False):
        return self._concat_lists('notes', keys=keys, ids=ids, quarterbeats=quarterbeats, unfold=unfold,
                                  interval_index=interval_index)

    def notes_and_rests(self, keys=None, ids=None, quarterbeats=False, unfold=False, interval_index=False):
        return self._concat_lists('notes_and_rests', keys=keys, ids=ids, quarterbeats=quarterbeats, unfold=unfold,
                                  interval_index=interval_index)

    def rests(self, keys=None, ids=None, quarterbeats=False, unfold=False, interval_index=False):
        return self._concat_lists('rests', keys=keys, ids=ids, quarterbeats=quarterbeats, unfold=unfold,
                                  interval_index=interval_index)

    def ids(self, keys=None):
        data = {}
        keys = self._treat_key_param(keys)
        for key in keys:
            fnames = self.fnames[key]
            for i, fname in enumerate(fnames):
                id = (key, i)
                data[id] = os.path.join(self.rel_paths[key][i], fname)
        result = pd.Series(data, name='file')
        result.index = pd.MultiIndex.from_tuples(result.index, names=['key', 'i'])
        return result


    @property
    def ms(self):
        return self._ms

    @ms.setter
    def ms(self, ms):
        self._ms = get_musescore(ms, logger=self.logger)


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


    def add_labels(self, use=None):
        for _, view in self:
            view.add_labels(use=use)



    def add_detached_annotations(self, keys=None, use=None, tsv_key=None, new_key='old', revision_specifier=None):
        """ Add :py:attr:`~.annotations.Annotations` objects generated from TSV files to the :py:attr:`~.score.Score`
        objects to which they are being matched based on their filenames or on ``match_dict``.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
            Key(s) under which score files are stored. By default, all keys are selected.
        use : :obj:`str`, optional
            By default, if several sets of annotation files are found, the user is asked to input
            in which order to pick them. Instead, they can specify the name of a column of
            _.pieces(), especially 'expanded' or 'labels' to be using only these.
        new_key : :obj:`str`, optional
            The key under which the :py:attr:`~.annotations.Annotations` objects will be available after attaching
            them to the :py:attr:`~.score.Score` objects (``Parsed.parsed_mscx[ID].new_key``).
        tsv_key : :obj:`str`, optional
            A key under which parsed TSV files are stored of which the type has been inferred as 'labels'.
            Note that passing ``tsv_key`` results in the deprecated use of Parse.match_files(). The preferred way
            is to parse the labels to be attached under the same key as the scores and use
            View.add_detached_labels().
        revision_specifier : :obj:`str`, optional
            If you want to retrieve a previous version of the TSV file from a git commit (e.g. for
            using compare_labels()), pass the commit's SHA.
        """
        keys = self._treat_key_param(keys)
        if tsv_key is None:
            for key in keys:
                view = self.get_view(key)
                view.add_detached_annotations(use=use, new_key=new_key, revision_specifier=revision_specifier)
            return
        matches = self.match_files(keys=keys + [tsv_key])
        matches = matches[matches.labels.notna() | matches.expanded.notna()]
        matches.labels.fillna(matches.expanded, inplace=True)
        list_of_pairs = list(matches[['scores', 'labels']].itertuples(name=None, index=False))
        if len(list_of_pairs) == 0:
            self.logger.error(f"No files could be matched based on file names, probably a bug due to the deprecated use of the tsv_key parameter."
                    f"The preferred way of adding labels is parsing them under the same key as the scores and use Parse[key].add_detached_annotations()")
            return
        self._add_detached_annotations_by_ids(list_of_pairs, new_key=new_key)



    def add_dir(self, directory, recursive=True, **logger_cfg) -> None:
        """
        This method decides if the directory ``directory`` contains several corpora or if it is a corpus
        itself, and calls _.add_corpus() for every corpus.

        Parameters
        ----------
        directory : :obj:`str`
            Directory to scan for files.
        file_re : :obj:`str`, optional
            Regular expression for filtering certain file names. By default, all parseable score files and TSV files are detected,
            depending on whether the MuseScore 3 executable is specified as :py:attr:``Parse.ms``, or not.
            The regEx is checked with search(), not match(), allowing for fuzzy search.
        folder_re : :obj:`str`, optional
            Regular expression for filtering certain folder names.
            The regEx is checked with search(), not match(), allowing for fuzzy search.
        exclude_re : :obj:`str`, optional
            Any files or folders (and their subfolders) including this regex will be disregarded. By default, files
            whose file names include ``_reviewed`` or start with ``.`` or ``_`` or ``concatenated`` are excluded.
        recursive : :obj:`bool`, optional
            By default, sub-directories are recursively scanned. Pass False to scan only ``dir``.
        """
        directory = resolve_dir(directory)
        if not os.path.isdir(directory):
            self.logger.warning(f"{directory} is not an existing directory.")
            return

        if not recursive:
            self.add_corpus(directory, )
            return

        # new corpus/corpora to be added
        subdir_corpora = sorted(get_first_level_corpora(directory, logger=self.logger))
        n_corpora = len(subdir_corpora)
        if n_corpora == 0:
            self.logger.debug(f"Treating {directory} as corpus.")
            self.add_corpus(directory, logger_cfg=logger_cfg, **logger_cfg)
        else:
            self.logger.debug(f"{n_corpora} individual corpora detected in {directory}.")
            for corpus_path in subdir_corpora:
                self.add_corpus(corpus_path, **logger_cfg)


    def add_corpus(self, directory, corpus_name=None, **logger_cfg) -> None:
        """
        This method scans the directory ``directory`` for files matching the criteria and groups their paths, file names, and extensions
        under the same key, considering them as one corpus.
        to the Parse object without looking at them. The

        Parameters
        ----------
        directory : :obj:`str`
            Directory to scan for files.
        corpus_name : :obj:`str`, optional
            By default, the folder name of ``directory`` is used as name for this corpus. Pass a string to
            use a different identifier.
        file_re : :obj:`str`, optional
            Regular expression for filtering certain file names. By default, all parseable score files and TSV files are detected,
            depending on whether the MuseScore 3 executable is specified as :py:attr:``Parse.ms``, or not.
            The regEx is checked with search(), not match(), allowing for fuzzy search.
        folder_re : :obj:`str`, optional
            Regular expression for filtering certain folder names.
            The regEx is checked with search(), not match(), allowing for fuzzy search.
        exclude_re : :obj:`str`, optional
            Any files or folders (and their subfolders) including this regex will be disregarded. By default, files
            whose file names include ``_reviewed`` or start with ``.`` or ``_`` or ``concatenated`` are excluded.
        recursive : :obj:`bool`, optional
            By default, sub-directories are recursively scanned. Pass False to scan only ``dir``.
        **logger_cfg:

        """
        directory = resolve_dir(directory)
        if not os.path.isdir(directory):
            self.logger.warning(f"{directory} is not an existing directory.")
            return
        new_logger_cfg = dict(self.logger_cfg)
        new_logger_cfg.update(logger_cfg)
        if corpus_name is None:
            corpus_name = os.path.basename(directory).strip(r"\/")
        logger_cfg['name'] = self.logger.name + '.' + corpus_name.replace('.', '')
        try:
            corpus = Corpus(directory=directory, view=self.get_view(), ms=self.ms, **logger_cfg)
        except AssertionError:
            self.logger.debug(f"{directory} contains no parseable files.")
            return
        corpus.set_view(**{view_name: view for view_name, view in self._views.items() if view_name is not None})
        if len(corpus.files) == 0:
            self.logger.info(f"No parseable files detected in {directory}. Skipping...")
            return
        if corpus_name is None:
            corpus_name = os.path.basename(directory).strip(r'\/')
        if corpus_name in self.corpus_paths:
            existing_path = self.corpus_paths[corpus_name]
            if existing_path == directory:
                self.logger.warning(f"Corpus '{corpus_name}' had already been present and was overwritten, i.e., reset.")
            else:
                self.logger.warning(f"Corpus '{corpus_name}' had already been present for the path {existing_path} and "
                                    f"was replaced by {directory}")
        self.corpus_paths[corpus_name] = directory
        self.corpus_objects[corpus_name] = corpus
        # convertible = self.ms is not None
        # if file_re is None:
        #     file_re = Score._make_extension_regex(tsv=True, convertible=convertible)
        # if exclude_re is None:
        #     exclude_re = r'(^(\.|_|concatenated_)|_reviewed)'
        # directory = resolve_dir(directory)
        # self.last_scanned_dir = directory
        # if key is None:
        #     key = os.path.basename(directory)
        # if key not in self.files:
        #     self.logger.debug(f"Adding {directory} as new corpus with key '{key}'.")
        #     self.files[key] = []
        #     self.corpus_paths[key] = resolve_dir(directory)
        # else:
        #     self.logger.info(f"Adding {directory} to existing corpus with key '{key}'.")
        #
        # top_level_folders, top_level_files = first_level_files_and_subdirs(directory)
        # self.logger.debug(f"Top level folders: {top_level_folders}\nTop level files: {top_level_files}")
        #
        # added_ids = []
        #
        # # look for scores
        # scores_folder = None
        # if 'scores' in kwargs and kwargs['scores'] in top_level_folders:
        #     scores_folder = kwargs['scores']
        # elif 'MS3' in top_level_folders:
        #     scores_folder = 'MS3'
        # elif 'scores' in top_level_folders:
        #     scores_folder = 'scores'
        # else:
        #     msg = f"No scores folder found among {top_level_folders}."
        #     if 'scores' not in kwargs:
        #         msg += " If one of them has MuseScore files, indicate it by passing scores='scores_folder'."
        #     self.logger.info(msg)
        # if scores_folder is not None:
        #     score_re = Score._make_extension_regex(convertible=convertible)
        #     scores_path = os.path.join(directory, scores_folder)
        #     score_paths = sorted(scan_directory(scores_path, file_re=score_re, recursive=recursive))
        #     score_ids = self.add_files(paths=score_paths, key=key)
        #     added_ids += score_ids
        #     score_fnames = self._get_unambiguous_fnames_from_ids(score_ids, key=key)
        #
        #     for fname, id in score_fnames.items():
        #         piece = self._get_piece(key, fname)
        #         piece.type2file_info['score'] = self.id2file_info[id]
        #         self.id2piece_id[id] = (key, fname)
        #         # if fname in self.corpus2fname2score[key]:
        #         #     if self.corpus2fname2score[key][fname] == id:
        #         #         self.debug(f"'{fname} had already been matched to {id}.")
        #         #     else:
        #         #         self.warning(f"'{fname} had already been matched to {self.corpus2fname2score[key][fname]}")
        #     self.corpus2fname2score[key].update(score_fnames)
        #
        # # look for metadata
        # if 'metadata.tsv' in top_level_files:
        #     default_metadata_path = os.path.join(directory, 'metadata.tsv')
        #     self.logger.debug(f"'metadata.tsv' was detected and added.")
        #     added_ids += self.add_files(paths=default_metadata_path, key=key)
        #     metadata_id = added_ids[-1]
        #     self.parse_tsv(ids=[metadata_id])
        #     metadata_tsv = self._parsed_tsv[metadata_id]
        #     metadata_fnames = metadata_tsv.fnames
        # else:
        #     metadata_id = None
        #
        #
        #
        # return
        # paths = sorted(
        #     scan_directory(directory, file_re=file_re, folder_re=folder_re, exclude_re=exclude_re,
        #                    recursive=recursive, logger=self.logger))
        # if len(paths) == 0:
        #     self.logger.info(f"No matching files found in {directory}.")
        #     return
        # added_ids = self.add_files(paths=paths, key=key)
        # if len(added_ids) == 0:
        #     self.logger.debug(f"No files from {directory} have been added.")
        #     return
        # _, first_i = added_ids[0]
        # if 'metadata.tsv' in self.files[key][first_i:]:
        #     self.logger.debug(f"Found metadata.tsv for corpus '{key}'.")
        # elif 'metadata.tsv' in self.files[key]:
        #     self.logger.debug(f"Had already found metadata.tsv for corpus '{key}'.")
        # else:
        #     # if no metadata have been found (e.g. because excluded via file_re), add them if they're there
        #     default_metadata_path = os.path.join(directory, 'metadata.tsv')
        #     if os.path.isfile(default_metadata_path):
        #         self.logger.info(f"'metadata.tsv' was detected and automatically added for corpus '{key}'.")
        #         metadata_id = self.add_files(paths=default_metadata_path, key=key)
        #         added_ids += metadata_id
        #     else:
        #         self.logger.info(f"No metadata found for corpus '{key}'.")
        # self.corpus_paths[key] = directory
        # self.look_for_ignored_warnings(directory, key)

    def look_for_ignored_warnings(self, directory, key):
        default_ignored_warnings_path = os.path.join(directory, 'IGNORED_WARNINGS')
        if os.path.isfile(default_ignored_warnings_path):
            self.logger.info(f"IGNORED_WARNINGS detected for {key}.")
            self.load_ignored_warnings(default_ignored_warnings_path)

    def load_ignored_warnings(self, path):
        ignored_warnings = parse_ignored_warnings_file(path)  # parse IGNORED_WARNINGS file into a {logger_name -> [message_id]} dict
        logger_names = set(self.logger_names.values())
        all_logger_names = set()
        for name in logger_names:
            while name != '' and name not in all_logger_names:
                all_logger_names.add(name)
                try:
                    split_pos = name.rindex('.')
                    name = name[:split_pos]
                except:
                    name = ''
        for to_be_configured, message_ids in ignored_warnings.items():
            self._ignored_warnings[to_be_configured].extend(message_ids)
            if to_be_configured not in all_logger_names:
                self.logger.warning(f"This Parse object is not using any logger called '{to_be_configured}', "
                                    f"which was, however, indicated in {path}.", extra={"message_id": (14, to_be_configured)})
            configured = get_logger(to_be_configured, ignored_warnings=message_ids)
            configured.debug(f"This logger has been configured to set warnings with the following IDs to DEBUG:\n{message_ids}.")

    def add_files(self, paths, corpus_name=None, exclude_re=None):
        """

        Parameters
        ----------
        paths : :obj:`~collections.abc.Collection`
            The paths of the files you want to add to the object.
        corpus_name : :obj:`str`
            | Pass a string to identify the loaded files.
            | If None is passed, :meth:`.utils.path2parent_corpus` will try to see if one of the
            | superdirectories is a corpus and use its name. Otherwise, if only one corpus
            | is available, that one will be chosen. If no corpus_name can be inferred, the files
            | will not be added.

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
        unpack_json_paths(paths, logger=self.logger)
        if corpus_name is None:
            # try to see if any of the paths is part of a corpus (superdir has 'metadata.tsv')
            for path in paths:
                parent_corpus = path2parent_corpus(path)
                if parent_corpus is not None:
                    corpus_name = os.path.basename(parent_corpus)
                    self.logger.info(f"Using key='{corpus_name}' because the directory contains 'metadata.tsv' or is a git repository.")
                    metadata_path = os.path.join(parent_corpus, 'metadata.tsv')
                    if metadata_path not in paths:
                        paths.append(metadata_path)
                        self.logger.info(f"{metadata_path} was automatically added.")
                    break
        if corpus_name is None:
            # if only one key is available, pick that one
            keys = self.keys()
            if len(keys) == 1:
                corpus_name = keys[0]
                self.logger.info(f"Using key='{corpus_name}' because it is the only one currently in use.")
            else:
                self.logger.error(f"Couldn't add individual files because no key was specified and no key could be inferred.",
                                  extra={"message_id": (8,)})
                return []
        if corpus_name not in self.files:
            self.logger.debug(f"Adding '{corpus_name}' as new corpus.")
            self.files[corpus_name] = []
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

        self.logger.debug(f"Attempting to add {len(paths)} files...")
        if corpus_name is None:
            ids = [self._handle_path(p, path2parent_corpus(p)) for p in paths]
        else:
            ids = [self._handle_path(p, corpus_name) for p in paths]
        if sum(True for x in ids if x[0] is not None) > 0:
            selector, added_ids = zip(*[(i, x) for i, x in enumerate(ids) if x[0] is not None])
            exts = self.count_extensions()
            self.logger.debug(f"{len(added_ids)} paths stored:\n{pretty_dict(exts, 'EXTENSIONS')}")
            return added_ids
        else:
            self.logger.info("No files added.")
            return []


    def annotation_objects(self):
        """Iterator through all annotation objects."""
        yield from self._annotations.items()




    def attach_labels(self, keys=None, annotation_key=None, staff=None, voice=None, harmony_layer=None, check_for_clashes=True):
        """ Attach all :py:attr:`~.annotations.Annotations` objects that are reachable via ``Score.annotation_key`` to their
        respective :py:attr:`~.score.Score`, changing their current XML. Calling :py:meth:`.output_mscx` will output
        MuseScore files where the annotations show in the score.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
            Key(s) under which parsed MuseScore files are stored. By default, all keys are selected.
        annotation_key : :obj:`str` or :obj:`list` or :obj:`tuple`, optional
            Key(s) under which the :py:attr:`~.annotations.Annotations` objects to be attached are stored in the
            :py:attr:`~.score.Score` objects. By default, all keys are selected.
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
                    r, g = self._parsed_mscx[id].attach_labels(anno_key, staff=staff, voice=voice, harmony_layer=harmony_layer, check_for_clashes=check_for_clashes)
                    self.logger.info(f"{r}/{g} labels successfully added to {self.files[id[0]][id[1]]}")
                    reached += r
                    goal += g
        self.logger.info(f"{reached}/{goal} labels successfully added to {len(ids)} files.")
        self._collect_annotations_objects_references(ids=ids)


    def change_labels_cfg(self, labels_cfg={}, staff=None, voice=None, harmony_layer=None, positioning=None, decode=None, column_name=None, color_format=None):
        """ Update :obj:`Parse.labels_cfg` and retrieve new 'labels' tables accordingly.

        Parameters
        ----------
        labels_cfg : :obj:`dict`
            Using an entire dictionary or, to change only particular options, choose from:
        staff, voice, harmony_layer, positioning, decode, column_name
            Arguments as they will be passed to :py:meth:`~ms3.annotations.Annotations.get_labels`
        """
        keys = ['staff', 'voice', 'harmony_layer', 'positioning', 'decode', 'column_name', 'color_format']
        for k in keys:
            val = locals()[k]
            if val is not None:
                labels_cfg[k] = val
        updated = update_labels_cfg(labels_cfg, logger=self.logger)
        self.labels_cfg.update(updated)
        for score in self._parsed_mscx.values():
            score.change_labels_cfg(labels_cfg=updated)
        ids = list(self._labellists.keys())
        if len(ids) > 0:
            self._extract_and_cache_dataframes(ids=ids, labels=True)


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

    def color_non_chord_tones(self, keys=None, ids=None, color_name='red'):
        if len(self._parsed_mscx) == 0:
            self.logger.info("No scores have been parsed so far. Use parse_mscx()")
            return
        if ids is None:
            ids = list(self._iterids(keys, only_parsed_mscx=True))
        else:
            ids = [id for id in ids if id in self._parsed_mscx]
        result = {}
        for id in ids:
            score = self._parsed_mscx[id]
            result[id] = score.color_non_chord_tones(color_name=color_name)
        return result


    def _extract_and_cache_dataframes(self, keys=None, ids=None, notes=False, rests=False, notes_and_rests=False, measures=False, events=False,
                                      labels=False, chords=False, expanded=False, form_labels=False, cadences=False, only_new=True):
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
        if len(self._parsed_mscx) == 0:
            self.logger.debug("No scores have been parsed so far. Use Parse.parse_mscx()")
            return
        if ids is None:
            only_new = False
            ids = list(self._iterids(keys, only_parsed_mscx=True))
        scores = {id: self._parsed_mscx[id] for id in ids if id in self._parsed_mscx}
        bool_params = Score.dataframe_types
        l = locals()
        params = {p: l[p] for p in bool_params}

        for i, score in scores.items():
            for param, li in self._dataframes.items():
                if params[param] and (i not in li or not only_new):
                    if self.simulate:
                        df = pd.DataFrame()
                    else:
                        df = score.mscx.__getattribute__(param)()
                    if df is not None:
                        li[i] = df


    def compare_labels(self, detached_key, new_color='ms3_darkgreen', old_color='red',
                       detached_is_newer=False):
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

        Returns
        -------
        :obj:`dict`
            {ID -> (n_new_labels, n_removed_labels)} dictionary.
        """
        assert detached_key != 'annotations', "Pass a key of detached labels, not 'annotations'."
        ids = list(self._iterids(None, only_detached_annotations=True))
        if len(ids) == 0:
            if len(self._parsed_mscx) == 0:
                self.logger.info("No scores have been parsed so far.")
                return {}
            self.logger.info("None of the parsed scores include detached labels to compare.")
            return {}
        available_keys = set(k for id in ids for k in self._parsed_mscx[id]._detached_annotations)
        if detached_key not in available_keys:
            self.logger.info(f"""None of the parsed scores include detached labels with the key '{detached_key}'.
Available keys: {available_keys}""")
            return {}
        ids = [id for id in ids if detached_key in self._parsed_mscx[id]._detached_annotations]
        self.logger.info(f"{len(ids)} parsed scores include detached labels with the key '{detached_key}'.")
        comparison_results = {}
        for id in ids:
            comparison_results[id] = self._parsed_mscx[id].compare_labels(detached_key=detached_key, new_color=new_color, old_color=old_color,
                                                 detached_is_newer=detached_is_newer)
        return comparison_results


    def count_annotation_layers(self, keys=None, which='attached', per_key=False):
        """ Counts the labels for each annotation layer defined as (staff, voice, harmony_layer).
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
            By default, the function returns a Counter of labels for every annotation layer (staff, voice, harmony_layer)
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
            names = ['staff', 'voice', 'harmony_layer', 'color'] #<[:levels]
            ix = pd.MultiIndex.from_tuples(ks, names=names)
            return pd.Series(data, ix)

        if per_key:
            res = {k: make_series(v) for k, v in res_dict.items()}
        else:
            res = make_series(sum(res_dict.values(), Counter()))
        if len(res) == 0:
            self.logger.info("No annotations found. Maybe no scores have been parsed using parse_mscx()?")
        return res


    def count_extensions(self, view_name: Optional[str] = None, per_piece: bool = False):
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
        per_subdir : :obj:`bool`, optional
            If set to True, the results are returned as {key: {subdir: Counter} }. ``per_key=True`` is therefore implied.

        Returns
        -------
        :obj:`dict`
            By default, the function returns a Counter of file extensions (Counters are converted to dicts).
            If ``per_key`` is set to True, a dictionary {key: Counter} is returned, separating the counts.
            If ``per_subdir`` is set to True, a dictionary {key: {subdir: Counter} } is returned.

        Args:
            view_name:
        """
        extension_counters = {corpus_name: corpus.count_extensions(view_name) for corpus_name, corpus in self.iter_corpora(view_name)}
        if per_piece:
            return {(corpus_name, fname): dict(cnt) for corpus_name, fname2cnt in extension_counters.items() for fname, cnt in fname2cnt.items()}
        return {corpus_name: dict(sum(fname2cnt.values(), start=Counter())) for corpus_name, fname2cnt in extension_counters.items()}




    def count_labels(self, keys=None, per_key=False):
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
            res_dict[key].update(self._annotations[(key, i)].harmony_layer_counts)
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



    def detach_labels(self, keys=None, annotation_key='detached', staff=None, voice=None, harmony_layer=None, delete=True):
        """ Calls :py:meth:`Score.detach_labels<ms3.score.Score.detach_labels` on every parsed score with key ``key``.
        """
        assert annotation_key != 'annotations', "The key 'annotations' is reserved, please choose a different one."
        ids = list(self._iterids(keys, only_attached_annotations=True))
        if len(ids) == 0:
            self.logger.info(f"Selection did not contain scores with labels: keys = '{keys}'")
        for id in ids:
            score = self._parsed_mscx[id]
            try:
                score.detach_labels(key=annotation_key, staff=staff, voice=voice, harmony_layer=harmony_layer, delete=delete)
            except:
                score.logger.error(f"Detaching labels failed with the following error:\n{sys.exc_info()[1]}")
        self._collect_annotations_objects_references(ids=ids)



    def fname2ids(self, fname, key=None, rel_path=None, allow_suffix=True):
        """For a given filename, return corresponding IDs.

        Parameters
        ----------
        fname : :obj:`str`
            Filename (without extension) to get IDs for.
        key : :obj:`str` or :obj:`~collections.abc.Collection`, optional
            If you want to scan through IDs of one or several particular keys, specify.
        rel_path : :obj:`str`, optional
            Passing a rel_path is useful for associating a row from metadata.tsv with a parsed
            MuseScore file.
        allow_suffix : :obj:`bool`, optional
            By default, filenames are matched even if they continue with a suffix. Pass False to return only exact matches.

        Returns
        -------
        :obj:`dict`
            {ID -> filename)
        """
        if allow_suffix:
            l = len(fname)
            ids = {(k, i): self.fnames[k][i]  for k, i in self._iterids(key) if self.fnames[k][i][:l] == fname}
        else:
            ids = {(k, i): self.fnames[k][i] for k, i in self._iterids(key) if self.fnames[k][i] == fname}
        if rel_path is not None:
            ids = {(k, i): fname for (k, i), fname in ids.items() if self.rel_paths[k][i] == rel_path}
        return ids




    def get_labels(self, keys=None, staff=None, voice=None, harmony_layer=None, positioning=True, decode=False, column_name=None,
                   color_format=None, concat=True):
        """ This function does not take into account self.labels_cfg """
        if len(self._annotations) == 0:
            self.logger.error("No labels available so far. Add files using add_dir() and parse them using parse().")
            return pd.DataFrame()
        keys = self._treat_key_param(keys)
        harmony_layer = self._treat_harmony_layer_param(harmony_layer)
        self._extract_and_cache_dataframes(labels=True, only_new=True)
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




    def get_dataframes(self, keys=None, ids=None, notes=False, rests=False, notes_and_rests=False, measures=False,
                       events=False, labels=False, chords=False, expanded=False, cadences=False, form_labels=False,
                       simulate=False, flat=False, unfold=False, quarterbeats=False, interval_index=False):
        """ Retrieve a dictionary with the selected feature matrices extracted from the parsed scores.
        If you want to retrieve parse TSV files, use :py:meth:`get_tsvs`.

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
        form_labels
        simulate
        flat : :obj:`bool`, optional
            By default, you get a nested dictionary {list_type -> {index -> list}}.
            By passing True you get a dictionary {(id, list_type) -> list}
        unfold : :obj:`bool`, optional
            Pass True if lists should reflect repetitions and voltas to create a correct playthrough.
            Defaults to False, meaning that all measures including second endings are included, unless ``quarterbeats``
            is set to True.
        quarterbeats : :obj:`bool`, optional
            Pass True to add a ``quarterbeats`` column with a continuous offset from the piece's beginning. If ``unfold``
            is False, first endings will not be assigned a quarterbeat position. If, additionally, ``interval_index``
            is set to True, they are removed from the DataFrame.
        interval_index : :obj:`bool`, optional
            Sets ``quarterbeats`` to True. Pass True to replace the indices of the returned DataFrames by
            :obj:`pandas.IntervalIndex <pandas.IntervalIndex>` with quarterbeat intervals. Rows that don't have a
            quarterbeat position are removed.

        Returns
        -------
        {feature -> {(key, i) -> pd.DataFrame}} if not flat (default) else {(key, i, feature) -> pd.DataFrame}
        """

        if self.count_parsed_scores() == 0:
            self.logger.error("No scores have been parsed so far.")
            return {}
        if ids is not None:
            grouped_ids = group_id_tuples(ids)
            key2fnames = {key: [self.fnames[key][i] for i in ints] for key, ints in grouped_ids.items()}
        else:
            keys = self._treat_key_param(keys)
            key2fnames = {k: None for k in keys}
        bool_params = Score.dataframe_types
        l = locals()
        columns = [tsv_type for tsv_type in Score.dataframe_types if l[tsv_type]]
        self.logger.debug(f"Looking up {columns} DataFrames for IDs {key2fnames}")
        #self.collect_lists(ids=ids, only_new=True, **params)
        res = {} if flat else defaultdict(dict)
        # if unfold:
        #     playthrough2mcs = self.get_playthrough2mc(ids=ids)
        if interval_index:
            quarterbeats = True
        for key, fnames in key2fnames.items():
            for md, *dfs in self[key].iter_transformed(columns, skip_missing=True, unfold=unfold, quarterbeats=quarterbeats, interval_index=interval_index, fnames=fnames):
                # TODO: Adapt to the fact that now all DataFrames come with quarterbeats already
                for tsv_type, id, df in zip(columns, md['ids'], dfs):
                    if df is not None:
                        if flat:
                            res[id + (tsv_type,)] = df
                        else:
                            res[tsv_type][id] = df
        return dict(res)
        # if unfold or quarterbeats:
        #     _ = self.match_files(ids=ids)
        # for param, li in self._lists.items():
        #     if params[param]:
        #         if not flat:
        #             res[param] = {}
        #         for id in (i for i in ids if i in li):
        #             logger = self.id_logger(id)
        #             df = li[id]
        #             if unfold:
        #                 if id in playthrough2mcs:
        #                     df = unfold_repeats(df, playthrough2mcs[id], logger=logger)
        #                     if quarterbeats:
        #                         offset_dict = self.get_continuous_offsets(id, unfold=True)
        #                         df = add_quarterbeats_col(df, offset_dict, interval_index=interval_index, logger=logger)
        #                 else:
        #                     logger.error(f"Skipped {param} for {id} because of unfolding error.")
        #                     continue
        #             elif quarterbeats:
        #                 if 'volta' in df.columns:
        #                     logger.debug("Only second voltas were included when computing quarterbeats.")
        #                 offset_dict = self.get_continuous_offsets(id, unfold=False)
        #                 df = add_quarterbeats_col(df, offset_dict, interval_index=interval_index, logger=logger)
        #             if id in self._parsed_mscx and len(self._parsed_mscx[id].mscx.volta_structure) == 0 and 'volta' in df.columns:
        #                 if df.volta.isna().all():
        #                     df = df.drop(columns='volta')
        #                 else:
        #                     logger.error(f"Score @ ID {id} does not include any voltas, yet the volta column in the {param} table is not empty.")
        #             if flat:
        #                 res[id + (param,)] = df
        #             else:
        #                 res[param][id] = df
        # return res


    def get_tsvs(self, keys=None, ids=None, metadata=True, notes=False, rests=False, notes_and_rests=False, measures=False,\
                 events=False, labels=False, chords=False, expanded=False, cadences=False, form_labels=False, flat=False):
        """Retrieve a dictionary with the selected feature matrices from the parsed TSV files.
        If you want to retrieve matrices from parsed scores, use :py:meth:`get_lists`.

        Parameters
        ----------
        keys
        ids
        metadata
        notes
        rests
        notes_and_rests
        measures
        events
        labels
        chords
        expanded
        cadences
        form_labels
        flat : :obj:`bool`, optional
            By default, you get a nested dictionary {list_type -> {index -> list}}.
            By passing True you get a dictionary {(id, list_type) -> list}

        Returns
        -------
        {feature -> {(key, i) -> pd.DataFrame}} if not flat (default) else {(key, i, feature) -> pd.DataFrame}
        """


        if ids is None:
            ids = list(self._iterids(keys, only_parsed_tsv=True))
        if len(self._parsed_tsv) == 0:
            self.info(f"No TSV files have been parsed, use method parse_tsv().")
            return {}
        bool_params = ('metadata',) + Score.dataframe_types
        l = locals()
        types = [p for p in bool_params if l[p]]
        res = {}
        if not flat:
            res.update({t: {} for t in types})
        for id in ids:
            tsv_type = self._tsv_types[id]
            if tsv_type in types:
                if flat:
                    res[(id + (tsv_type,))] = self._parsed_tsv[id]
                else:
                    res[tsv_type][id] = self._parsed_tsv[id]
        return res



    def get_playthrough2mc(self, keys=None, ids=None):
        if ids is None:
            ids = list(self._iterids(keys))
        _ = self.match_files(ids=ids)
        res = {}
        for id in ids:
            unf_mcs = self._get_playthrough2mc(id)
            if unf_mcs is not None:
                res[id] = unf_mcs
            # else:
            #     self.id_logger(id).(f"Unable to unfold.")
        return res

    def _get_measure_list(self, id, unfold=False):
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
        logger = self.id_logger(id)
        piece = self.get_piece(id)
        if piece is None:
            logger.warning(f"Could not associate ID {id} with a Piece object.")
            return
        return piece.get_dataframe('measures', unfold=unfold)



    def _get_playthrough2mc(self, id):
        logger = self.id_logger(id)
        if id in self._playthrough2mc:
            return self._playthrough2mc[id]
        ml = self._get_measure_list(id)
        if ml is None:
            logger.warning("No measures table available.")
            self._playthrough2mc[id] = None
            return
        mc_playthrough = make_playthrough2mc(ml, logger=logger)
        if len(mc_playthrough) == 0:
            logger.warning(f"Error in the repeat structure for ID {id}: Did not reach the stopping value -1 in measures.next:\n{ml.set_index('mc').next}")
            mc_playthrough = None
        else:
            logger.debug("Measures successfully unfolded.")
        self._playthrough2mc[id] = mc_playthrough
        return mc_playthrough

    def get_continuous_offsets(self, id, unfold):
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
        logger = self.id_logger(id)
        if id in self._quarter_offsets[unfold]:
            return self._quarter_offsets[unfold][id]
        ml = self._get_measure_list(id, unfold=unfold)
        if ml is None:
            logger.warning(f"Could not find measure list for id {id}.")
            return None
        offset_col = make_continuous_offset_series(ml, logger=logger)
        offsets = offset_col.to_dict()
        self._quarter_offsets[unfold][id] = offsets
        return offsets

    def id_logger(self, id):
        return get_logger(self.logger_names[id])


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
        idx = ids
        names = ['key', 'i']

        if pandas_index:
            idx = pd.MultiIndex.from_tuples(idx, names=names)
            return idx

        return idx, names

    def iter_corpora(self, view_name: Optional[str] = None) -> Generator[Tuple[str, Corpus], None, None]:
        """Iterate through corpora under the current or specified view."""
        view = self.get_view(view_name)
        for corpus_name, corpus in view.filter_by_token('corpora', self):
            if view_name not in corpus._views:
                if view_name is None:
                    corpus.set_view(view)
                else:
                    corpus.set_view(**{view_name: view})
            yield corpus_name, corpus

    def count_files(self,
                    types=True,
                    parsed=True,
                    as_dict: bool = False,
                    drop_zero: bool = True,
                    view_name: Optional[str] = None) -> Union[pd.DataFrame, dict]:
        all_counts = {corpus_name: corpus._summed_file_count(types=types, parsed=parsed, view_name=view_name) for corpus_name, corpus in self.iter_corpora(view_name=view_name)}
        counts_df = pd.DataFrame.from_dict(all_counts, orient='index')
        if drop_zero:
            empty_cols = counts_df.columns[counts_df.sum() == 0]
            counts_df = counts_df.drop(columns=empty_cols)
        if as_dict:
            return counts_df.to_dict(orient='index')
        counts_df.index.rename('corpus', inplace=True)
        return counts_df

    def get_parsed_score_files(self, view_name: Optional[str] = None) -> Dict[CorpusFnameTuple, FileList]:
        result = {}
        for corpus_name, corpus in self.iter_corpora(view_name=view_name):
            fname2files = corpus.get_files('scores', view_name=view_name, unparsed=False, flat=True)
            result[corpus_name] = sum(fname2files.values(), [])
        return result


    def get_unparsed_score_files(self, view_name: Optional[str] = None) -> Dict[CorpusFnameTuple, FileList]:
        result = {}
        for corpus_name, corpus in self.iter_corpora(view_name=view_name):
            fname2files = corpus.get_files('scores', view_name=view_name, parsed=False, flat=True)
            result[corpus_name] = sum(fname2files.values(), [])
        return result

    def get_parsed_tsv_files(self, view_name: Optional[str] = None, flat: bool = True) -> Dict[CorpusFnameTuple, Union[FileDict, FileList]]:
        result = {}
        for corpus_name, corpus in self.iter_corpora(view_name=view_name):
            fname2files = corpus.get_files('tsv', view_name=view_name, unparsed=False, flat=flat)
            if flat:
                result[corpus_name] = sum(fname2files.values(), [])
            else:
                dd = defaultdict(list)
                for fname, typ2files in fname2files.items():
                    for typ, files in typ2files.items():
                        dd[typ].extend(files)
                result[corpus_name] = dict(dd)
        return result

    def get_unparsed_tsv_files(self, view_name: Optional[str] = None, flat: bool = True) -> Dict[CorpusFnameTuple, Union[FileDict, FileList]]:
        result = {}
        for corpus_name, corpus in self.iter_corpora(view_name=view_name):
            fname2files = corpus.get_files('tsv', view_name=view_name, parsed=False, flat=flat)
            if flat:
                result[corpus_name] = sum(fname2files.values(), [])
            else:
                dd = defaultdict(list)
                for fname, typ2files in fname2files.items():
                    for typ, files in typ2files.items():
                        dd[typ].extend(files)
                result[corpus_name] = dict(dd)
        return result


    def count_parsed_scores(self, view_name: Optional[str] = None) -> int:
        return sum(map(len, self.get_parsed_score_files(view_name=view_name).values()))

    def count_parsed_tsvs(self, view_name: Optional[str] = None) -> int:
        return sum(map(len, self.get_parsed_tsv_files(view_name=view_name).values()))

    def count_unparsed_scores(self, view_name: Optional[str] = None) -> int:
        return sum(map(len, self.get_parsed_score_files(view_name=view_name).values()))

    def count_unparsed_tsvs(self, view_name: Optional[str] = None) -> int:
        return sum(map(len, self.get_parsed_tsv_files(view_name=view_name).values()))


    def info(self, return_str: bool = False,
             view_name: Optional[str] = None,
             show_discarded: bool = False):
        """"""
        header = f"All corpora"
        header += "\n" + "-" * len(header) + "\n"

        # start info message with the names of the available views, the header, and info on the active view.
        view = self.get_view(view_name)
        view.reset_filtering_data()
        msg = available_views2str(self._views, view_name)
        msg += header
        msg += f"View: {view}\n\n"

        # Show info on all pieces and files included in the active view
        counts_df = self.count_files(view_name=view_name)
        if counts_df.isna().any().any():
            counts_df = counts_df.fillna(0).astype('int')
        additional_columns = []
        for corpus_name in counts_df.index:
            corpus = self._get_corpus(corpus_name)
            has_metadata = 'no' if corpus.metadata_tsv is None else 'yes'
            corpus_view = corpus.get_view().name
            additional_columns.append([has_metadata, corpus_view])
        additional_columns = pd.DataFrame(additional_columns, columns=[('', 'metadata'), ('', 'view')], index=counts_df.index)
        info_df = pd.concat([additional_columns, counts_df], axis=1)
        info_df.columns = pd.MultiIndex.from_tuples(info_df.columns)
        msg += info_df.to_string() + '\n\n'
        msg += view.filtering_report(show_discarded=show_discarded)
        if return_str:
            return msg
        print(msg)
        # ids = list(self._iterids(keys))
        # info = f"{len(ids)} files.\n"
        # if subdirs:
        #     exts = self.count_extensions(keys, per_subdir=True)
        #     for key, subdir_exts in exts.items():
        #         info += key + '\n'
        #         for line in pretty_dict(subdir_exts).split('\n'):
        #             info += '    ' + line + '\n'
        # else:
        #     exts = self.count_extensions(keys, per_key=True)
        #     info += pretty_dict(exts, heading='EXTENSIONS')
        # parsed_mscx_ids = [id for id in ids if id in self._parsed_mscx]
        # parsed_mscx = len(parsed_mscx_ids)
        # ext_counts = self.count_extensions(keys, per_key=False)
        # others = len(self._score_ids(opposite=True))
        # mscx = len(self._score_ids())
        # by_conversion = len(self._score_ids(native=False))
        # if parsed_mscx > 0:
        #
        #     if parsed_mscx == mscx:
        #         info += f"\n\nAll {mscx} MSCX files have been parsed."
        #     else:
        #         info += f"\n\n{parsed_mscx}/{mscx} MSCX files have been parsed."
        #     annotated = sum(True for id in parsed_mscx_ids if id in self._annotations)
        #     if annotated == mscx:
        #         info += f"\n\nThey all have annotations attached."
        #     else:
        #         info += f"\n\n{annotated} of them have annotations attached."
        #     if annotated > 0:
        #         layers = self.count_annotation_layers(keys, which='attached', per_key=True)
        #         info += f"\n{pretty_dict(layers, heading='ANNOTATION LAYERS')}"
        #
        #     detached = sum(True for id in parsed_mscx_ids if self._parsed_mscx[id].has_detached_annotations)
        #     if detached > 0:
        #         info += f"\n\n{detached} of them have detached annotations:"
        #         layers = self.count_annotation_layers(keys, which='detached', per_key=True)
        #         try:
        #             info += f"\n{pretty_dict(layers, heading='ANNOTATION LAYERS')}"
        #         except:
        #             print(layers)
        #             raise
        # elif '.mscx' in ext_counts:
        #     if mscx > 0:
        #         info += f"\n\nNone of the {mscx} score files have been parsed."
        #         if by_conversion > 0 and self.ms is None:
        #             info += f"\n{by_conversion} files would need to be converted, for which you need to set the 'ms' property to your MuseScore 3 executable."
        # if self.ms is not None:
        #     info += "\n\nMuseScore 3 executable has been found."
        #
        #
        # parsed_tsv_ids = [id for id in ids if id in self._parsed_tsv]
        # parsed_tsv = len(parsed_tsv_ids)
        # if parsed_tsv > 0:
        #     annotations = sum(True for id in parsed_tsv_ids if id in self._annotations)
        #     if parsed_tsv == others:
        #         info += f"\n\nAll {others} tabular files have been parsed, {annotations} of them as Annotations object(s)."
        #     else:
        #         info += f"\n\n{parsed_tsv}/{others} tabular files have been parsed, {annotations} of them as Annotations object(s)."
        #     if annotations > 0:
        #         layers = self.count_annotation_layers(keys, which='tsv', per_key=True)
        #         info += f"\n{pretty_dict(layers, heading='ANNOTATION LAYERS')}"
        #
        # if return_str:
        #     return info
        # print(info)


    def iter(self, columns, keys=None, skip_missing=False):
        keys = self._treat_key_param(keys)
        for key in keys:
            for tup in self[key].iter(columns=columns, skip_missing=skip_missing):
                if tup is not None:
                    yield (key, *tup)

    def iter_transformed(self, columns, keys=None, skip_missing=False, unfold=False, quarterbeats=False, interval_index=False):
        keys = self._treat_key_param(keys)
        for key in keys:
            for tup in self[key].iter_transformed(columns=columns, skip_missing=skip_missing, unfold=unfold, quarterbeats=quarterbeats, interval_index=interval_index):
                if tup is not None:
                    yield (key, *tup)

    def iter_notes(self, keys=None, unfold=False, quarterbeats=False, interval_index=False, skip_missing=False, weight_grace_durations=0):
        keys = self._treat_key_param(keys)
        for key in keys:
            for tup in self[key].iter_notes(unfold=unfold, quarterbeats=quarterbeats, interval_index=interval_index, skip_missing=skip_missing,
                                            weight_grace_durations=weight_grace_durations):
                if tup is not None:
                    yield (key, *tup)


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
        return tuple(self.files.keys())

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
        lists.update(dict(self._dataframes))
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

        matching_candidates = {wh: {(key, i): self.fnames[key][i] for key, i in lists[wh].keys()} for wh in what}
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
                self._matches = pd.concat([self._matches, pd.DataFrame(row).T])
                if len(self._matches) == 1:
                    self._matches.index = pd.MultiIndex.from_tuples(self._matches.index)

        #res_ix = set()
        for j, wh in enumerate(what):
            for id, fname in matching_candidates[wh].items():
                ix = id
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
                            self.logger.debug(f"No {wha} matches for {wh} {id} with filename {file}. Candidates:\n{matching_candidates[wha].values()}")
                            break
                        longest = {id: prefix for id, prefix in matches.items() if lengths[id] == max_length}

                        if len(longest) == 0:
                            self.logger.info(
                                f"No {wha} match found for {wh} {file} among the candidates\n{pretty_dict(matching_candidates[wh])}")
                            continue
                        elif len(longest) > 1:
                            # try to disambiguate by keys
                            key_similarities = {(k, i): len(os.path.commonprefix([key, k])) for k, i in longest.keys()}
                            max_sim = max(key_similarities.values())
                            disambiguated = [id for id, length in key_similarities.items() if length == max_sim]
                            if max_sim == 0 or len(disambiguated) == 0:
                                ambiguity = {f"{key}: {self.full_paths[key][i]}": prefix for (key, i), prefix in
                                             longest.items()}
                                self.logger.info(
                                    f"Matching {wh} {file} to {wha} is ambiguous. Disambiguate using keys:\n{pretty_dict(ambiguity)}")
                                continue
                            elif len(disambiguated) > 1:
                                ambiguity = {f"{key}: {self.full_paths[key][i]}": longest[(key, i)] for key, i in
                                             disambiguated}
                                self.logger.info(
                                    f"Matching {wh} {file} to {wha} is ambiguous, even after comparing keys. Disambiguate using keys:\n{pretty_dict(ambiguity)}")
                                continue
                            match_id = disambiguated[0]
                            msg = " after disambiguation by keys."
                        else:
                            match_id = list(longest.keys())[0]
                            msg = "."
                        row[wha] = match_id
                        match_ix = match_id
                        match_row = get_row(match_ix)
                        match_row[wh] = id
                        update_row(match_ix, match_row)
                        #res_ix.add(match_ix)
                        match_file = self.files[match_id[0]][match_id[1]]
                        self.logger.debug(f"Matched {wh} {file} to {wha} {match_file} based on the prefix {longest[match_id]}{msg}")

                update_row(ix, row)
                #res_ix.add(ix)

        if ids is None:
            ids = list(self._iterids(keys))
        ids = [i for i in ids if any(i in lists[w] for w in what)]
        res_ix = self.ids2idx(ids, pandas_index=True)
        return self._matches.loc[res_ix, what].sort_index()


    def metadata(self, keys=None, from_tsv=False):
        """ Retrieve concatenated metadata from parsed scores or TSV files.

        Parameters
        ----------
        keys
        from_tsv : :obj:`bool`, optional
            If set to True, you'll get a concatenation of all parsed TSV files that have been
            recognized as containing metadata. If you want to specifically retrieve files called
            'metadata.tsv' only, use :py:meth:`metadata_tsv`.

        Returns
        -------
        :obj:`pandas.DataFrame`
        """
        if from_tsv:
            tsv_dfs = self.get_tsvs(keys=keys, metadata=True)['metadata']
            if len(tsv_dfs) > 0:
                df = pd.concat(tsv_dfs.values(), keys=tsv_dfs.keys())
                df.index.names = ['key', 'i', 'metadata_id']

            # this option does not give control over keys:
            tsv_dfs = self._metadata
        else:
            parsed_ids = list(self._iterids(keys, only_parsed_mscx=True))
            if len(parsed_ids) > 0:
                ids, meta_series = zip(*[(id, metadata2series(self._parsed_mscx[id].mscx.metadata)) for id in parsed_ids])
                ix = pd.MultiIndex.from_tuples(ids, names=['key', 'i'])
                df = pd.DataFrame(meta_series, index=ix)
                df['rel_paths'] = [self.rel_paths[k][i] for k, i in ids]
                df['fnames'] = [self.fnames[k][i] for k, i in ids]
            else:
                df = pd.DataFrame()

        if len(df) > 0:
            return column_order(df, METADATA_COLUMN_ORDER).sort_index()
        else:
            what = 'metadata TSVs' if from_tsv else 'scores'
            self.logger.info(f"No {what} have been parsed so far.")
            return df

    def score_metadata(self, view_name: Optional[str] = None) -> pd.DataFrame:
        metadata_dfs = {corpus_name: corpus.score_metadata(view_name=view_name) for corpus_name, corpus in self.iter_corpora(view_name=view_name)}
        metadata = pd.concat(metadata_dfs.values(), keys=metadata_dfs.keys(), names=['corpus', 'fname'])
        return metadata

    def metadata_tsv(self, view_name: Optional[str] = None) -> pd.DataFrame:
        metadata_dfs = {corpus_name: corpus.metadata_tsv
                        for corpus_name, corpus in self.iter_corpora(view_name=view_name)
                        if corpus.metadata_tsv is not None
                        }
        metadata = pd.concat(metadata_dfs.values(), keys=metadata_dfs.keys(), names=['corpus', 'fname'])
        return metadata




    def parse(self, view_name=None, level=None, parallel=True, only_new=True, labels_cfg={}, cols={}, infer_types=None, **kwargs):
        """ Shorthand for executing parse_mscx and parse_tsv at a time.
        Args:
            view_name:
        """
        self.parse_mscx(level=level, parallel=parallel, only_new=only_new, labels_cfg=labels_cfg, view_name=view_name)
        self.parse_tsv(view_name=view_name, level=level, cols=cols, infer_types=infer_types, only_new=only_new, **kwargs)



    def parse_mscx(self,
                   level: str = None,
                   parallel: bool = True,
                   only_new: bool = True,
                   labels_cfg: dict = {},
                   view_name:str = None):
        """ Parse MuseScore 3 files (MSCX or MSCZ) and store the resulting read-only Score objects. If they need
        to be writeable, e.g. for removing or adding labels, pass ``parallel=False`` which takes longer but prevents
        having to re-parse at a later point.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
            For which key(s) to parse all MSCX files.
        ids : :obj:`~collections.abc.Collection`
            To parse only particular files, pass their IDs. ``keys`` and ``fexts`` are ignored in this case.
        level : {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}, optional
            Pass a level name for which (and above which) you want to see log records.
        parallel : :obj:`bool`, optional
            Defaults to True, meaning that all CPU cores are used simultaneously to speed up the parsing. It implies
            that the resulting Score objects are in read-only mode and that you might not be able to use the computer
            during parsing. Set to False to parse one score after the other, which uses more memory but will allow
            making changes to the scores.
        only_new : :obj:`bool`, optional
            By default, score which already have been parsed, are not parsed again. Pass False to parse them, too.

        Returns
        -------
        None

        """
        if level is not None:
            self.change_logger_cfg(level=level)
        for corpus_name, corpus in self.iter_corpora(view_name=view_name):
            corpus.parse_mscx(level=level, parallel=parallel, only_new=only_new, labels_cfg=labels_cfg)


    def get_files_of_types(self, file_type: Union[str, Collection[str]],
                     view_name: Optional[str] = None,
                     parsed: bool = True,
                     unparsed: bool = True,
                     choose: Literal['all', 'auto', 'ask'] = 'all',
                     flat: bool = False) -> Dict[CorpusFnameTuple, Union[FileDict, FileList]]:
        """

        Args:
            file_type:
            view_name:
            parsed:
            unparsed:
            choose:
            flat:

        Returns:
            {(corpus_name, fname) -> {type -> [:obj:`File`]}} dict if flat=False (default).
            {(corpus_name, fname) -> [:obj:`File`]} dict if flat=True.
        """
        result = {}
        for corpus_name, corpus in self.iter_corpora(view_name=view_name):
            selected = corpus.get_files(facets=file_type,
                                        view_name=view_name,
                                        parsed=parsed,
                                        unparsed=unparsed,
                                        choose=choose,
                                        flat=flat)
            selected = {(corpus_name, fname): files for fname, files in selected.items()}
            result.update(selected)
        return result

    def parse_tsv(self, view_name=None, level=None, cols={}, infer_types=None, only_new=True, **kwargs):
        """ Parse TSV files (or other value-separated files such as CSV) to be able to do something with them.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
            Key(s) for which to parse all non-MSCX files.  By default, all keys are selected.
        ids : :obj:`~collections.abc.Collection`
            To parse only particular files, pass there IDs. ``keys`` and ``fexts`` are ignored in this case.
        fexts :  :obj:`str` or :obj:`~collections.abc.Collection`, optional
            If you want to parse only files with one or several particular file extension(s), pass the extension(s)
        cols : :obj:`dict`, optional
            By default, if a column called ``'label'`` is found, the TSV is treated as an annotation table and turned into
            an Annotations object. Pass one or several column name(s) to treat *them* as label columns instead. If you
            pass ``{}`` or no label column is found, the TSV is parsed as a "normal" table, i.e. a DataFrame.
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

        Args:
            only_new:
            view_name:
        """
        if level is not None:
            self.change_logger_cfg(level=level)
        for corpus_name, corpus in self.iter_corpora(view_name=view_name):
            corpus.parse_tsv(view_name=view_name, cols=cols, infer_types=infer_types, only_new=only_new, **kwargs)


    def _parse_tsv_from_git_revision(self, tsv_id, revision_specifier):
        """ Takes the ID of an annotation table, and parses the same file's previous version at ``revision_specifier``.

        Parameters
        ----------
        tsv_id
            ID of the TSV file containing an annotation table, for which to parse a previous version.
        revision_specifier : :obj:`str`
            String used by git.Repo.commit() to find the desired git revision.
            Can be a long or short SHA, git tag, branch name, or relative specifier such as 'HEAD~1'.

        Returns
        -------
        ID
            (key, i) of the newly added annotation table.
        """
        key, i = tsv_id
        corpus_path = self.corpus_paths[key]
        try:
            repo = Repo(corpus_path, search_parent_directories=True)
        except InvalidGitRepositoryError:
            self.logger.error(f"{corpus_path} seems not to be (part of) a git repository.")
            return
        try:
            git_repo = repo.remote("origin").url
        except ValueError:
            git_repo = os.path.basename()
        try:
            commit = repo.commit(revision_specifier)
            commit_sha = commit.hexsha
            short_sha = commit_sha[:7]
            commit_info = f"{short_sha} with message '{commit.message}'"
        except BadName:
            self.logger.error(f"{revision_specifier} does not resolve to a commit for repo {git_repo}.")
            return
        tsv_type = self._tsv_types[tsv_id]
        tsv_path = self.full_paths[key][i]
        rel_path = os.path.relpath(tsv_path, corpus_path)
        new_directory = os.path.join(corpus_path, short_sha)
        new_path = os.path.join(new_directory, self.files[key][i])
        if new_path in self.full_paths[key]:
            existing_i = self.full_paths[key].index(new_path)
            existing_tsv_type = self._tsv_types[(key, existing_i)]
            if tsv_type == existing_tsv_type:
                self.logger.error(f"Had already loaded a {tsv_type} table for commit {commit_info} of repo {git_repo}.")
                return
        if not tsv_type in ('labels', 'expanded'):
            raise NotImplementedError(f"Currently, only annotations are to be loaded from a git revision but {rel_path} is a {tsv_type}.")
        try:
            targetfile = commit.tree / rel_path
        except KeyError:
            # if the file was not found, try and see if at the time of the git revision the folder was still called 'harmonies'
            if tsv_type == 'expanded':
                folder, tsv_name = os.path.split(rel_path)
                if folder != 'harmonies':
                    old_rel_path = os.path.join('harmonies', tsv_name)
                    try:
                        targetfile = commit.tree / old_rel_path
                        self.logger.debug(f"{rel_path} did not exist at commit {commit_info}, using {old_rel_path} instead.")
                        rel_path = old_rel_path
                    except KeyError:
                        self.logger.error(f"Neither {rel_path} nor its older version {old_rel_path} existed at commit {commit_info}.")
                        return
            else:
                self.logger.error(f"{rel_path} did not exist at commit {commit_info}.")
                return
        self.logger.info(f"Successfully loaded {rel_path} from {commit_info}.")
        try:
            with io.BytesIO(targetfile.data_stream.read()) as f:
                df = load_tsv(f)
        except Exception:
            self.logger.error(f"Parsing {rel_path} @ commit {commit_info} failed with the following error:\n{sys.exc_info()[1]}")
            return
        new_id = self._handle_path(new_path, key, skip_checks=True)
        self._parsed_tsv[new_id] = df
        self._dataframes[tsv_type][new_id] = df
        self._tsv_types[new_id] = tsv_type
        logger_cfg = dict(self.logger_cfg)
        logger_cfg['name'] = self.logger_names[(key, i)]
        if tsv_id in self._annotations:
            anno_obj = self._annotations[tsv_id] # get Annotation object's settings from the existing one
            cols = anno_obj.cols
            infer_types = anno_obj.regex_dict
        else:
            cols = dict(label='label')
            infer_types = None
        self._annotations[new_id] = Annotations(df=df, cols=cols, infer_types=infer_types,
                                            logger_cfg=logger_cfg)
        self.logger.debug(
            f"{rel_path} successfully parsed from commit {short_sha}.")
        return new_id


    def pieces(self, parsed_only=False):
        pieces_dfs = [self[k].pieces(parsed_only=parsed_only) for k in self.keys()]
        result = pd.concat(pieces_dfs, keys=self.keys())
        result.index.names = ['key', 'metadata_row']
        return result

    def output_dataframes(self, root_dir=None, notes_folder=None, notes_suffix='',
                          rests_folder=None, rests_suffix='',
                          notes_and_rests_folder=None, notes_and_rests_suffix='',
                          measures_folder=None, measures_suffix='',
                          events_folder=None, events_suffix='',
                          labels_folder=None, labels_suffix='',
                          chords_folder=None, chords_suffix='',
                          expanded_folder=None, expanded_suffix='',
                          cadences_folder=None, cadences_suffix='',
                          form_labels_folder=None, form_labels_suffix='',
                          metadata_path=None, markdown=True,
                          simulate=None, unfold=False, quarterbeats=False,
                          silence_label_warnings=False):
        """ Store score information as TSV files.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
            Key(s) under which score files are stored. By default, all keys are selected.
        root_dir : :obj:`str`, optional
            Defaults to None, meaning that the original root directory is used that was added to the Parse object.
            Otherwise, pass a directory to rebuild the original directory structure. For ``_folder`` parameters describing
            absolute paths, ``root_dir`` is ignored.
        notes_folder, rests_folder, notes_and_rests_folder, measures_folder, events_folder, labels_folder, chords_folder, expanded_folder, cadences_folder, form_labels_folder : str, optional
            Specify directory where to store the corresponding TSV files.
        notes_suffix, rests_suffix, notes_and_rests_suffix, measures_suffix, events_suffix, labels_suffix, chords_suffix, expanded_suffix, cadences_suffix, form_labels_suffix : str, optional
            Optionally specify suffixes appended to the TSVs' file names.
        metadata_path : str, optional
            Where to store an overview file with the MuseScore files' metadata.
            If no file name is specified, the file will be named ``metadata.tsv``.
        markdown : bool, optional
            By default, when ``metadata_path`` is specified, a markdown file called ``README.rst.md`` containing
            the columns [file_name, measures, labels, standard, annotators, reviewers] is created. If it exists already,
            this table will be appended or overwritten after the heading ``# Overview``.
        simulate : bool, optional
            Defaults to ``None``. Pass a value to set the object attribute :py:attr:`~ms3.parse.Parse.simulate`.
        unfold : bool, optional
            By default, repetitions are not unfolded. Pass True to duplicate values so that they correspond to a full
            playthrough, including correct positioning of first and second endings.
        quarterbeats : bool, optional
            By default, no ``quarterbeats`` column is added with distances from the piece's beginning measured in quarter notes.
            Pass True to add the columns ``quarterbeats`` and ``duration_qb``. If a score has first and second endings,
            the behaviour depends on ``unfold``: If False, repetitions are not unfolded and only last endings are included in the
            continuous count. If repetitions are being unfolded, all endings are taken into account.

        Returns
        -------

        """
        if simulate is None:
            simulate = self.simulate
        else:
            self.simulate = simulate
        for corpus_name, corpus in self:
            corpus.output_dataframes(root_dir=root_dir,
                        notes_folder=notes_folder, notes_suffix=notes_suffix,
                        notes_and_rests_folder=notes_and_rests_folder, notes_and_rests_suffix=notes_and_rests_suffix,
                        labels_folder=labels_folder, labels_suffix=labels_suffix,
                        measures_folder=measures_folder, measures_suffix=measures_suffix,
                        rests_folder=rests_folder, rests_suffix=rests_suffix,
                        events_folder=events_folder, events_suffix=events_suffix,
                        chords_folder=chords_folder, chords_suffix=chords_suffix,
                        expanded_folder=expanded_folder, expanded_suffix=expanded_suffix,
                        cadences_folder=cadences_folder, cadences_suffix=cadences_suffix,
                        form_labels_folder=form_labels_folder, form_labels_suffix=form_labels_suffix,
                        metadata_path=metadata_path,
                        markdown=markdown,
                        simulate=simulate,
                        unfold=unfold,
                        quarterbeats=quarterbeats,
                        silence_label_warnings=silence_label_warnings,
                        )


    def output_mscx(self, keys=None, ids=None, root_dir=None, folder='.', suffix='', overwrite=False, simulate=False):
        """ Stores the parsed MuseScore files in their current state, e.g. after detaching or attaching annotations.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
            Key(s) for which to count file extensions.  By default, all keys are selected.
        ids : :obj:`~collections.abc.Collection`
            If you pass a collection of IDs, ``keys`` is ignored and only the selected extensions are counted.
        root_dir : :obj:`str`, optional
            Defaults to None, meaning that the original root directory is used that was added to the Parse object.
            Otherwise, pass a directory to rebuild the original substructure. If ``folder`` is an absolute path,
            ``root_dir`` is ignored.
        folder : :obj:`str`
            Where to store the file. Can be relative to ``root_dir`` or absolute, in which case ``root_dir`` is ignored.
            If ``folder`` is relative, the behaviour depends on whether it starts with a dot ``.`` or not: If it does,
            the folder is created at every end point of the relative tree structure under ``root_dir``. If it doesn't,
            it is created only once, relative to ``root_dir``, and the relative tree structure is build below.
        suffix : :obj:`str`, optional
            Suffix to append to the original file name.
        overwrite : :obj:`bool`, optional
            Pass True to overwrite existing files.
        simulate : :obj:`bool`, optional
            Set to True if no files are to be written.

        Returns
        -------

        """
        if ids is None:
            ids = [id for id in self._iterids(keys) if id in self._parsed_mscx]
        paths = []
        for key, i in ids:
            new_path = self._output_mscx(key=key, i=i, folder=folder, suffix=suffix, root_dir=root_dir, overwrite=overwrite, simulate=simulate)
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

    def _iter_subdir_selectors(self, keys=None, ids=None):
        """ Iterate through the specified ids grouped by subdirs.

        Yields
        ------
        :obj:`tuple`
            (key: str, subdir: str, ixs: list) tuples. IDs can be created by combining key with each i in ixs.
            The yielded ``ixs`` are typically used as parameter for ``.utils.iter_selection``.

        """
        grouped_ids = self._make_grouped_ids(keys, ids)
        for k, ixs in grouped_ids.items():
            subdirs = self.subdirs[k]
            for subdir in sorted(set(iter_selection(subdirs, ixs))):
                yield k, subdir, [i for i in ixs if subdirs[i] == subdir]




    def _parse(self, key, i, logger_cfg={}, labels_cfg={}, read_only=False):
        """Performs a single parse and returns the resulting Score object or None."""
        path = self.full_paths[key][i]
        file = self.files[key][i]
        self.logger.debug(f"Attempting to parse {file}")
        try:
            logger_cfg['name'] = self.logger_names[(key, i)]
            score = Score(path, read_only=read_only, labels_cfg=labels_cfg, logger_cfg=logger_cfg, ms=self.ms)
            if score is None:
                self.logger.debug(f"Encountered errors when parsing {file}")
            else:
                self.logger.debug(f"Successfully parsed {file}")
            return score
        except (KeyboardInterrupt, SystemExit):
            self.logger.info("Process aborted.")
            raise
        except:
            self.logger.error(f"Unable to parse {path} due to the following exception:\n" + traceback.format_exc())
            return None


    def _score_ids(self, keys=None, score_extensions=None, native=True, convertible=True, opposite=False):
        """ Return IDs of all detected scores with particular file extensions, or all others if ``opposite==True``.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`collections.abc.Iterable`, optional
            Only get IDs for particular keys.
        score_extensions : :obj:`collections.abc.Collection`, optional
            Get IDs for files with the given extensions (each starting with a dot). If this parameter is defined,
            ``native```and ``convertible`` are being ignored.
        native : :obj:`bool`, optional
            If ``score_extensions`` is not set, ``native=True`` selects all scores that ms3 can parse without using
            a MuseScore 3 executable.
        convertible : :obj:`bool`, optional
            If ``score_extensions`` is not set, ``convertible=True`` selects all scores that ms3 can parse as long as
            a MuseScore 3 executable is defined.
        opposite : :obj:`bool`, optional
            Set to True if you want to get the IDs of all the scores that do NOT have the specified extensions.

        Returns
        -------
        :obj:`list`
            A list of IDs.

        """
        if score_extensions is None:
            score_extensions = []
            if native:
                score_extensions.extend(Score.native_formats)
            if convertible:
                score_extensions.extend(Score.convertible_formats)
        if opposite:
            return [(k, i) for k, i in self._iterids(keys) if self.fexts[k][i][1:].lower() not in score_extensions]
        return [(k, i) for k, i in self._iterids(keys) if self.fexts[k][i][1:].lower() in score_extensions]



    def _output_mscx(self, key, i, folder, suffix='', root_dir=None, overwrite=False, simulate=False):
        """ Creates a MuseScore 3 file from the Score object at the given ID (key, i).

        Parameters
        ----------
        key, i : (:obj:`str`, :obj:`int`)
            ID from which to construct the new path and filename.
        root_dir : :obj:`str`, optional
            Defaults to None, meaning that the original root directory is used that was added to the Parse object.
            Otherwise, pass a directory to rebuild the original substructure. If ``folder`` is an absolute path,
            ``root_dir`` is ignored.
        folder : :obj:`str`
            Where to store the file. Can be relative to ``root_dir`` or absolute, in which case ``root_dir`` is ignored.
            If ``folder`` is relative, the behaviour depends on whether it starts with a dot ``.`` or not: If it does,
            the folder is created at every end point of the relative tree structure under ``root_dir``. If it doesn't,
            it is created only once, relative to ``root_dir``, and the relative tree structure is build below.
        suffix : :obj:`str`, optional
            Suffix to append to the original file name.
        overwrite : :obj:`bool`, optional
            Pass True to overwrite existing files.
        simulate : :obj:`bool`, optional
            Set to True if no files are to be written.

        Returns
        -------
        :obj:`str`
            Path of the stored file.

        """

        id = (key, i)
        logger = self.id_logger(id)
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
            self._parsed_mscx[id].output_mscx(file_path)
            logger.debug(f"Score written to {file_path}.")

        return file_path


    def _store_tsv(self, df, key, i, folder, suffix='', root_dir=None, what='DataFrame', simulate=False):
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

        Returns
        -------
        :obj:`str`
            Path of the stored file.

        """
        tsv_logger = self.id_logger((key, i))

        if df is None:
            tsv_logger.debug(f"No DataFrame for {what}.")
            return
        path = self._calculate_path(key=key, i=i, root_dir=root_dir, folder=folder)
        if path is None:
            return

        fname = self.fnames[key][i] + suffix + ".tsv"
        file_path = os.path.join(path, fname)
        if simulate:
            tsv_logger.debug(f"Would have written {what} to {file_path}.")
        else:
            tsv_logger.debug(f"Writing {what} to {file_path}.")
            write_tsv(df, file_path, logger=tsv_logger)
        return file_path



    def _treat_key_param(self, keys):
        if keys is None:
            keys = list(self.full_paths.keys())
        elif isinstance(keys, str):
            keys = [keys]
        return [k for k in sorted(set(keys)) if k in self.files]


    def _treat_harmony_layer_param(self, harmony_layer):
        if harmony_layer is None:
            return None
        all_types = {str(k): k for k in self.count_labels().keys()}
        if isinstance(harmony_layer, int) or isinstance(harmony_layer, str):
            harmony_layer = [harmony_layer]
        lt = [str(t) for t in harmony_layer]
        def matches_any_type(user_input):
            return any(True for t in all_types if user_input in t)
        def get_matches(user_input):
            return [t for t in all_types if user_input in t]

        not_found = [t for t in lt if not matches_any_type(t)]
        if len(not_found) > 0:
            plural = len(not_found) > 1
            plural_s = 's' if plural else ''
            self.logger.warning(
                f"No labels found with {'these' if plural else 'this'} label{plural_s} harmony_layer{plural_s}: {', '.join(not_found)}")
        return [all_types[t] for user_input in lt for t in get_matches(user_input)]

    def update_metadata(self, allow_suffix=False):
        """Uses all parsed metadata TSVs to update the information in the corresponding parsed MSCX files and returns
        the IDs of those that have been changed.

        Parameters
        ----------
        allow_suffix : :obj:`bool`, optional
            If set to True, this would also update the metadata for currently parsed MuseScore files
            corresponding to the columns 'rel_paths' and 'fnames' + [ANY SUFFIX]. For example,
            the row ('MS3', 'bwv846') would also update the metadata of 'MS3/bwv846_reviewed.mscx'.

        Returns
        -------
        :obj:`list`
            IDs of the parsed MuseScore files whose metadata has been updated.
        """
        metadata_dfs = self.metadata_tsv()
        if len(metadata_dfs) > 0:
            metadata = pd.concat(metadata_dfs.values(), keys=metadata_dfs.keys())
        else:
            metadata = self._metadata
        if len(metadata) == 0:
            self.logger.debug("No parsed metadata found.")
            return
        old = metadata
        if old.index.names != ['rel_paths', 'fnames']:
            try:
                old = old.set_index(['rel_paths', 'fnames'])
            except KeyError:
                self.logger.warning(f"Parsed metadata do not contain the columns 'rel_paths' and 'fnames' "
                                    f"needed to match information on identical files.")
                return []
        new = self.metadata(from_tsv=False).set_index(['rel_paths', 'fnames'])
        excluded_cols = ['ambitus', 'annotated_key', 'KeySig', 'label_count', 'last_mc', 'last_mn', 'musescore',
                         'TimeSig', 'length_qb', 'length_qb_unfolded', 'all_notes_qb', 'n_onsets', 'n_onset_positions']
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
                matches = self.fname2ids(fname=fname, rel_path=rel_path, allow_suffix=allow_suffix)
                match_ids = [id for id in matches.keys() if id in self._parsed_mscx]
                n_files_to_update = len(match_ids)
                if n_files_to_update == 0:
                    self.logger.debug(
                        f"rel_path={rel_path}, fname={fname} does not correspond to a currently parsed MuseScore file.")
                    continue
                for id in match_ids:
                    for name, val in new_dict.items():
                        self._parsed_mscx[id].mscx.parsed.metatags[name] = val
                    self._parsed_mscx[id].mscx.parsed.update_metadata()
                    self.id_logger(id).debug(f"Updated with {new_dict}")
                    ids.append(id)

            self.logger.info(f"{l} files updated.")
        else:
            self.logger.info("Nothing to update.")
        return ids


    def __getstate__(self):
        """ Override the method of superclass """
        return self.__dict__

    def iter_pieces(self) -> Tuple[CorpusFnameTuple, Piece]:
        for corpus_name, corpus in self:
            for fname, piece in corpus:
                yield (corpus_name, fname), piece

    def _get_corpus(self, name):
        assert name in self.corpus_objects, f"Don't have a corpus called '{name}', only {list(self.corpus_objects.keys())}"
        return self.corpus_objects[name]


    def __getitem__(self, item) -> Corpus:
        if isinstance(item, str):
            return self._get_corpus(item)
        elif isinstance(item, tuple):
            if len(item) == 1:
                return self._get_corpus(item[0])
            if len(item) == 2:
                corpus_name, fname_or_ix = item
                return self._get_corpus(corpus_name)[fname_or_ix]
            corpus_name, *remainder = item
            return self._get_corpus(corpus_name)[tuple(remainder)]

    def __iter__(self) -> Generator[Tuple[str, Corpus], None, None]:
        """  Iterate through all (corpus_name, Corpus) tuples, regardless of any Views.

        Yields: (corpus_name, Corpus) tuples
        """
        yield from self.corpus_objects.items()


    def __repr__(self):
        return self.info(return_str=True)

    def _get_unambiguous_fnames_from_ids(self, score_ids, key):

        file_info = [self.id2file_info[id] for id in score_ids]
        score_names = [F.fname for F in file_info]
        score_name_set = set(score_names)
        if len(score_names) == len(score_name_set):
            return dict(zip(score_names, score_ids))
        more_than_one = {name: [] for name, cnt in Counter(score_names).items() if cnt > 1}
        result = {} # fname -> score_id
        for F in file_info:
            if F.fname in more_than_one:
                more_than_one[F.fname].append(F)
            else:
                result[F.fname] = F.id
        for name, files in more_than_one.items():
            choice_between_n = len(files)
            df = pd.DataFrame.from_dict({F.id: dict(subdir=F.subdir, fext=F.fext, subdir_len=len(F.subdir)) for F in files}, orient='index')
            self.logger.debug(f"Trying to disambiguate between these {choice_between_n} with the same fname '{name}':\n{df}")
            shortest_subdir_length = df.subdir_len.min()
            shortest_length_selector = (df.subdir_len == shortest_subdir_length)
            n_have_shortest_length = shortest_length_selector.sum()
            # checking if the shortest path contains only 1 file and pick that
            if n_have_shortest_length == 1:
                id = df.subdir_len.idxmin()
                picked = df.loc[id]
                self.logger.info(f"In order to pick one from the {choice_between_n} scores with fname '{name}', the one with the shortest subdir '{picked.subdir}' was selected.")
                result[name] = id
                continue
            # otherwise, check if there is only a single MSCX or otherwise MSCZ file and pick that
            fexts = df.fext.value_counts()
            if '.mscx' in fexts:
                if fexts['.mscx'] == 1:
                    picked = df[df.fext == '.mscx'].iloc[0]
                    id = picked.name
                    self.logger.info(f"In order to pick one from the {choice_between_n} scores with fname '{name}', the one contained in '{picked.subdir}' was selected because it is the only "
                                     f"one in MSCX format.")
                    result[name] = id
                    continue
            elif '.mscz' in fexts and fexts['.mscz'] == 1:
                picked = df[df.fext == '.mscz'].iloc[0]
                id = picked.name
                self.logger.info(
                    f"In order to pick one from the {choice_between_n} scores with fname '{name}', the one contained in '{picked.subdir}' was selected because it is the only "
                    f"one in MuseScore format.")
                result[name] = id
                continue
            # otherwise, check if the shortest path contains only a single MSCX or MSCZ file as a last resort
            if n_have_shortest_length < choice_between_n:
                df = df[shortest_length_selector]
                self.logger.debug(f"Picking those from the shortest subdir has reduced the choice to {n_have_shortest_length}:\n{df}.")
            else:
                self.logger.warning(f"Unable to pick one of the available scores for fname '{name}', it will be disregarded until disambiguated:\n{df}")
                continue
            if '.mscx' in df.fext.values and fexts['.mscx'] == 1:
                pick_ext = '.mscx'
            elif '.mscz' in df.fext.values and fexts['.mscz'] == 1:
                pick_ext = '.mscz'
            else:
                self.logger.warning(f"Unable to pick one of the available scores for fname '{name}', it will be disregarded until disambiguated:\n{df}")
                continue
            picked = df[df.fext == pick_ext].iloc[0]
            id = picked.name
            self.logger.info(
                f"In order to pick one from the {choice_between_n} scores with fname '{name}', the '{pick_ext}' one contained in '{picked.subdir}' was selected because it is the only "
                f"one in that format contained in the shortest subdir.")
            result[name] = id
        return result


    def get_piece(self, corpus_name: str, fname: str) -> Piece:
        """Returns an existing :obj:`Piece` object."""
        assert corpus_name in self.corpus_objects, f"'{corpus_name}' is not an existing corpus. Choose from {list(self.corpus_objects.keys())}"
        return self.corpus_objects[corpus_name].get_piece(fname)
########################################################################################################################
########################################################################################################################
################################################# End of Parse() ########################################################
########################################################################################################################
########################################################################################################################


########################################################################################################################
########################################################################################################################
################################################# End of View() ########################################################
########################################################################################################################
########################################################################################################################

class PieceView(View):

    def __init__(self,
                 view: View,
                 fname: str):
        self.view = view  # parent View object
        self.p = view.p
        self.key = view.key
        self.fname = fname
        logger_cfg = self.p.logger_cfg
        logger_cfg['name'] = f"{self.view.logger.name}.{self.fname}"
        super(Parse, self).__init__(subclass='Piece', logger_cfg=logger_cfg)  # initialize loggers
        matches = view.detect_ids_by_fname(parsed_only=True, names=[fname])
        if len(matches) != 1:
            raise ValueError(f"{len(matches)} fnames match {fname} for key {self.key}")
        self.matches = matches[fname]
        self.score_available = 'scores' in self.matches
        self.measures_available = self.score_available or 'measures' in self.matches


    def info(self, return_str=False):
        info = f"View on {self.key} -> {self.fname}\n"
        info += "-" * len(info) + "\n\n"
        if self.score_available:
            plural = "s" if len(self.matches['scores']) > 1 else ""
            info += f"Parsed score{plural} available."
        else:
            info += "No parsed scores available."
        info += "\n\n"
        for typ, matched_files in self.matches.items():
            info += f"{typ}\n{'-' * len(typ)}\n"
            if len(matched_files) > 1:
                distinguish = make_distinguishing_strings(matched_files)
                matches = dict(zip(distinguish, matched_files))
                paths = {k: matches[k].full_path for k in sorted(matches, key=lambda s: len(s))}
                info += pretty_dict(paths)
            else:
                info += matched_files[0].full_path
            info += "\n\n"
        if return_str:
            return info
        print(info)


    @lru_cache()
    def get_dataframe(self, what: Literal['measures', 'notes', 'rests', 'labels', 'expanded', 'events', 'chords', 'metadata', 'form_labels'],
                      unfold: bool = False,
                      quarterbeats: bool = False,
                      interval_index: bool = False,
                      disambiguation: str = 'auto',
                      prefer_score: bool = True,
                      return_file_info: bool = False) -> pd.DataFrame:
        """ Retrieves one DataFrame for the piece.

        Args:
            what: What kind of DataFrame to retrieve.
            unfold: Pass True to unfold repeats.
            quarterbeats:
            interval_index:
            disambiguation: In case several DataFrames are available in :attr:`.matches`, pass its disambiguation string.
            prefer_score: By default, data from parsed scores is preferred to that from parsed TSVs. Pass False to prefer TSVs.
            return_file_info: Set to True if the method should also return a :obj:`namedtuple` with information on the DataFrame
                being returned. It comes with the fields "id", "full_path", "suffix", "fext", "subdir", "i_str" where the
                latter is the ID's second component as a string.

        Returns:
            The requested DataFrame if available and, if ``return_file_info`` is set to True, a namedtuple with information about its provenance.

        Raises:
            FileNotFoundError: If no DataFrame of the requested type is available
        """
        available = list(self.p._dataframes.keys())
        if what not in available:
            raise ValueError(f"what='{what}' is an invalid argument. Pass one of {available}.")
        if self.score_available and (prefer_score or what not in self.matches):
            file_info = disambiguate(self.matches['scores'], disambiguation=disambiguation)
            score = self.p[file_info.id]
            df = score.mscx.__getattribute__(what)()
        elif what in self.matches:
            file_info = disambiguate(self.matches[what], disambiguation=disambiguation)
            df = self.p[file_info.id]
        else:
            raise FileNotFoundError(f"No {what} available for {self.key} -> {self.fname}")
        if any((unfold, quarterbeats, interval_index)):
            measures = self.get_dataframe('measures', prefer_score=prefer_score)
            df = dfs2quarterbeats([df], measures, unfold=unfold, quarterbeats=quarterbeats,
                                   interval_index=interval_index, logger=self.logger)[0]
        if return_file_info:
            return df, file_info
        return df