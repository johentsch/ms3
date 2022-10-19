from typing import Literal, Collection, Dict, List, Union, Tuple, Iterator

import io
import sys, os, re
from functools import lru_cache
import json
import pathos.multiprocessing as mp
from collections import Counter, defaultdict, namedtuple

import pandas as pd
import numpy as np
from git import Repo, InvalidGitRepositoryError
from gitdb.exc import BadName

from .annotations import Annotations
from .logger import LoggedClass, get_logger, get_log_capture_handler, temporarily_suppress_warnings, function_logger
from .piece import Piece
from .score import Score
from .utils import File, column_order, get_musescore, get_path_component, group_id_tuples, infer_tsv_type, \
    iter_selection, join_tsvs, load_tsv, make_continuous_offset_series, \
    make_id_tuples, make_playthrough2mc, METADATA_COLUMN_ORDER, metadata2series, parse_ignored_warnings_file, path2type, \
    pretty_dict, resolve_dir, \
    update_labels_cfg, write_metadata, write_tsv, path2parent_corpus
from .transformations import add_weighted_grace_durations, dfs2quarterbeats
from .view import DefaultView, View


class Corpus(LoggedClass):
    """
    Collection of scores and TSV files that can be matched to each other based on their file names.
    """

    def __init__(self, directory, view: View=None, simulate=False, labels_cfg={}, logger_cfg={}, ms=None, level=None, **kwargs):
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
        directory = resolve_dir(directory)
        assert os.path.isdir(directory), f"{directory} is not an existing directory."
        self.corpus_path = directory
        """obj:`dict`
        {key -> path} dictionary with each corpus's base directory.
        """
        logger_cfg = dict(logger_cfg)
        if 'name' not in logger_cfg or logger_cfg['name'] is None or logger_cfg['name'] == '':
            logger_cfg['name'] = 'ms3.' + os.path.basename(directory).strip(r'\/').replace('.', '')
        if level is not None:
            logger_cfg['level'] = level
        if 'level' not in logger_cfg or (logger_cfg['level'] is None):
            logger_cfg['level'] = 'w'
        super().__init__(subclass='Corpus', logger_cfg=logger_cfg)
        self.simulate=simulate

        self.files: list = []
        """
        ``[File]`` list of :obj:`File` data objects containing information on the file location
        etc. for all detected files. 
        """

        self.files_df: pd.DataFrame = pd.DataFrame()
        """ DataFrame containing information on all detected files.
        """

        self._views: dict = {}
        self._views[None] = DefaultView('current') if view is None else view
        self._views['default'] = DefaultView('default')
        self._views['all'] = View('all')

        self._ms = get_musescore(ms, logger=self.logger)
        """:obj:`str`
        Path or command of the local MuseScore 3 installation if specified by the user."""

        self._ignored_warnings = defaultdict(list)
        """:obj:`collections.defaultdict`
        {'logger_name' -> [(message_id), ...]} This dictionary stores the warnings to be ignored
        upon loading them from an IGNORED_WARNINGS file.
        """

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


        self._pieces: Dict[str, Piece] = {}
        """{fname -> :class:`Piece`} The objects holding together information for the same piece
        from various files. 
        """

        self.metadata_tsv: pd.DataFrame = None
        """The parsed 'metadata.tsv' file for the corpus."""

        self.metadata_ix: int = None
        """The index of the 'metadata.tsv' file for the corpus."""

        self.ix2fname: dict = {}
        """{ix -> fname} dict for associating files with the piece they have been matched to.
        None for indices that could not be matched, e.g. metadata.
        """

        self.parsed_files: dict = {}
        """{ix -> Score or DataFrame}"""

        self.score_fnames: list = []
        """Sorted list of unique file names of all detected scores"""

        self.detect_parseable_files()
        self.collect_fnames_from_scores()
        self.find_and_load_metadata()
        self.create_pieces()
        #self.reset_piece_selection()
        # self.look_for_ignored_warnings(directory, key=top_level)
        # if len(mscx_files) > 0:
        #     self.logger.warning(f"The following MSCX files are lying on the top level '{top_level}' and have not been registered (call with recursive=False to add them):"
        #                         f"\n{mscx_files}", extra={"message_id": (7, top_level)})
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
        for fname, piece in self:
            if current is not None and current.check_token('fname', fname):
                piece.set_view(current)
            for view_name, view in views.items():
                if view.check_token('fname', fname):
                    piece.set_view(**{view_name:view})

    def get_view(self, view_name: str = None) -> View:
        """Retrieve an existing or create a new View object."""
        if view_name in self._views:
            return self._views[view_name]
        if view_name is not None and self._views[None].name == view_name:
            return self._views[None]
        self._views[view_name] = View(view_name)
        self.logger.info(f"New view '{view_name}' created.")
        return self._views[view_name]

    @property
    def views(self):
        print(pretty_dict({"[active]" if k is None else k: v for k, v in self._views.items()}, "view_name", "Description"))

    def switch_view(self, view_name: str,
                    show_info: bool = True) -> None:
        if view_name is None:
            return
        new_view = self.get_view(view_name)
        old_view = self.get_view()
        if old_view.name is not None:
            self._views[old_view.name] = old_view
        self._views[None] = new_view
        new_name = new_view.name
        if new_name in self._views:
            del(self._views[new_name])
        for fname, piece in self:
            if new_view.check_token('fname', fname):
                if new_view not in piece._views.values() and\
                    new_name not in piece._views and\
                    piece.get_view().name != new_name:
                    piece.set_view(new_view)
                else:
                    piece.switch_view(new_name, show_info=False)
        if show_info:
            self.info()


    def __getattr__(self, view_name):
        if view_name in self._views:
            self.info(view_name=view_name)
        elif view_name is not None and self._views[None].name == view_name:
            self.info()
        else:
            raise AttributeError(f"'{view_name}' is not an existing view. Use _.get_view('{view_name}') to create it.")

    def iter_pieces(self, view_name: str = None) -> Iterator[Tuple[str, Piece]]:
        """Iterate through corpora under the current or specified view."""
        view = self.get_view(view_name)
        param_sum = view.fnames_in_metadata + view.fnames_not_in_metadata
        if param_sum == 0:
            # all excluded, need to update filter counts accordingly
            key = 'filtered_fnames'
            discarded_items, *_ = list(zip(*view.filter_by_token('fnames', self)))
            view._discarded_items[key].extend(discarded_items)
            filtering_counts = view._last_filtering_counts[key]
            filtering_counts[[0,1]] = filtering_counts[[1, 0]]
            yield from []
        elif param_sum == 2:
            for fname, piece in view.filter_by_token('fnames', self):
                if view_name not in piece._views:
                    if view_name is None:
                        piece.set_view(view)
                    else:
                        piece.set_view(**{view_name: view})
                yield fname, piece
        else:
            # need to differentiate between files that are and are not included in the metadata.tsv file
            fnames = self.fnames_in_metadata() if view.fnames_in_metadata else self.fnames_not_in_metadata()
            n_kept, n_discarded = 0, 0
            discarded_items = []
            for fname, piece in view.filter_by_token('fnames', self):
                if fname in fnames:
                    if view_name not in piece._views:
                        if view_name is None:
                            piece.set_view(view)
                        else:
                            piece.set_view(**{view_name: view})
                    yield fname, piece
                else:
                    discarded_items.append(fname)
                    n_kept -= 1
                    n_discarded += 1
            key = 'filtered_fnames'
            view._last_filtering_counts[key] += np.array([n_kept, n_discarded, 0], dtype='int')
            view._discarded_items[key].extend(discarded_items)




    def select_files(self, file_type: Union[str, Collection[str]],
                     view_name: str = None,
                     parsed: bool = True,
                     unparsed: bool = True,
                     choose: Literal['all', 'auto', 'ask'] = 'all',
                     flat: bool = False) -> Dict[str, Union[Dict[str, File], List[File]]]:
        """

        Args:
            file_type:
            choose:

        Returns:
            {fname -> {type -> [:obj:`File`]}} dict if flat=False (default).
            {fname -> [:obj:`File`]} dict if flat=True.
        """
        assert parsed + unparsed > 0, "At least one of 'parsed' and 'unparsed' needs to be True."
        result = {}
        if choose == 'ask':
            choose = 'all'
            logger.warning(f"choose='ask' hasn't been implemented yet for Corpus.select_files(); setting to 'auto'")
        for fname, piece in self.iter_pieces(view_name=view_name):
            type2file = piece.select_files(file_type=file_type,
                                            view_name=view_name,
                                            parsed=parsed,
                                            unparsed=unparsed,
                                            choose=choose,
                                            flat=flat)
            result[piece.fname] = type2file
        return result

    def select_score_files(self, choose: Literal['all', 'auto', 'ask'] = 'all'):
        if choose == 'ask':
            choose = 'all'
            logger.warning(f"choose='ask' hasn't been implemented yet for Corpus.select_score_files(); setting to 'auto'")
        result = self.select_files('scores', choose=choose)
        result = {fname: list(type2file.values()) for fname, type2file in result.items()}

        if all(len(l) == 1 for l in result.values()):
            return {fname: l[0] for fname, l in result.items()}

    def fnames_in_metadata(self) -> List[str]:
        if self.metadata_tsv is None:
            return  []
        try:
            return sorted(self.metadata_tsv.fname.unique())
        except AttributeError:
            return sorted(self.metadata_tsv.fnames.unique())

    def fnames_not_in_metadata(self) -> List[str]:
        metadata_fnames = self.fnames_in_metadata()
        if len(metadata_fnames) == 0:
            return self.score_fnames
        return [f for f in self.score_fnames
                if not (f in metadata_fnames or any(f.startswith(md_fname) for md_fname in metadata_fnames))
                ]

    def create_pieces(self):
        """Creates and stores one :obj:`Piece` object per fname."""
        if self.metadata_tsv is None:
            fnames = self.score_fnames
        else:
            # if metadata.tsv was found and not all score names are present in it,
            # the file may reflect a selection of pieces or needs to be updated.
            # Creating additional pieces only for score names, that don't begin with
            # any of the names contained in the metadata file, interpreting these as
            # having a suffix.
            metadata_fnames = self.fnames_in_metadata()
            additional_fnames = []
            for fname in self.score_fnames:
                if fname in metadata_fnames:
                    continue
                if any(fname.startswith(md_fname) for md_fname in metadata_fnames):
                    continue
                additional_fnames.append(fname)
            fnames = sorted(metadata_fnames + additional_fnames)
        for fname in fnames:
            logger_cfg = dict(self.logger_cfg)
            logger_name = self.logger.name + '.' + fname
            logger_cfg['name'] = logger_name
            piece = Piece(fname, view=self.get_view(), logger_cfg=logger_cfg, ms=self.ms)
            piece.set_view(**{view_name: view for view_name, view in self._views.items() if view_name is not None})
            self._pieces[fname] = piece
            self.logger_names[fname] = logger_name
        fnames = sorted(fnames, key=len, reverse=True)
        if len(fnames) == 0:
            self.logger.warning(f"Corpus contains neither scores nor metadata.")
            return
        for file in self.files:
            # try to associate all detected files with one of the created pieces and
            # store the mapping in :attr:`ix2fname`
            piece_name = None
            if file.fname in fnames:
                piece_name = file.fname
            else:
                for fname in fnames:
                    if file.fname.startswith(fname):
                        piece_name = fname
                        break
            if piece_name is None:
                if 'metadata' not in file.fname:
                    self.logger.warning(f"Could not associate {file.file} with any of the pieces.")
                self.ix2fname[file.ix] = None
            else:
                if self._pieces[piece_name].add_file(file):
                    self.ix2fname[file.ix] = piece_name
                    self.logger_names[file.ix] = self._pieces[piece_name].logger.name
                else:
                    self.ix2fname[file.ix] = None


    def find_and_load_metadata(self) -> None:
        """Checks if a 'metadata.tsv' is present at the default path and parses or creates it."""
        metadata_path = os.path.join(self.corpus_path, 'metadata.tsv')
        full_paths = self.files_df['full_path'].values
        if metadata_path in full_paths:
            ixs, = np.where(full_paths == metadata_path)
            self.metadata_ix = ixs[0]
            self.metadata_tsv = load_tsv(metadata_path)
            self.parsed_files[self.metadata_ix] = self.metadata_tsv

    def get_files_of_type(self, file_type: str) -> pd.DataFrame:
        """Filters :attr:`files_df` by its 'type' column."""
        return self.files_df[self.files_df.type == file_type].copy()

    def collect_fnames_from_scores(self) -> None:
        """Construct sorted list of fnames from all detected scores."""
        detected_scores = self.get_files_of_type('scores')
        self.score_fnames = sorted(detected_scores.fname.unique())

    def detect_parseable_files(self) -> None:
        """Walks through the corpus_path and collects information on all parseable files."""
        score_extensions = ['.' + ext for ext in Score.parseable_formats]
        detected_extensions = score_extensions + ['.tsv']
        for current_path, subdirs, files in os.walk(self.corpus_path):
            current_subdir = os.path.relpath(current_path, self.corpus_path)
            subdirs[:] = sorted(sd for sd in subdirs if not sd.startswith('.'))
            parseable_files = [(f,) + os.path.splitext(f) for f in files if os.path.splitext(f)[1] in detected_extensions]
            for file, file_name, file_ext in sorted(parseable_files):
                full_path = os.path.join(current_path, file)
                rel_path = os.path.join(current_subdir, file)
                file_type = path2type(full_path, logger=self.logger)
                F = File(
                    ix=len(self.files),
                    type = file_type,
                    file=file,
                    fname=file_name,
                    fext=file_ext,
                    subdir=current_subdir,
                    corpus_path=self.corpus_path,
                    rel_path=rel_path,
                    full_path=full_path,
                    directory=current_path,
                    suffix='',
                )
                self.files.append(F)
        self.files_df = pd.DataFrame(self.files).set_index('ix')

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
            grouped_ids = {k: list(range(len(self.fnames[k]))) for k in self._treat_key_param(keys)}
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
                self.logger.info(f'This Corpus object does not include any {msg[which]}.')
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
            them to the :py:attr:`~.score.Score` objects (``Corpusd.parsed_mscx[ID].new_key``).
        tsv_key : :obj:`str`, optional
            A key under which parsed TSV files are stored of which the type has been inferred as 'labels'.
            Note that passing ``tsv_key`` results in the deprecated use of Corpus.match_files(). The preferred way
            is to parse the labels to be attached under the same key as the scores and use
            View.add_detached_labels().
        revision_specifier : :obj:`str`, optional
            If you want to retrieve a previous version of the TSV file from a git commit (e.g. for
            using compare_labels()), pass the commit's SHA.
        """
        keys = self._treat_key_param(keys)
        if tsv_key is None:
            for key in keys:
                view = self._get_view(key)
                view.add_detached_annotations(use=use, new_key=new_key, revision_specifier=revision_specifier)
            return
        matches = self.match_files(keys=keys + [tsv_key])
        matches = matches[matches.labels.notna() | matches.expanded.notna()]
        matches.labels.fillna(matches.expanded, inplace=True)
        list_of_pairs = list(matches[['scores', 'labels']].itertuples(name=None, index=False))
        if len(list_of_pairs) == 0:
            self.logger.error(f"No files could be matched based on file names, probably a bug due to the deprecated use of the tsv_key parameter."
                    f"The preferred way of adding labels is parsing them under the same key as the scores and use Corpus[key].add_detached_annotations()")
            return
        self._add_detached_annotations_by_ids(list_of_pairs, new_key=new_key)


    def add_files(self, paths, key=None, exclude_re=None):
        """

        Parameters
        ----------
        paths : :obj:`~collections.abc.Collection`
            The paths of the files you want to add to the object.
        key : :obj:`str`
            | Pass a string to identify the loaded files.
            | If None is passed, paths relative to :py:attr:`last_scanned_dir` are used as keys. If :py:meth:`add_dir`
              hasn't been used before, the longest common prefix of all paths is used.

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
        json_ixs = [i for i, p in enumerate(paths) if p.endswith('.json')]
        if len(json_ixs) > 0:
            for i in reversed(json_ixs):
                try:
                    with open(paths[i]) as f:
                        loaded_paths = json.load(f)
                    paths.extend(loaded_paths)
                    self.logger.info(f"Unpacked the {len(loaded_paths)} paths found in {paths[i]}.")
                    del(paths[i])
                except Exception:
                    self.logger.info(f"Could not load paths from {paths[i]} because of the following error(s):\n{sys.exc_info()[1]}")
        if key is None:
            # try to see if any of the paths is part of a corpus (superdir has 'metadata.tsv')
            for path in paths:
                parent_corpus = path2parent_corpus(path)
                if parent_corpus is not None:
                    key = os.path.basename(parent_corpus)
                    self.logger.info(f"Using key='{key}' because the directory contains 'metadata.tsv' or is a git repository.")
                    metadata_path = os.path.join(parent_corpus, 'metadata.tsv')
                    if metadata_path not in paths:
                        paths.append(metadata_path)
                        self.logger.info(f"{metadata_path} was automatically added.")
                    break
        if key is None:
            # if only one key is available, pick that one
            keys = self.keys()
            if len(keys) == 1:
                key = keys[0]
                self.logger.info(f"Using key='{key}' because it is the only one currently in use.")
            else:
                self.logger.error(f"Couldn't add individual files because no key was specified and no key could be inferred.",
                                  extra={"message_id": (8,)})
                return []
        if key not in self.files:
            self.logger.debug(f"Adding '{key}' as new corpus.")
            self.files[key] = []
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
        if key is None:
            ids = [self._handle_path(p, path2parent_corpus(p)) for p in paths]
        else:
            ids = [self._handle_path(p, key) for p in paths]
        if sum(True for x in ids if x[0] is not None) > 0:
            selector, added_ids = zip(*[(i, x) for i, x in enumerate(ids) if x[0] is not None])
            exts = self.count_extensions(ids=added_ids, per_key=True)
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
        """ Update :obj:`Corpus.labels_cfg` and retrieve new 'labels' tables accordingly.

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
            self.logger.debug("No scores have been parsed so far. Use Corpus.parse_mscx()")
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
            If ``which='detached'``, the keys are keys from Score objects, otherwise they are keys from this Corpus object.

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


    def count_extensions(self, keys=None, ids=None, per_key=False, per_subdir=False):
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
        """

        if per_subdir:
            res_dict = defaultdict(dict)
            for k, subdir, ixs in self._iter_subdir_selectors(keys=keys, ids=ids):
                res_dict[k][subdir] = dict(Counter(iter_selection(self.fexts[k], ixs)))
            return dict(res_dict)
        else:
            res_dict = {}
            if ids is not None:
                grouped_ids = group_id_tuples(ids)
                for k, ixs in grouped_ids.items():
                    res_dict[k] = Counter(iter_selection(self.fexts[k], ixs))
            else:
                keys = self._treat_key_param(keys)
                for k in keys:
                    res_dict[k] = Counter(self.fexts[k])
            if per_key:
                return {k: dict(v) for k, v in res_dict.items()}
            return dict(sum(res_dict.values(), Counter()))



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
        interval_index : :obj:`bool`, optional
            Sets ``quarterbeats`` to True. Pass True to replace the indices of the returned DataFrames by
            :obj:`pandas.IntervalIndex <pandas.IntervalIndex>` with quarterbeat intervals. Rows that don't have a
            quarterbeat position are removed.

        Returns
        -------
        {feature -> {(key, i) -> pd.DataFrame}} if not flat (default) else {(key, i, feature) -> pd.DataFrame}
        """

        if len(self.parsed_files) == 0:
            self.logger.error("No files have been parsed so far.")
            return {}
        bool_params = Score.dataframe_types
        l = locals()
        file_types = [tsv_type for tsv_type in Score.dataframe_types if l[tsv_type]]
        res = {} if flat else defaultdict(dict)
        for piece in self.iter_pieces():
            fname = piece.fname
            file, score_obj = piece.get_parsed_score()
            if score_obj is None:
                self.logger.info(f"No parsed score found for '{fname}'")
                continue
            for typ in file_types:
                df = getattr(score_obj.mscx, typ)()
                if df is None:
                    continue
                if flat:
                    res[(file.ix, typ)] = df
                else:
                    res[typ][file.ix] = df
        return dict(res)


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


    def get_piece(self, id):
        key, i = id
        return self[key].get_piece(id)


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
        logger = self.ix_logger(id)
        piece = self.get_piece(id)
        if piece is None:
            logger.warning(f"Could not associate ID {id} with a Piece object.")
            return
        return piece.get_dataframe('measures', unfold=unfold)



    def _get_playthrough2mc(self, id):
        logger = self.ix_logger(id)
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
        logger = self.ix_logger(id)
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

    def ix_logger(self, ix):
        return get_logger(self.logger_names[ix])


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

    def count(self, types: bool = True,
              parsed: bool = True,
              as_dict: bool = False,
              drop_zero: bool = False,
              view_name: str = None) -> Union[pd.DataFrame, dict]:
        assert types + parsed > 0, "At least one parameter needs to be True"
        fname2counts = {}
        for fname, piece in self.iter_pieces(view_name=view_name):
            if types:
                type_count = piece.count_types(view_name=view_name)
                if not parsed:
                    fname2counts[fname] = type_count
            if parsed:
                parsed_count = piece.count_parsed()
                if not types:
                    fname2counts[fname] = parsed_count
            if types & parsed:
                alternating_counts = {}
                for (k1, v1), (k2, v2) in zip(type_count.items(), parsed_count.items()):
                    alternating_counts[k1] = v1
                    alternating_counts[k2] = v2
                fname2counts[fname] = alternating_counts
        if as_dict:
            return fname2counts
        df = pd.DataFrame.from_dict(fname2counts, orient='index')
        if drop_zero:
            empty_cols = df.columns[df.sum() == 0]
            return df.drop(columns=empty_cols)
        return df



    def info(self,  return_str: bool = False,
             view_name: str = None,
             show_discarded: bool = False):
        """"""
        view = self.get_view(view_name)
        view.reset_filtering_data()
        msg = f"{view}\n\n"
        all_pieces_df = self.count(drop_zero=True, view_name=view_name)
        if self.metadata_tsv is None:
            msg += "No 'metadata.tsv' file is present.\n\n"
            msg += all_pieces_df.to_string()
        else:
            metadata_fnames = set(self.fnames_in_metadata())
            included_selector = all_pieces_df.index.isin(metadata_fnames)
            if included_selector.all():
                msg += "All pieces are listed in 'metadata.tsv':\n\n"
                msg += all_pieces_df.to_string()
            elif not included_selector.any():
                msg = "None of the pieces is actually listed in 'metadata.tsv'.\n\n"
                msg += all_pieces_df.to_string()
            else:
                msg = "Only the following pieces are listed in 'metadata.tsv':\n\n"
                msg += all_pieces_df[included_selector].to_string()
                not_included = ~included_selector
                plural = "These ones here are" if not_included.sum() > 1 else "This one is"
                msg += f"\n\n{plural} missing from 'metadata.tsv':\n\n"
                msg += all_pieces_df[not_included].to_string()
        msg += '\n\n' + view.filtering_report(show_discarded=show_discarded)
        if return_str:
            return msg
        print(msg)


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


    def metadata_from_parsed(self, ixs=None, from_tsv=False):
        """ Retrieve concatenated metadata from parsed scores or TSV files.

        Parameters
        ----------
        ixs
        from_tsv : :obj:`bool`, optional
            If set to True, you'll get a concatenation of all parsed TSV files that have been
            recognized as containing metadata. If you want to specifically retrieve files called
            'metadata.tsv' only, use :py:meth:`metadata_tsv`.

        Returns
        -------
        :obj:`pandas.DataFrame`
        """
        if from_tsv:
            raise NotImplementedError(f"Not there yet")

        rows = {}
        for piece in self.iter_pieces():
            fname = piece.fname
            file, score_obj = piece.get_parsed_score()
            metadata_dict = score_obj.mscx.metadata
            metadata_dict['subdirectory'] = file.subdir
            rows[fname] = metadata2series(metadata_dict)
        df = pd.DataFrame.from_dict(rows, orient='index').sort_index()
        df.index.rename('fname', inplace=True)
        df = df.reset_index()
        df = column_order(df, METADATA_COLUMN_ORDER)
        return df

    def metadata_tsv(self, keys=None, parse_if_necessary=True):
        """Returns a {id -> DataFrame} dictionary with all metadata.tsv files. To retrieve parsed
        files recognized as containing metadata independent of their names, use :py:meth:`get_tsvs`."""
        keys = self._treat_key_param(keys)
        if len(self._parsed_tsv) == 0:
            if not parse_if_necessary:
                self.logger.debug(f"No TSV files have been parsed so far. Use Corpus.parse_tsv().")
                return pd.DataFrame()
        metadata_dfs = {}
        for k in keys:
            try:
                i = self.files[k].index('metadata.tsv')
            except ValueError:
                self.logger.debug(f"Key '{k}' does not include a file named 'metadata.tsv'.")
                return metadata_dfs
            id = (k, i)
            if id not in self._parsed_tsv:
                if parse_if_necessary:
                    self.parse_tsv()
                else:
                    self.logger.debug(
                        f"Found unparsed metadata for key '{k}' but parse_if_necessary is set to False.")
            if id in self._parsed_tsv:
                metadata_dfs[id] = self._parsed_tsv[id]
            elif parse_if_necessary:
                self.logger.debug(f"Could not find the DataFrame for the freshly parsed {self.full_paths[k][i]}.")
        n_found = len(metadata_dfs)
        if n_found == 0:
            self.logger.debug(f"No metadata.tsv files have been found for they keys {', '.join(keys)}")
            return {}
        return metadata_dfs



    def parse(self, keys=None, level=None, parallel=True, only_new=True, labels_cfg={}, fexts=None, cols={}, infer_types=None, simulate=None, **kwargs):
        """ Shorthand for executing parse_mscx and parse_tsv at a time."""
        if simulate is not None:
            self.simulate = simulate
        self.parse_mscx(level=level, parallel=parallel, only_new=only_new, labels_cfg=labels_cfg)
        self.parse_tsv(fexts=fexts, cols=cols, infer_types=infer_types, level=level, **kwargs)



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
        self.labels_cfg.update(update_labels_cfg(labels_cfg, logger=self.logger))
        self.logger.debug(f"Parsing scores with parameters parallel={parallel}, only_new={only_new}")

        fname2type2files = self.select_files('scores', view_name=view_name, parsed=not only_new)
        fname2files = {fname: type2files['scores']
                       for fname, type2files in fname2type2files.items()
                       if len(type2files['scores']) > 0}
        selected_files = sum(fname2files.values(), start=[])
        selected_scores_df = pd.concat([pd.DataFrame(files) for files in fname2files.values()], keys=fname2files.keys())
        self.logger.debug(selected_scores_df.to_string())
        exts = selected_scores_df.fext.value_counts()

        if any(ext[1:].lower() in Score.convertible_formats for ext in exts.index) and parallel:
            msg = f"The file extensions [{', '.join(ext[1:] for ext in exts.index if ext[1:].lower() in Score.convertible_formats )}] " \
                  f"require temporary conversion with your local MuseScore, which is not possible with parallel " \
                  f"processing. Parse with parallel=False or set View.include_convertible=False."
            if self.ms is None:
                msg += "\nIn case you want to temporarily convert these files, you will also have to set the " \
                       "property ms of this object to the path of your MuseScore 3 executable."
            self.logger.error(msg)
            return


        configs = [dict(
            name=self.logger_names[ix]
        ) for ix in selected_scores_df.ix]

        ### collect argument tuples for calling parse_musescore_file
        parse_this = [(file, conf, self.labels_cfg, parallel, self.ms) for file, conf in zip(selected_files, configs)]
        target = len(parse_this)
        try:
            if parallel:
                pool = mp.Pool(mp.cpu_count())
                res = pool.starmap(parse_musescore_file, parse_this)
                pool.close()
                pool.join()
                successful_results = {file.ix: score for file, score in zip(selected_files, res) if score is not None}
                self.parsed_files.update(successful_results)
                with_captured_logs = [score for score in successful_results.values() if hasattr(score, 'captured_logs')]
                if len(with_captured_logs) > 0:
                    log_capture_handler = get_log_capture_handler(self.logger)
                    if log_capture_handler is not None:
                        for score in with_captured_logs:
                            log_capture_handler.log_queue.extend(score.captured_logs)
                successful = len(successful_results)
            else:
                parsing_results = [parse_musescore_file(*params) for params in parse_this]
                successful_results = {file.ix: score for file, score in zip(selected_files, parsing_results) if score is not None}
                self.parsed_files.update(successful_results)
                successful = len(successful_results)
            for ix, score in successful_results.items():
                self._get_piece(self.ix2fname[ix]).add_parsed_score(ix, score)
            if successful > 0:
                if successful == target:
                    self.logger.info(f"All {target} files have been parsed successfully.")
                else:
                    self.logger.info(f"Only {successful} of the {target} files have been parsed successfully.")
            else:
                self.logger.info(f"None of the {target} files have been parsed successfully.")
        except KeyboardInterrupt:
            self.logger.info("Parsing interrupted by user.")
            raise
        finally:
            #self._collect_annotations_objects_references(ids=ids)
            pass

    def parse_tsv(self, cols={}, infer_types=None, level=None, **kwargs):
        """ Parse TSV files to be able to do something with them.

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
        """
        if level is not None:
            self.change_logger_cfg(level=level)

        selected_scores = self.select_files('scores')

        for id in ids:
            key, i = id
            path = self.full_paths[key][i]
            logger = self.ix_logger(id)
            try:
                df = load_tsv(path, **kwargs)
            except Exception:
                logger.info(f"Couldn't be loaded, probably no tabular format or you need to specify 'sep', the delimiter."
                                 f"\n{path}\nError: {sys.exc_info()[1]}")
                continue
            label_col = cols['label'] if 'label' in cols else 'label'
            try:
                self._parsed_tsv[id] = df
                if 'label' in cols and label_col in df.columns:
                    tsv_type = 'labels'
                else:
                    tsv_type = infer_tsv_type(df)

                if tsv_type is None:
                    logger.debug(
                        f"No label column '{label_col}' was found in {self.rel_paths[key][i]} and its content could not be inferred. Columns: {df.columns.to_list()}")
                    self._tsv_types[id] = 'other'
                else:
                    self._tsv_types[id] = tsv_type
                    if tsv_type == 'metadata':
                        self._metadata = pd.concat([self._metadata, self._parsed_tsv[id]])
                        logger.debug(f"{self.rel_paths[key][i]} parsed as metadata.")
                    else:
                        self._dataframes[tsv_type][id] = self._parsed_tsv[id]
                        if tsv_type in ['labels', 'expanded']:
                            if label_col in df.columns:
                                logger_cfg = dict(self.logger_cfg)
                                logger_cfg['name'] = self.logger_names[(key, i)]
                                self._annotations[id] = Annotations(df=df, cols=cols, infer_types=infer_types,
                                                                          logger_cfg=logger_cfg, level=level)
                                logger.debug(
                                    f"{self.rel_paths[key][i]} parsed as annotation table and an Annotations object was created.")
                            else:
                                logger.info(
        f"""The file {self.rel_paths[key][i]} was recognized to contain labels but no label column '{label_col}' was found in {df.columns.to_list()}
        Specify parse_tsv(key='{key}', cols={{'label'=label_column_name}}).""")
                        else:
                            logger.debug(f"{self.rel_paths[key][i]} parsed as {tsv_type} table.")

            except Exception as e:
                self.logger.error(f"Parsing {self.rel_paths[key][i]} failed with the following error:\n{e}")

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

    def output_dataframes(self, keys=None, root_dir=None, notes_folder=None, notes_suffix='',
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
            Defaults to None, meaning that the original root directory is used that was added to the Corpus object.
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
            Defaults to ``None``. Pass a value to set the object attribute :py:attr:`~ms3.parse.Corpus.simulate`.
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
        l = locals()
        df_types = [t for t in Score.dataframe_types if t != 'metadata']
        folder_vars = [t + '_folder' for t in df_types]
        suffix_vars = [t + '_suffix' for t in df_types]
        folder_params = {t: l[p] for t, p in zip(df_types, folder_vars) if l[p] is not None}
        if len(folder_params) == 0 and metadata_path is None:
            self.logger.warning("Pass at least one parameter to store files.")
            return [] if simulate else None
        suffix_params = {t: '_unfolded' if l[p] is None and unfold else l[p] for t, p in zip(df_types, suffix_vars) if t in folder_params}
        df_params = {p: True for p in folder_params.keys()}
        if silence_label_warnings:
            with temporarily_suppress_warnings(self) as self:
                dataframes = self.get_dataframes(keys, unfold=unfold, quarterbeats=quarterbeats, flat=True, **df_params)
        else:
            dataframes = self.get_dataframes(keys, unfold=unfold, quarterbeats=quarterbeats, flat=True, **df_params)
        modus = 'would ' if simulate else ''
        if len(dataframes) == 0 and metadata_path is None:
            self.logger.info(f"No files {modus}have been written.")
            return [] if simulate else None
        paths = {}
        warnings, infos = [], []
        unf = 'Unfolded ' if unfold else ''
        for (ix, what), dataframe in dataframes.items():
            new_path = self._store_tsv(df=dataframe, ix=ix, folder=folder_params[what], suffix=suffix_params[what], root_dir=root_dir, what=what, simulate=simulate)
            if new_path in paths:
                warnings.append(f"The {paths[new_path]} at {new_path} {modus}have been overwritten with {what}.")
            else:
                infos.append(f"{unf}{what} {modus}have been stored as {new_path}.")
            paths[new_path] = what
        if len(warnings) > 0:
            self.logger.warning('\n'.join(warnings))
        l_infos = len(infos)
        l_target = len(dataframes)
        if l_target > 0:
            if l_infos == 0:
                self.logger.info(f"\n\nNone of the {l_target} {modus}have been written.")
            elif l_infos < l_target:
                msg = f"\n\nOnly {l_infos} out of {l_target} files {modus}have been stored."
            else:
                msg = f"\n\nAll {l_infos} {modus}have been written."
            self.logger.info('\n'.join(infos) + msg)
        if metadata_path is not None:
            full_path = self.update_metadata_from_parsed(metadata_path, markdown)
            if full_path is not None:
                paths[full_path] = 'metadata'
        return paths

    def update_metadata_from_parsed(self, metadata_path: str, markdown: bool = True) -> str:
        """ Gathers the metadata from parsed and currently selected scores and updates 'metadata.tsv' with the information.

        Args:
            metadata_path: Folder where to update or create the TSV file.
            markdown: Pass False if you don't want to also create or update an README.md file.

        Returns:

        """
        md = self.metadata_from_parsed()
        if len(md.index) == 0:
            self.logger.debug(f"\n\nNo metadata to write.")
            return None
        if os.path.isabs(metadata_path) or '~' in metadata_path:
            metadata_path = resolve_dir(metadata_path)
            path = metadata_path
        else:
            path = os.path.abspath(os.path.join(self.corpus_path, metadata_path))
        fname, ext = os.path.splitext(path)
        if ext != '':
            path, file = os.path.split(path)
        else:
            file = 'metadata.tsv'
        if not os.path.isdir(path):
            os.makedirs(path)
        full_path = os.path.join(path, file)
        write_metadata(md, full_path, markdown=markdown, logger=self.logger)
        return full_path

    def output_mscx(self, keys=None, ids=None, root_dir=None, folder='.', suffix='', overwrite=False, simulate=False):
        """ Stores the parsed MuseScore files in their current state, e.g. after detaching or attaching annotations.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
            Key(s) for which to count file extensions.  By default, all keys are selected.
        ids : :obj:`~collections.abc.Collection`
            If you pass a collection of IDs, ``keys`` is ignored and only the selected extensions are counted.
        root_dir : :obj:`str`, optional
            Defaults to None, meaning that the original root directory is used that was added to the Corpus object.
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




    def _calculate_path(self, ix, root_dir, folder, enforce_below_root=False):
        """ Constructs a path and file name from a loaded file based on the arguments.

        Parameters
        ----------
        ix : :obj:`int`
            ID from which to construct the new path and filename.
        folder : :obj:`str`
            Where to store the file. Can be relative to ``root_dir`` or absolute, in which case ``root_dir`` is ignored.
            If ``folder`` is relative, the behaviour depends on whether it starts with a dot ``.`` or not: If it does,
            the folder is created at every end point of the relative tree structure under ``root_dir``. If it doesn't,
            it is created only once, relative to ``root_dir``, and the relative tree structure is build below.
        root_dir : :obj:`str`, optional
            Defaults to None, meaning that the original root directory is used that was added to the Corpus object.
            Otherwise, pass a directory to rebuild the original substructure. If ``folder`` is an absolute path,
            ``root_dir`` is ignored.
        enforce_below_root : :obj:`bool`, optional
            If True is passed, the computed paths are checked to be within ``root_dir`` or ``folder`` respectively.
        """
        if folder is not None and (os.path.isabs(folder) or '~' in folder):
            folder = resolve_dir(folder)
            path = folder
        else:
            file = self.files[ix]
            root = self.corpus_path if root_dir is None else resolve_dir(root_dir)
            if folder is None:
                path = root
            elif folder[0] == '.':
                path = os.path.abspath(os.path.join(root, file.subdir, folder))
            else:
                path = os.path.abspath(os.path.join(root, folder, file.subdir))
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

    def _handle_path(self, full_path, key, skip_checks=False):
        """Store information about the file at ``full_path`` in their various fields under the given key."""
        full_path = resolve_dir(full_path)
        if not skip_checks and not os.path.isfile(full_path):
            self.logger.error("No file found at this path: " + full_path)
            return (None, None)
        file_path, file = os.path.split(full_path)
        file_name, file_ext = os.path.splitext(file)
        if not skip_checks and file_ext[1:] not in Score.parseable_formats + ('tsv',):
            ext_string = "without extension" if file_ext == '' else f"with extension {file_ext}"
            self.logger.debug(f"ms3 does not handle files {ext_string} -> discarding" + full_path)
            return (None, None)
        rel_path = os.path.relpath(file_path, self.last_scanned_dir)
        subdir = get_path_component(rel_path, key)
        if file in self.files[key]:
            same_name = [i for i, f in enumerate(self.files[key]) if f == file]
            if not skip_checks and any(True for i in same_name if self.rel_paths[key][i] == rel_path):
                self.logger.debug(
                    f"""The file name {file} is already registered for key '{key}' and both files have the relative path {rel_path}.
Load one of the identically named files with a different key using add_dir(key='KEY').""")
                return (None, None)
            self.logger.debug(
                f"The file {file} is already registered for key '{key}' but can be distinguished via the relative path {rel_path}.")

        i = len(self.files[key])
        self.logger_names[(key, i)] = f"{self.logger.name}.{key}.{file_name.replace('.', '')}{file_ext}"
        self.full_paths[key].append(full_path)
        self.scan_paths[key].append(self.last_scanned_dir)
        self.rel_paths[key].append(rel_path)
        self.subdirs[key].append(subdir)
        self.paths[key].append(file_path)
        self.files[key].append(file)
        self.logger_names[(key, i)] = f"{self.logger.name}.{key}.{file_name.replace('.', '')}{file_ext}"
        self.fnames[key].append(file_name)
        self.fexts[key].append(file_ext)
        F = File(
            id=(key, i),
            full_path=full_path,
            scan_path=self.last_scanned_dir,
            rel_path=rel_path,
            subdir=subdir,
            path=file_path,
            file=file,
            fname=file_name,
            fext=file_ext
        )
        self.id2file_info[(key, i)] = F
        return key, i

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
            Defaults to None, meaning that the original root directory is used that was added to the Corpus object.
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
        logger = self.ix_logger(id)
        fname = self.fnames[key][i]

        if id not in self._parsed_mscx:
            logger.error(f"No Score object found. Call parse_mscx() first.")
            return
        path = self._calculate_path(ix=i, root_dir=root_dir, folder=folder)
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


    def _store_tsv(self, df, ix, folder, suffix='', root_dir=None, what='unknown', simulate=False):
        """ Stores a given DataFrame by constructing path and file name from a loaded file based on the arguments.

        Parameters
        ----------
        df : :obj:`pandas.DataFrame`
            DataFrame to store as a TSV.
        ix : :obj:`int`
            index of the file from which to construct the new path and filename.
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
        tsv_logger = self.ix_logger(ix)

        if df is None:
            tsv_logger.debug(f"No DataFrame for {what}.")
            return
        path = self._calculate_path(ix=ix, root_dir=root_dir, folder=folder)
        if path is None:
            return

        fname = self.files[ix].fname + suffix + ".tsv"
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
                self.logger.warning(f"Corpusd metadata do not contain the columns 'rel_paths' and 'fnames' "
                                    f"needed to match information on identical files.")
                return []
        new = self.metadata_from_parsed(from_tsv=False).set_index(['rel_paths', 'fnames'])
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
                    self.ix_logger(id).debug(f"Updated with {new_dict}")
                    ids.append(id)

            self.logger.info(f"{l} files updated.")
        else:
            self.logger.info("Nothing to update.")
        return ids


    def __getstate__(self):
        """ Override the method of superclass """
        return self.__dict__


    def __getitem__(self, item) -> Piece:
        if isinstance(item, str):
            return self._get_piece(item)
        raise NotImplementedError("Currently subscripting works with fnames only.")


    def __iter__(self) -> Iterator[Tuple[str, Piece]]:
        """  Iterate through all (fname, Piece) tuples, regardless of any Views.

        Yields: (fname, Piece) tuples
        """
        yield from self._pieces.items()



    def __repr__(self):
        return self.info(return_str=True)



    def _get_piece(self, fname) -> Piece:
        """Returns the :obj:`Piece` object for fname."""
        assert fname in self._pieces, f"'{fname}' is not a piece in this corpus."
        return self._pieces[fname]
########################################################################################################################
########################################################################################################################
################################################# End of Corpus() ########################################################
########################################################################################################################
########################################################################################################################


@function_logger
def parse_musescore_file(file: File, logger_cfg={}, labels_cfg={}, read_only=False, ms=None) -> Score:
    """Performs a single parse and returns the resulting Score object or None."""
    path = file.full_path
    file = file.file
    logger.debug(f"Attempting to parse {file}")
    try:
        score = Score(path, read_only=read_only, labels_cfg=labels_cfg, logger_cfg=logger_cfg, ms=ms)
        if score is None:
            logger.debug(f"Encountered errors when parsing {file}")
        else:
            logger.debug(f"Successfully parsed {file}")
        return score
    except (KeyboardInterrupt, SystemExit):
        logger.info("Process aborted.")
        raise
    except Exception as e:
        logger.error(f"Unable to parse {path} due to the following exception:\n{e}")
        return None