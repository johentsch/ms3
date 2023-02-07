import re
from typing import Literal, Collection, Generator, Tuple, Union, Dict, Optional, List, Iterator

import sys, os
from collections import Counter, defaultdict

import pandas as pd

from .corpus import Corpus
from .logger import LoggedClass
from .piece import Piece
from ._typing import FileDict, FileList, CorpusFnameTuple, ScoreFacets, FacetArguments, FileParsedTuple, FileDataframeTuple, ScoreFacet, AnnotationsFacet, Facet
from .utils import get_musescore, get_first_level_corpora, pretty_dict, resolve_dir, \
    update_labels_cfg, available_views2str, path2parent_corpus, resolve_paths_argument, enforce_fname_index_for_metadata, File
from .view import View, create_view_from_parameters, DefaultView


class Parse(LoggedClass):
    """
    Class for creating one or several :class:`~.corpus.Corpus` objects and performing actions on all of them.
    """

    def __init__(self,
                 directory: Optional[Union[str, Collection[str]]] = None,
                 recursive: bool = True,
                 only_metadata_fnames: bool = True,
                 include_convertible: bool = False,
                 include_tsv: bool = True,
                 exclude_review: bool = True,
                 file_re: Optional[Union[str, re.Pattern]] = None,
                 folder_re: Optional[Union[str, re.Pattern]] = None,
                 exclude_re: Optional[Union[str, re.Pattern]] = None,
                 file_paths: Optional[Collection[str]] = None,
                 labels_cfg: dict = {},
                 ms=None,
                 **logger_cfg):
        """ Initialize a Parse object and try to create corpora if directories and/or file paths are specified.

        Args:
            directory: Path to scan for corpora.
            recursive: Pass False if you don't want to scan ``directory`` for subcorpora, but force making it a corpus instead.
            only_metadata_fnames:
                The default view excludes piece names that are not listed in the corpus' metadata.tsv file (e.g. when none was found).
                Pass False to include all pieces regardless. This might be needed when setting ``recursive`` to False.
            include_convertible:
                The default view excludes scores that would need conversion to MuseScore format prior to parsing.
                Pass True to include convertible scores in .musicxml, .midi, .cap or any other format that MuseScore 3 can open.
                For on-the-fly conversion, however, the parameter ``ms`` needs to be set.
            include_tsv: The default view includes TSV files. Pass False to disregard them and parse only scores.
            exclude_review:
                The default view excludes files and folders whose name contains 'review'.
                Pass False to include these as well.
            file_re: Pass a regular expression if you want to create a view filtering out all files that do not contain it.
            folder_re: Pass a regular expression if you want to create a view filtering out all folders that do not contain it.
            exclude_re: Pass a regular expression if you want to create a view filtering out all files or folders that contain it.
            file_paths:
                If ``directory`` is specified, the file names of these paths are used to create a filtering view excluding all other files.
                Otherwise, all paths are expected to be part of the same parent corpus which will be inferred from the first path by looking for the first parent directory that
                either contains a 'metadata.tsv' file or is a git. This parameter is deprecated and ``file_re`` should be used instead.
            labels_cfg: Pass a configuration dict to detect only certain labels or change their output format.
            ms:
                If you pass the path to your local MuseScore 3 installation, ms3 will attempt to parse musicXML, MuseScore 2,
                and other formats by temporarily converting them. If you're using the standard path, you may try 'auto', or 'win' for
                Windows, 'mac' for MacOS, or 'mscore' for Linux. In case you do not pass the 'file_re' and the MuseScore executable is
                detected, all convertible files are automatically selected, otherwise only those that can be parsed without conversion.
            **logger_cfg: Keyword arguments for changing the logger configuration. E.g. ``level='d'`` to see all debug messages.
        """
        if 'level' not in logger_cfg or (logger_cfg['level'] is None):
            logger_cfg['level'] = 'w'
        super().__init__(subclass='Parse', logger_cfg=logger_cfg)

        self.corpus_paths: Dict[str, str] = {}
        """{corpus_name -> path} dictionary with each corpus's base directory. Generally speaking, each corpus path is expected to contain a ``metadata.tsv`` and, maybe, to be a git.
        """

        self.corpus_objects: Dict[str, Corpus] = {}
        """{corpus_name -> Corpus} dictionary with one object per :attr:`corpus_path <corpus_paths>`.
        """

        self._ms = get_musescore(ms, logger=self.logger)

        self._views: dict = {}
        initial_view = create_view_from_parameters(only_metadata_fnames=only_metadata_fnames,
                                                   include_convertible=include_convertible,
                                                   include_tsv=include_tsv,
                                                   exclude_review=exclude_review,
                                                   file_paths=file_paths,
                                                   file_re=file_re,
                                                   folder_re=folder_re,
                                                   exclude_re=exclude_re,
                                                   level=self.logger.getEffectiveLevel())
        self._views[None] = initial_view
        if initial_view.name != 'default':
            self._views['default'] = DefaultView(level=self.logger.getEffectiveLevel())
        self._views['all'] = View(level=self.logger.getEffectiveLevel())
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

        if directory is not None:
            if isinstance(directory, str):
                directory = [directory]
            for d in directory:
                self.add_dir(directory=d, recursive=recursive)
        if file_paths is not None:
            self.add_files(file_paths)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END of __init__() %%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


    @property
    def ms(self) -> str:
        """Path or command of the local MuseScore 3 installation if specified by the user and recognized."""
        return self._ms

    @ms.setter
    def ms(self, ms):
        self._ms = get_musescore(ms, logger=self.logger)

    @property
    def n_detected(self) -> int:
        """Number of detected files aggregated from all :class:`~.corpus.Corpus` objects without taking views into account. Excludes metadata files."""
        return sum(corpus.n_detected for _, corpus in self)

    @property
    def n_orphans(self) -> int:
        """Number of files that are always disregarded because they could not be attributed to any of the fnames."""
        return sum(len(corpus.ix2orphan_file) for _, corpus in self)

    @property
    def n_parsed(self) -> int:
        """Number of parsed files aggregated from all :class:`~.corpus.Corpus` objects without taking views into account. Excludes metadata files."""
        return sum(corpus.n_parsed for _, corpus in self)

    @property
    def n_parsed_scores(self) -> int:
        """Number of parsed scores aggregated from all :class:`~.corpus.Corpus` objects without taking views into account. Excludes metadata files."""
        return sum(corpus.n_parsed_scores for _, corpus in self)

    @property
    def n_parsed_tsvs(self) -> int:
        """Number of parsed TSV files aggregated from all :class:`~.corpus.Corpus` objects without taking views into account. Excludes metadata files."""
        return sum(corpus.n_parsed_tsvs for _, corpus in self)

    @property
    def n_pieces(self) -> int:
        """Number of all available pieces ('fnames'), independent of the view."""
        return sum(corpus.n_pieces for _, corpus in self)

    @property
    def n_unparsed_scores(self) -> int:
        """Number of all detected but not yet parsed scores, aggregated from all :class:`~.corpus.Corpus` objects without taking views into account. Excludes metadata files."""
        return sum(corpus.n_unparsed_scores for _, corpus in self)

    @property
    def n_unparsed_tsvs(self) -> int:
        """Number of all detected but not yet parsed TSV files, aggregated from all :class:`~.corpus.Corpus` objects without taking views into account. Excludes metadata files."""
        return sum(corpus.n_unparsed_tsvs for _, corpus in self)


    @property
    def parsed_mscx(self) -> pd.DataFrame:
        """Deprecated property. Replaced by :attr:`n_parsed_scores`"""
        raise AttributeError(f"Property has been renamed to n_parsed_scores.")

    @property
    def parsed_tsv(self) -> pd.DataFrame:
        """Deprecated property. Replaced by :attr:`n_parsed_tsvs`"""
        raise AttributeError(f"Property has been renamed to n_parsed_tsvs.")

    @property
    def view(self) -> View:
        """Retrieve the current View object. Shorthand for :meth:`get_view`."""
        return self.get_view()

    @view.setter
    def view(self, new_view: View):
        if not isinstance(new_view, View):
            return TypeError("If you want to switch to an existing view, use its name like an attribute or "
                             "call _.switch_view().")
        self.set_view(new_view)

    @property
    def views(self) -> None:
        """Display a short description of the available views."""
        print(pretty_dict({"[active]" if k is None else k: v for k, v in self._views.items()}, "view_name", "Description"))

    @property
    def view_name(self) -> str:
        """Get the name of the active view."""
        return self.get_view().name

    @view_name.setter
    def view_name(self, new_name):
        view = self.get_view()
        view.name = new_name

    @property
    def view_names(self):
        return {view.name if name is None else name for name, view in self._views.items()}

    def add_corpus(self,
                   directory: str,
                   corpus_name: Optional[str] = None,
                   **logger_cfg) -> None:
        """
        This method creates a :class:`~.corpus.Corpus` object which scans the directory ``directory`` for parseable files.
        It inherits all :class:`Views <.view.View>` from the Parse object.

        Args:
            directory: Directory to scan for files.
            corpus_name:
                By default, the folder name of ``directory`` is used as name for this corpus. Pass a string to
                use a different identifier.
            **logger_cfg:
                Keyword arguments for configuring the logger of the new Corpus object. E.g. ``level='d'`` to see all debug messages.
                Note that the logger is a child logger of this Parse object's logger and propagates, so it might filter debug messages.
                You can use _.change_logger_cfg(level='d') to change the level post hoc.
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

    def add_detached_annotations(self, *args, **kwargs):
        """Deprecated method. Replaced by :meth:`insert_detached_labels`."""
        raise AttributeError(f"Method not in use any more. Use Parse.insert_detached_labels().")

    def add_dir(self, directory: str,
                recursive: bool = True,
                **logger_cfg) -> None:
        """
        This method decides if the directory ``directory`` contains several corpora or if it is a corpus
        itself, and calls :meth:`add_corpus` for each corpus.

        Args:
            directory: Directory to scan for corpora.
            recursive:
                By default, if any of the first-level subdirectories contains a 'metadata.tsv' or is a git, all first-level
                subdirectories of ``directory`` are treated as corpora, i.e. one :class:`~.corpus.Corpus` object per folder is created.
                Pass False to prevent this, which is equivalent to calling :meth:`add_corpus(directory) <add_corpus>`
            **logger_cfg:
                Keyword arguments for configuring the logger of the new Corpus objects. E.g. ``level='d'`` to see all debug messages.
                Note that the loggers are child loggers of this Parse object's logger and propagate, so it might filter debug messages.
                You can use _.change_logger_cfg(level='d') to change the level post hoc.
        """
        directory = resolve_dir(directory)
        if not os.path.isdir(directory):
            self.logger.warning(f"{directory} is not an existing directory.")
            return

        if not recursive:
            self.add_corpus(directory, **logger_cfg)
            return

        # new corpus/corpora to be added
        subdir_corpora = sorted(get_first_level_corpora(directory, logger=self.logger))
        n_corpora = len(subdir_corpora)
        if n_corpora == 0:
            self.logger.debug(f"Treating {directory} as corpus.")
            self.add_corpus(directory, **logger_cfg)
        else:
            self.logger.debug(f"{n_corpora} individual corpora detected in {directory}.")
            for corpus_path in subdir_corpora:
                self.add_corpus(corpus_path, **logger_cfg)


    def add_files(self, file_paths: Union[str, Collection[str]], corpus_name: Optional[str] = None) -> None:
        """
        Deprecated: To deal with particular files only, use :meth:`add_corpus` passing the directory containing them and
        configure the :class`~.view.View` accordingly. This method here does it for you but easily leads to unexpected behaviour.
        It expects the file paths to point to files located in a shared corpus folder
        on some higher level or in folders for which :class:`~.corpus.Corpus` objects have already been created.

        Args:
            file_paths: Collection of file paths. Only existing files can be added.
            corpus_name:

                * By default, I will try to attribute the files to existing :class:`~.corpus.Corpus` objects based on their paths. This makes sense only when new files have
                  been created after the directories were scanned.
                * For paths that do no not contain an existing corpus_path, I will try to detect the parent directory that is a corpus (based on it being a git or containing a ``metadata.tsv``).
                  If this is without success for the first path, I will raise an error. Otherwise, all subsequent paths will be considered to be part of that same corpus (watch out
                  meaningless relative paths!).
                * You can pass a folder name contained in the first path to create a new corpus, assuming that all other paths are contained in it (watch out meaningless relative paths!).
                * Pass an existing corpus_name to add the files to a particular corpus. Note that all parseable files under the corpus_path are detected anyway, and if you add files
                  from other directories, it will lead to invalid relative paths that work only on your system. If you're adding files that have been created after the Corpus object
                  has, you can leave this parameter empty; paths will be attributed to the existing corpora automatically.
        """
        resolved_paths = resolve_paths_argument(file_paths, logger=self.logger)
        if len(resolved_paths) == 0:
            return
        if corpus_name is None:
            add_to_existing = defaultdict(list)
            no_parent = []
            for path in resolved_paths:
                part_of = None
                for corpus_name, corpus_path in self.corpus_paths.items():
                    if corpus_path in path:
                        part_of = corpus_name
                        add_to_existing[part_of].append(path)
                        break
                if part_of is None:
                    no_parent.append(path)
            for corpus_name, file_paths in add_to_existing.items():
                self.get_corpus(corpus_name).add_file_paths(file_paths)
            if len(no_parent) > 0:
                # paths are expected to be contained in one and the same corpus directory
                first_path = no_parent[0]
                directory = path2parent_corpus(first_path)
                if directory is None:
                    raise ValueError(f"No parent of {first_path} has been recognized as a corpus by being a git or containing a 'metadata.tsv'. Use _.add_corpus()")
                self.add_corpus(directory)
        elif corpus_name in self.corpus_paths:
            self.get_corpus(corpus_name).add_file_paths(file_paths)
        else:
            # find the path according to the corpus_name
            first_path = resolved_paths[0]
            if corpus_name in first_path:
                while True:
                    tmp_path, last_component = os.path.split(first_path)
                    if tmp_path == first_path:
                        # reached the root
                        first_path = ''
                        break
                    if last_component == corpus_name:
                        # first_path is the corpus first_path
                        break
                    first_path = tmp_path
            else:
                first_path == ''
            if first_path == '':
                raise ValueError(f"corpus_name needs to be a folder contained in the first path, but '{corpus_name}' isn't.")
            self.add_corpus(first_path)
            # corpus = self.get_corpus(corpus_name)
            # new_view = create_view_from_parameters(only_metadata_fnames=False, exclude_review=False, paths=paths)
            # corpus.set_view(new_view)

    def color_non_chord_tones(self,
                              color_name: str = 'red',
                              view_name: Optional[str] = None,
                              force: bool = False,
                              choose: Literal['all', 'auto', 'ask'] = 'all', ) -> Dict[CorpusFnameTuple, List[FileDataframeTuple]]:
        result = {}
        for corpus_name, corpus in self.iter_corpora(view_name):
            fname2reports = corpus.color_non_chord_tones(color_name,
                                                    view_name=view_name,
                                                    force=force,
                                                    choose=choose)
            result.update({(corpus_name, fname): report for fname, report in fname2reports.items()})
        return result

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

    def compare_labels(self,
                       key: str = 'detached',
                       new_color: str = 'ms3_darkgreen',
                       old_color: str = 'ms3_darkred',
                       detached_is_newer: bool = False,
                       add_to_rna: bool = True,
                       view_name: Optional[str] = None) -> Tuple[int, int]:
        """ Compare detached labels ``key`` to the ones attached to the Score to create a diff.
        By default, the attached labels are considered as the reviewed version and labels that have changed or been added
        in comparison to the detached labels are colored in green; whereas the previous versions of changed labels are
        attached to the Score in red, just like any deleted label.

        Args:
            key: Key of the detached labels you want to compare to the ones in the score.
            new_color, old_color:
                The colors by which new and old labels are differentiated. Identical labels remain unchanged. Colors can be
                CSS colors or MuseScore colors (see :py:attr:`utils.MS3_COLORS`).
            detached_is_newer:
                Pass True if the detached labels are to be added with ``new_color`` whereas the attached changed labels
                will turn ``old_color``, as opposed to the default.
            add_to_rna:
                By default, new labels are attached to the Roman Numeral layer.
                Pass False to attach them to the chord layer instead.

        Returns:
            Number of scores in which labels have changed.
            Number of scores in which no label has chnged.
        """
        changed, unchanged = 0, 0
        for _, corpus in self.iter_corpora(view_name=view_name):
            c, u = corpus.compare_labels(key=key,
                                        new_color=new_color,
                                        old_color=old_color,
                                        detached_is_newer=detached_is_newer,
                                        add_to_rna=add_to_rna,
                                        view_name=view_name)
            changed += c
            unchanged += u
        return changed, unchanged


    def count_changed_scores(self, view_name: Optional[str] = None):
        return sum(corpus.count_changed_scores() for _, corpus in self.iter_corpora(view_name))

    def count_extensions(self,
                         view_name: Optional[str] = None,
                         per_piece: bool = False,
                         include_metadata: bool = False,
                         ):
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
        extension_counters = {corpus_name: corpus.count_extensions(view_name, include_metadata=include_metadata) for corpus_name, corpus in self.iter_corpora(view_name)}
        if per_piece:
            return {(corpus_name, fname): dict(cnt) for corpus_name, fname2cnt in extension_counters.items() for fname, cnt in fname2cnt.items()}
        return {corpus_name: dict(sum(fname2cnt.values(), Counter())) for corpus_name, fname2cnt in extension_counters.items()}


    def count_files(self,
                    detected=True,
                    parsed=True,
                    as_dict: bool = False,
                    drop_zero: bool = True,
                    view_name: Optional[str] = None) -> Union[pd.DataFrame, dict]:
        all_counts = {corpus_name: corpus._summed_file_count(types=detected, parsed=parsed, view_name=view_name) for corpus_name, corpus in self.iter_corpora(view_name=view_name)}
        counts_df = pd.DataFrame.from_dict(all_counts, orient='index', dtype='Int64')
        if drop_zero:
            empty_cols = counts_df.columns[counts_df.sum() == 0]
            counts_df = counts_df.drop(columns=empty_cols)
        if as_dict:
            return counts_df.to_dict(orient='index')
        counts_df.index.rename('corpus', inplace=True)
        return counts_df

    def count_parsed_scores(self, view_name: Optional[str] = None) -> int:
        return sum(map(len, self._get_parsed_score_files(view_name=view_name).values()))

    def count_parsed_tsvs(self, view_name: Optional[str] = None) -> int:
        return sum(map(len, self._get_parsed_tsv_files(view_name=view_name).values()))

    def count_unparsed_scores(self, view_name: Optional[str] = None) -> int:
        return sum(map(len, self._get_parsed_score_files(view_name=view_name).values()))

    def count_unparsed_tsvs(self, view_name: Optional[str] = None) -> int:
        return sum(map(len, self._get_parsed_tsv_files(view_name=view_name).values()))

    def create_missing_metadata_tsv(self,
                                    view_name: Optional[str] = None) -> None:
        for corpus_name, corpus in self.iter_corpora(view_name=view_name):
            if corpus.metadata_tsv is None:
                path = corpus.create_metadata_tsv()

    def disambiguate_facet(self,
                           facet: Facet,
                           view_name: Optional[str] = None,
                           ask_for_input=True) -> None:
        """Calls the method on every selected corpus."""
        for name, corpus in self.iter_corpora(view_name):
            corpus.disambiguate_facet(facet=facet,
                                      view_name=view_name,
                                      ask_for_input=ask_for_input)


    def extract_facets(self,
                       facets: ScoreFacets = None,
                       view_name: Optional[str] = None,
                       force: bool = False,
                       choose: Literal['auto', 'ask'] = 'auto',
                       unfold: bool = False,
                       interval_index: bool = False,
                       flat=False,
                       concatenate=True) -> Union[pd.DataFrame, Dict[CorpusFnameTuple, Union[Dict[str,  List[FileDataframeTuple]], List[FileDataframeTuple]]]]:
        return self._aggregate_corpus_data('extract_facets',
                                           facets=facets,
                                           view_name=view_name,
                                           force=force,
                                           choose=choose,
                                           unfold=unfold,
                                           interval_index=interval_index,
                                           flat=flat,
                                           concatenate=concatenate,
                                           )

    def get_all_parsed(self, facets: FacetArguments = None,
                       view_name: Optional[str] = None,
                       force: bool = False,
                       choose: Literal['all', 'auto', 'ask'] = 'all',
                       flat: bool = False,
                       include_empty=False,
                       concatenate: bool = True,
                       ) -> Union[pd.DataFrame, Dict[CorpusFnameTuple, Union[Dict[str, FileParsedTuple], List[FileParsedTuple]]]]:
        return self._aggregate_corpus_data('get_all_parsed',
                                           facets=facets,
                                           view_name=view_name,
                                           force=force,
                                           choose=choose,
                                           flat=flat,
                                           include_empty=include_empty,
                                           concatenate=concatenate)

    def get_corpus(self, name) -> Corpus:
        assert name in self.corpus_objects, f"Don't have a corpus called '{name}', only {list(self.corpus_objects.keys())}"
        return self.corpus_objects[name]


    def get_dataframes(self,
                       notes: bool = False,
                       rests: bool = False,
                       notes_and_rests: bool = False,
                       measures: bool = False,
                       events: bool = False,
                       labels: bool = False,
                       chords: bool = False,
                       expanded: bool = False,
                       form_labels: bool = False,
                       cadences: bool = False,
                       view_name: Optional[str] = None,
                       force: bool = False,
                       choose: Literal['all', 'auto', 'ask'] = 'all',
                       unfold: bool = False,
                       interval_index: bool = False,
                       flat=False,
                       include_empty: bool = False,
                      ) -> Union[pd.DataFrame, Dict[CorpusFnameTuple, Union[Dict[str,  List[FileDataframeTuple]], List[FileDataframeTuple]]]]:
        """Renamed to :meth:`get_facets`."""
        l = locals()
        facets = [facet for facet in ScoreFacet.__args__ if l[facet]]
        return self.get_facets(facets=facets,
                               view_name=view_name,
                               force=force,
                               choose=choose,
                               unfold=unfold,
                               interval_index=interval_index,
                               flat=flat,
                               include_empty=include_empty)

    def get_facet(self,
                   facet: ScoreFacet,
                   view_name: Optional[str] = None,
                   choose: Literal['auto', 'ask'] = 'auto',
                   unfold: bool = False,
                   interval_index: bool = False,
                   concatenate: bool = True,
                   ) -> Union[Dict[str, FileDataframeTuple], pd.DataFrame]:
        """Retrieves exactly one DataFrame per piece, if available."""
        return self._aggregate_corpus_data('get_facet',
                                           facet=facet,
                                           view_name=view_name,
                                           choose=choose,
                                           unfold=unfold,
                                           interval_index=interval_index,
                                           concatenate=concatenate,
                                           )

    def get_facets(self,
                   facets: ScoreFacets = None,
                   view_name: Optional[str] = None,
                   force: bool = False,
                   choose: Literal['all', 'auto', 'ask'] = 'all',
                   unfold: bool = False,
                   interval_index: bool = False,
                   flat=False,
                   include_empty=False,
                   concatenate=True,
                   ) -> Union[pd.DataFrame, Dict[CorpusFnameTuple, Union[Dict[str,  List[FileDataframeTuple]], List[FileDataframeTuple]]]]:
        return self._aggregate_corpus_data('get_facets',
                                           facets=facets,
                                           view_name=view_name,
                                           force=force,
                                           choose=choose,
                                           unfold=unfold,
                                           interval_index=interval_index,
                                           flat=flat,
                                           include_empty=include_empty,
                                           concatenate=concatenate,
                                           )

    def get_files(self, facets: FacetArguments = None,
                  view_name: Optional[str] = None,
                  parsed: bool = True,
                  unparsed: bool = True,
                  choose: Literal['all', 'auto', 'ask'] = 'all',
                  flat: bool = False,
                  include_empty=False,
                  ) -> Dict[CorpusFnameTuple, Union[FileDict, FileList]]:
        return self._aggregate_corpus_data('get_files',
                                           facets=facets,
                                           view_name=view_name,
                                           parsed=parsed,
                                           unparsed=unparsed,
                                           choose=choose,
                                           flat=flat,
                                           include_empty=include_empty
                                           )

    def get_lists(self, *args, **kwargs):
        """Deprecated method. Replaced by :meth:`get_facets`."""
        raise AttributeError(f"Method get_lists() not in use any more. Use Parse.get_facets() instead.")


    def get_piece(self, corpus_name: str, fname: str) -> Piece:
        """Returns an existing Piece object."""
        assert corpus_name in self.corpus_objects, f"'{corpus_name}' is not an existing corpus. Choose from {list(self.corpus_objects.keys())}"
        return self.corpus_objects[corpus_name].get_piece(fname)

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
            old_name = view.name
            view.name = view_name
            self._views[view_name] = view
            self.logger.info(f"New view '{view_name}' created as a copy of '{old_name}'.")
        if len(config) > 0:
            view.update_config(**config)
        return view


    def info(self, view_name: Optional[str] = None, return_str: bool = False, show_discarded: bool = False):
        """"""
        header = f"All corpora"
        header += "\n" + "-" * len(header) + "\n"

        # start info message with the names of the available views, the header, and info on the active view.
        view = self.get_view(view_name)
        view.reset_filtering_data()
        msg = available_views2str(self._views, view_name)
        msg += header
        view_info = f"View: {view}"
        if view_name is None:
            corpus_views = [corpus.get_view().name for _, corpus in self.iter_corpora(view_name=view_name)]
            if len(set(corpus_views)) > 1:
                view_info = f"This is a mixed view. Call _.info(view_name) to see a homogeneous one."
        msg += view_info + "\n\n"

        # Show info on all pieces and files included in the active view
        counts_df = self.count_files(view_name=view_name)
        if len(counts_df) == 0:
            if self.n_detected == 0:
                msg += 'No files detected. Use _.add_corpus().'
            else:
                msg += 'No files selected under the current view. You could use _.all to see everything.'
        else:
            if counts_df.isna().any().any():
                counts_df = counts_df.fillna(0).astype('int')
            additional_columns = []
            for corpus_name in counts_df.index:
                corpus = self.get_corpus(corpus_name)
                has_metadata = 'no' if corpus.metadata_tsv is None else 'yes'
                corpus_view = corpus.get_view().name
                additional_columns.append([has_metadata, corpus_view])
            additional_columns = pd.DataFrame(additional_columns, columns=[('has', 'metadata'), ('active', 'view')], index=counts_df.index)
            info_df = pd.concat([additional_columns, counts_df], axis=1)
            info_df.columns = pd.MultiIndex.from_tuples(info_df.columns)
            msg += info_df.to_string()
            n_changed_scores = self.count_changed_scores(view_name)
            if n_changed_scores > 0:
                msg += f"\n\n{n_changed_scores} scores have changed since parsing."
            filtering_report = view.filtering_report(show_discarded=show_discarded, return_str=True)
            if filtering_report != '':
                msg += '\n\n' + filtering_report
        if self.n_orphans > 0:
            msg += f"\n\nThere are {self.n_orphans} orphans that could not be attributed to any of the respective corpus's fnames"
            if show_discarded:
                msg += ':'
                for name, corpus in self:
                    if corpus.n_orphans > 0:
                        msg += f"\n\t{name}: {list(corpus.ix2orphan_file.values())}"
            else:
                msg += "."
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

    def insert_detached_labels(self,
                               view_name: Optional[str] = None,
                               key: str = 'detached',
                               staff: int = None,
                               voice: Literal[1,2,3,4] = None,
                               harmony_layer: Literal[0,1,2] = None,
                               check_for_clashes: bool = True):
        """ Attach all :py:attr:`~.annotations.Annotations` objects that are reachable via ``Score.key`` to their
        respective :py:attr:`~.score.Score`, altering the XML in memory. Calling :py:meth:`.store_scores` will output
        MuseScore files where the annotations show in the score.

        Parameters
        ----------
        key :
            Key under which the :py:attr:`~.annotations.Annotations` objects to be attached are stored in the
            :py:attr:`~.score.Score` objects. Defaults to 'detached'.
        staff : :obj:`int`, optional
            If you pass a staff ID, the labels will be attached to that staff where 1 is the upper stuff.
            By default, the staves indicated in the 'staff' column of :obj:`ms3.annotations.Annotations.df`
            will be used.
        voice : {1, 2, 3, 4}, optional
            If you pass the ID of a notational layer (where 1 is the upper voice, blue in MuseScore),
            the labels will be attached to that one.
            By default, the notational layers indicated in the 'voice' column of
            :obj:`ms3.annotations.Annotations.df` will be used.
        harmony_layer : :obj:`int`, optional
            | By default, the labels are written to the layer specified as an integer in the column ``harmony_layer``.
            | Pass an integer to select a particular layer:
            | * 0 to attach them as absolute ('guitar') chords, meaning that when opened next time,
            |   MuseScore will split and encode those beginning with a note name ( resulting in ms3-internal harmony_layer 3).
            | * 1 the labels are written into the staff's layer for Roman Numeral Analysis.
            | * 2 to have MuseScore interpret them as Nashville Numbers
        check_for_clashes : :obj:`bool`, optional
            By default, warnings are thrown when there already exists a label at a position (and in a notational
            layer) where a new one is attached. Pass False to deactivate these warnings.
        """
        reached, goal = 0, 0
        for i, (name, corpus) in enumerate(self.iter_corpora(view_name), 1):
            r, g = corpus.insert_detached_labels(view_name=view_name,
                                          key=key,
                                          staff=staff,
                                          voice=voice,
                                          harmony_layer=harmony_layer,
                                          check_for_clashes=check_for_clashes
                                          )
            reached += r
            goal += g
        if i > 1:
            self.logger.info(f"{reached}/{goal} labels successfully added.")


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

    def iter_pieces(self) -> Tuple[CorpusFnameTuple, Piece]:
        for corpus_name, corpus in self:
            for fname, piece in corpus:
                yield (corpus_name, fname), piece

    def keys(self) -> List[str]:
        """Return the names of all corpus objects."""
        return list(self.corpus_objects.keys())

    def load_facet_into_scores(self,
                               facet: AnnotationsFacet,
                               view_name: Optional[str] = None,
                               force: bool = False,
                               choose: Literal['auto', 'ask'] = 'auto',
                               git_revision: Optional[str] = None,
                               key: str = 'detached',
                               infer: bool = True,
                               **cols) -> Dict[str, int]:
        result = {}
        for corpus_name, corpus in self.iter_corpora(view_name):
            result[corpus_name] = corpus.load_facet_into_scores(facet=facet,
                                                                view_name=view_name,
                                                                force=force,
                                                                choose=choose,
                                                                git_revision=git_revision,
                                                                key=key,
                                                                infer=infer,
                                                                **cols)
        return result

    def load_ignored_warnings(self, path: str) -> None:
        """ Adds a filters to all loggers included in a IGNORED_WARNINGS file.

        Args:
            path: Path of the IGNORED_WARNINGS file.
        """
        for _, corpus in self:
            _ = corpus.load_ignored_warnings(path)

    def set_view(self, active: View = None, **views: View):
        """Register one or several view_name=View pairs."""
        if active is not None:
            new_name = active.name
            if new_name in self._views and active != self._views[new_name]:
                self.logger.info(f"The existing view called '{new_name}' has been overwritten")
                del(self._views[new_name])
            old_view = self._views[None]
            self._views[old_view.name] = old_view
            self._views[None] = active
        for view_name, view in views.items():
            if view.name is None:
                view.name = view_name
            self._views[view_name] = view
        for corpus_name, corpus in self:
            if active is not None and active.check_token('corpus', corpus_name):
                corpus.set_view(active)
            for view_name, view in views.items():
                if view.check_token('corpus', corpus_name):
                    corpus.set_view(**{view_name: view})

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

    def update_metadata_tsv_from_parsed_scores(self,
                                               root_dir: Optional[str] = None,
                                               suffix: str = '',
                                               markdown_file: Optional[str] = "README.md",
                                               view_name: Optional[str] = None,
                                               ) -> List[str]:
        """
        Gathers the metadata from parsed and currently selected scores and updates 'metadata.tsv' with the information.

        Args:
            root_dir: In case you want to output the metadata to folder different from :attr:`corpus_path`.
            suffix:
                Added to the filename: 'metadata{suffix}.tsv'. Defaults to ''. Metadata files with suffix may be
                used to store views with particular subselections of pieces.
            markdown_file:
                By default, a subset of metadata columns will be written to 'README.md' in the same folder as the TSV file.
                If the file exists, it will be scanned for a line containing the string '# Overview' and overwritten
                from that line onwards.
            view_name:
                The view under which you want to update metadata from the selected parsed files. Defaults to None,
                i.e. the active view.

        Returns:
            The file paths to which metadata was written.
        """
        metadata_paths = []
        for _, corpus in self.iter_corpora():
            paths = corpus.update_metadata_tsv_from_parsed_scores(root_dir=root_dir,
                                                                  suffix=suffix,
                                                                  markdown_file=markdown_file,
                                                                  view_name=view_name)
            metadata_paths.extend(paths)
        return metadata_paths

    def update_score_metadata_from_tsv(self,
                                       view_name: Optional[str] = None,
                                       force: bool = False,
                                       choose: Literal['all', 'auto', 'ask'] = 'all',
                                       write_empty_values: bool = False,
                                       remove_unused_fields: bool = False,
                                       write_text_fields: bool = False
                                       ) -> List[File]:
        """ Update metadata fields of parsed scores with the values from the corresponding row in metadata.tsv.

        Args:
            view_name:
            force:
            choose:
            write_empty_values:
                If set to True, existing values are overwritten even if the new value is empty, in which case the field
                will be set to ''.
            remove_unused_fields:
                If set to True, all non-default fields that are not among the columns of metadata.tsv (anymore) are removed.
            write_text_fields:
                If set to True, ms3 will write updated values from the columns ``title_text``, ``subtitle_text``, ``composer_text``,
                ``lyricist_text``, and ``part_name_text`` into the score headers.

        Returns:
            List of File objects of those scores of which the XML structure has been modified.
        """
        updated_scores = []
        for _, corpus in self.iter_corpora(view_name):
            modified = corpus.update_score_metadata_from_tsv(view_name=view_name,
                                                             force=force,
                                                             choose=choose,
                                                             write_empty_values=write_empty_values,
                                                             remove_unused_fields=remove_unused_fields,
                                                             write_text_fields=write_text_fields)
            updated_scores.extend(modified)
        return updated_scores

    def update_scores(self,
                      root_dir: Optional[str] = None,
                      folder: str = '.',
                      suffix: str = '',
                      overwrite: bool = False) -> List[str]:
        """ Update scores created with an older MuseScore version to the latest MuseScore 3 version.

                Args:
                    root_dir:
                        In case you want to create output paths for the updated MuseScore files based on a folder different
                        from :attr:`corpus_path`.
                    folder:
                        * The default '.' has the updated scores written to the same directory as the old ones, effectively
                          overwriting them if ``root_dir`` is None.
                        * If ``folder`` is None, the files will be written to ``{root_dir}/scores/``.
                        * If ``folder`` is an absolute path, ``root_dir`` will be ignored.
                        * If ``folder`` is a relative path starting with a dot ``.`` the relative path is appended to the file's subdir.
                          For example, ``..\scores`` will resolve to a sibling directory of the one where the ``file`` is located.
                        * If ``folder`` is a relative path that does not begin with a dot ``.``, it will be appended to the
                          ``root_dir``.
                    suffix: String to append to the file names of the updated files, e.g. '_updated'.
                    overwrite: By default, existing files are not overwritten. Pass True to allow this.

                Returns:
                    A list of all up-to-date paths, whether they had to be converted or were already in the latest version.
                """
        up2date_paths = []
        for _, corpus in self.iter_corpora():
            paths = corpus.update_scores(root_dir=root_dir,
                                         folder=folder,
                                         suffix=suffix,
                                         overwrite=overwrite)
            up2date_paths.extend(paths)
        return up2date_paths


    def update_tsvs_on_disk(self,
                       facets: ScoreFacets = 'tsv',
                       view_name: Optional[str] = None,
                       force: bool = False,
                       choose: Literal['auto', 'ask'] = 'auto',
                       ) -> List[str]:
        """
        Update existing TSV files corresponding to one or several facets with information freshly extracted from a parsed
        score, but only if the contents are identical. Otherwise, the existing TSV file is not overwritten and the
        differences are displayed in a log warning. The purpose is to safely update the format of existing TSV files,
        (for instance with respect to column order) making sure that the content doesn't change.

        Args:
            facets:
            view_name:
            force:
                By default, only TSV files that have already been parsed are updated. Set to True in order to
                force-parse for each facet one of the TSV files included in the given view, if necessary.
            choose:

        Returns:
            List of paths that have been overwritten.
        """
        paths = []
        for _, corpus in self.iter_corpora(view_name=view_name):
            paths.extend(corpus.update_tsvs_on_disk(facets=facets,
                                                   view_name=view_name,
                                                   force=force,
                                                   choose=choose))
        return paths

    def _aggregate_corpus_data(self,
                               method,
                               view_name=None,
                               concatenate=False,
                               **kwargs
                               ):
        result = {}
        for corpus_name, corpus in self.iter_corpora(view_name):
            corpus_method = getattr(corpus, method)
            if method == 'get_facet':
                kwargs['concatenate'] = False
            corpus_result = corpus_method(view_name=view_name, **kwargs)
            for fname, piece_result in corpus_result.items():
                if method == 'get_facet':
                    piece_result = [piece_result]
                result[(corpus_name, fname)] = piece_result
        if concatenate:
            keys, dataframes = [], []
            flat = 'flat' not in kwargs or kwargs['flat']
            if flat:
                add_index_level = any(len(piece_result) > 1 for piece_result in result.values())
            else:
                add_index_level = any(len(file_dataframe_tuples) > 1 for piece_result in result.values() for file_dataframe_tuples in piece_result.values())
            for corpus_fname, piece_result in result.items():
                if flat:
                    n_tuples = len(piece_result)
                    if n_tuples == 0:
                        continue
                    keys.append(corpus_fname)
                    if n_tuples == 1:
                        if add_index_level:
                            file, df = piece_result[0]
                            df = pd.concat([df], keys=[file.rel_path])
                            dataframes.append(df)
                        else:
                            dataframes.append(piece_result[0][1])
                    else:
                        files, dfs = list(zip(*piece_result))
                        ix_level = [file.rel_path for file in files]
                        concat = pd.concat(dfs, keys=ix_level)
                        dataframes.append(concat)
                else:
                    for facet, file_dataframe_tuples in piece_result.items():
                        n_tuples = len(file_dataframe_tuples)
                        if n_tuples == 0:
                            continue
                        keys.append(corpus_fname + (facet,))
                        if n_tuples == 1:
                            if add_index_level:
                                file, df = file_dataframe_tuples[0]
                                df = pd.concat([df], keys=[file.rel_path])
                                dataframes.append(df)
                            else:
                                dataframes.append(file_dataframe_tuples[0][1])
                        else:
                            files, dfs = list(zip(*file_dataframe_tuples))
                            ix_level = [file.rel_path for file in files]
                            concat = pd.concat(dfs, keys=ix_level)
                            dataframes.append(concat)
            if len(dataframes) > 0:
                try:
                    result = pd.concat(dataframes, keys=keys)
                except ValueError:
                    n_levels = [df.columns.nlevels for df in dataframes]
                    if len(set(n_levels)) > 1:
                        # this error might come from various form label dataframes with varying numbers of column levels
                        adapted_dataframes = []
                        for df in dataframes:
                            if df.columns.nlevels == 2:
                                adapted_dataframes.append(df)
                            else:
                                loc = df.columns.get_loc('form_label')
                                adapted_dataframes.append(pd.concat([df.iloc[:, :loc], df.iloc[:, loc:]], keys=['', 'a'], axis=1))
                        result = pd.concat(adapted_dataframes, keys=keys)
                    else:
                        raise
                nlevels = result.index.nlevels
                level_names = ['corpus', 'fname']
                if not flat:
                    level_names.append('facet')
                if len(level_names) < nlevels - 1:
                    level_names.append('ix')
                level_names.append('i')
                result.index.rename(level_names, inplace=True)
            else:
                return pd.DataFrame()
        return result



    def _get_parsed_score_files(self, view_name: Optional[str] = None) -> Dict[CorpusFnameTuple, FileList]:
        result = {}
        for corpus_name, corpus in self.iter_corpora(view_name=view_name):
            fname2files = corpus.get_files('scores', view_name=view_name, unparsed=False, flat=True)
            result[corpus_name] = sum(fname2files.values(), [])
        return result


    def _get_unparsed_score_files(self, view_name: Optional[str] = None) -> Dict[CorpusFnameTuple, FileList]:
        result = {}
        for corpus_name, corpus in self.iter_corpora(view_name=view_name):
            fname2files = corpus.get_files('scores', view_name=view_name, parsed=False, flat=True)
            result[corpus_name] = sum(fname2files.values(), [])
        return result

    def _get_parsed_tsv_files(self, view_name: Optional[str] = None, flat: bool = True) -> Dict[CorpusFnameTuple, Union[FileDict, FileList]]:
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

    def _get_unparsed_tsv_files(self, view_name: Optional[str] = None, flat: bool = True) -> Dict[CorpusFnameTuple, Union[FileDict, FileList]]:
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


    def __getattr__(self, view_name) -> View:
        if view_name in self.view_names:
            if view_name != self.view_name:
                self.switch_view(view_name, show_info=False)
            return self
        else:
            raise AttributeError(f"'{view_name}' is not an existing view. Use _.get_view('{view_name}') to create it.")


#
#
#     def count_annotation_layers(self, keys=None, which='attached', per_key=False):
#         """ Counts the labels for each annotation layer defined as (staff, voice, harmony_layer).
#         By default, only labels attached to a score are counted.
#
#         Parameters
#         ----------
#         keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
#             Key(s) for which to count annotation layers.  By default, all keys are selected.
#         which : {'attached', 'detached', 'tsv'}, optional
#             'attached': Counts layers from annotations attached to a score.
#             'detached': Counts layers from annotations that are in a Score object, but detached from the score.
#             'tsv': Counts layers from Annotation objects that have been loaded from or into annotation tables.
#         per_key : :obj:`bool`, optional
#             If set to True, the results are returned as a dict {key: Counter}, otherwise the counts are summed up in one Counter.
#             If ``which='detached'``, the keys are keys from Score objects, otherwise they are keys from this Parse object.
#
#         Returns
#         -------
#         :obj:`dict` or :obj:`collections.Counter`
#             By default, the function returns a Counter of labels for every annotation layer (staff, voice, harmony_layer)
#             If ``per_key`` is set to True, a dictionary {key: Counter} is returned, separating the counts.
#         """
#         res_dict = defaultdict(Counter)
#
#         if which == 'detached':
#             for id in self._iterids(keys, only_detached_annotations=True):
#                 for key, annotations in self._parsed_mscx[id]._detached_annotations.items():
#                     if key != 'annotations':
#                         _, layers = annotations.annotation_layers
#                         layers_dict = {tuple(None if pd.isnull(e) else e for e in t): count for t, count in
#                                        layers.to_dict().items()}
#                         res_dict[key].update(layers_dict)
#         elif which in ['attached', 'tsv']:
#             for key, i in self._iterids(keys):
#                 if (key, i) in self._annotations:
#                     ext = self.fexts[key][i]
#                     if (which == 'attached' and ext == '.mscx') or (which == 'tsv' and ext != '.mscx'):
#                         _, layers = self._annotations[(key, i)].annotation_layers
#                         layers_dict = {tuple(None if pd.isnull(e) else e for e in t): count for t, count in
#                                        layers.to_dict().items()}
#                         res_dict[key].update(layers_dict)
#         else:
#             self.logger.error(f"Parameter 'which' needs to be one of {{'attached', 'detached', 'tsv'}}, not {which}.")
#             return {} if per_key else pd.Series()
#
#
#         def make_series(counts):
#             if len(counts) == 0:
#                 return pd.Series()
#             data = counts.values()
#             ks = list(counts.keys())
#             #levels = len(ks[0])
#             names = ['staff', 'voice', 'harmony_layer', 'color'] #<[:levels]
#             ix = pd.MultiIndex.from_tuples(ks, names=names)
#             return pd.Series(data, ix)
#
#         if per_key:
#             res = {k: make_series(v) for k, v in res_dict.items()}
#         else:
#             res = make_series(sum(res_dict.values(), Counter()))
#         if len(res) == 0:
#             self.logger.info("No annotations found. Maybe no scores have been parsed using parse_scores()?")
#         return res
#
#
#
#
#
#
#     def count_labels(self, keys=None, per_key=False):
#         """ Count label types.
#
#         Parameters
#         ----------
#         keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
#             Key(s) for which to count label types.  By default, all keys are selected.
#         per_key : :obj:`bool`, optional
#             If set to True, the results are returned as a dict {key: Counter},
#             otherwise the counts are summed up in one Counter.
#
#         Returns
#         -------
#         :obj:`dict` or :obj:`collections.Counter`
#             By default, the function returns a Counter of label types.
#             If ``per_key`` is set to True, a dictionary {key: Counter} is returned, separating the counts.
#         """
#         annotated = [id for id in self._iterids(keys) if id in self._annotations]
#         res_dict = defaultdict(Counter)
#         for key, i in annotated:
#             res_dict[key].update(self._annotations[(key, i)].harmony_layer_counts)
#         if len(res_dict) == 0:
#             if len(self._parsed_mscx) == 0:
#                 self.logger.error("No scores have been parsed so far. Use parse_scores().")
#             else:
#                 self.logger.info("None of the scores contain annotations.")
#         if per_key:
#             return {k: dict(v) for k, v in res_dict.items()}
#         return dict(sum(res_dict.values(), Counter()))
#

    def detach_labels(self,
                      view_name: Optional[str] = None,
                      force: bool = False,
                      choose: Literal['auto', 'ask'] = 'auto',
                      key: str = 'removed',
                      staff: int = None,
                      voice: Literal[1, 2, 3, 4] = None,
                      harmony_layer: Literal[0, 1, 2, 3] = None,
                      delete: bool = True):
        for name, corpus in self.iter_corpora(view_name):
            corpus.detach_labels(view_name=view_name,
                                 force=force,
                                 choose=choose,
                                 key=key,
                                 staff=staff,
                                 voice=voice,
                                 harmony_layer=harmony_layer,
                                 delete=delete)



    def score_metadata(self, view_name: Optional[str] = None) -> pd.DataFrame:
        metadata_dfs = {corpus_name: corpus.score_metadata(view_name=view_name) for corpus_name, corpus in self.iter_corpora(view_name=view_name)}
        metadata = pd.concat(metadata_dfs.values(), keys=metadata_dfs.keys(), names=['corpus', 'fname'])
        return metadata

    def metadata(self,
                 view_name: Optional[str] = None,
                 choose: Optional[Literal['auto', 'ask']] = None) -> pd.DataFrame:
        metadata_dfs = {corpus_name: corpus.metadata(view_name=view_name, choose=choose)
                        for corpus_name, corpus in self.iter_corpora(view_name=view_name)}
        metadata = pd.concat(metadata_dfs.values(), keys=metadata_dfs.keys(), names=['corpus', 'fname'])
        return metadata

    def metadata_tsv(self, view_name: Optional[str] = None) -> pd.DataFrame:
        """Concatenates the 'metadata.tsv' (as they come) files for all corpora with a [corpus, fname] MultiIndex. If
        you need metadata that filters out fnames according to the current view, use :meth:`metadata`.
        """
        metadata_dfs = {corpus_name: enforce_fname_index_for_metadata(corpus.metadata_tsv)
                        for corpus_name, corpus in self.iter_corpora(view_name=view_name)
                        if corpus.metadata_tsv is not None
                        }
        metadata = pd.concat(metadata_dfs.values(), keys=metadata_dfs.keys(), names=['corpus', 'fname'])
        return metadata



    def store_extracted_facets(self,
                               view_name: Optional[str] = None,
                               root_dir: Optional[str] = None,
                               measures_folder: Optional[str] = None, measures_suffix: str = '',
                               notes_folder: Optional[str] = None, notes_suffix: str = '',
                               rests_folder: Optional[str] = None, rests_suffix: str = '',
                               notes_and_rests_folder: Optional[str] = None, notes_and_rests_suffix: str = '',
                               labels_folder: Optional[str] = None, labels_suffix: str = '',
                               expanded_folder: Optional[str] = None, expanded_suffix: str = '',
                               form_labels_folder: Optional[str] = None, form_labels_suffix: str = '',
                               cadences_folder: Optional[str] = None, cadences_suffix: str = '',
                               events_folder: Optional[str] = None, events_suffix: str = '',
                               chords_folder: Optional[str] = None, chords_suffix: str = '',
                               metadata_suffix: Optional[str] = None, markdown: bool = True,
                               simulate: bool = False,
                               unfold: bool = False,
                               interval_index: bool = False,
                               silence_label_warnings: bool = False):
        """  Store facets extracted from parsed scores as TSV files.

        Args:
            view_name:
            root_dir:
                ('measures', 'notes', 'rests', 'notes_and_rests', 'labels', 'expanded', 'form_labels', 'cadences', 'events', 'chords')

            measures_folder, notes_folder, rests_folder, notes_and_rests_folder, labels_folder, expanded_folder, form_labels_folder, cadences_folder, events_folder, chords_folder:
                Specify directory where to store the corresponding TSV files.
            measures_suffix, notes_suffix, rests_suffix, notes_and_rests_suffix, labels_suffix, expanded_suffix, form_labels_suffix, cadences_suffix, events_suffix, chords_suffix:
                Optionally specify suffixes appended to the TSVs' file names. If ``unfold=True`` the suffixes default to ``_unfolded``.
            metadata_suffix:
                Specify a suffix to update the 'metadata{suffix}.tsv' file for each corpus. For the main file, pass ''
            markdown:
                By default, when ``metadata_path`` is specified, a markdown file called ``README.md`` containing
                the columns [file_name, measures, labels, standard, annotators, reviewers] is created. If it exists already,
                this table will be appended or overwritten after the heading ``# Overview``.
            simulate:
            unfold:
                By default, repetitions are not unfolded. Pass True to duplicate values so that they correspond to a full
                playthrough, including correct positioning of first and second endings.
            interval_index:
            silence_label_warnings:

        Returns:

        """
        for corpus_name, corpus in self.iter_corpora(view_name=view_name):
            corpus.store_extracted_facets(view_name=view_name, root_dir=root_dir, measures_folder=measures_folder, measures_suffix=measures_suffix, notes_folder=notes_folder, notes_suffix=notes_suffix,
                                          rests_folder=rests_folder, rests_suffix=rests_suffix, notes_and_rests_folder=notes_and_rests_folder,
                                          notes_and_rests_suffix=notes_and_rests_suffix, labels_folder=labels_folder, labels_suffix=labels_suffix, expanded_folder=expanded_folder,
                                          expanded_suffix=expanded_suffix, form_labels_folder=form_labels_folder, form_labels_suffix=form_labels_suffix,
                                          cadences_folder=cadences_folder, cadences_suffix=cadences_suffix, events_folder=events_folder, events_suffix=events_suffix,
                                          chords_folder=chords_folder, chords_suffix=chords_suffix, metadata_suffix=metadata_suffix, markdown=markdown, simulate=simulate,
                                          unfold=unfold, interval_index=interval_index, silence_label_warnings=silence_label_warnings)

    def store_parsed_scores(self,
                            view_name: Optional[str] = None,
                            only_changed: bool = True,
                            root_dir: Optional[str] = None,
                            folder: str = '.',
                            suffix: str = '',
                            overwrite: bool = False,
                            simulate=False) -> Dict[str, List[str]]:
        """ Stores all parsed scores under this view as MuseScore 3 files.

        Args:
            view_name: Name of another view if another than the current one is to be used.
            only_changed:
                By default, only scores that have been modified since parsing are written. Set to False to store
                all scores regardless.
            root_dir: Directory where to re-build the sub-directory tree of the :obj:`Corpus` in question.
            folder:
                Different behaviours are available. Note that only the third option ensures that file paths are distinct for
                files that have identical fnames but are located in different subdirectories of the same corpus.

                * If ``folder`` is None (default), the files' type will be appended to the ``root_dir``.
                * If ``folder`` is an absolute path, ``root_dir`` will be ignored.
                * If ``folder`` is a relative path starting with a dot ``.`` the relative path is appended to the file's subdir.
                  For example, ``..\notes`` will resolve to a sibling directory of the one where the ``file`` is located.
                * If ``folder`` is a relative path that does not begin with a dot ``.``, it will be appended to the
                  ``root_dir``.
            suffix: Suffix to append to the original file name.
            overwrite: Pass True to overwrite existing files.
            simulate: Set to True if no files are to be written.

        Returns:
            Paths of the stored files.
        """
        paths = {}
        for corpus_name, corpus in self.iter_corpora(view_name):
            paths[corpus_name] = corpus.store_parsed_scores(view_name=view_name,
                                                    only_changed=only_changed,
                                                    root_dir=root_dir,
                                                    folder=folder,
                                                    suffix=suffix,
                                                    overwrite=overwrite,
                                                    simulate=simulate)
        return paths

    def parse(self, view_name=None, level=None, parallel=True, only_new=True, labels_cfg={}, cols={}, infer_types=None, **kwargs):
        """ Shorthand for executing parse_scores and parse_tsv at a time.
        Args:
            view_name:
        """
        self.parse_scores(level=level, parallel=parallel, only_new=only_new, labels_cfg=labels_cfg, view_name=view_name)
        self.parse_tsv(view_name=view_name, level=level, cols=cols, infer_types=infer_types, only_new=only_new, **kwargs)

    def parse_mscx(self, *args, **kwargs):
        """Deprecated method. Replaced by :meth:`parse_scores`."""
        raise AttributeError(f"Method not in use any more. Use Parse.parse_scores().")

    def parse_scores(self,
                     level: str = None,
                     parallel: bool = True,
                     only_new: bool = True,
                     labels_cfg: dict = {},
                     view_name: str = None,
                     choose: Literal['all', 'auto', 'ask'] = 'all'):
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
            during parsing. Pass False to parse one score after the other, which uses more memory but will allow
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
            corpus.parse_scores(level=level,
                                parallel=parallel,
                                only_new=only_new,
                                labels_cfg=labels_cfg,
                                view_name=view_name,
                                choose=choose)



    def parse_tsv(self,
                  view_name=None,
                  level=None,
                  cols={},
                  infer_types=None,
                  only_new=True,
                  choose: Literal['all', 'auto', 'ask'] = 'all',
                  **kwargs):
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
            corpus.parse_tsv(view_name=view_name,
                             cols=cols,
                             infer_types=infer_types,
                             only_new=only_new,
                             choose=choose,
                             **kwargs)


    # def _parse_tsv_from_git_revision(self, tsv_id, revision_specifier):
    #     """ Takes the ID of an annotation table, and parses the same file's previous version at ``revision_specifier``.
    #
    #     Parameters
    #     ----------
    #     tsv_id
    #         ID of the TSV file containing an annotation table, for which to parse a previous version.
    #     revision_specifier : :obj:`str`
    #         String used by git.Repo.commit() to find the desired git revision.
    #         Can be a long or short SHA, git tag, branch name, or relative specifier such as 'HEAD~1'.
    #
    #     Returns
    #     -------
    #     ID
    #         (key, i) of the newly added annotation table.
    #     """
    #     key, i = tsv_id
    #     corpus_path = self.corpus_paths[key]
    #     try:
    #         repo = Repo(corpus_path, search_parent_directories=True)
    #     except InvalidGitRepositoryError:
    #         self.logger.error(f"{corpus_path} seems not to be (part of) a git repository.")
    #         return
    #     try:
    #         git_repo = repo.remote("origin").url
    #     except ValueError:
    #         git_repo = os.path.basename()
    #     try:
    #         commit = repo.commit(revision_specifier)
    #         commit_sha = commit.hexsha
    #         short_sha = commit_sha[:7]
    #         commit_info = f"{short_sha} with message '{commit.message}'"
    #     except BadName:
    #         self.logger.error(f"{revision_specifier} does not resolve to a commit for repo {git_repo}.")
    #         return
    #     tsv_type = self._tsv_types[tsv_id]
    #     tsv_path = self.full_paths[key][i]
    #     rel_path = os.path.relpath(tsv_path, corpus_path)
    #     new_directory = os.path.join(corpus_path, short_sha)
    #     new_path = os.path.join(new_directory, self.files[key][i])
    #     if new_path in self.full_paths[key]:
    #         existing_i = self.full_paths[key].index(new_path)
    #         existing_tsv_type = self._tsv_types[(key, existing_i)]
    #         if tsv_type == existing_tsv_type:
    #             self.logger.error(f"Had already loaded a {tsv_type} table for commit {commit_info} of repo {git_repo}.")
    #             return
    #     if not tsv_type in ('labels', 'expanded'):
    #         raise NotImplementedError(f"Currently, only annotations are to be loaded from a git revision but {rel_path} is a {tsv_type}.")
    #     try:
    #         targetfile = commit.tree / rel_path
    #     except KeyError:
    #         # if the file was not found, try and see if at the time of the git revision the folder was still called 'harmonies'
    #         if tsv_type == 'expanded':
    #             folder, tsv_name = os.path.split(rel_path)
    #             if folder != 'harmonies':
    #                 old_rel_path = os.path.join('harmonies', tsv_name)
    #                 try:
    #                     targetfile = commit.tree / old_rel_path
    #                     self.logger.debug(f"{rel_path} did not exist at commit {commit_info}, using {old_rel_path} instead.")
    #                     rel_path = old_rel_path
    #                 except KeyError:
    #                     self.logger.error(f"Neither {rel_path} nor its older version {old_rel_path} existed at commit {commit_info}.")
    #                     return
    #         else:
    #             self.logger.error(f"{rel_path} did not exist at commit {commit_info}.")
    #             return
    #     self.logger.info(f"Successfully loaded {rel_path} from {commit_info}.")
    #     try:
    #         with io.BytesIO(targetfile.data_stream.read()) as f:
    #             df = load_tsv(f)
    #     except Exception:
    #         self.logger.error(f"Parsing {rel_path} @ commit {commit_info} failed with the following error:\n{sys.exc_info()[1]}")
    #         return
    #     new_id = self._handle_path(new_path, key, skip_checks=True)
    #     self._parsed_tsv[new_id] = df
    #     self._dataframes[tsv_type][new_id] = df
    #     self._tsv_types[new_id] = tsv_type
    #     logger_cfg = dict(self.logger_cfg)
    #     logger_cfg['name'] = self.logger_names[(key, i)]
    #     if tsv_id in self._annotations:
    #         anno_obj = self._annotations[tsv_id] # get Annotation object's settings from the existing one
    #         cols = anno_obj.cols
    #         infer_types = anno_obj.regex_dict
    #     else:
    #         cols = dict(label='label')
    #         infer_types = None
    #     self._annotations[new_id] = Annotations(df=df, cols=cols, infer_types=infer_types,
    #                                         **logger_cfg)
    #     self.logger.debug(
    #         f"{rel_path} successfully parsed from commit {short_sha}.")
    #     return new_id
    #
    #
    # def pieces(self, parsed_only=False):
    #     pieces_dfs = [self[k].pieces(parsed_only=parsed_only) for k in self.keys()]
    #     result = pd.concat(pieces_dfs, keys=self.keys())
    #     result.index.names = ['key', 'metadata_row']
    #     return result

    #
    # def store_scores(self, keys=None, ids=None, root_dir=None, folder='.', suffix='', overwrite=False, simulate=False):
    #     """ Stores the parsed MuseScore files in their current state, e.g. after detaching or attaching annotations.
    #
    #     Parameters
    #     ----------
    #     keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
    #         Key(s) for which to count file extensions.  By default, all keys are selected.
    #     ids : :obj:`~collections.abc.Collection`
    #         If you pass a collection of IDs, ``keys`` is ignored and only the selected extensions are counted.
    #     root_dir : :obj:`str`, optional
    #         Defaults to None, meaning that the original root directory is used that was added to the Parse object.
    #         Otherwise, pass a directory to rebuild the original substructure. If ``folder`` is an absolute path,
    #         ``root_dir`` is ignored.
    #     folder : :obj:`str`
    #         Where to store the file. Can be relative to ``root_dir`` or absolute, in which case ``root_dir`` is ignored.
    #         If ``folder`` is relative, the behaviour depends on whether it starts with a dot ``.`` or not: If it does,
    #         the folder is created at every end point of the relative tree structure under ``root_dir``. If it doesn't,
    #         it is created only once, relative to ``root_dir``, and the relative tree structure is build below.
    #     suffix : :obj:`str`, optional
    #         Suffix to append to the original file name.
    #     overwrite : :obj:`bool`, optional
    #         Pass True to overwrite existing files.
    #     simulate : :obj:`bool`, optional
    #         Pass True if no files are to be written.
    #
    #     Returns
    #     -------
    #
    #     """
    #     if ids is None:
    #         ids = [id for id in self._iterids(keys) if id in self._parsed_mscx]
    #     paths = []
    #     for key, i in ids:
    #         new_path = self._store_scores(key=key, i=i, folder=folder, suffix=suffix, root_dir=root_dir, overwrite=overwrite, simulate=simulate)
    #         if new_path is not None:
    #             if new_path in paths:
    #                 modus = 'would have' if simulate else 'has'
    #                 self.logger.info(f"The score at {new_path} {modus} been overwritten.")
    #             else:
    #                 paths.append(new_path)
    #     if simulate:
    #         return list(set(paths))
    #
    #

    #
    #
    # def _collect_annotations_objects_references(self, keys=None, ids=None):
    #     """ Updates the dictionary self._annotations with all parsed Scores that have labels attached (or not any more). """
    #     if ids is None:
    #         ids = list(self._iterids(keys, only_parsed_mscx=True))
    #     updated = {}
    #     for id in ids:
    #         if id in self._parsed_mscx:
    #             score = self._parsed_mscx[id]
    #             if score is not None:
    #                 if 'annotations' in score:
    #                     updated[id] = score.annotations
    #                 elif id in self._annotations:
    #                     del (self._annotations[id])
    #             else:
    #                 del (self._parsed_mscx[id])
    #     self._annotations.update(updated)
    #
    #
    # def _iterids(self, keys=None, only_parsed_mscx=False, only_parsed_tsv=False, only_attached_annotations=False, only_detached_annotations=False):
    #     """Iterator through IDs for a given set of keys.
    #
    #     Parameters
    #     ----------
    #     keys
    #     only_parsed_mscx
    #     only_attached_annotations
    #     only_detached_annotations
    #
    #     Yields
    #     ------
    #     :obj:`tuple`
    #         (str, int)
    #
    #     """
    #     keys = self._treat_key_param(keys)
    #     for key in sorted(keys):
    #         for id in make_id_tuples(key, len(self.fnames[key])):
    #             if only_parsed_mscx  or only_attached_annotations or only_detached_annotations:
    #                 if id not in self._parsed_mscx:
    #                     continue
    #                 if only_attached_annotations:
    #                     if 'annotations' in self._parsed_mscx[id]:
    #                         pass
    #                     else:
    #                         continue
    #                 elif only_detached_annotations:
    #                     if self._parsed_mscx[id].has_detached_annotations:
    #                         pass
    #                     else:
    #                         continue
    #             elif only_parsed_tsv:
    #                 if id in self._parsed_tsv:
    #                     pass
    #                 else:
    #                     continue
    #
    #             yield id
    #
    # def _iter_subdir_selectors(self, keys=None, ids=None):
    #     """ Iterate through the specified ids grouped by subdirs.
    #
    #     Yields
    #     ------
    #     :obj:`tuple`
    #         (key: str, subdir: str, ixs: list) tuples. IDs can be created by combining key with each i in ixs.
    #         The yielded ``ixs`` are typically used as parameter for ``.utils.iter_selection``.
    #
    #     """
    #     grouped_ids = self._make_grouped_ids(keys, ids)
    #     for k, ixs in grouped_ids.items():
    #         subdirs = self.subdirs[k]
    #         for subdir in sorted(set(iter_selection(subdirs, ixs))):
    #             yield k, subdir, [i for i in ixs if subdirs[i] == subdir]
    #
    #
    #
    #
    # def _parse(self, key, i, logger_cfg={}, labels_cfg={}, read_only=False):
    #     """Performs a single parse and returns the resulting Score object or None."""
    #     path = self.full_paths[key][i]
    #     file = self.files[key][i]
    #     self.logger.debug(f"Attempting to parse {file}")
    #     try:
    #         logger_cfg['name'] = self.logger_names[(key, i)]
    #         score = Score(path, read_only=read_only, labels_cfg=labels_cfg, logger_cfg=logger_cfg, ms=self.ms)
    #         if score is None:
    #             self.logger.debug(f"Encountered errors when parsing {file}")
    #         else:
    #             self.logger.debug(f"Successfully parsed {file}")
    #         return score
    #     except (KeyboardInterrupt, SystemExit):
    #         self.logger.info("Process aborted.")
    #         raise
    #     except:
    #         self.logger.error(f"Unable to parse {path} due to the following exception:\n" + traceback.format_exc())
    #         return None
    #
    #
    # def _score_ids(self, keys=None, score_extensions=None, native=True, convertible=True, opposite=False):
    #     """ Return IDs of all detected scores with particular file extensions, or all others if ``opposite==True``.
    #
    #     Parameters
    #     ----------
    #     keys : :obj:`str` or :obj:`collections.abc.Iterable`, optional
    #         Only get IDs for particular keys.
    #     score_extensions : :obj:`collections.abc.Collection`, optional
    #         Get IDs for files with the given extensions (each starting with a dot). If this parameter is defined,
    #         ``native```and ``convertible`` are being ignored.
    #     native : :obj:`bool`, optional
    #         If ``score_extensions`` is not set, ``native=True`` selects all scores that ms3 can parse without using
    #         a MuseScore 3 executable.
    #     convertible : :obj:`bool`, optional
    #         If ``score_extensions`` is not set, ``convertible=True`` selects all scores that ms3 can parse as long as
    #         a MuseScore 3 executable is defined.
    #     opposite : :obj:`bool`, optional
    #         Pass True if you want to get the IDs of all the scores that do NOT have the specified extensions.
    #
    #     Returns
    #     -------
    #     :obj:`list`
    #         A list of IDs.
    #
    #     """
    #     if score_extensions is None:
    #         score_extensions = []
    #         if native:
    #             score_extensions.extend(Score.native_formats)
    #         if convertible:
    #             score_extensions.extend(Score.convertible_formats)
    #     if opposite:
    #         return [(k, i) for k, i in self._iterids(keys) if self.fexts[k][i][1:].lower() not in score_extensions]
    #     return [(k, i) for k, i in self._iterids(keys) if self.fexts[k][i][1:].lower() in score_extensions]
    #
    #
    #
    # def _store_scores(self, key, i, folder, suffix='', root_dir=None, overwrite=False, simulate=False):
    #     """ Creates a MuseScore 3 file from the Score object at the given ID (key, i).
    #
    #     Parameters
    #     ----------
    #     key, i : (:obj:`str`, :obj:`int`)
    #         ID from which to construct the new path and filename.
    #     root_dir : :obj:`str`, optional
    #         Defaults to None, meaning that the original root directory is used that was added to the Parse object.
    #         Otherwise, pass a directory to rebuild the original substructure. If ``folder`` is an absolute path,
    #         ``root_dir`` is ignored.
    #     folder : :obj:`str`
    #         Where to store the file. Can be relative to ``root_dir`` or absolute, in which case ``root_dir`` is ignored.
    #         If ``folder`` is relative, the behaviour depends on whether it starts with a dot ``.`` or not: If it does,
    #         the folder is created at every end point of the relative tree structure under ``root_dir``. If it doesn't,
    #         it is created only once, relative to ``root_dir``, and the relative tree structure is build below.
    #     suffix : :obj:`str`, optional
    #         Suffix to append to the original file name.
    #     overwrite : :obj:`bool`, optional
    #         Pass True to overwrite existing files.
    #     simulate : :obj:`bool`, optional
    #         Pass True if no files are to be written.
    #
    #     Returns
    #     -------
    #     :obj:`str`
    #         Path of the stored file.
    #
    #     """
    #
    #     id = (key, i)
    #     logger = self.id_logger(id)
    #     fname = self.fnames[key][i]
    #
    #     if id not in self._parsed_mscx:
    #         logger.error(f"No Score object found. Call parse_scores() first.")
    #         return
    #     path = self._calculate_path(key=key, i=i, root_dir=root_dir, folder=folder)
    #     if path is None:
    #         return
    #
    #     fname = fname + suffix + '.mscx'
    #     file_path = os.path.join(path, fname)
    #     if os.path.isfile(file_path):
    #         if simulate:
    #             if overwrite:
    #                 logger.warning(f"Would have overwritten {file_path}.")
    #                 return
    #             logger.warning(f"Would have skipped {file_path}.")
    #             return
    #         elif not overwrite:
    #             logger.warning(f"Skipped {file_path}.")
    #             return
    #     if simulate:
    #         logger.debug(f"Would have written score to {file_path}.")
    #     else:
    #         os.makedirs(path, exist_ok=True)
    #         self._parsed_mscx[id].store_scores(file_path)
    #         logger.debug(f"Score written to {file_path}.")
    #
    #     return file_path
    #
    #
    # def _store_tsv(self, df, key, i, folder, suffix='', root_dir=None, what='DataFrame', simulate=False):
    #     """ Stores a given DataFrame by constructing path and file name from a loaded file based on the arguments.
    #
    #     Parameters
    #     ----------
    #     df : :obj:`pandas.DataFrame`
    #         DataFrame to store as a TSV.
    #     key, i : (:obj:`str`, :obj:`int`)
    #         ID from which to construct the new path and filename.
    #     folder, root_dir : :obj:`str`
    #         Parameters passed to :py:meth:`_calculate_path`.
    #     suffix : :obj:`str`, optional
    #         Suffix to append to the original file name.
    #     what : :obj:`str`, optional
    #         Descriptor, what the DataFrame contains for more informative log message.
    #     simulate : :obj:`bool`, optional
    #         Pass True if no files are to be written.
    #
    #     Returns
    #     -------
    #     :obj:`str`
    #         Path of the stored file.
    #
    #     """
    #     tsv_logger = self.id_logger((key, i))
    #
    #     if df is None:
    #         tsv_logger.debug(f"No DataFrame for {what}.")
    #         return
    #     path = self._calculate_path(key=key, i=i, root_dir=root_dir, folder=folder)
    #     if path is None:
    #         return
    #
    #     fname = self.fnames[key][i] + suffix + ".tsv"
    #     file_path = os.path.join(path, fname)
    #     if simulate:
    #         tsv_logger.debug(f"Would have written {what} to {file_path}.")
    #     else:
    #         tsv_logger.debug(f"Writing {what} to {file_path}.")
    #         write_tsv(df, file_path, logger=tsv_logger)
    #     return file_path
    #
    #
    #
    # def _treat_key_param(self, keys):
    #     if keys is None:
    #         keys = list(self.full_paths.keys())
    #     elif isinstance(keys, str):
    #         keys = [keys]
    #     return [k for k in sorted(set(keys)) if k in self.files]
    #
    #
    # def _treat_harmony_layer_param(self, harmony_layer):
    #     if harmony_layer is None:
    #         return None
    #     all_types = {str(k): k for k in self.count_labels().keys()}
    #     if isinstance(harmony_layer, int) or isinstance(harmony_layer, str):
    #         harmony_layer = [harmony_layer]
    #     lt = [str(t) for t in harmony_layer]
    #     def matches_any_type(user_input):
    #         return any(True for t in all_types if user_input in t)
    #     def get_matches(user_input):
    #         return [t for t in all_types if user_input in t]
    #
    #     not_found = [t for t in lt if not matches_any_type(t)]
    #     if len(not_found) > 0:
    #         plural = len(not_found) > 1
    #         plural_s = 's' if plural else ''
    #         self.logger.warning(
    #             f"No labels found with {'these' if plural else 'this'} label{plural_s} harmony_layer{plural_s}: {', '.join(not_found)}")
    #     return [all_types[t] for user_input in lt for t in get_matches(user_input)]
    #
    # def update_metadata(self, allow_suffix=False):
    #     """Uses all parsed metadata TSVs to update the information in the corresponding parsed MSCX files and returns
    #     the IDs of those that have been changed.
    #
    #     Parameters
    #     ----------
    #     allow_suffix : :obj:`bool`, optional
    #         If set to True, this would also update the metadata for currently parsed MuseScore files
    #         corresponding to the columns 'rel_paths' and 'fnames' + [ANY SUFFIX]. For example,
    #         the row ('MS3', 'bwv846') would also update the metadata of 'MS3/bwv846_reviewed.mscx'.
    #
    #     Returns
    #     -------
    #     :obj:`list`
    #         IDs of the parsed MuseScore files whose metadata has been updated.
    #     """
    #     metadata_dfs = self.metadata_tsv()
    #     if len(metadata_dfs) > 0:
    #         metadata = pd.concat(metadata_dfs.values(), keys=metadata_dfs.keys())
    #     else:
    #         metadata = self._metadata
    #     if len(metadata) == 0:
    #         self.logger.debug("No parsed metadata found.")
    #         return
    #     old = metadata
    #     if old.index.names != ['rel_paths', 'fnames']:
    #         try:
    #             old = old.set_index(['rel_paths', 'fnames'])
    #         except KeyError:
    #             self.logger.warning(f"Parsed metadata do not contain the columns 'rel_paths' and 'fnames' "
    #                                 f"needed to match information on identical files.")
    #             return []
    #     new = self.metadata(from_tsv=False).set_index(['rel_paths', 'fnames'])
    #     excluded_cols = ['ambitus', 'annotated_key', 'KeySig', 'label_count', 'last_mc', 'last_mn', 'musescore',
    #                      'TimeSig', 'length_qb', 'length_qb_unfolded', 'all_notes_qb', 'n_onsets', 'n_onset_positions']
    #     old_cols = sorted([c for c in old.columns if c not in excluded_cols and c[:5] != 'staff'])
    #
    #     parsed = old.index.map(lambda i: i in new.index)
    #     relevant = old.loc[parsed, old_cols]
    #     updates = defaultdict(dict)
    #     for i, row in relevant.iterrows():
    #         new_row = new.loc[i]
    #         for j, val in row[row.notna()].iteritems():
    #             val = str(val)
    #             if j not in new_row or str(new_row[j]) != val:
    #                 updates[i][j] = val
    #
    #     l = len(updates)
    #     ids = []
    #     if l > 0:
    #         for (rel_path, fname), new_dict in updates.items():
    #             matches = self.fname2ids(fname=fname, rel_path=rel_path, allow_suffix=allow_suffix)
    #             match_ids = [id for id in matches.keys() if id in self._parsed_mscx]
    #             n_files_to_update = len(match_ids)
    #             if n_files_to_update == 0:
    #                 self.logger.debug(
    #                     f"rel_path={rel_path}, fname={fname} does not correspond to a currently parsed MuseScore file.")
    #                 continue
    #             for id in match_ids:
    #                 for name, val in new_dict.items():
    #                     self._parsed_mscx[id].mscx.parsed.metatags[name] = val
    #                 self._parsed_mscx[id].mscx.parsed.update_metadata()
    #                 self.id_logger(id).debug(f"Updated with {new_dict}")
    #                 ids.append(id)
    #
    #         self.logger.info(f"{l} files updated.")
    #     else:
    #         self.logger.info("Nothing to update.")
    #     return ids
    #
    #
    # def __getstate__(self):
    #     """ Override the method of superclass """
    #     return self.__dict__




    def __getitem__(self, item) -> Corpus:
        if isinstance(item, str):
            return self.get_corpus(item)
        elif isinstance(item, tuple):
            if len(item) == 1:
                return self.get_corpus(item[0])
            if len(item) == 2:
                corpus_name, fname_or_ix = item
                return self.get_corpus(corpus_name)[fname_or_ix]
            corpus_name, *remainder = item
            return self.get_corpus(corpus_name)[tuple(remainder)]

    def __iter__(self) -> Iterator[Tuple[str, Corpus]]:
        """  Iterate through all (corpus_name, Corpus) tuples, regardless of any Views.

        Yields: (corpus_name, Corpus) tuples
        """
        yield from self.corpus_objects.items()


    def __repr__(self):
        """Show the :meth:`info` under the active view."""
        return self.info(return_str=True)

    # def _get_unambiguous_fnames_from_ids(self, score_ids, key):
    #
    #     file_info = [self.id2file_info[id] for id in score_ids]
    #     score_names = [F.fname for F in file_info]
    #     score_name_set = set(score_names)
    #     if len(score_names) == len(score_name_set):
    #         return dict(zip(score_names, score_ids))
    #     more_than_one = {name: [] for name, cnt in Counter(score_names).items() if cnt > 1}
    #     result = {} # fname -> score_id
    #     for F in file_info:
    #         if F.fname in more_than_one:
    #             more_than_one[F.fname].append(F)
    #         else:
    #             result[F.fname] = F.id
    #     for name, files in more_than_one.items():
    #         choice_between_n = len(files)
    #         df = pd.DataFrame.from_dict({F.id: dict(subdir=F.subdir, fext=F.fext, subdir_len=len(F.subdir)) for F in files}, orient='index')
    #         self.logger.debug(f"Trying to disambiguate between these {choice_between_n} with the same fname '{name}':\n{df}")
    #         shortest_subdir_length = df.subdir_len.min()
    #         shortest_length_selector = (df.subdir_len == shortest_subdir_length)
    #         n_have_shortest_length = shortest_length_selector.sum()
    #         # checking if the shortest path contains only 1 file and pick that
    #         if n_have_shortest_length == 1:
    #             id = df.subdir_len.idxmin()
    #             picked = df.loc[id]
    #             self.logger.info(f"In order to pick one from the {choice_between_n} scores with fname '{name}', the one with the shortest subdir '{picked.subdir}' was selected.")
    #             result[name] = id
    #             continue
    #         # otherwise, check if there is only a single MSCX or otherwise MSCZ file and pick that
    #         fexts = df.fext.value_counts()
    #         if '.mscx' in fexts:
    #             if fexts['.mscx'] == 1:
    #                 picked = df[df.fext == '.mscx'].iloc[0]
    #                 id = picked.name
    #                 self.logger.info(f"In order to pick one from the {choice_between_n} scores with fname '{name}', the one contained in '{picked.subdir}' was selected because it is the only "
    #                                  f"one in MSCX format.")
    #                 result[name] = id
    #                 continue
    #         elif '.mscz' in fexts and fexts['.mscz'] == 1:
    #             picked = df[df.fext == '.mscz'].iloc[0]
    #             id = picked.name
    #             self.logger.info(
    #                 f"In order to pick one from the {choice_between_n} scores with fname '{name}', the one contained in '{picked.subdir}' was selected because it is the only "
    #                 f"one in MuseScore format.")
    #             result[name] = id
    #             continue
    #         # otherwise, check if the shortest path contains only a single MSCX or MSCZ file as a last resort
    #         if n_have_shortest_length < choice_between_n:
    #             df = df[shortest_length_selector]
    #             self.logger.debug(f"Picking those from the shortest subdir has reduced the choice to {n_have_shortest_length}:\n{df}.")
    #         else:
    #             self.logger.warning(f"Unable to pick one of the available scores for fname '{name}', it will be disregarded until disambiguated:\n{df}")
    #             continue
    #         if '.mscx' in df.fext.values and fexts['.mscx'] == 1:
    #             pick_ext = '.mscx'
    #         elif '.mscz' in df.fext.values and fexts['.mscz'] == 1:
    #             pick_ext = '.mscz'
    #         else:
    #             self.logger.warning(f"Unable to pick one of the available scores for fname '{name}', it will be disregarded until disambiguated:\n{df}")
    #             continue
    #         picked = df[df.fext == pick_ext].iloc[0]
    #         id = picked.name
    #         self.logger.info(
    #             f"In order to pick one from the {choice_between_n} scores with fname '{name}', the '{pick_ext}' one contained in '{picked.subdir}' was selected because it is the only "
    #             f"one in that format contained in the shortest subdir.")
    #         result[name] = id
    #     return result


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
#
# class PieceView(View):
#
#     def __init__(self,
#                  view: View,
#                  fname: str):
#         self.view = view  # parent View object
#         self.p = view.p
#         self.key = view.key
#         self.fname = fname
#         logger_cfg = self.p.logger_cfg
#         logger_cfg['name'] = f"{self.view.logger.name}.{self.fname}"
#         super(Parse, self).__init__(subclass='Piece', logger_cfg=logger_cfg)  # initialize loggers
#         matches = view.detect_ids_by_fname(parsed_only=True, names=[fname])
#         if len(matches) != 1:
#             raise ValueError(f"{len(matches)} fnames match {fname} for key {self.key}")
#         self.matches = matches[fname]
#         self.score_available = 'scores' in self.matches
#         self.measures_available = self.score_available or 'measures' in self.matches
#
#
#
#     @lru_cache()
#     def get_dataframe(self, what: Literal['measures', 'notes', 'rests', 'labels', 'expanded', 'events', 'chords', 'metadata', 'form_labels'],
#                       unfold: bool = False,
#                       quarterbeats: bool = False,
#                       interval_index: bool = False,
#                       disambiguation: str = 'auto',
#                       prefer_score: bool = True,
#                       return_file_info: bool = False) -> pd.DataFrame:
#         """ Retrieves one DataFrame for the piece.
#
#         Args:
#             what: What kind of DataFrame to retrieve.
#             unfold: Pass True to unfold repeats.
#             quarterbeats:
#             interval_index:
#             disambiguation: In case several DataFrames are available in :attr:`.matches`, pass its disambiguation string.
#             prefer_score: By default, data from parsed scores is preferred to that from parsed TSVs. Pass False to prefer TSVs.
#             return_file_info: Pass True if the method should also return a :obj:`namedtuple` with information on the DataFrame
#                 being returned. It comes with the fields "id", "full_path", "suffix", "fext", "subdir", "i_str" where the
#                 latter is the ID's second component as a string.
#
#         Returns:
#             The requested DataFrame if available and, if ``return_file_info`` is set to True, a namedtuple with information about its provenance.
#
#         Raises:
#             FileNotFoundError: If no DataFrame of the requested type is available
#         """
#         available = list(self.p._dataframes.keys())
#         if what not in available:
#             raise ValueError(f"what='{what}' is an invalid argument. Pass one of {available}.")
#         if self.score_available and (prefer_score or what not in self.matches):
#             file_info = disambiguate(self.matches['scores'], disambiguation=disambiguation)
#             score = self.p[file_info.id]
#             df = score.mscx.__getattribute__(what)()
#         elif what in self.matches:
#             file_info = disambiguate(self.matches[what], disambiguation=disambiguation)
#             df = self.p[file_info.id]
#         else:
#             raise FileNotFoundError(f"No {what} available for {self.key} -> {self.fname}")
#         if any((unfold, quarterbeats, interval_index)):
#             measures = self.get_dataframe('measures', prefer_score=prefer_score)
#             df = dfs2quarterbeats([df], measures, unfold=unfold, quarterbeats=quarterbeats,
#                                    interval_index=interval_index, logger=self.logger)[0]
#         if return_file_info:
#             return df, file_info
#         return df