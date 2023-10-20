import os
import re
from collections import Counter, defaultdict
from typing import (
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    overload,
)

import pandas as pd

from ._typing import (
    AnnotationsFacet,
    CorpusFnameTuple,
    Facet,
    FacetArguments,
    FileDataframeTuple,
    FileDict,
    FileList,
    FileParsedTuple,
    ScoreFacet,
    ScoreFacets,
)
from .corpus import Corpus
from .logger import LoggedClass
from .piece import Piece
from .utils import (
    File,
    available_views2str,
    enforce_piece_index_for_metadata,
    get_first_level_corpora,
    get_musescore,
    path2parent_corpus,
    pretty_dict,
    resolve_dir,
    resolve_paths_argument,
    update_labels_cfg,
)
from .view import DefaultView, View, create_view_from_parameters


class Parse(LoggedClass):
    """
    Class for creating one or several :class:`~.corpus.Corpus` objects and performing actions on all of them.
    """

    _deprecated_elements = [
        "parsed_mscx",
        "parsed_tsv",
        "add_detached_annotations",
        "count_annotation_layers",
        "count_labels",
        "get_lists",
        "iter",
        "parse_mscx",
        "pieces",
        "store_scores",
        "update_metadata",
    ]

    def __init__(
        self,
        directory: Optional[Union[str, Collection[str]]] = None,
        recursive: bool = True,
        only_metadata_pieces: bool = True,
        include_convertible: bool = False,
        include_tsv: bool = True,
        exclude_review: bool = True,
        file_re: Optional[Union[str, re.Pattern]] = None,
        folder_re: Optional[Union[str, re.Pattern]] = None,
        exclude_re: Optional[Union[str, re.Pattern]] = None,
        file_paths: Optional[Collection[str]] = None,
        labels_cfg: dict = {},
        ms=None,
        **logger_cfg,
    ):
        """Initialize a Parse object and try to create corpora if directories and/or file paths are specified.

        Args:
          directory: Path to scan for corpora.
          recursive:
            Pass False if you don't want to scan ``directory`` for subcorpora, but force making it a corpus instead.
          only_metadata_pieces:
              The default view excludes piece names that are not listed in the corpus' metadata.tsv file
              (e.g. when none was found).
              Pass False to include all pieces regardless. This might be needed when setting ``recursive`` to False.
          include_convertible:
              The default view excludes scores that would need conversion to MuseScore format prior to parsing.
              Pass True to include convertible scores in .musicxml, .midi, .cap or any other format that MuseScore 3 can
              open.
              For on-the-fly conversion, however, the parameter ``ms`` needs to be set.
          include_tsv: The default view includes TSV files. Pass False to disregard them and parse only scores.
          exclude_review:
              The default view excludes files and folders whose name contains 'review'.
              Pass False to include these as well.
          file_re:
            Pass a regular expression if you want to create a view filtering out all files that do not contain it.
          folder_re:
            Pass a regular expression if you want to create a view filtering out all folders that do not contain it.
          exclude_re:
            Pass a regular expression if you want to create a view filtering out all files or folders that contain it.
          file_paths:
              If ``directory`` is specified, the file names of these paths are used to create a filtering view excluding
              all other files. Otherwise, all paths are expected to be part of the same parent corpus which will be
              inferred from the first path by looking for the first parent directory that either contains a
              'metadata.tsv' file or is a git. This parameter is deprecated and ``file_re`` should be used instead.
          labels_cfg: Pass a configuration dict to detect only certain labels or change their output format.
          ms:
              If you pass the path to your local MuseScore 3 installation, ms3 will attempt to parse musicXML,
              MuseScore 2, and other formats by temporarily converting them. If you're using the standard path,
              you may try 'auto', or 'win' for Windows, 'mac' for MacOS, or 'mscore' for Linux. In case you do not
              pass the 'file_re' and the MuseScore executable is detected, all convertible files are automatically
              selected, otherwise only those that can be parsed without conversion.
          **logger_cfg:
            Keyword arguments for changing the logger configuration. E.g. ``level='d'`` to see all debug messages.
        """
        if "level" not in logger_cfg or (logger_cfg["level"] is None):
            logger_cfg["level"] = "w"
        super().__init__(subclass="Parse", logger_cfg=logger_cfg)

        self.corpus_paths: Dict[str, str] = {}
        """
        {corpus_name -> path} dictionary with each corpus's base directory. Generally speaking, each corpus path is
        expected to contain a ``metadata.tsv`` and, maybe, to be a git.
        """

        self.corpus_objects: Dict[str, Corpus] = {}
        """{corpus_name -> Corpus} dictionary with one object per :attr:`corpus_path <corpus_paths>`.
        """

        self._ms = get_musescore(ms, logger=self.logger)

        self._views: dict = {}
        initial_view = create_view_from_parameters(
            only_metadata_pieces=only_metadata_pieces,
            include_convertible=include_convertible,
            include_tsv=include_tsv,
            exclude_review=exclude_review,
            file_paths=file_paths,
            file_re=file_re,
            folder_re=folder_re,
            exclude_re=exclude_re,
            level=self.logger.getEffectiveLevel(),
        )
        self._views[None] = initial_view
        if initial_view.name != "default":
            self._views["default"] = DefaultView(level=self.logger.getEffectiveLevel())
        self._views["all"] = View(level=self.logger.getEffectiveLevel())
        #
        # self._ignored_warnings = defaultdict(list)
        # """:obj:`collections.defaultdict`
        # {'logger_name' -> [(message_id), ...]} This dictionary stores the warnings to be ignored
        # upon loading them from an IGNORED_WARNINGS file.
        # """
        #
        self.labels_cfg = {
            "staff": None,
            "voice": None,
            "harmony_layer": None,
            "positioning": False,
            "decode": True,
            "column_name": "label",
            "color_format": None,
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

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%% END of __init__() %%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    @property
    def ms(self) -> str:
        """Path or command of the local MuseScore 3 installation if specified by the user and recognized."""
        return self._ms

    @ms.setter
    def ms(self, ms: Union[str, Literal["auto", "win", "mac"]]):
        executable = get_musescore(ms, logger=self.logger)
        if executable is None:
            raise FileNotFoundError(
                f"'{ms}' did not lead me to a MuseScore executable."
            )
        if executable is not None:
            self._ms = executable
            for _, corpus in self.__iter__():
                corpus.ms = executable

    @property
    def n_detected(self) -> int:
        """
        Number of detected files aggregated from all :class:`~.corpus.Corpus` objects without taking views into account.
        Excludes metadata files.
        """
        return sum(corpus.n_detected for _, corpus in self)

    @property
    def n_orphans(self) -> int:
        """Number of files that are always disregarded because they could not be attributed to any of the pieces."""
        return sum(len(corpus.ix2orphan_file) for _, corpus in self)

    @property
    def n_parsed(self) -> int:
        """
        Number of parsed files aggregated from all :class:`~.corpus.Corpus` objects without taking views into account.
        Excludes metadata files.
        """
        return sum(corpus.n_parsed for _, corpus in self)

    @property
    def n_parsed_scores(self) -> int:
        """
        Number of parsed scores aggregated from all :class:`~.corpus.Corpus` objects without taking views into account.
        Excludes metadata files.
        """
        return sum(corpus.n_parsed_scores for _, corpus in self)

    @property
    def n_parsed_tsvs(self) -> int:
        """
        Number of parsed TSV files aggregated from all :class:`~.corpus.Corpus` objects without taking views into
        account. Excludes metadata files.
        """
        return sum(corpus.n_parsed_tsvs for _, corpus in self)

    @property
    def n_pieces(self) -> int:
        """Number of all available pieces ('pieces'), independent of the view."""
        return sum(corpus.n_pieces for _, corpus in self)

    @property
    def n_unparsed_scores(self) -> int:
        """
        Number of all detected but not yet parsed scores, aggregated from all :class:`~.corpus.Corpus` objects without
        taking views into account. Excludes metadata files.
        """
        return sum(corpus.n_unparsed_scores for _, corpus in self)

    @property
    def n_unparsed_tsvs(self) -> int:
        """
        Number of all detected but not yet parsed TSV files, aggregated from all :class:`~.corpus.Corpus` objects
        without taking views into account. Excludes metadata files.
        """
        return sum(corpus.n_unparsed_tsvs for _, corpus in self)

    @property
    def view(self) -> View:
        """Retrieve the current View object. Shorthand for :meth:`get_view`."""
        return self.get_view()

    @view.setter
    def view(self, new_view: View):
        if not isinstance(new_view, View):
            return TypeError(
                "If you want to switch to an existing view, use its name like an attribute or "
                "call _.switch_view()."
            )
        self.set_view(new_view)

    @property
    def views(self) -> None:
        """Display a short description of the available views."""
        print(
            pretty_dict(
                {"[active]" if k is None else k: v for k, v in self._views.items()},
                "view_name",
                "Description",
            )
        )

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
        return {
            view.name if name is None else name for name, view in self._views.items()
        }

    def add_corpus(
        self,
        directory: str,
        corpus_name: Optional[str] = None,
        only_metadata_pieces: Optional[bool] = None,
        include_convertible: Optional[bool] = None,
        include_tsv: Optional[bool] = None,
        exclude_review: Optional[bool] = None,
        file_re: Optional[Union[str, re.Pattern]] = None,
        folder_re: Optional[Union[str, re.Pattern]] = None,
        exclude_re: Optional[Union[str, re.Pattern]] = None,
        paths: Optional[Collection[str]] = None,
        **logger_cfg,
    ) -> None:
        """
        This method creates a :class:`~.corpus.Corpus` object which scans the directory ``directory`` for parseable
        files.
        It inherits all :class:`Views <.view.View>` from the Parse object.

        Args:
          directory: Directory to scan for files.
          corpus_name:
              By default, the folder name of ``directory`` is used as name for this corpus. Pass a string to
              use a different identifier.
          **logger_cfg:
              Keyword arguments for configuring the logger of the new Corpus object. E.g. ``level='d'`` to see all debug
              messages. Note that the logger is a child logger of this Parse object's logger and propagates,
              so it might filter debug messages. You can use _.change_logger_cfg(level='d') to change the level post
              hoc.
        """
        directory = resolve_dir(directory)
        if not os.path.isdir(directory):
            self.logger.warning(f"{directory} is not an existing directory.")
            return
        new_logger_cfg = dict(self.logger_cfg)
        new_logger_cfg.update(logger_cfg)
        if corpus_name is None:
            corpus_name = os.path.basename(directory).strip(r"\/")
        new_logger_cfg["name"] = self.logger.name + "." + corpus_name.replace(".", "")
        view_params = any(
            param is not None
            for param in (
                paths,
                file_re,
                folder_re,
                exclude_re,
                only_metadata_pieces,
                include_tsv,
                exclude_review,
                include_convertible,
            )
        )
        if view_params:
            initial_view = create_view_from_parameters(
                only_metadata_pieces=only_metadata_pieces,
                include_convertible=include_convertible,
                include_tsv=include_tsv,
                exclude_review=exclude_review,
                file_paths=paths,
                file_re=file_re,
                folder_re=folder_re,
                exclude_re=exclude_re,
                level=self.logger.getEffectiveLevel(),
            )
            self._views[initial_view.name] = initial_view
        else:
            initial_view = self.get_view()
        corpus = self.make_corpus(directory, initial_view, **new_logger_cfg)
        if corpus is None or len(corpus.files) == 0:
            self.logger.info(f"No parseable files detected in {directory}. Skipping...")
            return
        if corpus_name is None:
            corpus_name = os.path.basename(directory).strip(r"\/")
        if corpus_name in self.corpus_paths:
            existing_path = self.corpus_paths[corpus_name]
            if existing_path == directory:
                self.logger.warning(
                    f"Corpus '{corpus_name}' had already been present and was overwritten, i.e., reset."
                )
            else:
                self.logger.warning(
                    f"Corpus '{corpus_name}' had already been present for the path {existing_path} and "
                    f"was replaced by {directory}"
                )
        self.corpus_paths[corpus_name] = directory
        self.corpus_objects[corpus_name] = corpus

    def make_corpus(self, directory, initial_view, **logger_cfg) -> Optional[Corpus]:
        try:
            corpus = Corpus(
                directory=directory,
                view=initial_view,
                labels_cfg=self.labels_cfg,
                ms=self.ms,
                **logger_cfg,
            )
        except AssertionError:
            self.logger.debug(f"{directory} contains no parseable files.")
            return
        corpus.set_view(
            **{
                view_name: view
                for view_name, view in self._views.items()
                if view_name is not None
            }
        )
        return corpus

    def add_dir(
        self,
        directory: str,
        recursive: bool = True,
        only_metadata_pieces: Optional[bool] = None,
        include_convertible: Optional[bool] = None,
        include_tsv: Optional[bool] = None,
        exclude_review: Optional[bool] = None,
        file_re: Optional[Union[str, re.Pattern]] = None,
        folder_re: Optional[Union[str, re.Pattern]] = None,
        exclude_re: Optional[Union[str, re.Pattern]] = None,
        paths: Optional[Collection[str]] = None,
        **logger_cfg,
    ) -> None:
        """
        This method decides if the directory ``directory`` contains several corpora or if it is a corpus
        itself, and calls :meth:`add_corpus` for each corpus.

        Args:
          directory: Directory to scan for corpora.
          recursive:
              By default, if any of the first-level subdirectories contains a 'metadata.tsv' or is a git, all
              first-level subdirectories of ``directory`` are treated as corpora, i.e. one :class:`~.corpus.Corpus`
              object per folder is created. Pass False to prevent this, which is equivalent to calling
              :meth:`add_corpus(directory) <add_corpus>`
          **logger_cfg:
              Keyword arguments for configuring the logger of the new Corpus objects. E.g. ``level='d'`` to see all
              debug messages. Note that the loggers are child loggers of this Parse object's logger and propagate,
              so it might filter debug messages. You can use _.change_logger_cfg(level='d') to change the level post
              hoc.
        """
        directory = resolve_dir(directory)
        if not os.path.isdir(directory):
            self.logger.warning(f"{directory} is not an existing directory.")
            return
        L = locals()
        arguments = (
            "only_metadata_pieces",
            "include_convertible",
            "include_tsv",
            "exclude_review",
            "file_re",
            "folder_re",
            "exclude_re",
            "paths",
        )
        corpus_config = {arg: L[arg] for arg in arguments if L[arg] is not None}
        if "level" not in logger_cfg:
            logger_cfg["level"] = self.logger.getEffectiveLevel()
        corpus_config.update(logger_cfg)
        if not recursive:
            self.add_corpus(directory=directory, **corpus_config)
            return

        # new corpus/corpora to be added
        subdir_corpora = sorted(get_first_level_corpora(directory, logger=self.logger))
        n_corpora = len(subdir_corpora)
        if n_corpora == 0:
            self.logger.debug(
                f"Treating {directory} as corpus because none of its children seems to be a corpus."
            )
            self.add_corpus(directory, **corpus_config)
        else:
            self.logger.debug(
                f"{n_corpora} individual corpora detected in {directory}."
            )
            for corpus_path in subdir_corpora:
                self.add_corpus(corpus_path, **corpus_config)

    def add_files(
        self, file_paths: Union[str, Collection[str]], corpus_name: Optional[str] = None
    ) -> None:
        """
        Deprecated: To deal with particular files only, use :meth:`add_corpus` passing the directory containing them and
        configure the :class`~.view.View` accordingly. This method here does it for you but easily leads to unexpected
        behaviour. It expects the file paths to point to files located in a shared corpus folder on some higher level or
        in folders for which :class:`~.corpus.Corpus` objects have already been created.

        Args:
          file_paths: Collection of file paths. Only existing files can be added.
          corpus_name:

              * By default, I will try to attribute the files to existing :class:`~.corpus.Corpus` objects based on
                their paths. This makes sense only when new files have been created after the directories were scanned.
              * For paths that do no not contain an existing corpus_path, I will try to detect the parent directory that
                is a corpus (based on it being a git or containing a ``metadata.tsv``). If this is without success
                for the first path, I will raise an error. Otherwise, all subsequent paths will be considered to be
                part of that same corpus (watch out meaningless relative paths!).
              * You can pass a folder name contained in the first path to create a new corpus, assuming that all other
                paths are contained in it (watch out meaningless relative paths!).
              * Pass an existing corpus_name to add the files to a particular corpus. Note that all parseable files
                under the corpus_path are detected anyway, and if you add files from other directories, it will lead
                to invalid relative paths that work only on your system. If you're adding files that have been created
                after the Corpus object has, you can leave this parameter empty; paths will be attributed to the
                existing corpora automatically.
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
                    raise ValueError(
                        f"No parent of {first_path} has been recognized as a corpus by being a git or containing a "
                        f"'metadata.tsv'. Use _.add_corpus()"
                    )
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
                        first_path = ""
                        break
                    if last_component == corpus_name:
                        # first_path is the corpus first_path
                        break
                    first_path = tmp_path
            else:
                first_path == ""
            if first_path == "":
                raise ValueError(
                    f"corpus_name needs to be a folder contained in the first path, but '{corpus_name}' isn't."
                )
            self.add_corpus(first_path)
            # corpus = self.get_corpus(corpus_name)
            # new_view = create_view_from_parameters(only_metadata_pieces=False, exclude_review=False, paths=paths)
            # corpus.set_view(new_view)

    def color_non_chord_tones(
        self,
        color_name: str = "red",
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["all", "auto", "ask"] = "all",
    ) -> Dict[CorpusFnameTuple, List[FileDataframeTuple]]:
        result = {}
        for corpus_name, corpus in self.iter_corpora(view_name):
            piece2reports = corpus.color_non_chord_tones(
                color_name, view_name=view_name, force=force, choose=choose
            )
            result.update(
                {
                    (corpus_name, piece): report
                    for piece, report in piece2reports.items()
                }
            )
        return result

    def change_labels_cfg(
        self,
        labels_cfg=(),
        staff=None,
        voice=None,
        harmony_layer=None,
        positioning=None,
        decode=None,
        column_name=None,
        color_format=None,
    ):
        """Update :obj:`Parse.labels_cfg` and retrieve new 'labels' tables accordingly.

        Parameters
        ----------
        labels_cfg : :obj:`dict`
            Using an entire dictionary or, to change only particular options, choose from:
        staff, voice, harmony_layer, positioning, decode, column_name
            Arguments as they will be passed to :py:meth:`~ms3.annotations.Annotations.get_labels`
        """
        keys = [
            "staff",
            "voice",
            "harmony_layer",
            "positioning",
            "decode",
            "column_name",
            "color_format",
        ]
        labels_cfg = dict(labels_cfg)
        for k in keys:
            val = locals()[k]
            if val is not None:
                labels_cfg[k] = val
        updated = update_labels_cfg(labels_cfg, logger=self.logger)
        self.labels_cfg.update(updated)
        for corpus_name, corpus in self:
            corpus.change_labels_cfg(labels_cfg=self.labels_cfg)

    def compare_labels(
        self,
        key: str = "detached",
        new_color: str = "ms3_darkgreen",
        old_color: str = "ms3_darkred",
        detached_is_newer: bool = False,
        add_to_rna: bool = True,
        view_name: Optional[str] = None,
        metadata_update: Optional[dict] = None,
        force_metadata_update: bool = False,
    ) -> Tuple[int, int]:
        """Compare detached labels ``key`` to the ones attached to the Score to create a diff.
        By default, the attached labels are considered as the reviewed version and labels that have changed or been
        added in comparison to the detached labels are colored in green; whereas the previous versions of changed
        labels are attached to the Score in red, just like any deleted label.

        Args:
          key: Key of the detached labels you want to compare to the ones in the score.
          new_color, old_color:
              The colors by which new and old labels are differentiated. Identical labels remain unchanged. Colors can
              be CSS colors or MuseScore colors (see :py:attr:`utils.MS3_COLORS`).
          detached_is_newer:
              Pass True if the detached labels are to be added with ``new_color`` whereas the attached changed labels
              will turn ``old_color``, as opposed to the default.
          add_to_rna:
              By default, new labels are attached to the Roman Numeral layer.
              Pass False to attach them to the chord layer instead.
          metadata_update:
             Dictionary containing metadata that is to be included in the comparison score. Notably, ms3 uses the key
             'compared_against' when the comparison is performed against a given git_revision.
          force_metadata_update:
             By default, the metadata is only updated if the comparison yields at least one difference to avoid
             outputting comparison scores not displaying any changes. Pass True to force the metadata update, which
             results in the properts :attr:`changed` being set to True.

        Returns:
          Number of scores in which labels have changed.
            Number of scores in which no label has chnged.
        """
        changed, unchanged = 0, 0
        for _, corpus in self.iter_corpora(view_name=view_name):
            c, u = corpus.compare_labels(
                key=key,
                new_color=new_color,
                old_color=old_color,
                detached_is_newer=detached_is_newer,
                add_to_rna=add_to_rna,
                view_name=view_name,
                metadata_update=metadata_update,
                force_metadata_update=force_metadata_update,
            )
            changed += c
            unchanged += u
        return changed, unchanged

    def count_changed_scores(self, view_name: Optional[str] = None):
        return sum(
            corpus.count_changed_scores() for _, corpus in self.iter_corpora(view_name)
        )

    def count_extensions(
        self,
        view_name: Optional[str] = None,
        per_piece: bool = False,
        include_metadata: bool = False,
    ):
        """Count file extensions.

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
            If set to True, the results are returned as {key: {subdir: Counter} }. ``per_key=True`` is therefore
            implied.

        Returns
        -------
        :obj:`dict`
            By default, the function returns a Counter of file extensions (Counters are converted to dicts).
            If ``per_key`` is set to True, a dictionary {key: Counter} is returned, separating the counts.
            If ``per_subdir`` is set to True, a dictionary {key: {subdir: Counter} } is returned.
        """
        extension_counters = {
            corpus_name: corpus.count_extensions(
                view_name, include_metadata=include_metadata
            )
            for corpus_name, corpus in self.iter_corpora(view_name)
        }
        if per_piece:
            return {
                (corpus_name, piece): dict(cnt)
                for corpus_name, piece2cnt in extension_counters.items()
                for piece, cnt in piece2cnt.items()
            }
        return {
            corpus_name: dict(sum(piece2cnt.values(), Counter()))
            for corpus_name, piece2cnt in extension_counters.items()
        }

    def count_files(
        self,
        detected=True,
        parsed=True,
        as_dict: bool = False,
        drop_zero: bool = True,
        view_name: Optional[str] = None,
    ) -> Union[pd.DataFrame, dict]:
        all_counts = {
            corpus_name: corpus._summed_file_count(
                types=detected, parsed=parsed, view_name=view_name
            )
            for corpus_name, corpus in self.iter_corpora(view_name=view_name)
        }
        counts_df = pd.DataFrame.from_dict(all_counts, orient="index", dtype="Int64")
        if drop_zero:
            empty_cols = counts_df.columns[counts_df.sum() == 0]
            counts_df = counts_df.drop(columns=empty_cols)
        if as_dict:
            return counts_df.to_dict(orient="index")
        counts_df.index.rename("corpus", inplace=True)
        return counts_df

    def count_parsed_scores(self, view_name: Optional[str] = None) -> int:
        return sum(map(len, self._get_parsed_score_files(view_name=view_name).values()))

    def count_parsed_tsvs(self, view_name: Optional[str] = None) -> int:
        return sum(map(len, self._get_parsed_tsv_files(view_name=view_name).values()))

    def count_unparsed_scores(self, view_name: Optional[str] = None) -> int:
        return sum(map(len, self._get_parsed_score_files(view_name=view_name).values()))

    def count_unparsed_tsvs(self, view_name: Optional[str] = None) -> int:
        return sum(map(len, self._get_parsed_tsv_files(view_name=view_name).values()))

    def count_pieces(self, view_name: Optional[str] = None) -> int:
        """Number of selected pieces under the given view."""
        return sum(
            corpus.count_pieces(view_name=view_name)
            for _, corpus in self.iter_corpora(view_name=view_name)
        )

    def create_missing_metadata_tsv(self, view_name: Optional[str] = None) -> None:
        for corpus_name, corpus in self.iter_corpora(view_name=view_name):
            if corpus.metadata_tsv is None:
                _ = corpus.create_metadata_tsv()

    def disambiguate_facet(
        self, facet: Facet, view_name: Optional[str] = None, ask_for_input=True
    ) -> None:
        """Calls the method on every selected corpus."""
        for name, corpus in self.iter_corpora(view_name):
            corpus.disambiguate_facet(
                facet=facet, view_name=view_name, ask_for_input=ask_for_input
            )

    def extract_facets(
        self,
        facets: ScoreFacets = None,
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
        flat=False,
        concatenate=True,
    ) -> Union[
        pd.DataFrame,
        Dict[
            CorpusFnameTuple,
            Union[Dict[str, List[FileDataframeTuple]], List[FileDataframeTuple]],
        ],
    ]:
        return self._aggregate_corpus_data(
            "extract_facets",
            facets=facets,
            view_name=view_name,
            force=force,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
            flat=flat,
            concatenate=concatenate,
        )

    def get_all_parsed(
        self,
        facets: FacetArguments = None,
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["all", "auto", "ask"] = "all",
        flat: bool = False,
        include_empty=False,
        concatenate: bool = True,
    ) -> Union[
        pd.DataFrame,
        Dict[
            CorpusFnameTuple, Union[Dict[str, FileParsedTuple], List[FileParsedTuple]]
        ],
    ]:
        return self._aggregate_corpus_data(
            "get_all_parsed",
            facets=facets,
            view_name=view_name,
            force=force,
            choose=choose,
            flat=flat,
            include_empty=include_empty,
            concatenate=concatenate,
        )

    def get_corpus(self, name) -> Corpus:
        assert (
            name in self.corpus_objects
        ), f"Don't have a corpus called '{name}', only {list(self.corpus_objects.keys())}"
        return self.corpus_objects[name]

    def get_dataframes(
        self,
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
        choose: Literal["all", "auto", "ask"] = "all",
        unfold: bool = False,
        interval_index: bool = False,
        flat=False,
        include_empty: bool = False,
    ) -> Union[
        pd.DataFrame,
        Dict[
            CorpusFnameTuple,
            Union[Dict[str, List[FileDataframeTuple]], List[FileDataframeTuple]],
        ],
    ]:
        """Renamed to :meth:`get_facets`."""
        lcls = locals()
        facets = [facet for facet in ScoreFacet.__args__ if lcls[facet]]
        return self.get_facets(
            facets=facets,
            view_name=view_name,
            force=force,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
            flat=flat,
            include_empty=include_empty,
        )

    def get_facet(
        self,
        facet: ScoreFacet,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
        concatenate: bool = True,
    ) -> Union[Dict[str, FileDataframeTuple], pd.DataFrame]:
        """Retrieves exactly one DataFrame per piece, if available."""
        return self._aggregate_corpus_data(
            "get_facet",
            facet=facet,
            view_name=view_name,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
            concatenate=concatenate,
        )

    @overload
    def get_facets(
        self,
        facets,
        view_name,
        force,
        choose,
        unfold,
        interval_index,
        flat,
        include_empty,
        concatenate: Literal[True],
    ) -> pd.DataFrame:
        ...

    @overload
    def get_facets(
        self,
        facets,
        view_name,
        force,
        choose,
        unfold,
        interval_index,
        flat: Literal[True],
        include_empty,
        concatenate: Literal[False],
    ) -> Dict[CorpusFnameTuple, List[FileDataframeTuple]]:
        ...

    @overload
    def get_facets(
        self,
        facets,
        view_name,
        force,
        choose,
        unfold,
        interval_index,
        flat: Literal[False],
        include_empty,
        concatenate: Literal[False],
    ) -> Dict[CorpusFnameTuple, Dict[str, List[FileDataframeTuple]]]:
        ...

    def get_facets(
        self,
        facets: ScoreFacets = None,
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["all", "auto", "ask"] = "all",
        unfold: bool = False,
        interval_index: bool = False,
        flat=False,
        include_empty=False,
        concatenate=True,
    ) -> Union[
        pd.DataFrame,
        Dict[
            CorpusFnameTuple,
            Union[Dict[str, List[FileDataframeTuple]], List[FileDataframeTuple]],
        ],
    ]:
        return self._aggregate_corpus_data(
            "get_facets",
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

    @overload
    def get_files(
        self,
        facets: FacetArguments = None,
        view_name: Optional[str] = None,
        parsed: bool = True,
        unparsed: bool = True,
        choose: Literal["all", "auto", "ask"] = "all",
        flat: bool = False,
        include_empty=False,
    ) -> Dict[CorpusFnameTuple, FileDict]:
        ...

    @overload
    def get_files(
        self,
        facets: FacetArguments = None,
        view_name: Optional[str] = None,
        parsed: bool = True,
        unparsed: bool = True,
        choose: Literal["all", "auto", "ask"] = "all",
        flat: bool = True,
        include_empty=False,
    ) -> Dict[CorpusFnameTuple, FileList]:
        ...

    def get_files(
        self,
        facets: FacetArguments = None,
        view_name: Optional[str] = None,
        parsed: bool = True,
        unparsed: bool = True,
        choose: Literal["all", "auto", "ask"] = "all",
        flat: bool = False,
        include_empty=False,
    ) -> Dict[CorpusFnameTuple, Union[FileDict, FileList]]:
        return self._aggregate_corpus_data(
            "get_files",
            facets=facets,
            view_name=view_name,
            parsed=parsed,
            unparsed=unparsed,
            choose=choose,
            flat=flat,
            include_empty=include_empty,
        )

    def get_piece(self, corpus_name: str, piece: str) -> Piece:
        """Returns an existing Piece object."""
        assert (
            corpus_name in self.corpus_objects
        ), f"'{corpus_name}' is not an existing corpus. Choose from {list(self.corpus_objects.keys())}"
        return self.corpus_objects[corpus_name].get_piece(piece)

    def get_view(self, view_name: Optional[str] = None, **config) -> View:
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
            self.logger.info(
                f"New view '{view_name}' created as a copy of '{old_name}'."
            )
        if len(config) > 0:
            view.update_config(**config)
        return view

    def info(
        self,
        view_name: Optional[str] = None,
        return_str: bool = False,
        show_discarded: bool = False,
    ):
        """"""
        header = "All corpora"
        header += "\n" + "-" * len(header) + "\n"

        # start info message with the names of the available views, the header, and info on the active view.
        view = self.get_view(view_name)
        view.reset_filtering_data()
        msg = available_views2str(self._views, view_name)
        msg += header
        view_info = f"View: {view}"
        if view_name is None:
            corpus_views = [
                corpus.get_view().name
                for _, corpus in self.iter_corpora(view_name=view_name)
            ]
            if len(set(corpus_views)) > 1:
                view_info = "This is a mixed view. Call _.info(view_name) to see a homogeneous one."
        msg += view_info + "\n\n"

        # Show info on all pieces and files included in the active view
        counts_df = self.count_files(view_name=view_name)
        if len(counts_df) == 0:
            if self.n_detected == 0:
                msg += "No files detected. Use _.add_corpus()."
            else:
                msg += "No files selected under the current view. You could use _.all to see everything."
        else:
            if counts_df.isna().any().any():
                counts_df = counts_df.fillna(0).astype("int")
            additional_columns = []
            for corpus_name in counts_df.index:
                corpus = self.get_corpus(corpus_name)
                has_metadata = "no" if corpus.metadata_tsv is None else "yes"
                corpus_view = corpus.get_view().name
                additional_columns.append([has_metadata, corpus_view])
            additional_columns = pd.DataFrame(
                additional_columns,
                columns=[("has", "metadata"), ("active", "view")],
                index=counts_df.index,
            )
            info_df = pd.concat([additional_columns, counts_df], axis=1)
            info_df.columns = pd.MultiIndex.from_tuples(info_df.columns)
            msg += info_df.to_string()
            n_changed_scores = self.count_changed_scores(view_name)
            if n_changed_scores > 0:
                msg += f"\n\n{n_changed_scores} scores have changed since parsing."
            filtering_report = view.filtering_report(
                show_discarded=show_discarded, return_str=True
            )
            if filtering_report != "":
                msg += "\n\n" + filtering_report
        if self.n_orphans > 0:
            msg += (
                f"\n\nThere are {self.n_orphans} orphans that could not be attributed to any of the respective "
                f"corpus's pieces"
            )
            if show_discarded:
                msg += ":"
                for name, corpus in self:
                    if corpus.n_orphans > 0:
                        msg += f"\n\t{name}: {list(corpus.ix2orphan_file.values())}"
            else:
                msg += "."
        if return_str:
            return msg
        print(msg)

    def insert_detached_labels(
        self,
        view_name: Optional[str] = None,
        key: str = "detached",
        staff: int = None,
        voice: Literal[1, 2, 3, 4] = None,
        harmony_layer: Optional[Literal[0, 1, 2]] = None,
        check_for_clashes: bool = True,
    ):
        """Attach all :py:attr:`~.annotations.Annotations` objects that are reachable via ``Score.key`` to their
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
            |   MuseScore will split and encode those beginning with a note name ( resulting in ms3-internal
                harmony_layer 3).
            | * 1 the labels are written into the staff's layer for Roman Numeral Analysis.
            | * 2 to have MuseScore interpret them as Nashville Numbers
        check_for_clashes : :obj:`bool`, optional
            By default, warnings are thrown when there already exists a label at a position (and in a notational
            layer) where a new one is attached. Pass False to deactivate these warnings.
        """
        reached, goal = 0, 0
        for i, (name, corpus) in enumerate(self.iter_corpora(view_name), 1):
            r, g = corpus.insert_detached_labels(
                view_name=view_name,
                key=key,
                staff=staff,
                voice=voice,
                harmony_layer=harmony_layer,
                check_for_clashes=check_for_clashes,
            )
            reached += r
            goal += g
        if i > 1:
            self.logger.info(f"{reached}/{goal} labels successfully added.")

    def iter_corpora(
        self, view_name: Optional[str] = None
    ) -> Iterator[Tuple[str, Corpus]]:
        """Iterate through corpora under the current or specified view."""
        view = self.get_view(view_name)
        for corpus_name, corpus in view.filter_by_token("corpora", self):
            if view_name not in corpus._views:
                if view_name is None:
                    corpus.set_view(view)
                else:
                    corpus.set_view(**{view_name: view})
            yield corpus_name, corpus

    def iter_independent_corpora(
        self, view_name: Optional[str] = None
    ) -> Iterator[Tuple[str, Corpus]]:
        """Like iter_corpora() but creating new Corpus objects that are not stored in this Parse object to avoid
        filling up memory when parsing many files."""
        initial_view = self.get_view(view_name)
        for corpus_name, corpus in self.iter_corpora(view_name=view_name):
            yield corpus_name, self.make_corpus(
                directory=corpus.corpus_path, initial_view=initial_view
            )

    def iter_pieces(self) -> Tuple[CorpusFnameTuple, Piece]:
        for corpus_name, corpus in self:
            for piece, piece_obj in corpus:
                yield (corpus_name, piece), piece_obj

    def keys(self) -> List[str]:
        """Return the names of all corpus objects."""
        return list(self.corpus_objects.keys())

    def load_facet_into_scores(
        self,
        facet: AnnotationsFacet,
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["auto", "ask"] = "auto",
        git_revision: Optional[str] = None,
        key: str = "detached",
        infer: bool = True,
        **cols,
    ) -> Dict[str, int]:
        result = {}
        for corpus_name, corpus in self.iter_corpora(view_name):
            result[corpus_name] = corpus.load_facet_into_scores(
                facet=facet,
                view_name=view_name,
                force=force,
                choose=choose,
                git_revision=git_revision,
                key=key,
                infer=infer,
                **cols,
            )
        return result

    def load_ignored_warnings(self, path: str) -> None:
        """Adds a filters to all loggers included in a IGNORED_WARNINGS file.

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
                self.logger.info(
                    f"The existing view called '{new_name}' has been overwritten"
                )
                del self._views[new_name]
            old_view = self._views[None]
            self._views[old_view.name] = old_view
            self._views[None] = active
        for view_name, view in views.items():
            if view.name is None:
                view.name = view_name
            self._views[view_name] = view
        for corpus_name, corpus in self:
            if active is not None and active.check_token("corpus", corpus_name):
                corpus.set_view(active)
            for view_name, view in views.items():
                if view.check_token("corpus", corpus_name):
                    corpus.set_view(**{view_name: view})

    def switch_view(
        self,
        view_name: str,
        show_info: bool = True,
        propagate=True,
    ) -> None:
        if view_name is None:
            return
        new_view = self.get_view(view_name)
        old_view = self.get_view()
        self._views[old_view.name] = old_view
        self._views[None] = new_view
        new_name = new_view.name
        if new_name in self._views:
            del self._views[new_name]
        if propagate:
            for corpus_name, corpus in self:
                active_view = corpus.get_view()
                if active_view.name != new_name or active_view != new_view:
                    corpus.set_view(new_view)
        if show_info:
            self.info()

    def update_metadata_tsv_from_parsed_scores(
        self,
        root_dir: Optional[str] = None,
        suffix: str = "",
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
              By default, a subset of metadata columns will be written to 'README.md' in the same folder as the TSV
              file. If the file exists, it will be scanned for a line containing the string '# Overview' and overwritten
              from that line onwards.
          view_name:
              The view under which you want to update metadata from the selected parsed files. Defaults to None,
              i.e. the active view.

        Returns:
          The file paths to which metadata was written.
        """
        metadata_paths = []
        for _, corpus in self.iter_corpora():
            paths = corpus.update_metadata_tsv_from_parsed_scores(
                root_dir=root_dir,
                suffix=suffix,
                markdown_file=markdown_file,
                view_name=view_name,
            )
            metadata_paths.extend(paths)
        return metadata_paths

    def update_score_metadata_from_tsv(
        self,
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["all", "auto", "ask"] = "all",
        write_empty_values: bool = False,
        remove_unused_fields: bool = False,
        write_text_fields: bool = False,
        update_instrumentation: bool = False,
    ) -> List[File]:
        """Update metadata fields of parsed scores with the values from the corresponding row in metadata.tsv.

        Args:
          view_name:
          force:
          choose:
          write_empty_values:
              If set to True, existing values are overwritten even if the new value is empty, in which case the field
              will be set to ''.
          remove_unused_fields:
              If set to True, all non-default fields that are not among the columns of metadata.tsv (anymore) are
              removed.
          write_text_fields:
              If set to True, ms3 will write updated values from the columns ``title_text``, ``subtitle_text``,
              ``composer_text``, ``lyricist_text``, and ``part_name_text`` into the score headers.
          update_instrumentation:
              Set to True to update the score's instrumentation based on changed values from 'staff_<i>_instrument'
              columns.

        Returns:
          List of File objects of those scores of which the XML structure has been modified.
        """
        updated_scores = []
        for _, corpus in self.iter_corpora(view_name):
            modified = corpus.update_score_metadata_from_tsv(
                view_name=view_name,
                force=force,
                choose=choose,
                write_empty_values=write_empty_values,
                remove_unused_fields=remove_unused_fields,
                write_text_fields=write_text_fields,
                update_instrumentation=update_instrumentation,
            )
            updated_scores.extend(modified)
        return updated_scores

    def update_scores(
        self,
        root_dir: Optional[str] = None,
        folder: str = ".",
        suffix: str = "",
        overwrite: bool = False,
    ) -> List[str]:
        """Update scores created with an older MuseScore version to the latest MuseScore 3 version.

        Args:
          root_dir:
              In case you want to create output paths for the updated MuseScore files based on a folder different
              from :attr:`corpus_path`.
          folder:
              * The default '.' has the updated scores written to the same directory as the old ones, effectively
                overwriting them if ``root_dir`` is None.
              * If ``folder`` is None, the files will be written to ``{root_dir}/scores/``.
              * If ``folder`` is an absolute path, ``root_dir`` will be ignored.
              * If ``folder`` is a relative path starting with a dot ``.`` the relative path is appended to the file's
                subdir.
                For example, ``../scores`` will resolve to a sibling directory of the one where the ``file`` is located.
              * If ``folder`` is a relative path that does not begin with a dot ``.``, it will be appended to the
                ``root_dir``.
          suffix: String to append to the file names of the updated files, e.g. '_updated'.
          overwrite: By default, existing files are not overwritten. Pass True to allow this.

        Returns:
          A list of all up-to-date paths, whether they had to be converted or were already in the latest version.
        """
        up2date_paths = []
        for _, corpus in self.iter_corpora():
            paths = corpus.update_scores(
                root_dir=root_dir, folder=folder, suffix=suffix, overwrite=overwrite
            )
            up2date_paths.extend(paths)
        return up2date_paths

    def update_tsvs_on_disk(
        self,
        facets: ScoreFacets = "tsv",
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["auto", "ask"] = "auto",
    ) -> List[str]:
        """
        Update existing TSV files corresponding to one or several facets with information freshly extracted from a
        parsed score, but only if the contents are identical. Otherwise, the existing TSV file is not overwritten and
        the differences are displayed in a log warning. The purpose is to safely update the format of existing TSV
        files, (for instance with respect to column order) making sure that the content doesn't change.

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
            paths.extend(
                corpus.update_tsvs_on_disk(
                    facets=facets, view_name=view_name, force=force, choose=choose
                )
            )
        return paths

    def _aggregate_corpus_data(
        self, method, view_name=None, concatenate=False, **kwargs
    ):
        result = {}
        for corpus_name, corpus in self.iter_corpora(view_name):
            corpus_method = getattr(corpus, method)
            if method == "get_facet":
                kwargs["concatenate"] = False
            corpus_result = corpus_method(view_name=view_name, **kwargs)
            for piece, piece_result in corpus_result.items():
                if method == "get_facet":
                    piece_result = [piece_result]
                result[(corpus_name, piece)] = piece_result
        if concatenate:
            keys, dataframes = [], []
            flat = "flat" not in kwargs or kwargs["flat"]
            if flat:
                add_index_level = any(
                    len(piece_result) > 1 for piece_result in result.values()
                )
            else:
                add_index_level = any(
                    len(file_dataframe_tuples) > 1
                    for piece_result in result.values()
                    for file_dataframe_tuples in piece_result.values()
                )
            for corpus_piece, piece_result in result.items():
                if flat:
                    n_tuples = len(piece_result)
                    if n_tuples == 0:
                        continue
                    keys.append(corpus_piece)
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
                        keys.append(corpus_piece + (facet,))
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
                                loc = df.columns.get_loc("form_label")
                                adapted_dataframes.append(
                                    pd.concat(
                                        [df.iloc[:, :loc], df.iloc[:, loc:]],
                                        keys=["", "a"],
                                        axis=1,
                                    )
                                )
                        result = pd.concat(adapted_dataframes, keys=keys)
                    else:
                        raise
                nlevels = result.index.nlevels
                level_names = ["corpus", "piece"]
                if not flat:
                    level_names.append("facet")
                if len(level_names) < nlevels - 1:
                    level_names.append("ix")
                level_names.append("i")
                result.index.rename(level_names, inplace=True)
            else:
                return pd.DataFrame()
        return result

    def _get_parsed_score_files(
        self, view_name: Optional[str] = None
    ) -> Dict[CorpusFnameTuple, FileList]:
        result = {}
        for corpus_name, corpus in self.iter_corpora(view_name=view_name):
            piece2files = corpus.get_files(
                "scores", view_name=view_name, unparsed=False, flat=True
            )
            result[corpus_name] = sum(piece2files.values(), [])
        return result

    def _get_unparsed_score_files(
        self, view_name: Optional[str] = None
    ) -> Dict[CorpusFnameTuple, FileList]:
        result = {}
        for corpus_name, corpus in self.iter_corpora(view_name=view_name):
            piece2files = corpus.get_files(
                "scores", view_name=view_name, parsed=False, flat=True
            )
            result[corpus_name] = sum(piece2files.values(), [])
        return result

    def _get_parsed_tsv_files(
        self, view_name: Optional[str] = None, flat: bool = True
    ) -> Dict[CorpusFnameTuple, Union[FileDict, FileList]]:
        result = {}
        for corpus_name, corpus in self.iter_corpora(view_name=view_name):
            piece2files = corpus.get_files(
                "tsv", view_name=view_name, unparsed=False, flat=flat
            )
            if flat:
                result[corpus_name] = sum(piece2files.values(), [])
            else:
                dd = defaultdict(list)
                for piece, typ2files in piece2files.items():
                    for typ, files in typ2files.items():
                        dd[typ].extend(files)
                result[corpus_name] = dict(dd)
        return result

    def _get_unparsed_tsv_files(
        self, view_name: Optional[str] = None, flat: bool = True
    ) -> Dict[CorpusFnameTuple, Union[FileDict, FileList]]:
        result = {}
        for corpus_name, corpus in self.iter_corpora(view_name=view_name):
            piece2files = corpus.get_files(
                "tsv", view_name=view_name, parsed=False, flat=flat
            )
            if flat:
                result[corpus_name] = sum(piece2files.values(), [])
            else:
                dd = defaultdict(list)
                for piece, typ2files in piece2files.items():
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
            raise AttributeError(
                f"'{view_name}' is not an existing view. Use _.get_view('{view_name}') to create it."
            )

    def detach_labels(
        self,
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["auto", "ask"] = "auto",
        key: str = "removed",
        staff: int = None,
        voice: Literal[1, 2, 3, 4] = None,
        harmony_layer: Optional[Literal[0, 1, 2, 3]] = None,
        delete: bool = True,
    ):
        for name, corpus in self.iter_corpora(view_name):
            corpus.detach_labels(
                view_name=view_name,
                force=force,
                choose=choose,
                key=key,
                staff=staff,
                voice=voice,
                harmony_layer=harmony_layer,
                delete=delete,
            )

    def score_metadata(self, view_name: Optional[str] = None) -> pd.DataFrame:
        metadata_dfs = {
            corpus_name: corpus.score_metadata(view_name=view_name)
            for corpus_name, corpus in self.iter_corpora(view_name=view_name)
        }
        metadata = pd.concat(
            metadata_dfs.values(), keys=metadata_dfs.keys(), names=["corpus", "piece"]
        )
        return metadata

    def metadata(
        self,
        view_name: Optional[str] = None,
        choose: Optional[Literal["auto", "ask"]] = None,
    ) -> pd.DataFrame:
        metadata_dfs = {
            corpus_name: corpus.metadata(view_name=view_name, choose=choose)
            for corpus_name, corpus in self.iter_corpora(view_name=view_name)
        }
        metadata = pd.concat(
            metadata_dfs.values(), keys=metadata_dfs.keys(), names=["corpus", "piece"]
        )
        return metadata

    def metadata_tsv(self, view_name: Optional[str] = None) -> pd.DataFrame:
        """Concatenates the 'metadata.tsv' (as they come) files for all corpora with a [corpus, piece] MultiIndex. If
        you need metadata that filters out pieces according to the current view, use :meth:`metadata`.
        """
        metadata_dfs = {
            corpus_name: enforce_piece_index_for_metadata(corpus.metadata_tsv)
            for corpus_name, corpus in self.iter_corpora(view_name=view_name)
            if corpus.metadata_tsv is not None
        }
        for corpus_name, df in metadata_dfs.items():
            try:
                rel_path_col = next(
                    col for col in ("subdirectory", "rel_paths") if col in df.columns
                )
            except StopIteration:
                raise ValueError(
                    "Metadata is expected to come with a column called 'subdirectory' or (previously) 'rel_paths'."
                )
            subdirectories = [
                "/".join((corpus_name, subdirectory))
                for subdirectory in df[rel_path_col]
            ]
            df.loc[:, rel_path_col] = subdirectories
            if "rel_path" in df.columns:
                rel_paths = [
                    "/".join((corpus_name, rel_path)) for rel_path in df["rel_path"]
                ]
                df.loc[:, "rel_path"] = rel_paths
        metadata = pd.concat(metadata_dfs, names=["corpus", "piece"])
        return metadata

    def store_extracted_facets(
        self,
        view_name: Optional[str] = None,
        root_dir: Optional[str] = None,
        measures_folder: Optional[str] = None,
        notes_folder: Optional[str] = None,
        rests_folder: Optional[str] = None,
        notes_and_rests_folder: Optional[str] = None,
        labels_folder: Optional[str] = None,
        expanded_folder: Optional[str] = None,
        form_labels_folder: Optional[str] = None,
        cadences_folder: Optional[str] = None,
        events_folder: Optional[str] = None,
        chords_folder: Optional[str] = None,
        metadata_suffix: Optional[str] = None,
        markdown: bool = True,
        simulate: bool = False,
        unfold: bool = False,
        interval_index: bool = False,
    ):
        """Store facets extracted from parsed scores as TSV files.

        Args:
          view_name:
          root_dir:
              ('measures', 'notes', 'rests', 'notes_and_rests', 'labels', 'expanded', 'form_labels', 'cadences',
               'events', 'chords')

          measures_folder, notes_folder, rests_folder, notes_and_rests_folder, labels_folder, expanded_folder,
          form_labels_folder, cadences_folder, events_folder, chords_folder:
              Specify directory where to store the corresponding TSV files.
          metadata_suffix:
              Specify a suffix to update the 'metadata{suffix}.tsv' file for each corpus. For the main file, pass ''
          markdown:
              By default, when ``metadata_path`` is specified, a markdown file called ``README.md`` containing
              the columns [file_name, measures, labels, standard, annotators, reviewers] is created. If it exists
              already, this table will be appended or overwritten after the heading ``# Overview``.
          simulate:
          unfold:
              By default, repetitions are not unfolded. Pass True to duplicate values so that they correspond to a full
              playthrough, including correct positioning of first and second endings.
          interval_index:

        Returns:

        """
        for corpus_name, corpus in self.iter_corpora(view_name=view_name):
            corpus.store_extracted_facets(
                view_name=view_name,
                root_dir=root_dir,
                measures_folder=measures_folder,
                notes_folder=notes_folder,
                rests_folder=rests_folder,
                notes_and_rests_folder=notes_and_rests_folder,
                labels_folder=labels_folder,
                expanded_folder=expanded_folder,
                form_labels_folder=form_labels_folder,
                cadences_folder=cadences_folder,
                events_folder=events_folder,
                chords_folder=chords_folder,
                metadata_suffix=metadata_suffix,
                markdown=markdown,
                simulate=simulate,
                unfold=unfold,
                interval_index=interval_index,
            )

    def store_parsed_scores(
        self,
        view_name: Optional[str] = None,
        only_changed: bool = True,
        root_dir: Optional[str] = None,
        folder: str = ".",
        suffix: str = "",
        overwrite: bool = False,
        simulate=False,
    ) -> Dict[str, List[str]]:
        """Stores all parsed scores under this view as MuseScore 3 files.

        Args:
          view_name: Name of another view if another than the current one is to be used.
          only_changed:
              By default, only scores that have been modified since parsing are written. Set to False to store
              all scores regardless.
          root_dir: Directory where to re-build the sub-directory tree of the :obj:`Corpus` in question.
          folder:

              * Different behaviours are available. Note that only the third option ensures that file paths are distinct
                for files that have identical pieces but are located in different subdirectories of the same corpus.
              * If ``folder`` is None (default), the files' type will be appended to the ``root_dir``.
              * If ``folder`` is an absolute path, ``root_dir`` will be ignored.
              * If ``folder`` is a relative path that does not begin with a dot ``.``, it will be appended to the
                ``root_dir``.
              * If ``folder`` is a relative path starting with a dot ``.`` the relative path is appended to the file's
                subdir. For example, ``..\notes`` will resolve to a sibling directory of the one where the ``file``
                is located.

          suffix: Suffix to append to the original file name.
          overwrite: Pass True to overwrite existing files.
          simulate: Set to True if no files are to be written.

        Returns:
          Paths of the stored files.
        """
        paths = {}
        for corpus_name, corpus in self.iter_corpora(view_name):
            paths[corpus_name] = corpus.store_parsed_scores(
                view_name=view_name,
                only_changed=only_changed,
                root_dir=root_dir,
                folder=folder,
                suffix=suffix,
                overwrite=overwrite,
                simulate=simulate,
            )
        return paths

    def parse(
        self,
        view_name=None,
        level=None,
        parallel=True,
        only_new=True,
        labels_cfg={},
        cols={},
        infer_types=None,
        **kwargs,
    ):
        """Shorthand for executing parse_scores and parse_tsv at a time.
        Args:
          view_name:
        """
        self.parse_scores(
            level=level,
            parallel=parallel,
            only_new=only_new,
            labels_cfg=labels_cfg,
            view_name=view_name,
        )
        self.parse_tsv(
            view_name=view_name,
            level=level,
            cols=cols,
            infer_types=infer_types,
            only_new=only_new,
            **kwargs,
        )

    def parse_scores(
        self,
        level: str = None,
        parallel: bool = True,
        only_new: bool = True,
        labels_cfg: dict = {},
        view_name: str = None,
        choose: Literal["all", "auto", "ask"] = "all",
    ):
        """Parse MuseScore 3 files (MSCX or MSCZ) and store the resulting read-only Score objects. If they need
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
            corpus.parse_scores(
                level=level,
                parallel=parallel,
                only_new=only_new,
                labels_cfg=labels_cfg,
                view_name=view_name,
                choose=choose,
            )

    def parse_tsv(
        self,
        view_name=None,
        level=None,
        cols={},
        infer_types=None,
        only_new=True,
        choose: Literal["all", "auto", "ask"] = "all",
        **kwargs,
    ):
        """Parse TSV files (or other value-separated files such as CSV) to be able to do something with them.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`~collections.abc.Collection`, optional
            Key(s) for which to parse all non-MSCX files.  By default, all keys are selected.
        ids : :obj:`~collections.abc.Collection`
            To parse only particular files, pass there IDs. ``keys`` and ``fexts`` are ignored in this case.
        fexts :  :obj:`str` or :obj:`~collections.abc.Collection`, optional
            If you want to parse only files with one or several particular file extension(s), pass the extension(s)
        cols : :obj:`dict`, optional
            By default, if a column called ``'label'`` is found, the TSV is treated as an annotation table and turned
            into an Annotations object. Pass one or several column name(s) to treat *them* as label columns instead.
            If you pass ``{}`` or no label column is found, the TSV is parsed as a "normal" table, i.e. a DataFrame.
        infer_types : :obj:`dict`, optional
            To recognize one or several custom label type(s), pass ``{name: regEx}``.
        level : {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}, optional
            Pass a level name for which (and above which) you want to see log records.
        **kwargs:
            Arguments for :py:meth:`pandas.DataFrame.to_csv`. Defaults to ``{'sep': '\t', 'index': False}``. In
            particular, you might want to update the default dictionaries for ``dtypes`` and ``converters`` used in
            :py:func:`load_tsv`.

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
            corpus.parse_tsv(
                view_name=view_name,
                cols=cols,
                infer_types=infer_types,
                only_new=only_new,
                choose=choose,
                **kwargs,
            )

    def __getitem__(self, item) -> Corpus:
        if isinstance(item, str):
            return self.get_corpus(item)
        elif isinstance(item, tuple):
            if len(item) == 1:
                return self.get_corpus(item[0])
            if len(item) == 2:
                corpus_name, piece_or_ix = item
                return self.get_corpus(corpus_name)[piece_or_ix]
            corpus_name, *remainder = item
            return self.get_corpus(corpus_name)[tuple(remainder)]

    def __iter__(self) -> Iterator[Tuple[str, Corpus]]:
        """Iterate through all (corpus_name, Corpus) tuples, regardless of any Views.

        Yields: (corpus_name, Corpus) tuples
        """
        yield from self.corpus_objects.items()

    def __repr__(self):
        """Show the :meth:`info` under the active view."""
        return self.info(return_str=True)

    @property
    def parsed_mscx(self, *args, **kwargs) -> pd.DataFrame:
        """Deprecated property. Replaced by :attr:`n_parsed_scores`"""
        raise DeprecationWarning("Property has been renamed to n_parsed_scores.")

    @property
    def parsed_tsv(self, *args, **kwargs) -> pd.DataFrame:
        """Deprecated property. Replaced by :attr:`n_parsed_tsvs`"""
        raise DeprecationWarning("Property has been renamed to n_parsed_tsvs.")

    def add_detached_annotations(self, *args, **kwargs):
        """Deprecated method. Replaced by :meth:`insert_detached_labels`."""
        raise DeprecationWarning(
            "Method not in use any more. Use Parse.insert_detached_labels()."
        )

    def count_annotation_layers(self, *args, **kwargs):
        """Deprecated method."""
        raise DeprecationWarning("Method not in use any more.")

    def count_labels(self, *args, **kwargs):
        """Deprecated method."""
        raise DeprecationWarning("Method not in use any more.")

    def get_lists(self, *args, **kwargs):
        """Deprecated method. Replaced by :meth:`get_facets`."""
        raise DeprecationWarning(
            "Method get_lists() not in use any more. Use Parse.get_facets() instead."
        )

    def iter(self, *args, **kwargs):
        """Deprecated method. Replaced by :meth:`ms3.corpus.Corpus.iter_facets`."""
        raise DeprecationWarning(
            "Method iter() not in use any more. Use Corpus.iter_facets() instead."
        )

    def parse_mscx(self, *args, **kwargs):
        """Deprecated method. Replaced by :meth:`parse_scores`."""
        raise DeprecationWarning(
            "Method not in use any more. Use Parse.parse_scores()."
        )

    def pieces(self, *args, **kwargs):
        """Deprecated method. Replaced by :meth:`info`."""
        raise DeprecationWarning(
            "This method is deprecated. To view pieces, call Corpus.info() or Corpus.info('all'). "
            "A DataFrame showing all detected files is available under the property Corpus.files_df"
        )

    def store_scores(self, *args, **kwargs):
        """Deprecated method. Replaced by :meth:`store_parsed_scores`."""
        raise DeprecationWarning(
            "Method not in use any more. Use Parse.store_parsed_scores()."
        )

    def update_metadata(self, *args, **kwargs):
        """Deprecated method. Replaced by :meth:`update_score_metadata_from_tsv`."""
        raise DeprecationWarning(
            "Method not in use any more. Use Parse.update_score_metadata_from_tsv()."
        )


# ######################################################################################################################
# ######################################################################################################################
# ############################################## End of Parse() ########################################################
# ######################################################################################################################
# ######################################################################################################################
