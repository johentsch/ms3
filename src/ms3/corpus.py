import os
import re
from collections import Counter, defaultdict
from functools import cache
from itertools import zip_longest
from logging import Logger
from typing import (
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import git
import numpy as np
import pandas as pd
import pathos.multiprocessing as mp
from ms3.utils.frictionless_helpers import store_dataframe_resource
from ms3.utils.functions import (
    compute_path_from_file,
    get_git_repo,
    get_git_version_info,
    get_name_of_highest_version_tag,
)

from ._typing import (
    AnnotationsFacet,
    Facet,
    FacetArguments,
    FileDataframeTuple,
    FileDict,
    FileList,
    FileParsedTuple,
    FileScoreTuple,
    ParsedFile,
    ScoreFacet,
    ScoreFacets,
)
from .logger import (
    LoggedClass,
    get_log_capture_handler,
    get_logger,
    normalize_logger_name,
)
from .piece import Piece
from .score import Score, compare_two_score_objects
from .utils import (
    File,
    ask_user_to_choose,
    available_views2str,
    check_argument_against_literal_type,
    column_order,
    convert,
    disambiguate_files,
    enforce_piece_index_for_metadata,
    files2disambiguation_dict,
    get_musescore,
    join_tsvs,
    load_tsv,
    make_file_path,
    parse_ignored_warnings_file,
    parse_tsv_file_at_git_revision,
    path2type,
    prepare_metadata_for_writing,
    pretty_dict,
    resolve_dir,
    resolve_facets_param,
    resolve_paths_argument,
    scan_directory,
    string2identifier,
    update_labels_cfg,
    write_markdown,
    write_metadata,
)
from .utils.constants import LATEST_MUSESCORE_VERSION, METADATA_COLUMN_ORDER
from .view import DefaultView, View, create_view_from_parameters


class Corpus(LoggedClass):
    """
    Collection of scores and TSV files that can be matched to each other based on their file names.
    """

    default_count_index = pd.MultiIndex.from_product(
        [("scores",) + Score.dataframe_types, ("found", "parsed")]
    )

    def __init__(
        self,
        directory: str,
        view: View = None,
        only_metadata_pieces: bool = True,
        include_convertible: bool = False,
        include_tsv: bool = True,
        exclude_review: bool = True,
        file_re: Optional[Union[str, re.Pattern]] = None,
        folder_re: Optional[Union[str, re.Pattern]] = None,
        exclude_re: Optional[Union[str, re.Pattern]] = None,
        paths: Optional[Collection[str]] = None,
        labels_cfg={},
        ms=None,
        **logger_cfg,
    ):
        """

        Parameters
        ----------
        directory, key, index, file_re, folder_re, exclude_re, recursive : optional
            Arguments for the method :py:meth:`~ms3.parse.add_dir`.
            If ``dir`` is not passed, no files are added to the new object except if you pass ``paths``
        paths : :obj:`~collections.abc.Collection` or :obj:`str`, optional
            List of file paths you want to add. If ``directory`` is also passed, all files will be combined in the same
            object.
        logger_cfg : :obj:`dict`, optional
            | The following options are available:
            | 'name': LOGGER_NAME -> by default the logger name is based on the parsed file(s)
            | 'level': {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}
            | 'path': Directory in which log files are stored. If 'file' is relative, this path is used as root,
              otherwise, it is ignored.
            | 'file': PATH_TO_LOGFILE Pass absolute path to store all log messages in a single log file.
              If PATH_TO_LOGFILE is relative, multiple log files are created dynamically, relative to the original
              MSCX files' paths.  If 'path' is set, the corresponding subdirectory structure is created there.
        ms : :obj:`str`, optional
            If you pass the path to your local MuseScore 3 installation, ms3 will attempt to parse musicXML,
            MuseScore 2, and other formats by temporarily converting them. If you're using the standard path,
            you may try 'auto', or 'win' for Windows, 'mac' for MacOS, or 'mscore' for Linux. In case you do not pass
            the 'file_re' and the MuseScore executable is detected, all convertible files are automatically selected,
            otherwise only those that can be parsed without conversion.
        """
        directory = resolve_dir(directory)
        assert os.path.isdir(directory), f"{directory} is not an existing directory."
        self.corpus_path: str = directory
        """Path where the corpus is located."""
        self.name = os.path.basename(directory).strip(r"\/")
        """Folder name of the corpus."""
        if (
            "name" not in logger_cfg
            or logger_cfg["name"] is None
            or logger_cfg["name"] == ""
        ):
            logger_cfg["name"] = "ms3.Corpus." + self.name.replace(".", "")
        # if 'level' not in logger_cfg or (logger_cfg['level'] is None):
        #     logger_cfg['level'] = 'w'
        super().__init__(subclass="Corpus", logger_cfg=logger_cfg)

        self.repo: Optional[git.Repo] = None
        """If the corpus is part of a git repository, this attribute holds the corresponding :obj:`git.Repo` object."""
        self.repo = get_git_repo(directory, logger=self.logger)

        self.files: List[File] = []
        """
        ``[File]`` list of :obj:`File` data objects containing information on the file location
        etc. for all detected files.
        """

        self._views: dict = {}
        if view is None:
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
            self._views[None] = initial_view
        else:
            legacy_params = any(
                param is not None for param in (paths, file_re, folder_re, exclude_re)
            )
            not_default = (
                not only_metadata_pieces
                or not include_tsv
                or not exclude_review
                or include_convertible
            )
            if legacy_params or not_default:
                self.logger.warning(
                    "If you pass an existing view, other view-related parameters are ignored."
                )
            self._views[None] = view
        if "default" not in self.view_names:
            self._views["default"] = DefaultView(level=self.logger.getEffectiveLevel())
        if "all" not in self.view_names:
            self._views["all"] = View(level=self.logger.getEffectiveLevel())

        self._ms = get_musescore(ms, logger=self.logger)
        """:obj:`str`
        Path or command of the local MuseScore 3 installation if specified by the user."""

        self._ignored_warnings = defaultdict(list)
        """:obj:`collections.defaultdict`
        {'logger_name' -> [(message_id), ...]} This dictionary stores the warnings to be ignored
        upon loading them from an IGNORED_WARNINGS file.
        """

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

        self._pieces: Dict[str, Piece] = {}
        """{piece -> :class:`Piece`} The objects holding together information for the same piece
        from various files.
        """

        self.metadata_tsv: pd.DataFrame = None
        """The parsed 'metadata.tsv' file for the corpus."""

        self.metadata_ix: int = None
        """The index of the 'metadata.tsv' file for the corpus."""

        self.ix2pname: Dict[int, str] = {}
        """{ix -> piece name} dict for associating files with the piece they have been matched to.
        None for indices that could not be matched, e.g. metadata.
        """

        self.ix2metadata_file: Dict[int, File] = {}
        """{ix -> File} dict for collecting all metadata files."""

        self.ix2orphan_file: Dict[int, File] = {}
        """{ix -> File} dict for collecting all metadata files."""

        self._ix2parsed: Dict[int, pd.DataFrame] = {}
        """{ix -> DataFrame} dictionary for parsed metadata and orphan TSV files. Managed by :attr:`ix2parsed`."""

        self.score_fnames: List[str] = []
        """Sorted list of unique file names of all detected scores"""

        self.detect_parseable_files()
        assert (
            len(self.files) > 0
        ), f"The path {self.corpus_path} contains no files that I can parse."
        self.collect_fnames_from_scores()
        self.find_and_load_metadata()
        self.create_pieces()
        self.look_for_ignored_warnings()
        self.register_files_with_pieces()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%% END of __init__() %%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    @property
    def pnames(self) -> List[str]:
        """All piece names including those of scores that are not listed in metadata.tsv"""
        return self.get_all_pnames()

    @property
    def ix2file(self) -> dict:
        return dict(enumerate(self.files))

    @property
    def ix2parsed(self) -> Dict[int, ParsedFile]:
        result = dict(self._ix2parsed)
        for _, piece in self:
            result.update(piece.ix2parsed)
        return result

    @property
    def ix2parsed_score(self) -> Dict[int, Score]:
        result = {}
        for _, piece in self:
            result.update(piece.ix2parsed_score)
        return result

    @property
    def ix2parsed_tsv(self) -> Dict[int, pd.DataFrame]:
        result = {}
        for _, piece in self:
            result.update(piece.ix2parsed_tsv)
        return result

    @property
    def ix2annotations(self) -> Dict[int, ParsedFile]:
        result = {}
        for _, piece in self:
            result.update(piece.ix2annotations)
        return result

    @property
    def ms(self):
        return self._ms

    @ms.setter
    def ms(self, ms):
        executable = get_musescore(ms, logger=self.logger)
        if executable is None:
            raise FileNotFoundError(
                f"'{ms}' did not lead me to a MuseScore executable."
            )
        if executable is not None:
            self._ms = executable
            for _, piece in self.__iter__():
                piece._ms = executable

    @property
    def n_detected(self):
        return len(self.files)

    @property
    def n_orphans(self):
        return len(self.ix2orphan_file)

    @property
    def n_parsed(self):
        return sum(file.ix not in self._ix2parsed for file in self.ix2parsed.values())

    @property
    def n_parsed_scores(self):
        return len(self.ix2parsed_score)

    @property
    def n_unparsed_scores(self):
        return sum(
            file.ix not in self.ix2parsed_score
            for file in self.files
            if file.type == "scores"
        )

    @property
    def n_parsed_tsvs(self):
        return len(self.ix2parsed_tsv)

    @property
    def n_pieces(self) -> int:
        """Number of all available pieces ('pieces'), independent of the view."""
        return len(self._pieces)

    @property
    def n_unparsed_tsvs(self):
        return sum(
            file.type != "scores"
            and file.ix not in self.ix2parsed_tsv
            and file.ix not in self.ix2orphan_file
            and file.ix not in self.ix2metadata_file
            for file in self.files
        )

    @property
    def n_annotations(self):
        return len(self.ix2annotations)

    def add_dir(
        self,
        directory: str,
        filter_other_pieces: bool = False,
        file_re: str = r".*",
        folder_re: str = r".*",
        exclude_re: str = r"^(\.|_)",
    ) -> FileList:
        """Add additional files pertaining to the already existing pieces of the corpus.

        If you want to use a directory with other pieces, create another :obj:`Corpus` object or combine several
        corpora in a :obj:`Parse` object.

        Args:
          directory:
              Directory to scan for parseable (score or TSV) files. Only those that begin with one of the corpus's
              pieces will be matched and registered, the others will be kept under :attr:`ix2orphan_file`.
          filter_other_pieces:
              Set to True if you want to filter out all pieces that were not matched up with one of the added files.
              This can be useful if you're loading TSV files with labels and want to parse only the scores for which
              you have added labels.
          file_re, folder_re:
              Regular expressions for filtering certain file names or folder names.
              The regEx are checked with search(), not match(), allowing for fuzzy search.
          exclude_re:
              Exclude files and folders containing this regular expression.

        Returns:
          List of :obj:`File` objects pertaining to the matched, newly added paths.
        """
        directory = resolve_dir(directory)
        all_file_paths = list(
            scan_directory(
                directory, file_re=file_re, folder_re=folder_re, exclude_re=exclude_re
            )
        )
        added_files = self.add_file_paths(all_file_paths)
        self.logger.debug(f"{len(added_files)} files added to the corpus.")
        if filter_other_pieces:
            new_view = self.view.copy()
            new_view.include("pieces", *(re.escape(f.piece) for f in added_files))
            self.set_view(new_view)
        return added_files

    def cadences(
        self,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
        concatenate: bool = True,
    ) -> Union[Dict[str, FileParsedTuple], pd.DataFrame]:
        return self.get_facet(
            "cadences",
            view_name=view_name,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
            concatenate=concatenate,
        )

    def chords(
        self,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
        concatenate: bool = True,
    ) -> Union[Dict[str, FileParsedTuple], pd.DataFrame]:
        return self.get_facet(
            "chords",
            view_name=view_name,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
            concatenate=concatenate,
        )

    def events(
        self,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
        concatenate: bool = True,
    ) -> Union[Dict[str, FileParsedTuple], pd.DataFrame]:
        return self.get_facet(
            "events",
            view_name=view_name,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
            concatenate=concatenate,
        )

    def expanded(
        self,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
        concatenate: bool = True,
    ) -> Union[Dict[str, FileParsedTuple], pd.DataFrame]:
        return self.get_facet(
            "expanded",
            view_name=view_name,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
            concatenate=concatenate,
        )

    def form_labels(
        self,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
        concatenate: bool = True,
    ) -> Union[Dict[str, FileParsedTuple], pd.DataFrame]:
        return self.get_facet(
            "form_labels",
            view_name=view_name,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
            concatenate=concatenate,
        )

    def labels(
        self,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
        concatenate: bool = True,
    ) -> Union[Dict[str, FileParsedTuple], pd.DataFrame]:
        return self.get_facet(
            "labels",
            view_name=view_name,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
            concatenate=concatenate,
        )

    def measures(
        self,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
        concatenate: bool = True,
    ) -> Union[Dict[str, FileParsedTuple], pd.DataFrame]:
        return self.get_facet(
            "measures",
            view_name=view_name,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
            concatenate=concatenate,
        )

    def notes(
        self,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
        concatenate: bool = True,
    ) -> Union[Dict[str, FileParsedTuple], pd.DataFrame]:
        return self.get_facet(
            "notes",
            view_name=view_name,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
            concatenate=concatenate,
        )

    def notes_and_rests(
        self,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
        concatenate: bool = True,
    ) -> Union[Dict[str, FileParsedTuple], pd.DataFrame]:
        return self.get_facet(
            "notes_and_rests",
            view_name=view_name,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
            concatenate=concatenate,
        )

    def rests(
        self,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
        concatenate: bool = True,
    ) -> Union[Dict[str, FileParsedTuple], pd.DataFrame]:
        return self.get_facet(
            "rests",
            view_name=view_name,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
            concatenate=concatenate,
        )

    def __getattr__(self, view_name):
        if view_name in self.view_names:
            if view_name != self.view_name:
                self.switch_view(view_name, show_info=False)
            return self
        else:
            raise AttributeError(
                f"'{view_name}' is not an existing view. Use _.get_view('{view_name}') to create it."
            )

    def ix_logger(self, ix):
        return get_logger(self.logger_names[ix])

    def count_changed_scores(self, view_name: Optional[str] = None):
        return sum(
            piece.count_changed_scores(view_name)
            for _, piece in self.iter_pieces(view_name)
        )

    def count_files(
        self,
        detected: bool = True,
        parsed: bool = True,
        as_dict: bool = False,
        drop_zero: bool = True,
        view_name: Optional[str] = None,
    ) -> Union[pd.DataFrame, dict]:
        assert detected + parsed > 0, "At least one parameter needs to be True"
        piece2counts = {}
        prefix = detected + parsed == 2
        for pname, piece_obj in self.iter_pieces(view_name=view_name):
            if detected:
                type_count = piece_obj.count_detected(
                    view_name=view_name, include_empty=True, prefix=prefix
                )
                if not parsed:
                    piece2counts[pname] = type_count
            if parsed:
                parsed_count = piece_obj.count_parsed(
                    view_name=view_name, include_empty=True, prefix=prefix
                )
                if not detected:
                    piece2counts[pname] = parsed_count
            if detected & parsed:
                alternating_counts = {}
                for (k1, v1), (k2, v2) in zip_longest(
                    type_count.items(), parsed_count.items(), fillvalue=(None, None)
                ):
                    if k1 is not None:
                        alternating_counts[k1] = v1
                    if k2 is not None:
                        alternating_counts[k2] = v2
                piece2counts[pname] = alternating_counts
        if as_dict:
            return piece2counts
        df = pd.DataFrame.from_dict(piece2counts, orient="index")
        if prefix:
            try:
                df.columns = (
                    df.columns.astype(str).str.split("_", n=1, expand=True).swaplevel()
                )
            except TypeError:
                pass
        if drop_zero:
            empty_cols = df.columns[df.sum() == 0]
            return df.drop(columns=empty_cols)
        return df

    def _summed_file_count(
        self, types=True, parsed=True, view_name: Optional[str] = None
    ) -> pd.Series:
        """The sum of _.count_files() but returning zero-filled Series if no pieces have been selected."""
        file_count = self.count_files(
            detected=types, parsed=parsed, drop_zero=False, view_name=view_name
        )
        if len(file_count) == 0 and types and parsed:
            return pd.Series(0, index=self.default_count_index)
        return file_count.sum()

    def add_file_paths(self, paths: Collection[str]) -> FileList:
        """Iterates through the given paths, converts those that correspond to parseable files to :obj:`File` objects
        (trying to infer their type from the path), and appends those to :attr:`files`.

        Args:
          paths: File paths that are to be registered with this Corpus object.

        Returns:
          A list of :obj:`File` objects corresponding to parseable files (based on their extensions).
        """
        resolved_paths = resolve_paths_argument(paths, logger=self.logger)
        if len(resolved_paths) == 0:
            return
        score_extensions = ["." + ext for ext in Score.parseable_formats]
        detected_extensions = score_extensions + [".tsv"]
        newly_added = []
        existing_paths = self.files_df.full_path.to_list()
        for full_path in resolved_paths:
            if full_path in existing_paths:
                self.logger.debug(f"Skipping {full_path} because it was already there.")
                continue
            current_path, file = os.path.split(full_path)
            file_name, file_ext = os.path.splitext(file)
            if file_ext not in detected_extensions:
                continue
            current_subdir = os.path.relpath(current_path, self.corpus_path)
            rel_path = os.path.join(current_subdir, file)
            file_type = path2type(full_path, logger=self.logger)
            if current_subdir.startswith("..") and file_type == "scores":
                self.logger.info(
                    f"The score {rel_path} lies outside the corpus folder {self.corpus_path}. "
                    f"In case this is the only score detected for piece '{file_name}', this will result in "
                    f"an invalid relative path in the metadata.tsv file, i.e. one that will not exist on other systems."
                )
            F = File(
                ix=len(self.files),
                type=file_type,
                file=file,
                piece=file_name,
                fext=file_ext,
                subdir=current_subdir,
                corpus_path=self.corpus_path,
                rel_path=rel_path,
                full_path=full_path,
                directory=current_path,
                suffix="",
            )
            self.files.append(F)
            newly_added.append(F)
        if len(self._pieces) > 0:
            self.register_files_with_pieces(newly_added)
        return newly_added

    def collect_fnames_from_scores(self) -> None:
        """Construct sorted list of pieces from all detected scores."""
        files_df = self.files_df
        detected_scores = files_df.loc[files_df.type == "scores"]
        self.score_fnames = sorted(detected_scores.piece.unique())

    def create_metadata_tsv(
        self,
        suffix="",
        view_name: Optional[str] = None,
        overwrite: bool = False,
        force: bool = True,
    ) -> Optional[str]:
        """Creates a 'metadata.tsv' file for the current view."""
        path = os.path.join(self.corpus_path, "metadata" + suffix + ".tsv")
        already_there = os.path.isfile(path)
        if already_there:
            if overwrite and suffix == "":
                self.logger.warning(
                    "For security reasons I won't overwrite the 'metadata.tsv' file of this corpus. "
                    "Consider using Corpus.update_metadata_tsv_from_parsed_scores() or delete it yourself."
                )
                return
            elif not overwrite:
                self.logger.warning(
                    f"{path} existed already. Consider using Corpus.update_metadata_tsv_from_parsed_scores()."
                )
                return
        metadata = self.score_metadata(view_name=view_name, choose=force)
        metadata = prepare_metadata_for_writing(metadata)
        metadata.to_csv(path, sep="\t", index=False)
        if already_there:
            self.logger.info(f"{path} overwritten.")
        else:
            self.logger.info(f"{path} created.")
        new_files = self.add_file_paths([path])
        if len(new_files) == 0:
            return
        file = new_files[0]
        self.load_metadata_file(file)

    def create_pieces(self, pnames: Union[Collection[str], str] = None) -> None:
        """Creates and stores one :obj:`Piece` object per piece."""
        if pnames is None:
            pnames = self.get_all_pnames()
        elif isinstance(pnames, str):
            pnames = [pnames]
        for pname in pnames:
            if pname in self._pieces:
                self.logger.debug(f"Piece({pname}) existed already, skipping...")
                continue
            logger_cfg = dict(self.logger_cfg)
            logger_name = self.logger.name + "." + pname.replace(".", "")
            logger_cfg["name"] = logger_name
            piece = Piece(
                pname,
                view=self.get_view(),
                labels_cfg=self.labels_cfg,
                ms=self.ms,
                **logger_cfg,
            )
            piece.set_view(
                **{
                    view_name: view
                    for view_name, view in self._views.items()
                    if view_name is not None
                }
            )
            self._pieces[pname] = piece
            self.logger_names[pname] = logger_name
        if self.metadata_tsv is not None:
            try:
                piece_col = next(
                    col
                    for col in ("piece", "fname", "fnames", "name", "names")
                    if col in self.metadata_tsv.columns
                )
            except StopIteration:
                file = self.files[self.metadata_ix]
                self.logger.warning(
                    f"Could not attribute metadata to Pieces because {file.rel_path} has no 'piece' column."
                )
                return
            metadata_rows = self.metadata_tsv.to_dict(orient="records")
            for row in metadata_rows:
                pname = row[piece_col]
                if pd.isnull(pname):
                    continue
                piece = self.get_piece(pname)
                piece._tsv_metadata = row

    def detect_parseable_files(self) -> None:
        """Walks through the corpus_path and collects information on all parseable files."""
        score_extensions = ["." + ext for ext in Score.parseable_formats]
        detected_extensions = score_extensions + [".tsv"]
        for current_path, subdirs, files in os.walk(self.corpus_path):
            subdirs[:] = sorted(sd for sd in subdirs if not sd.startswith("."))
            current_subdir = os.path.relpath(current_path, self.corpus_path)
            parseable_files = [
                (f,) + os.path.splitext(f)
                for f in files
                if os.path.splitext(f)[1] in detected_extensions
            ]
            for file, file_name, file_ext in sorted(parseable_files):
                full_path = os.path.join(current_path, file)
                rel_path = os.path.join(current_subdir, file)
                file_type = path2type(full_path, logger=self.logger)
                F = File(
                    ix=len(self.files),
                    type=file_type,
                    file=file,
                    piece=file_name,
                    fext=file_ext,
                    subdir=current_subdir,
                    corpus_path=self.corpus_path,
                    rel_path=rel_path,
                    full_path=full_path,
                    directory=current_path,
                    suffix="",
                )
                self.files.append(F)

    def disambiguate_facet(
        self, facet: Facet, view_name: Optional[str] = None, ask_for_input=True
    ) -> None:
        """Make sure that, for a given facet, the current view includes only one or zero files. If at least one piece
        has more than one file, the user will be asked which ones to use. The others will be excluded from the view.

        Args:
          facet: Which facet to disambiguate.
          ask_for_input:
              By default, if there is anything to disambiguate, the user is asked to select a group of files.
              Pass False to see only the questions and choices without actually disambiguating.
        """
        assert isinstance(
            facet, str
        ), f"Let's disambiguate one facet at a time. Received invalid argument {facet}"
        assert (
            facet in Facet.__args__
        ), f"'{facet}' is not a valid facet. Choose one of {Facet.__args__}."
        disambiguated = {}
        no_need = {}
        missing = []
        all_available_files = []
        selected_ixs = []
        piece2files = self.get_files(
            facet, view_name=view_name, choose="all", flat=True, include_empty=False
        )
        for peace, files in piece2files.items():
            if len(files) == 0:
                missing.append(peace)
            elif len(files) == 1:
                no_need[peace] = files[0]
            else:
                all_available_files.extend(files)
                disamb2file = files2disambiguation_dict(
                    files, include_disambiguator=True
                )
                disambiguated[peace] = {
                    disamb: file.ix for disamb, file in disamb2file.items()
                }
        if len(missing) > 0:
            self.logger.info(f"{len(missing)} files don't come with '{facet}'.")
        if len(disambiguated) == 0:
            self.logger.info(f"No files require disambiguation for facet '{facet}'.")
            return
        if len(no_need) > 0:
            self.logger.info(
                f"{len(no_need)} files do not require disambiguation for facet '{facet}'."
            )
        N_target = len(disambiguated)
        N_remaining = N_target
        print(f"{N_target} pieces require disambiguation for facet '{facet}'.")
        df = pd.DataFrame.from_dict(disambiguated, orient="index", dtype="Int64")
        n_files_per_disambiguator = df.notna().sum().sort_values(ascending=False)
        df = df.loc[:, n_files_per_disambiguator.index]
        n_choices_groups = df.notna().sum(axis=1).sort_values()
        gpb: Iterator[Tuple[int, pd.DataFrame]] = df.groupby(
            n_choices_groups
        )  # introduces variable just for type hinting
        for n_choices, chunk in gpb:
            range_str = f"1-{n_choices}"
            chunk = chunk.dropna(axis=1, how="all")
            choices_groups = chunk.apply(lambda S: tuple(S.index[S.notna()]), axis=1)
            # question_gpb: Iterator[Tuple[Dict[int, str], pd.DataFrame]] = chunk.groupby(choices_groups)
            question_gpb = chunk.groupby(choices_groups)
            n_questions = question_gpb.ngroups
            for i, (choices, piece_group) in enumerate(question_gpb, 1):
                N_current = len(piece_group)
                choices = dict(enumerate(choices, 1))
                piece_group = piece_group.dropna(axis=1, how="all").sort_index(
                    axis=1, key=lambda S: S.str.len()
                )
                remaining_string = (
                    "" if N_current == N_remaining else f" ({N_remaining} remaining)"
                )
                if N_current == 1:
                    for piece, first_row in piece_group.iterrows():
                        break
                    prompt = f"Choose one of the files (or 0 for none):\n{pretty_dict(choices)}"
                    choices_representation = f"'{piece}'{remaining_string} by choosing one of the {n_choices} files:\n"
                else:
                    try:
                        prompt = f"Choose one of the columns (or 0 for none):\n{pretty_dict(choices)}"
                    except ValueError:
                        print(choices, piece_group)
                        raise
                    choices_representation = (
                        f"the following {N_current} pieces{remaining_string} by choosing one of"
                        f" the {n_choices} columns:\n\n{piece_group.to_string()}\n"
                    )
                question_number = f"({i}/{n_questions}) " if n_questions > 1 else ""
                print(f"{question_number} Disambiguation for {choices_representation}")
                if ask_for_input:
                    print(prompt)
                    query = f"Selection [{range_str}]: "
                    column = ask_user_to_choose(query, list(choices.values()))
                    if column is not None:
                        selected_ixs.extend(piece_group[column].values)
                        print()
                N_remaining -= N_current
        if ask_for_input:
            excluded_file_paths = [
                file.full_path
                for file in all_available_files
                if file.ix not in selected_ixs
            ]
            view = self.get_view(view_name)
            view.excluded_file_paths.extend(excluded_file_paths)
        return

    def extract_facet(
        self,
        facet: ScoreFacet,
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
        concatenate: bool = True,
    ) -> Union[Dict[str, FileParsedTuple], pd.DataFrame]:
        view_name_display = self.view_name if view_name is None else view_name
        result = {}
        for piece, piece_obj in self.iter_pieces(view_name):
            file, df = piece_obj.extract_facet(
                facet=facet,
                view_name=view_name,
                force=force,
                choose=choose,
                unfold=unfold,
                interval_index=interval_index,
            )
            if df is None:
                self.logger.info(
                    f"The view '{view_name_display}' does not comprise a score for '{piece}' from which to "
                    f"extract {facet}."
                )
            else:
                result[piece] = df if concatenate else (file, df)
        if concatenate:
            if len(result) > 0:
                df = pd.concat(result.values(), keys=result.keys())
                df.index.rename(["piece", "i"], inplace=True)
                return df
            else:
                return pd.DataFrame()
        return result

    def extract_facets(
        self,
        facets: ScoreFacets = None,
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
        flat=False,
    ) -> Dict[
        str, Union[Dict[str, List[FileDataframeTuple]], List[FileDataframeTuple]]
    ]:
        """Retrieve a dictionary with the selected feature matrices extracted from the parsed scores.
        If you want to retrieve parsed TSV files, use :py:meth:`get_all_parsed`.

        """
        selected_facets = resolve_facets_param(facets, ScoreFacet, logger=self.logger)
        if choose == "ask":
            for facet in selected_facets:
                self.disambiguate_facet(facet, ask_for_input=True)
            choose = "auto"
        if force:
            self.check_number_of_unparsed_scores(view_name, choose)
        result = {}
        for piece, piece_obj in self.iter_pieces(view_name=view_name):
            type2file = piece_obj.extract_facets(
                facets=selected_facets,
                view_name=view_name,
                force=force,
                choose=choose,
                unfold=unfold,
                interval_index=interval_index,
                flat=flat,
            )
            result[piece_obj.name] = type2file
        return result

    @property
    def files_df(self):
        if len(self.files) == 0:
            return pd.DataFrame()
        return pd.DataFrame(self.files).set_index("ix")

    def find_and_load_metadata(self) -> None:
        """Checks if a 'metadata.tsv' is present at the default path and parses it."""
        metadata_path = os.path.join(self.corpus_path, "metadata.tsv")
        if not os.path.isfile(metadata_path):
            return
        try:
            file = next(file for file in self.files if file.full_path == metadata_path)
        except StopIteration:
            self.logger.warning(
                f"Metadata file exists but had not been among the detected files: {metadata_path}"
            )
            new_files = self.add_file_paths([metadata_path])
            if len(new_files) == 0:
                self.logger.error(
                    f"Unable to add {metadata_path} to the Corpus object."
                )
                return
            file = new_files[0]
        self.load_metadata_file(file)

    @cache
    def pieces_in_metadata(self, metadata_ix: Optional[int] = None) -> List[str]:
        """pieces (file names without extension and suffix) serve as IDs for pieces. Retrieve
        those that are listed in the 'metadata.tsv' file for this corpus. The argument is simply
        self.metadata_ix and serves caching of the results for multiple metadata.tsv files.
        """
        if metadata_ix is None:
            if self.metadata_ix is None:
                msg = "No metadata.tsv file has been detected for this Corpus object."
                if len(self.ix2metadata_file) > 0:
                    available = [
                        file
                        for ix, file in self.ix2metadata_file.items()
                        if ix in self._ix2parsed
                    ]
                    if len(available) > 0:
                        msg += f" However, the following metadata files are available: {available}"
                if self.view.pieces_not_in_metadata:
                    self.logger.info(msg)
                else:
                    self.logger.warning(msg)
                return []
            metadata_ix = self.metadata_ix
        if metadata_ix not in self.ix2metadata_file:
            self.logger.error(
                f"Index {metadata_ix} does not seem to belong to a metadata file."
            )
            return []
        if metadata_ix not in self._ix2parsed:
            self.logger.error(
                f"The metadata file at index {metadata_ix} has no parsed DataFrame."
            )
            return []
        metadata_df = self._ix2parsed[metadata_ix]
        try:
            piece_col = next(
                col
                for col in ("piece", "fname", "fnames", "name", "names")
                if col in metadata_df.columns
            )
            return sorted(
                str(piece)
                for piece in metadata_df[piece_col].unique()
                if not pd.isnull(piece)
            )
        except StopIteration:
            file = self.files[metadata_ix]
            self.logger.warning(
                f"The file {file.rel_path} is missing a column called 'piece' or 'pieces':\n{metadata_df.columns}"
            )
            return []

    def pieces_not_in_metadata(self) -> List[str]:
        """pieces (file names without extension and suffix) serve as IDs for pieces. Retrieve
        those that are not listed in the 'metadata.tsv' file for this corpus.
        """
        metadata_pieces = self.pieces_in_metadata(self.metadata_ix)
        # view = self.get_view(view_name)
        # filtered_score_pieces = view.filtered_tokens('pieces', self.score_pieces)
        if len(metadata_pieces) == 0:
            return self.score_fnames
        return [
            f
            for f in self.score_fnames
            if not (
                f in metadata_pieces
                or any(f.startswith(md_piece) for md_piece in metadata_pieces)
            )
        ]

    def get_changed_scores(
        self, view_name: Optional[str] = None, include_empty: bool = False
    ) -> Dict[str, List[FileScoreTuple]]:
        result = {
            piece: piece_obj.get_changed_scores(view_name)
            for piece, piece_obj in self.iter_pieces(view_name)
        }
        if include_empty:
            return result
        return {k: v for k, v in result.items() if len(v) > 0}

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
    ) -> Dict[str, Union[Dict[str, FileDataframeTuple], List[FileDataframeTuple]]]:
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
        view_name_display = self.view_name if view_name is None else view_name
        result = {}
        for piece, piece_obj in self.iter_pieces(view_name):
            file, df = piece_obj.get_facet(
                facet=facet,
                view_name=view_name,
                choose=choose,
                unfold=unfold,
                interval_index=interval_index,
            )
            if df is None:
                self.logger.info(
                    f"The view '{view_name_display}' does not comprise a score or TSV file for '{piece}' from which to "
                    f"retrieve {facet}."
                )
            else:
                result[piece] = df if concatenate else (file, df)
        if concatenate:
            if len(result) > 0:
                df = pd.concat(result.values(), keys=result.keys())
                df.index.rename(["piece", "i"], inplace=True)
                return df
            else:
                return pd.DataFrame()
        return result

    def get_facets(
        self,
        facets: ScoreFacets = None,
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["all", "auto", "ask"] = "all",
        unfold: bool = False,
        interval_index: bool = False,
        flat=False,
        include_empty: bool = False,
    ) -> Dict[str, Union[Dict[str, FileDataframeTuple], List[FileDataframeTuple]]]:
        """

        Args:
          facets:
          view_name:
          force:
              Only relevant when ``choose='all'``. By default, only scores and TSV files that have already been
              parsed are taken into account. Set ``force=True`` to force-parse all scores and TSV files selected
              under the given view.
          choose:
          unfold:
          interval_index:
          flat:
          include_empty:

        Returns:

        """
        selected_facets = resolve_facets_param(facets, ScoreFacet, logger=self.logger)
        if choose == "ask":
            for facet in selected_facets:
                self.disambiguate_facet(facet, ask_for_input=True)
            choose = "auto"
        if force:
            self.check_number_of_unparsed_scores(view_name, choose)
        result = {}
        for piece, piece_obj in self.iter_pieces(view_name=view_name):
            facet2parsed = piece_obj.get_facets(
                facets=selected_facets,
                view_name=view_name,
                force=force,
                choose=choose,
                unfold=unfold,
                interval_index=interval_index,
                flat=flat,
            )
            if not include_empty:
                if not flat:
                    facet2parsed = {
                        facet: parsed
                        for facet, parsed in facet2parsed.items()
                        if len(parsed) > 0
                    }
                if len(facet2parsed) == 0:
                    continue
            result[piece_obj.name] = facet2parsed
        return result

    def check_number_of_unparsed_scores(self, view_name, choose):
        tmp_choose = "auto" if choose == "ask" else choose
        unparsed_files = self.get_files(
            "scores", view_name=view_name, parsed=False, choose=tmp_choose, flat=True
        )
        n_unparsed = len(sum(unparsed_files.values(), []))
        if n_unparsed > 10:
            self.logger.warning(
                f"You have set force=True, which forces me to parse {n_unparsed} scores iteratively. "
                f"Next time, call _.parse() on me, so we can speed this up!"
            )

    def get_file_from_path(self, full_path: str) -> Optional[File]:
        for file in self.files:
            if file.full_path == full_path:
                return file

    def get_files(
        self,
        facets: FacetArguments = None,
        view_name: Optional[str] = None,
        parsed: bool = True,
        unparsed: bool = True,
        choose: Literal["all", "auto", "ask"] = "all",
        flat: bool = False,
        include_empty=False,
    ) -> Dict[str, Union[Dict[str, FileList], FileList]]:
        """"""
        assert (
            parsed + unparsed > 0
        ), "At least one of 'parsed' and 'unparsed' needs to be True."
        selected_facets = resolve_facets_param(facets, logger=self.logger)
        if choose == "ask":
            for facet in selected_facets:
                self.disambiguate_facet(facet, ask_for_input=True)
            choose = "auto"
        result = {}
        for piece, piece_obj in self.iter_pieces(view_name=view_name):
            facet2files = piece_obj.get_files(
                facets=facets,
                view_name=view_name,
                parsed=parsed,
                unparsed=unparsed,
                choose=choose,
                flat=flat,
                include_empty=include_empty,
            )
            if not include_empty:
                if not flat:
                    facet2files = {
                        facet: files
                        for facet, files in facet2files.items()
                        if len(files) > 0
                    }
                if len(facet2files) == 0:
                    continue
            result[piece_obj.name] = facet2files
            result[piece_obj.name] = facet2files
        return result

    def get_all_parsed(
        self,
        facets: FacetArguments = None,
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["all", "auto", "ask"] = "all",
        flat: bool = False,
        include_empty=False,
    ) -> Dict[str, Union[Dict[str, List[FileParsedTuple]], List[FileParsedTuple]]]:
        """"""
        selected_facets = resolve_facets_param(facets, logger=self.logger)
        if choose == "ask":
            for facet in selected_facets:
                self.disambiguate_facet(facet, ask_for_input=True)
            choose = "auto"
        if force:
            self.check_number_of_unparsed_scores(view_name, choose)
        result = {}
        for piece, piece_obj in self.iter_pieces(view_name=view_name):
            facet2parsed = piece_obj.get_all_parsed(
                facets=selected_facets,
                view_name=view_name,
                force=force,
                choose=choose,
                flat=flat,
                include_empty=include_empty,
            )
            if not include_empty:
                if not flat:
                    facet2parsed = {
                        facet: parsed
                        for facet, parsed in facet2parsed.items()
                        if len(parsed) > 0
                    }
                if len(facet2parsed) == 0:
                    continue
            result[piece_obj.name] = facet2parsed
        return result

    def get_all_pnames(
        self, pieces_in_metadata: bool = True, pieces_not_in_metadata: bool = True
    ) -> List[str]:
        """pieces (file names without extension and suffix) serve as IDs for pieces. Use
        this function to retrieve the comprehensive list, ignoring views.

        Args:
          pieces_in_metadata: pieces that are listed in the 'metadata.tsv' file for this corpus, if present
          pieces_not_in_metadata: pieces that are not listed in the 'metadata.tsv' file for this corpus

        Returns:
          The file names included in 'metadata.tsv' and/or those of all other scores.
        """
        result = []
        if pieces_in_metadata:
            result.extend(self.pieces_in_metadata(self.metadata_ix))
        if pieces_not_in_metadata:
            result.extend(self.pieces_not_in_metadata())
        return sorted(result)

    def get_pieces(self, view_name: Optional[str] = None) -> List[str]:
        """Retrieve pieces included in the current or selected view."""
        return [piece for piece, _ in self.iter_pieces(view_name)]

    def get_piece(self, pname) -> Piece:
        """Returns the :obj:`Piece` object for piece."""
        assert pname in self._pieces, f"'{pname}' is not a piece in this corpus."
        return self._pieces[pname]

    def get_present_facets(self, view_name: Optional[str] = None) -> List[str]:
        view = self.get_view(view_name)
        selected_pieces = []
        if view.pieces_in_metadata:
            selected_pieces.extend(self.pieces_in_metadata(self.metadata_ix))
        if view.pieces_not_in_metadata:
            selected_pieces.extend(self.pieces_not_in_metadata())
        result: Set[str] = set()
        for piece, piece_obj in self:
            detected_facets = piece_obj.count_detected(
                include_empty=False, prefix=False
            )
            result.update(detected_facets.keys())
        return list(result)

    def get_version_info(self, only_if_clean: bool = True) -> Dict[str, str]:
        if self.repo is None:
            self.logger.debug("No git repo, no version info.")
            return {}
        version_info = get_git_version_info(self.repo, only_if_clean=only_if_clean)
        if only_if_clean and not version_info:
            self.logger.info(
                "Git repo is dirty and only_if_clean=True. Returning empty version info."
            )
        return version_info

    def get_view(self, view_name: Optional[str] = None, **config) -> View:
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

    def info(
        self,
        view_name: Optional[str] = None,
        return_str: bool = False,
        show_discarded: bool = False,
    ):
        """"""
        header = f"Corpus '{self.name}'"
        header += "\n" + "-" * len(header) + "\n"
        header += f"Location: {self.corpus_path}\n"

        # get parsed scores before resetting the view's filtering counts to prevent counting twice
        parsed_scores = self.get_all_parsed("scores", view_name=view_name, flat=True)

        # start info message with the names of the available views, the header, and info on the active view.
        view = self.get_view(view_name)
        view.reset_filtering_data()
        msg = available_views2str(self._views, view_name)
        msg += header
        view_info = f"View: {view}"
        if view_name is None:
            piece_views = [
                piece.get_view().name
                for _, piece in self.iter_pieces(view_name=view_name)
            ]
            if len(set(piece_views)) > 1:
                view_info = "This is a mixed view. Call _.info(view_name) to see a homogeneous one."
        msg += view_info + "\n\n"

        # a table counting the number of parseable files under the active view
        counts_df = self.count_files(drop_zero=True, view_name=view_name)
        if counts_df.isna().any().any():
            counts_df = counts_df.fillna(0).astype("int")
        n_pieces = len(counts_df)
        if n_pieces == 0:
            if self.metadata_tsv is None:
                msg += "No 'metadata.tsv' file is present.\n\n"
            msg += "All pieces are excluded from the current view.\n\n"
        elif self.metadata_tsv is None:
            msg += (
                f"No 'metadata.tsv' file is present. Use _.create_metadata_tsv() to create one for these "
                f"{n_pieces} pieces.\n\n"
            )
            msg += counts_df.to_string()
        else:
            metadata_pieces = set(self.pieces_in_metadata(self.metadata_ix))
            included_selector = counts_df.index.isin(metadata_pieces)
            if included_selector.all():
                msg += f"All {n_pieces} pieces are listed in 'metadata.tsv':\n\n"
                msg += counts_df.to_string()
            elif not included_selector.any():
                msg = f"None of the {n_pieces} pieces is actually listed in 'metadata.tsv'.\n\n"
                msg += counts_df.to_string()
            else:
                msg += f"Only the following {included_selector.sum()} pieces are listed in 'metadata.tsv':\n\n"
                msg += counts_df[included_selector].to_string()
                not_included = ~included_selector
                plural = (
                    f"These {not_included.sum()} here are"
                    if not_included.sum() > 1
                    else "This one is"
                )
                msg += f"\n\n{plural} missing from 'metadata.tsv':\n\n"
                msg += counts_df[not_included].to_string()
        n_changed_scores = 0
        detached_key_counter = Counter()
        for piece, tuples in parsed_scores.items():
            for file, score in tuples:
                n_changed_scores += score.mscx.changed
                detached_key_counter.update(score._detached_annotations.keys())
        has_changed = n_changed_scores > 0
        has_detached = len(detached_key_counter) > 0
        if has_changed or has_detached:
            msg += "\n\n"
            if has_changed:
                plural = "s have" if n_changed_scores > 1 else " has"
                msg += f"{n_changed_scores} score{plural} changed since parsing."
            if has_detached:
                msg += pretty_dict(detached_key_counter, "key", "#scores")
        filtering_report = view.filtering_report(
            show_discarded=show_discarded, return_str=True
        )
        if filtering_report != "":
            msg += "\n" + filtering_report
        if self.n_orphans > 0:
            msg += f"\n\nThe corpus contains {self.n_orphans} orphans that could not be attributed to any of the pieces"
            if show_discarded:
                msg += f":\n{list(self.ix2orphan_file.values())}"
            else:
                msg += "."
        if return_str:
            return msg
        print(msg)

    def iter_facet(
        self,
        facet: ScoreFacet,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
    ) -> Iterator[FileDataframeTuple]:
        view_name_display = self.view_name if view_name is None else view_name
        for pname, piece_obj in self.iter_pieces(view_name):
            file, df = piece_obj.get_facet(
                facet=facet,
                view_name=view_name,
                choose=choose,
                unfold=unfold,
                interval_index=interval_index,
            )
            if df is None:
                self.logger.info(
                    f"The view '{view_name_display}' does not comprise a score or TSV file for '{pname}' from which to "
                    f"retrieve {facet}."
                )
            else:
                yield file, df

    def iter_facets(
        self,
        facets: ScoreFacets = None,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
        include_files: bool = False,
    ) -> Iterator:
        """Iterate through (piece, *DataFrame) tuples containing exactly one or zero DataFrames per requested facet.

        Args:
          facets:
          view_name:
          choose:
          unfold:
          interval_index:
          include_files:

        Returns:
          (piece, *DataFrame) tuples containing exactly one or zero DataFrames per requested facet per piece (piece).
        """
        selected_facets = resolve_facets_param(facets, ScoreFacet, logger=self.logger)
        if choose == "ask":
            for facet in selected_facets:
                self.disambiguate_facet(facet, ask_for_input=True)
            choose = "auto"
        for piece, piece_obj in self.iter_pieces(view_name=view_name):
            facet2parsed = piece_obj.get_facets(
                facets=selected_facets,
                view_name=view_name,
                force=True,
                choose=choose,
                unfold=unfold,
                interval_index=interval_index,
                flat=False,
            )
            if include_files:
                result = [
                    tup for facet in selected_facets for tup in facet2parsed[facet]
                ]
            else:
                result = [
                    df for facet in selected_facets for file, df in facet2parsed[facet]
                ]
            yield (piece, *result)

    def iter_parsed(
        self,
        facet: Facet = None,
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["all", "auto", "ask"] = "all",
        include_empty=False,
        unfold: bool = False,
        interval_index: bool = False,
    ) -> Iterator[FileParsedTuple]:
        """"""
        facet = check_argument_against_literal_type(facet, Facet, logger=self.logger)
        assert facet is not None, f"Pass a valid facet {Facet.__args__}"
        if choose == "ask":
            self.disambiguate_facet(facet, ask_for_input=True)
            choose = "auto"
        if force:
            self.check_number_of_unparsed_scores(view_name, choose)
        for piece, piece_obj in self.iter_pieces(view_name=view_name):
            for file, parsed in piece_obj.iter_parsed(
                facet=facet,
                view_name=view_name,
                force=force,
                choose=choose,
                include_empty=include_empty,
                unfold=unfold,
                interval_index=interval_index,
            ):
                yield file, parsed

    def iter_pieces(
        self, view_name: Optional[str] = None
    ) -> Iterator[Tuple[str, Piece]]:
        """Iterate through (name, corpus) tuples under the current or specified view."""
        view = self.get_view(view_name)
        view.reset_filtering_data(categories="pieces")
        param_sum = view.pieces_in_metadata + view.pieces_not_in_metadata
        if param_sum == 0:
            # all excluded, need to update filter counts accordingly
            key = "pieces"
            discarded_items, *_ = list(zip(*view.filter_by_token("pieces", self)))
            view._discarded_items[key].update(discarded_items)
            filtering_counts = view._last_filtering_counts[key]
            filtering_counts[[0, 1]] = filtering_counts[
                [1, 0]
            ]  # swapping counts for included & discarded in the array
            yield from []
        else:
            n_kept, n_discarded = 0, 0
            discarded_items = []
            differentiate_by_presence_in_metadata = param_sum == 1
            if differentiate_by_presence_in_metadata:
                selected_pieces = (
                    self.pieces_in_metadata(self.metadata_ix)
                    if view.pieces_in_metadata
                    else self.pieces_not_in_metadata()
                )
            filter_incomplete_facets = not view.pieces_with_incomplete_facets
            if filter_incomplete_facets:
                selected_facets = view.selected_facets
                if len(selected_facets) == len(View.available_facets):
                    # No facets have been excluded from the view, therefore the completeness criterion is based
                    # on which facets the corpus has rather than which ones have been selected
                    selected_facets = self.get_present_facets(view_name)
            for piece, piece_obj in view.filter_by_token("pieces", self):
                if len(piece_obj.count_detected()) == 0:
                    # no facets to show, probably due to other filters; do not include in 'pieces' filter counts
                    continue
                metadata_check = (
                    not differentiate_by_presence_in_metadata
                    or piece in selected_pieces
                )
                facet_check = (
                    not filter_incomplete_facets
                    or piece_obj.all_facets_present(
                        view_name=view_name, selected_facets=selected_facets
                    )
                )
                if metadata_check and facet_check:
                    if view_name not in piece_obj._views:
                        if view_name is None:
                            piece_obj.set_view(view)
                        else:
                            piece_obj.set_view(**{view_name: view})
                    yield piece, piece_obj
                else:
                    discarded_items.append(piece)
                    n_kept -= 1
                    n_discarded += 1
            key = "pieces"
            view._last_filtering_counts[key] += np.array(
                [n_kept, n_discarded, 0], dtype="int"
            )
            view._discarded_items[key].update(discarded_items)

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
    ) -> int:
        """Loads annotations from maximum one TSV file to maximum one score per piece. Each score will contain the
        annotations as a 'detached' annotation object accessible via the indicated ``key`` (defaults to 'detached').
        """
        facet = check_argument_against_literal_type(
            facet, AnnotationsFacet, logger=self.logger
        )
        assert facet is not None, f"Pass a valid facet {AnnotationsFacet.__args__}"
        assert (
            choose != "all"
        ), "Only one set of annotations can be added under a given key."
        if not git_revision:
            piece2tuples = self.get_all_parsed(
                facets=facet,
                view_name=view_name,
                force=force,
                choose=choose,
                flat=True,
            )
            piece2tuple = {piece: tuples[0] for piece, tuples in piece2tuples.items()}
            revision_str = ""
        else:
            piece2tuple = self.get_facet_at_git_revision(
                facet=facet,
                git_revision=git_revision,
                view_name=view_name,
                choose=choose,
            )
            revision_str = f" @ git revision {git_revision}"
        n_pieces = len(piece2tuple)
        if n_pieces == 0:
            self.logger.debug(f"No parsed '{facet}' TSV files found{revision_str}.")
        else:
            plural = "s" if n_pieces > 1 else ""
            self.logger.debug(
                f"Parsed '{facet}' file{revision_str} added to {n_pieces} piece{plural}."
            )
        for piece, (file, df) in piece2tuple.items():
            self[piece].load_annotation_table_into_score(
                df=df, view_name=view_name, choose=choose, key=key, infer=infer, **cols
            )
        return n_pieces

    def look_for_ignored_warnings(self, directory: Optional[str] = None):
        """
        Looks for a text file called IGNORED_WARNINGS and, if it exists, loads it, configuring loggers as indicated.
        """
        if directory is None:
            directory = self.corpus_path
        default_ignored_warnings_path = os.path.join(directory, "IGNORED_WARNINGS")
        if os.path.isfile(default_ignored_warnings_path):
            loggers, unsuccessful = self.load_ignored_warnings(
                default_ignored_warnings_path
            )
            n_lgrs, n_unscfl = len(loggers), len(unsuccessful)
            lgrs_plural = (
                f"{n_lgrs} logger has" if n_lgrs == 1 else f"{n_lgrs} loggers have"
            )
            if n_unscfl == 0:
                uscfl_plural = ""
            else:
                uscfl_plural = (
                    f", {n_unscfl} hasn't" if n_unscfl == 1 else f", {n_unscfl} haven't"
                )
            self.logger.info(
                f"{lgrs_plural} successfully been configured{uscfl_plural}."
            )

    def load_ignored_warnings(self, path: str) -> Tuple[List[Logger], List[str]]:
        """
        Loads in a text file containing warnings that are to be ignored, i.e., wrapped in DEBUG messages. The purpose
        is to mark certain warnings as OK, warranted by a human, to allow checks to pass regardless.
        """
        try:
            ignored_warnings = parse_ignored_warnings_file(path)
        except ValueError as e:
            self.logger.warning(e)
            return
        self.logger.debug(f"Ignored warnings contained in {path}:\n{ignored_warnings}")
        logger_names = set(self.logger_names.values())
        filtered_loggers = []
        unsuccessful = []
        for name, message_ids in ignored_warnings.items():
            normalized_name = normalize_logger_name(name)
            to_be_configured = [
                logger_name
                for logger_name in logger_names
                if normalized_name in logger_name
            ]
            if len(to_be_configured) == 0:
                self.logger.warning(
                    f"None of the logger names contains '{normalized_name}', which is the normalized name for loggers "
                    f"supposed to ignore "
                    f"warnings with message IDs {message_ids}"
                )
                unsuccessful.append(name)
                continue
            for name_to_configure in to_be_configured:
                filtered_loggers.append(name_to_configure)
                self._ignored_warnings[name_to_configure].extend(message_ids)
                configured = get_logger(
                    name_to_configure,
                    ignored_warnings=message_ids,
                    level=self.logger.getEffectiveLevel(),
                )
                configured.debug(
                    f"This logger has been configured to set warnings with the following IDs to DEBUG:\n{message_ids}."
                )
        return filtered_loggers, unsuccessful

    def load_metadata_file(self, file: File, allow_prefixed: bool = False) -> None:
        """Loads the TSV file at the given path and stores it as metadata. If the file is called 'metadata.tsv' it will
        be treated as the corpus' main file for determining pieces. Otherwise it is expected to be named
        'metadata{suffix}.tsv' and the suffix will be used as name for an additionally created view.
        """
        if not file.piece.startswith("metadata") and not allow_prefixed:
            self.logger.info(
                f"The file {file.rel_path} has a prefix and is disregarded as metadata file."
            )
            self.ix2metadata_file[file.ix] = file
            return
        if not file.piece.endswith("metadata"):
            match = re.search("metadata", file.piece)
            match_end = match.end()
            suffix = file.piece[match_end:]
            file.suffix = suffix
        else:
            suffix = ""
        self.ix2metadata_file[file.ix] = file
        try:
            metadata_df = load_tsv(file.full_path)
        except Exception as e:
            self.logger.warning(
                f"load_tsv({file.full_path!r}) failed with the exception '{e}'"
            )
            return
        if len(metadata_df) == 0:
            self.logger.warning(f"Parsed metadata file {file.rel_path} was empty.")
            return
        self._ix2parsed[file.ix] = metadata_df
        rel_dir = os.path.normpath(os.path.dirname(file.rel_path))
        if suffix == "" and rel_dir == ".":
            self.metadata_ix = file.ix
            self.metadata_tsv = metadata_df
            return
        # create views for other metadata files
        view_name = ""
        if rel_dir != ".":
            view_name = string2identifier(rel_dir)
        if suffix != "":
            if view_name != "":
                view_name += "_"
            view_name += string2identifier(suffix)
        pieces = self.pieces_in_metadata(file.ix)
        pieces = [re.escape(piece) for piece in pieces]
        new_view = DefaultView(view_name, only_metadata_pieces=False)
        new_view.include("pieces", *pieces)
        action = "Replaced" if view_name in self.view_names else "Added"
        self.set_view(**{view_name: new_view})
        self.logger.debug(
            f"{action} view '{view_name}' corresponding to {file.rel_path}."
        )

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

    def parse_mscx(self, *args, **kwargs):
        """Renamed to :meth:`parse_scores`."""
        self.parse_scores(*args, **kwargs)

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
        self.logger.debug(
            f"Parsing scores with parameters parallel={parallel}, only_new={only_new}"
        )
        piece2files = self.get_files(
            "scores", view_name=view_name, parsed=not only_new, choose=choose, flat=True
        )
        selected_files = sum(piece2files.values(), start=[])
        target = len(selected_files)
        if target == 0:
            self.logger.debug("Nothing to parse.")
            return
        selected_scores_df = pd.concat(
            [pd.DataFrame(files) for files in piece2files.values()],
            keys=piece2files.keys(),
        )
        self.logger.debug(
            f"Selected scores to parse:\n{selected_scores_df.set_index('ix')['rel_path'].to_string()}"
        )
        exts = selected_scores_df.fext.value_counts()

        if (
            any(ext[1:].lower() in Score.convertible_formats for ext in exts.index)
            and parallel
        ):
            needs_conversion = [
                ext
                for ext in exts.index
                if ext[1:].lower() in Score.convertible_formats
            ]
            msg = (
                f"The file extensions [{', '.join(needs_conversion)}] "
                f"require temporary conversion with your local MuseScore, which is not possible with parallel "
                f"processing. Parse with parallel=False or set View.include_convertible=False."
            )
            if self.ms is None:
                msg += (
                    "\nIn case you want to temporarily convert these files, you will also have to set the "
                    "property ms of this object to the path of your MuseScore 3 executable."
                )
            selected_files = [
                file for file in selected_files if file.fext not in needs_conversion
            ]
            msg += f"\nAfter filtering out the files, the target was reduced from N={target} to {len(selected_files)}."
            target = len(selected_files)
            self.logger.warning(msg)
            if target == 0:
                return

        configs = [
            dict(
                name=self.logger_names[ix],
            )
            for ix in selected_scores_df.ix
        ]
        # collect argument tuples for calling parse_musescore_file
        parse_this = [
            (
                file,
                self.logger,
                self.logger.parent,
                self.labels_cfg,
                conf,
                parallel,
                self.ms,
            )
            for file, conf in zip(selected_files, configs)
        ]
        try:
            if parallel:
                pool = mp.Pool(mp.cpu_count())
                res = pool.starmap(parse_musescore_file, parse_this)
                pool.close()
                pool.join()
                successful_results = {
                    file.ix: score
                    for file, score in zip(selected_files, res)
                    if score is not None
                }
                with_captured_logs = [
                    score
                    for score in successful_results.values()
                    if hasattr(score, "captured_logs")
                ]
                if len(with_captured_logs) > 0:
                    log_capture_handler = get_log_capture_handler(self.logger)
                    if log_capture_handler is not None:
                        for score in with_captured_logs:
                            log_capture_handler.log_queue.extend(score.captured_logs)
            else:
                parsing_results = [
                    parse_musescore_file(*params) for params in parse_this
                ]
                successful_results = {
                    file.ix: score
                    for file, score in zip(selected_files, parsing_results)
                    if score is not None
                }
            successful = len(successful_results)
            for ix, score in successful_results.items():
                self.get_piece(self.ix2pname[ix]).add_parsed_score(ix, score)
            if successful > 0:
                if successful == target:
                    quantifier = (
                        "The score has" if target == 1 else f"All {target} scores have"
                    )
                    self.logger.info(f"{quantifier} been parsed successfully.")
                else:
                    self.logger.info(
                        f"Only {successful} of the {target} scores have been parsed successfully."
                    )
            else:
                self.logger.info(
                    f"None of the {target} scores have been parsed successfully."
                )
        except KeyboardInterrupt:
            self.logger.info("Parsing interrupted by user.")
            raise
        finally:
            # self._collect_annotations_objects_references(ids=ids)
            pass

    def parse_tsv(
        self,
        view_name: Optional[str] = None,
        cols={},
        infer_types=None,
        level=None,
        only_new: bool = True,
        choose: Literal["all", "auto", "ask"] = "all",
        **kwargs,
    ):
        """Parse TSV files to be able to do something with them.

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
            Arguments for :py:meth:`pandas.DataFrame.to_csv`. Defaults to ``{'sep': '\t', 'index': False}``.
            In particular, you might want to update the default dictionaries for ``dtypes`` and ``converters`` used
            in :py:func:`load_tsv`. Passing kwargs prevents ms3 from parsing TSVs in parallel, so it will be a bit
            slower.

        Returns
        -------
        None
        """
        if level is not None:
            self.change_logger_cfg(level=level)
        piece2files = self.get_files(
            "tsv", view_name=view_name, parsed=not only_new, choose=choose, flat=True
        )
        selected_files = sum(piece2files.values(), start=[])
        target = len(selected_files)
        if target == 0:
            self.logger.debug("Nothing to parse.")
            return
        parse_this = [
            (file, get_logger(self.ix_logger(file.ix))) for file in selected_files
        ]
        if len(kwargs) == 0:
            pool = mp.Pool(mp.cpu_count())
            parsing_results = pool.starmap(parse_tsv_file, parse_this)
            pool.close()
            pool.join()
            successful_results = {
                file.ix: df
                for file, df in zip(selected_files, parsing_results)
                if df is not None
            }
        else:
            parsing_results = [load_tsv(*params, **kwargs) for params in parse_this]
            for file, logger in parse_this:
                logger.debug(f"Trying to load {file.rel_path}")
                try:
                    df = load_tsv(file.full_path, logger=logger, **kwargs)
                    parsing_results.append(df)
                except Exception as e:
                    parsing_results.append(None)
                    logger.info(
                        f"Couldn't be loaded, probably no tabular format or you need to specify 'sep', the delimiter "
                        f"as **kwargs.\n{file.rel_path}\nError: {e}"
                    )
            successful_results = {
                file.ix: score
                for file, score in zip(selected_files, parsing_results)
                if score is not None
            }
        successful = len(successful_results)
        for ix, df in successful_results.items():
            self.get_piece(self.ix2pname[ix]).add_parsed_tsv(ix, df)
        if successful > 0:
            if successful == target:
                quantifier = (
                    "The TSV file" if target == 1 else f"All {target} TSV files"
                )
                self.logger.info(f"{quantifier} have been parsed successfully.")
            else:
                self.logger.info(
                    f"Only {successful} of the {target} TSV files have been parsed successfully."
                )
        else:
            self.logger.info(
                f"None of the {target} TSV files have been parsed successfully."
            )

    def register_files_with_pieces(
        self,
        files: Optional[FileList] = None,
        pnames: Optional[Union[Collection[str], str]] = None,
    ) -> None:
        """Iterates through the ``files`` and tries to match it with the ``pieces`` and registered matched
        :obj:`File` objects with the corresponding :obj:`Piece` objects (unless already registered).

        By default, the method uses this object's :attr:`files` and :attr:`pieces`. To match with a Piece, the file name
        (without extension) needs to start with the Piece's ``piece``; otherwise, it will be stored under
        :attr:`ix2orphan_file`.

        Args:
          files: :obj:`File` objects to register with the corresponding :obj:`Piece` objects based on their file names.
          pnames: Names of the pieces that the files are to be matched to. Those that don't match any will be stored
          under :attr:`ix2orphan_file`.
        """
        if pnames is None:
            pnames = self.pnames
        elif isinstance(pnames, str):
            pnames = [pnames]
        pnames = sorted(pnames, key=len, reverse=True)
        # sort so that longest pieces come first for preventing errors when detecting suffixes
        if len(pnames) == 0:
            self.logger.info("Corpus contains neither scores nor metadata.")
            return
        if files is None:
            files = self.files
        for file in files:
            # try to associate all detected files with one of the created pieces and
            # store the mapping in :attr:`ix2pname`
            piece_name = None
            if file.piece in pnames:
                piece_name = file.piece
            else:
                for pname in pnames:
                    if file.piece.startswith(pname):
                        piece_name = pname
                        break
            if piece_name is None:
                if "metadata" in file.piece:
                    self.load_metadata_file(file)
                else:
                    self.logger.debug(
                        f"Could not associate {file.file} with any of the pieces. Stored as orphan."
                    )
                    self.ix2orphan_file[file.ix] = file
            else:
                piece = self.get_piece(piece_name)
                registration_result = piece.register_file(file)
                if registration_result is None:
                    self.logger.debug(
                        f"Skipping '{file.rel_path}' because it had already been registered with Piece('{piece_name}')."
                    )
                elif registration_result:
                    self.ix2pname[file.ix] = piece_name
                    self.logger_names[file.ix] = piece.logger.name
                else:
                    self.logger.warning(
                        f"Stored '{file.rel_path}' as orphan because it could not be registered with "
                        f"Piece('{piece_name}')"
                    )
                    self.ix2orphan_file[file.ix] = file

    def metadata(
        self,
        view_name: Optional[str] = None,
        choose: Optional[Literal["auto", "ask"]] = None,
    ) -> pd.DataFrame:
        """Returns metadata.tsv but only for pieces included in the current or indicated view. If no TSV file is
        present, get metadata from the current scores.
        """
        rows = [
            piece_obj.metadata() for piece, piece_obj in self.iter_pieces(view_name)
        ]
        metadata = pd.DataFrame(rows)
        if len(metadata) == 0:
            return metadata
        metadata = enforce_piece_index_for_metadata(metadata)
        return column_order(metadata, METADATA_COLUMN_ORDER, sort=False).sort_index()
        # tsv_metadata, score_metadata = None, None
        # if view.pieces_in_metadata:
        #     tsv_pieces = [piece for piece in self.pieces_in_metadata(self.metadata_ix)
        #                   if view.check_token('pieces', piece)]
        #     tsv_metadata = enforce_piece_index_for_metadata(self.metadata_tsv)
        #     tsv_metadata = tsv_metadata.loc[tsv_pieces]
        #     print(f"tsv_pieces: {tsv_pieces}")
        # if view.pieces_not_in_metadata:
        #     score_pieces = [piece for piece in self.pieces_not_in_metadata() if view.check_token('pieces', piece)]
        #     rows = [self.get_piece(piece).score_metadata(view_name=view_name, choose=choose)
        #             for piece in score_pieces]
        #     if len(rows) > 0:
        #         score_metadata = pd.DataFrame(rows).set_index('piece')
        #     print(f"score_pieces: {score_pieces}")
        # n_dataframes = (tsv_metadata is not None) + (score_metadata is not None)
        # if n_dataframes == 0:
        #     return pd.DataFrame()
        # if n_dataframes == 1:
        #     if tsv_metadata is None:
        #         return column_order(score_metadata, METADATA_COLUMN_ORDER, sort=False).sort_index()
        #     else:
        #         return column_order(tsv_metadata, METADATA_COLUMN_ORDER, sort=False).sort_index()
        # result = pd.concat([tsv_metadata, score_metadata])
        # return column_order(result, METADATA_COLUMN_ORDER, sort=False).sort_index()

    def score_metadata(
        self, view_name: Optional[str] = None, choose: Literal["auto", "ask"] = "auto"
    ):
        pieces, rows = [], []
        for piece, piece_obj in self.iter_pieces(view_name=view_name):
            try:
                row = piece_obj.score_metadata(view_name=view_name, choose=choose)
                if row is not None:
                    rows.append(row)
                    pieces.append(piece)
            except ValueError:
                print(piece)
                print(piece_obj.score_metadata())
                raise
        if len(rows) > 0:
            df = pd.DataFrame(rows).set_index("piece")
            return column_order(df, METADATA_COLUMN_ORDER, sort=False).sort_index()
        return pd.DataFrame()

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
            if view.name != view_name:
                view.name = view_name
            self._views[view_name] = view
        for piece, piece_obj in self:
            if active is not None:
                piece_obj.set_view(active)
            for view_name, view in views.items():
                piece_obj.set_view(**{view_name: view})

    def switch_view(
        self,
        view_name: str,
        show_info: bool = True,
        propagate: bool = True,
    ) -> None:
        if view_name is None:
            return
        new_view = self.get_view(view_name)
        old_view = self.get_view()
        if old_view.name is not None:
            self._views[old_view.name] = old_view
        self._views[None] = new_view
        new_name = new_view.name
        if new_name in self._views:
            del self._views[new_name]
        if propagate:
            for piece, piece_obj in self:
                active_view = piece_obj.get_view()
                if active_view.name != new_name or active_view != new_view:
                    piece_obj.set_view(new_view)
        if show_info:
            self.info()

    @property
    def view(self):
        return self.get_view()

    @view.setter
    def view(self, new_view: View):
        if not isinstance(new_view, View):
            return TypeError(
                "If you want to switch to an existing view by its name, use its name like an attribute or "
                "call _.switch_view()."
            )
        self.set_view(new_view)

    @property
    def views(self):
        print(
            pretty_dict(
                {"[active]" if k is None else k: v for k, v in self._views.items()},
                "view_name",
                "Description",
            )
        )

    @property
    def view_name(self):
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

    def update_labels(
        self,
        staff: Optional[int] = None,
        voice: Optional[Literal[1, 2, 3, 4]] = None,
        harmony_layer: Optional[Literal[0, 1, 2, 3]] = None,
        above: bool = False,
        safe: bool = True,
    ) -> FileList:
        altered_files = []
        for piece, piece_obj in self.iter_pieces():
            for file, score in piece_obj.iter_parsed("scores"):
                successfully_moved = score.move_labels_to_layer(
                    staff=staff,
                    voice=voice,
                    harmony_layer=harmony_layer,
                    above=above,
                    safe=safe,
                )
                if successfully_moved:
                    altered_files.append(file)
        return altered_files

    def update_scores(
        self,
        root_dir: Optional[str] = None,
        folder: Optional[str] = ".",
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
        assert (
            self.ms is not None
        ), "Set the attribute 'ms' to your MuseScore 3 executable to update scores."
        up2date_paths = []
        latest_version = LATEST_MUSESCORE_VERSION.split(".")
        for piece, piece_obj in self.iter_pieces():
            for file, score in piece_obj.iter_parsed("scores"):
                logger = self.ix_logger(file.ix)
                new_path = make_file_path(
                    file, root_dir=root_dir, folder=folder, suffix=suffix, fext=".mscx"
                )
                up2date_paths.append(new_path)
                updated_existed = os.path.isfile(new_path)
                if updated_existed:
                    if not overwrite:
                        logger.info(
                            f"Skipped updating {file.rel_path} because the target file exists already and "
                            f"overwrite=False: {new_path}"
                        )
                        continue
                    else:
                        if new_path == file.full_path:
                            updated_file = file
                        else:
                            updated_file = self.get_file_from_path(new_path)
                score_version = score.mscx.metadata["musescore"].split(".")
                if score_version < latest_version:
                    convert(file.full_path, new_path, self.ms, logger=logger)
                    if not updated_existed:
                        new_files = self.add_file_paths([new_path])
                        updated_file = new_files[0]
                    new_score = Score(new_path)
                    piece_obj.add_parsed_score(updated_file.ix, new_score)
                    compare_two_score_objects(score, new_score, logger=logger)
                else:
                    updated_file = file
                    logger.debug(
                        f"{file.rel_path} has version {LATEST_MUSESCORE_VERSION} already."
                    )
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
        for piece, piece_obj in self.iter_pieces(view_name=view_name):
            paths.extend(
                piece_obj.update_tsvs_on_disk(
                    facets=facets, view_name=view_name, force=force, choose=choose
                )
            )
        return paths

    #####################
    # OLD, needs adapting
    #####################
    def _add_annotations_by_ids(
        self,
        list_of_pairs,
        staff=None,
        voice=None,
        harmony_layer=1,
        check_for_clashes=False,
    ):
        """For each pair, adds the labels at tsv_id to the score at score_id.

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
            |   MuseScore will split and encode those beginning with a note name ( resulting in ms3-internal
                harmony_layer 3).
            | * 2 to have MuseScore interpret them as Nashville Numbers
        check_for_clashes : :obj:`bool`, optional
            Defaults to True, meaning that the positions where the labels will be inserted will be checked for existing
            labels.
        """
        self._add_detached_annotations_by_ids(list_of_pairs, new_key="labels_to_attach")
        for score_id, tsv_id in list_of_pairs:
            score = self[score_id]
            score.attach_labels(
                "labels_to_attach",
                staff=staff,
                voice=voice,
                harmony_layer=harmony_layer,
                check_for_clashes=check_for_clashes,
                remove_detached=True,
            )

    def insert_detached_labels(
        self,
        view_name: Optional[str] = None,
        key: str = "detached",
        staff: int = None,
        voice: Literal[1, 2, 3, 4] = None,
        harmony_layer: Optional[Literal[0, 1, 2]] = None,
        check_for_clashes: bool = True,
    ) -> Tuple[int, int]:
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
        reached, goal, i = 0, 0, 0
        for i, (file, score) in enumerate(
            self.iter_parsed("scores", view_name=view_name), 1
        ):
            r, g = score.attach_labels(
                key,
                staff=staff,
                voice=voice,
                harmony_layer=harmony_layer,
                check_for_clashes=check_for_clashes,
            )
            self.logger.debug(f"{r}/{g} labels successfully added to {file}")
            reached += r
            goal += g
        self.logger.info(f"{reached}/{goal} labels successfully added to {i} files.")
        return reached, goal

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
        """Update :obj:`Corpus.labels_cfg` and retrieve new 'labels' tables accordingly.

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
        for piece, piece_obj in self:
            piece_obj.change_labels_cfg(labels_cfg=self.labels_cfg)

    def check_labels(self, keys=None, ids=None):
        if len(self._parsed_mscx) == 0:
            self.logger.info("No scores have been parsed so far. Use parse_scores()")
            return
        if ids is None:
            ids = list(self._iterids(keys, only_parsed_mscx=True))
        checks = {id: self._parsed_mscx[id].check_labels() for id in ids}
        checks = {k: v for k, v in checks.items() if v is not None and len(v) > 0}
        if len(checks) > 0:
            idx = self.ids2idx(checks.keys(), pandas_index=True)
            return pd.concat(checks.values(), keys=idx, names=idx.names)
        return pd.DataFrame()

    def color_non_chord_tones(
        self,
        color_name: str = "red",
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["all", "auto", "ask"] = "all",
    ) -> Dict[str, List[FileDataframeTuple]]:
        if self.n_parsed_scores == 0:
            self.logger.info("No scores have been parsed so far. Use parse_scores()")
            return dict()
        result = defaultdict(list)
        for file, score in self.iter_parsed(
            "scores",
            view_name=view_name,
            force=force,
            choose=choose,
        ):
            report = score.color_non_chord_tones(color_name=color_name)
            if report is not None:
                result[file.piece].append((file, report))
        return dict(result)

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
        for piece, piece_obj in self.iter_pieces(view_name=view_name):
            c, u = piece_obj.compare_labels(
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

    def count_annotation_layers(self, keys=None, which="attached", per_key=False):
        """Counts the labels for each annotation layer defined as (staff, voice, harmony_layer).
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
            If set to True, the results are returned as a dict {key: Counter}, otherwise the counts are summed up in one
            Counter. If ``which='detached'``, the keys are keys from Score objects, otherwise they are keys from this
            Corpus object.

        Returns
        -------
        :obj:`dict` or :obj:`collections.Counter`
            By default, the function returns a Counter of labels for every annotation layer (staff, voice,
            harmony_layer) If ``per_key`` is set to True, a dictionary {key: Counter} is returned, separating the
            counts.
        """
        res_dict = defaultdict(Counter)

        if which == "detached":
            for id in self._iterids(keys, only_detached_annotations=True):
                for key, annotations in self._parsed_mscx[
                    id
                ]._detached_annotations.items():
                    if key != "annotations":
                        _, layers = annotations.annotation_layers
                        layers_dict = {
                            tuple(None if pd.isnull(e) else e for e in t): count
                            for t, count in layers.to_dict().items()
                        }
                        res_dict[key].update(layers_dict)
        elif which in ["attached", "tsv"]:
            for key, i in self._iterids(keys):
                if (key, i) in self._annotations:
                    ext = self.fexts[key][i]
                    if (which == "attached" and ext == ".mscx") or (
                        which == "tsv" and ext != ".mscx"
                    ):
                        _, layers = self._annotations[(key, i)].annotation_layers
                        layers_dict = {
                            tuple(None if pd.isnull(e) else e for e in t): count
                            for t, count in layers.to_dict().items()
                        }
                        res_dict[key].update(layers_dict)
        else:
            self.logger.error(
                f"Parameter 'which' needs to be one of {{'attached', 'detached', 'tsv'}}, not {which}."
            )
            return {} if per_key else pd.Series()

        def make_series(counts):
            if len(counts) == 0:
                return pd.Series()
            data = counts.values()
            ks = list(counts.keys())
            # levels = len(ks[0])
            names = ["staff", "voice", "harmony_layer", "color"]  # <[:levels]
            ix = pd.MultiIndex.from_tuples(ks, names=names)
            return pd.Series(data, ix)

        if per_key:
            res = {k: make_series(v) for k, v in res_dict.items()}
        else:
            res = make_series(sum(res_dict.values(), Counter()))
        if len(res) == 0:
            self.logger.info(
                "No annotations found. Maybe no scores have been parsed using parse_scores()?"
            )
        return res

    def count_extensions(
        self, view_name: Optional[str] = None, include_metadata: bool = False
    ):
        """"""
        selected_files = self.get_files(view_name=view_name, flat=True)
        result = {
            piece: Counter(file.fext for file in files)
            for piece, files in selected_files.items()
        }
        if include_metadata and "metadata" not in result:
            result["metadata"] = Counter(
                file.fext for file in self.ix2metadata_file.values()
            )
        return result

    def count_pieces(self, view_name: Optional[str] = None) -> int:
        """Number of selected pieces under the given view."""
        return sum(1 for _ in self.iter_pieces(view_name=view_name))

    def _get_parsed_score_files(
        self, view_name: Optional[str] = None, flat=True
    ) -> Union[FileList, FileDict]:
        file_dict = self.get_files(
            "scores", view_name=view_name, unparsed=False, flat=True
        )
        all_files = sum(file_dict.values(), [])
        if flat:
            return all_files
        else:
            return {"scores": all_files}

    def _get_unparsed_score_files(
        self, view_name: Optional[str] = None, flat=True
    ) -> Union[FileList, FileDict]:
        file_dict = self.get_files(
            "scores", view_name=view_name, parsed=False, flat=True
        )
        all_files = sum(file_dict.values(), [])
        if flat:
            return all_files
        else:
            return {"scores": all_files}

    def _get_parsed_tsv_files(
        self, view_name: Optional[str] = None, flat=True
    ) -> Union[FileList, FileDict]:
        if flat:
            file_dict = self.get_files(
                "tsv", view_name=view_name, unparsed=False, flat=True
            )
            return sum(file_dict.values(), [])
        else:
            result = defaultdict(list)
            for piece, piece_obj in self.iter_pieces(view_name=view_name):
                file_dict = piece_obj._get_parsed_tsv_files(
                    view_name=view_name, flat=False
                )
                for facet, files in file_dict.items():
                    result[facet].extend(files)
            return dict(result)

    def _get_unparsed_tsv_files(
        self, view_name: Optional[str] = None, flat: bool = True
    ) -> Union[FileList, FileDict]:
        if flat:
            file_dict = self.get_files(
                "tsv", view_name=view_name, parsed=False, flat=True
            )
            return sum(file_dict.values(), [])
        else:
            result = defaultdict(list)
            for piece, piece_obj in self.iter_pieces(view_name=view_name):
                file_dict = piece_obj._get_unparsed_tsv_files(
                    view_name=view_name, flat=False
                )
                for facet, files in file_dict.items():
                    result[facet].extend(files)
            return dict(result)

    def count_labels(self, keys=None, per_key=False):
        """Count label types.

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
                self.logger.error(
                    "No scores have been parsed so far. Use parse_scores()."
                )
            else:
                self.logger.info("None of the scores contain annotations.")
        if per_key:
            return {k: dict(v) for k, v in res_dict.items()}
        return dict(sum(res_dict.values(), Counter()))

    def count_tsv_types(self, keys=None, per_key=False):
        """Count inferred TSV types.

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
        """Calls :py:meth:`Score.detach_labels <ms3.score.Score.detach_labels` on every parsed score under the
        current or selected view.
        """

        for file, score in self.iter_parsed(
            "scores", view_name=view_name, force=force, choose=choose
        ):
            try:
                score.detach_labels(
                    key=key,
                    staff=staff,
                    voice=voice,
                    harmony_layer=harmony_layer,
                    delete=delete,
                )
            except Exception as e:
                score.logger.error(
                    f"Detaching labels failed with the following error:\n'{e}'"
                )

    # def get_labels(self, keys=None, staff=None, voice=None, harmony_layer=None, positioning=True, decode=False,
    # column_name=None,
    #                color_format=None, concat=True):
    #     """ This function does not take into account self.labels_cfg """
    #     if len(self._annotations) == 0:
    #         self.logger.error("No labels available so far. Add files using add_dir() and parse them using parse().")
    #         return pd.DataFrame()
    #     keys = self._treat_key_param(keys)
    #     harmony_layer = self._treat_harmony_layer_param(harmony_layer)
    #     self._extract_and_cache_dataframes(labels=True, only_new=True)
    #     l = locals()
    #     params = {p: l[p] for p in self.labels_cfg.keys()}
    #     ids = [id for id in self._iterids(keys) if id in self._annotations]
    #     if len(ids) == 0:
    #         self.logger.info(f"No labels match the criteria.")
    #         return pd.DataFrame()
    #     annotation_tables = [self._annotations[id].get_labels(**params, inverse=False) for id in ids]
    #     idx, names = self.ids2idx(ids)
    #     if names is None:
    #         names = (None,) * len(idx[0])
    #     names += tuple(annotation_tables[0].index.names)
    #     if concat:
    #         return pd.concat(annotation_tables, keys=idx, names=names)
    #     return annotation_tables

    def get_facet_at_git_revision(
        self,
        facet: ScoreFacet,
        git_revision: str,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        concatenate: bool = False,
    ) -> Union[Dict[str, FileDataframeTuple], pd.DataFrame]:
        facet = check_argument_against_literal_type(
            facet, ScoreFacet, logger=self.logger
        )
        assert facet is not None, f"Pass a valid facet {ScoreFacet.__args__}"
        if choose == "ask":
            self.disambiguate_facet(facet, ask_for_input=True)
            choose = "auto"
        piece2files = self.get_files(facets=facet, view_name=view_name, flat=True)
        piece2selected = {}
        for piece, files in piece2files.items():
            parsed_files = [file for file in files if file.ix in self.ix2parsed]
            unparsed_files = [file for file in files if file.ix not in parsed_files]
            n_parsed = len(parsed_files)
            n_unparsed = len(unparsed_files)
            if n_parsed == 1:
                piece2selected[piece] = parsed_files[0]
            else:
                if n_parsed == 0:
                    if n_unparsed > 1:
                        selected = disambiguate_files(
                            unparsed_files,
                            self.name,
                            facet,
                            choose=choose,
                            logger=self.logger,
                        )
                    else:
                        selected = unparsed_files[0]
                else:  # n_parsed > 1
                    selected = disambiguate_files(
                        parsed_files,
                        self.name,
                        facet,
                        choose=choose,
                        logger=self.logger,
                    )
                if selected is None:
                    continue
                piece2selected[piece] = selected
        piece2tuples = {}
        for piece, file in piece2selected.items():
            new_file, parsed = parse_tsv_file_at_git_revision(
                file=file,
                git_revision=git_revision,
                repo_path=self.corpus_path,
                logger=self.ix_logger(file.ix),
            )
            if parsed is None:
                self.logger.warning(
                    f"Could not retrieve {file.rel_path} @ {git_revision!r}."
                )
            else:
                if git_revision == "LATEST_VERSION":
                    git_info = get_name_of_highest_version_tag(self.repo)
                else:
                    git_info = git_revision
                self.logger.info(
                    f"Successfully retrieved {file.rel_path} @ {git_info!r} "
                    f"({new_file.commit_sha})."
                )
                piece2tuples[piece] = (new_file, parsed)
        if concatenate:
            if len(piece2tuples) > 0:
                dfs = [df for file, df in piece2tuples.values()]
                df = pd.concat(dfs, keys=piece2tuples.keys())
                df.index.rename(["piece", "i"], inplace=True)
                return df
            else:
                return pd.DataFrame()
        return piece2tuples

    def get_parsed_at_index(self, ix: int) -> Optional[ParsedFile]:
        parsed = self.ix2parsed
        if ix in parsed:
            return parsed[ix]
        if ix in self.ix2pname:
            piece = self.get_piece(self.ix2pname[ix])
            try:
                return piece._get_parsed_at_index(ix)
            except RuntimeError:
                return None
        if ix in self.ix2orphan_file:
            file = self.ix2orphan_file[ix]
            try:
                df = load_tsv(file.full_path)
            except Exception:
                df = pd.read_csv(file.full_path, sep="\t")
            self._ix2parsed[file.ix] = df
            return df
        self.logger.warning(f"This Corpus does not include a file with index {ix}.")

    def join(self, keys=None, ids=None, what=None, use_index=True):
        if what is not None:
            what = [w for w in what if w != "scores"]
        matches = self.match_files(keys=keys, ids=ids, what=what)
        join_this = set(map(tuple, matches.values))
        key_ids, *_ = zip(*join_this)
        if use_index:
            key_ids = self.ids2idx(key_ids, pandas_index=True)
        join_this = [
            [self._parsed_tsv[id] for id in ids if not pd.isnull(id)]
            for ids in join_this
        ]
        joined = [join_tsvs(dfs) for dfs in join_this]
        return pd.concat(joined, keys=key_ids)

    def keys(self) -> List[str]:
        """Return the names of all Piece objects."""
        return list(self._pieces.keys())

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
        frictionless: bool = True,
    ) -> List[str]:
        """Store facets extracted from parsed scores as TSV files.

        Args:
          view_name:
          root_dir:
          measures_folder, notes_folder, rests_folder, notes_and_rests_folder, labels_folder, expanded_folder,
          form_labels_folder, cadences_folder, events_folder, chords_folder:
              Specify directory where to store the corresponding TSV files.
          metadata_suffix:
              Specify a suffix to update the 'metadata{suffix}.tsv' file for this corpus. For the main file, pass ''
          markdown:
              By default, when ``metadata_path`` is specified, a markdown file called ``README.md`` containing
              the columns [file_name, measures, labels, standard, annotators, reviewers] is created. If it exists
              already, this table will be appended or overwritten after the heading ``# Overview``.
          simulate:
          unfold:
              By default, repetitions are not unfolded. Pass True to duplicate values so that they correspond to a full
              playthrough, including correct positioning of first and second endings.
          interval_index:
          frictionless:
            If True (default), the file is written together with a frictionless resource descriptor JSON file
            whose column schema is used to validate the stored TSV file.


        Returns:
            A list of file stored to disk. If ``frictionless=True`` (default), it will be the list of descriptor file
            paths describing the stored TSV files (i.e., the list contains one file for every two files written to
            disk). Otherwise, it will be the list of TSV file paths.
        """
        lcls = locals()
        df_types = [t for t in Score.dataframe_types if t != "metadata"]
        folder_vars = [t + "_folder" for t in df_types]
        folder_params = {
            t: lcls[p] for t, p in zip(df_types, folder_vars) if lcls[p] is not None
        }
        output_metadata = metadata_suffix is not None
        if len(folder_params) == 0 and not output_metadata:
            self.logger.warning("Pass at least one parameter to store files.")
            return []
        facets = list(folder_params.keys())
        df_params = {p: True for p in folder_params.keys()}
        n_scores = len(self._get_parsed_score_files(view_name=view_name, flat=True))
        paths = []
        target = len(facets) * n_scores

        # (deprecated for now) if the view is default (no additional filters have been set), write one CSVW metadata
        # file per facet
        # view = self.get_view(view_name)
        self.logger.info(
            f"Extracting {len(facets)} facets from {n_scores} of the {self.n_parsed_scores} parsed scores."
        )
        if frictionless:
            version_info = self.get_version_info()
        else:
            version_info = None
        if target > 0:
            for piece, piece_obj in self.iter_pieces(view_name=view_name):
                for file, facet2dataframe in piece_obj.iter_extracted_facets(
                    facets,
                    view_name=view_name,
                    unfold=unfold,
                    interval_index=interval_index,
                ):
                    for facet, df in facet2dataframe.items():
                        if df is None:
                            continue
                        folder = folder_params[facet]
                        directory = compute_path_from_file(
                            file, root_dir=root_dir, folder=folder
                        )
                        piece_name = file.piece
                        if unfold:
                            piece_name += "_unfolded"
                        facet_param = "harmonies" if facet == "expanded" else facet
                        if simulate:
                            file_path = os.path.join(
                                directory, f"{piece}.{facet_param}.tsv"
                            )
                            self.logger.info(
                                f"Would have stored the {facet} from {file.rel_path} as {file_path}."
                            )
                        else:
                            create_descriptor = frictionless and facet != "events"
                            descriptor_or_resource_path = store_dataframe_resource(
                                df=df,
                                directory=directory,
                                piece_name=piece_name,
                                facet=facet_param,
                                zipped=False,
                                frictionless=create_descriptor,
                                raise_exception=False,
                                write_or_remove_errors_file=True,
                                logger=self.logger,
                                custom_metadata=version_info,
                            )
                        paths.append(descriptor_or_resource_path)
        if output_metadata:
            if not markdown:
                metadata_paths = self.update_metadata_tsv_from_parsed_scores(
                    root_dir=root_dir, suffix=metadata_suffix, markdown_file=None
                )
            else:
                metadata_paths = self.update_metadata_tsv_from_parsed_scores(
                    root_dir=root_dir, suffix=metadata_suffix
                )
            paths.extend(metadata_paths)
        return paths

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
        md = self.score_metadata(view_name=view_name)
        if len(md.index) == 0:
            self.logger.debug("\n\nNo metadata to write.")
            return []
        if root_dir is None:
            root_dir = self.corpus_path
        else:
            root_dir = resolve_dir(root_dir)
            assert not os.path.isfile(
                root_dir
            ), "Pass a path to a folder, not to a file."
            if not os.path.isdir(root_dir):
                os.makedirs(root_dir)
        metadata_path = os.path.join(root_dir, "metadata" + suffix + ".tsv")
        if not write_metadata(md, metadata_path, logger=self.logger):
            return []
        # if the metadata<suffix>.tsv has been created or updated, register/reload it with the Corpus object
        new_files = self.add_file_paths([metadata_path])
        if len(new_files) == 0:
            file = self.get_file_from_path(metadata_path)
        else:
            file = new_files[0]
        self.load_metadata_file(file)
        updated_metadata = self[file.ix]
        if markdown_file is not None:
            if os.path.isabs(markdown_file):
                markdown_path = markdown_file
            else:
                markdown_path = os.path.join(root_dir, markdown_file)
            write_markdown(updated_metadata, markdown_path, logger=self.logger)
            return [metadata_path, markdown_path]
        return [metadata_path]

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
        for piece, piece_obj in self.iter_pieces(view_name):
            modified = piece_obj.update_score_metadata_from_tsv(
                view_name=view_name,
                force=force,
                choose=choose,
                write_empty_values=write_empty_values,
                remove_unused_fields=remove_unused_fields,
                write_text_fields=write_text_fields,
                update_instrumentation=update_instrumentation,
            )
            updated_scores.extend(modified)
        if len(updated_scores) > 0:
            self.update_metadata_tsv_from_parsed_scores(markdown_file=None)
        return updated_scores

    #
    #
    # def _collect_annotations_objects_references(self, keys=None, ids=None):
    #     """ Updates the dictionary self._annotations with all parsed Scores that have labels attached (or not any
    #     more). """
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

    def store_parsed_scores(
        self,
        view_name: Optional[str] = None,
        only_changed: bool = True,
        root_dir: Optional[str] = None,
        folder: str = ".",
        suffix: str = "",
        overwrite: bool = False,
        simulate=False,
    ) -> List[str]:
        """Stores all parsed scores under this view as MuseScore 3 files.

        Args:
          view_name:
          only_changed:
              By default, only scores that have been modified since parsing are written. Set to False to store
              all scores regardless.
          root_dir:
          folder:
          suffix: Suffix to append to the original file name.
          overwrite: Pass True to overwrite existing files.
          simulate: Set to True if no files are to be written.

        Returns:
          Paths of the stored files.
        """
        file_paths = []
        for piece, piece_obj in self.iter_pieces(view_name):
            if only_changed:
                score_iterator = piece_obj.get_changed_scores(view_name)
            else:
                score_iterator = piece_obj.get_parsed_scores(view_name)
            for file, score in score_iterator:
                path = piece_obj.store_parsed_score_at_ix(
                    ix=file.ix,
                    root_dir=root_dir,
                    folder=folder,
                    suffix=suffix,
                    overwrite=overwrite,
                    simulate=simulate,
                )
                if path is None:
                    continue
                file_paths.append(path)
                if path == file.full_path:
                    continue
                if self.corpus_path not in path:
                    continue
                new_files = self.add_file_paths([path])
                if len(new_files) == 0:
                    file_to_register = self.get_file_from_path(path)
                else:
                    file_to_register = new_files[0]
                piece_obj.add_parsed_score(file_to_register.ix, score)
        return file_paths

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
    #             f"No labels found with {'these' if plural else 'this'} label{plural_s} harmony_layer{plural_s}:
    #             {', '.join(not_found)}")
    #     return [all_types[t] for user_input in lt for t in get_matches(user_input)]

    # def update_metadata(self, allow_suffix=False):
    #     """Uses all parsed metadata TSVs to update the information in the corresponding parsed MSCX files and returns
    #     the IDs of those that have been changed.
    #
    #     Parameters
    #     ----------
    #     allow_suffix : :obj:`bool`, optional
    #         If set to True, this would also update the metadata for currently parsed MuseScore files
    #         corresponding to the columns 'rel_paths' and 'pieces' + [ANY SUFFIX]. For example,
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
    #     if old.index.names != ['rel_paths', 'pieces']:
    #         try:
    #             old = old.set_index(['rel_paths', 'pieces'])
    #         except KeyError:
    #             self.logger.warning(f"Corpusd metadata do not contain the columns 'rel_paths' and 'pieces' "
    #                                 f"needed to match information on identical files.")
    #             return []
    #     new = self.metadata_from_parsed(from_tsv=False).set_index(['rel_paths', 'pieces'])
    #     excluded_cols = ['ambitus', 'annotated_key', 'KeySig', 'label_count', 'last_mc', 'last_mn', 'musescore',
    #                      'TimeSig', 'length_qb', 'length_qb_unfolded', 'all_notes_qb', 'n_onsets',
    #                      'n_onset_positions']
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
    #         for (rel_path, piece), new_dict in updates.items():
    #             matches = self.piece2ids(piece=piece, rel_path=rel_path, allow_suffix=allow_suffix)
    #             match_ids = [id for id in matches.keys() if id in self._parsed_mscx]
    #             n_files_to_update = len(match_ids)
    #             if n_files_to_update == 0:
    #                 self.logger.debug(
    #                     f"rel_path={rel_path}, piece={piece} does not correspond to a currently parsed MuseScore
    #                     file.")
    #                 continue
    #             for id in match_ids:
    #                 for name, val in new_dict.items():
    #                     self._parsed_mscx[id].mscx.parsed.metatags[name] = val
    #                 self._parsed_mscx[id].mscx.parsed.update_metadata()
    #                 self.ix_logger(id).debug(f"Updated with {new_dict}")
    #                 ids.append(id)
    #
    #         self.logger.info(f"{l} files updated.")
    #     else:
    #         self.logger.info("Nothing to update.")
    #     return ids

    def __getstate__(self):
        """Override the method of superclass"""
        return self.__dict__

    def __getitem__(self, piece_or_ix: Union[str, int]) -> Union[Piece, ParsedFile]:
        if isinstance(piece_or_ix, str):
            return self.get_piece(piece_or_ix)
        if isinstance(piece_or_ix, int):
            return self.get_parsed_at_index(piece_or_ix)
        if isinstance(piece_or_ix, tuple):
            if len(piece_or_ix) == 1:
                return self.get_piece(piece_or_ix[0])
            piece, ix = piece_or_ix
            return self.get_piece(piece)[ix]
        raise TypeError(
            f"Index needs to be piece (str) or ix (int), not {piece_or_ix} ({type(piece_or_ix)})"
        )

    def __iter__(self) -> Iterator[Tuple[str, Piece]]:
        """Iterate through all (piece, Piece) tuples, regardless of any Views.

        Yields: (piece, Piece) tuples
        """
        yield from self._pieces.items()

    def __repr__(self):
        return self.info(return_str=True)


# ######################################################################################################################
# ######################################################################################################################
# ############################################### End of Corpus() ######################################################
# ######################################################################################################################
# ######################################################################################################################


def parse_musescore_file(
    file: File,
    logger: Logger,
    logger_parent: Logger,
    labels_cfg: dict = {},
    logger_cfg: dict = {},
    read_only: bool = False,
    ms: Optional[str] = None,
) -> Score:
    """Performs a single parse and returns the resulting Score object or None.

    Args:
      file: File object with path information of a score that can be opened (or converted) with MuseScore 3.
      logger: Logger to be used within this function (not for the parsing itself).
      logger_cfg: Logger config for the new Score object (and therefore for the parsing itself).
      read_only:
          Pass True to return smaller objects that do not keep a copy of the original XML structure in memory.
          In order to make changes to the score after parsing, this needs to be False (default).
      ms: MuseScore executable in case the file needs to be converted.

    Returns:
      The parsed score.
    """
    path = file.full_path
    file = file.file
    logger.parent = logger_parent
    logger.debug(f"Attempting to parse {file}")
    try:
        score = Score(
            path, read_only=read_only, labels_cfg=labels_cfg, ms=ms, **logger_cfg
        )
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


def parse_tsv_file(file, logger) -> pd.DataFrame:
    path = file.full_path
    logger.debug(f"Trying to load {file.rel_path}")
    try:
        df = load_tsv(path)
        return df
    except Exception as e:
        logger.info(
            f"Couldn't be loaded, probably no tabular format or you need to specify 'sep', the delimiter as "
            f"**kwargs.\n{path}\nError: {e}"
        )
