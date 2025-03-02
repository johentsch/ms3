import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union, overload

import pandas as pd

from ._typing import (
    AnnotationsFacet,
    DataframeDict,
    Facet,
    FacetArguments,
    Facets,
    FileDataframeTuple,
    FileDataframeTupleMaybe,
    FileDict,
    FileList,
    FileParsedTuple,
    FileParsedTupleMaybe,
    FileScoreTuple,
    FileScoreTupleMaybe,
    ParsedFile,
    ScoreFacet,
    ScoreFacets,
    TSVtype,
    TSVtypes,
)
from .annotations import Annotations
from .logger import LoggedClass
from .score import Score
from .transformations import dfs2quarterbeats
from .utils import (
    File,
    argument_and_literal_type2list,
    ask_user_to_choose_from_disambiguated_files,
    assert_dfs_equal,
    automatically_choose_from_disambiguated_files,
    available_views2str,
    check_argument_against_literal_type,
    compute_path_from_file,
    disambiguate_files,
    files2disambiguation_dict,
    get_git_version_info,
    get_musescore,
    infer_tsv_type,
    literal_type2tuple,
    load_tsv,
    make_file_path,
    metadata2series,
    parse_tsv_file_at_git_revision,
    pretty_dict,
    replace_index_by_intervals,
    resolve_facets_param,
    store_dataframe_resource,
    update_labels_cfg,
    write_tsv,
)
from .utils.constants import (
    AUTOMATIC_COLUMNS,
    LEGACY_COLUMNS,
    MUSESCORE_HEADER_FIELDS,
    MUSESCORE_METADATA_FIELDS,
)
from .view import DefaultView, View


class Piece(LoggedClass):
    """Wrapper around :class:`~.score.Score` for associating it with parsed TSV files"""

    _deprecated_elements = ["get_dataframe"]

    def __init__(
        self,
        pname: str,
        view: Optional[View] = None,
        labels_cfg: Optional[dict] = None,
        ms=None,
        **logger_cfg,
    ):
        """

        Args:
            pname: Piece name, that is the file name without any suffixes or extensions.
            view: :obj:`View` object to be used as default.
            labels_cfg:
                Configuration dictionary to determine the output format of :py:attr:`~.score.Score.labels`.
            ms: MuseScore executable if convertible files (not MSCX or MSCZ) are to be parsed.
            **logger_cfg
        """
        super().__init__(subclass="Piece", logger_cfg=logger_cfg)
        self.name = pname
        available_types = ("scores",) + Score.dataframe_types
        self.facet2files: Dict[str, FileList] = defaultdict(list)
        """{typ -> [:obj:`File`]} dict storing file information for associated types.
        """
        self.facet2files.update({typ: [] for typ in available_types})
        self.ix2file: Dict[int, File] = defaultdict(list)
        """{ix -> :obj:`File`} dict storing the registered file information for access via index.
        """
        self.facet2parsed: Dict[str, Dict[int, ParsedFile]] = defaultdict(dict)
        """{typ -> {ix -> :obj:`pandas.DataFrame`|:obj:`Score`}} dict storing parsed files for associated types.
        """
        self.facet2parsed.update({typ: {} for typ in available_types})
        self.ix2parsed: Dict[int, ParsedFile] = {}
        """{ix -> :obj:`pandas.DataFrame`|:obj:`Score`} dict storing the parsed files for access via index.
        """
        self.ix2parsed_score: Dict[int, Score] = {}
        """Subset of :attr:`ix2parsed`"""
        self.ix2parsed_tsv: Dict[int, pd.DataFrame] = {}
        """Subset of :attr:`ix2parsed`"""
        self.ix2annotations: Dict[int, Annotations] = {}
        """{ix -> :obj:`Annotations`} dict storing Annotations objects for the parsed labels and expanded labels.
        """
        self._views: dict = {}
        if view is None:
            self._views[None] = DefaultView(level=self.logger.getEffectiveLevel())
        else:
            self._views[None] = view
            if view.name != "default":
                self._views["default"] = DefaultView(
                    level=self.logger.getEffectiveLevel()
                )
        self._views["all"] = View(level=self.logger.getEffectiveLevel())
        self._ms = get_musescore(ms, logger=self.logger)
        """:obj:`str`
        Path or command of the local MuseScore 3 installation if specified by the user."""
        self._tsv_metadata: Optional[Dict[str, str]] = None
        """If the :class:`~.corpus.Corpus` has :attr:`~.corpus.Corpus.metadata_tsv`, this field will contain the
        {column: value} pairs of the row pertaining to this piece.
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
        if labels_cfg is not None:
            self.labels_cfg.update(update_labels_cfg(labels_cfg, logger=self.logger))

    def all_facets_present(
        self, view_name: Optional[str] = None, selected_facets: Optional[Facets] = None
    ) -> bool:
        """Checks if parsed TSV files have been detected for all selected facets under the active or indicated view.

        Args:
          view_name: Name of the view to check.
          selected_facets: If passed, needs to be a subset of the facets selected by the view, otherwise the
              result will be False. If no ``selected_facets`` are passed, check for those selected by the
              active or indicated view.

        Returns:
          True if for each selected facet at least one file has been registered.
        """
        view = self.get_view(view_name)
        view_facets = view.selected_facets
        if selected_facets is None:
            facets = view_facets
        else:
            facets = resolve_facets_param(selected_facets, none_means_all=False)
            missing = [f for f in facets if f not in view_facets]
            if len(missing) > 0:
                plural = "s are" if len(missing) > 1 else " is"
                self.logger.warning(
                    f"The following facet{plural} excluded from the view '{view.name}': {missing}"
                )
                return False
        present_facets = [
            typ
            for typ, _ in self.iter_facet2files(
                view_name=view_name, include_empty=False
            )
        ]
        result = all(f in present_facets for f in facets)
        if not result:
            missing = [f for f in facets if f not in present_facets]
            plural = "s are" if len(missing) > 1 else " is"
            self.logger.debug(
                f"The following facet{plural} not present under the view '{view.name}': {missing}"
            )
        return result

    @property
    def files(self):
        return list(self.ix2file.values())

    @property
    def has_changed_scores(self) -> bool:
        """Whether any of the parsed scores has been altered."""
        for ix, score in self.ix2parsed_score.items():
            if score.mscx.changed:
                return True
        return False

    @property
    def ms(self):
        return self._ms

    @ms.setter
    def ms(self, ms):
        self._ms = get_musescore(ms, logger=self.logger)

    @overload
    def score_metadata(
        self,
        view_name: Optional[str],
        choose: Literal["auto", "ask"],
        as_dict: Literal[False],
    ) -> pd.Series: ...

    @overload
    def score_metadata(
        self,
        view_name: Optional[str],
        choose: Literal["auto", "ask"],
        as_dict: Literal[True],
    ) -> dict: ...

    def score_metadata(
        self,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        as_dict: bool = False,
    ) -> Union[pd.Series, dict, Literal[None]]:
        """

        Args:
          choose:
          as_dict: Set to True to change the return type from :obj:`pandas.Series` to :obj:`dict`.

        Returns:

        """
        file, score = self.get_parsed_score(view_name=view_name, choose=choose)
        if score is None:
            return None
        meta_dict = score.mscx.metadata
        meta_dict["subdirectory"] = file.subdir
        meta_dict["piece"] = self.name
        meta_dict["rel_path"] = file.rel_path
        if as_dict:
            return meta_dict
        return metadata2series(meta_dict)

    @property
    def tsv_metadata(self) -> Optional[Dict[str, str]]:
        """If the :class:`~.corpus.Corpus` has :attr:`~.corpus.Corpus.metadata_tsv`, this field will contain the
        {column: value} pairs of the row pertaining to this piece.
        """
        return self._tsv_metadata

    def metadata(self, view_name: Optional[str] = None) -> Optional[pd.Series]:
        """If a row of 'metadata.tsv' has been stored, return that, otherwise extract from a (force-)parsed score."""
        if self.tsv_metadata is not None:
            return self.tsv_metadata
        return self.score_metadata(view_name)

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

    @property
    def view(self):
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

    def __getattr__(self, view_name):
        if view_name in self.view_names:
            if view_name != self.view_name:
                self.switch_view(view_name, show_info=False)
            return self
        else:
            raise AttributeError(
                f"'{view_name}' is not an existing view. Use _.get_view('{view_name}') to create it."
            )

    def __getitem__(self, ix) -> ParsedFile:
        return self._get_parsed_at_index(ix)

    def __repr__(self):
        return self.info(return_str=True)

    def cadences(
        self,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
    ) -> Optional[pd.DataFrame]:
        file, df = self.get_facet(
            "cadences",
            view_name=view_name,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
        )
        self.logger.debug(f"Returning {file}.")
        return df

    def chords(
        self,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
    ) -> Optional[pd.DataFrame]:
        file, df = self.get_facet(
            "chords",
            view_name=view_name,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
        )
        self.logger.debug(f"Returning {file}.")
        return df

    def events(
        self,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
    ) -> Optional[pd.DataFrame]:
        file, df = self.get_facet(
            "events",
            view_name=view_name,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
        )
        self.logger.debug(f"Returning {file}.")
        return df

    def expanded(
        self,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
    ) -> Optional[pd.DataFrame]:
        file, df = self.get_facet(
            "expanded",
            view_name=view_name,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
        )
        self.logger.debug(f"Returning {file}.")
        return df

    def form_labels(
        self,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
    ) -> Optional[pd.DataFrame]:
        file, df = self.get_facet(
            "form_labels",
            view_name=view_name,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
        )
        self.logger.debug(f"Returning {file}.")
        return df

    def labels(
        self,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
    ) -> Optional[pd.DataFrame]:
        file, df = self.get_facet(
            "labels",
            view_name=view_name,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
        )
        self.logger.debug(f"Returning {file}.")
        return df

    def measures(
        self,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
    ) -> Optional[pd.DataFrame]:
        file, df = self.get_facet(
            "measures",
            view_name=view_name,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
        )
        self.logger.debug(f"Returning {file}.")
        return df

    def notes(
        self,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
    ) -> Optional[pd.DataFrame]:
        file, df = self.get_facet(
            "notes",
            view_name=view_name,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
        )
        self.logger.debug(f"Returning {file}.")
        return df

    def notes_and_rests(
        self,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
    ) -> Optional[pd.DataFrame]:
        file, df = self.get_facet(
            "notes_and_rests",
            view_name=view_name,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
        )
        self.logger.debug(f"Returning {file}.")
        return df

    def rests(
        self,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
    ) -> Optional[pd.DataFrame]:
        file, df = self.get_facet(
            "rests",
            view_name=view_name,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
        )
        self.logger.debug(f"Returning {file}.")
        return df

    def score(
        self, view_name: Optional[str] = None, choose: Literal["auto", "ask"] = "auto"
    ) -> Optional[Score]:
        file, score = self.get_parsed("scores", view_name=view_name, choose=choose)
        self.logger.debug(f"Returning {file}.")
        return score

    def add_parsed_score(self, ix: int, score_obj: Score) -> None:
        assert (
            ix in self.ix2file
        ), f"Piece '{self.name}' does not include a file with index {ix}."
        if score_obj is None:
            file = self.ix2file[ix]
            self.logger.debug(
                f"I was promised the parsed score for '{file.rel_path}' but received None."
            )
            return
        self.ix2parsed[ix] = score_obj
        self.ix2parsed_score[ix] = score_obj
        self.facet2parsed["scores"][ix] = score_obj

    def add_parsed_tsv(self, ix: int, parsed_tsv: pd.DataFrame) -> None:
        assert (
            ix in self.ix2file
        ), f"Piece '{self.name}' does not include a file with index {ix}."
        if parsed_tsv is None:
            file = self.ix2file[ix]
            self.logger.debug(
                f"I was promised the parsed DataFrame for '{file.rel_path}' but received None."
            )
            return
        self.ix2parsed[ix] = parsed_tsv
        self.ix2parsed_tsv[ix] = parsed_tsv
        inferred_type = infer_tsv_type(parsed_tsv)
        file = self.ix2file[ix]
        if file.type != inferred_type:
            if inferred_type == "unknown":
                self.logger.info(
                    f"After parsing '{file.rel_path}', the original guess that it contains '{file.type}' "
                    f"seems to be False and I'm attributing it to the facet '{file.type}'."
                )
            else:
                self.logger.info(
                    f"File {file.rel_path} turned out to contain '{inferred_type}' instead of '{file.type}', "
                    f"as I had guessed from its path."
                )
            self.facet2files[file.type].remove(file)
            file.type = inferred_type
            self.facet2files[inferred_type].append(file)
        self.facet2parsed[inferred_type][ix] = parsed_tsv
        if inferred_type in ("labels", "expanded"):
            self.ix2annotations = Annotations(df=parsed_tsv)

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
        """Update :obj:`Piece.labels_cfg` and retrieve new 'labels' tables accordingly.

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
        for score in self.ix2parsed_score.values():
            score.change_labels_cfg(labels_cfg=self.labels_cfg)

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
        for file, score in self.get_parsed_scores(view_name=view_name):
            if key in score._detached_annotations:
                changes = score.compare_labels(
                    key=key,
                    new_color=new_color,
                    old_color=old_color,
                    detached_is_newer=detached_is_newer,
                    add_to_rna=add_to_rna,
                    metadata_update=metadata_update,
                    force_metadata_update=force_metadata_update,
                )
                if changes > (0, 0):
                    changed += 1
                else:
                    unchanged += 1
        return changed, unchanged

    def count_changed_scores(self, view_name: Optional[str]) -> int:
        parsed_scores = self.get_parsed_scores(view_name=view_name)
        return sum(score.mscx.changed for _, score in parsed_scores)

    def count_parsed(
        self, include_empty=False, view_name: Optional[str] = None, prefix: bool = False
    ) -> Dict[str, int]:
        result = {}
        for typ, parsed in self.iter_facet2parsed(
            view_name=view_name, include_empty=include_empty
        ):
            key = "parsed_" + typ if prefix else typ
            result[key] = len(parsed)
        return result

    def count_detected(
        self,
        include_empty: bool = False,
        view_name: Optional[str] = None,
        prefix: bool = False,
    ) -> Dict[str, int]:
        """Count how many files per facet have been detected.

        Args:
          include_empty:
              By default, facets without files are not included in the dict. Pass True to include zero counts.
          view_name:
          prefix: Pass True if you want the facets prefixed with 'detected_'.

        Returns:
          {facet -> count of detected files}
        """
        result = {}
        for facet, files in self.iter_facet2files(
            view_name=view_name, include_empty=include_empty
        ):
            key = "detected_" + facet if prefix else facet
            result[key] = len(files)
        return result

    def extract_facet(
        self,
        facet: ScoreFacet,
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
    ) -> FileDataframeTupleMaybe:
        facet = check_argument_against_literal_type(
            facet, ScoreFacet, logger=self.logger
        )
        assert facet is not None, f"Pass a valid facet {ScoreFacet.__args__}"
        assert (
            choose != "all"
        ), "If you want to choose='all', use _.extract_facets() (plural)."
        df_list = self.extract_facets(
            facets=facet,
            view_name=view_name,
            force=force,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
            flat=True,
        )
        if len(df_list) == 0:
            return None, None
        if len(df_list) == 1:
            return df_list[0]

    def extract_facets(
        self,
        facets: ScoreFacets = None,
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["all", "auto", "ask"] = "all",
        unfold: bool = False,
        interval_index: bool = False,
        flat=False,
    ) -> Union[Dict[str, List[FileDataframeTuple]], List[FileDataframeTuple]]:
        """Retrieve a dictionary with the selected feature matrices extracted from the parsed scores.
        If you want to retrieve parsed TSV files, use :py:meth:`get_all_parsed`.
        """
        selected_facets = resolve_facets_param(facets, ScoreFacet, logger=self.logger)
        assert selected_facets is not None, f"Pass a valid facet {ScoreFacet.__args__}"
        result = defaultdict(list)
        score_files = self.get_parsed_scores(
            view_name=view_name, force=force, choose=choose
        )
        if len(score_files) == 0:
            return [] if flat else {facet: [] for facet in selected_facets}
        for file, score_obj in score_files:
            if score_obj is None:
                self.logger.info(f"No parsed score found for '{self.name}'")
                continue
            for facet in selected_facets:
                df = getattr(score_obj.mscx, facet)(
                    interval_index=interval_index, unfold=unfold
                )
                if df is None:
                    self.logger.debug(
                        f"Score({file.rel_path}).{facet}(unfold={unfold}, interval_index={interval_index}) returned "
                        f"None."
                    )
                else:
                    result[facet].append((file, df))
        if flat:
            return sum(result.values(), [])
        else:
            result = {
                facet: result[facet] if facet in result else []
                for facet in selected_facets
            }
        return result

    def get_changed_scores(self, view_name: Optional[str]) -> List[FileScoreTuple]:
        parsed_scores = self.get_parsed_scores(view_name=view_name)
        return [(file, score) for file, score in parsed_scores if score.mscx.changed]

    def get_facets(
        self,
        facets: ScoreFacets = None,
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["all", "auto", "ask"] = "all",
        unfold: bool = False,
        interval_index: bool = False,
        flat=False,
    ) -> Union[Dict[str, FileDataframeTuple], List[FileDataframeTuple]]:
        """
        Retrieve score facets both freshly extracted from parsed scores and from parsed TSV files, depending on
        the parameters and the view in question.

        If choose != 'all', the goal will be to return one DataFrame per facet. Preference is given to a DataFrame
        freshly extracted from an already parsed score; otherwise, from an already parsed TSV file. If both
        are not available, preference will be given to a force-parsed TSV, then to a force-parsed score.


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

        Returns:

        """
        selected_facets = resolve_facets_param(facets, ScoreFacet, logger=self.logger)
        assert (
            selected_facets is not None
        ), f"Pass at least one valid facet {ScoreFacet.__args__}"

        def merge_dicts(extracted_facets, parsed_facets):
            nonlocal selected_facets
            result = defaultdict(list)
            for facet in selected_facets:
                present_in_score = facet in extracted_facets
                present_as_tsv = facet in parsed_facets
                if present_in_score:
                    result[facet].extend(extracted_facets[facet])
                if present_as_tsv:
                    result[facet].extend(parsed_facets[facet])
                if not (present_in_score or present_as_tsv):
                    result[facet] = []
            return result

        def make_result(extracted_facets, parsed_facets=None):
            if parsed_facets is None:
                result = extracted_facets
            else:
                result = merge_dicts(extracted_facets, parsed_facets)
            if flat:
                return sum(result.values(), [])
            return dict(result)

        if choose == "all":
            extracted_facets = self.extract_facets(
                facets=selected_facets,
                view_name=view_name,
                force=force,
                unfold=unfold,
                interval_index=interval_index,
            )
            parsed_facets = self.get_all_parsed(
                facets=selected_facets,
                view_name=view_name,
                force=force,
                unfold=unfold,
                interval_index=interval_index,
            )
            # TODO: Unfold & interval_index for parsed facets
            return make_result(extracted_facets, parsed_facets)

        # The rest below makes sure that there is only one DataFrame per facet, if available
        extracted_facets = self.extract_facets(
            facets=selected_facets,
            view_name=view_name,
            force=False,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
        )
        missing_facets = [
            facet for facet in selected_facets if len(extracted_facets[facet]) == 0
        ]
        if len(missing_facets) == 0:
            return make_result(extracted_facets)
        # facets missing, look for parsed TSV files
        parsed_facets = self.get_all_parsed(
            facets=missing_facets,
            view_name=view_name,
            force=False,
            choose=choose,
            include_empty=True,
            unfold=unfold,
            interval_index=interval_index,
        )
        result = merge_dicts(extracted_facets, parsed_facets)
        missing_facets = [facet for facet in selected_facets if len(result[facet]) == 0]
        if len(missing_facets) == 0 or not force:
            return make_result(result)
        # there are still facets missing; force-parse TSV files first
        parsed_facets = self.get_all_parsed(
            facets=missing_facets,
            view_name=view_name,
            force=True,
            choose=choose,
            include_empty=True,
            unfold=unfold,
            interval_index=interval_index,
        )
        result = merge_dicts(result, parsed_facets)
        missing_facets = [facet for facet in selected_facets if len(result[facet]) == 0]
        if len(missing_facets) == 0 or not force:
            return make_result(result)
        # there are still facets missing; force-parse scores as last resort
        extracted_facets = self.extract_facets(
            facets=selected_facets,
            view_name=view_name,
            force=True,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
        )
        return make_result(result, extracted_facets)

    def get_facet(
        self,
        facet: ScoreFacet,
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
    ) -> FileDataframeTupleMaybe:
        """Retrieve a DataFrame from a parsed score or, if unavailable, from a parsed TSV. If none have been
        parsed, first force-parse a TSV and, if not included in the given view, force-parse a score.
        """
        facet = check_argument_against_literal_type(
            facet, ScoreFacet, logger=self.logger
        )
        assert facet is not None, f"Pass a valid facet {ScoreFacet.__args__}"
        assert (
            choose != "all"
        ), "If you want to choose='all', use _.extract_facets() (plural)."
        df_list = self.get_facets(
            facets=facet,
            view_name=view_name,
            force=force,
            choose=choose,
            unfold=unfold,
            interval_index=interval_index,
            flat=True,
        )
        if len(df_list) == 0:
            return None, None
        if len(df_list) == 1:
            return df_list[0]

    def get_file(
        self,
        facet: Facet,
        view_name: Optional[str] = None,
        parsed: bool = True,
        unparsed: bool = True,
        choose: Literal["auto", "ask"] = "auto",
    ) -> Optional[File]:
        """

        Args:
          facet:
          choose:

        Returns:
          A {file_type -> [:obj:`File`] dict containing the selected Files or, if flat=True, just a list.
        """
        facet = check_argument_against_literal_type(facet, Facet, logger=self.logger)
        assert facet is not None, f"Pass a valid facet {Facet.__args__}"
        assert (
            choose != "all"
        ), "If you want to choose='all', use _.get_files() (plural)."
        files = self.get_files(
            facets=facet,
            view_name=view_name,
            parsed=parsed,
            unparsed=unparsed,
            choose=choose,
            flat=True,
        )
        if len(files) == 0:
            return None
        if len(files) == 1:
            return files[0]

    def get_file_from_path(self, full_path: Optional[str] = None) -> Optional[File]:
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
        include_empty: bool = False,
    ) -> Union[Dict[str, FileList], FileList]:
        """

        Args:
          facets:

        Returns:
          A {file_type -> [:obj:`File`] dict containing the selected Files or, if flat=True, just a list.
        """
        assert (
            parsed + unparsed > 0
        ), "At least one of 'parsed' and 'unparsed' needs to be True."
        selected_facets = resolve_facets_param(facets, logger=self.logger)
        if selected_facets is None:
            return
        view = self.get_view(view_name=view_name)
        selected_facets = [
            facet for facet in selected_facets if facet in view.selected_facets
        ]
        if unparsed:
            facet2files = {f: self.facet2files[f] for f in selected_facets}
        else:
            # i.e., parsed must be True
            facet2files = {f: self.facet2files[f] for f in selected_facets}
            facet2files = {
                typ: [f for f in files if f.ix in self.ix2parsed]
                for typ, files in facet2files.items()
            }
        if not parsed:
            # i.e., unparsed must be True
            facet2files = {
                typ: [f for f in files if f.ix not in self.ix2parsed]
                for typ, files in facet2files.items()
            }
        facet2files = {
            typ: view.filtered_file_list(files) for typ, files in facet2files.items()
        }
        result = {}
        needs_choice = []
        for facet, files in facet2files.items():
            n_files = len(files)
            if n_files == 0 and not include_empty:
                continue
            elif choose == "all" or n_files < 2:
                result[facet] = files
            else:
                selected = files2disambiguation_dict(files, logger=self.logger)
                needs_choice.append(facet)
                result[facet] = selected
        if choose == "auto":
            for typ in needs_choice:
                result[typ] = [
                    automatically_choose_from_disambiguated_files(
                        result[typ], self.name, typ
                    )
                ]
        elif choose == "ask":
            for typ in needs_choice:
                choices = result[typ]
                selected = ask_user_to_choose_from_disambiguated_files(
                    choices, self.name, typ
                )
                if selected is None:
                    if include_empty:
                        result[typ] = []
                    else:
                        del result[typ]
                else:
                    result[typ] = [selected]
        elif choose == "all" and "scores" in needs_choice:
            # check if any scores can be differentiated solely by means of their file extension
            several_score_files = result["scores"].values()
            subdir_pieces = [(file.subdir, file.piece) for file in several_score_files]
            if len(set(subdir_pieces)) < len(subdir_pieces):
                duplicates = {
                    tup: [] for tup, cnt in Counter(subdir_pieces).items() if cnt > 1
                }
                for file in several_score_files:
                    if (file.subdir, file.piece) in duplicates:
                        duplicates[(file.subdir, file.piece)].append(file.rel_path)
                display_duplicates = "\n".join(
                    str(sorted(files)) for files in duplicates.values()
                )
                self.logger.warning(
                    f"The following scores are lying in the same subfolder and have the same name:\n"
                    f"{display_duplicates}.\n"
                    f"TSV files extracted from them will be overwriting each other. Consider excluding certain "
                    f"file extensions or letting me choose='auto'."
                )
        if flat:
            return sum(result.values(), start=[])
        return result

    def _get_parsed_at_index(self, ix: int) -> ParsedFile:
        assert (
            ix in self.ix2file
        ), f"Piece '{self.name}' does not include a file with index {ix}."
        if ix not in self.ix2parsed:
            self._parse_file_at_index(ix)
        if ix not in self.ix2parsed:
            file = self.ix2file[ix]
            raise RuntimeError(f"Unable to parse '{file.rel_path}'.")
        return self.ix2parsed[ix]

    def get_parsed(
        self,
        facet: Facet,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        git_revision: Optional[str] = None,
        unfold: bool = False,
        interval_index: bool = False,
    ) -> FileParsedTupleMaybe:
        """Retrieve exactly one parsed score or TSV file. If none has been parsed, parse one automatically.

        Args:
          facet:
          view_name:
          choose:
          git_revision:

        Returns:

        """
        facet = check_argument_against_literal_type(facet, Facet, logger=self.logger)
        assert facet is not None, f"Pass a valid facet {Facet.__args__}"
        assert choose != "all", "If you want to choose='all', use _.get_all_parsed()."
        if git_revision is not None:
            assert (
                facet != "scores"
            ), f"I don't parse scores from older commits. Check out {git_revision} yourself."
        files = self.get_all_parsed(
            facets=facet,
            view_name=view_name,
            choose=choose,
            flat=True,
            unfold=unfold,
            interval_index=interval_index,
        )
        if len(files) == 0:
            file = self.get_file(
                facet, view_name=view_name, parsed=False, choose=choose
            )
            if file is None:
                return None, None
            if git_revision is None:
                parsed = self._get_parsed_at_index(file.ix)
                if file.type != facet:
                    # i.e., after parsing the file turned out to be of a different type than inferred from its path
                    return None, None
                return file, parsed
        else:
            if git_revision is None:
                return files[0]
            file = files[0][0]
        return parse_tsv_file_at_git_revision(file=file, git_revision=git_revision)

    def get_all_parsed(
        self,
        facets: FacetArguments = None,
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["all", "auto", "ask"] = "all",
        flat: bool = False,
        include_empty: bool = False,
        unfold: bool = False,
        interval_index: bool = False,
    ) -> Union[Dict[Facet, List[FileParsedTuple]], List[FileParsedTuple]]:
        """Return multiple parsed files."""
        selected_facets = resolve_facets_param(facets, logger=self.logger)
        if selected_facets is None:
            return [] if flat else {}
        facet2files = self.get_files(
            selected_facets, view_name=view_name, include_empty=include_empty
        )
        result = {}
        for facet, files in facet2files.items():
            if len(files) == 0:
                # implies include_empty=True
                result[facet] = []
                continue
            parsed_files = [file for file in files if file.ix in self.ix2parsed]
            unparsed_files = [file for file in files if file.ix not in parsed_files]
            n_parsed = len(parsed_files)
            n_unparsed = len(unparsed_files)
            if choose == "all":
                if force:
                    if n_unparsed > 0:
                        for file in unparsed_files:
                            self._parse_file_at_index(file.ix)
                else:
                    files = parsed_files
            # otherwise, each facet will have a list of 1 or 0 elements (or is skipped if include_empty=False)
            elif n_parsed == 0:
                if force:
                    if n_unparsed > 1:
                        selected = disambiguate_files(
                            unparsed_files,
                            self.name,
                            facet,
                            choose=choose,
                            logger=self.logger,
                        )
                        if selected is None:
                            if include_empty:
                                result[facet] = []
                            continue
                    else:
                        selected = unparsed_files[0]
                    self._parse_file_at_index(selected.ix)
                    files = [selected]
                else:
                    files = []
            elif n_parsed == 1:
                files = parsed_files
            else:
                selected = disambiguate_files(
                    parsed_files, self.name, facet, choose=choose, logger=self.logger
                )
                if selected is None:
                    if include_empty:
                        result[facet] = []
                    continue
                files = [selected]
            if n_unparsed > 0:
                plural = "files" if n_unparsed > 1 else "file"
                try:
                    self.logger.debug(
                        f"Disregarded {n_unparsed} unparsed {facet} {plural}. Set force=True to automatically parse."
                    )
                except AttributeError:
                    if self.logger is None:
                        raise RuntimeError(
                            "The logger is None. This happens when __getstate__ is called. Did you use copy()?"
                        )
                    raise
            parsed_files = [
                (
                    file,
                    self._get_transformed_facet_at_ix(
                        ix=file.ix, unfold=unfold, interval_index=interval_index
                    ),
                )
                for file in files
                if file.ix in self.ix2parsed
            ]
            parsed_files = [(file, df) for file, df in parsed_files if df is not None]
            n_parsed = len(parsed_files)
            if n_parsed == 0 and not include_empty:
                continue
            result[facet] = parsed_files
        if flat:
            return sum(result.values(), start=[])
        return result

    def _get_transformed_facet_at_ix(
        self, ix: int, unfold: bool = False, interval_index: bool = False
    ) -> Optional[ParsedFile]:
        """Retrieves a parsed TSV file, adds quarterbeats if missing and, if requested, unfolds repeats or adds a
        :obj:`pandas.IntervalIndex`.
        """
        if ix not in self.ix2parsed:
            return None
        if ix not in self.ix2parsed_tsv:
            # this is a score and will not be transformed in any way
            return self.ix2parsed[ix]
        df = self.ix2parsed_tsv[ix]
        qb_missing = any(c not in df.columns for c in ("quarterbeats", "duration_qb"))
        if interval_index or unfold or qb_missing:
            file = self.ix2file[ix]
            if unfold or qb_missing:
                if file.type != "measures":
                    _, measures = self.get_facet("measures")
                else:
                    measures = df
                if measures is None:
                    if unfold:
                        self.logger.warning(
                            f"Piece.get_facet('measures') did not return a measures table, which is required for "
                            f"unfolding repeats. Make sure that the view includes a TSV file or a score for "
                            f"'{self.name}' so I can get it. Returning None for now."
                        )
                        return None
                    # else: qb_missing
                    self.logger.warning(
                        f"Piece.get_facet('measures') did not return a measures table, which is required for adding "
                        f"the missing columns 'quarterbeats' and 'duration_qb'. Make sure that the view includes a "
                        f"TSV file or a score for '{self.name}' so I can get it. Returning a DataFrame without these "
                        f"columns for now."
                    )
                    return df
                transformed = dfs2quarterbeats(
                    df,
                    measures=measures,
                    unfold=unfold,
                    interval_index=interval_index,
                    logger=self.logger,
                )
                if len(transformed) == 0:
                    return
                df = transformed[0]
            else:
                df = replace_index_by_intervals(df, logger=self.logger)
        return df

    def get_parsed_score(
        self, view_name: Optional[str] = None, choose: Literal["auto", "ask"] = "auto"
    ) -> FileScoreTupleMaybe:
        return self.get_parsed("scores", view_name=view_name, choose=choose)

    def get_parsed_scores(
        self,
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["all", "auto", "ask"] = "all",
    ) -> List[FileScoreTuple]:
        return self.get_all_parsed(
            "scores", view_name=view_name, force=force, choose=choose, flat=True
        )

    def get_parsed_tsv(
        self,
        facet: TSVtype,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        unfold: bool = False,
        interval_index: bool = False,
    ) -> FileDataframeTupleMaybe:
        facets = argument_and_literal_type2list(facet, TSVtype, logger=self.logger)
        assert (
            len(facets) == 1
        ), f"Pass exactly one valid TSV type {literal_type2tuple(TSVtype)} or use _.get_parsed_tsvs()\nGot: {facets}"
        facet = facets[0]
        return self.get_parsed(facet, view_name=view_name, choose=choose)

    def get_parsed_tsvs(
        self,
        facets: TSVtypes,
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["all", "auto", "ask"] = "all",
    ) -> List[FileDataframeTupleMaybe]:
        facets = argument_and_literal_type2list(facets, TSVtype, logger=self.logger)
        return self.get_all_parsed(
            facets, view_name=view_name, force=force, choose=choose, flat=True
        )

    def _get_parsed_score_files(self, view_name: Optional[str] = None) -> FileList:
        return self.get_files("scores", view_name=view_name, unparsed=False, flat=True)

    def _get_parsed_tsv_files(
        self, view_name: Optional[str] = None, flat: bool = True
    ) -> Union[FileDict, FileList]:
        return self.get_files("tsv", view_name=view_name, unparsed=False, flat=flat)

    def _get_unparsed_score_files(self, view_name: Optional[str] = None) -> FileList:
        return self.get_files("scores", view_name=view_name, parsed=False, flat=True)

    def _get_unparsed_tsv_files(
        self, view_name: Optional[str] = None, flat: bool = True
    ) -> Union[FileDict, FileList]:
        return self.get_files("tsv", view_name=view_name, parsed=False, flat=flat)

    def info(self, return_str=False, view_name=None, show_discarded: bool = False):
        header = f"Piece '{self.name}'"
        header += "\n" + "-" * len(header) + "\n"

        # get parsed scores before resetting the view's filtering counts to prevent counting twice
        parsed_scores = self.get_parsed_scores(view_name=view_name)

        # start info message with the names of the available views, the header, and info on the active view.
        view = self.get_view(view_name)
        view.reset_filtering_data()
        msg = available_views2str(self._views, view_name)
        msg += header
        msg += f"View: {view}\n\n"

        # Show info on all files included in the active view
        facet2files = dict(
            self.iter_facet2files(view_name=view_name, include_empty=False)
        )
        if len(facet2files) == 0:
            msg += "No files selected."
        else:
            files_df = pd.concat(
                [pd.DataFrame(files).set_index("ix") for files in facet2files.values()],
                keys=facet2files.keys(),
                names=["facet", "ix"],
            )
            if len(files_df) == 0:
                msg += "No files selected."
            else:
                is_parsed, has_changed = [], []
                for facet, ix in files_df.index:
                    parsed = ix in self.ix2parsed
                    is_parsed.append(parsed)
                    changed_score = (
                        parsed and facet == "scores" and self[ix].mscx.changed
                    )
                    has_changed.append(changed_score)
                files_df["is_parsed"] = is_parsed
                if any(has_changed):
                    files_df["has_changed"] = has_changed
                    info_columns = ["rel_path", "is_parsed", "has_changed"]
                else:
                    info_columns = ["rel_path", "is_parsed"]
                msg += files_df[info_columns].to_string()
        changed_score_ixs = []
        ix2detached_annotations = {}
        for file, score in parsed_scores:
            if score.mscx.changed:
                changed_score_ixs.append(file.ix)
            if len(score._detached_annotations) > 0:
                ix2detached_annotations[file.ix] = list(
                    score._detached_annotations.keys()
                )
        has_changed = len(changed_score_ixs) > 0
        has_detached = len(ix2detached_annotations) > 0
        if has_changed or has_detached:
            msg += "\n\n"
            if has_changed:
                plural = (
                    f"Scores {changed_score_ixs} have"
                    if len(changed_score_ixs) > 1
                    else f"Score {changed_score_ixs[0]} has"
                )
                msg += f"{plural} changed since parsing."
            if has_detached:
                msg += pretty_dict(ix2detached_annotations, "ix", "Loaded annotations")
        msg += "\n\n" + view.filtering_report(
            show_discarded=show_discarded, return_str=True
        )
        if return_str:
            return msg
        print(msg)

    def iter_extracted_facet(
        self,
        facet: ScoreFacet,
        view_name: Optional[str] = None,
        force: bool = False,
        unfold: bool = False,
        interval_index: bool = False,
    ) -> Iterator[FileDataframeTupleMaybe]:
        """Iterate through the selected facet extracted from all parsed or yet-to-parse scores."""
        facet = check_argument_against_literal_type(
            facet, ScoreFacet, logger=self.logger
        )
        assert facet is not None, f"Pass a valid facet {ScoreFacet.__args__}"
        for file in self.iter_files(
            "scores",
            view_name=view_name,
            flat=True,
        ):
            if file.ix not in self.ix2parsed and not force:
                continue
            score_obj = self._get_parsed_at_index(file.ix)
            if score_obj is None:
                self.logger.info(f"No parsed score found for '{file.rel_path}'")
                continue
            df = getattr(score_obj.mscx, facet)(
                interval_index=interval_index, unfold=unfold
            )
            if df is None:
                self.logger.debug(
                    f"Score({file.rel_path}).{facet}(unfold={unfold}, interval_index={interval_index}) returned None."
                )
                continue
            yield file, df

    def iter_extracted_facets(
        self,
        facets: ScoreFacets,
        view_name: Optional[str] = None,
        force: bool = False,
        unfold: bool = False,
        interval_index: bool = False,
    ) -> Iterator[Tuple[File, DataframeDict]]:
        """Iterate through the selected facets extracted from all parsed or yet-to-parse scores."""
        selected_facets = resolve_facets_param(facets, ScoreFacet, logger=self.logger)
        assert (
            selected_facets is not None
        ), f"Pass at least one valid facet {ScoreFacet.__args__}"
        facet2dataframe = {}
        for file in self.iter_files(
            "scores",
            view_name=view_name,
            flat=True,
        ):
            if file.ix not in self.ix2parsed and not force:
                continue
            score_obj = self._get_parsed_at_index(file.ix)
            if score_obj is None:
                self.logger.info(f"No parsed score found for '{file.rel_path}'")
                continue
            for facet in selected_facets:
                df = getattr(score_obj.mscx, facet)(
                    interval_index=interval_index, unfold=unfold
                )
                if df is None:
                    self.logger.debug(
                        f"Score({file.rel_path}).{facet}(unfold={unfold}, interval_index={interval_index}) returned "
                        f"None."
                    )
                facet2dataframe[facet] = df
            yield file, facet2dataframe

    def iter_facet2files(
        self, view_name: Optional[str] = None, include_empty: bool = False
    ) -> Iterator[Tuple[str, FileList]]:
        """Iterating through :attr:`facet2files` under the current or specified view."""
        view = self.get_view(view_name=view_name)
        for facet, files in self.facet2files.items():
            if facet not in view.selected_facets:
                continue
            filtered_files = view.filtered_file_list(files)
            # the files need to be filtered even if the facet is excluded, for counting excluded files
            if len(filtered_files) == 0 and not include_empty:
                continue
            yield facet, filtered_files

    def iter_facet2parsed(
        self, view_name: Optional[str] = None, include_empty: bool = False
    ) -> Iterator[Dict[str, FileList]]:
        """
        Iterating through :attr:`facet2parsed` under the current or specified view and selecting only parsed files.
        """
        view = self.get_view(view_name=view_name)
        for facet, ix2parsed in self.facet2parsed.items():
            if facet not in view.selected_facets:
                continue
            files = [self.ix2file[ix] for ix in ix2parsed.keys()]
            filtered_ixs = [
                file.ix for file in view.filtered_file_list(files, "parsed")
            ]
            # the files need to be filtered even if the facet is excluded, for counting excluded files
            if len(filtered_ixs) == 0 and not include_empty:
                continue
            yield facet, {ix: ix2parsed[ix] for ix in filtered_ixs}

    def iter_files(
        self,
        facets: FacetArguments = None,
        view_name: Optional[str] = None,
        parsed: bool = True,
        unparsed: bool = True,
        choose: Literal["all", "auto", "ask"] = "all",
        flat: bool = False,
        include_empty: bool = False,
    ) -> Union[Iterator[FileDict], Iterator[FileList]]:
        """Equivalent to iterating through the result of :meth:`get_files`."""
        selected_files = self.get_files(
            facets=facets,
            view_name=view_name,
            parsed=parsed,
            unparsed=unparsed,
            choose=choose,
            flat=flat,
            include_empty=include_empty,
        )
        if flat:
            yield from selected_files
        else:
            yield from selected_files.items()

    def iter_parsed(
        self,
        facet: Facet = None,
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["all", "auto", "ask"] = "all",
        include_empty: bool = False,
        unfold: bool = False,
        interval_index: bool = False,
    ) -> Iterator[FileParsedTuple]:
        facet = check_argument_against_literal_type(facet, Facet, logger=self.logger)
        assert facet is not None, f"Pass a valid facet {Facet.__args__}"
        files = self.get_all_parsed(
            facets=facet,
            view_name=view_name,
            force=force,
            choose=choose,
            flat=True,
            include_empty=include_empty,
            unfold=unfold,
            interval_index=interval_index,
        )
        yield from files

    def _parse_file_at_index(self, ix: int) -> None:
        assert (
            ix in self.ix2file
        ), f"Piece '{self.name}' does not include a file with index {ix}."
        file = self.ix2file[ix]
        if file.type == "scores":
            logger_cfg = dict(self.logger_cfg)
            score = Score(
                file.full_path, labels_cfg=self.labels_cfg, ms=self.ms, **logger_cfg
            )
            if score is None:
                self.logger.warning(f"Parsing {file.rel_path} failed.")
            else:
                self.add_parsed_score(ix, score)
        else:
            df = load_tsv(file.full_path)
            if df is None:
                self.logger.warning(f"Parsing {file.rel_path} failed.")
            else:
                self.add_parsed_tsv(ix, df)

    def keys(self) -> List[int]:
        """Return the indices of all Files registered with this Piece."""
        return list(self.ix2file.keys())

    def load_annotation_table_into_score(
        self,
        ix: Optional[int] = None,
        df: Optional[pd.DataFrame] = None,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        key: str = "detached",
        infer: bool = True,
        **cols,
    ) -> None:
        """Attach an :py:class:`~.annotations.Annotations` object to the score and make it available as ``Score.{key}``.
        It can be an existing object or one newly created from the TSV file ``tsv_path``.

        Args:
          ix: Either pass the index of a TSV file containing annotations, or
          df: A DataFrame containing annotations.
          key:
              Specify a new key for accessing the set of annotations. The string needs to be usable
              as an identifier, e.g. not start with a number, not contain special characters etc. In return you
              may use it as a property: For example, passing ``'chords'`` lets you access the :py:class:`~.annotations.
              Annotations` as ``Score.chords``. The key 'annotations' is reserved for all annotations attached to the
              score.
          infer:
              By default, the label types are inferred in the currently configured order (see :py:attr:`name2regex`).
              Pass False to not add and not change any label types.
          **cols:
              If the columns in the specified TSV file diverge from the :ref:`standard column names<column_names>`,
              pass them as standard_name='custom name' keywords.
        """
        assert (ix is None) + (
            df is None
        ) == 1, "Pass either the index of a TSV file or a DataFrame with annotations."
        if ix is not None:
            assert (
                ix in self.ix2file
            ), f"Index {ix} is not associated with Piece '{self.name}'."
            file = self.ix2file[ix]
            df = self._get_parsed_at_index(ix)
            assert file.type in (
                "labels",
                "expanded",
            ), f"File needs to contain annotations, but {file.rel_path} is of type '{file.type}'."
        score_file, score = self.get_parsed_score(view_name=view_name, choose=choose)
        score.load_annotations(df=df, key=key, infer=infer, **cols)

    def load_facet_into_score(
        self,
        facet: AnnotationsFacet,
        view_name: Optional[str] = None,
        choose: Literal["auto", "ask"] = "auto",
        git_revision: Optional[str] = None,
        key: str = "detached",
        infer: bool = True,
        **cols,
    ) -> None:
        facet = check_argument_against_literal_type(
            facet, AnnotationsFacet, logger=self.logger
        )
        assert facet is not None, f"Pass a valid facet {AnnotationsFacet.__args__}"
        file, df = self.get_parsed(
            facet=facet,
            view_name=view_name,
            choose=choose,
            git_revision=git_revision,
        )
        self.load_annotation_table_into_score(
            df=df, view_name=view_name, choose=choose, key=key, infer=infer, **cols
        )

    def register_file(self, file: File, reject_incongruent_pnames: bool = True) -> bool:
        ix = file.ix
        if ix in self.ix2file:
            existing_file = self.ix2file[ix]
            if file.full_path == existing_file.full_path:
                self.logger.debug(
                    f"File '{file.rel_path}' was already registered for {self.name}."
                )
                return None
            else:
                self.logger.debug(
                    f"File '{existing_file.rel_path}' replaced with '{file.rel_path}'"
                )
        if file.piece != self.name:
            if file.piece.startswith(self.name):
                name_len = len(self.name)
                file.suffix = file.piece[name_len:]
                self.logger.debug(f"Recognized suffix '{file.suffix}' for {file.file}.")
            elif reject_incongruent_pnames:
                if self.name in file.piece:
                    self.logger.info(
                        f"{file.file} seems to come with a prefix w.r.t. '{self.name}' and is ignored."
                    )
                    return False
                else:
                    self.logger.warning(
                        f"{file.file} does not contain '{self.name}' and is ignored."
                    )
                    return False
        self.facet2files[file.type].append(file)
        self.ix2file[file.ix] = file
        return True

    def store_extracted_facet(
        self,
        facet: ScoreFacet,
        root_dir: Optional[str] = None,
        folder: Optional[str] = None,
        view_name: Optional[str] = None,
        force: bool = False,
        choose: Literal["all", "auto", "ask"] = "all",
        unfold: bool = False,
        interval_index: bool = False,
        frictionless: bool = True,
        raise_exception: bool = True,
        write_or_remove_errors_file: bool = True,
    ):
        """
        Extract a facet from one or several available scores and store the results as TSV files, the paths of which
        are computed from the respective score's location.

        Args:
          facet:
          root_dir:
              Defaults to None, meaning that the path is constructed based on the corpus_path.
              Pass a directory to construct the path relative to it instead. If ``folder`` is an absolute path,
              ``root_dir`` is ignored.
          folder:
              * If ``folder`` is None (default), the files' type will be appended to the ``root_dir``.
              * If ``folder`` is an absolute path, ``root_dir`` will be ignored.
              * If ``folder`` is a relative path starting with a dot ``.`` the relative path is appended to the file's
                subdir. For example, ``..\notes`` will resolve to a sibling directory of the one where the ``file``
                is located.
              * If ``folder`` is a relative path that does not begin with a dot ``.``, it will be appended to the
                ``root_dir``.
          view_name:
          force:
          choose:
          unfold:
          interval_index:
          frictionless:
            If True (default), the file is written together with a frictionless resource descriptor JSON file
            whose column schema is used to validate the stored TSV file.
          raise_exception:
              If True (default) raise if the resource is not valid. Only relevant when frictionless=True (i.e.,
              by default).
          write_or_remove_errors_file:
            If True (default) write a .errors file if the resource is not valid, otherwise remove it if it exists.
            Only relevant when frictionless=True (i.e., by default).




        Returns:

        """
        if choose == "all":
            extracted_facet = self.iter_extracted_facet(
                facet=facet,
                view_name=view_name,
                force=force,
                unfold=unfold,
                interval_index=interval_index,
            )
        else:
            extracted_facet = self.extract_facets(
                facets=facet, view_name=view_name, force=force, choose=choose, flat=True
            )
        create_descriptor = frictionless and facet != "events"
        for file, df in extracted_facet:
            piece_name = file.piece
            if unfold:
                piece_name += "_unfolded"
            directory = compute_path_from_file(file, root_dir=root_dir, folder=folder)
            version_info = None
            if create_descriptor:
                try:
                    version_info = get_git_version_info(
                        repo_path=file.directory,
                        only_if_clean=True,
                    )
                except AssertionError:
                    pass
            store_dataframe_resource(
                df=df,
                directory=directory,
                piece_name=piece_name,
                facet=facet,
                zipped=False,
                frictionless=create_descriptor,
                raise_exception=raise_exception,
                write_or_remove_errors_file=write_or_remove_errors_file,
                logger=self.logger,
                custom_metadata=version_info,
            )

    # def store_parsed_scores(self,
    #                         view_name: Optional[str] = None,
    #                         root_dir: Optional[str] = None,
    #                         folder: str = '.',
    #                         suffix: str = '',
    #                         overwrite: bool = False,
    #                         simulate=False) -> List[str]:
    #     stored_file_paths = []
    #     for file in self._get_parsed_score_files(view_name):
    #         file_path = self.store_parsed_score_at_ix(ix=file.ix,
    #                                       root_dir=root_dir,
    #                                       folder=folder,
    #                                       suffix=suffix,
    #                                       overwrite=overwrite,
    #                                       simulate=simulate)
    #         stored_file_paths.append(file_path)
    #     return stored_file_paths

    def store_parsed_score_at_ix(
        self,
        ix,
        root_dir: Optional[str] = None,
        folder: str = ".",
        suffix: str = "",
        overwrite: bool = False,
        simulate=False,
    ) -> Optional[str]:
        """
        Creates a MuseScore file from the Score object at the given index.

        Args:
          ix:
          folder:
          suffix: Suffix to append to the original file name.
          root_dir:
          overwrite: Pass True to overwrite existing files.
          simulate: Set to True if no files are to be written.

        Returns:
          Path of the stored file.
        """
        if ix not in self.ix2parsed_score:
            self.logger.error("No Score object found. Call parse_scores() first.")
            return
        file = self.ix2file[ix]
        file_path = make_file_path(
            file=file, root_dir=root_dir, folder=folder, suffix=suffix, fext=".mscx"
        )
        if os.path.isfile(file_path):
            if simulate:
                if overwrite:
                    self.logger.warning(f"Would have overwritten {file_path}.")
                    return
                self.logger.warning(f"Would have skipped {file_path}.")
                return
            elif not overwrite:
                self.logger.warning(
                    f"Skipping {file.rel_path} because the target file exists already and overwrite=False: {file_path}."
                )
                return
        if simulate:
            self.logger.debug(f"Would have written score to {file_path}.")
        else:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self[ix].store_score(file_path)
            self.logger.debug(f"Score written to {file_path}.")
        return file_path

    def switch_view(self, view_name: str, show_info: bool = True) -> None:
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
        if show_info:
            self.info()

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
              ``composer_text``, ``lyricist_text``, and ``part_name_text`` into the score header.
          update_instrumentation:
              Set to True to update the score's instrumentation based on changed values from 'staff_<i>_instrument'
              columns.

        Returns:
          List of File objects of those scores of which the XML structure has been modified.
        """
        if self.tsv_metadata is None:
            self.logger.info("No metadata available for updating scores.")
            return []
        updated_scores = []
        ignore_columns = AUTOMATIC_COLUMNS + LEGACY_COLUMNS + ["piece"]
        for file, score in self.iter_parsed(
            "scores", view_name=view_name, force=force, choose=choose
        ):
            MSCX = score.mscx.parsed
            current_metadata = MSCX.metatags.fields
            current_metadata.update(MSCX.prelims.fields)
            row_dict = {
                column: value
                for column, value in self.tsv_metadata.items()
                if column not in ignore_columns and not re.match(r"^staff_\d", column)
            }
            unused_fields = [
                field
                for field in current_metadata.keys()
                if field not in row_dict and field not in MUSESCORE_METADATA_FIELDS
            ]
            if write_empty_values:
                row_dict = {
                    column: "" if pd.isnull(value) else str(value)
                    for column, value in row_dict.items()
                }
            else:
                row_dict = {
                    column: str(value)
                    for column, value in row_dict.items()
                    if value != "" and not pd.isnull(value)
                }
            to_be_updated = {
                field: value
                for field, value in row_dict.items()
                if (field not in current_metadata and value != "")
                or (field in current_metadata and current_metadata[field] != value)
            }
            fields_to_be_created = [
                field for field in to_be_updated.keys() if field not in current_metadata
            ]
            metadata_fields, text_fields = {}, {}
            for field, value in to_be_updated.items():
                if field in MUSESCORE_HEADER_FIELDS:
                    text_fields[field] = value
                else:
                    metadata_fields[field] = value
            changed = False
            if write_text_fields and len(text_fields) > 0:
                for field, value in text_fields.items():
                    if pd.isnull(value):
                        continue
                    if value == "" and not write_empty_values:
                        continue
                    MSCX.prelims[field] = value
                    self.logger.debug(f"{file.rel_path}: {field} set to '{value}'.")
                    changed = True
            for field, value in metadata_fields.items():
                specifier = "New field" if field in fields_to_be_created else "Field"
                self.logger.debug(
                    f"{file.rel_path}: {specifier} '{field}' set to {value}."
                )
                MSCX.metatags[field] = value
                changed = True
            if remove_unused_fields:
                for field in unused_fields:
                    MSCX.metatags.remove(field)
                    self.logger.debug(f"{file.rel_path}: Field {field} removed.")
                    changed = True
            if update_instrumentation:
                current_values = MSCX.get_instrumentation()
                to_be_updated = {}
                for column, value in self.tsv_metadata.items():
                    if (m := re.match(r"^(staff_\d+)_instrument", column)) is None:
                        continue
                    staff_id = m.group(1)
                    if staff_id not in current_values:
                        continue
                    if pd.isnull(value):
                        self.logger.warning(
                            f"{file.full_path}: Instrumentation for staff {staff_id} is empty."
                        )
                    if value != current_values[staff_id]:
                        to_be_updated[staff_id] = value
                if len(to_be_updated) > 0:
                    changed = True
                    self.logger.debug(
                        f"This instrumentation will be written into the score:\n{to_be_updated}"
                    )
                    for staff, instrument in to_be_updated.items():
                        self.logger.debug(
                            f"{staff}: {current_values[staff]} => {instrument}"
                        )
                        MSCX.instrumentation.set_instrument(staff, instrument)
            if changed:
                MSCX.update_metadata()
                score.mscx.changed = True
                updated_scores.append(file)
            else:
                self.logger.debug(f"No metadata need updating in {file.rel_path}")
        return updated_scores

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
        selected_facets = resolve_facets_param(facets, ScoreFacet, logger=self.logger)
        assert selected_facets is not None, f"Pass a valid facet {ScoreFacet.__args__}"
        assert choose != "all", "This method does not accept choose='all'."
        written_paths = []
        parsed_tsvs = self.get_all_parsed(
            selected_facets, view_name=view_name, force=force, choose=choose
        )
        if len(parsed_tsvs) == 0:
            self.logger.info("No parsed TSV files to update.")
            return []
        score_file, score_obj = self.get_parsed_score(
            view_name=view_name, choose=choose
        )
        if score_obj is None:
            self.logger.info(
                "This method updates TSV files based on a score but the current view includes none."
            )
            return []
        for facet, file_df_tuples in parsed_tsvs.items():
            if len(file_df_tuples) > 1:
                self.logger.warning(
                    f"There are more than one TSV files containing {facet} and they will be compared with "
                    f"the same score."
                )
            score_file, new_df = self.extract_facet(
                facet, view_name=view_name, choose=choose
            )
            for tsv_file, old_df in file_df_tuples:
                try:
                    # missing_columns = [c for c in old_df.columns if c not in new_df.columns]
                    # if len(missing_columns) > 0:
                    #     plural = 'These columns are' if len(missing_columns) > 1 else 'This column is'
                    #     self.logger.warning(f"{plural} missing in the updated {facet}:\n{old_df[missing_columns]}")
                    # tmp_new = new_df[[c for c in old_df.columns if c in new_df.columns]]
                    # assert_frame_equal(old_df, tmp_new, check_dtype=False, obj=facet) # from pandas.testing import
                    # assert_frame_equal
                    assert_dfs_equal(old_df, new_df)
                    # TODO: make utils.assert_dfs_equal() use math.isclose() for comparing floats
                except AssertionError as e:
                    if facet == "expanded":
                        facet += " labels"
                    self.logger.warning(
                        f"The {facet} extracted from {score_file.rel_path} is not identical with the "
                        f"ones in {tsv_file.rel_path}:\n{e}"
                    )
                    continue
                write_tsv(new_df, tsv_file.full_path, logger=self.logger)
                written_paths.append(tsv_file.full_path)
        return written_paths

    def get_dataframe(self, *args, **kwargs) -> None:
        """Deprecated method. Replaced by :meth:`get_parsed`, :meth:`extract_facet`, and :meth:`get_facet()`."""
        raise DeprecationWarning(
            "Method not in use any more. Use _.get_parsed() to retrieve a parsed TSV file, "
            "_.extract_facet() to retrieve a freshly extracted DataFrame, "
            "or _.get_facet() to retrieve either, according to availability."
        )
