import os
from collections import defaultdict, Counter
from typing import Dict, Literal, Union, Iterator, Optional, overload, List, Tuple

import pandas as pd

from .annotations import Annotations
from ._typing import FileList, ParsedFile, FileDict, Facet, TSVtype, Facets, ScoreFacets, ScoreFacet, FileParsedTuple, FacetArguments, FileScoreTuple, \
    FileDataframeTupleMaybe, DataframeDict, FileDataframeTuple, TSVtypes, FileScoreTupleMaybe, FileParsedTupleMaybe
from .utils import File, infer_tsv_type, automatically_choose_from_disambiguated_files, ask_user_to_choose_from_disambiguated_files, \
    files2disambiguation_dict, get_musescore, load_tsv, metadata2series, pretty_dict, resolve_facets_param, \
    disambiguate_parsed_files, available_views2str, argument_and_literal_type2list, check_argument_against_literal_type, make_file_path, write_tsv, assert_dfs_equal
from .score import Score
from .logger import LoggedClass
from .view import View, DefaultView


class Piece(LoggedClass):
    """Wrapper around :class:`~.score.Score` for associating it with parsed TSV files"""

    def __init__(self, fname: str, view: View = None, logger_cfg={}, ms=None):
        super().__init__(subclass='Piece', logger_cfg=logger_cfg)
        self.name = fname
        available_types = ('scores',) + Score.dataframe_types
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
            self._views[None] = DefaultView()
        else:
            self._views[None] = view
            if view.name != 'default':
                self._views['default'] = DefaultView()
        self._views['all'] = View('all')
        self._ms = get_musescore(ms, logger=self.logger)
        """:obj:`str`
        Path or command of the local MuseScore 3 installation if specified by the user."""
        self._tsv_metadata: dict = None
        # self.score_obj: Score = None
        # self.score_metadata: pd.Series = None
        # """:obj:`pandas.Series`
        # Metadata from :attr:`.Score.metadata`, transformed by :func:`.utils.metadata2series`.
        # """
        # self.measures: pd.DataFrame = None
        # self.notes: pd.DataFrame = None
        # self.rests: pd.DataFrame = None
        # self.notes_and_rests: pd.DataFrame = None
        # self.labels: pd.DataFrame = None
        # self.expanded: pd.DataFrame = None
        # self.events: pd.DataFrame = None
        # self.chords: pd.DataFrame = None
        # self.form_labels: pd.DataFrame = None

    def all_facets_present(self, view_name: Optional[str] = None,
                           selected_facets: Optional[Facets] = None) -> bool:
        """ Checks if parsed files have been detected for all selected facets under the active or indicated view.

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
                plural = 's are' if len(missing) > 1 else ' is'
                self.logger.warning(f"The following facet{plural} excluded from the view '{view.name}': {missing}")
                return False
        present_facets = [typ for typ, _ in self.iter_facet2files(view_name=view_name, include_empty=False)]
        result = all(f in present_facets for f in facets)
        if not result:
            missing = [f for f in facets if f not in present_facets]
            plural = 's are' if len(missing) > 1 else ' is'
            self.logger.debug(f"The following facet{plural} are not present under the view '{view.name}': {missing}")
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
    def score_metadata(self, as_dict: Literal[False]) -> pd.Series:
        ...
    @overload
    def score_metadata(self, as_dict: Literal[True]) -> dict:
        ...
    def score_metadata(self, as_dict: bool = False) -> Union[pd.Series, dict]:
        """

        Args:
            force: Make sure you retrieve metadata, even if a score has to be automatically
            selected and parsed for that (default).

        Returns:

        """
        parsed_score = self.get_parsed_score()
        if parsed_score is None:
            return None
        file, score = parsed_score
        meta_dict = score.mscx.metadata
        meta_dict['subdirectory'] = file.subdir
        meta_dict['fname'] = self.name
        meta_dict['rel_path'] = file.rel_path
        if as_dict:
            return meta_dict
        return metadata2series(meta_dict)

    @property
    def tsv_metadata(self):
        return self._tsv_metadata

    def set_view(self, active: View = None, **views: View):
        """Register one or several view_name=View pairs."""
        if active is not None:
            new_name = active.name
            if new_name in self._views and active != self._views[new_name]:
                self.logger.info(f"The existing view called '{new_name}' has been overwritten")
                del (self._views[new_name])
            old_view = self._views[None]
            self._views[old_view.name] = old_view
            self._views[None] = active
        for view_name, view in views.items():
            if view.name is None:
                view.name = view_name
            self._views[view_name] = view

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
    def view(self):
        return self.get_view()


    @view.setter
    def view(self, new_view: View):
        if not isinstance(new_view, View):
            return TypeError("If you want to switch to an existing view, use its name like an attribute or "
                             "call _.switch_view().")
        self.set_view(new_view)

    @property
    def views(self):
        print(pretty_dict({"[active]" if k is None else k: v for k, v in self._views.items()}, "view_name",
                          "Description"))

    @property
    def view_name(self):
        return self.get_view().name

    @view_name.setter
    def view_name(self, new_name):
        view = self.get_view()
        view.name = new_name

    @property
    def view_names(self):
        return {view.name if name is None else name for name, view in self._views.items()}

    def __getattr__(self, view_name):
        if view_name in self.view_names:
            if view_name != self.view_name:
                self.switch_view(view_name, show_info=False)
            return self
        else:
            raise AttributeError(f"'{view_name}' is not an existing view. Use _.get_view('{view_name}') to create it.")

    def __getitem__(self, ix) -> ParsedFile:
        return self._get_parsed_at_index(ix)


    def __repr__(self):
        return self.info(return_str=True)
    def add_parsed_score(self, ix: int, score_obj: Score) -> None:
        assert ix in self.ix2file, f"Piece '{self.name}' does not include a file with index {ix}."
        if score_obj is None:
            file = self.ix2file[ix]
            self.logger.debug(f"I was promised the parsed score for '{file.rel_path}' but received None.")
            return
        self.ix2parsed[ix] = score_obj
        self.ix2parsed_score[ix] = score_obj
        self.facet2parsed['scores'][ix] = score_obj

    def add_parsed_tsv(self, ix: int, parsed_tsv: pd.DataFrame) -> None:
        assert ix in self.ix2file, f"Piece '{self.name}' does not include a file with index {ix}."
        if parsed_tsv is None:
            file = self.ix2file[ix]
            self.logger.debug(f"I was promised the parsed DataFrame for '{file.rel_path}' but received None.")
            return
        self.ix2parsed[ix] = parsed_tsv
        self.ix2parsed_tsv[ix] = parsed_tsv
        inferred_type = infer_tsv_type(parsed_tsv)
        file = self.ix2file[ix]
        if file.type != inferred_type:
            if inferred_type == 'unknown':
                self.logger.info(f"After parsing '{file.rel_path}', the original guess that it contains '{file.type}' "
                                 f"seems to be False and I'm attributing it to the facet '{file.type}'.")
            else:
                self.logger.info(f"File {file.rel_path} turned out to contain '{inferred_type}' instead of '{file.type}', "
                             f"as I had guessed from its path.")
            self.facet2files[file.type].remove(file)
            file.type = inferred_type
            self.facet2files[inferred_type].append(file)
        self.facet2parsed[inferred_type][ix] = parsed_tsv
        if inferred_type in ('labels', 'expanded'):
            self.ix2annotations = Annotations(df = parsed_tsv)

    # def make_annotation_objects(self, view_name: Optional[str] = None, label_col='label'):
    #     selected_files = self.get_files_of_types(['labels', 'expanded'], unparsed=False, view_name=view_name)
    #     try:
    #         self._parsed_tsv[id] = df
    #         if 'label' in cols and label_col in df.columns:
    #             tsv_type = 'labels'
    #         else:
    #             tsv_type = infer_tsv_type(df)
    #
    #         if tsv_type is None:
    #             logger.debug(
    #                 f"No label column '{label_col}' was found in {self.rel_paths[key][i]} and its content could not be inferred. Columns: {df.columns.to_list()}")
    #             self._tsv_types[id] = 'other'
    #         else:
    #             self._tsv_types[id] = tsv_type
    #             if tsv_type == 'metadata':
    #                 self._metadata = pd.concat([self._metadata, self._parsed_tsv[id]])
    #                 logger.debug(f"{self.rel_paths[key][i]} parsed as metadata.")
    #             else:
    #                 self._dataframes[tsv_type][id] = self._parsed_tsv[id]
    #                 if tsv_type in ['labels', 'expanded']:
    #                     if label_col in df.columns:
    #                         logger_cfg = dict(self.logger_cfg)
    #                         logger_cfg['name'] = self.logger_names[(key, i)]
    #                         self._annotations[id] = Annotations(df=df, cols=cols, infer_types=infer_types,
    #                                                             logger_cfg=logger_cfg, level=level)
    #                         logger.debug(
    #                             f"{self.rel_paths[key][i]} parsed as annotation table and an Annotations object was created.")
    #                     else:
    #                         logger.info(
    #                             f"""The file {self.rel_paths[key][i]} was recognized to contain labels but no label column '{label_col}' was found in {df.columns.to_list()}
    #             Specify parse_tsv(key='{key}', cols={{'label'=label_column_name}}).""")
    #                 else:
    #                     logger.debug(f"{self.rel_paths[key][i]} parsed as {tsv_type} table.")
    #
    #     except Exception as e:
    #         self.logger.error(f"Parsing {self.rel_paths[key][i]} failed with the following error:\n{e}")

    def count_changed_scores(self, view_name: Optional[str]) -> int:
        parsed_scores = self.get_parsed_scores(view_name=view_name)
        return sum(score.mscx.changed for _, score in parsed_scores)

    def count_parsed(self, include_empty=False, view_name: Optional[str] = None, prefix: bool = False) -> Dict[str, int]:
        result = {}
        for typ, parsed in self.iter_facet2parsed(view_name=view_name, include_empty=include_empty):
            key = 'parsed_' + typ if prefix else typ
            result[key] = len(parsed)
        return result

    def count_detected(self, include_empty=False, view_name: Optional[str] = None, prefix: bool = False) -> Dict[str, int]:
        result = {}
        for facet, files in self.iter_facet2files(view_name=view_name, include_empty=include_empty):
            key = 'detected_' + facet if prefix else facet
            result[key] = len(files)
        return result

    def extract_facet(self,
                      facet: ScoreFacet,
                      view_name: Optional[str] = None,
                      force: bool = False,
                      choose: Literal['auto', 'ask'] = 'auto',
                      unfold: bool = False,
                      interval_index: bool = False,
                 ) -> FileDataframeTupleMaybe:
        facet = check_argument_against_literal_type(facet, ScoreFacet, logger=self.logger)
        assert facet is not None, f"Pass a valid facet {ScoreFacet.__args__}"
        assert choose != 'all', "If you want to choose='all', use _.extract_facets() (plural)."
        df_list = self.extract_facets(facets=facet,
                                      view_name=view_name,
                                      force=force,
                                      choose=choose,
                                      unfold=unfold,
                                      interval_index=interval_index,
                                      flat=True
                                      )
        if len(df_list) == 0:
            return None, None
        if len(df_list) == 1:
            return df_list[0]

    def extract_facets(self,
                       facets: ScoreFacets = None,
                       view_name: Optional[str] = None,
                       force: bool = False,
                       choose: Literal['all', 'auto', 'ask'] = 'all',
                       unfold: bool = False,
                       interval_index: bool = False,
                       flat=False) -> Union[Dict[str,  List[FileDataframeTuple]], List[FileDataframeTuple]]:
        """ Retrieve a dictionary with the selected feature matrices extracted from the parsed scores.
        If you want to retrieve parsed TSV files, use :py:meth:`get_all_parsed`.
        """
        selected_facets = resolve_facets_param(facets, ScoreFacet, logger=self.logger)
        assert selected_facets is not None, f"Pass a valid facet {ScoreFacet.__args__}"
        result = defaultdict(list)
        score_files = self.get_parsed_scores(view_name=view_name,
                                             force=force,
                                             choose=choose
                                             )
        if len(score_files) == 0:
            return [] if flat else {facet: [] for facet in selected_facets}
        for file, score_obj in score_files:
            if score_obj is None:
                self.logger.info(f"No parsed score found for '{self.name}'")
                continue
            for facet in selected_facets:
                df = getattr(score_obj.mscx, facet)(interval_index=interval_index) # unfold=unfold,
                if df is None:
                    self.logger.debug(f"Score({file.rel_path}).{facet}(unfold={unfold}, interval_index={interval_index}) returned None.")
                else:
                    result[facet].append((file, df))
        if flat:
            return sum(result.values(), [])
        else:
            result = {facet: result[facet] if facet in result else [] for facet in selected_facets}
        return result

    def get_changed_scores(self, view_name: Optional[str]) -> List[FileScoreTuple]:
        parsed_scores = self.get_parsed_scores(view_name=view_name)
        return [(file, score) for file, score in parsed_scores if score.mscx.changed]

    def get_facets(self,
                   facets: ScoreFacets = None,
                   view_name: Optional[str] = None,
                   force: bool = False,
                   choose: Literal['all', 'auto', 'ask'] = 'all',
                   unfold: bool = False,
                   interval_index: bool = False,
                   flat=False) -> Union[Dict[str, FileDataframeTuple], List[FileDataframeTuple]]:
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
        assert selected_facets is not None, f"Pass at least one valid facet {ScoreFacet.__args__}"

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

        if choose == 'all':
            extracted_facets = self.extract_facets(facets=selected_facets,
                                                   view_name=view_name,
                                                   force=force,
                                                   unfold=unfold,
                                                   interval_index=interval_index,
                                                   )
            parsed_facets = self.get_all_parsed(facets=selected_facets,
                                                view_name=view_name,
                                                force=force,
                                                unfold=unfold,
                                                interval_index=interval_index,
                                                )
            # TODO: Unfold & interval_index for parsed facets
            return make_result(extracted_facets, parsed_facets)

        # The rest below makes sure that there is only one DataFrame per facet, if available
        extracted_facets = self.extract_facets(facets=selected_facets,
                                               view_name=view_name,
                                               force=False,
                                               choose=choose,
                                               unfold=unfold,
                                               interval_index=interval_index,
                                               )
        missing_facets = [facet for facet in selected_facets if len(extracted_facets[facet]) == 0]
        if len(missing_facets) == 0:
            return make_result(extracted_facets)
        # facets missing, look for parsed TSV files
        parsed_facets = self.get_all_parsed(facets=missing_facets,
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
        parsed_facets = self.get_all_parsed(facets=missing_facets,
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
        extracted_facets = self.extract_facets(facets=selected_facets,
                                               view_name=view_name,
                                               force=True,
                                               choose=choose,
                                               unfold=unfold,
                                               interval_index=interval_index,
                                               )
        return make_result(result, extracted_facets)


    def get_facet(self,
                  facet: ScoreFacet,
                  view_name: Optional[str] = None,
                  choose: Literal['auto', 'ask'] = 'auto',
                  unfold: bool = False,
                  interval_index: bool = False) -> FileDataframeTupleMaybe:
        """Retrieve a DataFrame from a parsed score or, if unavailable, from a parsed TSV. If none have been
        parsed, first force-parse a TSV and, if not included in the given view, force-parse a score.
        """
        facet = check_argument_against_literal_type(facet, ScoreFacet, logger=self.logger)
        assert facet is not None, f"Pass a valid facet {ScoreFacet.__args__}"
        assert choose != 'all', "If you want to choose='all', use _.extract_facets() (plural)."
        df_list = self.get_facets(facets=facet,
                                  view_name=view_name,
                                  choose=choose,
                                  unfold=unfold,
                                  interval_index=interval_index,
                                  flat=True
                                  )
        if len(df_list) == 0:
            return None, None
        if len(df_list) == 1:
            return df_list[0]

    def get_file(self, facet: Facet,
                 view_name: Optional[str] = None,
                 parsed: bool = True,
                 unparsed: bool = True,
                 choose: Literal['auto', 'ask'] = 'auto',
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
        assert choose != 'all', "If you want to choose='all', use _.get_files() (plural)."
        files = self.get_files(facets=facet,
                               view_name=view_name,
                               parsed=parsed,
                               unparsed=unparsed,
                               choose=choose,
                               flat=True)
        if len(files) == 0:
            return None
        if len(files) == 1:
            return files[0]


    def get_file_from_path(self, full_path: Optional[str] = None) -> Optional[File]:
        for file in self.files:
            if file.full_path == full_path:
                return file


    def get_files(self,
                  facets: FacetArguments = None,
                  view_name: Optional[str] = None,
                  parsed: bool = True,
                  unparsed: bool = True,
                  choose: Literal['all', 'auto', 'ask'] = 'all',
                  flat: bool = False,
                  include_empty: bool = False,
                  ) -> Union[Dict[str, FileList], FileList]:
        """

        Args:
            facets:

        Returns:
            A {file_type -> [:obj:`File`] dict containing the selected Files or, if flat=True, just a list.
        """
        assert parsed + unparsed > 0, "At least one of 'parsed' and 'unparsed' needs to be True."
        selected_facets = resolve_facets_param(facets, logger=self.logger)
        if selected_facets is None:
            return
        view = self.get_view(view_name=view_name)
        selected_facets = [facet for facet in selected_facets if facet in view.selected_facets]
        if unparsed:
            facet2files = {f: self.facet2files[f] for f in selected_facets}
        else:
            # i.e., parsed must be True
            facet2files = {f: self.facet2files[f] for f in selected_facets}
            facet2files = {typ: [f for f in files if f.ix in self.ix2parsed] for typ, files in facet2files.items()}
        if not parsed:
            # i.e., unparsed must be True
            facet2files = {typ: [f for f in files if f.ix not in self.ix2parsed] for typ, files in facet2files.items()}
        facet2files = {typ: view.filtered_file_list(files) for typ, files in facet2files.items()}
        result = {}
        needs_choice = []
        for facet, files in facet2files.items():
            n_files = len(files)
            if n_files == 0 and not include_empty:
                continue
            elif choose == 'all' or n_files < 2:
                result[facet] = files
            else:
                selected = files2disambiguation_dict(files, logger=self.logger)
                needs_choice.append(facet)
                result[facet] = selected
        if choose == 'auto':
            for typ in needs_choice:
                result[typ] = [automatically_choose_from_disambiguated_files(result[typ], self.name, typ)]
        elif choose == 'ask':
            for typ in needs_choice:
                choices = result[typ]
                selected = ask_user_to_choose_from_disambiguated_files(choices, self.name, typ)
                if selected is None:
                    if include_empty:
                        result[typ] = []
                else:
                    result[typ] = [selected]
        elif choose == 'all' and 'scores' in needs_choice:
            # check if any scores can be differentiated solely by means of their file extension
            several_score_files = result['scores'].values()
            subdir_fnames = [(file.subdir, file.fname) for file in several_score_files]
            if len(set(subdir_fnames)) < len(subdir_fnames):
                duplicates = {tup: [] for tup, cnt in Counter(subdir_fnames).items() if cnt > 1}
                for file in several_score_files:
                    if (file.subdir, file.fname) in duplicates:
                        duplicates[(file.subdir, file.fname)].append(file.rel_path)
                display_duplicates = '\n'.join(str(sorted(files)) for files in duplicates.values())
                self.logger.warning(f"The following scores are lying in the same subfolder and have the same name:\n{display_duplicates}.\n"
                                    f"TSV files extracted from them will be overwriting each other. Consider excluding certain "
                                    f"file extensions or letting me choose='auto'.")
        if flat:
            return sum(result.values(), start=[])
        return result

    def _get_parsed_at_index(self,
                             ix: int) -> ParsedFile:
        assert ix in self.ix2file, f"Piece '{self.name}' does not include a file with index {ix}."
        if ix not in self.ix2parsed:
            self._parse_file_at_index(ix)
        if ix not in self.ix2parsed:
            file = self.ix2file[ix]
            raise RuntimeError(f"Unable to parse '{file.rel_path}'.")
        return self.ix2parsed[ix]

    def get_parsed(self,
                   facet: Facet,
                   view_name: Optional[str] = None,
                   choose: Literal['auto', 'ask'] = 'auto') -> FileParsedTupleMaybe:
        facet = check_argument_against_literal_type(facet, Facet, logger=self.logger)
        assert facet is not None, f"Pass a valid facet {Facet.__args__}"
        assert choose != 'all', "If you want to choose='all', use _.get_all_parsed()."
        files = self.get_all_parsed(facets=facet,
                                    view_name=view_name,
                                    choose=choose,
                                    flat=True)
        if len(files) == 0:
            unparsed_file = self.get_file(facet, view_name=view_name, parsed=False, choose=choose)
            if unparsed_file is not None:
                return unparsed_file, self._get_parsed_at_index(unparsed_file.ix)
            else:
                return None, None
        if len(files) == 1:
            return files[0]


    def get_all_parsed(self,
                       facets: FacetArguments = None,
                       view_name: Optional[str] = None,
                       force: bool = False,
                       choose: Literal['all', 'auto', 'ask'] = 'all',
                       flat: bool = False,
                       include_empty: bool = False,
                       unfold: bool = False,
                       interval_index: bool = False,
                       ) -> Union[Dict[Facet, List[FileParsedTuple]], List[FileParsedTuple]]:
        """Return multiple parsed files."""
        if unfold + interval_index > 0:
            raise NotImplementedError(f"Unfolding and interval index currently only available for extracted facets.")
        selected_facets = resolve_facets_param(facets, logger=self.logger)
        if selected_facets is None:
            return [] if flat else {}
        facet2files = self.get_files(selected_facets, view_name=view_name, choose=choose, include_empty=include_empty)
        result = {}
        for facet, files in facet2files.items():
            unparsed_ixs = [file.ix for file in files if file.ix not in self.ix2parsed]
            n_unparsed = len(unparsed_ixs)
            if force:
                if len(unparsed_ixs) > 0:
                    for ix in unparsed_ixs:
                        self._parse_file_at_index(ix)
            elif n_unparsed > 0:
                plural = 'files' if n_unparsed > 1 else 'facet'
                self.logger.debug(f"Disregarded {n_unparsed} unparsed {facet} {plural}. Set force=True to automatically parse.")
            parsed_files = [(file, self.ix2parsed[file.ix]) for file in files if file.ix in self.ix2parsed]
            n_parsed = len(parsed_files)
            if n_parsed == 0 and not include_empty:
                continue
            elif choose == 'all' or n_parsed < 2:
                # no disambiguation required
                result[facet] = parsed_files
            else:
                selected = disambiguate_parsed_files(parsed_files,
                                                     self.name,
                                                     facet,
                                                     choose=choose,
                                                     logger=self.logger
                                                     )
                if selected is None:
                    if include_empty:
                        result[facet] = []
                else:
                    result[facet] = [selected]
        if flat:
            return sum(result.values(), start=[])
        return result




    def get_parsed_score(self,
                         view_name: Optional[str] = None,
                         choose: Literal['auto', 'ask'] = 'auto') -> FileScoreTupleMaybe:
        return self.get_parsed('scores', view_name=view_name, choose=choose)

    def get_parsed_scores(self,
                         view_name: Optional[str] = None,
                         force: bool = False,
                         choose: Literal['all', 'auto', 'ask'] = 'all') -> List[FileScoreTuple]:
        return self.get_all_parsed('scores', view_name=view_name, force=force, choose=choose, flat=True)

    def get_parsed_tsv(self,
                       facet: TSVtype,
                       view_name: Optional[str] = None,
                       choose: Literal['auto', 'ask'] = 'auto',
                       ) -> FileDataframeTupleMaybe:
        facets = argument_and_literal_type2list(facet, TSVtype, logger=self.logger)
        assert len(facet) == 1, f"Pass exactly one valid TSV type {TSVtype.__args__} or use _.get_parsed_tsvs()"
        facet = facets[0]
        return self.get_parsed(facet, view_name=view_name, choose=choose)

    def get_parsed_tsvs(self,
                        facets: TSVtypes,
                        view_name: Optional[str] = None,
                        force: bool = False,
                        choose: Literal['all', 'auto', 'ask'] = 'all',
                        ) -> List[FileDataframeTupleMaybe]:
        facets = argument_and_literal_type2list(facets, TSVtype, logger=self.logger)
        return self.get_all_parsed(facets, view_name=view_name, force=force, choose=choose, flat=True)

    def _get_parsed_score_files(self, view_name: Optional[str] = None) -> FileList:
        return self.get_files('scores', view_name=view_name, unparsed=False, flat=True)

    def _get_parsed_tsv_files(self, view_name: Optional[str] = None, flat: bool = True) -> Union[FileDict, FileList]:
        return self.get_files('tsv', view_name=view_name, unparsed=False, flat=flat)

    def _get_unparsed_score_files(self, view_name: Optional[str] = None) -> FileList:
        return self.get_files('scores', view_name=view_name, parsed=False, flat=True)

    def _get_unparsed_tsv_files(self, view_name: Optional[str] = None, flat: bool = True) -> Union[FileDict, FileList]:
        return self.get_files('tsv', view_name=view_name, parsed=False, flat=flat)

    def info(self, return_str=False, view_name=None, show_discarded: bool = False):
        header = f"Piece '{self.name}'"
        header += "\n" + "-" * len(header) + "\n"

        # start info message with the names of the available views, the header, and info on the active view.
        view = self.get_view(view_name)
        view.reset_filtering_data()
        msg = available_views2str(self._views, view_name)
        msg += header
        msg += f"View: {view}\n\n"

        # Show info on all files included in the active view
        facet2files = dict(self.iter_facet2files(view_name=view_name, include_empty=False))
        if len(facet2files) == 0:
            msg += "No files selected."
        else:
            files_df = pd.concat([pd.DataFrame(files).set_index('ix') for files in facet2files.values()],
                                 keys=facet2files.keys(),
                                 names=['facet', 'ix'])
            if len(files_df) == 0:
                msg += "No files selected."
            else:
                is_parsed, has_changed = [], []
                for facet, ix in files_df.index:
                    parsed = ix in self.ix2parsed
                    is_parsed.append(parsed)
                    changed_score = parsed and facet == 'scores' and self[ix].mscx.changed
                    has_changed.append(changed_score)
                files_df['is_parsed'] = is_parsed
                if any(has_changed):
                    files_df['has_changed'] = has_changed
                    info_columns = ['rel_path', 'is_parsed', 'has_changed']
                else:
                    info_columns = ['rel_path', 'is_parsed']
                msg += files_df[info_columns].to_string()
        msg += '\n\n' + view.filtering_report(show_discarded=show_discarded, return_str=True)
        if return_str:
            return msg
        print(msg)

    def iter_extracted_facet(self,
                       facet: ScoreFacet,
                       view_name: Optional[str] = None,
                       force: bool = False,
                       unfold: bool = False,
                       interval_index: bool = False) -> Iterator[FileDataframeTupleMaybe]:
        """ Iterate through the selected facet extracted from all parsed or yet-to-parse scores.
        """
        facet = check_argument_against_literal_type(facet, ScoreFacet, logger=self.logger)
        assert facet is not None, f"Pass a valid facet {ScoreFacet.__args__}"
        for file in self.iter_files('scores',
                                    view_name=view_name,
                                    flat=True,
                                    ):
            if file.ix not in self.ix2parsed and not force:
                continue
            score_obj = self._get_parsed_at_index(file.ix)
            if score_obj is None:
                self.logger.info(f"No parsed score found for '{file.rel_path}'")
                continue
            df = getattr(score_obj.mscx, facet)(interval_index=interval_index) # unfold=unfold,
            if df is None:
                self.logger.debug(f"Score({file.rel_path}).{facet}(unfold={unfold}, interval_index={interval_index}) returned None.")
                continue
            yield file, df

    def iter_extracted_facets(self,
                              facets: ScoreFacets,
                              view_name: Optional[str] = None,
                              force: bool = False,
                              unfold: bool = False,
                              interval_index: bool = False) -> Iterator[Tuple[File, DataframeDict]]:
        """ Iterate through the selected facets extracted from all parsed or yet-to-parse scores.
        """
        selected_facets = resolve_facets_param(facets, ScoreFacet, logger=self.logger)
        assert selected_facets is not None, f"Pass at least one valid facet {ScoreFacet.__args__}"
        facet2dataframe = {}
        for file in self.iter_files('scores',
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
                df = getattr(score_obj.mscx, facet)(interval_index=interval_index) # unfold=unfold,
                if df is None:
                    self.logger.debug(f"Score({file.rel_path}).{facet}(unfold={unfold}, interval_index={interval_index}) returned None.")
                facet2dataframe[facet] = df
            yield file, facet2dataframe


    def iter_facet2files(self, view_name: Optional[str] = None, include_empty: bool = False) -> Iterator[Tuple[str, FileList]]:
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


    def iter_facet2parsed(self, view_name: Optional[str] = None, include_empty: bool = False) -> Iterator[Dict[str, FileList]]:
        """Iterating through :attr:`facet2parsed` under the current or specified view and selecting only parsed files."""
        view = self.get_view(view_name=view_name)
        for facet, ix2parsed in self.facet2parsed.items():
            if facet not in view.selected_facets:
                continue
            files = [self.ix2file[ix] for ix in ix2parsed.keys()]
            filtered_ixs = [file.ix for file in view.filtered_file_list(files, 'parsed')]
            # the files need to be filtered even if the facet is excluded, for counting excluded files
            if len(filtered_ixs) == 0 and not include_empty:
                continue
            yield facet, {ix: ix2parsed[ix] for ix in filtered_ixs}


    def iter_files(self,
                  facets: FacetArguments = None,
                  view_name: Optional[str] = None,
                  parsed: bool = True,
                  unparsed: bool = True,
                  choose: Literal['all', 'auto', 'ask'] = 'all',
                  flat: bool = False,
                  include_empty: bool = False,
                  ) -> Union[Iterator[FileDict], Iterator[FileList]]:
        """Equivalent to iterating through the result of :meth:`get_files`."""
        selected_files = self.get_files(facets=facets,
                                        view_name=view_name,
                                        parsed=parsed,
                                        unparsed=unparsed,
                                        choose=choose,
                                        flat=flat,
                                        include_empty=include_empty)
        if flat:
            yield from selected_files
        else:
            yield from selected_files.items()

    def iter_parsed(self,
                       facet: Facet = None,
                       view_name: Optional[str] = None,
                       force: bool = False,
                       choose: Literal['all', 'auto', 'ask'] = 'all',
                       include_empty: bool = False,
                       unfold: bool = False,
                       interval_index: bool = False,
                       ) -> Iterator[FileParsedTuple]:
        facet = check_argument_against_literal_type(facet, Facet, logger=self.logger)
        assert facet is not None, f"Pass a valid facet {Facet.__args__}"
        files = self.get_all_parsed(facets=facet,
                                         view_name=view_name,
                                         force=force,
                                         choose=choose,
                                         flat=True,
                                         include_empty=include_empty,
                                         unfold=unfold,
                                         interval_index=interval_index)
        yield from files

    def _parse_file_at_index(self, ix: int) -> None:
        assert ix in self.ix2file, f"Piece '{self.name}' does not include a file with index {ix}."
        file = self.ix2file[ix]
        if file.type == 'scores':
            logger_cfg = dict(self.logger_cfg)
            score = Score(file.full_path, ms=self.ms, **logger_cfg)
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



    def register_file(self, file: File, reject_incongruent_fnames: bool = True) -> bool:
        ix = file.ix
        if ix in self.ix2file:
            existing_file = self.ix2file[ix]
            if file.full_path == existing_file.full_path:
                self.logger.debug(f"File '{file.rel_path}' was already registered for {self.name}.")
                return None
            else:
                self.logger.debug(f"File '{existing_file.rel_path}' replaced with '{file.rel_path}'")
        if file.fname != self.name:
            if file.fname.startswith(self.name):
                file.suffix = file.fname[len(self.name):]
                self.logger.debug(f"Recognized suffix '{file.suffix}' for {file.file}.")
            elif reject_incongruent_fnames:
                if self.name in file.fname:
                    self.logger.info(f"{file.file} seems to come with a prefix w.r.t. '{self.name}' and is ignored.")
                    return False
                else:
                    self.logger.warning(f"{file.file} does not contain '{self.name}' and is ignored.")
                    return False
        self.facet2files[file.type].append(file)
        self.ix2file[file.ix] = file
        return True
    #
    # def store_changed_scores(self,
    #                         view_name: Optional[str] = None,
    #                         root_dir: Optional[str] = None,
    #                         folder: str = '.',
    #                         suffix: str = '',
    #                         overwrite: bool = False,
    #                         simulate=False) -> List[str]:
    #     stored_file_paths = []
    #     for file, _ in self.get_changed_scores(view_name):
    #         file_path = self.store_parsed_score_at_ix(ix=file.ix,
    #                                                   root_dir=root_dir,
    #                                                   folder=folder,
    #                                                   suffix=suffix,
    #                                                   overwrite=overwrite,
    #                                                   simulate=simulate)
    #         stored_file_paths.append(file_path)
    #     return stored_file_paths

    def store_extracted_facet(self,
                              facet: ScoreFacet,
                              root_dir: Optional[str] = None,
                              folder: Optional[str] = None,
                              suffix: str = '',
                              view_name: Optional[str] = None,
                              force: bool = False,
                              choose: Literal['all', 'auto', 'ask'] = 'all',
                              unfold: bool = False,
                              interval_index: bool = False
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
                * If ``folder`` is a relative path starting with a dot ``.`` the relative path is appended to the file's subdir.
                  For example, ``..\notes`` will resolve to a sibling directory of the one where the ``file`` is located.
                * If ``folder`` is a relative path that does not begin with a dot ``.``, it will be appended to the
                  ``root_dir``.
            suffix: String to append to the file's fname.
            view_name:
            force:
            choose:
            unfold:
            interval_index:

        Returns:

        """
        if choose == 'all':
            extracted_facet = self.iter_extracted_facet(facet=facet,
                                                        view_name=view_name,
                                                        force=force,
                                                        unfold=unfold,
                                                        interval_index=interval_index)
        else:
            extracted_facet = self.extract_facets(facets=facet,
                                                   view_name=view_name,
                                                   force=force,
                                                   choose=choose,
                                                   flat=True)
        for file, df in extracted_facet:
            file_path = make_file_path(file=file,
                                  root_dir=root_dir,
                                  folder=folder,
                                  suffix=suffix)
            write_tsv(df, file_path, logger=self.logger)

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

    def store_parsed_score_at_ix(self,
                                 ix,
                                 root_dir: Optional[str] = None,
                                 folder: str = '.',
                                 suffix: str = '',
                                 overwrite: bool = False,
                                 simulate=False) -> Optional[str]:
        """
        Creates a MuseScore 3 file from the Score object at the given index.

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
            logger.error(f"No Score object found. Call parse_scores() first.")
            return
        file = self.ix2file[ix]
        file_path = make_file_path(file=file,
                                   root_dir=root_dir,
                                   folder=folder,
                                   suffix=suffix,
                                   fext='.mscx')
        if os.path.isfile(file_path):
            if simulate:
                if overwrite:
                    self.logger.warning(f"Would have overwritten {file_path}.")
                    return
                self.logger.warning(f"Would have skipped {file_path}.")
                return
            elif not overwrite:
                self.logger.warning(f"Skipping {file.rel_path} because the target file exists already and overwrite=False: {file_path}.")
                return
        if simulate:
            self.logger.debug(f"Would have written score to {file_path}.")
        else:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self[ix].store_score(file_path)
            self.logger.debug(f"Score written to {file_path}.")
        return file_path

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
            del (self._views[new_name])
        if show_info:
            self.info()

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
        selected_facets = resolve_facets_param(facets, ScoreFacet, logger=self.logger)
        assert selected_facets is not None, f"Pass a valid facet {ScoreFacet.__args__}"
        assert choose != 'all', "This method does not accept choose='all'."
        written_paths = []
        parsed_tsvs = self.get_all_parsed(selected_facets,
                                          view_name=view_name,
                                          force=force,
                                          choose=choose)
        if len(parsed_tsvs) == 0:
            self.logger.info(f"No parsed TSV files to update.")
            return []
        score_file, score_obj = self.get_parsed_score(view_name=view_name,
                                             choose=choose
                                             )
        if score_obj is None:
            self.logger.info(f"This method updates TSV files based on a score but the current view includes none.")
            return []
        for facet, file_df_tuples in parsed_tsvs.items():
            if len(file_df_tuples) > 1:
                self.logger.warning(f"There are more than one TSV files containing {facet} and they will be compared with "
                                    f"the same score.")
            score_file, new_df = self.extract_facet(facet,
                                                    view_name=view_name,
                                                    choose=choose)
            for tsv_file, old_df in file_df_tuples:
                try:
                    # missing_columns = [c for c in old_df.columns if c not in new_df.columns]
                    # if len(missing_columns) > 0:
                    #     plural = 'These columns are' if len(missing_columns) > 1 else 'This column is'
                    #     self.logger.warning(f"{plural} missing in the updated {facet}:\n{old_df[missing_columns]}")
                    # tmp_new = new_df[[c for c in old_df.columns if c in new_df.columns]]
                    # assert_frame_equal(old_df, tmp_new, check_dtype=False, obj=facet) # from pandas.testing import assert_frame_equal
                    assert_dfs_equal(old_df, new_df)
                    # TODO: make utils.assert_dfs_equal() use math.isclose() for comparing floats
                except AssertionError as e:
                    if facet == 'expanded':
                        facet += ' labels'
                    self.logger.warning(f"The {facet} extracted from {score_file.rel_path} is not identical with the "
                                        f"ones in {tsv_file.rel_path}:\n{e}")
                    continue
                write_tsv(new_df, tsv_file.full_path, logger=self.logger)
                written_paths.append(tsv_file.full_path)
        return written_paths


