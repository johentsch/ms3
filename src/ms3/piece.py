from collections import defaultdict, Counter
from typing import Dict, Literal, Union, Iterator, Optional, overload, List

import pandas as pd

from .annotations import Annotations
from ._typing import FileList, ParsedFile, FileDict, Facet, TSVtype, Facets, ScoreFacets, ScoreFacet, FileParsedTuple, FacetArguments, FileScoreTuple, \
    FileDataframeTuple
from .utils import File, infer_tsv_type, automatically_choose_from_disambiguated_files, ask_user_to_choose_from_disambiguated_files, \
    files2disambiguation_dict, get_musescore, load_tsv, metadata2series, pretty_dict, treat_facets_argument, \
    disambiguate_parsed_files, available_views2str, argument_and_literal_type2list
from .score import Score
from .logger import LoggedClass
from .view import View, DefaultView


class Piece(LoggedClass):
    """Wrapper around :class:`~.score.Score` for associating it with parsed TSV files"""

    def __init__(self, fname: str, view: View = None, logger_cfg={}, ms=None):
        super().__init__(subclass='Piece', logger_cfg=logger_cfg)
        self.piece_name = fname
        available_types = ('scores',) + Score.dataframe_types
        self.facet2files: Dict[str, FileList] = defaultdict(list)
        """{typ -> [:obj:`File`]} dict storing file information for associated types.
        """
        self.facet2files.update({typ: [] for typ in available_types})
        self.files: Dict[int, File] = defaultdict(list)
        """{ix -> :obj:`File`} dict storing the registered file information for access via index.
        """
        self.type2parsed: Dict[str, Dict[int, ParsedFile]] = defaultdict(dict)
        """{typ -> {ix -> :obj:`pandas.DataFrame`|:obj:`Score`}} dict storing parsed files for associated types.
        """
        self.type2parsed.update({typ: {} for typ in available_types})
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
        self._views[None] = DefaultView() if view is None else view
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
        file, score = self.get_parsed_score()
        if score is None:
            return None
        meta_dict = score.mscx.metadata
        meta_dict['subdirectory'] = file.subdir
        meta_dict['fname'] = self.piece_name
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
            active_name = active.name
            if active_name in self._views:
                self.switch_view(active_name, show_info=False)
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
    def views(self):
        print(pretty_dict({"[active]" if k is None else k: v for k, v in self._views.items()}, "view_name", "Description"))

    def __getattr__(self, view_name):
        if view_name in self._views:
            self.switch_view(view_name, show_info=False)
            return self
        elif view_name is not None and self._views[None].name == view_name:
            return self
        else:
            raise AttributeError(f"'{view_name}' is not an existing view. Use _.get_view('{view_name}') to create it.")

    def __getitem__(self, ix) -> ParsedFile:
        return self._get_parsed_at_index(ix)


    def __repr__(self):
        return self.info(return_str=True)
    def add_parsed_score(self, ix: int, score_obj: Score) -> None:
        assert ix in self.files, f"Piece '{self.piece_name}' does not include a file with index {ix}."
        if score_obj is None:
            file = self.files[ix]
            self.logger.debug(f"I was promised the parsed score for '{file.rel_path}' but received None.")
            return
        self.ix2parsed[ix] = score_obj
        self.ix2parsed_score[ix] = score_obj
        self.type2parsed['scores'][ix] = score_obj

    def add_parsed_tsv(self, ix: int, parsed_tsv: pd.DataFrame) -> None:
        assert ix in self.files, f"Piece '{self.piece_name}' does not include a file with index {ix}."
        if parsed_tsv is None:
            file = self.files[ix]
            self.logger.debug(f"I was promised the parsed DataFrame for '{file.rel_path}' but received None.")
            return
        self.ix2parsed[ix] = parsed_tsv
        self.ix2parsed_tsv[ix] = parsed_tsv
        inferred_type = infer_tsv_type(parsed_tsv)
        file = self.files[ix]
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
        self.type2parsed[inferred_type][ix] = parsed_tsv
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

    def count_parsed(self, include_empty=False, view_name: Optional[str] = None) -> Dict[str, int]:
       return {"parsed_" + typ: len(parsed) for typ, parsed in self.iter_type2parsed_files(view_name=view_name, include_empty=include_empty)}

    def count_types(self, include_empty=False, view_name: Optional[str] = None) -> Dict[str, int]:
        return {"found_" + typ: len(files) for typ, files in self.iter_facet2files(view_name=view_name, include_empty=include_empty)}

    def extract_facet(self,
                      facet: ScoreFacet,
                      view_name: Optional[str] = None,
                      force: bool = False,
                      choose: Literal['auto', 'ask'] = 'auto',
                      unfold: bool = False,
                      interval_index: bool = False,
                 ) -> Optional[FileDataframeTuple]:
        if isinstance(facet, list):
            facet = tuple(facet)
        facets = argument_and_literal_type2list(facet, ScoreFacet)
        assert facets is not None, f"Pass a valid facet {ScoreFacet.__args__}"
        assert len(facets) == 1, "Request one facet at a time or use _.extract_facets"
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
            return None
        if len(df_list) == 1:
            return df_list[0]

    def extract_facets(self,
                       facets: ScoreFacets = None,
                       view_name: Optional[str] = None,
                       force: bool = False,
                       choose: Literal['all', 'auto', 'ask'] = 'all',
                       unfold: bool = False,
                       interval_index: bool = False,
                       flat=False) -> Union[Dict[str, FileDataframeTuple], List[FileDataframeTuple]]:
        """ Retrieve a dictionary with the selected feature matrices extracted from the parsed scores.
        If you want to retrieve parsed TSV files, use :py:meth:`get_all_parsed`.
        """
        selected_facets = treat_facets_argument(facets, ScoreFacet, logger=self.logger)
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
                self.logger.info(f"No parsed score found for '{self.piece_name}'")
                continue
            for facet in selected_facets:
                df = getattr(score_obj.mscx, facet)(interval_index=interval_index)
                if unfold:
                    raise NotImplementedError(f"Unfolding is currently not available.")
                if df is None:
                    self.logger.debug(f"{file.rel_path} doesn't contain {facet}.")
                else:
                    result[facet].append((file, df))
        if flat:
            return sum(result.values(), [])
        else:
            result = {facet: result[facet] if facet in result else [] for facet in selected_facets}
        return result



    def get_facets(self,
                   facets: ScoreFacets = None,
                   view_name: Optional[str] = None,
                   force: bool = False,
                   choose: Literal['all', 'auto', 'ask'] = 'all',
                   unfold: bool = False,
                   interval_index: bool = False,
                   flat=False) -> Union[Dict[str, FileDataframeTuple], List[FileDataframeTuple]]:
        selected_facets = treat_facets_argument(facets, ScoreFacet, logger=self.logger)
        assert selected_facets is not None, f"Pass a valid facet {ScoreFacet.__args__}"

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
        # facets missing, looked for parsed TSV files
        parsed_facets = self.get_all_parsed(facets=missing_facets,
                                        view_name=view_name,
                                        force=False,
                                        choose=choose,
                                        include_empty=True
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
                                            include_empty=True
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
                  force: bool = False,
                  choose: Literal['auto', 'ask'] = 'auto',
                  unfold: bool = False,
                  interval_index: bool = False,
                  ) -> Optional[FileDataframeTuple]:
        if isinstance(facet, list):
            facet = tuple(facet)
        facets = argument_and_literal_type2list(facet, ScoreFacet)
        assert facets is not None, f"Pass a valid facet {ScoreFacet.__args__}"
        assert len(facets) == 1, "Request one facet at a time or use _.extract_facets"
        assert choose != 'all', "If you want to choose='all', use _.extract_facets() (plural)."
        df_list = self.get_facets(facets=facet,
                                  view_name=view_name,
                                  force=force,
                                  choose=choose,
                                  unfold=unfold,
                                  interval_index=interval_index,
                                  flat=True
                                  )
        if len(df_list) == 0:
            return None
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
        if isinstance(facet, list):
            facet = tuple(facet)
        facets = argument_and_literal_type2list(facet, Facet)
        assert facets is not None, f"Pass a valid facet {Facet.__args__}"
        assert len(facets) == 1, "Request one facet at a time or use _.get_files"
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


    def get_files(self,
                  facets: FacetArguments = None,
                  view_name: Optional[str] = None,
                  parsed: bool = True,
                  unparsed: bool = True,
                  choose: Literal['all', 'auto', 'ask'] = 'all',
                  flat: bool = False,
                  include_empty: bool = False,
                  ) -> Union[FileDict, FileList]:
        """

        Args:
            facets:

        Returns:
            A {file_type -> [:obj:`File`] dict containing the selected Files or, if flat=True, just a list.
        """
        assert parsed + unparsed > 0, "At least one of 'parsed' and 'unparsed' needs to be True."
        selected_facets = treat_facets_argument(facets, logger=self.logger)
        if selected_facets is None:
            return
        if unparsed:
            facet2files = {f: self.facet2files[f] for f in selected_facets}
        else:
            # i.e., parsed must be True
            facet2files = {f: self.facet2files[f] for f in selected_facets}
            facet2files = {typ: [f for f in files if f.ix in self.ix2parsed] for typ, files in facet2files.items()}
        if not parsed:
            # i.e., unparsed must be True
            facet2files = {typ: [f for f in files if f.ix not in self.ix2parsed] for typ, files in facet2files.items()}
        view = self.get_view(view_name=view_name)
        facet2files = {typ: view.filtered_file_list(files, 'files') for typ, files in facet2files.items()}
        result = {}
        needs_choice = []
        for facet, files in facet2files.items():
            n_files = len(files)
            if n_files == 0 and not include_empty:
                continue
            elif choose == 'all' or n_files < 2:
                result[facet] = files
            else:
                disambiguated = files2disambiguation_dict(files, logger=self.logger)
                needs_choice.append(facet)
                result[facet] = disambiguated
        if choose == 'auto':
            for typ in needs_choice:
                result[typ] = [automatically_choose_from_disambiguated_files(result[typ], self.piece_name, typ)]
        elif choose == 'ask':
            for typ in needs_choice:
                choices = result[typ]
                result[typ] = [ask_user_to_choose_from_disambiguated_files(choices, self.piece_name, typ)]
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
                self.logger.warning(f"The following scores lie in the same subfolder and have the same name:\n{display_duplicates}.\n"
                                    f"TSV files extracted from them will be overwriting each other. Consider excluding certain "
                                    f"file extensions or letting me choose='auto'.")
        if flat:
            return sum(result.values(), start=[])
        return result

    def _get_parsed_at_index(self,
                             ix: int) -> ParsedFile:
        assert ix in self.files, f"Piece '{self.piece_name}' does not include a file with index {ix}."
        if ix not in self.ix2parsed:
            self._parse_file_at_index(ix)
        if ix not in self.ix2parsed:
            file = self.files[ix]
            raise RuntimeError(f"Unable to parse '{file.rel_path}'.")
        return self.ix2parsed[ix]

    def get_parsed(self,
                   facet: Facet,
                   view_name: Optional[str] = None,
                   choose: Literal['auto', 'ask'] = 'auto') -> FileParsedTuple:
        if isinstance(facet, list):
            facet = tuple(facet)
        facets = argument_and_literal_type2list(facet, Facet)
        assert facets is not None, f"Pass a valid facet {Facet.__args__}"
        assert len(facets) == 1, "Request one facet at a time or use _.get_all_parsed"
        assert choose != 'all', "If you want to choose='all', use _.get_all_parsed()."
        files = self.get_all_parsed(facets=facet,
                                    view_name=view_name,
                                    choose=choose,
                                    flat=True)
        if len(files) == 0:
            unparsed_file = self.get_file(facet, view_name=view_name, parsed=False, choose=choose)
            return unparsed_file, self._get_parsed_at_index(unparsed_file.ix)
        if len(files) == 1:
            return files[0]


    def get_all_parsed(self,
                       facets: FacetArguments = None,
                       view_name: Optional[str] = None,
                       force: bool = False,
                       choose: Literal['all', 'auto', 'ask'] = 'all',
                       flat: bool = False,
                       include_empty: bool = False
                       ) -> Union[Dict[str, FileParsedTuple], List[FileParsedTuple]]:
        """Return multiple parsed files."""
        selected_facets = treat_facets_argument(facets, logger=self.logger)
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
                                                     self.piece_name,
                                                     facet,
                                                     choose=choose,
                                                     logger=self.logger
                                                     )
                result[facet] = [selected]
        if flat:
            return sum(result.values(), start=[])
        return result

    # def get_all_parsed_of_type(self,
    #                             facet: Facet = None,
    #                             view_name: Optional[str] = None,
    #                             force: bool = False,
    #                             choose: Literal['all', 'auto', 'ask'] = 'all',
    #                             flat: bool = False
    #                             ) -> Union[Dict[str, FileParsedTuple], List[FileParsedTuple]]:
    #     """Alias for :meth:`get_all_parsed_of_types`"""
    #     return self.get_all_parsed(
    #         facets=facet,
    #         view_name=view_name,
    #         force=force,
    #         choose=choose,
    #         flat=flat
    #     )



    def get_parsed_score(self,
                         view_name: Optional[str] = None,
                         choose: Literal['auto', 'ask'] = 'auto') -> FileScoreTuple:
        return self.get_parsed('scores', view_name=view_name, choose=choose)

    def get_parsed_scores(self,
                         view_name: Optional[str] = None,
                         force: bool = False,
                         choose: Literal['all', 'auto', 'ask'] = 'all') -> List[FileScoreTuple]:
        return self.get_all_parsed('scores', view_name=view_name, force=force, choose=choose, flat=True)

    def get_parsed_tsv(self,
                       file_type: TSVtype,
                       view_name: Optional[str] = None,
                       choose: Literal['auto', 'ask'] = 'auto',
         ) -> FileDataframeTuple:
        return self.get_parsed(file_type, view_name=view_name, choose=choose)

    def get_parsed_tsvs(self,
                       file_type: TSVtype,
                       view_name: Optional[str] = None,
                       force: bool = False,
                       choose: Literal['all', 'auto', 'ask'] = 'all',
         ) -> List[FileDataframeTuple]:
        return self.get_all_parsed(file_type, view_name=view_name, force=force, choose=choose, flat=True)

    def _get_parsed_score_files(self, view_name: Optional[str] = None) -> FileList:
        return self.get_files('scores', view_name=view_name, unparsed=False, flat=True)

    def _get_parsed_tsv_files(self, view_name: Optional[str] = None, flat: bool = True) -> Union[FileDict, FileList]:
        return self.get_files('tsv', view_name=view_name, unparsed=False, flat=flat)

    def _get_unparsed_score_files(self, view_name: Optional[str] = None) -> FileList:
        return self.get_files('scores', view_name=view_name, parsed=False, flat=True)

    def _get_unparsed_tsv_files(self, view_name: Optional[str] = None, flat: bool = True) -> Union[FileDict, FileList]:
        return self.get_files('tsv', view_name=view_name, parsed=False, flat=flat)

    def info(self, return_str=False, view_name=None, show_discarded: bool = False):
        header = f"Piece '{self.piece_name}'"
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
                ixs = files_df.index.get_level_values(1)
                is_parsed = [str(ix in self.ix2parsed) for ix in ixs]
                files_df['is_parsed'] = is_parsed
                msg += files_df[['rel_path', 'is_parsed']].to_string()
        msg += '\n\n' + view.filtering_report(show_discarded=show_discarded)
        if return_str:
            return msg
        print(msg)


    def iter_facet2files(self, view_name: Optional[str] = None, include_empty: bool = False) -> Iterator[Dict[str, FileList]]:
        """Iterating through _.facet2files.items() under the current or specified view."""
        view = self.get_view(view_name=view_name)
        for typ, files in self.facet2files.items():
            filtered_files = view.filtered_file_list(files, 'files')
            # the files need to be filtered even if the facet is excluded, for counting excluded files
            if len(filtered_files) == 0 and not include_empty:
                continue
            if typ in view.selected_facets:
                yield typ, filtered_files


    def iter_type2parsed_files(self, view_name: Optional[str] = None, include_empty: bool = False) -> Iterator[Dict[str, FileList]]:
        """Iterating through _.facet2files.items() under the current or specified view and selecting only parsed files."""
        view = self.get_view(view_name=view_name)
        for typ, ix2parsed in self.type2parsed.items():
            files = [self.files[ix] for ix in ix2parsed.keys()]
            filtered_ixs = [file.ix for file in view.filtered_file_list(files, 'parsed')]
            # the files need to be filtered even if the facet is excluded, for counting excluded files
            if len(filtered_ixs) == 0 and not include_empty:
                continue
            if typ in view.selected_facets:
                yield typ, {ix: ix2parsed[ix] for ix in filtered_ixs}


    def _parse_file_at_index(self, ix: int) -> None:
        assert ix in self.files, f"Piece '{self.piece_name}' does not include a file with index {ix}."
        file = self.files[ix]
        if file.type == 'scores':
            logger_cfg = dict(self.logger_cfg)
            score = Score(file.full_path, logger_cfg=logger_cfg, ms=self.ms)
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
        if ix in self.files:
            existing_file = self.files[ix]
            if file.full_path == existing_file.full_path:
                self.logger.debug(f"File '{file.rel_path}' was already registered for {self.piece_name}.")
                return
            else:
                self.logger.debug(f"File '{existing_file.rel_path}' replaced with '{file.rel_path}'")
        if file.fname != self.piece_name:
            if file.fname.startswith(self.piece_name):
                file.suffix = file.fname[len(self.piece_name):]
                self.logger.debug(f"Recognized suffix '{file.suffix}' for {file.file}.")
            elif reject_incongruent_fnames:
                if self.piece_name in file.fname:
                    self.logger.info(f"{file.file} seems to come with a prefix w.r.t. '{self.piece_name}' and is ignored.")
                    return False
                else:
                    self.logger.warning(f"{file.file} does not contain '{self.piece_name}' and is ignored.")
                    return False
        self.facet2files[file.type].append(file)
        self.files[file.ix] = file
        return True

    def switch_view(self, view_name: Optional[str],
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



