import os
from collections import defaultdict, Counter
from typing import Dict, Collection, Literal, Tuple, Union, List, Iterator, TypeAlias

import pandas as pd

from .annotations import Annotations
from .utils import File, infer_tsv_type, automatically_choose_from_disambiguated_files, ask_user_to_choose_from_disambiguated_files, pretty_dict, get_musescore, load_tsv
from .score import Score
from .logger import LoggedClass, function_logger
from .view import View, DefaultView

ParsedFile: TypeAlias = Union[Score, pd.DataFrame]

class Piece(LoggedClass):
    """Wrapper around :class:`~.score.Score` for associating it with parsed TSV files"""

    def __init__(self, fname: str, view: View = None, logger_cfg={}, ms=None):
        super().__init__(subclass='Piece', logger_cfg=logger_cfg)
        self.fname = fname
        available_types = ('scores',) + Score.dataframe_types
        self.type2files: Dict[str, List[File]] = defaultdict(list)
        """{typ -> [:obj:`File`]} dict storing file information for associated types.
        """
        self.type2files.update({typ: [] for typ in available_types})
        self.ix2file: Dict[int, File] = defaultdict(list)
        """{ix -> :obj:`File`} dict storing the registered file information for access via index.
        """
        self.type2parsed: Dict[str, Dict[int, ParsedFile]] = defaultdict(dict)
        """{typ -> {ix -> :obj:`pandas.DataFrame`|:obj:`Score`}} dict storing parsed files for associated types.
        """
        self.type2parsed.update({typ: {} for typ in available_types})
        self.ix2parsed: Dict[int, ParsedFile] = {}
        """{ix -> :obj:`pandas.DataFrame`|:obj:`Score`} dict storing the parsed files for access via index.
        """
        self.ix2annotations: Dict[int, Annotations] = {}
        """{ix -> :obj:`Annotations`} dict storing Annotations objects for the parsed labels and expanded labels.
        """
        self._views: dict = {}
        self._views[None] = DefaultView('current') if view is None else view
        self._views['default'] = DefaultView('default')
        self._views['all'] = View('all')
        self._ms = get_musescore(ms, logger=self.logger)
        """:obj:`str`
        Path or command of the local MuseScore 3 installation if specified by the user."""
        # self.parsed_scores: dict = {}
        # """{ix -> :obj:`Score`}"""
        # self.parsed_tsvs: dict = {}
        # """{ix -> :obj:`pandas.DataFrame`}"""
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

    def set_view(self, current: View = None, **views: View):
        """Register one or several view_name=View pairs."""
        if current is not None:
            self._views[None] = current
        for view_name, view in views.items():
            if view.name is None:
                view.name = view_name
            self._views[view_name] = view

    def get_view(self, view_name: str = None) -> View:
        """Retrieve an existing or create a new View object."""
        if view_name in self._views:
            return self._views[view_name]
        elif view_name is not None and self._views[None].name == view_name:
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
            del (self._views[new_name])
        if show_info:
            self.info()

    def __getattr__(self, view_name):
        if view_name in self._views:
            self.info(view_name=view_name)
        elif view_name is not None and self._views[None].name == view_name:
            self.info()
        else:
            raise AttributeError(f"'{view_name}' is not an existing view. Use _.get_view('{view_name}') to create it.")

    def parse_file_at_index(self, ix: int) -> None:
        assert ix in self.ix2file, f"Piece '{self.fname}' does not include a file with index {ix}."
        file = self.ix2file[ix]
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



    def __getitem__(self, ix) -> ParsedFile:
        assert ix in self.ix2file, f"Piece '{self.fname}' does not include a file with index {ix}."
        if ix not in self.ix2parsed:
            self.parse_file_at_index(ix)
        if ix not in self.ix2parsed:
            file = self.ix2file[ix]
            raise RuntimeError(f"Unable to parse '{file.rel_path}'.")
        return self.ix2parsed[ix]


    def get_first_parsed_score(self, view_name: str = None) -> Tuple[File, Score]:
        parsed_scores = self.get_parsed_score_files(view_name)
        if len(parsed_scores) == 0:
            if len(self.type2parsed['scores']) > 0:
                ixs = list(self.type2parsed['scores'].keys())
                plural = f"a parsed score at index {ixs[0]}" if len(ixs) == 1 else f"parsed scores at indices {ixs}"
                if view_name is None:
                    view_name = self.get_view().name
                self.logger.info(f"Piece('{self.fname}') has {plural}, but they don't seem to be included in this View (name {view_name}).")
            return None, None
        file = parsed_scores[0]
        score_obj = self.ix2parsed[file.ix]
        if len(parsed_scores) > 1:
            self.info(f"Several parsed scores are available for '{self.fname}'. Picked '{file.rel_path}'")
        return file, score_obj

    def add_parsed_score(self, ix: int, score_obj: Score) -> None:
        assert ix in self.ix2file, f"Piece '{self.fname}' does not include a file with index {ix}."
        if score_obj is None:
            file = self.ix2file[ix]
            self.logger.debug(f"I was promised the parsed score for '{file.rel_path}' but received None.")
            return
        self.ix2parsed[ix] = score_obj
        self.type2parsed['scores'][ix] = score_obj

    def add_parsed_tsv(self, ix: int, parsed_tsv: pd.DataFrame) -> None:
        assert ix in self.ix2file, f"Piece '{self.fname}' does not include a file with index {ix}."
        if parsed_tsv is None:
            file = self.ix2file[ix]
            self.logger.debug(f"I was promised the parsed DataFrame for '{file.rel_path}' but received None.")
            return
        self.ix2parsed[ix] = parsed_tsv
        inferred_type = infer_tsv_type(parsed_tsv)
        file = self.ix2file[ix]
        if file.type != inferred_type:
            self.logger.info(f"File {file.rel_path} turned out to be a '{inferred_type}' instead of a '{file.type}', "
                             f"as I had guessed from its path.")
            self.type2files[file.type].remove(file)
            file.type = inferred_type
            self.type2files[inferred_type].append(file)
        self.type2parsed[inferred_type][ix] = parsed_tsv
        if inferred_type in ('labels', 'expanded'):
            self.ix2annotations = Annotations(df = parsed_tsv)

    def make_annotation_objects(self, view_name: str = None, label_col='label'):
        selected_files = self.get_files(['labels', 'expanded'], unparsed=False, view_name=view_name)
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


    def get_files(self, file_type: str|Collection[str] = None,
                     view_name: str = None,
                     parsed: bool = True,
                     unparsed: bool = True,
                     choose: Literal['all', 'auto', 'ask'] = 'all',
                     flat: bool = False) -> Union[Dict[str, File], List[File]]:
        """

        Args:
            file_type:
            choose: If choose == 'all', all dict values will be Lists

        Returns:
            A {file_type -> [:obj:`File`] dict containing the selected Files or, if flat=True, just a list.
        """
        assert parsed + unparsed > 0, "At least one of 'parsed' and 'unparsed' needs to be True."
        if file_type is None:
            file_type = list(self.type2files.keys())
        elif isinstance(file_type, str):
            if file_type in ('tsv', 'tsvs'):
                file_type = list(self.type2files.keys())
                file_type.remove('scores')
            else:
                file_type = [file_type]
        if any(t not in self.type2files for t in file_type):
            unknown = [t for t in file_type if t not in self.type2files]
            file_type = [t for t in file_type if t in self.type2files]
            self.logger.warning(f"Unknown file type(s): {unknown}")
        if unparsed:
            type2files = {t: self.type2files[t] for t in file_type}
        else:
            # i.e., parsed must be True
            type2files = {t: self.type2files[t] for t in file_type}
            type2files = {typ: [f for f in files if f.ix in self.ix2parsed] for typ, files in type2files.items()}
        if not parsed:
            # i.e., unparsed must be True
            type2files = {typ: [f for f in files if f.ix not in self.ix2parsed] for typ, files in type2files.items()}
        view = self.get_view(view_name=view_name)
        type2files = {typ: view.filtered_file_list(files, 'files') for typ, files in type2files.items()}
        result = {}
        needs_choice = []
        for t, files in type2files.items():
            if choose == 'all' or len(files) < 2:
                result[t] = files
            else:
                disambiguated = disambiguate_files(files, logger=self.logger)
                needs_choice.append(t)
                result[t] = disambiguated
        if choose == 'auto':
            for typ in needs_choice:
                result[typ] = automatically_choose_from_disambiguated_files(result[typ], self.fname, typ)
        elif choose == 'ask':
            for typ in needs_choice:
                choices = result[typ]
                result[typ] = ask_user_to_choose_from_disambiguated_files(choices, self.fname, typ)
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

    def get_parsed_score_files(self, view_name: str = None, flat: bool = True):
        return self.get_files('scores', view_name=view_name, unparsed=False, flat=flat)

    def get_parsed_tsv_files(self, view_name: str = None, flat: bool = True):
        return self.get_files('tsv', view_name=view_name, unparsed=False, flat=flat)

    def get_unparsed_score_files(self, view_name: str = None, flat: bool = True):
        return self.get_files('scores', view_name=view_name, parsed=False, flat=flat)

    def get_unparsed_tsv_files(self, view_name: str = None, flat: bool = True):
        return self.get_files('tsv', view_name=view_name, parsed=False, flat=flat)

    def register_file(self, file: File, reject_incongruent_fnames: bool = True) -> bool:
        ix = file.ix
        if ix in self.ix2file:
            existing_file = self.ix2file[ix]
            if file.full_path == existing_file.full_path:
                self.logger.debug(f"File '{file.rel_path}' was already registered for {self.fname}.")
                return
            else:
                self.logger.debug(f"File '{existing_file.rel_path}' replaced with '{file.rel_path}'")
        if file.fname != self.fname:
            if file.fname.startswith(self.fname):
                file.suffix = file.fname[len(self.fname):]
                self.logger.debug(f"Recognized suffix '{file.suffix}' for {file.file}.")
            elif reject_incongruent_fnames:
                if self.fname in file.fname:
                    self.logger.info(f"{file.file} seems to come with a prefix w.r.t. '{self.fname}' and is ignored.")
                    return False
                else:
                    self.logger.warning(f"{file.file} does not contain '{self.fname}' and is ignored.")
                    return False
        self.type2files[file.type].append(file)
        self.ix2file[file.ix] = file
        return True

    def iter_type2files(self, view_name: str = None, drop_zero: bool = False) -> Iterator[Dict[str, List[File]]]:
        """Iterating through _.type2files.items() under the current or specified view."""
        view = self.get_view(view_name=view_name)
        for typ, files in self.type2files.items():
            filtered_files = view.filtered_file_list(files, 'files')
            # the files need to be filtered even if the facet is excluded, for counting excluded files
            if drop_zero and len(filtered_files) == 0:
                continue
            if typ in view.selected_facets:
                yield typ, filtered_files


    def iter_type2parsed_files(self, view_name: str = None, drop_zero: bool = False) -> Iterator[Dict[str, List[File]]]:
        """Iterating through _.type2files.items() under the current or specified view and selecting only parsed files."""
        view = self.get_view(view_name=view_name)
        for typ, ix2parsed in self.type2parsed.items():
            files = [self.ix2file[ix] for ix in ix2parsed.keys()]
            filtered_ixs = [file.ix for file in view.filtered_file_list(files, 'parsed')]
            # the files need to be filtered even if the facet is excluded, for counting excluded files
            if drop_zero and len(filtered_ixs) == 0:
                continue
            if typ in view.selected_facets:
                yield typ, {ix: ix2parsed[ix] for ix in filtered_ixs}


    def count_types(self, drop_zero=False, view_name: str = None) -> Dict[str, int]:
        return {"found_" + typ: len(files) for typ, files in self.iter_type2files(view_name=view_name, drop_zero=drop_zero)}

    def count_parsed(self, drop_zero=False, view_name: str = None) -> Dict[str, int]:
       return {"parsed_" + typ: len(parsed) for typ, parsed in self.iter_type2parsed_files(view_name=view_name, drop_zero=drop_zero)}

    def info(self, return_str=False, view_name=None):
        print(self.get_view(view_name))
        type2files = dict(self.iter_type2files(view_name=view_name, drop_zero=True))
        if len(type2files) == 0:
            msg = "No files selected."
        else:
            files_df = pd.concat([pd.DataFrame(files).set_index('ix') for files in type2files.values()],
                                 keys=type2files.keys(),
                                 names=['facet', 'ix'])
            if len(files_df) == 0:
                msg = "No files selected."
            else:
                ixs = files_df.index.get_level_values(1)
                is_parsed = [str(ix in self.ix2parsed) for ix in ixs]
                files_df['is_parsed'] = is_parsed
                msg = files_df[['rel_path', 'is_parsed']].to_string()
        if return_str:
            return msg
        print(msg)


    def __repr__(self):
        return self.info(return_str=True)


@function_logger
def disambiguate_files(files: Collection[File]) -> Dict[str, File]:
    """Takes a list of :class:`File` returns a dictionary with disambiguating strings based on path components.
    of distinct strings to distinguish files pertaining to the same type."""
    N_target = len(files)
    if N_target == 0:
        return {}
    if N_target== 1:
        f = files[0]
        return {f.type: f}
    disambiguation = [f.type for f in files]
    if len(set(disambiguation)) == N_target:
        # done disambiguating
        return dict(zip(disambiguation, files))
    # first, try to disambiguate based on the files' sub-directories
    subdirs = []
    for f in files:
        file_type = f.type
        subdir = f.subdir.strip(r"\/.")
        if subdir.startswith(file_type):
            subdir = subdir[len(file_type):]
        if subdir.strip(r"\/") == '':
            subdir = ''
        subdirs.append(subdir)
    if len(set(subdirs)) > 1:
        # files can (partially) be disambiguated because they are in different sub-directories
        disambiguation = [os.path.join(disamb, subdir) for disamb, subdir in zip(disambiguation, subdirs)]
    if len(set(disambiguation)) == N_target:
        # done disambiguating
        return dict(zip(disambiguation, files))
    # next, try adding detected suffixes
    for ix, f in enumerate(files):
        if f.suffix != '':
            disambiguation[ix] += f"[{f.suffix}]"
    if len(set(disambiguation)) == N_target:
        # done disambiguating
        return dict(zip(disambiguation, files))
    # now, add file extensions to disambiguate further
    if len(set(f.fext for f in files)) > 1:
        for ix, f in enumerate(files):
            disambiguation[ix] += f.fext
    if len(set(disambiguation)) == N_target:
        # done disambiguating
        return dict(zip(disambiguation, files))
    str_counts = Counter(disambiguation)
    duplicate_disambiguation_strings = [s for s, cnt in str_counts.items() if cnt > 1]
    ambiguate_files = {s: [f for disamb, f in zip(disambiguation, files) if disamb == s] for s in duplicate_disambiguation_strings}
    result = dict(zip(disambiguation, files))
    remaining_ones = {s: result[s] for s in duplicate_disambiguation_strings}
    logger.warning(f"The following files could not be ambiguated: {ambiguate_files}.\n"
                   f"In the result, only these remain: {remaining_ones}.")
    return result
