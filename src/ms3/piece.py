import os
from collections import defaultdict, Counter
from typing import Dict, Collection, Literal, Tuple, Union, List, Iterator

import pandas as pd

from .utils import File, infer_tsv_type, automatically_choose_from_disambiguated_files, ask_user_to_choose_from_disambiguated_files, pretty_dict
from .score import Score
from .logger import LoggedClass, function_logger
from .view import View, DefaultView


class Piece(LoggedClass):
    """Wrapper around :class:`~.score.Score` for associating it with parsed TSV files"""

    def __init__(self, fname: str, view: View = None, logger_cfg={}):
        super().__init__(subclass='Piece', logger_cfg=logger_cfg)
        self.fname = fname
        available_types = ('scores',) + Score.dataframe_types
        self.type2files: defaultdict = defaultdict(list)
        """{typ -> [:obj:`File`]} dict storing file information for associated types.
        """
        self.type2files.update({typ: [] for typ in available_types})
        self.ix2file: defaultdict = defaultdict(list)
        """{ix -> :obj:`File`} dict storing the identical file information for access via index.
        """
        self.type2parsed: defaultdict = defaultdict(dict)
        """{typ -> {ix -> :obj:`pandas.DataFrame`|:obj:`Score`}} dict storing parsed files for associated types.
        """
        self.type2parsed.update({typ: {} for typ in available_types})
        self.ix2parsed: defaultdict = defaultdict(list)
        """{ix -> :obj:`pandas.DataFrame`|:obj:`Score`} dict storing the identical file information for access via index.
        """
        self._views: dict = {}
        self._views[None] = DefaultView('current') if view is None else view
        self._views['default'] = DefaultView('default')
        self._views['all'] = View('all')
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

    def get_parsed_score(self) -> Tuple[File, Score]:
        parsed_scores = self.type2parsed['scores']
        if len(parsed_scores) == 0:
            return None, None
        for ix, score_obj in parsed_scores.items():
            break
        file = self.ix2file[ix]
        if len(parsed_scores) > 1:
            self.info(f"Several parsed scores are available for '{self.fname}'. Picked '{file.rel_path}'")
        return file, score_obj

    def add_parsed_score(self, ix: int, score_obj: Score):
        assert ix in self.ix2file, f"Piece '{self.fname}' does not include a file with index {ix}."
        self.ix2parsed[ix] = score_obj
        self.type2parsed['scores'][ix] = score_obj

    def add_parsed_tsv(self, ix: int, parsed_tsv: pd.DataFrame):
        assert ix in self.ix2file, f"Piece '{self.fname}' does not include a file with index {ix}."
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



    def select_files(self, file_type: str|Collection[str],
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
        if isinstance(file_type, str):
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



    def add_file(self, file: File, reject_incongruent_fnames: bool = True) -> bool:
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

    def iter_type2files(self, view_name: str = None) -> Iterator[Dict[str, List[File]]]:
        """Iterating through _.type2files.items() under the current or specified view."""
        view = self.get_view(view_name=view_name)
        for typ, files in self.type2files.items():
            yield typ, view.filtered_file_list(files, 'files')


    def iter_type2parsed_files(self, view_name: str = None) -> Iterator[Dict[str, List[File]]]:
        """Iterating through _.type2files.items() under the current or specified view."""
        view = self.get_view(view_name=view_name)
        for typ, ix2file in self.type2parsed.items():
            files = [self.ix2file[ix] for ix in ix2file.keys()]
            yield typ, view.filtered_file_list(files, 'parsed')


    def count_types(self, drop_zero=False, view_name: str = None) -> Dict[str, int]:
        if drop_zero:
            return {"found_" + typ: len(files) for typ, files in self.iter_type2files(view_name=view_name) if len(files) > 0}
        return {"found_" + typ: len(files) for typ, files in self.iter_type2files(view_name=view_name)}

    def count_parsed(self, drop_zero=False, view_name: str = None) -> Dict[str, int]:
        if drop_zero:
            return {"parsed_" + typ: len(parsed) for typ, parsed in self.iter_type2parsed_files(view_name=view_name) if len(parsed) > 0}
        return {"parsed_" + typ: len(parsed) for typ, parsed in self.iter_type2parsed_files(view_name=view_name)}

    def info(self, return_str=False, view_name=None):
        print(self.get_view(view_name))
        counts = self.count_parsed(drop_zero=True)
        counts.update(self.count_types(drop_zero=True))
        if return_str:
            return str(counts)
        print(counts)


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
