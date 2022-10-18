import os
from collections import defaultdict, Counter
from typing import Dict, Collection, Literal, Tuple, Union, List, Generator

import pandas as pd

from .utils import File, file_type2path_component_map, infer_tsv_type
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
        self._views[None] = view if view is None else DefaultView('current')
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
        self._views[view_name] = View(view_name)
        self.logger.info(f"New view '{view_name}' created.")
        return self._views[view_name]

    def __getattr__(self, view_name):
        if view_name in self._views:
            self.info(view_name=view_name)
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


    def automatically_choose_from_disambiguated_files(self, file_type: str, disambiguation : Dict[str, File], view_name: str = None):
        if len(disambiguation) == 1:
            return list(disambiguation.keys())[0]
        disamb_series = pd.Series(disambiguation)
        files = list(disambiguation.values())
        files_df = pd.DataFrame(files, index=disamb_series.index)
        choice_between_n = len(files)
        if file_type == 'scores':
            # if a score is requested, check if there is only a single MSCX or otherwise MSCZ file and pick that
            fexts = files_df.fext.str.lower()
            fext_counts = fexts.value_counts()
            if '.mscx' in fext_counts:
                if fext_counts['.mscx'] == 1:
                    selected_file = disamb_series[fexts == '.mscx'].iloc[0]
                    self.logger.debug(f"In order to pick one from the {choice_between_n} scores with fname '{self.fname}', '{selected_file.rel_path}' was selected because it is the only "
                                     f"one in MSCX format.")
                    return selected_file
            elif '.mscz' in fext_counts and fext_counts['.mscz'] == 1:
                selected_file = disamb_series[fexts == '.mscz'].iloc[0]
                self.logger.debug(f"In order to pick one from the {choice_between_n} scores with fname '{self.fname}', '{selected_file.rel_path}' was selected because it is the only "
                                 f"one in MSCZ format.")
                return selected_file
        # as first disambiguation criterion, check if the shortest disambiguation string pertains to 1 file only and pick that
        disamb_str_lengths = pd.Series(disamb_series.index.map(len), index=disamb_series.index)
        shortest_length_selector = (disamb_str_lengths == disamb_str_lengths.min())
        n_have_shortest_length = shortest_length_selector.sum()
        if n_have_shortest_length == 1:
            ix = disamb_str_lengths.idxmin()
            selected_file = disamb_series.loc[ix]
            self.logger.debug(f"In order to pick one from the {choice_between_n} '{file_type}' with fname '{self.fname}', the one with the shortest disambiguating string '{ix}' was selected.")
            return selected_file
        if file_type != 'unknown':
            # otherwise, check if only one file is lying in a directory with default name
            subdirs = files_df.subdir
            default_components = file_type2path_component_map()[file_type]
            default_components_regex = '|'.join(comp.replace('.', r'\.') for comp in default_components)
            default_selector = subdirs.str.contains(default_components_regex, regex=True)
            if default_selector.sum() == 1:
                subdir = subdirs[default_selector].iloc[0]
                selected_file = disamb_series[default_selector].iloc[0]
                self.logger.debug(f"In order to pick one from the {choice_between_n} '{file_type}' with fname '{self.fname}', the one in the default subdir '{subdir}' was selected.")
                return selected_file
            # or if only one file contains a default name in its suffix
            suffixes = files_df.suffix
            default_selector = suffixes.str.contains(default_components_regex, regex=True)
            if default_selector.sum() == 1:
                suffix = suffixes[default_selector].iloc[0]
                selected_file = disamb_series[default_selector].iloc[0]
                self.logger.debug(f"In order to pick one from the {choice_between_n} '{file_type}' with fname '{self.fname}', the one in the default suffix '{suffix}' was selected.")
                return selected_file
        # if no file was selected, try again with only those having the shortest disambiguation strings
        if shortest_length_selector.all():
            # if all disambiguation strings already have the shortest length, as a last resort
            # fall back to the lexigographically first
            sorted_disamb_series = disamb_series.sort_index()
            disamb = sorted_disamb_series.index[0]
            selected_file = sorted_disamb_series.iloc[0]
            self.logger.warning(f"Unable to automatically choose from the {choice_between_n} '{file_type}' with fname '{self.fname}', I selected '{selected_file.rel_path}' "
                                f"because it's disambiguation string '{disamb}' is the lexicographically first among {sorted_disamb_series.index.to_list()}")
            return selected_file
        only_shortest_disamb_str = disamb_series[shortest_length_selector].to_dict()
        self.logger.info(f"After the first unsuccessful attempt to choose from {choice_between_n} '{file_type}' with fname '{self.fname}', trying again "
                            f"after reducing the choices to the {shortest_length_selector.sum()} with the shortest disambiguation strings.")
        return self.automatically_choose_from_disambiguated_files(file_type, only_shortest_disamb_str)

    def select_files(self, file_type: str|Collection[str], view_name: str = None, choose: Literal['all', 'auto', 'ask'] = 'auto') -> Tuple[Dict[str, Union[Literal[None], File, Dict[str, File]]], List[str]]:
        """

        Args:
            file_type:
            choose:

        Returns:
            An dict mapping a file_type to an unambiguous :obj:`File` or to a {disambiguated_file_type -> :obj:`File`} dict where keys start with the file type and may have subdirs, suffixes, or file extensions appended.
            A [file_type] list containing those file_types that have several files to choose from.

        """
        if isinstance(file_type, str):
            file_type = [file_type]
        if any(t not in self.type2files for t in file_type):
            unknown = [t for t in file_type if t not in self.type2files]
            file_type = [t for t in file_type if t in self.type2files]
            self.logger.warning(f"Unknown file type(s): {unknown}")
        type2files = {t: self.type2files[t] for t in file_type}
        #if any(len(files) > 1 for files in type2files.values()):
        result = {}
        needs_choice = []
        for t, files in type2files.items():
            if len(files) == 0:
                result[t] = None
            elif len(files) == 1:
                result[t] = files[0]
            else:
                disambiguated = disambiguate_files(files, logger=self.logger)
                needs_choice.append(t)
                result[t] = disambiguated
        if choose == 'auto':
            for typ in needs_choice:
                result[typ] = self.automatically_choose_from_disambiguated_files(typ, result[typ])
        elif choose == 'ask':
            for typ in needs_choice:
                choices = result[typ]
                sorted_keys = sorted(choices.keys(), key=lambda s: (len(s), s))
                choices = {k: choices[k] for k in sorted_keys}
                file_list = list(choices.values())
                disamb_strings = pd.Series(choices.keys(), name='disambiguation_str')
                choices_df = pd.concat([disamb_strings,
                                        pd.DataFrame(file_list)[['rel_path', 'type', 'ix']]],
                                       axis=1)
                choices_df.index.rename('select:', inplace=True)
                range_str = f"0-{len(choices) - 1}"
                query = f"Selection [{range_str}]: "
                print(f"Several '{typ}' available for '{self.fname}':\n{choices_df.to_string()}")
                print(f"Please select one of the files by passing an integer between {range_str}:")
                permitted = list(range(len(choices)))

                def test_integer(s):
                    nonlocal permitted, range_str
                    try:
                        int_i = int(s)
                    except:
                        print(f"Value '{s}' could not be converted to an integer.")
                        return None
                    if int_i not in permitted:
                        print(f"Value '{s}' is not between {range_str}.")
                        return None
                    return int_i

                ask_user = True
                while ask_user:
                    selection = input(query)
                    int_i = test_integer(selection)
                    if int_i is not None:
                        result[typ] = file_list[int_i]
                        ask_user = False
        return result, needs_choice



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

    def iter_type2files(self, view_name: str = None) -> Generator[Dict[str, List[File]], None, None]:
        """Iterating through _.type2files.items() under the current or specified view."""
        view = self.get_view(view_name=view_name)
        for typ, files in self.type2files.items():
            yield typ, view.filtered_file_list(files)


    def iter_type2parsed(self, view_name: str = None) -> Generator[Dict[str, List[File]], None, None]:
        """Iterating through _.type2files.items() under the current or specified view."""
        view = self.get_view(view_name=view_name)
        for typ, files in self.type2parsed.items():
            yield typ, view.filtered_file_list(files)


    def count_types(self, drop_zero=False, view_name: str = None) -> Dict[str, int]:
        if drop_zero:
            return {"found_" + typ: len(files) for typ, files in self.iter_type2files(view_name=view_name) if len(files) > 0}
        return {"found_" + typ: len(files) for typ, files in self.iter_type2files(view_name=view_name)}

    def count_parsed(self, drop_zero=False, view_name: str = None) -> Dict[str, int]:
        if drop_zero:
            return {"parsed_" + typ: len(parsed) for typ, parsed in self.iter_type2parsed(view_name=view_name) if len(parsed) > 0}
        return {"parsed_" + typ: len(parsed) for typ, parsed in self.iter_type2parsed(view_name=view_name)}

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
