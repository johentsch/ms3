import re
from collections import defaultdict
from copy import deepcopy
from typing import Collection, Union, Iterable, List, Dict, Iterator, Optional

import numpy as np
import numpy.typing as npt

from .score import Score
from ._typing import FileList, Category, Categories
from .utils import File
from .logger import LoggedClass


def empty_counts():
    """Array for counting kept items, discarded items, and their sum."""
    return np.zeros(3, dtype=int)

class View(LoggedClass):
    """"""
    review_regex = "review"
    categories = (
        'corpora',
        'folders',
        'fnames',
        'files',
        'suffixes',
        'facets',
    )
    available_facets = ('scores',) + Score.dataframe_types + ('unknown',)
    singular2category = dict(zip(('corpus', 'folder', 'fname', 'file', 'suffix', 'facet'),
                                   categories))
    tsv_regex = re.compile(r"\.tsv$", re.IGNORECASE)
    convertible_regex = Score.make_extension_regex(native=False, convertible=True, tsv=False)
    registered_regexes = (convertible_regex, review_regex, tsv_regex)

    def __init__(self,
                 view_name: Optional[str] = 'all',
                 only_metadata: bool = False,
                 include_convertible: bool = True,
                 include_tsv: bool = True,
                 exclude_review: bool = False,
                 **logger_cfg
                 ):
        super().__init__(subclass='View', logger_cfg=logger_cfg)
        assert isinstance(view_name, str), f"Name of the view should be a string, not '{type(view_name)}'"
        if view_name is not None and not view_name.isidentifier():
            self.logger.info(f"The string '{view_name}' cannot be used as attribute name.")
        self.name: str = view_name
        self.including: dict = {c: [] for c in self.categories}
        self.excluding: dict = {c: [] for c in self.categories}
        self.selected_facets = self.available_facets
        self.fnames_in_metadata: bool = True
        self.fnames_not_in_metadata: bool = not only_metadata
        self.include_convertible = include_convertible
        self.include_tsv = include_tsv
        self.exclude_review = exclude_review
        self._last_filtering_counts: Dict[str, npt.NDArray[int, int, int]] = defaultdict(empty_counts)
        """For each filter method, store the counts of the last run as [n_kept, n_discarded, N (the sum)].
        Keys are f"filtered_{category}" for :meth:`filter_by_token` and 'files' or 'parsed' for :meth:`filtered_file_list`.
        To inspect, you can use the method :meth:`filtering_report`
        """
        self._discarded_items: Dict[str, List[str]] = defaultdict(list)
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%% END of __init__() %%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    def copy(self, new_name: str):
        new_view = self.__class__(view_name=new_name)
        new_view.including = deepcopy(self.including)
        new_view.excluding = deepcopy(self.excluding)
        new_view.update_facet_selection()
        new_view.fnames_in_metadata = self.fnames_in_metadata
        return new_view

    def update_config(self,
                     view_name: Optional[str] = None,
                     only_metadata: bool = None,
                     include_convertible: bool = None,
                     include_tsv: bool = None,
                     exclude_review: bool = None,
                     **logger_cfg):
        for param, value in zip(('view_name', 'only_metadata', 'include_convertible', 'include_tsv', 'exclude_review'),
                                (view_name, only_metadata, include_convertible, include_tsv, exclude_review)
                                ):
            if value is None:
                continue
            old_value = getattr(self, param)
            if value != old_value:
                setattr(self, param, value)
                self.logger.debug(f"Set '{param}' (previously {old_value}) to {value}.")
        if 'level' in logger_cfg:
            self.change_logger_cfg(level=logger_cfg['level'])

    @property
    def include_convertible(self):
        return self.convertible_regex not in self.excluding['files']

    @include_convertible.setter
    def include_convertible(self, yes: bool):
        if yes:
            self.unexclude('files', self.convertible_regex)
        else:
            self.exclude('files', self.convertible_regex)


    @property
    def include_tsv(self):
        return self.tsv_regex not in self.excluding['files']

    @include_tsv.setter
    def include_tsv(self, yes: bool):
        if yes:
            self.unexclude('files', self.tsv_regex)
        else:
            self.exclude('files', self.tsv_regex)


    @property
    def exclude_review(self):
        return all(self.review_regex in self.excluding[what_to_exclude]
                   for what_to_exclude in ('suffixes', 'folders'))

    @exclude_review.setter
    def exclude_review(self, yes: bool):
        if yes:
            self.exclude(('suffixes', 'folders'), self.review_regex)
        else:
            self.unexclude(('suffixes', 'folders'), self.review_regex)

    def check_token(self, category: Category, token: str) -> bool:
        """Checks if a string pertaining to a certain category should be included in the view or not."""
        category = self.resolve_categories(category)
        if any(re.search(rgx, token) is not None for rgx in self.excluding[category]):
            return False
        if len(self.including[category]) == 0:
            return True
        return any(re.search(rgx, token) is not None for rgx in self.including[category])


    def check_file(self, file: File) -> bool:
        """Check if an individual File passes all filters w.r.t. its subdirectories, file name and suffix."""
        category2file_component = dict(zip(('folders', 'files', 'suffixes'),
                                           (file.subdir, file.file, file.suffix)
                                           ))
        for category, component in category2file_component.items():
            if any(re.search(rgx, component) is not None for rgx in self.excluding[category]):
                return False
        for category, component in category2file_component.items():
            if len(self.including[category]) == 0 or any(re.search(rgx, component) is not None for rgx in self.including[category]):
                continue
            else:
                return False
        return True

    def reset_filtering_data(self):
        self._last_filtering_counts = defaultdict(empty_counts)
        self._discarded_items = defaultdict(list)
        self.update_facet_selection()


    def filter_by_token(self, category: Category, tuples: Iterable[tuple]) -> Iterator[tuple]:
        """Filters out those tuples where the token (first element) does not pass _.check_token(category, token)."""
        category = self.resolve_categories(category)
        n_kept, n_discarded, N = 0, 0, 0
        discarded_items = []
        for tup in tuples:
            N += 1
            token, *_ = tup
            if self.check_token(category=category, token=token):
                n_kept += 1
                yield tup
            else:
                n_discarded += 1
                discarded_items.append(token)
        key = f"filtered_{category}"
        self._last_filtering_counts[key] += np.array([n_kept, n_discarded, N], dtype='int')
        self._discarded_items[key].extend(discarded_items)


    def filtered_file_list(self, files: Collection[File], key: str = None) -> FileList:
        """ Keep only the files that pass _.check_file().

        Args:
            files: :obj:`File` objects to be filtered.
            key: Aggregate results from several filter runs under this dictionary key.

        Returns:

        """
        if len(files) == 0:
            return files
        result, discarded_items = [], []
        for file in files:
            if self.check_file(file):
                result.append(file)
            else:
                discarded_items.append(file.rel_path)
        N, n_kept = len(files), len(result)
        n_discarded = N - n_kept
        if key is None:
            key = 'filtered_file_list'
        self._last_filtering_counts[key] += np.array([n_kept, n_discarded, N], dtype='int')
        self._discarded_items[key].extend(discarded_items)
        return result

    def filtering_report(self, drop_zero=True, show_discarded=False):
        aggregated_counts = defaultdict(empty_counts)
        for key, counts in self._last_filtering_counts.items():
            key = key.replace('filtered_', '')
            aggregated_counts[key] += counts
        if show_discarded:
            discarded = defaultdict(list)
            for key, items in self._discarded_items.items():
                key = key.replace('filtered_', '')
                discarded[key].extend(items)
        msg = ''
        for key, (_, n_discarded, N) in aggregated_counts.items():
            if not drop_zero or n_discarded > 0:
                msg += f"{n_discarded}/{N} {key} have been discarded"
                if show_discarded:
                    if len(discarded[key]) > 0:
                        msg += f":\n{discarded[key]}\n\n"
                    else:
                        msg += ", but unfortunately I don't know which ones.\n"
                else:
                    msg += '.\n'
        return msg

    def info(self, return_str=False):
        msg_components = []
        if self.fnames_in_metadata + self.fnames_not_in_metadata == 0:
            msg = f"This view is called '{self.name}'. It excludes everything because both its attributes " \
                   f"fnames_in_metadata and fnames_not_in_metadata are set to False."
            if return_str:
                return msg
            print(msg)
            return
        if not self.fnames_not_in_metadata:
            msg_components.append("excludes fnames that are not contained in the metadata")
        if not self.include_convertible:
            msg_components.append("filters out file extensions requiring conversion (such as .xml)")
        if not self.include_tsv:
            msg_components.append("disregards all TSV files")
        if self.exclude_review:
            msg_components.append("excludes review files and folders")
        included_re = {what_to_include: [f'"{rgx}"' for rgx in regexes if rgx not in self.registered_regexes]
                       for what_to_include, regexes in self.including.items()}
        excluded_re = {what_to_exclude: [f'"{rgx}"' for rgx in regexes if rgx not in self.registered_regexes]
                       for what_to_exclude, regexes in self.excluding.items()}
        msg_components.extend([f"includes only {what_to_include} containing {re_strings[0] if len(re_strings) == 1 else 'one of ' + str(re_strings)}"
                            for what_to_include, re_strings in included_re.items()
                            if len(re_strings) > 0])
        msg_components.extend([f"excludes any {what_to_exclude} containing {re_strings[0] if len(re_strings) == 1 else 'one of ' + str(re_strings)}"
                            for what_to_exclude, re_strings in excluded_re.items()
                            if len(re_strings) > 0])
        msg = f"This view is called '{self.name}'. It "
        n_components = len(msg_components)
        if n_components == 0:
            msg += "selects everything."
        elif n_components == 1:
            msg += msg_components[0] + "."
        else:
            msg += ', '.join(msg_components[:-1])
            msg += f", and {msg_components[-1]}."
        if return_str:
            return msg
        print(msg)


    def resolve_categories(self, category):
        if isinstance(category, str):
            if category not in self.categories:
                if category in self.singular2category:
                    return self.singular2category[category]
                else:
                    self.logger.error(f"'{category}' is not one of the known categories {self.categories}")
            return category
        else:
            # assumes this to be iterable
            return [self.resolve_categories(categ) for categ in category]

    def update_facet_selection(self):
        selected, discarded = [], []
        for facet in self.available_facets:
            if self.check_token('facet', facet):
                selected.append(facet)
            else:
                discarded.append(facet)
        self.selected_facets = selected
        key = 'filtered_facets'
        if len(discarded) == 0:
            if key in self._last_filtering_counts:
                del(self._last_filtering_counts[key])
            if key in self._discarded_items:
                del(self._discarded_items[key])
            return
        n_kept, n_discarded = len(selected), len(discarded)
        counts = np.array([n_kept, n_discarded, n_kept+n_discarded])
        self._last_filtering_counts[key] = counts
        self._discarded_items[key] = discarded

    def include(self, categories: Categories, *regex: Union[str, re.Pattern]):
        categories = self.resolve_categories(categories)
        if isinstance(categories, str):
            categories = [categories]
        for what_to_include in categories:
            for rgx in regex:
                if rgx not in self.including[what_to_include]:
                    self.including[what_to_include].append(rgx)
            if what_to_include == 'facets':
                self.update_facet_selection()


    def exclude(self, categories: Categories, *regex: Union[str, re.Pattern]):
        categories = self.resolve_categories(categories)
        if isinstance(categories, str):
            categories = [categories]
        for what_to_exclude in categories:
            for rgx in regex:
                if rgx not in self.excluding[what_to_exclude]:
                    self.excluding[what_to_exclude].append(rgx)
            if what_to_exclude == 'facets':
                self.update_facet_selection()

    def uninclude(self, categories: Categories, *regex: Union[str, re.Pattern]):
        categories = self.resolve_categories(categories)
        if isinstance(categories, str):
            categories = [categories]
        for what_to_uninclude in categories:
            for rgx in regex:
                try:
                    self.including[what_to_uninclude].remove(rgx)
                except ValueError:
                    pass


    def unexclude(self, categories: Categories, *regex: Union[str, re.Pattern]):
        categories = self.resolve_categories(categories)
        if isinstance(categories, str):
            categories = [categories]
        for what_to_unexclude in categories:
            for rgx in regex:
                try:
                    self.excluding[what_to_unexclude].remove(rgx)
                except ValueError:
                    pass

    def __repr__(self):
        return self.info(return_str=True)

class DefaultView(View):

    def __init__(self,
                 view_name: Optional[str] = 'default',
                 only_metadata: bool = True,
                 include_convertible: bool = False,
                 include_tsv: bool = True,
                 exclude_review: bool = True,
                 **logger_cfg
                 ):
        super().__init__(view_name=view_name,
                         only_metadata=only_metadata,
                         include_convertible=include_convertible,
                         include_tsv=include_tsv,
                         exclude_review=exclude_review,
                         **logger_cfg
                         )