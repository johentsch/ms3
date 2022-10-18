import re
from typing import TypeAlias, Collection, Literal, Union, Iterable, Tuple, Any, Generator, List

from .score import Score
from .utils import File
from .logger import LoggedClass

Category: TypeAlias = Literal['corpora',
                              'folders',
                              'fnames',
                              'files',
                              'suffixes'
                              'extensions',]

Categories: TypeAlias = Union[Category, Collection[Category]]

class View(LoggedClass):
    """"""
    review_regex = "review"
    categories = (
        'corpora',
        'folders',
        'fnames',
        'files',
        'suffixes'
    )
    singular2category = dict(zip(('corpus', 'folder', 'fname', 'file', 'suffix'),
                                   categories))
    native_regex = Score.make_extension_regex(native=True, convertible=False, tsv=True)
    convertible_regex = Score.make_extension_regex(native=False, convertible=True, tsv=False)
    registered_regexes = (convertible_regex, native_regex, review_regex)

    def __init__(self,
                 view_name: str = None,
                 logger_cfg: dict = {},
                 only_metadata: bool = False,
                 include_convertible: bool = True,
                 exclude_review: bool = False,
                 ):
        super().__init__(subclass='View', logger_cfg=logger_cfg)
        assert isinstance(view_name, str), f"Name of the view should be a string, not '{type(view_name)}'"
        self.name = view_name
        self.including = {c: [] for c in self.categories}
        self.excluding = {c: [] for c in self.categories}
        self.fnames_from_metadata = True
        self.fnames_outside_metadata = not only_metadata
        self.include_convertible = include_convertible
        self.exclude_review = exclude_review

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

    @property
    def include_convertible(self):
        return self.convertible_regex not in self.excluding['files']

    @include_convertible.setter
    def include_convertible(self, yes: bool):
        if yes:
            self.unexclude('files', self.convertible_regex)
        else:
            self.exclude('files', self.convertible_regex)

    def check_token(self, category: Category, token: str) -> bool:
        """Checks if a string pertaining to a certain category should be included in the view or not."""
        if category not in self.categories:
            if category in self.singular2category:
                category = self.singular2category[category]
            else:
                self.logger.error(f"'{category}' is not one of the known categories {self.singular2category}")
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




    def filter_by_token(self, category: Category, tuples: Iterable[tuple]) -> Generator[tuple, None, None]:
        """Filters out those tuples where the token (first element) does not pass _.check_token(category, token)."""
        for tup in tuples:
            token, *_ = tup
            if self.check_token(category=category, token=token):
                yield tup

    def filtered_file_list(self, files: Collection[File]) -> List[File]:
        """Keep only the files that pass _.check_file()"""
        return [file for file in files if self.check_file(file)]

    def info(self, return_str=False):
        msg_components = []
        if self.include_convertible:
            msg_components.append("includes convertible file extensions (such as .xml)")
        if self.exclude_review:
            msg_components.append("excludes review files and folders")
        included_re = {what_to_include: [f'"{rgx}"' for rgx in regexes if rgx not in self.registered_regexes]
                       for what_to_include, regexes in self.including.items()}
        excluded_re = {what_to_exclude: [f'"{rgx}"' for rgx in regexes if rgx not in self.registered_regexes]
                       for what_to_exclude, regexes in self.excluding.items()}
        msg_components.extend([f"includes {what_to_include} matching {', '.join(re_strings)}"
                            for what_to_include, re_strings in included_re.items()
                            if len(re_strings) > 0])
        msg_components.extend([f"excludes {what_to_exclude} matching {', '.join(re_strings)}, "
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

    def __repr__(self):
        return self.info(return_str=True)

    def include(self, categories: Categories, *regex: Union[str, re.Pattern]):
        if isinstance(categories, str):
            categories = [categories]
        for what_to_include in categories:
            for rgx in regex:
                if rgx not in self.including[what_to_include]:
                    self.including[what_to_include].append(rgx)


    def exclude(self, categories: Categories, *regex: Union[str, re.Pattern]):
        if isinstance(categories, str):
            categories = [categories]
        for what_to_exclude in categories:
            for rgx in regex:
                if rgx not in self.excluding[what_to_exclude]:
                    self.excluding[what_to_exclude].append(rgx)


    def uninclude(self, categories: Categories, *regex: Union[str, re.Pattern]):
        if isinstance(categories, str):
            categories = [categories]
        for what_to_uninclude in categories:
            for rgx in regex:
                try:
                    self.including[what_to_uninclude].remove(rgx)
                except ValueError:
                    pass


    def unexclude(self, categories: Categories, *regex: Union[str, re.Pattern]):
        if isinstance(categories, str):
            categories = [categories]
        for what_to_unexclude in categories:
            for rgx in regex:
                try:
                    self.excluding[what_to_unexclude].remove(rgx)
                except ValueError:
                    pass

class DefaultView(View):

    def __init__(self,
                 view_name: str = None,
                 logger_cfg: dict = {},
                 only_metadata: bool = True,
                 include_convertible: bool = False,
                 exclude_review: bool = True,
                 ):
        super().__init__(view_name=view_name,
                         logger_cfg=logger_cfg,
                         only_metadata=only_metadata,
                         include_convertible=include_convertible,
                         exclude_review=exclude_review,
                         )