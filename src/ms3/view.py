import os
import random
import re
from collections import Counter, defaultdict
from copy import deepcopy
from typing import (
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt

from ._typing import Categories, Category, FileList
from .logger import LoggedClass
from .score import Score
from .utils import File, resolve_dir, resolve_paths_argument, unpack_json_paths


def empty_counts():
    """Array for counting kept items, discarded items, and their sum."""
    return np.zeros(3, dtype=int)


class View(LoggedClass):
    """
    Object storing regular expressions and filter lists, storing and keeping track of things filtered out.
    """

    _deprecated_elements = ("fnames_with_incomplete_facets",)
    review_regex = "review"
    categories = (
        "corpora",
        "folders",
        "pieces",
        "files",
        "suffixes",
        "facets",
        "paths",
    )
    available_facets = ("scores",) + Score.dataframe_types + ("unknown",)
    singular2category: Dict[str, Category] = dict(
        zip(
            ("corpus", "folder", "piece", "file", "suffix", "facet", "path"), categories
        )
    )
    tsv_regex = re.compile(r"\.tsv$", re.IGNORECASE)
    convertible_regex = Score.make_extension_regex(
        native=False, convertible=True, tsv=False
    )
    registered_regexes = (convertible_regex, review_regex, tsv_regex)

    def __init__(
        self,
        view_name: Optional[str] = "all",
        only_metadata_pieces: bool = False,
        include_convertible: bool = True,
        include_tsv: bool = True,
        exclude_review: bool = False,
        **logger_cfg,
    ):
        super().__init__(subclass="View", logger_cfg=logger_cfg)
        # fields
        self._name: str = ""
        # the two main dicts
        self.including: dict = {c: [] for c in self.categories}
        self.excluding: dict = {c: [] for c in self.categories}
        self.excluded_file_paths: List[str] = []
        self.selected_facets = self.available_facets
        self._last_filtering_counts: Dict[
            str, npt.NDArray[int, int, int]
        ] = defaultdict(empty_counts)
        """For each filter method, store the counts of the last run as [n_kept, n_discarded, N (the sum)].
        Keys are "category" for :meth:`filter_by_token` and 'files' or 'parsed' for :meth:`filtered_file_list`.
        To inspect, you can use the method :meth:`filtering_report`
        """
        self._discarded_items: Dict[str, Set[str]] = defaultdict(set)
        self._discarded_file_criteria: dict[
            Literal["subdir", "file", "suffix", "path"], Counter
        ] = defaultdict(Counter)
        """{criterion -> {excluded_name -> n_excluded}} dict for keeping track of which file was discarded based on
        which criterion.
        """
        # booleans
        self.pieces_in_metadata: bool = True
        self.pieces_not_in_metadata: bool = not only_metadata_pieces
        self.pieces_with_incomplete_facets = True
        # properties
        self.name = view_name
        self.include_convertible = include_convertible
        self.include_tsv = include_tsv
        self.exclude_review = exclude_review
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%% END of __init__() %%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    @staticmethod
    def check_name(view_name) -> Tuple[bool, str]:
        if not isinstance(view_name, str):
            return (
                False,
                f"Name of the view should be a string, not '{type(view_name)}'",
            )
        if not view_name.isidentifier():
            return False, f"The string '{view_name}' cannot be used as attribute name."
        return True, ""

    @property
    def fnames_with_incomplete_facets(self):
        raise DeprecationWarning(
            "'fnames_with_incomplete_facets' was renamed to  'pieces_with_incomplete_facets' in "
            "ms3 v2."
        )

    @fnames_with_incomplete_facets.setter
    def fnames_with_incomplete_facets(self, value):
        raise DeprecationWarning(
            "'fnames_with_incomplete_facets' was renamed to  'pieces_with_incomplete_facets' in "
            "ms3 v2."
        )

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str):
        name_valid, msg = self.check_name(new_name)
        if not name_valid:
            raise ValueError(msg)
        self._name = new_name

    @property
    def only_metadata_pieces(self) -> bool:
        return self.pieces_in_metadata and not self.pieces_not_in_metadata

    @only_metadata_pieces.setter
    def only_metadata_pieces(self, value):
        self.pieces_in_metadata = True
        self.pieces_not_in_metadata = False

    def is_default(self, relax_for_cli: bool = False) -> bool:
        """Checks includes and excludes that may influence the selection of pieces. Returns True if the settings
        do not filter out any pieces. Only if ``relax_for_cli`` is set to True, the filters :attr:`include_convertible`
        and :attr:`exclude_review` are permitted, too."""
        # define the expected number of filter regexes per category (ignore 'corpora' and 'facets')
        default_excluding_lengths = {
            "suffixes": 0,
            "folders": 0,
            "pieces": 0,
            "files": 0,
            "paths": 0,
        }
        if relax_for_cli:
            if self.exclude_review:
                default_excluding_lengths.update(
                    {"folders": 1, "pieces": 1, "files": 1}
                )
            default_excluding_lengths["files"] += not self.include_convertible
        # # debugging:
        #         print(f"""no includes: {all(len(self.including[category]) == 0 for category in
        #         default_excluding_lengths.keys())}
        # default_excludes: {all(len(self.excluding[category]) == expected for category, expected in
        # default_excluding_lengths.items())}
        # exclude_review: {not self.exclude_review or relax_for_cli}
        # include_convertible: {self.include_convertible or relax_for_cli}
        # no paths excluded: {len(self.excluded_file_paths) == 0}
        # pieces in metadata: {self.pieces_in_metadata}
        # not in metadata excluded: {self.pieces_not_in_metadata or relax_for_cli}
        # incomplete facets: {self.pieces_with_incomplete_facets}""")
        return (
            all(
                len(self.including[category]) == 0
                for category in default_excluding_lengths.keys()
            )
            and all(
                len(self.excluding[category]) == expected
                for category, expected in default_excluding_lengths.items()
            )
            and len(self.excluded_file_paths) == 0
            and self.pieces_in_metadata
            and self.pieces_with_incomplete_facets
            and (
                relax_for_cli
                or (self.include_convertible and self.pieces_not_in_metadata)
            )
        )

    def copy(self, new_name: Optional[str] = None) -> "View":
        """Returns a copy of this view, i.e., a new View object."""
        if new_name is None:
            new_name = get_ferocious_name()
        new_view = self.__class__(view_name=new_name)
        new_view.including = deepcopy(self.including)
        new_view.excluding = deepcopy(self.excluding)
        new_view.update_facet_selection()
        new_view.excluded_file_paths = list(self.excluded_file_paths)
        new_view.pieces_in_metadata = self.pieces_in_metadata
        new_view.pieces_not_in_metadata = self.pieces_not_in_metadata
        new_view.pieces_with_incomplete_facets = self.pieces_with_incomplete_facets
        return new_view

    def update_config(
        self,
        view_name: Optional[str] = None,
        only_metadata_pieces: Optional[bool] = None,
        include_convertible: Optional[bool] = None,
        include_tsv: Optional[bool] = None,
        exclude_review: Optional[bool] = None,
        file_paths: Optional[Union[str, Collection[str]]] = None,
        file_re: Optional[str] = None,
        folder_re: Optional[str] = None,
        exclude_re: Optional[str] = None,
        folder_paths: Optional[Union[str, Collection[str]]] = None,
        **logger_cfg,
    ):
        """Update the configuration of the View. This is a shorthand for issuing several calls to :meth:`include` and
        :meth:`exclude` at once.

        Args:
          view_name: New name of the view.
          only_metadata_pieces: Whether or not pieces that are not included in a metadata.tsv should be excluded.
          include_convertible: Whether or not scores that need conversion via MuseScore before parsing should be
          included.
          include_tsv: Whether or not TSV files should be included.
          exclude_review: Whether or not files and folder that include 'review' should be excluded.
          file_paths:
              The exact file names will be extracted and used as exclusive filter, that is, all files that do not have
              one of these file names will be excluded. This is regardless of eventual relative or absolute paths
              included
              in the argument.
          file_re: Include only files whose file name includes this regular expression.
          folder_re: Include only files from folders whose name includes this regular expression.
          exclude_re: Exclude all file and folders whose name includes this regular expression.
          folder_paths: Include only files from these folders.
          **logger_cfg:

        Returns:

        """
        for param, value in zip(
            (
                "view_name",
                "only_metadata_pieces",
                "include_convertible",
                "include_tsv",
                "exclude_review",
            ),
            (
                view_name,
                only_metadata_pieces,
                include_convertible,
                include_tsv,
                exclude_review,
            ),
        ):
            if value is None:
                continue
            old_value = getattr(self, param)
            if value != old_value:
                setattr(self, param, value)
                self.logger.debug(f"Set '{param}' (previously {old_value}) to {value}.")
        if file_re is not None and file_re != ".*":
            self.include("files", file_re)
        if folder_re is not None and folder_re != ".*":
            self.include("folders", folder_re)
        if exclude_re is not None:
            self.exclude(("files", "folders"), exclude_re)
        if file_paths is not None:
            resolved_paths = resolve_paths_argument(file_paths)
            if len(resolved_paths) > 0:
                unpack_json_paths(resolved_paths)
                regexes = [re.escape(os.path.basename(p)) for p in resolved_paths]
                self.include("files", *regexes)
        if folder_paths is not None:
            resolved_paths = resolve_paths_argument(folder_paths, files=False)
            if len(resolved_paths) > 0:
                self.include("paths", *resolved_paths)
        if len(logger_cfg) > 0:
            self.change_logger_cfg(**logger_cfg)

    @property
    def include_convertible(self):
        return self.convertible_regex not in self.excluding["files"]

    @include_convertible.setter
    def include_convertible(self, yes: bool):
        if yes:
            self.unexclude("files", self.convertible_regex)
        else:
            self.exclude("files", self.convertible_regex)

    @property
    def include_tsv(self):
        return self.tsv_regex not in self.excluding["files"]

    @include_tsv.setter
    def include_tsv(self, yes: bool):
        if yes:
            self.unexclude("files", self.tsv_regex)
        else:
            self.exclude("files", self.tsv_regex)

    @property
    def exclude_review(self):
        return all(
            self.review_regex in self.excluding[what_to_exclude]
            for what_to_exclude in ("files", "pieces", "folders")
        )

    @exclude_review.setter
    def exclude_review(self, yes: bool):
        if yes:
            self.exclude(("files", "pieces", "folders"), self.review_regex)
        else:
            self.unexclude(("files", "pieces", "folders"), self.review_regex)

    def check_token(self, category: Category, token: str) -> bool:
        """Checks if a string pertaining to a certain category should be included in the view or not."""
        category = self.resolve_category(category)
        if category == "paths":
            path = resolve_dir(token)
            if os.path.isfile(path):
                path = os.path.dirname(path)
            if any(
                path.startswith(excluded_path)
                for excluded_path in self.excluding["paths"]
            ):
                return False
            if len(self.including["paths"]) == 0:
                return True
            return any(
                path.startswith(included_path)
                for included_path in self.including["paths"]
            )
        if any(re.search(rgx, token) is not None for rgx in self.excluding[category]):
            return False
        if len(self.including[category]) == 0:
            return True
        return any(
            re.search(rgx, token) is not None for rgx in self.including[category]
        )

    def check_file(self, file: File) -> Tuple[bool, str]:
        """Check if an individual File passes all filters w.r.t. its subdirectories, file name and suffix.

        Args:
          file:

        Returns:
          False if file is to be discarded from this view.
          The criterion based on which the file is being excluded.
        """
        if file.full_path in self.excluded_file_paths:
            return False, "file"
        if not self.check_token("paths", file.directory):
            return False, "directory"
        category2file_component = dict(
            zip(
                (("folders", "subdir"), ("files", "file"), ("suffixes", "suffix")),
                (file.subdir, file.file, file.suffix),
            )
        )
        for (category, criterion), component in category2file_component.items():
            if any(
                re.search(rgx, component) is not None
                for rgx in self.excluding[category]
            ):
                return False, criterion
        for (category, criterion), component in category2file_component.items():
            if len(self.including[category]) == 0:
                continue
            if not any(
                re.search(rgx, component) is not None
                for rgx in self.including[category]
            ):
                return False, criterion
        return True, "files"

    def reset_filtering_data(self, categories: Categories = None):
        if categories is None:
            # reset everything
            self._last_filtering_counts = defaultdict(empty_counts)
            self._discarded_items = defaultdict(set)
            self._discarded_file_criteria = defaultdict(Counter)
        else:
            categories = self.resolve_categories(categories)
            for ctgr in categories:
                if ctgr in self._last_filtering_counts:
                    del self._last_filtering_counts[ctgr]
                if ctgr in self._discarded_items:
                    del self._discarded_items[ctgr]
            if "files" in categories:
                self._discarded_file_criteria = defaultdict(Counter)
        self.update_facet_selection()

    def reset_view(self):
        self.__init__()

    def filter_by_token(
        self, category: Category, tuples: Iterable[tuple]
    ) -> Iterator[tuple]:
        """Filters out those tuples where the token (first element) does not pass _.check_token(category, token)."""
        category = self.resolve_category(category)
        n_kept, n_discarded, N = 0, 0, 0
        discarded_items = []
        for tup in tuples:
            N += 1
            token, *_ = tup
            if self.check_token(category, token=token):
                n_kept += 1
                yield tup
            else:
                n_discarded += 1
                discarded_items.append(token)
        key = category
        self._last_filtering_counts[key] += np.array(
            [n_kept, n_discarded, N], dtype="int"
        )
        self._discarded_items[key].update(discarded_items)

    def filtered_tokens(self, category: Category, tokens: Collection[str]) -> List[str]:
        """Applies :meth:`filter_by_token` to a collection of tokens."""
        return [
            token[0] for token in self.filter_by_token(category, ((t,) for t in tokens))
        ]

    def filtered_file_list(self, files: Collection[File], key: str = None) -> FileList:
        """Keep only the files that pass _.check_file().

        Args:
          files: :obj:`File` objects to be filtered.
          key: Aggregate results from several filter runs under this dictionary key.

        Returns:

        """
        if len(files) == 0:
            return []
        result, discarded_items = [], []
        for file in files:
            accept, criterion = self.check_file(file)
            if accept:
                result.append(file)
            else:
                discarded_items.append(file.rel_path)
                if key is None:
                    # do not track discarding criteria for special keys such as 'parsed', used by View.iter_facet2parsed
                    self._discarded_file_criteria[criterion][
                        getattr(file, criterion)
                    ] += 1
        n_kept, n_discarded, N = len(result), len(discarded_items), len(files)
        if key is None:
            key = "files"
        self._last_filtering_counts[key] += np.array(
            [n_kept, n_discarded, N], dtype="int"
        )
        self._discarded_items[key].update(discarded_items)
        return result

    def filtering_report(
        self, drop_zero=True, show_discarded=True, return_str=False
    ) -> Optional[str]:
        aggregated_counts = defaultdict(empty_counts)
        for key, counts in self._last_filtering_counts.items():
            aggregated_counts[key] += counts
        if show_discarded:
            discarded = defaultdict(list)
            for key, items in self._discarded_items.items():
                discarded[key].extend(items)
        msg = ""
        for key, (_, n_discarded, N) in aggregated_counts.items():
            if not drop_zero or n_discarded > 0:
                msg += f"{n_discarded}/{N} {key} are excluded from this view"
                if show_discarded:
                    if len(discarded[key]) > 0:
                        msg += f":\n{sorted(discarded[key])}\n\n"
                    else:
                        msg += ", but unfortunately I don't know which ones.\n"
                else:
                    msg += ".\n"
        if len(self._discarded_file_criteria) > 0:
            msg += "\n"
            for criterion, cntr in self._discarded_file_criteria.items():
                crit = "file name" if criterion == "file" else criterion
                msg += f"{sum(cntr.values())} files have been excluded based on their {crit}"
                if show_discarded:
                    msg += ":\n"
                    for excluded_name, n in cntr.items():
                        msg += f"\t- '{excluded_name}': {n}\n"
                else:
                    msg += ".\n"
        if return_str:
            return msg
        print(msg)

    def info(self, return_str=False):
        msg_components = []
        if self.pieces_in_metadata + self.pieces_not_in_metadata == 0:
            msg = (
                f"This view is called '{self.name}'. It excludes everything because both its attributes "
                f"pieces_in_metadata and pieces_not_in_metadata are set to False."
            )
            if return_str:
                return msg
            print(msg)
            return
        if not self.pieces_in_metadata:
            msg_components.append("excludes pieces that are contained in the metadata")
        if not self.pieces_not_in_metadata:
            msg_components.append(
                "excludes pieces that are not contained in the metadata"
            )
        if not self.include_convertible:
            msg_components.append(
                "filters out file extensions requiring conversion (such as .xml)"
            )
        if not self.include_tsv:
            msg_components.append("disregards all TSV files")
        if self.exclude_review:
            msg_components.append("excludes review files and folders")
        included_re = {
            what_to_include: [
                rgx for rgx in regexes if rgx not in self.registered_regexes
            ]
            for what_to_include, regexes in self.including.items()
        }
        excluded_re = {
            what_to_exclude: [
                rgx for rgx in regexes if rgx not in self.registered_regexes
            ]
            for what_to_exclude, regexes in self.excluding.items()
        }
        for what_to_include, re_strings in included_re.items():
            n_included = len(re_strings)
            if n_included == 0:
                continue
            if n_included == 1:
                included = f"'{re_strings[0]}'"
            elif n_included < 11:
                included = "one of " + str(re_strings)
            else:
                included = (
                    "one of ["
                    + ", ".join(f"'{regex}'" for regex in re_strings[:10])
                    + "... "
                )
                included += f" ({n_included - 10} more, see filtering_report()))"
            msg_components.append(
                f"includes only {what_to_include} containing {included}"
            )
        for what_to_exclude, re_strings in excluded_re.items():
            n_excluded = len(re_strings)
            if n_excluded == 0:
                continue
            if n_excluded == 1:
                excluded = f"'{re_strings[0]}'"
            elif n_excluded < 11:
                excluded = "one of " + str(re_strings)
            else:
                excluded = (
                    "one of ["
                    + ", ".join(f"'{regex}'" for regex in re_strings[:10])
                    + "... "
                )
                excluded += f" ({n_excluded - 10} more, see filtering_report())"
            msg_components.append(
                f"excludes any {what_to_exclude} containing {excluded}"
            )
        if not self.pieces_with_incomplete_facets:
            msg_components.append(
                f"excludes pieces that do not have at least one file per selected facet ("
                f"{', '.join(self.selected_facets)})"
            )
        if len(self.excluded_file_paths) > 0:
            msg_components.append(
                f"excludes {len(self.excluded_file_paths)} files based on user input"
            )
        msg = f"This view is called '{self.name}'. It "
        n_components = len(msg_components)
        if n_components == 0:
            msg += "selects everything."
        elif n_components == 1:
            msg += msg_components[0] + "."
        else:
            separator = "\n\t- "
            msg += separator + ("," + separator).join(msg_components[:-1])
            msg += f", and{separator}{msg_components[-1]}."
        if return_str:
            return msg
        print(msg)

    def resolve_category(self, category: Category) -> Category:
        if isinstance(category, str):
            if category not in self.categories:
                if category in self.singular2category:
                    return self.singular2category[category]
                else:
                    raise ValueError(
                        f"'{category}' is not one of the known categories {self.categories}"
                    )
            return category
        else:
            raise ValueError(
                f"Pass a single category string ∈ {self.categories}, not a '{type(category)}'"
            )

    def resolve_categories(self, categories: Categories) -> List[str]:
        if isinstance(categories, str):
            categories = [categories]
        return [self.resolve_category(categ) for categ in categories]

    def update_facet_selection(self):
        selected, discarded = [], []
        for facet in self.available_facets:
            if self.check_token("facet", facet):
                selected.append(facet)
            else:
                discarded.append(facet)
        self.selected_facets = selected
        key = "facets"
        if len(discarded) == 0:
            if key in self._last_filtering_counts:
                del self._last_filtering_counts[key]
            if key in self._discarded_items:
                del self._discarded_items[key]
            return
        n_kept, n_discarded = len(selected), len(discarded)
        counts = np.array([n_kept, n_discarded, n_kept + n_discarded])
        self._last_filtering_counts[key] = counts
        self._discarded_items[key] = set(discarded)

    def include(self, categories: Categories, *regex: Union[str, re.Pattern]):
        categories = self.resolve_categories(categories)
        if "paths" in categories:
            paths = [resolve_dir(rgx) for rgx in regex]
        for what_to_include in categories:
            regex_or_paths = paths if what_to_include == "paths" else regex
            for rgx in regex_or_paths:
                if rgx not in self.including[what_to_include]:
                    self.including[what_to_include].append(rgx)
            if what_to_include == "facets":
                self.update_facet_selection()

    def exclude(self, categories: Categories, *regex: Union[str, re.Pattern]):
        categories = self.resolve_categories(categories)
        if "paths" in categories:
            paths = [resolve_dir(rgx) for rgx in regex]
        for what_to_exclude in categories:
            regex_or_paths = paths if what_to_exclude == "paths" else regex
            for rgx in regex_or_paths:
                if rgx not in self.excluding[what_to_exclude]:
                    self.excluding[what_to_exclude].append(rgx)
            if what_to_exclude == "facets":
                self.update_facet_selection()

    def uninclude(self, categories: Categories, *regex: Union[str, re.Pattern]):
        categories = self.resolve_categories(categories)
        if "paths" in categories:
            paths = [resolve_dir(rgx) for rgx in regex]
        for what_to_uninclude in categories:
            regex_or_paths = paths if what_to_uninclude == "paths" else regex
            for rgx in regex_or_paths:
                try:
                    self.including[what_to_uninclude].remove(rgx)
                except ValueError:
                    pass

    def unexclude(self, categories: Categories, *regex: Union[str, re.Pattern]):
        categories = self.resolve_categories(categories)
        if "paths" in categories:
            paths = [resolve_dir(rgx) for rgx in regex]
        for what_to_unexclude in categories:
            regex_or_paths = paths if what_to_unexclude == "paths" else regex
            for rgx in regex_or_paths:
                try:
                    self.excluding[what_to_unexclude].remove(rgx)
                except ValueError:
                    pass

    def __repr__(self):
        return self.info(return_str=True)


class DefaultView(View):
    def __init__(
        self,
        view_name: Optional[str] = "default",
        only_metadata_pieces: bool = True,
        include_convertible: bool = False,
        include_tsv: bool = True,
        exclude_review: bool = True,
        **logger_cfg,
    ):
        super().__init__(
            view_name=view_name,
            only_metadata_pieces=only_metadata_pieces,
            include_convertible=include_convertible,
            include_tsv=include_tsv,
            exclude_review=exclude_review,
            **logger_cfg,
        )

    def is_default(self, relax_for_cli: bool = False) -> bool:
        default_excluding_lengths = {
            "folders": 1,
            "pieces": 1,
            "files": 2,
            "suffixes": 0,
        }
        if relax_for_cli:
            default_excluding_lengths["files"] -= self.include_convertible
        # # debugging:
        #         print(f"""no includes: {all(len(self.including[category]) == 0 for category in
        #         default_excluding_lengths.keys())}
        # default_excludes: {all(len(self.excluding[category]) == expected for category, expected in
        # default_excluding_lengths.items())}
        # exclude_review: {self.exclude_review}
        # include_convertible: {not self.include_convertible or relax_for_cli}
        # no paths excluded: {len(self.excluded_file_paths) == 0}
        # pieces in metadata: {self.pieces_in_metadata}
        # not in metadata excluded: {not self.pieces_not_in_metadata or relax_for_cli}
        # incomplete facets: {self.pieces_with_incomplete_facets}""")
        return (
            all(
                len(self.including[category]) == 0
                for category in default_excluding_lengths.keys()
            )
            and all(
                len(self.excluding[category]) == expected
                for category, expected in default_excluding_lengths.items()
            )
            and len(self.excluded_file_paths) == 0
            and self.pieces_in_metadata
            and self.pieces_with_incomplete_facets
            and (
                relax_for_cli
                or (not self.include_convertible and not self.pieces_not_in_metadata)
            )
        )


def create_view_from_parameters(
    only_metadata_pieces: bool = True,
    include_convertible: bool = False,
    include_tsv: bool = True,
    exclude_review: bool = True,
    file_paths=None,
    file_re=None,
    folder_re=None,
    exclude_re=None,
    level=None,
) -> View:
    """From the arguments of an __init__ method, create either a DefaultView or a custom view."""
    no_legacy_params = all(
        param is None for param in (file_paths, file_re, folder_re, exclude_re)
    )
    all_default = (
        only_metadata_pieces
        and include_tsv
        and exclude_review
        and not include_convertible
    )
    if no_legacy_params and all_default:
        return DefaultView(level=level)
    ferocious_name = get_ferocious_name()
    view = View(
        ferocious_name,
        only_metadata_pieces=only_metadata_pieces,
        include_convertible=include_convertible,
        include_tsv=include_tsv,
        exclude_review=exclude_review,
        level=level,
    )
    view.update_config(
        file_paths=file_paths,
        file_re=file_re,
        folder_re=folder_re,
        exclude_re=exclude_re,
    )
    return view


def get_ferocious_name():
    path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "ferocious_names.txt"
    )
    return random.choice(open(path, "r", encoding="utf-8").readlines()).strip("\n")
