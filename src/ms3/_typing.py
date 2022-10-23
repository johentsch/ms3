from typing import TypeAlias, Dict, List, Tuple, Literal, Union, Collection, TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from .score import Score
    from .utils import File
    from .view import View

CorpusFnameTuple = Tuple[str, str]
FileDict: TypeAlias = Dict[str, 'File']
FileList: TypeAlias = List['File']
Category: TypeAlias = Literal['corpora',
                              'folders',
                              'fnames',
                              'files',
                              'suffixes',
                              'facets'
                                ]
Categories: TypeAlias = Union[Category, Collection[Category]]
ParsedFile: TypeAlias = Union['Score', pd.DataFrame]
FileParsedTuple: TypeAlias = Tuple[Optional['File'], Optional[ParsedFile]]
FileScoreTuple: TypeAlias = Tuple[Optional['File'], Optional['Score']]
FileDataframeTuple: TypeAlias = Tuple[Optional['File'], Optional[pd.DataFrame]]
Facet: TypeAlias = Literal[
    'scores',
    'measures',
    'notes',
    'rests',
    'notes_and_rests',
    'labels',
    'expanded',
    'form_labels',
    'cadences',
    'events',
    'chords',
    'unknown',
]
"""All score facets, including the score itself and parsed TSV files of unknown type."""
FacetArgument: TypeAlias = Union[Facet, Literal['tsv', 'tsvs']]
"""Strings that can be used to identify a :obj:`Facet`, including shortcuts for specifying groups."""
ScoreFacet: TypeAlias = Literal[
    'measures',
    'notes',
    'rests',
    'notes_and_rests',
    'labels',
    'expanded',
    'form_labels',
    'cadences',
    'events',
    'chords',
]
"""All facets that can be extracted from a score, (but excluding metadata)."""
TSVtype: TypeAlias = Union[ScoreFacet, Literal['metadata', 'unknown']]
"""All available types a TSV file can be recognized as."""
MultipleFacets: TypeAlias = Collection[Facet]
MultipleFacetArguments: TypeAlias = Collection[FacetArgument]
MultipleScoreFacets: TypeAlias = Collection[ScoreFacet]
MultipleTSVtypes: TypeAlias = Collection[TSVtype]
Facets: TypeAlias = Union[Facet, MultipleFacets]
FacetArguments = Union[FacetArgument, MultipleFacetArguments]
ScoreFacets: TypeAlias = Union[ScoreFacet, MultipleScoreFacets]
TSVtypes = Union[TSVtype, MultipleTSVtypes]
ViewDict = Dict[str, 'View']