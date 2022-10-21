from typing import TypeAlias, Dict, List, Tuple, Literal, Union, Collection

import pandas as pd

from . import Score
from .utils import File

CorpusFnameTuple = Tuple[str, str]
FileDict: TypeAlias = Dict[str, File]
FileList: TypeAlias = List[File]
Category: TypeAlias = Literal['corpora',
                              'folders',
                              'fnames',
                              'files',
                              'suffixes',
                              'facets'
                                ]
Categories: TypeAlias = Union[Category, Collection[Category]]
ParsedFile: TypeAlias = Union[Score, pd.DataFrame]
