# -*- coding: utf-8 -*-
"""
All functionality of the library is available through creating a ``ms3.Score`` object for a single score and a
``ms3.Parse`` object for multiple scores. Parsing a list of annotation labels only can be done by creating a
``ms3.Annotations`` object.
"""
import logging
import os
from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

import pandas as pd

# ms3 was written against pandas < 3.0, where text columns are NumPy ``object`` arrays.
# pandas 3.0 enables a dedicated string dtype by default (PDEP-14), which changes several
# behaviours ms3 relies on:
#   * constructed DataFrames get the immutable ``str`` dtype, which rejects in-place
#     assignment of non-string values (e.g. replacing duration strings with Fractions via
#     ``df.loc[:, col] = series.map(...)``);
#   * the nullable ``string`` dtype (from explicit ``.astype("string")``) becomes
#     pyarrow-backed under the "auto" storage, so boolean results of string comparisons
#     are pyarrow-backed and lack compute kernels ms3 uses (e.g. ``cumsum``).
# Restoring the pre-3.0 defaults keeps behaviour uniform across supported pandas versions.
# Both calls are guarded because the options do not exist on every supported pandas release
# (``future.infer_string`` was added in 2.1; on < 2.1 object strings are already the default).
for _option, _value in (
    ("future.infer_string", False),
    ("mode.string_storage", "python"),
):
    try:
        pd.set_option(_option, _value)
    except (
        KeyError,
        ValueError,
    ):  # pragma: no cover - option absent on this pandas version
        pass
del _option, _value

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

version_file_path = os.path.join(os.path.dirname(__file__), "_version.py")
# store version in the "once canonical place" (https://stackoverflow.com/a/7071358)
with open(version_file_path, "w") as f:
    f.write(f'__version__ = "{__version__}"')

from .annotations import Annotations
from .corpus import Corpus
from .logger import config_logger
from .operations import *
from .parse import Parse
from .piece import Piece
from .score import Score
from .transformations import *
from .utils import *

_ = config_logger("ms3", level="w")
logging.getLogger("git").setLevel(20)
