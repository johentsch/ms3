# -*- coding: utf-8 -*-
"""
All functionality of the library is available through creating a ``ms3.Score`` object for a single score and a
``ms3.Parse`` object for multiple scores. Parsing a list of annotation labels only can be done by creating a
``ms3.Annotations`` object.
"""
import logging
import os
from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

version_file_path = os.path.join(os.path.dirname(__file__), "_version.py")
# store version in the "once canonical place" (https://stackoverflow.com/a/7071358);
# only rewrite when stale, and tolerate read-only installs — an unconditional write
# races when parallel processes import ms3 concurrently (e.g. pytest-xdist workers)
_version_line = f'__version__ = "{__version__}"'
try:
    with open(version_file_path) as f:
        _current = f.read()
except OSError:
    _current = None
if _current != _version_line:
    try:
        with open(version_file_path, "w") as f:
            f.write(_version_line)
    except OSError:
        pass
del _version_line

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
