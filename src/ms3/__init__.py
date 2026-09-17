# -*- coding: utf-8 -*-
"""
All functionality of the library is available through creating a ``ms3.Score`` object for a single score and a
``ms3.Parse`` object for multiple scores. Parsing a list of annotation labels only can be done by creating a
``ms3.Annotations`` object.
"""
import logging

from ._version import __version__
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
