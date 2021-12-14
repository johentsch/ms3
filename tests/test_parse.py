#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

from ms3 import Parse, resolve_dir

def test_folder_parse():
    p = Parse(".")
    assert isinstance(p, Parse), "Failed to parse folder."

def test_paths_parse():
    here = Path(resolve_dir('.'))
    exts = [".mscx", ".tsv"]
    paths = [str(p) for p in here.glob("**/*") if p.suffix in exts]
    p = Parse(paths=paths, logger_cfg=dict(level='d'))
    assert isinstance(p, Parse), "Failed to parse list of paths."