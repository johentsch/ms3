#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
import pytest
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


@pytest.fixture
def parsed_mscx():
    p = Parse('MS3', file_re='mscx$', logger_cfg=dict(level='d'))
    p.parse()
    return p


class TestParse:

    test_folder = os.path.dirname(os.path.realpath(__file__))

    def test_extract(self, parsed_mscx):
        target = resolve_dir('test_results')
        path_dict = parsed_mscx.store_lists(measures_folder=target, measures_suffix="_measures",
                                notes_folder=target, notes_suffix='_notes',
                                labels_folder=target, labels_suffix='_labels')
        for path, what in path_dict.items():
            original_path = os.path.join(test_folder, what, )
