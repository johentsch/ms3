#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
import pytest, os
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

def test_json_parse():
    paths = ['paths2.json', 'truth/metadata.tsv', 'paths1.json']
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    directory = 'test_results'
    goal = 72
    p = Parse(directory=directory, paths=paths, key='tests', logger_cfg=dict(level='d'))
    assert len(p.files['tests']) >= goal, f"Failed to parse list of paths that included several JSON files containing paths: Loaded only {len(p.files['tests'])} instead of {goal} or more."



@pytest.fixture
def parsed_mscx():
    p = Parse('.', file_re='mscx$', folder_re='MS3', logger_cfg=dict(level='d'))
    p.parse()
    return p


class TestParse:

    test_folder = os.path.dirname(os.path.realpath(__file__))

    def test_extract(self, parsed_mscx):
        target = resolve_dir('test_results')
        path_dict = parsed_mscx.store_lists(measures_folder=target, measures_suffix="_measures",
                                notes_folder=target, notes_suffix='_notes',
                                labels_folder=target, labels_suffix='_labels')
        # for path, what in path_dict.items():
        #     original_path = os.path.join(test_folder, what, )
