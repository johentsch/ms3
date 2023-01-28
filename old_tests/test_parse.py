#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
import pytest, os
import pandas as pd
from ms3 import Parse, assert_dfs_equal, load_tsv, resolve_dir

def test_folder_parse():
    p = Parse(".")
    assert isinstance(p, Parse), "Failed to parse folder."

def test_paths_parse():
    here = Path(resolve_dir('.'))
    exts = [".mscx", ".tsv"]
    paths = [str(p) for p in here.glob("**/*") if p.suffix in exts]
    p = Parse(file_paths=paths, logger_cfg=dict(level='d'))
    assert isinstance(p, Parse), "Failed to parse list of paths."

def test_json_parse():
    paths = ['paths2.json', 'truth/metadata.tsv', 'paths1.json']
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    directory = 'test_results'
    goal = 72
    p = Parse(directory=directory, file_paths=paths, key='old_tests', logger_cfg=dict(level='d'))
    assert len(p.files['old_tests']) >= goal, f"Failed to parse list of paths that included several JSON files containing paths: Loaded only {len(p.files['old_tests'])} instead of {goal} or more."



@pytest.fixture
def parsed_mscx():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    ms3_dir = os.path.dirname(test_dir)
    p = Parse(ms3_dir, file_re='mscx$', folder_re='MS3', logger_cfg=dict(level='d'))
    p.parse()
    return p


class TestParse:

    test_folder = os.path.dirname(os.path.realpath(__file__))
    test_results = os.path.join(test_folder, 'test_results')

    def test_extract(self, parsed_mscx):
        target = self.test_results
        path_dict = parsed_mscx.store_extracted_facets(notes_folder=target, notes_suffix='_notes', measures_folder=target, measures_suffix="_measures", labels_folder=target,
                                                       labels_suffix='_labels')
        # for path, what in path_dict.items():
        #     original_path = os.path.join(test_folder, what, )

    def test_metadata_extract(self, parsed_mscx):
        target_path = os.path.join(self.test_results, 'metadata.tsv')
        old = load_tsv(target_path, dtype='string')
        new = parsed_mscx.metadata().reset_index(drop=True).astype('string').replace('', pd.NA)
        assert_dfs_equal(old, new, exclude=['rel_paths'])

