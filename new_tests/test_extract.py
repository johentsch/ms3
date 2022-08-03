import pytest
from collections import defaultdict
import os
from pathlib import Path

def inspect_object(obj, no_magic=True):
    result = {}
    for attr in dir(obj):
        if no_magic and attr[:2] == '__':
            continue
        result[attr] = obj.__getattribute__(attr)
    return result

def node_name2cfg_name(node_name):
    """Takes 'test_function[cfg_name]' and returns 'cfg_name'."""
    start_pos = node_name.index("[") + 1
    cfg_name = node_name[start_pos:-1]
    return cfg_name

TEST_NAMES = defaultdict(list)
@pytest.fixture()
def collect_test_name(request):
    """Helper for test_collect_cfg_names. Collects names in CFG_NAMES"""
    function_name = request.function.__name__
    test_name = node_name2cfg_name(request.node.name)
    TEST_NAMES[function_name].append(test_name)

class TestNameCollection():

    # def test_collect_parse_obj_names(self, parse_obj, collect_test_name):
    #     """Run this if the parametrization of Parse has changed and you need to update TEST_NAMES."""
    #     print('\ntest_names = [')
    #     for name in sorted(TEST_NAMES['test_collect_parse_obj_names']):
    #         print(f" '{name}',")
    #     print("]")

    def test_collect_parsed_parse_obj_names(self, parsed_parse_obj, collect_test_name):
        """Run this if the parametrization of Parse has changed and you need to update TEST_NAMES."""
        print('\nname2expected.update({')
        for name in sorted(TEST_NAMES['test_collect_parsed_parse_obj_names']):
            print(f" {name}:" + " {},")
        print("})")


################################## Actual tests ############################################



@pytest.mark.usefixtures("parse_objects")
class TestEmptyParse():
    """Tests Parse objects where no files have been parsed yet."""

    @pytest.fixture()
    def expected_keys(self, request):
        name2expected = defaultdict(dict)
        name2expected.update(dict(
            chaotic_dirs = {'mixed_files': {'.mscx': 8, '.mscz': 1, '.tsv': 1}, 'outputs': {'.mscx': 1, '.tsv': 3}},
            everything = {'mixed_files': {'.mscx': 8, '.mscz': 1, '.tsv': 1},
                             'outputs': {'.mscx': 1, '.tsv': 3},
                             'ravel_piano': {'.mscx': 5, '.tsv': 14},
                             'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
                             'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
            files_with_correct_key = {'sweelinck_keyboard': {'.mscx': 2, '.tsv': 7}},
            redundant = {'classic': {'.mscx': 1, '.mscz': 1}},
            regular_dirs = {'ravel_piano': {'.mscx': 5, '.tsv': 14},
                             'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
                             'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
            without_metadata = {'custom_key': {'.mscx': 4, '.mscz': 1}},
        ))
        name = node_name2cfg_name(request.node.name)
        return name2expected[name]

    def test_keys(self, expected_keys):
        assert self.parse_obj.count_extensions(per_key=True) == expected_keys

    def test_empty(self):
        assert len(self.parse_obj._parsed_mscx) == 0
        assert len(self.parse_obj._parsed_tsv) == 0



@pytest.mark.usefixtures("parsed_parse_objects")
class TestParsedParse():
    """Test Parse objects containing either parsed MSCX, parsed TSV, or both."""

    @pytest.fixture()
    def expected_keys(self, request):
        name2expected = defaultdict(dict)
        name2expected.update({
            "parsed_all-chaotic_dirs": {'mixed_files': {'.mscx': 8, '.mscz': 1, '.tsv': 1},
                                        'outputs': {'.mscx': 1, '.tsv': 3}},
            "parsed_mscx-chaotic_dirs": {'mixed_files': {'.mscx': 8, '.mscz': 1, '.tsv': 1},
                                         'outputs': {'.mscx': 1, '.tsv': 3}},
            "parsed_tsv-chaotic_dirs": {'mixed_files': {'.mscx': 8, '.mscz': 1, '.tsv': 1},
                                        'outputs': {'.mscx': 1, '.tsv': 3}},
            "parsed_all-everything": {'mixed_files': {'.mscx': 8, '.mscz': 1, '.tsv': 1},
                                         'outputs': {'.mscx': 1, '.tsv': 3},
                                         'ravel_piano': {'.mscx': 5, '.tsv': 14},
                                         'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
                                         'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
            "parsed_mscx-everything": {'mixed_files': {'.mscx': 8, '.mscz': 1, '.tsv': 1},
                                         'outputs': {'.mscx': 1, '.tsv': 3},
                                         'ravel_piano': {'.mscx': 5, '.tsv': 14},
                                         'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
                                         'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
            "parsed_tsv-everything": {'mixed_files': {'.mscx': 8, '.mscz': 1, '.tsv': 1},
                                         'outputs': {'.mscx': 1, '.tsv': 3},
                                         'ravel_piano': {'.mscx': 5, '.tsv': 14},
                                         'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
                                         'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
            "parsed_all-files_with_correct_key": {'sweelinck_keyboard': {'.mscx': 2, '.tsv': 7}},
            "parsed_mscx-files_with_correct_key": {'sweelinck_keyboard': {'.mscx': 2, '.tsv': 7}},
            "parsed_tsv-files_with_correct_key": {'sweelinck_keyboard': {'.mscx': 2, '.tsv': 7}},
            "parsed_all-redundant": {'classic': {'.mscx': 1, '.mscz': 1}},
            "parsed_mscx-redundant": {'classic': {'.mscx': 1, '.mscz': 1}},
            "parsed_tsv-redundant": {'classic': {'.mscx': 1, '.mscz': 1}},
            "parsed_all-regular_dirs": {'ravel_piano': {'.mscx': 5, '.tsv': 14},
                                         'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
                                         'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
            "parsed_mscx-regular_dirs": {'ravel_piano': {'.mscx': 5, '.tsv': 14},
                                         'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
                                         'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
            "parsed_tsv-regular_dirs": {'ravel_piano': {'.mscx': 5, '.tsv': 14},
                                         'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
                                         'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
            "parsed_all-without_metadata": {'custom_key': {'.mscx': 4, '.mscz': 1}},
            "parsed_mscx-without_metadata": {'custom_key': {'.mscx': 4, '.mscz': 1}},
            "parsed_tsv-without_metadata": {'custom_key': {'.mscx': 4, '.mscz': 1}},
        })
        name = node_name2cfg_name(request.node.name)
        return name2expected[name]

    @pytest.fixture()
    def n_parsed_files(self, request):
        name2expected = defaultdict(lambda: (0,0))
        name2expected.update({
            "parsed_all-chaotic_dirs": (10, 4),
            "parsed_mscx-chaotic_dirs": (10, 0),
            "parsed_tsv-chaotic_dirs": (0, 4),
            "parsed_all-files_with_correct_key": (2, 7),
            "parsed_mscx-files_with_correct_key": (2, 0),
            "parsed_tsv-files_with_correct_key": (0, 7),
            "parsed_all-everything": (18, 29),
            "parsed_mscx-everything": (18, 0),
            "parsed_tsv-everything": (0, 29),
            "parsed_all-redundant": (2, 0),
            "parsed_mscx-redundant": (2, 0),
            "parsed_all-regular_dirs": (8, 25),
            "parsed_mscx-regular_dirs": (8, 0),
            "parsed_tsv-regular_dirs": (0, 25),
            "parsed_all-without_metadata": (5, 0),
            "parsed_mscx-without_metadata": (5, 0),
        })
        name = node_name2cfg_name(request.node.name)
        return name2expected[name]

    @pytest.fixture()
    def get_ignored_warnings(self, filter_path='unittest_metacorpus/mixed_files'):
        # get warnings from file
        with open(os.path.join(str(Path.home()), filter_path, 'IGNORED_WARNINGS')) as f:
            ignored_messages = f.read()
        return ignored_messages

    def test_keys(self, expected_keys):
        assert self.parsed_parse_obj.count_extensions(per_key=True) == expected_keys

    def test_n_parsed(self, n_parsed_files):
        p = self.parsed_parse_obj
        parsed_files = (len(p._parsed_mscx), len(p._parsed_tsv))
        assert parsed_files == n_parsed_files

    def test_check(self, caplog, get_ignored_warnings):
        _ = self.parsed_parse_obj.get_dataframes(expanded=True)
        for record in caplog.records:
            if " ".join(record.getMessage().split(sep="--")[0].split(sep=" ")) in get_ignored_warnings:
                assert record.levelname == "DEBUG"

# add_dir (different keys)
# file_re

