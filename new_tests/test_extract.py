import pytest
from collections import defaultdict


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

    def test_collect_parse_obj_names(self, parse_obj, collect_test_name):
        """Run this if the parametrization of Parse has changed and you need to update TEST_NAMES."""
        print('\ntest_names = [')
        for name in sorted(TEST_NAMES['test_collect_parse_obj_names']):
            print(f" '{name}',")
        print("]")

    def test_collect_parsed_parse_obj_names(self, parsed_parse_obj, collect_test_name):
        """Run this if the parametrization of Parse has changed and you need to update TEST_NAMES."""
        print('\nname2expected = {')
        for name in sorted(TEST_NAMES['test_collect_parsed_parse_obj_names']):
            print(f" '{name}'" + ": {},")
        print("}")


################################## Actual tests ############################################



@pytest.mark.usefixtures("parse_objects")
class TestEmptyParse():

    test_names = [
        'multiple-all_paths',
        'multiple-directory',
        'multiple-directory+paths',
        'multiple-mscx_paths',
        'multiple-tsv_paths',
        'single-all_paths',
        'single-directory',
        'single-directory+paths',
        'single-mscx_paths',
        'single-tsv_paths',
    ]

    @pytest.fixture()
    def expected_keys(self, request):
        expected = [
            {'mixed_files': {'.mscx': 8,  # multiple-all_paths
                             '.mscz': 1,
                             '.musicxml': 1,
                             '.mxl': 1,
                             '.tsv': 1,
                             '.xml': 1},
             'ravel_piano': {'.mscx': 5, '.tsv': 14},
             'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
             'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
            {'mixed_files': {'.mscx': 8, '.mscz': 1, '.tsv': 1},  # multiple-directory
             'ravel_piano': {'.mscx': 5, '.tsv': 14},
             'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
             'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
            {'mixed_files': {'.mscx': 8,  # multiple-directory+paths
                             '.mscz': 1,
                             '.musicxml': 1,
                             '.mxl': 1,
                             '.tsv': 1,
                             '.xml': 1},
             'ravel_piano': {'.mscx': 5, '.tsv': 14},
             'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
             'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
            {'mixed_files': {'.mscx': 8},  # multiple-mscx_paths
             'ravel_piano': {'.mscx': 5},
             'sweelinck_keyboard': {'.mscx': 1},
             'wagner_overtures': {'.mscx': 2}},
            {'mixed_files': {'.tsv': 1},  # multiple-tsv_paths
             'ravel_piano': {'.tsv': 14},
             'sweelinck_keyboard': {'.tsv': 4},
             'wagner_overtures': {'.tsv': 7}},
            {'pleyel_quartets': {'.mscx': 6, '.tsv': 13}},  # single-all_paths
            {'pleyel_quartets': {'.mscx': 6, '.tsv': 13}},  # single-directory
            {'pleyel_quartets': {'.mscx': 6, '.tsv': 13}},  # single-directory+paths
            {'pleyel_quartets': {'.mscx': 6}},  # single-mscx_paths
            {'pleyel_quartets': {'.tsv': 13}},  # single-tsv_paths
        ]
        name2expected = dict(zip(self.test_names, expected))
        name = node_name2cfg_name(request.node.name)
        return name2expected[name]

    def test_keys(self, expected_keys):
        assert self.parse_obj.count_extensions(per_key=True) == expected_keys

    def test_empty(self):
        assert len(self.parse_obj._parsed_mscx) == 0
        assert len(self.parse_obj._parsed_tsv) == 0



@pytest.mark.usefixtures("parsed_parse_objects")
class TestParsedParse():

    @pytest.fixture()
    def expected_keys(self, request):
        name2expected = {
            'parsed_all-multiple-all_paths': {},
            'parsed_all-multiple-directory': {},
            'parsed_all-multiple-directory+paths': {},
            'parsed_all-multiple-mscx_paths': {},
            'parsed_all-multiple-tsv_paths': {},
            'parsed_all-single-all_paths': {},
            'parsed_all-single-directory': {},
            'parsed_all-single-directory+paths': {},
            'parsed_all-single-mscx_paths': {},
            'parsed_all-single-tsv_paths': {},
            'parsed_mscx-multiple-all_paths': {},
            'parsed_mscx-multiple-directory': {},
            'parsed_mscx-multiple-directory+paths': {},
            'parsed_mscx-multiple-mscx_paths': {},
            'parsed_mscx-multiple-tsv_paths': {},
            'parsed_mscx-single-all_paths': {},
            'parsed_mscx-single-directory': {},
            'parsed_mscx-single-directory+paths': {},
            'parsed_mscx-single-mscx_paths': {'pleyel_quartets': {'.mscx': 6}},
            'parsed_mscx-single-tsv_paths': {},
            'parsed_tsv-multiple-all_paths': {},
            'parsed_tsv-multiple-directory': {},
            'parsed_tsv-multiple-directory+paths': {},
            'parsed_tsv-multiple-mscx_paths': {},
            'parsed_tsv-multiple-tsv_paths': {},
            'parsed_tsv-single-all_paths': {},
            'parsed_tsv-single-directory': {},
            'parsed_tsv-single-directory+paths': {},
            'parsed_tsv-single-mscx_paths': {},
            'parsed_tsv-single-tsv_paths': {},
        }
        #name2expected = dict(zip_longest(self.test_names, expected, fillvalue={}))
        name = node_name2cfg_name(request.node.name)
        return name2expected[name]

    def test_keys(self, expected_keys):
        assert self.parsed_parse_obj.count_extensions(per_key=True) == expected_keys


# add_dir (different keys)
# file_re

