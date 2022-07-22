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

    # def test_collect_parse_obj_names(self, parse_obj, collect_test_name):
    #     """Run this if the parametrization of Parse has changed and you need to update TEST_NAMES."""
    #     print('\ntest_names = [')
    #     for name in sorted(TEST_NAMES['test_collect_parse_obj_names']):
    #         print(f" '{name}',")
    #     print("]")

    def test_collect_parsed_parse_obj_names(self, parsed_parse_obj, collect_test_name):
        """Run this if the parametrization of Parse has changed and you need to update TEST_NAMES."""
        print('\nname2expected.update(dict(')
        for name in sorted(TEST_NAMES['test_collect_parsed_parse_obj_names']):
            print(f"\t{name}" + " = {},")
        print("))")


################################## Actual tests ############################################



@pytest.mark.usefixtures("parse_objects")
class TestEmptyParse():

    @pytest.fixture()
    def expected_keys(self, request):
        name2expected = defaultdict(dict)
        name2expected.update(dict(
            everything = {'mixed_files': {'.mscx': 8, '.mscz': 1},
                         'outputs': {'.tsv': 3},
                         'ravel_piano': {'.mscx': 5, '.tsv': 14},
                         'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
                         'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
            regular_dirs = {'mixed_files': {'.mscx': 8, '.mscz': 1},
                             'outputs': {'.tsv': 3},
                             'ravel_piano': {'.mscx': 5, '.tsv': 14},
                             'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
                             'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
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

    @pytest.fixture()
    def expected_keys(self, request):
        name2expected = defaultdict(dict)
        name2expected.update(dict(
            parsed_mscx-regular_dirs = {'ravel_piano': {'.mscx': 5, '.tsv': 14},
E          'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
E          'wagner_overtures': {'.mscx': 2, '.tsv': 7}},

            everything={'mixed_files': {'.mscx': 8, '.mscz': 1},
                       'outputs': {'.tsv': 3},
                       'ravel_piano': {'.mscx': 5, '.tsv': 14},
                       'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
                       'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
            regular_dirs={'mixed_files': {'.mscx': 8, '.mscz': 1},
                            'outputs': {'.tsv': 3},
                            'ravel_piano': {'.mscx': 5, '.tsv': 14},
                            'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
                            'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
        ))
        name = node_name2cfg_name(request.node.name)
        return name2expected[name]

    def test_keys(self, expected_keys):
        assert self.parsed_parse_obj.count_extensions(per_key=True) == expected_keys


# add_dir (different keys)
# file_re

