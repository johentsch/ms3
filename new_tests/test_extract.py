import logging

import pytest
from collections import defaultdict

from ms3.logger import MessageType, LEVELS


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
    def expected_keys(self, request, caplog):
        name2expected = defaultdict(dict)
        name2expected.update(dict(
            everything = {'mixed_files': {'.mscx': 9, '.mscz': 1, '.tsv': 1},
                             'outputs': {'.mscx': 1, '.tsv': 3},
                             'ravel_piano': {'.mscx': 5, '.tsv': 14},
                             'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
                             'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
            file_re_with_key = {'sweelinck': {'.mscx': 2, '.tsv': 6}},
            file_re_without_key = {'outputs': {'.mscx': 1, '.tsv': 3},
                                   'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4}},
            without_metadata = {'orchestral': {'.mscx': 3}},
            redundant = {'classic': {'.mscx': 1, '.mscz': 1}},
            regular_dirs = {'ravel_piano': {'.mscx': 5, '.tsv': 14},
                             'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
                             'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
            chaotic_dirs = {'keyboard': {'.mscx': 4, '.mscz': 1, '.tsv': 1},
                             'orchestral': {'.mscx': 3},
                             'outputs': {'.mscx': 1, '.tsv': 3}},
            files_without_key = {},
            files_with_inferred_key = {'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4}},
            files_with_wrong_key = {'sweelinck_keyboard': {'.mscx': 4, '.tsv': 4}},
            files_correct_without_metadata = {'frankenstein': {'.mscx': 2, '.tsv': 6}},
            files_with_correct_key = {'sweelinck_keyboard': {'.mscx': 2, '.tsv': 7}},
        ))
        name = node_name2cfg_name(request.node.name)
        ## Check for particular log message types in particular test cases
        if name == 'chaotic_dirs':
            assert any(record._message_type == 7 for record in caplog.records), f"Expected a {MessageType(7)}. Got {caplog.records}."
        if name == 'files_without_key':
            assert any(record._message_type == 8 for record in caplog.records), f"Expected a {MessageType(7)}. Got {caplog.records}."
        return name2expected[name]

    @pytest.fixture()
    def expected_tsv_lines(self, request):
        name2expected = defaultdict(lambda: 0)
        name2expected.update(dict(
            chaotic_dirs = 5,
            everything = 13,
            #file_re = 1,
            file_re_without_key = 1,
            files_with_correct_key = 1,
            files_with_inferred_key = 1,
            files_with_wrong_key = 1,
            regular_dirs = 8
        ))
        name = node_name2cfg_name(request.node.name)
        return name2expected[name]

    def test_keys(self, expected_keys):
        assert self.parse_obj.count_extensions(per_key=True) == expected_keys

    def test_metadata(self, expected_tsv_lines):
        p = self.parse_obj
        metadata_tsv_lines = sum(len(p[k].metadata()) for k in p.keys())
        p._parsed_tsv = {}
        assert metadata_tsv_lines == expected_tsv_lines
        assert len(p.metadata()) == 0

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
            "parsed_all-chaotic_dirs": {'keyboard': {'.mscx': 4, '.mscz': 1, '.tsv': 1},
                                         'orchestral': {'.mscx': 3},
                                         'outputs': {'.mscx': 1, '.tsv': 3}},
            "parsed_mscx-chaotic_dirs": {'keyboard': {'.mscx': 4, '.mscz': 1, '.tsv': 1},
                                         'orchestral': {'.mscx': 3},
                                         'outputs': {'.mscx': 1, '.tsv': 3}},
            "parsed_tsv-chaotic_dirs": {'keyboard': {'.mscx': 4, '.mscz': 1, '.tsv': 1},
                                         'orchestral': {'.mscx': 3},
                                         'outputs': {'.mscx': 1, '.tsv': 3}},
            "parsed_all-everything": {'mixed_files': {'.mscx': 9, '.mscz': 1, '.tsv': 1},
                                         'outputs': {'.mscx': 1, '.tsv': 3},
                                         'ravel_piano': {'.mscx': 5, '.tsv': 14},
                                         'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
                                         'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
            "parsed_mscx-everything": {'mixed_files': {'.mscx': 9, '.mscz': 1, '.tsv': 1},
                                         'outputs': {'.mscx': 1, '.tsv': 3},
                                         'ravel_piano': {'.mscx': 5, '.tsv': 14},
                                         'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
                                         'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
            "parsed_tsv-everything": {'mixed_files': {'.mscx': 9, '.mscz': 1, '.tsv': 1},
                                         'outputs': {'.mscx': 1, '.tsv': 3},
                                         'ravel_piano': {'.mscx': 5, '.tsv': 14},
                                         'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
                                         'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
            # "parsed_all-file_re": {'sweelinck': {'.mscx': 2, '.tsv': 7}},
            # "parsed_mscx-file_re": {'sweelinck': {'.mscx': 2, '.tsv': 7}},
            # "parsed_tsv-file_re": {'sweelinck': {'.mscx': 2, '.tsv': 7}},
            "parsed_all-file_re_with_key": {'sweelinck': {'.mscx': 2, '.tsv': 6}},
            "parsed_mscx-file_re_with_key": {'sweelinck': {'.mscx': 2, '.tsv': 6}},
            "parsed_tsv-file_re_with_key": {'sweelinck': {'.mscx': 2, '.tsv': 6}},
            "parsed_all-file_re_without_key": {'outputs': {'.mscx': 1, '.tsv': 3}, 'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4}},
            "parsed_mscx-file_re_without_key": {'outputs': {'.mscx': 1, '.tsv': 3}, 'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4}},
            "parsed_tsv-file_re_without_key": {'outputs': {'.mscx': 1, '.tsv': 3}, 'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4}},
            "parsed_all-files_correct_without_metadata": {'frankenstein': {'.mscx': 2, '.tsv': 6}},
            "parsed_mscx-files_correct_without_metadata": {'frankenstein': {'.mscx': 2, '.tsv': 6}},
            "parsed_tsv-files_correct_without_metadata": {'frankenstein': {'.mscx': 2, '.tsv': 6}},
            "parsed_all-files_with_inferred_key": {'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4}},
            "parsed_mscx-files_with_inferred_key": {'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4}},
            "parsed_tsv-files_with_inferred_key": {'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4}},
            "parsed_all-files_with_correct_key": {'sweelinck_keyboard': {'.mscx': 2, '.tsv': 7}},
            "parsed_mscx-files_with_correct_key": {'sweelinck_keyboard': {'.mscx': 2, '.tsv': 7}},
            "parsed_tsv-files_with_correct_key": {'sweelinck_keyboard': {'.mscx': 2, '.tsv': 7}},
            "parsed_all-files_with_wrong_key": {'sweelinck_keyboard': {'.mscx': 4, '.tsv': 4}},
            "parsed_mscx-files_with_wrong_key": {'sweelinck_keyboard': {'.mscx': 4, '.tsv': 4}},
            "parsed_tsv-files_with_wrong_key": {'sweelinck_keyboard': {'.mscx': 4, '.tsv': 4}},
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
            "parsed_all-without_metadata": {'orchestral': {'.mscx': 3}},
            "parsed_mscx-without_metadata": {'orchestral': {'.mscx': 3}},
            "parsed_tsv-without_metadata": {'orchestral': {'.mscx': 3}},
        })
        name = node_name2cfg_name(request.node.name)
        return name2expected[name]

    @pytest.fixture()
    def n_parsed_files(self, request):
        name2expected = defaultdict(lambda: (0,0))
        name2expected.update({
            "parsed_all-chaotic_dirs": (9, 4),
            "parsed_mscx-chaotic_dirs": (9, 0),
            "parsed_tsv-chaotic_dirs": (0, 4),
            "parsed_all-file_re_with_key": (2, 6),
            "parsed_mscx-file_re_with_key": (2, 0),
            "parsed_tsv-file_re_with_key": (0, 6),
            "parsed_all-file_re_without_key": (2, 7),
            "parsed_mscx-file_re_without_key": (2, 0),
            "parsed_tsv-file_re_without_key": (0, 7),
            "parsed_all-files_correct_without_metadata": (2, 6),
            "parsed_mscx-files_correct_without_metadata": (2, 0),
            "parsed_tsv-files_correct_without_metadata": (0, 6),
            "parsed_all-files_with_correct_key": (2, 7),
            "parsed_mscx-files_with_correct_key": (2, 0),
            "parsed_tsv-files_with_correct_key": (0, 7),
            "parsed_all-files_with_wrong_key": (4, 4),
            "parsed_mscx-files_with_wrong_key": (4, 0),
            "parsed_tsv-files_with_wrong_key": (0, 4),
            "parsed_all-files_with_inferred_key": (1, 4),
            "parsed_mscx-files_with_inferred_key": (1, 0),
            "parsed_tsv-files_with_inferred_key": (0, 4),
            "parsed_all-everything": (19, 29),
            "parsed_mscx-everything": (19, 0),
            "parsed_tsv-everything": (0, 29),
            "parsed_all-redundant": (2, 0),
            "parsed_mscx-redundant": (2, 0),
            "parsed_all-regular_dirs": (8, 25),
            "parsed_mscx-regular_dirs": (8, 0),
            "parsed_tsv-regular_dirs": (0, 25),
            "parsed_all-without_metadata": (3, 0),
            "parsed_mscx-without_metadata": (3, 0),
        })
        name = node_name2cfg_name(request.node.name)
        return name2expected[name]

    @pytest.fixture()
    def get_ignored_warnings(self):
        return {"ms3.Parse.sweelinck": [(9, 'SwWV258_fantasia_cromatica')],"ms3.Parse.wagner_overtures.WWV090_Tristan_01_Vorspiel-Prelude_Ricordi1888Floridia.mscx": [(6, 87, 'V64(6b5)')],
                "ms3.Parse.ravel_piano.Ravel_-_Miroirs_III_Une_Barque_sur_l'ocean.mscx": [(3, 45)], "ms3.Parse.ravel_piano.Ravel_-_Miroirs_II_Oiseaux_tristes.mscx": [(3, 17), (1, 14, 16, 18, 20, 25, 28)],
                "ms3.Parse.mixed_files.Did03M-Son_regina-1762-Sarti.mscx": [(2, 94)], "ms3.Parse.mixed_files.BWV_0815.mscx": [(1, 1, 40, 85, 97, 131, 139)]}

    def test_keys(self, expected_keys):
        assert self.parsed_parse_obj.count_extensions(per_key=True) == expected_keys

    def test_n_parsed(self, n_parsed_files):
        p = self.parsed_parse_obj
        parsed_files = (len(p._parsed_mscx), len(p._parsed_tsv))
        assert parsed_files == n_parsed_files

    def test_loggers_level(self, level="D"):
        p = self.parsed_parse_obj
        _ = p.get_dataframes(expanded=True)
        p.change_logger_cfg()
        for logger_name in p.logger_names.values():
            current_logger_level = logging.getLogger(logger_name).level
            assert current_logger_level == LEVELS[level], f"Logger {logger_name} has level {current_logger_level}, not {LEVELS[level]}"

    def test_check(self, caplog, get_ignored_warnings):
        _ = self.parsed_parse_obj.get_dataframes(expanded=True)
        for record in caplog.records:
            if record.name in get_ignored_warnings:
                if record._message_id in get_ignored_warnings[record.name]:
                    assert record.levelname == "DEBUG", f"IGNORED_WARNINGS not filtered for logger {record.name}"

# add_dir (different keys)
# file_re

