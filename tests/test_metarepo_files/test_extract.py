from collections import defaultdict
from itertools import product

import pytest

DOCUMENTED_COLUMNS = {
    "measures": [
        "mc",
        "mn",
        "quarterbeats",
        "duration_qb",
        "keysig",
        "timesig",
        "act_dur",
        "mc_offset",
        "volta",
        "numbering_offset",
        "dont_count",
        "barline",
        "breaks",
        "repeats",
        "next",
    ],
    "notes": [
        "quarterbeats",
        "duration_qb",
        "mc",
        "mn",
        "mc_onset",
        "mn_onset",
        "timesig",
        "staff",
        "voice",
        "duration",
        "gracenote",
        "tremolo",
        "nominal_duration",
        "scalar",
        "tied",
        "tpc",
        "midi",
        "volta",
        "chord_id",
    ],
    "notes_and_rests": [
        "quarterbeats",
        "duration_qb",
        "mc",
        "mn",
        "mc_onset",
        "mn_onset",
        "timesig",
        "staff",
        "voice",
        "duration",
        "gracenote",
        "tremolo",
        "nominal_duration",
        "scalar",
        "tied",
        "tpc",
        "midi",
        "volta",
        "chord_id",
    ],
    "rests": [
        "quarterbeats",
        "duration_qb",
        "mc",
        "mn",
        "mc_onset",
        "mn_onset",
        "timesig",
        "staff",
        "voice",
        "duration",
        "nominal_duration",
        "scalar",
        "volta",
    ],
    "labels": [
        "quarterbeats",
        "duration_qb",
        "mc",
        "mn",
        "mc_onset",
        "mn_onset",
        "timesig",
        "staff",
        "voice",
        "volta",
        "harmony_layer",
        "label",
        "offset_x",
        "offset_y",
        "regex_match",
    ],
    "expanded": [
        "quarterbeats",
        "duration_qb",
        "mc",
        "mn",
        "mc_onset",
        "mn_onset",
        "timesig",
        "staff",
        "voice",
        "volta",
        "label",
        "alt_label",
        "offset_x",
        "offset_y",
        "regex_match",
        "globalkey",
        "localkey",
        "pedal",
        "chord",
        "numeral",
        "form",
        "figbass",
        "changes",
        "relativeroot",
        "cadence",
        "phraseend",
        "chord_type",
        "globalkey_is_minor",
        "localkey_is_minor",
        "chord_tones",
        "added_tones",
        "root",
        "bass_note",
    ],
    "events": [],
    "chords": [
        "quarterbeats",
        "tremolo",
        "duration_qb",
        "mc",
        "mn",
        "mc_onset",
        "mn_onset",
        "timesig",
        "staff",
        "voice",
        "duration",
        "gracenote",
        "nominal_duration",
        "scalar",
        "volta",
        "chord_id",
        "dynamics",
        "articulation",
        "staff_text",
        "slur",
        "Ottava:8va",
        "Ottava:8vb",
        "pedal",
        "TextLine",
        "crescendo_line",
        "diminuendo_line",
        "crescendo_hairpin",
        "decrescendo_hairpin",
        "tempo",
        "qpm",
        "lyrics:1",
        "Ottava:15mb",
    ],
    "metadata": [],
    "form_labels": [],
}
"""Columns that are represented in the respective functions' docstrings."""


# region test helpers


def inspect_object(obj, no_magic=True):
    result = {}
    for attr in dir(obj):
        if no_magic and attr[:2] == "__":
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


class TestNameCollection:
    # def test_collect_parse_obj_names(self, parse_obj, collect_test_name):
    #     """Run this if the parametrization of Parse has changed and you need to update TEST_NAMES."""
    #     print('\ntest_names = [')
    #     for name in sorted(TEST_NAMES['test_collect_parse_obj_names']):
    #         print(f" '{name}',")
    #     print("]")

    def test_collect_parsed_parse_obj_names(self, parsed_parse_obj, collect_test_name):
        """Run this if the parametrization of Parse has changed and you need to update TEST_NAMES."""
        print("\nname2expected.update({")
        for name in sorted(TEST_NAMES["test_collect_parsed_parse_obj_names"]):
            print(f" {name}:" + " {},")
        print("})")


# endregion test helpers


@pytest.mark.usefixtures("parse_objects")
class TestEmptyParse:
    """Tests Parse objects where no files have been parsed yet."""

    @pytest.fixture()
    def expected_keys(self, request):
        name2expected = defaultdict(dict)
        name2expected.update(
            dict(
                regex={
                    "mixed_files": {},
                    "outputs": {},
                    "ravel_piano": {},
                    "sweelinck_keyboard": {},
                    "wagner_overtures": {".mscx": 2},
                },
                everything={
                    "mixed_files": {
                        ".mscx": 22,
                        ".mscz": 1,
                        ".musicxml": 1,
                        ".mxl": 1,
                        ".xml": 1,
                    },
                    "outputs": {".mscx": 1},
                    "ravel_piano": {".mscx": 8, ".tsv": 21},
                    "sweelinck_keyboard": {".mscx": 2, ".tsv": 5},
                    "wagner_overtures": {".mscx": 4, ".tsv": 10},
                },
                # file_re_without_key = {'outputs': {'.mscx': 1, '.tsv': 3},
                #                        'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4}},
                # without_metadata = {'orchestral': {'.mscx': 3}},
                # redundant = {'classic': {'.mscx': 1, '.mscz': 1}},
                # regular_dirs = {'ravel_piano': {'.mscx': 5, '.tsv': 14},
                #                  'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
                #                  'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
                # regular_dirs_at_once = {'ravel_piano': {'.mscx': 5, '.tsv': 14},
                #                  'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
                #                  'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
                # chaotic_dirs = {'chamber': {'.mscx': 1},
                #                 'keyboard': {'.mscx': 5, '.mscz': 1, '.tsv': 1},
                #                 'orchestral': {'.mscx': 3},
                #                 'outputs': {'.mscx': 1, '.tsv': 3}},
                # files_without_key = {'unittest_metacorpus': {'.mscx': 3}},
                # files_with_inferred_key = {'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4}},
                # files_with_wrong_key = {'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
                #                         'unittest_metacorpus': {'.mscx': 3}},
                # files_correct_without_metadata = {'frankenstein': {'.mscx': 2, '.tsv': 6}},
                # files_with_correct_key = {'sweelinck_keyboard': {'.mscx': 2, '.tsv': 7}},
            )
        )
        name = node_name2cfg_name(request.node.name)
        return name2expected[name]

    @pytest.fixture()
    def expected_tsv_lines(self, request):
        name2expected = defaultdict(lambda: 0)
        name2expected.update(
            dict(
                regex=8,
                everything=8,
                # chaotic_dirs = 5,
                # file_re = 1,
                # file_re_without_key = 1,
                # files_with_correct_key = 1,
                # files_with_inferred_key = 1,
                # files_with_wrong_key = 1,
                # regular_dirs = 8,
                # regular_dirs_at_once = 8
            )
        )
        name = node_name2cfg_name(request.node.name)
        return name2expected[name]

    def test_keys(self, expected_keys, caplog):
        # Check for particular log message types in particular test cases
        # if name == 'chaotic_dirs':
        #     assert any(record._message_type == 7 for record in caplog.records), f"Expected a {MessageType(7)}. Got
        #     {caplog.records}."
        # if name == 'files_without_key':
        #     assert any(record._message_type == 8 for record in caplog.records), print(f"Expected a {MessageType(
        #     7)}. Got {caplog.records}.")
        assert self.parse_obj.count_extensions() == expected_keys

    def test_metadata(self, expected_tsv_lines):
        p = self.parse_obj
        for name, corpus in p:
            if corpus.metadata_tsv is not None:
                print(name)
                print(corpus.metadata_tsv)
        metadata_tsv_lines = sum(
            len(corpus.metadata_tsv)
            for _, corpus in p
            if corpus.metadata_tsv is not None
        )
        assert metadata_tsv_lines == expected_tsv_lines

    def test_empty(self):
        assert self.parse_obj.n_parsed_scores == 0
        assert self.parse_obj.n_parsed_tsvs == 0


@pytest.mark.usefixtures("parsed_parse_objects")
class TestParsedParse:
    """Test Parse objects containing either parsed MSCX, parsed TSV, or both."""

    @pytest.fixture()
    def n_parsed_files(self, request):
        expected = {  # (n_scores, n_tsvs)
            "regex": (2, 0),
            "everything": (38, 36),
            "chaotic_dirs": (11, 4),
            "file_re_without_key": (2, 7),
            "files_correct_without_metadata": (2, 6),
            "files_with_correct_key": (2, 7),
            "files_with_wrong_key": (4, 4),
            "files_with_inferred_key": (1, 4),
            "files_without_key": (3, 0),
            "redundant": (2, 0),
            "regular_dirs": (8, 25),
            "regular_dirs_at_once": (8, 25),
            "without_metadata": (3, 0),
        }
        name2expected = {}  # defaultdict(lambda: (0,0))
        parse_modes = "parsed_all-", "parse_scores-", "parsed_tsv-"
        mode2selector = dict(
            zip(parse_modes, ((True, True), (True, False), (False, True)))
        )

        def mode_filter(expected_pair, selector):
            (n_mscx, n_tsv), (a, b) = expected_pair, selector
            return (n_mscx if a else 0, n_tsv if b else 0)

        name2expected.update(
            {
                parse_mode
                + test_mode: mode_filter(expected_pair, mode2selector[parse_mode])
                for parse_mode, (test_mode, expected_pair) in product(
                    ["parsed_all-", "parse_scores-", "parsed_tsv-"], expected.items()
                )
            }
        )
        name = node_name2cfg_name(request.node.name)
        return name2expected[name]

    def test_n_parsed(self, n_parsed_files):
        p = self.parsed_parse_obj
        n_parsed_scores = p.n_parsed_scores
        n_parsed_tsvs = p.n_parsed_tsvs
        parsed_files = (n_parsed_scores, n_parsed_tsvs)
        assert parsed_files == n_parsed_files

    def test_extracting_concatenated_dataframes(self):
        p = self.parsed_parse_obj
        if p.n_parsed_scores == 0:
            return
        dataframes = dict(
            measures=p.extract_facets("measures", flat=True),
            notes=p.extract_facets("notes", flat=True),
            notes_and_rests=p.extract_facets("notes_and_rests", flat=True),
            rests=p.extract_facets("rests", flat=True),
            labels=p.extract_facets("labels", flat=True),
            expanded=p.extract_facets("expanded", flat=True),
            chords=p.extract_facets("chords", flat=True),
            form_labels=p.extract_facets("form_labels", flat=True),
        )

        for typ, df in dataframes.items():
            if len(df) == 0:
                continue
            print("\n\n", typ.upper())
            print(df)
            for col in ("quarterbeats", "duration_qb"):
                if col not in df and ("", col) not in df:
                    print(f"{typ} is missing {col}".upper(), df)
                    assert False
            if "volta" in df.columns:
                if not df.quarterbeats.isna().any():
                    print(typ)
                    print(df.loc[df.volta.notna(), ["quarterbeats", "volta"]])
                    assert False
            print(df.columns.to_list())
            for col in df.columns:
                print(f"|{col}|,", end=" ")
            documented = DOCUMENTED_COLUMNS[typ]
            missing = [col for col in df.columns if col not in documented]
            if len(missing) > 0:
                print(f"Columns missing in documentation: {missing}")
                for col in missing:
                    print(f"|{col}|,", end=" ")
                # assert False

    # def test_check(self, caplog, all_ignored_warnings):
    #     _ = self.parsed_parse_obj.get_dataframes(expanded=True)
    #     for logger_name, message_ids in all_ignored_warnings.items():
    #         eligible_records = [record for record in caplog.records if record.name == logger_name]
    #         if len(eligible_records) == 0:
    #             print(set(record.name for record in caplog.records))
    #             assert False
    #         for record in eligible_records:
    #             assert record._message_id not in message_ids


def test_fixture(get_all_warnings_parsed):
    print(get_all_warnings_parsed)
    assert len(get_all_warnings_parsed) > 0


# add_dir (different keys)
# file_re
