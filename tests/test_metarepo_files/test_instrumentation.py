#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unittests for changing the instrumentation within MuseScore files.
Instrumentation is encoded in <Part> tags. Each <Part> includes exactly one <Instrument> tag and
one or several <Staff> tags which are assigned the same instrument. The relevant tags can be seen
in the TypedDict PartInfo.
"""
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, TypedDict

import bs4
import pytest
from git import Repo
from ms3 import Score
from ms3.bs4_parser import INSTRUMENT_DEFAULTS, Instrumentation

from ..conftest import TEST_COMMIT, UNITTEST_METACORPUS


@lru_cache()
def check_metarepo_commit(directory) -> str:
    repo = Repo(directory)
    commit = repo.commit()
    sha = commit.hexsha[: len(TEST_COMMIT)]
    if sha != TEST_COMMIT:
        print(f"Please checkout unittest_metarepo to {TEST_COMMIT}")
        assert sha == TEST_COMMIT
    return directory


# region test utilities


def get_source_target_paths() -> Dict[str, Tuple[str, str]]:
    """Returns a {file_name -> (source_path, target_path)} dictionary."""
    metarepo_path = check_metarepo_commit(os.path.expanduser(UNITTEST_METACORPUS))
    source_files_folder = os.path.join(metarepo_path, "mixed_files")

    target_files_folder = os.path.join(source_files_folder, "changed_instruments")
    target_files = os.listdir(target_files_folder)
    print(target_files)
    file2source_target_path = {}
    for path, subdirs, files in os.walk(source_files_folder):
        current_folder = os.path.basename(path)
        if current_folder.startswith(".") or current_folder == "changed_instruments":
            continue
        musescore_files = [f for f in files if f.endswith(".mscx")]
        for file in musescore_files:
            if file not in target_files:
                continue
            source_path = os.path.join(path, file)
            target_path = os.path.join(target_files_folder, file)
            file2source_target_path[file] = (source_path, target_path)
    assert (
        len(file2source_target_path) > 0
    ), f"Didn't find any relevant files at {target_files_folder}"
    return file2source_target_path


def get_tag_string_or_none(parent_tag: bs4.Tag, tag_name: str) -> Optional[str]:
    """Looks for a tag and, if found, returns its non-tag content as a string, otherwise none."""
    found_tag = parent_tag.find(tag_name)
    if found_tag is None:
        return
    if found_tag.string is None:
        return
    return found_tag.string.strip()


class PartInfo(TypedDict):
    """Relevant instrumentation info of one <Part> tag."""

    part_trackName: str
    staves: List[str]
    longName: str
    shortName: str
    trackName: str
    instrumentId: str


def get_soup(source_path: str) -> bs4.BeautifulSoup:
    """Parse MSCX file."""
    with open(source_path, "r", encoding="utf-8") as F:
        soup = bs4.BeautifulSoup(F.read(), "xml")
    return soup


def get_instrumentation(soup: bs4.BeautifulSoup) -> List[PartInfo]:
    parts = []
    for part_tag in soup.find_all("Part"):
        part_info = PartInfo(
            part_trackName=get_tag_string_or_none(part_tag, "trackName"),
            staves=[
                f"staff_{staff_tag['id']}" for staff_tag in part_tag.find_all("Staff")
            ],
        )
        instrument_tags = part_tag.find_all("Instrument")
        assert len(instrument_tags) == 1, (
            f"Expected exactly 1 <Instrument> tag but "
            f"the <Part> containing {part_info['staves']} contains {len(instrument_tags)}."
        )
        instrument = instrument_tags[0]
        for tag_name in ("longName", "shortName", "trackName", "instrumentId"):
            part_info[tag_name] = get_tag_string_or_none(instrument, tag_name)
        parts.append(part_info)
    return parts


def get_instrumentation_from_path(source_path: str) -> List[PartInfo]:
    """Get the full instrumentation info from a path to a MSCX file."""
    soup = get_soup(source_path)
    try:
        return get_instrumentation(soup)
    except AssertionError as e:
        raise AssertionError(f"Assertion error in {source_path}: {e}")


def part_info_without_staves(part_info: PartInfo) -> PartInfo:
    """Returns a copy of a PartInfo dictionary without the key 'staves'"""
    return PartInfo({k: v for k, v in part_info.items() if k != "staves"})


# endregion test utilities


# region pytest fixtures


@pytest.fixture(scope="session")
def source_instrumentation() -> Dict[str, List[PartInfo]]:
    """Reads instrumentation from all source files and returns it as {file_name -> List[TypedDict]} dictionary."""
    source_target_paths = get_source_target_paths()
    result = {}
    for file, (source_path, _) in source_target_paths.items():
        parts = get_instrumentation_from_path(source_path)
        result[file] = parts
    return result


@pytest.fixture()
def target_instrumentation() -> Dict[str, List[PartInfo]]:
    """Reads instrumentation from all source files and returns it as {file_name -> List[TypedDict]} dictionary."""
    source_target_paths = get_source_target_paths()
    result = {}
    for file, (_, target_path) in source_target_paths.items():
        parts = get_instrumentation_from_path(target_path)
        result[file] = parts
    return result


@pytest.fixture(
    scope="session",
    ids=get_source_target_paths().keys(),
    params=[source_path for (source_path, _) in get_source_target_paths().values()],
)
def source_path(request):
    """Get the path to one source MSCX file."""
    return request.param


# endregion pytest fixtures

# region tests


def test_accessing_source_instrument_names(source_path):
    """Test if the Instrumentation object returns the instrument name as it appears in metadata.tsv"""
    file_name = os.path.basename(source_path)
    soup = get_soup(source_path)
    parts = get_instrumentation(soup)
    staff2groundtruth = {}
    for part in parts:
        for staff_name in part["staves"]:
            assert (
                staff_name not in staff2groundtruth
            ), f"Competing instrument information for {staff_name} in {file_name}"
            if part["trackName"] is not None:
                # this if-clause corresponds to the current behaviour of bs4_parser.get_part_info
                staff2groundtruth[staff_name] = part["trackName"]
            else:
                staff2groundtruth[staff_name] = part["part_trackName"]
    tested_object = Instrumentation(soup=soup)
    for staff_name, instrument_name in staff2groundtruth.items():
        tested_value = tested_object.get_instrument_name(staff_name)
        assert tested_value == instrument_name


# {file_name: {(staff_to_modify, 'new_instrument'): {staff_to_test: 'expected_instrument'}
TEST_CASES = {
    "BWV_0815.mscx": {
        (1, "harpsichord"): {1: "harpsichord", 2: "harpsichord"},
        (2, "hch"): {1: "harpsichord", 2: "harpsichord"},
    },
    "K281-3.mscx": {
        (1, "drumset"): {1: "drumset", 2: "drumset"},
        (2, "Pnoo."): {1: "piano", 2: "piano"},
        (2, "Pno"): {1: "piano"},
    },
    "Brahms Op. 99iv.mscx": {
        (2, "pno"): {2: "piano", 3: "piano"},
        # (3, 'cello'): {3: 'violoncello'},
        (3, "violoncello"): {3: "violoncello"},
    },
}


def test_instrumentation_after_instrument_change(source_path):
    """For each file for which test cases have been defined in TEST_CASES, this test iterates through the
    cases, changes one instrument change (the one in the dictionary key), and tests if the resulting
    instrumentation corresponds to the dictionary value.
    """
    file_name = os.path.basename(source_path)
    if file_name not in TEST_CASES:
        pytest.skip(f"No test cases defined for {file_name}")
    for (staff_to_modify, new_instrument), staff_id2expected_instrument in TEST_CASES[
        file_name
    ].items():
        print(f"Creating new Instrumentation object from {source_path}...")
        soup = get_soup(source_path)  # re-parse everytime because soup is mutable
        tested_object = Instrumentation(soup=soup)
        print(f"INITIAL STATE: {tested_object}")
        print(f"TEST SETTING {staff_to_modify} TO {new_instrument!r}...")
        tested_object.set_instrument(staff_to_modify, new_instrument)
        expectation = {
            f"staff_{staff_id}": INSTRUMENT_DEFAULTS[
                tested_object.instrumentation_fields
            ]
            .loc[expected_instrument_name]
            .to_dict()
            for staff_id, expected_instrument_name in staff_id2expected_instrument.items()
        }
        parts = get_instrumentation(soup)
        test_results = {}
        for part in parts:
            print("PART", part)
            # actual_result = part_info_without_staves(part)
            actual_results = tested_object.fields
            for staff_name in part["staves"]:
                if staff_name not in expectation:
                    continue
                test_results[staff_name] = {
                    k: actual_results[staff_name][k]
                    for k in tested_object.instrumentation_fields
                }
        print(f"ASSERT: {test_results} == \n {expectation}")
        if test_results != expectation:
            print(
                f"Setting {staff_to_modify} to {new_instrument!r} did not result in the expected instrumentation."
            )
            assert test_results == expectation


def test_accessing_instrumentation_after_instrument_change(source_path):
    """Analogous to test_instrumentation_after_instrument_change but using a Score object."""
    file_name = os.path.basename(source_path)
    if file_name not in TEST_CASES:
        pytest.skip(f"No test cases defined for {file_name}")
    for (staff_to_modify, new_instrument), staff_id2expected_instrument in TEST_CASES[
        file_name
    ].items():
        print(f"Creating new Score object from {source_path}...")
        score = Score(source_path)
        tested_object = score.mscx.parsed.instrumentation
        print(f"INITIAL STATE: {tested_object}")
        print(f"TEST SETTING {staff_to_modify} TO {new_instrument!r}...")
        tested_object.set_instrument(staff_to_modify, new_instrument)
        expectation = {
            f"staff_{staff_id}": INSTRUMENT_DEFAULTS[
                tested_object.instrumentation_fields
            ]
            .loc[expected_instrument_name]
            .to_dict()
            for staff_id, expected_instrument_name in staff_id2expected_instrument.items()
        }
        test_results = {}
        for staff_name, actual_result in tested_object.fields.items():
            if staff_name not in expectation:
                continue
            test_results[staff_name] = {
                k: actual_result[k] for k in tested_object.instrumentation_fields
            }
        print(f"ASSERT: {test_results} == {expectation}")
        if test_results != expectation:
            print(
                f"Setting {staff_to_modify} to {new_instrument!r} did not result in the expected instrumentation."
            )
            assert test_results == expectation


# endregion tests

# region Accessories for developing tests

# # auxiliary function for getting instrument defaults
# def test_see_instrumentation(directory):
#     fp = os.path.join(directory, "mixed_files/changed_instruments/Brahms Op. 99iv.mscx")
#     instrumentation = get_instrumentation_from_path(fp)
#     print()
#     for part in instrumentation:
#         pprint(part_info_without_staves(part))
#
# # auxiliary function for getting complete source instrumentation
# def test_complete_source_instrumentation(source_instrumentation):
#     print(source_instrumentation)
#
#
# # status before instrument changes SOURCE_INSTRUMENTATION = { 'stabat_03_coloured.mscx': [ {'part_trackName':
# 'Soprano', 'staves': ['staff_1'], 'longName': 'Soprano', 'shortName': None, 'trackName': 'Soprano', 'instrumentId':
# 'voice.soprano'}, {'part_trackName': 'Alto', 'staves': ['staff_2'], 'longName': 'Alto', 'shortName': None,
# 'trackName': 'Alto', 'instrumentId': 'voice.alto'}, {'part_trackName': 'Piano', 'staves': ['staff_3', 'staff_4'],
# 'longName': 'Violino I/II\nViola\nVioloncello\nContrabasso\ne organo', 'shortName': None, 'trackName': 'Piano',
# 'instrumentId': 'keyboard.piano'}], '76CASM34A33UM.mscx': [ {'part_trackName': 'Piano', 'staves': ['staff_1',
# 'staff_2'], 'longName': 'Piano', 'shortName': 'Pno.', 'trackName': 'Piano', 'instrumentId':
# 'keyboard.piano.grand'}, {'part_trackName': 'Double Bass', 'staves': ['staff_3'], 'longName': 'Double Bass',
# 'shortName': 'Db.', 'trackName': 'Double Bass', 'instrumentId': 'pluck.bass.acoustic'}, {'part_trackName':
# 'Trommesæt', 'staves': ['staff_4'], 'longName': 'Trommesæt', 'shortName': 'D. Set', 'trackName': 'Trommesæt',
# 'instrumentId': 'drum.group.set'}], 'Brahms Op. 99iv.mscx': [{'part_trackName': 'Cello', 'staves': ['staff_1'],
# 'longName': 'Violoncello', 'shortName': 'Vc.', 'trackName': 'Violoncello', 'instrumentId': 'strings.cello'},
# {'part_trackName': 'MusicXML Part', 'staves': ['staff_2', 'staff_3'], 'longName': 'Piano', 'shortName': 'Pno.',
# 'trackName': 'Piano', 'instrumentId': 'keyboard.piano'}], '05_symph_fant.mscx': [ {'part_trackName': 'Flute',
# 'staves': ['staff_1'], 'longName': 'Flauto I.', 'shortName': 'Fl. I', 'trackName': 'Flute', 'instrumentId':
# 'wind.flutes.flute'}, {'part_trackName': 'Piccolo', 'staves': ['staff_2'], 'longName': 'Flauto piccolo.',
# 'shortName': 'Fl. picc.', 'trackName': 'Piccolo', 'instrumentId': 'wind.flutes.flute.piccolo'}, {'part_trackName':
# 'Oboe', 'staves': ['staff_3'], 'longName': '2 Oboi.', 'shortName': 'Ob.', 'trackName': 'Oboe', 'instrumentId':
# 'wind.reed.oboe'}, {'part_trackName': 'E♭ Clarinet', 'staves': ['staff_4'], 'longName': '\n\n\n\nI in Es (
# Mi♭).\n\n2 Clarinetti.              \n\nII in C (Ut).', 'shortName': 'Clar. I', 'trackName': 'E♭ Clarinet',
# 'instrumentId': 'wind.reed.clarinet.eflat'}, {'part_trackName': 'C Clarinet', 'staves': ['staff_5'], 'longName':
# None, 'shortName': 'Clar. II', 'trackName': 'C Clarinet', 'instrumentId': 'wind.reed.clarinet'}, {'part_trackName':
# 'Horn in E♭', 'staves': ['staff_6'], 'longName': '\n\n\n\nI e II in E (Mi♭)\n\n4 Corni.                   \n\nIII e
# IV in C (Ut).', 'shortName': 'Cor. I.II.', 'trackName': 'Horn in E♭', 'instrumentId': 'brass.natural-horn'},
# {'part_trackName': 'High C Horn', 'staves': ['staff_7'], 'longName': None, 'shortName': 'Cor. III.IV', 'trackName':
# 'High C Horn', 'instrumentId': 'brass.natural-horn'}, {'part_trackName': 'Bassoon', 'staves': ['staff_8'],
# 'longName': '\n\n\n\nI.II.\n\n4 Fagotti        \n\nIII.IV.', 'shortName': 'Fag. I.II.', 'trackName': 'Bassoon',
# 'instrumentId': 'wind.reed.bassoon'}, {'part_trackName': 'Bassoon', 'staves': ['staff_9'], 'longName': None,
# 'shortName': 'Fag. III.IV.', 'trackName': 'Bassoon', 'instrumentId': 'wind.reed.bassoon'}, {'part_trackName': 'E♭
# Trumpet', 'staves': ['staff_10'], 'longName': '2 Trombe in Es (Mi♭).', 'shortName': 'Tr.', 'trackName': 'E♭
# Trumpet', 'instrumentId': 'brass.trumpet'}, {'part_trackName': 'B♭ Cornet', 'staves': ['staff_11'], 'longName': '2
# Cornetti in B (Si♭).', 'shortName': 'Ctti.', 'trackName': 'B♭ Cornet', 'instrumentId': 'brass.cornet'},
# {'part_trackName': 'Trombone', 'staves': ['staff_12'], 'longName': 'Tromboni I e II.', 'shortName': 'Tromb. I.II.',
# 'trackName': 'Trombone', 'instrumentId': 'brass.trombone'}, {'part_trackName': 'Bass Trombone', 'staves': [
# 'staff_13'], 'longName': 'Trombone III.', 'shortName': 'Tromb. III.', 'trackName': 'Bass Trombone', 'instrumentId':
# 'brass.trombone.bass'}, {'part_trackName': 'Tuba', 'staves': ['staff_14'], 'longName': '2 Tube.', 'shortName':
# 'Tube.', 'trackName': 'Tuba', 'instrumentId': 'brass.tuba'}, {'part_trackName': 'Timpani', 'staves': ['staff_15'],
# 'longName': 'Timpani I    \nin H (Si) E (Mi)', 'shortName': 'Timp. I', 'trackName': 'Timpani', 'instrumentId':
# 'drum.timpani'}, {'part_trackName': 'Timpani', 'staves': ['staff_16'], 'longName': 'Timpani II        \nin Gis (
# Sol#) Cis (Ut#)', 'shortName': 'Timp. II', 'trackName': 'Timpani', 'instrumentId': 'drum.timpani'},
# {'part_trackName': 'Cymbal', 'staves': ['staff_17'], 'longName': 'Cinelli.', 'shortName': 'Cinelli.', 'trackName':
# 'Cymbal', 'instrumentId': 'metal.cymbal.crash'}, {'part_trackName': 'Concert Bass Drum', 'staves': ['staff_18'],
# 'longName': 'Gran Tamburo.', 'shortName': 'Gr. Tamb.', 'trackName': 'Concert Bass Drum', 'instrumentId':
# 'drum.bass-drum'}, {'part_trackName': 'Piano', 'staves': ['staff_19', 'staff_20'], 'longName': 'Campane',
# 'shortName': 'Camp.', 'trackName': 'Piano', 'instrumentId': 'keyboard.piano'}, {'part_trackName': 'Violins',
# 'staves': ['staff_21'], 'longName': None, 'shortName': 'Viol. I', 'trackName': 'Violins', 'instrumentId':
# 'strings.group'}, {'part_trackName': 'Violins', 'staves': ['staff_22'], 'longName': 'Violino I.', 'shortName':
# 'Viol. I\n(div. 1)', 'trackName': 'Violins', 'instrumentId': 'strings.group'}, {'part_trackName': 'Violins',
# 'staves': ['staff_23'], 'longName': None, 'shortName': 'Viol. I\n(div. 2)', 'trackName': 'Violins', 'instrumentId':
# 'strings.group'}, {'part_trackName': 'Violins', 'staves': ['staff_24'], 'longName': None, 'shortName': 'Viol. II',
# 'trackName': 'Violins', 'instrumentId': 'strings.group'}, {'part_trackName': 'Violins', 'staves': ['staff_25'],
# 'longName': 'Violino II.', 'shortName': 'Viol. II\n(div. 1)', 'trackName': 'Violins', 'instrumentId':
# 'strings.group'}, {'part_trackName': 'Violins', 'staves': ['staff_26'], 'longName': None, 'shortName': 'Viol. II\n(
# div. 2)', 'trackName': 'Violins', 'instrumentId': 'strings.group'}, {'part_trackName': 'Violas', 'staves': [
# 'staff_27'], 'longName': '\n\n\nViola.', 'shortName': 'Viola.', 'trackName': 'Violas', 'instrumentId':
# 'strings.group'}, {'part_trackName': 'Violas', 'staves': ['staff_28'], 'longName': None, 'shortName': 'Viola.\n(
# div. 1)', 'trackName': 'Violas', 'instrumentId': 'strings.group'}, {'part_trackName': 'Violoncellos', 'staves': [
# 'staff_29'], 'longName': 'Violoncello', 'shortName': 'Vcello.', 'trackName': 'Violoncellos', 'instrumentId':
# 'strings.group'}, {'part_trackName': 'Contrabasses', 'staves': ['staff_30'], 'longName': 'Contrabasso',
# 'shortName': 'C.B.', 'trackName': 'Contrabasses', 'instrumentId': 'strings.group'}],
# 'Did03M-Son_regina-1762-Sarti.mscx': [ {'part_trackName': 'Oboe I', 'staves': ['staff_1'], 'longName': 'Oboe I',
# 'shortName': 'Ob. I', 'trackName': 'SmartMusic SoftSynth 1', 'instrumentId': 'wind.reed.oboe'}, {'part_trackName':
# 'Oboe II', 'staves': ['staff_2'], 'longName': 'Oboe II', 'shortName': 'Ob. II', 'trackName': 'SmartMusic SoftSynth
# 2', 'instrumentId': 'wind.reed.oboe'}, {'part_trackName': 'Corno I en Re', 'staves': ['staff_3'], 'longName':
# 'Corno I en Re', 'shortName': 'Cor. I (Re)', 'trackName': 'SmartMusic SoftSynth 1', 'instrumentId':
# 'brass.french-horn'}, {'part_trackName': 'Corno II en Re', 'staves': ['staff_4'], 'longName': 'Corno II en Re',
# 'shortName': 'Cor. II (Re)', 'trackName': 'SmartMusic SoftSynth 2', 'instrumentId': 'brass.french-horn'},
# {'part_trackName': 'DIDONE', 'staves': ['staff_5'], 'longName': 'DIDONE', 'shortName': 'DID.', 'trackName':
# 'SmartMusic SoftSynth 1', 'instrumentId': 'voice.soprano'}, {'part_trackName': 'Violino I', 'staves': ['staff_6'],
# 'longName': 'Violino I', 'shortName': 'Vn. I', 'trackName': 'SmartMusic SoftSynth 1', 'instrumentId':
# 'strings.violin'}, {'part_trackName': 'Violino II', 'staves': ['staff_7'], 'longName': 'Violino II', 'shortName':
# 'Vn. II', 'trackName': 'SmartMusic SoftSynth 1', 'instrumentId': 'strings.violin'}, {'part_trackName': 'Viole',
# 'staves': ['staff_8'], 'longName': 'Viole', 'shortName': 'Ve.', 'trackName': 'SmartMusic SoftSynth 1',
# 'instrumentId': 'strings.viola'}, {'part_trackName': 'Bassi', 'staves': ['staff_9'], 'longName': 'Bassi',
# 'shortName': 'Bs.', 'trackName': 'SmartMusic SoftSynth 1', 'instrumentId': 'strings.cello'}], 'caldara_form.mscx':
# [ {'part_trackName': 'TIMANTE', 'staves': ['staff_1'], 'longName': 'TIMANTE', 'shortName': 'TIM.', 'trackName':
# 'SmartMusic SoftSynth 1', 'instrumentId': 'voice.soprano'}, {'part_trackName': 'Violino I', 'staves': ['staff_2'],
# 'longName': 'Violino I', 'shortName': 'Vn. I', 'trackName': 'SmartMusic SoftSynth 1', 'instrumentId':
# 'strings.violin'}, {'part_trackName': 'Violino II', 'staves': ['staff_3'], 'longName': 'Violino II', 'shortName':
# 'Vn. II', 'trackName': 'SmartMusic SoftSynth 1', 'instrumentId': 'strings.violin'}, {'part_trackName': 'Viola',
# 'staves': ['staff_4'], 'longName': 'Viola', 'shortName': 'Va.', 'trackName': 'SmartMusic SoftSynth 1',
# 'instrumentId': 'strings.viola'}, {'part_trackName': 'Basso', 'staves': ['staff_5'], 'longName': 'Basso',
# 'shortName': 'Bs.', 'trackName': 'SmartMusic SoftSynth 1', 'instrumentId': 'strings.cello'}], 'BWV_0815.mscx': [{
# 'part_trackName': 'Klavier linke Hand', 'staves': ['staff_1', 'staff_2'], 'longName': None, 'shortName': None,
# 'trackName': None, 'instrumentId': 'keyboard.piano'}], '12.16_Toccata_cromaticha_per_l’elevatione_phrygian.mscx': [
# {'part_trackName': 'Organ', 'staves': ['staff_1', 'staff_2'], 'longName': None, 'shortName': None, 'trackName':
# 'Organ', 'instrumentId': 'keyboard.organ'}], 'Tempest_1st.mscx': [{'part_trackName': 'Piano', 'staves': ['staff_1',
# 'staff_2'], 'longName': 'Piano', 'shortName': 'Pno.', 'trackName': 'Piano', 'instrumentId': 'keyboard.piano'}],
# 'D973deutscher01.mscx': [ {'part_trackName': 'Piano', 'staves': ['staff_1', 'staff_2'], 'longName': 'Piano',
# 'shortName': 'Pno.', 'trackName': 'Piano', 'instrumentId': 'keyboard.piano'}], 'K281-3.mscx': [{'part_trackName':
# 'Klavier linke Hand', 'staves': ['staff_1', 'staff_2'], 'longName': None, 'shortName': None, 'trackName': None,
# 'instrumentId': 'keyboard.piano'}]}

# endregion Accessories for developing tests
