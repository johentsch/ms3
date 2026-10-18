import logging
import os

import pytest
from ms3 import Parse, capture_parse_logs, first_level_subdirs
from ms3.logger import LogCaptureHandler, get_logger, iter_ms3_loggers

# a single, small piece emitting warnings, in both MS3 and MS4 format
SMALL_PIECE = dict(file_re="Oiseaux")


class TestEquivalence:
    def test_parallel(self, small_directory):
        """Parsing in parallel or iteratively yields the same files and log messages."""
        a = Parse(small_directory, level="i", **SMALL_PIECE)
        b = Parse(small_directory, level="i", **SMALL_PIECE)
        # the file_re filter creates a view with a random name
        a.view_name = b.view_name = "small_piece"
        with capture_parse_logs(a.logger) as captured_msgs:
            a.parse_scores(parallel=False)
            non_parallel_msgs = captured_msgs.content_list
        with capture_parse_logs(b.logger) as captured_msgs:
            b.parse_scores(parallel=True)
            parallel_msgs = captured_msgs.content_list
        assert a.info(return_str=True) == b.info(return_str=True)
        not_captured = [msg for msg in non_parallel_msgs if msg not in parallel_msgs]
        assert len(not_captured) == 0

    def test_add_corpus(self, directory):
        a = Parse(directory)
        b = Parse()
        c = Parse()
        b.add_corpus(directory)
        for sd in first_level_subdirs(directory):
            corpus = os.path.join(directory, sd)
            c.add_corpus(corpus)
        assert a.count_extensions() == c.count_extensions()
        assert len(b.count_extensions()) == 1


def is_ms3_handler(handler: logging.Handler) -> bool:
    """Since pytest 9, pytest attaches its own capturing handlers to all non-propagating loggers, which includes
    ms3's head loggers. Their levels are not ms3's business."""
    return isinstance(handler, LogCaptureHandler) or not type(
        handler
    ).__module__.startswith("_pytest")


def assert_all_loggers_level(level):
    for name, logger in iter_ms3_loggers():
        if not (name == "ms3.Parse" or name.startswith("ms3.Parse.")):
            continue
        eff_level = logger.getEffectiveLevel()
        if eff_level != level:
            head = get_logger("ms3.Parse")
            print(
                f"LOGGER '{logger.name}' SHOULD HAVE LEVEL {level}, not {eff_level}. ms3.Parse: {head}"
            )
        assert eff_level == level
        for h in logger.handlers:
            if not is_ms3_handler(h):
                continue
            h_level = h.level
            if h_level != level:
                print(
                    f"THE {h!r} of {logger.name} SHOULD HAVE LEVEL {level}, not {h_level}."
                )
            assert h_level == level


class RecordCollector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.mark.parametrize(
    "level, expected_level",
    [(None, 30), (0, 30), ("d", 10), ("i", 20), ("w", 30), ("e", 40), ("c", 50)],
    ids=["default", "notset", "debug", "info", "warning", "error", "critical"],
)
def test_logging_level(small_directory, level, expected_level):
    """All loggers involved in parsing have the requested level and no message below that level is emitted.
    Parsing iteratively because log messages from parallel workers are not passed on as LogRecords (equivalence with
    parallel parsing is covered by TestEquivalence.test_parallel)."""
    if level is None:
        p = Parse(small_directory, **SMALL_PIECE)
    else:
        p = Parse(small_directory, level=level, **SMALL_PIECE)
    collector = RecordCollector()
    p.logger.addHandler(collector)
    try:
        p.parse(parallel=False)
        _ = p.get_dataframes(expanded=True)
    finally:
        p.logger.removeHandler(collector)
    assert_all_loggers_level(expected_level)
    too_verbose = [
        f"{r.levelname}: {r.getMessage()}"
        for r in collector.records
        if r.levelno < expected_level
    ]
    assert too_verbose == []
    if expected_level <= 30:
        # make sure the corpus emits messages at all, otherwise the test would be meaningless
        assert any(r.levelno >= 30 for r in collector.records)


@pytest.mark.xdist_group("all_warnings")
class TestIgnoredWarnings:
    def test_ignoring_all_warnings(self, parse_with_all_warnings_ignored):
        assert parse_with_all_warnings_ignored["warnings"] == []

    def test_capturing_suppressed_warnings(
        self, get_all_warnings, parse_with_all_warnings_ignored
    ):
        """Every warning emitted without ignoring warnings shows up as an ignored message."""
        assert len(get_all_warnings) > 0
        suppressed = parse_with_all_warnings_ignored["ignored"]
        not_found = [msg for msg in get_all_warnings if msg not in suppressed]
        if len(not_found) > 0:
            print("NOT FOUND:\n" + "\n".join(not_found))
        assert not_found == []
