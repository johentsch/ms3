import os

from ms3 import Parse, capture_parse_logs, first_level_subdirs, ignored_warnings2dict
from ms3.logger import get_logger, iter_ms3_loggers


class TestEquivalence:
    def test_parallel(self, directory):
        a = Parse(directory)
        b = Parse(directory)
        a.parse_scores()
        b.parse_scores(parallel=False)
        assert a.info(return_str=True) == b.info(return_str=True)

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


def assert_all_loggers_level(level):
    for name, logger in iter_ms3_loggers():
        if not name.startswith("ms3.Parse"):
            continue
        eff_level = logger.getEffectiveLevel()
        if eff_level != level:
            head = get_logger("ms3.Parse")
            print(
                f"LOGGER '{logger.name}' SHOULD HAVE LEVEL {level}, not {eff_level}. ms3.Parse: {head}"
            )
        assert eff_level == level
        for h in logger.handlers:
            h_level = h.level
            if eff_level != level:
                print(
                    f"THE {h.__class__} of {logger.name} SHOULD HAVE LEVEL {level}, not {h_level}."
                )
            assert h_level == level


class TestLogging:
    def test_parallel_log_capture(self, small_directory):
        """Compare log messages emitted when parsing the same thing in parallel or iteratively."""
        a = Parse(small_directory, level="i")
        b = Parse(small_directory, level="i")
        with capture_parse_logs(a.logger) as captured_msgs:
            a.parse_scores(parallel=False)
            non_parallel_msgs = captured_msgs.content_list
        with capture_parse_logs(b.logger) as captured_msgs:
            b.parse_scores(parallel=True)
            parallel_msgs = captured_msgs.content_list
        not_captured = [msg for msg in non_parallel_msgs if msg not in parallel_msgs]
        assert len(not_captured) == 0

    def test_default(self, small_directory):
        p = Parse(small_directory)
        p.parse()
        _ = p.get_dataframes(expanded=True)
        assert_all_loggers_level(30)

    def test_debug(self, small_directory):
        p = Parse(small_directory, level="d")
        p.parse()
        _ = p.get_dataframes(expanded=True)
        assert_all_loggers_level(10)

    def test_info(self, small_directory):
        p = Parse(small_directory, level="i")
        with capture_parse_logs(p.logger) as captured_msgs:
            p.parse()
            _ = p.get_dataframes(expanded=True)
            all_msgs = captured_msgs.content_list
        assert_all_loggers_level(20)
        wrong_msgs = [msg for msg in all_msgs if msg.startswith("DEBUG")]
        if len(wrong_msgs) > 0:
            print("\n".join(wrong_msgs))
            assert False

    def test_warning(self, small_directory):
        p = Parse(small_directory, level="w")
        with capture_parse_logs(p.logger) as captured_msgs:
            p.parse()
            _ = p.get_dataframes(expanded=True)
            all_msgs = captured_msgs.content_list
        assert_all_loggers_level(30)
        wrong_msgs = [
            msg for msg in all_msgs if msg.startswith("INFO") or msg.startswith("DEBUG")
        ]
        if len(wrong_msgs) > 0:
            print("\n".join(wrong_msgs))
            assert False

    def test_error(self, small_directory):
        p = Parse(small_directory, level="e")
        with capture_parse_logs(p.logger) as captured_msgs:
            p.parse()
            _ = p.get_dataframes(expanded=True)
            all_msgs = captured_msgs.content_list
        assert_all_loggers_level(40)
        wrong_msgs = [msg for msg in all_msgs if "WARNING" in msg]
        if len(wrong_msgs) > 0:
            print("\n".join(wrong_msgs))
            assert False

    def test_critical(self, small_directory):
        p = Parse(small_directory, level="c")
        with capture_parse_logs(p.logger) as captured_msgs:
            p.parse()
            _ = p.get_dataframes(expanded=True)
            all_msgs = captured_msgs.content_list
        assert_all_loggers_level(50)
        if len(all_msgs) > 0:
            print("\n".join(all_msgs))
            assert False

    def test_notset(self, small_directory):
        p = Parse(small_directory, level=0)
        p.parse()
        _ = p.get_dataframes(expanded=True)
        assert_all_loggers_level(30)

    def test_ignoring_all_warnings(self, directory):
        ignored_warnings_file = os.path.join(
            directory, "mixed_files", "ALL_WARNINGS_IGNORED"
        )
        p = Parse(directory)
        p.load_ignored_warnings(ignored_warnings_file)
        with capture_parse_logs(p.logger) as captured_msgs:
            p.parse()
            _ = p.get_dataframes(expanded=True)
            all_msgs = captured_msgs.content_list
        assert all_msgs == []

    def test_capturing_suppressed_warnings(
        self, get_all_warnings, get_all_supressed_warnings
    ):
        for usual_warning in get_all_warnings:
            if usual_warning not in get_all_supressed_warnings:
                print(("NOT FOUND:", usual_warning))
                print(get_all_supressed_warnings)
            assert usual_warning in get_all_supressed_warnings

    def test_seeing_ignored_warnings(self, directory, get_all_warnings_parsed):
        ignored_warnings_file = os.path.join(
            directory, "mixed_files", "ALL_WARNINGS_IGNORED"
        )
        p = Parse(directory, level="d")
        p.load_ignored_warnings(ignored_warnings_file)
        with capture_parse_logs(p.logger, level="d") as captured_msgs:
            p.parse()
            _ = p.extract_facets("expanded")
            all_msgs = captured_msgs.content_list
        ignored = [
            "\n".join(msg.split("\n\t")[1:])
            for msg in all_msgs
            if msg.startswith("IGNORED")
        ]
        try:
            ignored_parsed = ignored_warnings2dict(ignored)
        except ValueError:
            print(ignored)
            raise
        for logger_name, message_ids in get_all_warnings_parsed.items():
            assert logger_name in ignored_parsed
            emitted_and_ignored = ignored_parsed[logger_name]
            for id in message_ids:
                assert id in emitted_and_ignored


def test_fixture(get_all_supressed_warnings):
    print(get_all_supressed_warnings)
    assert len(get_all_supressed_warnings) > 0
