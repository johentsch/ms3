import os

from ms3 import Parse, first_level_subdirs, capture_parse_logs, parse_ignored_warnings, ignored_warnings2dict
from ms3.logger import iter_ms3_loggers


class TestEquivalence():

    def test_parallel(self, directory):
        a = Parse(directory)
        b = Parse(directory)
        a.parse_mscx()
        b.parse_mscx(parallel=False)
        assert a.info(return_str=True) == b.info(return_str=True)

    def test_add_corpus(self, directory):
        a = Parse(directory)
        b = Parse()
        c = Parse()
        b.add_corpus(directory)
        for sd in first_level_subdirs(directory):
            corpus = os.path.join(directory, sd)
            c.add_corpus(corpus)
        assert a.count_extensions(per_key=True) == c.count_extensions(per_key=True)
        assert len(b.count_extensions(per_key=True)) == 1

def assert_all_loggers_level(level):
    for logger in iter_ms3_loggers():
        expected_level = 30 if logger.name == 'ms3' else level
        eff_level = logger.getEffectiveLevel()
        if eff_level != expected_level:
            print(f"THIS LOGGER SHOULD HAVE LEVEL {expected_level}, not {eff_level}: {logger.name}")
        assert eff_level == expected_level
        for h in logger.handlers:
            h_level = h.level
            if eff_level != expected_level:
                print(f"THE {h.__class__} of {logger.name} SHOULD HAVE LEVEL {expected_level}, not {h_level}.")
            assert h_level == expected_level


class TestLogging():

    def test_parallel_log_capture(self, directory):
        cfg = dict(level='d')
        a = Parse(directory, logger_cfg=cfg)
        b = Parse(directory, logger_cfg=cfg)
        with capture_parse_logs(a.logger) as captured_msgs:
            a.parse_mscx(parallel=False)
            non_parallel_msgs = captured_msgs.content_list
        with capture_parse_logs(b.logger) as captured_msgs:
            b.parse_mscx(parallel=True)
            parallel_msgs = captured_msgs.content_list
        for msg in non_parallel_msgs:
            assert msg in parallel_msgs

    def test_default(self, directory):
        p = Parse(directory)
        p.parse()
        _ = p.get_dataframes(expanded=True)
        assert_all_loggers_level(30)

    def test_debug(self, directory):
        p = Parse(directory, logger_cfg=dict(level='d'))
        p.parse()
        _ = p.get_dataframes(expanded=True)
        assert_all_loggers_level(10)


    def test_info(self, directory):
        p = Parse(directory, logger_cfg=dict(level='i'))
        with capture_parse_logs(p.logger) as captured_msgs:
            p.parse()
            _ = p.get_dataframes(expanded=True)
            all_msgs = captured_msgs.content_list
        assert_all_loggers_level(20)
        wrong_msgs = [msg for msg in all_msgs if msg.startswith('DEBUG')]
        if len(wrong_msgs) > 0:
            print('\n'.join(wrong_msgs))
            assert False

    def test_warning(self, directory):
        p = Parse(directory, logger_cfg=dict(level='w'))
        with capture_parse_logs(p.logger) as captured_msgs:
            p.parse()
            _ = p.get_dataframes(expanded=True)
            all_msgs = captured_msgs.content_list
        assert_all_loggers_level(30)
        wrong_msgs = [msg for msg in all_msgs if msg.startswith('INFO') or msg.startswith('DEBUG')]
        if len(wrong_msgs) > 0:
            print('\n'.join(wrong_msgs))
            assert False


    def test_error(self, directory):
        p = Parse(directory, logger_cfg=dict(level='e'))
        with capture_parse_logs(p.logger) as captured_msgs:
            p.parse()
            _ = p.get_dataframes(expanded=True)
            all_msgs = captured_msgs.content_list
        assert_all_loggers_level(40)
        wrong_msgs = [msg for msg in all_msgs if 'WARNING' in msg]
        if len(wrong_msgs) > 0:
            print('\n'.join(wrong_msgs))
            assert False


    def test_critical(self, directory):
        p = Parse(directory, logger_cfg=dict(level='c'))
        with capture_parse_logs(p.logger) as captured_msgs:
            p.parse()
            _ = p.get_dataframes(expanded=True)
            all_msgs = captured_msgs.content_list
        assert_all_loggers_level(50)
        if len(all_msgs) > 0:
            print('\n'.join(all_msgs))
            assert False


    def test_notset(self, directory):
        p = Parse(directory, logger_cfg=dict(level=0))
        p.parse()
        _ = p.get_dataframes(expanded=True)
        assert_all_loggers_level(30)

    def test_ignoring_all_warnings(self, directory):
        ignored_warnings_file = os.path.join(directory, 'mixed_files', 'ALL_WARNINGS_IGNORED')
        p = Parse(directory)
        p.load_ignored_warnings(ignored_warnings_file)
        with capture_parse_logs(p.logger) as captured_msgs:
            p.parse()
            _ = p.get_dataframes(expanded=True)
            all_msgs = captured_msgs.content_list
        assert len(all_msgs) == 0

    def test_seeing_ignored_warnings(self, directory, get_all_warnings_parsed):
        ignored_warnings_file = os.path.join(directory, 'mixed_files', 'ALL_WARNINGS_IGNORED')
        p = Parse(directory, logger_cfg=dict(level='d'))
        p.load_ignored_warnings(ignored_warnings_file)
        with capture_parse_logs(p.logger, level='d') as captured_msgs:
            p.parse(parallel=False)
            _ = p.get_dataframes(expanded=True)
            all_msgs = captured_msgs.content_list
        ignored = [msg for msg in all_msgs if msg.startswith('IGNORED')]
        ignored = ['\n'.join(msg for msg in message.split('\n')[2:]).strip("\n\t ") for message in ignored]
        ignored_parsed = ignored_warnings2dict(ignored)
        for logger_name, message_ids in get_all_warnings_parsed.items():
            assert logger_name in ignored_parsed
            emitted_and_ignored = ignored_parsed[logger_name]
            for id in message_ids:
                assert id in emitted_and_ignored

    def test_suppressing_warning(self, get_all_warnings, get_all_supressed_warnings):
        for usual_warning in get_all_warnings:
            if usual_warning not in get_all_supressed_warnings:
                print("NOT FOUND: ", usual_warning)
                print(get_all_supressed_warnings)
            assert usual_warning in get_all_supressed_warnings