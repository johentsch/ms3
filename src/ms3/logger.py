import logging
import os
import sys
from contextlib import contextmanager
from enum import Enum, unique
from typing import Iterable, List, Optional, Set, Tuple

LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "NOTSET": logging.NOTSET,
    "D": logging.DEBUG,
    "I": logging.INFO,
    "W": logging.WARNING,
    "E": logging.ERROR,
    "C": logging.CRITICAL,
    "N": logging.NOTSET,
}


@unique
class MessageType(Enum):
    """Enumerated constants of message types."""

    NO_TYPE = 0  # 0 is reserved as no type message
    MCS_NOT_EXCLUDED_FROM_BARCOUNT_WARNING = 1
    INCORRECT_VOLTA_MN_WARNING = 2
    INCOMPLETE_MC_WRONGLY_COMPLETED_WARNING = 3
    VOLTAS_WITH_DIFFERING_LENGTHS_WARNING = 4
    MISSING_END_REPEAT_WARNING = 5
    DCML_HARMONY_SUPERFLUOUS_TONE_REPLACEMENT_WARNING = 6
    not_in_use = 7
    DCML_HARMONY_KEY_NOT_SPECIFIED_WARNING = 8
    COMPETING_MEASURE_INFO_WARNING = 9
    IGNORED = 10
    MISSING_FACET_INFO = 11
    DCML_HARMONY_INCOMPLETE_LOCALKEY_COLUMN_WARNING = 12
    DCML_HARMONY_INCOMPLETE_PEDAL_COLUMN_WARNING = 13
    LOGGER_NOT_IN_USE_WARNING = 14
    DCML_HARMONY_SYNTAX_WARNING = 15
    DCML_PHRASE_INCONGRUENCY_WARNING = 16
    DCML_EXPANSION_FAILED_WARNING = 17
    DCML_SEVENTH_CORD_WITH_ALTERED_SEVENTH_WARNING = 18
    DCML_NON_CHORD_TONES_ABOVE_THRESHOLD_WARNING = 19
    UNUSED_FINE_MARKER_WARNING = 20
    PLAY_UNTIL_IS_MISSING_LABEL_WARNING = 21
    JUMP_TO_IS_MISSING_LABEL_WARNING = 22
    MISSING_TIME_SIGNATURE_WARNING = 23  # no timesig through the piece
    BEGINNING_WITHOUT_TIME_SIGNATURE_WARNING = (
        24  # no timesig in more than only the first measure (which could be an incipit)
    )
    INVALID_REPEAT_STRUCTURE = 25
    UNFOLDING_REPEATS_FAILED_WARNING = 26
    DCML_DEFAULT_CORRECTION_WARNING = 27
    WRONGLY_ENCODED_POSITION_WARNING = 28
    FIRST_BAR_MISSING_TEMPO_MARK_WARNING = 29
    CORRECTED_INSTRUMENT_TRACKNAME_WARNING = 30
    INCONSISTENT_INSTRUMENT_CHANGE_WITHIN_PART = 31
    FRICTIONLESS_VALIDATION_ERROR_WARNING = 32


class CustomFormatter(logging.Formatter):
    """Formats message depending on whether there is a specified message type"""

    def format(self, record):
        if not hasattr(record, "_message_type"):
            raise ValueError(
                f"Logger {record.name} has not been correctly defined and is missing the default _message_type field."
            )
        msg = record.msg.replace("\n", "\n\t")
        if record._message_type == 0:  # if there is no message type
            record.msg = "%-8s %s -- %s (line %s) %s():\n\t%s" % (
                record.levelname,
                record.name,
                record.pathname,
                record.lineno,
                record.funcName,
                msg,
            )
        elif record._message_type == 10:
            record.msg = "IGNORED  %s -- %s (line %s) %s():\n\t%s" % (
                record.name,
                record.pathname,
                record.lineno,
                record.funcName,
                msg,
            )
        else:
            record.msg = "%s %s %s -- %s (line %s) %s():\n\t%s" % (
                record._message_type_full,
                record._message_id,
                record.name,
                record.pathname,
                record.lineno,
                record.funcName,
                msg,
            )
        return super(CustomFormatter, self).format(record)


class LoggedClass:
    """

    logger : :obj:`logging.Logger` or :obj:`logging.LoggerAdapter`
        Current logger that the object is using.
    parser : {'bs4'}
        The only XML parser currently implemented is BeautifulSoup 4.
    paths, files, pieces, fexts, logger_names : :obj:`dict`
        Dictionaries for keeping track of file information handled by .
    """

    _deprecated_elements: List[str] = []
    """Methods and properties named here will be removed from the object's tab completion."""

    def __init__(self, subclass: str, logger_cfg: Optional[dict] = None):
        if logger_cfg is None:
            logger_cfg = {}
        old_code_warnings = []
        if "logger_cfg" in logger_cfg:
            old_code_warnings.append("logger_cfg")
            logger_cfg = logger_cfg["logger_cfg"]
        # deprecated arguments may make it into the logger_cfg because the latter is specified as **kwargs
        deprecated_arguments = {
            "key": None,
            "only_metadata_fnames": "only_metadata_pieces",
        }
        used_deprecated_arguments = [
            arg for arg in deprecated_arguments if arg in logger_cfg
        ]
        if used_deprecated_arguments:
            old_code_warnings += used_deprecated_arguments
            logger_cfg = {
                k: v
                for k, v in logger_cfg.items()
                if k not in used_deprecated_arguments
            }
        self.logger_cfg: dict = dict(logger_cfg)
        if "name" not in self.logger_cfg or self.logger_cfg["name"] is None:
            name = subclass if subclass == "ms3" else "ms3." + subclass
            self.logger_cfg["name"] = name
        self.logger_names: dict = {}
        self.logger: logging.Logger = get_logger(**self.logger_cfg)
        for arg in old_code_warnings:
            if arg == "logger_cfg":
                self.logger.warning(
                    f"You are using old code that initiated a {subclass!r} object with the argument 'logger_cfg'. "
                    f"New code passes the logger config as **kwargs."
                )
            elif arg == "key":
                self.logger.warning(
                    f"You are using old code that initiated a {subclass!r} object with the argument 'key'. "
                    f"Since version 1.0.0, ms3 does not use this argument to match files anymore and you will probably "
                    f"see get errors in the following."
                )
            else:
                self.logger.warning(
                    f"You are using old code that initiated a {subclass!r} object with the argument {arg!r}. "
                    f"Since version 2.0.0, this argument has been replaced by {deprecated_arguments[arg]!r} and is "
                    f"ignored."
                )
        self.logger_names["class"] = self.logger.name

    def change_logger_cfg(self, level):
        level = resolve_level_param(level)
        self.logger.setLevel(level)
        self.logger_cfg["level"] = level
        for h in self.logger.handlers:
            if not isinstance(h, LogCaptureHandler):
                h.setLevel(level)

    def __getstate__(self):
        """Loggers pose problems when pickling: Remove the reference."""
        self.logger = None
        return self.__dict__

    def __setstate__(self, state):
        """Restore the reference to the root logger."""
        self.__dict__.update(state)
        self.logger = get_logger(**self.logger_cfg)

    def __dir__(self) -> Iterable[str]:
        if len(self._deprecated_elements) == 0:
            return super().__dir__()
        elements = super().__dir__()
        return sorted(
            element for element in elements if element not in self._deprecated_elements
        )


def get_logger(name=None, level=None, path=None, ignored_warnings=[]) -> logging.Logger:
    """
    The function gets or creates the logger `name` and returns it, by default through the given LoggerAdapter class.
    """
    if isinstance(name, logging.Logger):
        name = name.name
    logger = config_logger(
        name, level=level, path=path, ignored_warnings=ignored_warnings
    )
    if name is None:
        logging.critical("The root logger has been altered.")
    return logger


def resolve_level_param(level):
    if isinstance(level, str):
        level = LEVELS[level.upper()]
    assert isinstance(
        level, int
    ), f"Logging level needs to be an integer, not {level.__class__}"
    return level


def get_parent_level(logger):
    if logger.parent is None:
        return None
    parent = logger.parent
    if parent.level == 0:
        return get_parent_level(parent)
    return parent


def get_log_capture_handler(logger):
    name = logger.name
    if not name.startswith("ms3") or name == "ms3":
        return
    head = ".".join(name.split(".")[:2])
    head_logger = get_logger(head)
    try:
        return next(h for h in head_logger.handlers if isinstance(h, LogCaptureHandler))
    except StopIteration:
        return


class WarningFilter(logging.Filter):
    """Filters messages. If message is in ignored_warnings, its level is changed to debug."""

    def __init__(self, logger, ignored_warnings):
        super().__init__()
        self.logger = logger
        self.ignored_warnings = set(ignored_warnings)

    def filter(self, record):
        ignored = record._message_id in self.ignored_warnings
        if ignored:
            # f"The following warning has been ignored in an IGNORED_WARNINGS file:\n{CustomFormatter().format(record)}"
            self.logger.debug(
                CustomFormatter().format(record),
                extra={"message_id": (10,)},
            )
        return not ignored

    def __repr__(self):
        return f"WarningFilter({self.ignored_warnings})"

    def __str__(self):
        def __repr__(self):
            return f"WarningFilter('{self.logger.name}', {self.ignored_warnings})"


def resolve_log_path_argument(path, name, logger):
    log_file = None
    if path is not None:
        path = resolve_dir(path)
        if os.path.isdir(path):
            piece = name + ".log"
            log_file = os.path.join(path, piece)
        else:
            path_component, piece = os.path.split(path)
            if os.path.isdir(path_component):
                log_file = path
            else:
                logger.error(
                    f"Log file output cannot be configured for '{name}' because '{path_component}' is "
                    f"not an existing directory."
                )
    return log_file


def config_logger(name, level=None, path=None, ignored_warnings=[]):
    """Configs the logger with name `name`. Overwrites existing config."""
    assert name is not None, "I don't want to change the root logger."
    is_new_logger = name not in logging.root.manager.loggerDict or isinstance(
        logging.root.manager.loggerDict[name], logging.PlaceHolder
    )
    is_top_level = name == "ms3"
    logger = logging.getLogger(name)
    if level is not None:
        level = resolve_level_param(level)
    if is_top_level:
        # # uncomment if you want to check for what's described in the log message
        # last_8 = ', '.join(f"-{i}: {stack()[i].function}()" for i in range(1, 9))
        set_level = 0 if level is None else level
        logger.debug(f"Setting top-level logger 'ms3' to level {set_level}")
        logger.setLevel(set_level)
    is_head_logger = logger.parent.name == "ms3"
    adding_file_handler = path is not None
    adding_any_handlers = adding_file_handler or is_top_level or is_head_logger

    if level is not None:
        if level > 0:
            logger.setLevel(level)
        elif is_head_logger:
            logger.setLevel(30)
    effective_level = logger.getEffectiveLevel()

    if is_head_logger:
        # checking if any loggers exist from previous runs and need to be adapted
        for logger_name, lggr in logging.Logger.manager.loggerDict.items():
            if (
                logger_name.startswith(name)
                and logger_name != name
                and isinstance(lggr, logging.Logger)
            ):
                if lggr.getEffectiveLevel() not in (0, effective_level):
                    lggr.setLevel(effective_level)

    if len(ignored_warnings) > 0:
        try:
            existing_filter = next(
                filter for filter in logger.filters if isinstance(filter, WarningFilter)
            )
            existing_filter.ignored_warnings.update(
                ID
                for ID in ignored_warnings
                if ID not in existing_filter.ignored_warnings
            )
        except StopIteration:
            logger.addFilter(WarningFilter(logger, ignored_warnings=ignored_warnings))

    if not is_new_logger and not adding_any_handlers:
        return logger

    if is_new_logger:
        logger.propagate = not (is_top_level or is_head_logger)
        # for each newly defined logger we replace the makeRecord method which allows us to add the property
        # '_message_type_full' to each log record. This enables replacing the LEVEL in the output message with
        # the message type name defined in the enum class MessageType each time something is logged with
        # extra={message_id: (<message_id>, ...)} (where ... are arbitrary elements to identify one specific instance,
        # e.g. a particular measure number)
        original_makeRecord = logger.makeRecord

        def make_record_with_extra(
            name, level, fn, lno, msg, args, exc_info, func, extra, sinfo
        ):
            """
            Rewrites the method of record logging to pass extra parameter.
            Returns
            -------
                record with fields: _info - label of message
                                    _message_type - index of message type accordingly to enum class MessageType
                                    _message_type_full - name of message type accordingly to enum class MessageType
            """
            record = original_makeRecord(
                name,
                level,
                fn,
                lno,
                msg,
                args,
                exc_info,
                func,
                extra=extra,
                sinfo=sinfo,
            )
            if extra is None:
                record._message_id = ()
                record._message_type = 0
            else:
                record._message_id = extra["message_id"]
                record._message_type = record._message_id[0]
            record._message_type_full = MessageType(record._message_type).name
            return record

        logger.makeRecord = make_record_with_extra

    if adding_any_handlers:
        if level is not None:
            if effective_level != level:
                if level == 0 and (is_head_logger or is_top_level):
                    logger.info(
                        f"Cannot unset (i.e., set to 0) logging level of top-level logger. Use > 0 instead. "
                        f"Left level at {effective_level}"
                    )
                else:
                    logger.warning(
                        f"The call to .setLevel() did not result in changing the level from {effective_level} to "
                        f"{level}"
                    )
        level = effective_level
        formatter = CustomFormatter()
        diverging_level = [
            h
            for h in logger.handlers
            if h.level != level and not isinstance(h, LogCaptureHandler)
        ]
        for h in diverging_level:
            h.setLevel(level)

        if is_head_logger or is_top_level:
            stream_handlers = [
                h for h in logger.handlers if h.__class__ == logging.StreamHandler
            ]
            n_stream_handlers = len(stream_handlers)
            if n_stream_handlers == 0:
                streamHandler = logging.StreamHandler(sys.stdout)
                streamHandler.setFormatter(formatter)
                streamHandler.setLevel(level)
                logger.addHandler(streamHandler)
            elif n_stream_handlers > 1:
                logger.warning(
                    f"The logger {name} has been setup with {stream_handlers} StreamHandlers and is probably sending "
                    f"every message twice."
                )

        log_file = resolve_log_path_argument(path, name, logger)
        if log_file is not None:
            file_handlers = [
                h for h in logger.handlers if h.__class__ == logging.FileHandler
            ]
            n_file_handlers = len(file_handlers)
            if n_file_handlers > 0:
                logger.error(
                    f"Logger '{name}' already has {n_file_handlers} FileHandlers attached."
                )
            else:
                logger.debug(f"Storing logs as {log_file}")
                fileHandler = logging.FileHandler(log_file, mode="a", delay=True)
                fileHandler.setLevel(level)
                # file_formatter = logging.Formatter("%(asctime)s "+format, datefmt='%Y-%m-%d %H:%M:%S')
                fileHandler.setFormatter(formatter)
                logger.addHandler(fileHandler)
    return logger


def resolve_dir(d):
    """Resolves '~' to HOME directory and turns ``dir`` into an absolute path."""
    if d is None:
        return None
    d = str(d)
    if "~" in d:
        return os.path.expanduser(d)
    return os.path.abspath(d)


def update_cfg(cfg_dict: dict, admitted_keys: Iterable) -> Tuple[dict, dict]:
    correct = {k: v for k, v in cfg_dict.items() if k in admitted_keys}
    incorrect = {k: v for k, v in cfg_dict.items() if k not in admitted_keys}
    return correct, incorrect


class LogCaptureHandler(logging.Handler):
    def __init__(self, log_queue):
        logging.Handler.__init__(self)
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.append(self.format(record).strip("\n\t "))


class LogCapturer(object):
    """Adapted from https://stackoverflow.com/a/37967421"""

    def __init__(self, level="W"):
        self._log_queue = (
            list()
        )  # original example was using collections.deque() to set maxlength
        self._log_handler = LogCaptureHandler(self._log_queue)
        # self._log_handler.setFormatter(CustomFormatter())
        self._log_handler.setLevel(resolve_level_param(level))

    @property
    def content_string(self):
        return "\n".join(self._log_queue)

    @property
    def content_list(self):
        return self._log_queue

    @property
    def log_handler(self):
        return self._log_handler


def iter_ms3_loggers(exclude_placeholders=True) -> tuple:
    for name in logging.Logger.manager.loggerDict:
        if name.startswith("ms3"):
            logger = logging.getLogger(name)
            if not exclude_placeholders or isinstance(logger, logging.Logger):
                yield name, logger


def inspect_loggers(exclude_placeholders=False) -> dict:
    return dict(iter_ms3_loggers(exclude_placeholders=exclude_placeholders))


@contextmanager
def temporarily_suppress_warnings(logged_object: LoggedClass):
    prev_level = logged_object.logger.level
    logged_object.change_logger_cfg(level="c")
    yield logged_object
    logged_object.change_logger_cfg(level=prev_level)


def normalize_logger_name(name: str) -> str:
    """
    Shorten a logger name to Corpus.Piece so that it can be used to configure all associated loggers, no matter what.
    """
    components = name.split(".")
    for remove in ("ms3", "Corpus", "Parse"):
        try:
            components.remove(remove)
        except ValueError:
            pass
    for extension in (
        "tsv",
        "mscx",
        "mscz",
        "cap",
        "capx",
        "midi",
        "mid",
        "musicxml",
        "mxl",
        "xml",
    ):
        if components[-1] == extension:
            components = components[:-1]
            break
    return ".".join(components)


def get_ignored_warning_ids(logger: logging.Logger) -> Set[tuple]:
    try:
        existing_filter = next(
            filter for filter in logger.filters if isinstance(filter, WarningFilter)
        )
        return set(existing_filter.ignored_warnings)
    except StopIteration:
        return set()


def inspect_logger_parents(inspected_logger):
    """Print all parents until the first one is found that has at least one handler."""
    if not isinstance(inspected_logger, logging.Logger):
        inspected_logger = logging.getLogger(inspected_logger)
    LL = inspected_logger
    info_string = f"LOOKING FOR HANDLER OF {LL}"
    while LL.parent:
        LL = LL.parent
        info_string += (
            f"\nparent: {LL.name} (propagates: {LL.propagate}), handlers: {LL.handlers}"
        )
        if LL.handlers:
            break
    print(info_string)
