import logging, sys, os
from functools import wraps
from enum import Enum
import re


LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
    'D': logging.DEBUG,
    'I': logging.INFO,
    'W': logging.WARNING,
    'E': logging.ERROR,
    'C': logging.CRITICAL,
}

class MessageType(Enum):
    """Enumerated constants of message types."""
    NO_TYPE = 0  # 0 is reserved as no type message
    MCS_NOT_EXCLUDED_FROM_BARCOUNT_WARNING = 1
    INCORRECT_VOLTA_MN_WARNING = 2
    INCOMPLETE_MC_WRONGLY_COMPLETED_WARNING = 3
    VOLTAS_WITH_DIFFERING_LENGTHS_WARNING = 4
    MISSING_END_REPEAT_WARNING = 5
    SUPERFLUOUS_TONE_REPLACEMENT_WARNING = 6
    OVERLOOKED_MSCX_FILES_WARNING = 7
    KEY_NOT_SPECIFIED_ERROR = 8
    COMPETING_MEASURE_INFO = 9


class CustomFormatter(logging.Formatter):
    """Formats message depending on whether there is a specified message type"""
    def format(self, record):
        if record._message_type == 0:  # if there is no message type
            record.msg = '%-8s %s -- %s (line %s) %s(): \n\t %s' % (record.levelname, record.name, record.pathname, record.lineno, record.funcName, record.msg)
        else:
            record.msg = '%s %s %s -- %s (line %s) %s(): \n\t %s' % (record._message_type_full, record._message_id, record.name, record.pathname, record.lineno, record.funcName, record.msg)
        return super(CustomFormatter, self).format(record)


class LoggedClass():
    """

    logger : :obj:`logging.Logger` or :obj:`logging.LoggerAdapter`
        Current logger that the object is using.
    parser : {'bs4'}
        The only XML parser currently implemented is BeautifulSoup 4.
    paths, files, fnames, fexts, logger_names : :obj:`dict`
        Dictionaries for keeping track of file information handled by .
    """
    def __init__(self, subclass, logger_cfg={}):
        logger_cfg = dict(logger_cfg)
        if 'name' not in logger_cfg or logger_cfg['name'] is None:
            name = subclass if subclass == 'ms3' else 'ms3.' + subclass
            logger_cfg['name'] = name
        if 'propagate' not in logger_cfg:
            logger_cfg['propagate'] = True
        self.logger_cfg = logger_cfg
        self.logger_names = {}
        self.logger = get_logger(**self.logger_cfg)
        self.logger_names['class'] = self.logger.name



    def __getstate__(self):
        """ Loggers pose problems when pickling: Remove the reference."""
        self.logger = None
        return self.__dict__

    def __setstate__(self, state):
        """ Restore the reference to the root logger. """
        self.__dict__.update(state)
        self.logger = get_logger(self.logger_names['class'])


def get_logger(name=None, level=None, path=None, propagate=True, ignored_warnings=[]):
    """The function gets or creates the logger `name` and returns it, by default through the given LoggerAdapter class."""
    #assert name != 'ms3', "logged function called without passing logger (or logger name)" # TODO: comment out before release
    # if isinstance(name, logging.LoggerAdapter):
    #     name = name.logger
    if isinstance(name, logging.Logger):
        if level is None:
            level = level.level
        name = name.name
    logger = config_logger(name, level=level, path=path, propagate=propagate, ignored_warnings=ignored_warnings)
    if name is None:
        logging.critical("The root logger has been altered.")
    return logger

def resolve_level_param(level):
    if level.__class__ == str:
        level = LEVELS[level.upper()]
    assert isinstance(level, int), f"Logging level needs to be an integer, not {level.__class__}"
    return level


def get_parent_level(logger):
    if logger.parent is None:
        return None
    parent = logger.parent
    if parent.level == 0:
        return get_parent_level(parent)
    return parent

class WarningFilter(logging.Filter):
    """Filters messages. If message is in ignored_warnings, its level is changed to debug."""
    def __init__(self, logger, ignored_warnings):
        super().__init__()
        self.logger = logger
        self.ignored_warnings = ignored_warnings

    def filter(self, record):
        ignored = record._message_id in self.ignored_warnings
        if ignored:
            self.logger.debug(f"The following warning has been ignored through an IGNORED_WARNINGS file:\n{record.getMessage()}")
        return not ignored

def config_logger(name, level=None, path=None, propagate=True, ignored_warnings=[]):
    """Configs the logger with name `name`. Overwrites existing config."""
    assert name is not None, "I don't want to change the root logger."
    new_logger = name not in logging.root.manager.loggerDict
    logger = logging.getLogger(name)
    original_makeRecord = logger.makeRecord

    def make_record_with_extra(name, level, fn, lno, msg, args, exc_info, func, extra, sinfo):
        """
        Rewrites the method of record logging to pass extra parameter.
        Returns
        -------
            record with fields: _info - label of message
                                _message_type - index of message type accordingly to enum class MessageType
                                _message_type_full - name of message type accordingly to enum class MessageType
        """
        record = original_makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra=extra, sinfo=sinfo)
        if extra is None:
            record._message_id = ()
            record._message_type = 0
        else:
            filter_function = lambda elem: list(map(convert_to_int, filter(None, re.split("[(, :')]+", elem)))) if type(
                elem) == str else [elem]
            record._message_id = tuple([elem_  for elem in extra["message_id"] for elem_ in filter_function(elem)])
            record._message_type = record._message_id[0]
        record._message_type_full = MessageType(record._message_type).name
        return record

    logger.makeRecord = make_record_with_extra
    logger.propagate = propagate
    if level is not None:
        level = resolve_level_param(level)
        logger.setLevel(level)
    elif new_logger:
        parent = get_parent_level(logger)
        if parent is None:
            level = 20
            logger.setLevel(level)
            logger.info(f"New logger '{name}' has no parent with level > 0. Using default level INFO.")
        else:
            level = parent.level
            logger.setLevel(level)
            logger.debug(f"New logger '{name}' initialized with level {level}, inherited from parent {parent.name}.")
    else:
        level = logger.level
    if len(ignored_warnings) > 0:
        logger.addFilter(WarningFilter(logger, ignored_warnings=ignored_warnings))
    if logger.parent.name != 'root':
        # go to the following setup of handlers only for the top level logger
        if new_logger:
            logger.debug(f"Configured {logger.name} for the first time.")
        return logger
    formatter = CustomFormatter()
    existing_handlers = [h for h in logger.handlers]
    stream_handlers = [h for h in existing_handlers if h.__class__ == logging.StreamHandler]
    n_stream_handlers = len(stream_handlers)
    if n_stream_handlers == 0:
        streamHandler = logging.StreamHandler(sys.stdout)
        streamHandler.setFormatter(formatter)
        streamHandler.setLevel(level)
        logger.addHandler(streamHandler)
    elif n_stream_handlers == 1:
        streamHandler = stream_handlers[0]
        streamHandler.setLevel(level)
    else:
        logger.info(f"The logger {name} has been setup with {stream_handlers} StreamHandlers and is probably sending every message twice.")
    log_file = None
    if path is not None:
        path = resolve_dir(path)
        if os.path.isdir(path):
            fname = name + ".log"
            log_file = os.path.join(path, fname)
        else:
            path_component, fname = os.path.split(path)
            if os.path.isdir(path_component):
                log_file = path
            else:
                logger.error(f"Log file output cannot be configured for '{name}' because '{path_component}' is "
                             f"not an existing directory.")

    if log_file is not None:
        file_handlers = [h for h in existing_handlers if h.__class__ == logging.FileHandler]
        n_file_handlers = len(file_handlers)
        if n_file_handlers > 0:
            logger.error(f"Logger '{name}' already has {n_file_handlers} FileHandlers attached.")
        else:
            # log_dir, _ = os.path.split(log_file)
            # if not os.path.isdir(log_dir):
            #     os.makedirs(log_dir, exist_ok=True)
            logger.debug(f"Storing logs as {log_file}")
            fileHandler = logging.FileHandler(log_file, mode='a', delay=True)
            fileHandler.setLevel(level)
            #file_formatter = logging.Formatter("%(asctime)s "+format, datefmt='%Y-%m-%d %H:%M:%S')
            fileHandler.setFormatter(formatter)
            logger.addHandler(fileHandler)
    return logger


def function_logger(f):
    """This decorator ensures that the decorated function can use the variable `logger` for logging and
       makes it possible to pass the function the keyword argument `logger` with either a Logger object or
       the name of one. If the keyword argument is not passed, the root logger is used.

    Example
    -------
    This is how the decorator can be used:

    .. code-block:: python

        from ms3.logger import function_logger

        @function_logger
        def log_this(msg):
            logger.warning(msg)


        if __name__ == '__main__':
            log_this('First test', logger='my_logger')
            log_this('Second Test')

    Output:

    .. code-block:: python

        WARNING my_logger -- function_logger.py (line 5) log_this():
            First test
        WARNING root -- function_logger.py (line 5) log_this():
            Second Test

    """

    @wraps(f)
    def logger(*args, **kwargs):
        l = kwargs.pop('logger', None)
        if l is None:
            l = 'ms3'
        if l.__class__ == str:
            logg = get_logger(l)
        else:
            logg = l

        func_globals = f.__globals__
        saved_values = func_globals.copy()
        f.__globals__.update({'logger': logg})
        try:
            result = f(*args, **kwargs)
        finally:
            func_globals = saved_values  # Undo changes.
        return result

    return logger


def resolve_dir(d):
    """ Resolves '~' to HOME directory and turns ``dir`` into an absolute path.
    """
    if d is None:
        return None
    if '~' in d:
        return os.path.expanduser(d)
    return os.path.abspath(d)


@function_logger
def update_cfg(cfg_dict, admitted_keys):
    correct = {k: v for k, v in cfg_dict.items() if k in admitted_keys}
    incorrect = {k: v for k, v in cfg_dict.items() if k not in admitted_keys}
    if len(incorrect) > 0:
        corr = '' if len(correct) == 0 else f"\nRecognized options: {correct}"
        logger.warning(f"Unknown config options: {incorrect}{corr}")
    return correct


class LogCaptureHandler(logging.Handler):

    def __init__(self, log_queue):
        logging.Handler.__init__(self)
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.append(self.format(record))


class LogCapturer(object):
    """Adapted from https://stackoverflow.com/a/37967421"""
    def __init__(self, level="W"):
        self._log_queue = list() # original example was using collections.deque() to set maxlength
        self._log_handler = LogCaptureHandler(self._log_queue)
        self._log_handler.setFormatter(CustomFormatter())
        self._log_handler.setLevel(resolve_level_param(level))

    @property
    def content_string(self):
        return '\n'.join(self._log_queue)

    @property
    def content_list(self):
        return self._log_queue

    @property
    def log_handler(self):
        return self._log_handler

def convert_to_int(input_str: str):
    """Convert list of strings to list of integer if possible"""
    try:
        return int(input_str)
    except ValueError:
        return input_str
