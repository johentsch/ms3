import logging, sys, os
from collections import defaultdict
from functools import wraps
from enum import Enum


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

DEFAULT_LOG_FORMAT = '%(levelname)-8s %(_message_type_full)s (%(_message_type)s, %(_info)s) %(name)s -- %(pathname)s (line %(lineno)s) %(funcName)s(): \n\t %(message)s'


class MessageType(Enum):
    NO_TYPE = 0  # 0 is reserved as no type message
    MCS_NOT_EXCLUDED_FROM_BARCOUNT_WARNING = 1
    INCORRECT_VOLTA_MN_WARNING = 2
    INCOMPLETE_MC_WRONGLY_COMPLETED_WARNING = 3
    VOLTAS_WITH_DIFFERING_LENGTHS_WARNING = 4
    MISSING_END_REPEAT_WARNING = 5
    SUPERFLUOUS_TONE_REPLACEMENT_WARNING = 6

def get_default_formatter():
    format = DEFAULT_LOG_FORMAT
    return logging.Formatter(format)


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


def get_logger(name=None, level=None, path=None, propagate=True):
    """The function gets or creates the logger `name` and returns it, by default through the given LoggerAdapter class."""
    assert name != 'ms3' # TODO: comment out before release
    if isinstance(name, logging.LoggerAdapter):
        name = name.logger
    if isinstance(name, logging.Logger):
        if level is None:
            level = level.level
        name = name.name
    logger = config_logger(name, level=level, path=path, propagate=propagate)
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
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.ignore = defaultdict(lambda: [()])
        self.ignore.update({2: [(96)], 1: [()]})

    def filter(self, record):
        if record._message_type != 0 and record._info in self.ignore[record._message_type]:
            return False
        else:
            return True

def config_logger(name, level=None, path=None, propagate=True):
    """Configs the logger with name `name`. Overwrites existing config."""
    assert name is not None, "I don't want to change the root logger."
    new_logger = name not in logging.root.manager.loggerDict
    logger = logging.getLogger(name)
    original_makeRecord = logger.makeRecord

    def make_record_with_extra(name, level, fn, lno, msg, args, exc_info, func, extra, sinfo):
        record = original_makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra=extra, sinfo=sinfo)
        record._message_type = extra["message_type"] if extra is not None else 0
        record._info = extra["info"] if extra is not None else ""
        record._message_type_full = MessageType(extra["message_type"]).name if extra is not None else ""
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
    if logger.parent.name != 'root':
        # go to the following setup of handlers only for the top level logger
        return logger
    formatter = get_default_formatter()
    existing_handlers = [h for h in logger.handlers]
    stream_handlers = [h for h in existing_handlers if h.__class__ == logging.StreamHandler]
    n_stream_handlers = len(stream_handlers)
    if n_stream_handlers == 0:
        streamHandler = logging.StreamHandler(sys.stdout)
        streamHandler.setFormatter(formatter)
        streamHandler.setLevel(level)
        logger.addHandler(streamHandler)
        streamHandler.addFilter(WarningFilter(logger))
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
            fileHandler.addFilter(WarningFilter(logger))
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
        formatter = get_default_formatter()
        self._log_handler.setFormatter(formatter)
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
