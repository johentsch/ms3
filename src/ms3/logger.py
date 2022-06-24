import logging, sys, os
from functools import wraps


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

CURRENT_LEVEL = logging.INFO

DEFAULT_LOG_FORMAT = '%(levelname)-8s %(name)s -- %(message)s'

def get_default_formatter():
    format = DEFAULT_LOG_FORMAT
    return logging.Formatter(format)

class ContextAdapter(logging.LoggerAdapter):
    """ This LoggerAdapter is designed to include the module and function that called the logger."""
    def process(self, msg, overwrite={}, stack_info=False, **kwargs):
        # my_context = kwargs.pop('my_context', self.extra['my_context'])
        fn, l, f, s = self.logger.findCaller(stack_info=stack_info)
        fname = os.path.basename(overwrite.pop('fname', fn))
        line = overwrite.pop('line', l)
        func = overwrite.pop('func', f)
        stack = overwrite.pop('stack', s)
        msg = msg.replace('\n', '\n\t')
        message = f"{fname} (line {line}) {func}():\n\t{msg}" if stack is None else f"{fname} line {line}, {func}():\n\t{msg}\n{stack}"
        return message, kwargs


class LoggedClass:
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
        self.logger = None
        self.logger_names = {}
        self.update_logger_cfg(**logger_cfg)

    def update_logger_cfg(self, name=None, level=None, path=None, propagate=True):
        if name is not None and not isinstance(name, str):
            raise ValueError(f"name needs to be a string, not a {name.__class__}")
        config_options = ['name', 'level', 'path', 'propagate']
        params = locals()
        logger_cfg = {param: value for param, value in params.items() if param in config_options}
        tested_cfg = update_cfg(logger_cfg, config_options)
        for o in config_options:
            if o not in tested_cfg and o not in self.logger_cfg:
                tested_cfg[o] = None
        self.logger_cfg.update(tested_cfg)
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



def get_logger(name=None, level=None, path=None, propagate=True, adapter=ContextAdapter):
    """The function gets or creates the logger `name` and returns it, by default through the given LoggerAdapter class."""
    global CURRENT_LEVEL
    if isinstance(name, logging.LoggerAdapter):
        name = name.logger
    if isinstance(name, logging.Logger):
        if level is None:
            level = level.level
        name = name.name
    if level is None:
        level = CURRENT_LEVEL
    else:
        CURRENT_LEVEL = LEVELS[level.upper()] if level.__class__ == str else level
    if name not in logging.root.manager.loggerDict:
        config_logger(name, level=level, path=path, propagate=propagate)
    logger = logging.getLogger(name)
    logger.setLevel(CURRENT_LEVEL)
    # for h in logger.handlers:
    #     if h.__class__ != logging.FileHandler:
    #         h.setLevel(CURRENT_LEVEL)

    if adapter is not None:
        logger = adapter(logger, {})
    if name is None:
        logging.critical("The root logger has been altered.")
    return logger

def resolve_level_param(level):
    if level.__class__ == str:
        level = LEVELS[level.upper()]
    assert isinstance(level, int), f"Logging level needs to be an integer, not {level.__class__}"
    return level

def config_logger(name, level=None, path=None, propagate=True):
    """Configs the logger with name `name`. Overwrites existing config."""
    global CURRENT_LEVEL
    assert name is not None, "I don't want to change the root logger."
    logger = logging.getLogger(name)
    logger.propagate = propagate
    if level is not None:
        level = resolve_level_param(level)
        logger.setLevel(level)
    else:
        level = CURRENT_LEVEL
    if logger.parent.name != 'root':
        # go to the following setup of handlers only for the top level logger
        return
    formatter = get_default_formatter()
    existing_handlers = [h for h in logger.handlers]
    stream_handlers = sum(True for h in existing_handlers if h.__class__ == logging.StreamHandler)
    if stream_handlers == 0:
        streamHandler = logging.StreamHandler(sys.stdout)
        streamHandler.setFormatter(formatter)
        streamHandler.setLevel(level)
        logger.addHandler(streamHandler)
    elif stream_handlers > 1:
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
        if any(True for h in existing_handlers if h.__class__ == logging.FileHandler and h.baseFilename == log_file):
            logger.error(f"Logger '{name}' already has a FileHandler attached.")
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
            logger.file_handler = fileHandler
    else:
        logger.file_handler = None


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
            l = 'class'
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
