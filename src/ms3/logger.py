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
    def __init__(self, subclass='root', logger_cfg={}):
        self.logger_cfg = {'name': subclass}
        if 'name' in logger_cfg:
            name = logger_cfg['name']
        else:
            name = subclass
        # if name in logging.root.manager.loggerDict:
        #     del(logging.root.manager.loggerDict[name])
        self.logger_names = {'root': subclass}
        self.update_logger_cfg(logger_cfg=logger_cfg)

    def update_logger_cfg(self, name=None, level=None, path=None, file=None, logger_cfg={}):
        if name is not None and not isinstance(name, str):
            raise ValueError(f"name needs to be a string, not a {name.__class__}")
        config_options = ['name', 'level', 'path', 'file']
        params = locals()
        for param in config_options:
            if params[param] is not None:
                logger_cfg[param] = params[param]
        tested_cfg = update_cfg(logger_cfg, config_options)
        for o in config_options:
            if o not in tested_cfg and o not in self.logger_cfg:
                tested_cfg[o] = None
        self.logger_cfg.update(tested_cfg)
        self.logger = get_logger(logger_cfg=self.logger_cfg)
        self.logger_names['root'] = self.logger.name

    def __getstate__(self):
        """ Loggers pose problems when pickling: Remove the reference."""
        self.logger = None
        return self.__dict__

    def __setstate__(self, state):
        """ Restore the reference to the root logger. """
        self.__dict__.update(state)
        self.logger = get_logger(self.logger_names['root'])



def get_logger(name=None, level=None, path=None, file=None, logger_cfg={}, adapter=ContextAdapter):
    """The function gets or creates the logger `name` and returns it, by default through the given LoggerAdapter class."""
    global CURRENT_LEVEL
    params = locals()
    for param in ['name', 'level', 'path', 'file']:
        if param in logger_cfg:
            params[param] = logger_cfg[param]

    if isinstance(params['name'], logging.LoggerAdapter):
        params['name'] = params['name'].logger
    if isinstance(params['name'], logging.Logger):
        if params['level'] is None:
            params['level'] = params['name'].level
        params['name'] = params['name'].name
    if params['level'] is None:
        params['level'] = CURRENT_LEVEL
    else:
        CURRENT_LEVEL = LEVELS[params['level'].upper()] if params['level'].__class__ == str else params['level']
    try:
        if params['name'] not in logging.root.manager.loggerDict:
            config_logger(params['name'], level=params['level'], path=params['path'],
            file=params['file'])
    except:
        print(f"params: {params}")
        raise
    logger = logging.getLogger(params['name'])
    logger.setLevel(CURRENT_LEVEL)
    for h in logger.handlers:
        if h.__class__ != logging.FileHandler:
            h.setLevel(CURRENT_LEVEL)

    if adapter is not None:
        return adapter(logger, {})

    return logger



def config_logger(name, level=None, path=None, file=None):
    """Configs the logger with name `name`. Overwrites existing config."""
    logger = logging.getLogger(name)
    logger.propagate = False
    format = '%(levelname)-8s %(name)s -- %(message)s'
    formatter = logging.Formatter(format)
    if level is not None:
        if level.__class__ == str:
            level = LEVELS[level.upper()]
        logger.setLevel(level)
    existing_handlers = [h for h in logger.handlers]
    stream_handlers = sum(True for h in existing_handlers if h.__class__ == logging.StreamHandler)
    if stream_handlers == 0:
        streamHandler = logging.StreamHandler(sys.stdout)
        streamHandler.setFormatter(formatter)
        logger.addHandler(streamHandler)
    elif stream_handlers > 1:
        logger.info(f"The logger {name} has been setup with {stream_handlers} StreamHandlers and is probably sending every message twice.")

    log_file = None
    if file is not None:
        file = os.path.expanduser(file)
        if os.path.isabs(file):
            log_file = os.path.abspath(file)
        elif path is None:
            logger.warning(f"""Log file output cannot be configured for '{name}' because 'file' is relative ({file})
but no 'path' has been configured.""")
    if log_file is None and path is not None:
        path = os.path.expanduser(path)
        log_file = os.path.abspath(os.path.join(path, f"{name}.log"))

    if log_file is not None and not any(True for h in existing_handlers if h.__class__ == logging.FileHandler and h.baseFilename == log_file):
        log_dir, _ = os.path.split(log_file)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        logger.debug(f"Storing logs as {log_file}")
        fileHandler = logging.FileHandler(log_file, mode='a', delay=True)
        fileHandler.setLevel(LEVELS['W'])
        file_formatter = logging.Formatter("%(asctime)s "+format, datefmt='%Y-%m-%d %H:%M:%S')
        fileHandler.setFormatter(file_formatter)
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
        if l is None or l.__class__ == str:
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