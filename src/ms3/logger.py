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

CURRENT_LEVEL = logging.WARNING

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


def get_logger(name=None, level=None, adapter=ContextAdapter):
    """The function gets or creates the logger `name` and returns it, by default through the given LoggerAdapter class."""
    global CURRENT_LEVEL
    if isinstance(name, logging.LoggerAdapter):
        name = name.logger
    if isinstance(name, logging.Logger):
        if level is None:
            level = name.level
        name = name.name
    if name not in logging.root.manager.loggerDict:
        config_logger(name)
    logger = logging.getLogger(name)

    if level is not None:
        CURRENT_LEVEL = LEVELS[level.upper()] if level.__class__ == str else level
    logger.setLevel(CURRENT_LEVEL)
    for h in logger.handlers:
        h.setLevel(CURRENT_LEVEL)

    if adapter is not None:
        return adapter(logger, {})

    return logger



def config_logger(name, level=None, logfile=None):
    """Configs the logger with name `name`. Overwrites existing config."""
    logger = logging.getLogger(name)
    logger.propagate = False
    format = '%(levelname)-7s %(name)s -- %(message)s'
    formatter = logging.Formatter(format)
    if level is not None:
        if level.__class__ == str:
            level = LEVELS[level]
        logger.setLevel(level)
    existing_handlers = [h for h in logger.handlers]
    stream_handlers = sum(True for h in existing_handlers if h.__class__ == logging.StreamHandler)
    if stream_handlers == 0:
        streamHandler = logging.StreamHandler(sys.stdout)
        streamHandler.setFormatter(formatter)
        logger.addHandler(streamHandler)
    elif stream_handlers > 1:
        logger.info(f"The logger {name} has been setup with {stream_handlers} StreamHandlers and is probably sending every message twice.")
    if logfile is not None:
        if not any(True for h in existing_handlers if h.__class__ == logging.FileHandler and h.baseFilename == logfile):
            fileHandler = logging.FileHandler(logfile, mode='w')
            fileHandler.setFormatter(formatter)
            logger.addHandler(fileHandler)


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
        func_globals.update({'logger': logg})
        try:
            result = f(*args, **kwargs)
        finally:
            func_globals = saved_values  # Undo changes.
        return result

    return logger
