import os

from .bs4_parser import _MSCX_bs4
from .logger import get_logger






class Score:
    """ Object representing a score.

    Attributes
    ----------
    mscx_src : :obj:`str`, optional
        Path to the MuseScore file to be parsed.
    parser : {'bs4'}, optional
        Which parser to use.
    logger_name : :obj:`str`, optional
        If you have defined a logger, pass its name.
    level : {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}, optional
        Pass a level name for which (and above which) you want to see log records.

    paths, files, fnames, fexts, logger_names : :obj:`dict`
        Dictionaries for keeping track of file information
    _mscx : :obj:`MSCX`
        After parsing, holds the `MSCX` object with the parsed score.

    Methods
    -------
    handle_path(path, key)
        Puts the path into `paths, files, fnames, fexts` dicts with the given key.
    """

    def __init__(self, mscx_src=None, parser='bs4', logger_name='Score', level=None):
        self.logger = get_logger(logger_name, level)
        self.full_paths, self.paths, self.files, self.fnames, self.fexts, self.logger_names = {}, {}, {}, {}, {}, {}
        self._mscx = None
        self.handle_path(mscx_src)
        self.parser = parser
        if 'mscx' in self.fnames:
            self.parse_mscx()
        elif mscx_src is not None:
            self.logger.error("At the moment, only .mscx files are accepted.")

    def handle_path(self, path, key=None):
        if path is not None:
            file_path, file = os.path.split(os.path.abspath(path))
            file_name, file_ext = os.path.splitext(file)
            if key is None:
                key = file_ext.replace('.', '', 1)
            self.full_paths[key] = path
            self.paths[key] = file_path
            self.files[key] = file
            self.fnames[key] = file_name
            self.fexts[key] = file_ext
            self.logger_names[key] = file_name.replace('.', '')



    def parse_mscx(self, mscx_src=None, parser=None, logger_name=None):
        self.handle_path(mscx_src)
        if parser is not None:
            self.parser = parser
        if 'mscx' in self.fnames:
            ln = self.logger_names['mscx'] if logger_name is None else logger_name
            self._mscx = MSCX(self.full_paths['mscx'], self.parser, logger_name=ln)
        else:
            self.logger.error("At the moment, only .mscx files are accepted.")

    def output_mscx(self, filepath):
        self.mscx.output_mscx(filepath)

    @property
    def mscx(self):
        """ Returns the `MSCX` object with the parsed score.
        """
        if self._mscx is None:
            raise LookupError("No XML has been parsed so far. Use the method parse_mscx().")
        return self._mscx


class MSCX:
    """ Object for interacting with the XML structure of a MuseScore 3 file.

    Attributes
    ----------
    mscx_src : :obj:`str`
        MuseScore 3 file to parse.
    parsed : :obj:`_MSCX_bs4`
        Holds the MSCX score parsed by the selected parser.
    parser : :obj:`str`, optional
        Which XML parser to use.
    logger_name : :obj:`str`, optional
        If you have defined a logger, pass its name.
    level : {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}, optional
        Pass a level name for which (and above which) you want to see log records.

    Methods
    -------
    output_mscx(filepath)
        Write the internal score representation to a file.
    """

    def __init__(self, mscx_src=None, parser='bs4', logger_name='MSCX', level=None):
        self.logger = get_logger(logger_name, level=level)
        self.mscx_src = mscx_src
        if parser is not None:
            self.parser = parser
        self.parsed = None

        assert os.path.isfile(self.mscx_src), f"{self.mscx_src} does not exist."

        implemented_parsers = ['bs4']
        if self.parser == 'bs4':
            self.parsed = _MSCX_bs4(self.mscx_src, logger_name=self.logger.name)
        else:
            raise NotImplementedError(f"Only the following parsers are available: {', '.join(implemented_parsers)}")

        self.output_mscx = self.parsed.output_mscx
        self.get_chords = self.parsed.get_chords
        self.get_harmonies = self.parsed.get_harmonies

    @property
    def measures(self):
        return self.parsed.measures

    @property
    def events(self):
        return self.parsed._events

    @property
    def chords(self):
        return self.parsed.chords

    @property
    def notes(self):
        return self.parsed.notes

    @property
    def rests(self):
        return self.parsed.rests

    @property
    def notes_and_rests(self):
        return self.parsed.notes_and_rests

    @property
    def version(self):
        """MuseScore version with which the file was created (read-only)."""
        return self.parsed.version



