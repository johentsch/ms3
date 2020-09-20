import os

from .bs4_parser import _MSCX_bs4
from .annotations import Annotations
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
        self._annotations = {}
        self.parser = parser
        if mscx_src is not None:
            self.parse_mscx(mscx_src)


    def handle_path(self, path, key=None):
        full_path = os.path.abspath(path)
        if os.path.isfile(full_path):
            file_path, file = os.path.split(full_path)
            file_name, file_ext = os.path.splitext(file)
            if key is None:
                key = file_ext[1:]
            elif key == 'file':
                key = file
            self.full_paths[key] = full_path
            self.paths[key] = file_path
            self.files[key] = file
            self.fnames[key] = file_name
            self.fexts[key] = file_ext
            self.logger_names[key] = file_name.replace('.', '')
            return key
        else:
            self.logger.error("No file found at this path: " + full_path)
            return None



    def parse_mscx(self, mscx_src, parser=None, logger_name=None):
        _ = self.handle_path(mscx_src)
        if parser is not None:
            self.parser = parser
        if 'mscx' in self.fnames:
            ln = self.logger_names['mscx'] if logger_name is None else logger_name
            self._mscx = MSCX(self.full_paths['mscx'], self.parser, logger_name=ln)
            if self._mscx.has_annotations:
                self._annotations['annotations'] = self._mscx._annotations
        else:
            self.logger.error("No .mscx file specified.")

    def output_mscx(self, filepath):
        self.mscx.output_mscx(filepath)

    def detach_labels(self, key, staff=None, voice=None, harmony_type=None):
        if 'annotations' not in self._annotations:
            self.logger.info("No annotations present in score.")
            return False
        df = self.annotations.get_labels(staff=staff, voice=voice, harmony_type=harmony_type, drop=True)
        if len(df) == 0:
            self.logger.info(f"No labels found for staff {staff}, voice {voice}, harmony_type {harmony_type}.")
            return False
        self._annotations[key] = Annotations(df=df)
        if len(self.annotations.df) == 0:
            self.mscx.has_annotations = False
            del (self._annotations['annotations'])
        self._mscx.delete_labels(df)
        return True


    def attach_labels(self, key):
        if key not in self._annotations:
            self.logger.info(f"""Key '{key}' doesn't correspond to a detached set of annotations.
Use on of the existing keys or load a new set with the method load_annotations().""")
            return False
        annotations = self._annotations[key]
        df = annotations.df
        if len(df) == 0:
            self.logger.warning(f"The annotation set '{key}' does not contain any labels meeting the criteria.")
            return False
        self._mscx.add_labels(df)
        self._annotations['annotations'] = self._mscx._annotations


    @property
    def mscx(self):
        """ Returns the `MSCX` object with the parsed score.
        """
        if self._mscx is None:
            raise LookupError("No XML has been parsed so far. Use the method parse_mscx().")
        return self._mscx


    def load_annotations(self, tsv_path, key=None):
        if key is None:
            key = 'file'
        key = self.handle_path(tsv_path, key)
        self._annotations[key] = Annotations(tsv_path)

    def __repr__(self):
        msg = ''
        if 'mscx' in self.full_paths:
            msg = f"MuseScore file"
            if self.mscx.changed:
                msg += " (CHANGED!!!)\n---------------!!!!!!!!!!!!"
            else:
                msg += "\n--------------"
            msg += f"\n\n{self.full_paths['mscx']}\n\n"
        if 'annotations' in self._annotations:
            msg += f"Attached annotations\n--------------------\n\n{self.annotations}\n\n"
        else:
            msg += "No annotations attached.\n\n"
        if sum(True for key in self._annotations if key != 'annotations') > 0:
            msg += "Detached annotations\n--------------------\n\n"
            for key, obj in self._annotations.items():
                if key != 'annotations':
                    msg += f"{key} -> {obj}\n\n"
        return msg

    def __getattr__(self, item):
        return self._annotations[item]

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
        self._annotations = None
        if parser is not None:
            self.parser = parser
        self.parsed = None
        self.changed = False

        assert os.path.isfile(self.mscx_src), f"{self.mscx_src} does not exist."

        implemented_parsers = ['bs4']
        if self.parser == 'bs4':
            self.parsed = _MSCX_bs4(self.mscx_src, logger_name=self.logger.name)
        else:
            raise NotImplementedError(f"Only the following parsers are available: {', '.join(implemented_parsers)}")

        self.output_mscx = self.parsed.output_mscx
        self.get_chords = self.parsed.get_chords
        self.get_harmonies = self.parsed.get_annotations
        self.get_metadata = self.parsed.get_metadata
        self.has_annotations = self.parsed.has_annotations
        if self.parsed.has_annotations:
            self._annotations = Annotations(df=self.parsed.get_annotations())

    def delete_labels(self, df):
        changes = sum(self.parsed.delete_label(mc, staff, voice, onset)
                      for mc, staff, voice, onset
                      in df[['mc', 'staff', 'voice', 'onset']].itertuples(name=None, index=False)
                      )
        if changes > 0:
            self.changed = True
            self.parsed.parse_measures()


    def add_labels(self, df, label='label', mc='mc', onset='onset', staff='staff', voice='voice', **kwargs):
        parameters = ['label', 'mc', 'onset', 'staff', 'voice'] + list(kwargs.keys())
        columns = [label, mc, onset, staff, voice] + list(kwargs.values())
        changes = sum(self.parsed.add_label(**{a: b for a, b in zip(parameters, t)})
                      for t
                      in df[columns].itertuples(index=False)
                      )
        if changes > 0:
            self.changed = True
            self.parsed.parse_measures()
            self._annotations = Annotations(df=self.parsed.get_annotations())


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

