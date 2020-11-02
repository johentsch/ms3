import os, re

import pandas as pd

from .utils import decode_harmonies, no_collections_no_booleans, resolve_dir, unpack_mscz, update_labels_cfg, update_cfg
from .bs4_parser import _MSCX_bs4
from .annotations import Annotations
from .logger import LoggedClass


class Score(LoggedClass):
    """ Object representing a score.
    """

    ABS_REGEX = r"^\(?[A-G|a-g](b*|#*).*?(/[A-G|a-g](b*|#*))?$"
    """ :obj:`str`
    Class variable with a regular expression that
    recognizes absolute chord symbols in their decoded (string) form; they start with a note name.
    """

    DCML_REGEX = re.compile(r"""
                                ^(?P<first>
                                  (\.?
                                    ((?P<globalkey>[a-gA-G](b*|\#*))\.)?
                                    ((?P<localkey>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\.)?
                                    ((?P<pedal>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\[)?
                                    (?P<chord>
                                        (?P<numeral>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))
                                        (?P<form>(%|o|\+|M|\+M))?
                                        (?P<figbass>(7|65|43|42|2|64|6))?
                                        (\((?P<changes>((\+|-|\^)?(b*|\#*)\d)+)\))?
                                        (/(?P<relativeroot>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)*))?
                                    )
                                    (?P<pedalend>\])?
                                  )?
                                  (?P<phraseend>(\\\\|\{|\}|\}\{)
                                  )?
                                 )
                                 (-
                                  (?P<second>
                                    ((?P<globalkey2>[a-gA-G](b*|\#*))\.)?
                                    ((?P<localkey2>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\.)?
                                    ((?P<pedal2>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\[)?
                                    (?P<chord2>
                                        (?P<numeral2>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))
                                        (?P<form2>(%|o|\+|M|\+M))?
                                        (?P<figbass2>(7|65|43|42|2|64|6))?
                                        (\((?P<changes2>((\+|-|\^)?(b*|\#*)\d)+)\))?
                                        (/(?P<relativeroot2>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)*))?
                                    )
                                    (?P<pedalend2>\])?
                                  )?
                                  (?P<phraseend2>(\\\\|\{|\}|\}\{)
                                  )?
                                 )?
                                $
                                """,
                            re.VERBOSE)
    """:obj:`str`
    Class variable with a regular expression that
    recognizes labels conforming to the DCML harmony annotation standard.
    """

    NASHVILLE_REGEX = r"^(b*|#*)(\d).*$"
    """:obj:`str`
    Class variable with a regular expression that
    recognizes labels representing a Nashville numeral, which MuseScore is able to encode.
    """

    RN_REGEX = r"^$"
    """:obj:`str`
    Class variable with a regular expression for Roman numerals that
    romentarily matches nothing because ms3 tries interpreting Roman Numerals
    als DCML harmony annotations.
    """

    def __init__(self, musescore_file=None, infer_label_types=['dcml'], read_only=False, labels_cfg={}, logger_cfg={},
                 parser='bs4'):
        """

        Parameters
        ----------
        musescore_file : :obj:`str`, optional
            Path to the MuseScore file to be parsed.
        infer_label_types : :obj:`list` or :obj:`dict`, optional
            Determine which label types are determined automatically. Defaults to ['dcml'].
            Pass ``[]`` to infer only main types 0 - 3.
            Pass ``{'type_name': r"^(regular)(Expression)$"}`` to call :meth:`ms3.Score.new_type`.
        read_only : :obj:`bool`, optional
            Defaults to ``False``, meaning that the parsing is slower and uses more memory in order to allow for manipulations
            of the score, such as adding and deleting labels. Set to ``True`` if you're only extracting information.
        labels_cfg : :obj:`dict`
            Store a configuration dictionary to determine the output format of the :py:class:`~ms3.annotations.Annotations`
            object representing the currently attached annotations. See :obj:`MSCX.labels_cfg`.
        logger_cfg : :obj:`dict`, optional
            The following options are available:
            'name': LOGGER_NAME -> by default the logger name is based on the parsed file(s)
            'level': {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}
            'file': PATH_TO_LOGFILE to store all log messages under the given path.
        parser : 'bs4', optional
            The only XML parser currently implemented is BeautifulSoup 4.
        """
        super().__init__(subclass='Score', logger_cfg=logger_cfg)

        self.full_paths = {}
        """:obj:`dict`
        ``{KEY: {i: full_path}}`` dictionary holding the full paths of all parsed MuseScore and TSV files,
        including file names. Handled internally by :py:meth:`~ms3.score.Score._handle_path`.
        """

        self.paths = {}
        """:obj:`dict`
        ``{KEY: {i: file path}}`` dictionary holding the paths of all parsed MuseScore and TSV files,
        excluding file names. Handled internally by :py:meth:`~ms3.score.Score._handle_path`.
        """

        self.files = {}
        """:obj:`dict`
        ``{KEY: {i: file name with extension}}`` dictionary holding the complete file name  of each parsed file,
        including the extension. Handled internally by :py:meth:`~ms3.score.Score._handle_path`.
        """

        self.fnames = {}
        """:obj:`dict`
        ``{KEY: {i: file name without extension}}`` dictionary holding the file name  of each parsed file,
        without its extension. Handled internally by :py:meth:`~ms3.score.Score._handle_path`.
        """

        self.fexts = {}
        """:obj:`dict`
        ``{KEY: {i: file extension}}`` dictionary holding the file extension of each parsed file.
        Handled internally by :py:meth:`~ms3.score.Score._handle_path`.
        """

        self._mscx = None
        """:obj:`MSCX`
        The object representing the parsed MuseScore file.
        """

        self._annotations = {}
        """:obj:`dict`
        ``{(key, i): Annotations object}`` dictionary for accessing all :py:class:`~ms3.annotations.Annotations` objects.            """

        self._types_to_infer = []
        """:obj:`list`
        Current order in which types are being recognized."""

        self._label_types = {
            0: "Simple string (should not begin with a note name, otherwise MS3 will turn it into type 3; prevent through leading dot)",
            1: "MuseScore's Roman Numeral Annotation format",
            2: "MuseScore's Nashville Number format",
            3: "Absolute chord encoded by MuseScore",
            'dcml': "Latest version of the DCML harmonic annotation standard.",
        }
        """:obj:`dict`
        Mapping label types to their descriptions.
        0: "Simple string (should not begin with a note name, otherwise MS3 will turn it into type 3; prevent through leading dot)",
        1: "MuseScore's Roman Numeral Annotation format",
        2: "MuseScore's Nashville Number format",
        3: "Absolute chord encoded by MuseScore",
        'dcml': "Latest version of the DCML harmonic annotation standard.",
        """

        self._label_regex = {
            1: self.RN_REGEX,
            2: self.NASHVILLE_REGEX,
            3: self.ABS_REGEX,
            'dcml': self.DCML_REGEX,
        }
        """:obj:`dict`
        Mapping label types to their corresponding regex. Managed via the property :py:meth:`infer_label_types`.
        1: self.rn_regex,
        2: self.nashville_regex,
        3: self.abs_regex,
        'dcml': self.dcml_regex,
        """

        self.parser = parser
        """{'bs4'}
        Currently only one XML parser has been implemented which uses BeautifulSoup 4.
        """

        self.infer_label_types = infer_label_types
        if musescore_file is not None:
            self._parse_mscx(musescore_file, read_only=read_only, labels_cfg=labels_cfg)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END of __init__() %%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    @property
    def infer_label_types(self):
        """:obj:`list` or :obj:`dict`, optional
        The order in which label types are to be inferred.
        Assigning a new value results in a call to :py:meth:`~ms3.annotations.Annotations.infer_types`.
        Passing a {label type: regex} dictionary is a shortcut to update type regex's or to add new ones.
        The inference will take place in the order in which they appear in the dictionary. To reuse an existing
        regex will updating others, you can refer to them as None, e.g. ``{'dcml': None, 'my_own': r'^(PAC|HC)$'}``.
        """
        return self._types_to_infer

    @infer_label_types.setter
    def infer_label_types(self, val):
        if val is None:
            val = []
        before_inf, before_reg = self._types_to_infer, self.get_infer_regex()
        if isinstance(val, list):
            exist = [v for v in val if v in self._label_regex]
            if len(exist) < len(val):
                logger.warning(
                    f"The following harmony types have not been added via the new_type() method:\n{[v for v in val if v not in self._label_regex]}")
            self._types_to_infer = exist
        elif isinstance(val, dict):
            for k, v in val.items():
                if k in self._label_regex:
                    if v is None:
                        val[k] = self._label_regex[k]
                    else:
                        self._label_regex[k] = v
                else:
                    self.new_type(name=k, regex=v)
            self._types_to_infer = list(val.keys())
        after_reg = self.get_infer_regex()
        if before_inf != self._types_to_infer or before_reg != after_reg:
            for ann in self._annotations.values():
                ann.infer_types(after_reg)

    @property
    def has_detached_annotations(self):
        """:obj:`bool`
        Is True as long as the score contains :py:class:`~ms3.annotations.Annotations` objects, that are not attached to the :obj:`MSCX` object.
        """
        return sum(True for key in self._annotations if key != 'annotations') > 0

    @property
    def mscx(self):
        """:obj:`MSCX`
        Standard way of accessing the parsed MuseScore file."""
        if self._mscx is None:
            raise LookupError("No XML has been parsed so far. Use the method parse_mscx().")
        return self._mscx

    @property
    def types(self):
        """:obj:`dict`
        Shows the mapping of label types to their descriptions."""
        return self._label_types


    def attach_labels(self, key, staff=None, voice=None, check_for_clashes=True):
        """ Insert detached labels ``key`` into this score's :obj:`MSCX` object.

        Parameters
        ----------
        key : :obj:`str`
            Key of the detached labels you want to insert into the score.
        staff, voice : :obj:`int`, optional
            Pass one or both of these arguments to change the original annotation layer or if there was none.
        check_for_clashes : :obj:`bool`, optional
            Defaults to True, meaning that the positions where the labels will be inserted will be checked for existing
            labels.

        Returns
        -------
        :obj:`int`
            Number of newly attached labels.
        :obj:`int`
            Number of labels that were to be attached.
        """
        assert key != 'annotations', "Labels with key 'annotations' are already attached."
        if key not in self._annotations:
            self.logger.info(f"""Key '{key}' doesn't correspond to a detached set of annotations.
Use one of the existing keys or load a new set with the method load_annotations().\nExisting keys: {list(self._annotations.keys())}""")
            return 0, 0

        annotations = self._annotations[key]
        goal = len(annotations.df)
        if goal == 0:
            self.logger.warning(f"The Annotation object '{key}' does not contain any labels.")
            return 0, 0
        df = annotations.prepare_for_attaching(staff=staff, voice=voice, check_for_clashes=check_for_clashes)
        reached = len(df)
        if reached == 0:
            self.logger.error(f"No labels from '{key}' have been attached due to aforementioned errors.")
            return reached, goal

        reached = self._mscx.add_labels(df, label=annotations.cols['label'])
        self._annotations['annotations'] = self._mscx._annotations
        if len(self._mscx._annotations.df) > 0:
            self._mscx.has_annotations = True
        return reached, goal


    def detach_labels(self, key, staff=None, voice=None, label_type=None, delete=True):
        """ Detach all annotations labels from this score's :obj:`MSCX` object or just a selection of them.
        The extracted labels are stored as a new :py:class:`~ms3.annotations.Annotations` object that is accessible via ``Score.{key}``.
        By default, ``delete`` is set to True, meaning that if you call :py:meth:`store_mscx` afterwards,
        the created MuseScore file will not contain the detached labels.

        Parameters
        ----------
        key : :obj:`str`
            Specify a new key for accessing the detached set of annotations. The string needs to be usable
            as an identifier, e.g. not start with a number, not contain special characters etc. In return you
            may use it as a property: For example, passing ``'chords'`` lets you access the detached labels as
            ``Score.chords``. The key 'annotations' is reserved for all annotations attached to the score.
        staff : :obj:`int`, optional
            Pass a staff ID to select only labels from this staff. The upper staff has ID 1.
        voice : {1, 2, 3, 4}, optional
            Can be used to select only labels from one of the four notational layers.
            Layer 1 is MuseScore's main, 'upper voice' layer, coloured in blue.
        label_type : :obj:`int` or :obj:`str`, optional
            Select one of the predefined or custom label types to select only labels of this type.
            Predefined types are {0, 1, 2, 3, 'dcml'} (see :py:attr:`_label_types`).
        delete : :obj:`bool`, optional
            By default, the labels are removed from the XML structure in :obj:`MSCX`.
            Pass False if you want them to remain. This could be useful if you only want to extract a subset
            of the annotations for storing them separately but without removing the labels from the score.
        """
        if 'annotations' not in self._annotations:
            self.logger.info("No annotations present in score.")
            return
        assert key != 'annotations', "The key 'annotations' is reserved, please choose a different one."
        if not key.isidentifier():
            self.logger.warning(
                f"'{key}' can not be used as an identifier. The extracted labels need to be accessed via self._annotations['{key}']")
        df = self.annotations.get_labels(staff=staff, voice=voice, label_type=label_type, drop=delete)
        if len(df) == 0:
            self.logger.info(f"No labels found for staff {staff}, voice {voice}, label_type {label_type}.")
            return
        logger_cfg = self.logger_cfg.copy()
        logger_cfg['name'] += f"{self.logger_names['mscx']}:{key}"
        self._annotations[key] = Annotations(df=df, infer_types=self.get_infer_regex(), mscx_obj=self._mscx,
                                             logger_cfg=logger_cfg)
        if delete:
            self._mscx.delete_labels(df)
        if len(self._annotations['annotations'].df) == 0:
            self._mscx.has_annotations = False
            del (self._annotations['annotations'])
        return


    def get_infer_regex(self):
        """
        Returns
        -------
        :obj:`dict`
            Mapping of label types to the corresponding regular expressions
            in the order in which they are currently set to be inferred.
        """
        return {t: self._label_regex[t] for t in self._types_to_infer}


    def new_type(self, name, regex, description='', infer=True):
        """ Declare a custom label type. A type consists of a name, a regular expression and,
        falculatively, of a description.

        Parameters
        ----------
        name : :obj:`str` or :obj:`int`
            Name of the custom label type.
        regex : :obj:`str`
            Regular expression that matches all labels of the custom type.
        description : :obj:`str`, optional
            Human readable description that appears when calling the property ``Score.types``.
        infer : :obj:`bool`, optional
            By default, the labels of all :py:class:`~ms3.annotations.Annotations` objects are matched against the new type.
            Pass False to not change any label's type.
        """
        assert name not in self._label_types, f"'{name}' already added to types: {self._label_types[name]}"
        self._label_types[name] = description
        self._label_regex[name] = regex
        if infer:
            self._types_to_infer.insert(0, name)
            for ann in self._annotations.values():
                ann.infer_types(self.get_infer_regex())

    def load_annotations(self, tsv_path=None, anno_obj=None, key='tsv', cols={}, infer=True):
        """ Attach an :py:class:`~ms3.annotations.Annotations` object to the score and make it available as ``Score.{key}``.
        It can be an existing object or one newly created from the TSV file ``tsv_path``.

        Parameters
        ----------
        tsv_path : :obj:`str`
            If you want to create a new :py:class:`~ms3.annotations.Annotations` object from a TSV file, pass its path.
        anno_obj : :py:class:`~ms3.annotations.Annotations`
            Instead, you can pass an existing object.
        key : :obj:`str`, defaults to 'tsv'
            Specify a new key for accessing the set of annotations. The string needs to be usable
            as an identifier, e.g. not start with a number, not contain special characters etc. In return you
            may use it as a property: For example, passing ``'chords'`` lets you access the :py:class:`~ms3.annotations.Annotations` as
            ``Score.chords``. The key 'annotations' is reserved for all annotations attached to the score.
        cols : :obj:`dict`, optional
            If the columns in the specified TSV file diverge from the :ref:`standard column names<column_names>`,
            pass a {standard name: custom name} dictionary.
        infer : :obj:`bool`, optional
            By default, the label types are inferred in the currently configured order (see :py:attr:`infer_label_types`).
            Pass False to not add and not change any label types.
        """
        assert sum(True for arg in [tsv_path, anno_obj] if arg is not None) == 1, "Pass either tsv_path or anno_obj."
        inf_dict = self.get_infer_regex() if infer else {}
        mscx = None if self._mscx is None else self._mscx
        if tsv_path is not None:
            key = self._handle_path(tsv_path, key)
            logger_cfg = self.logger_cfg.copy()
            logger_cfg['name'] = f"{self.logger_names[key]}"
            self._annotations[key] = Annotations(tsv_path=tsv_path, infer_types=inf_dict, cols=cols, mscx_obj=mscx,
                                                 logger_cfg=logger_cfg)
        else:
            anno_obj.mscx_obj = mscx
            self._annotations[key] = anno_obj

    def store_annotations(self, key=None, tsv_path=None, **kwargs):
        """ Save a set of annotations as TSV file. While ``store_list`` stores attached labels only, this method
        can also store detached labels by passing a ``key``.

        Parameters
        ----------
        key : :obj:`str`, optional
            Key of the :py:class:`~ms3.annotations.Annotations` object which you want to output as TSV file.
            By default, the annotations attached to the score (key='annotations') are stored.
        tsv_path : :obj:`str`, optional
            Path of the newly created TSV file including the file name.
            By default, the TSV file is stored next to t
        kwargs
            Additional keyword arguments will be passed to the function :py:meth:`pandas.DataFrame.to_csv` to
            customise the format of the created file (e.g. to change the separator to commas instead of tabs,
            you would pass ``sep=','``).
        """
        if key is None:
            assert self._mscx.has_annotations, "Score has no labels attached."
            key = 'annotations'
        assert key in self._annotations, f"Key '{key}' not found. Available keys: {list(self._annotations.keys())}"
        if tsv_path is None:
            if 'mscx' in self.paths:
                path = self.paths['mscx']
                fname = self.fnames['mscx']
                tsv_path = os.path.join(path, fname + '_labels.tsv')
            else:
                self.logger.warning(f"No tsv_path has been specified and no MuseScore file has been parsed to infer one.")
                return
        if self._annotations[key].store_tsv(tsv_path, **kwargs):
            new_key = self._handle_path(tsv_path, key=key)
            if key != 'annotations':
                self._annotations[key].update_logger_cfg({'name': self.logger_names[new_key]})


    def store_mscx(self, filepath):
        """ Store the current :obj:`MSCX` object attached to this score as uncompressed MuseScore file.
        Just a shortcut for ``Score.mscx.store_mscx()``.

        Parameters
        ----------
        filepath : :obj:`str`
            Path of the newly created MuseScore file, including the file name ending on '.mscx'.
            Uncompressed files ('.mscz') are not supported.
        """
        return self.mscx.store_mscx(filepath)

    def _handle_path(self, path, key=None, logger_name=None):
        """ Puts the path into ``paths, files, fnames, fexts`` dicts with the given key.

        Parameters
        ----------
        path : :obj:`str`
            Full file path.
        key : :obj:`str`, optional
            The key chosen by the user. By default, the key is automatically assigend to be the file's extension.
        logger_name : :obj:`str`, optional
            By default, a logger name is generated for the file from its file name with extension, stripped of all dots.
            To designate a different logger to go with this file, pass its name.
        """
        full_path = resolve_dir(path)
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
            if logger_name is None:
                self.logger_names[key] = file_name.replace('.', '')
            else:
                self.logger_names[key] = logger_name
            return key
        else:
            self.logger.error("No file found at this path: " + full_path)
            return None

    def _parse_mscx(self, musescore_file, read_only=False, parser=None, labels_cfg={}):
        """ 
        This method is called by :py:meth:`.__init__` to parse the score. It checks the file extension
        and in the case of a compressed MuseScore file (.mscz), a temporary uncompressed file is generated
        which is removed after the parsing process.
        Essentially, parsing means to initiate a :obj:`MSCX` object and to make it available as ``Score.mscx``
        and, if the score includes annotations, to initiate an :py:class:`~ms3.annotations.Annotations` object that
        can be accessed as ``Score.annotations``.
        The method doesn't systematically clean up data from a hypothetical previous parse.

        Parameters
        ----------
        musescore_file : :obj:`str`
            Path to the MuseScore file to be parsed.
        read_only : :obj:`bool`, optional
            Defaults to ``False``, meaning that the parsing is slower and uses more memory in order to allow for manipulations
            of the score, such as adding and deleting labels. Set to ``True`` if you're only extracting information.
        parser : 'bs4', optional
            The only XML parser currently implemented is BeautifulSoup 4.
        labels_cfg : :obj:`dict`, optional
            Store a configuration dictionary to determine the output format of the :py:class:`~ms3.annotations.Annotations`
            object representing the currently attached annotations. See :obj:`MSCX.labels_cfg`.
        """
        if parser is not None:
            self.parser = parser
        if musescore_file[-4:] not in ('mscx', 'mscz'):
            raise ValueError(f"The extension of a MuseScore file should be mscx or mscz, not {extensions}.")
        extension = self._handle_path(musescore_file)
        logger_cfg = self.logger_cfg.copy()
        logger_cfg['name'] = self.logger_names[extension]
        if extension == 'mscz':
            fake_path = musescore_file[:-4] + 'mscx'
            self._handle_path(fake_path)
            with unpack_mscz(musescore_file) as tmp_mscx:
                self._mscx = MSCX(tmp_mscx, read_only=read_only, labels_cfg=labels_cfg, parser=self.parser,
                                  logger_cfg=logger_cfg)
        else:
            self._mscx = MSCX(self.full_paths['mscx'], read_only=read_only, labels_cfg=labels_cfg, parser=self.parser,
                              logger_cfg=logger_cfg)
        if self._mscx.has_annotations:
            self._annotations['annotations'] = self._mscx._annotations
            self._annotations['annotations'].infer_types(self.get_infer_regex())


    def __repr__(self):
        msg = ''
        if 'mscx' in self.full_paths:
            msg = f"MuseScore file"
            if self._mscx.changed:
                msg += " (CHANGED!!!)\n---------------!!!!!!!!!!!!"
            else:
                msg += "\n--------------"
            msg += f"\n\n{self.full_paths['mscx']}\n\n"
        if 'annotations' in self._annotations:
            msg += f"Attached annotations\n--------------------\n\n{self.annotations}\n\n"
        else:
            msg += "No annotations attached.\n\n"
        if self.has_detached_annotations:
            msg += "Detached annotations\n--------------------\n\n"
            for key, obj in self._annotations.items():
                if key != 'annotations':
                    key_info = key + f" (stored as {self.files[key]})" if key in self.files else key
                    msg += f"{key_info} -> {obj}\n\n"
        return msg



    def __getattr__(self, item):
        try:
            return self._annotations[item]
        except:
            raise AttributeError(item)

    def __getitem__(self, item):
        try:
            return self._annotations[item]
        except:
            raise AttributeError(item)

    # def __setattr__(self, key, value):
    #     assert key != 'annotations', "The key 'annotations' is managed automatically, please pick a different one."
    #     assert key.isidentifier(), "Please use an alphanumeric key without special characters."
    #     if key in self.__dict__:
    #         self.__dict__[key] = value
    #     else:
    #         self._annotations[key] = value


class MSCX(LoggedClass):
    """ Object for interacting with the XML structure of a MuseScore 3 file. Is usually attached to a
    :obj:`Score` object and exposed as ``Score.mscx``.
    An object is only created if a score was successfully parsed.
    """

    def __init__(self, mscx_src, read_only=False, parser='bs4', labels_cfg={}, logger_cfg={}, level=None):
        """ Object for interacting with the XML structure of a MuseScore 3 file.

        Parameters
        ----------
        mscx_src : :obj:`str`
            Uncompressed MuseScore 3 file to parse.
        read_only : :obj:`bool`, optional
            Defaults to ``False``, meaning that the parsing is slower and uses more memory in order to allow for manipulations
            of the score, such as adding and deleting labels. Set to ``True`` if you're only extracting information.
        parser : :obj:`str`, optional
            Which XML parser to use.
        labels_cfg : :obj:`dict`, optional
            Store a configuration dictionary to determine the output format of the :py:class:`~ms3.annotations.Annotations`
            object representing the currently attached annotations. See :obj:`MSCX.labels_cfg`.
        logger_cfg : :obj:`dict`, optional
            The following options are available:
            'name': LOGGER_NAME -> by default the logger name is based on the parsed file(s)
            'level': {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}
            'file': PATH_TO_LOGFILE to store all log messages under the given path.
        level : :obj:`str` or :obj:`int`
            Quick way to change the logging level which defaults to the one of the parent :obj:`Score`.
        """
        if level is not None:
            logger_cfg['level'] = level
        super().__init__(subclass='MSCX', logger_cfg=logger_cfg)
        if os.path.isfile(mscx_src):
            self.mscx_src = mscx_src
            """:obj:`str`
            Full path of the parsed MuseScore file."""
        else:
            raise ValueError(f"File does not exist: {mscx_src}")

        self.read_only = read_only
        """:obj:`bool`, optional
        Shortcut for ``MSCX.parsed.read_only``.
        Defaults to ``False``, meaning that the parsing is slower and uses more memory in order to allow for manipulations
        of the score, such as adding and deleting labels. Set to ``True`` if you're only extracting information."""

        self._annotations = None
        """:py:class:`~ms3.annotations.Annotations` or None
        If the score contains at least one <Harmony> tag, this attribute points to the object representing all
        annotations, otherwise it is None."""

        self.parser = parser
        """{'bs4'}
        The currently used parser."""

        self._parsed = None
        """{:obj:`_MSCX_bs4`}
        Holds the MSCX score parsed by the selected parser (currently only BeautifulSoup 4 available)."""


        self.labels_cfg = {
            'staff': None,
            'voice': None,
            'label_type': None,
            'positioning': True,
            'decode': False,
            'column_name': 'label',
        }
        """:obj:`dict`
        Configuration dictionary to determine the output format of the loaded :py:class:`~ms3.annotations.Annotations`
        objects. The default options correspond to the default parameters of
        :py:meth:`Annotations.get_labels()<ms3.annotations.Annotations.get_labels>`.
        """
        self.labels_cfg.update(update_labels_cfg(labels_cfg, logger=self.logger))
        self.parse_mscx()
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END of __init__() %%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


    @property
    def changed(self):
        """:obj:`bool`
        Shortcut for ``MSCX.parsed.changed``.
        Switches to True as soon as the original XML structure is changed. Does not automatically switch back to False.
        """
        return self.parsed.changed

    @changed.setter
    def changed(self, val):
        self.parsed.changed = val


    @property
    def chords(self):
        """:obj:`pandas.DataFrame`
        DataFrame representing all <Chord> tags in the score. A chord in that sense is a grouping of all
        synchronous notes occurring in the same notational layer of the same staff. The DataFrame contains
        all kinds of score markup that is not attached to particular notes but to a <Chord>, such as
        slurs, lyrics, staff text, ottava lines etc.
        """
        return self._parsed.chords


    @property
    def events(self):
        """:obj:`pandas.DataFrame`
        DataFrame representating a raw skeleton of the score's XML structure and contains all score events,
        i.e. <Chord>, <Rest>, <Harmony> and markup tags such as <Beam> together with, in the columns the values of their
        XML properties and children. It serves as master for computing :obj:`.chords`, :obj:`rests`, and :obj:`labels`
        (and therefore :obj:`.expanded`, too)."""
        return self._parsed.events


    @property
    def expanded(self):
        """:obj:`pandas.DataFrame`
        DataFrame of labels that have been split into various features using a regular expression."""
        if self._annotations is None:
            return None
        labels_cfg = self.labels_cfg.copy()
        labels_cfg['decode'] = False
        return self._annotations.expand_dcml(**labels_cfg)


    @property
    def has_annotations(self):
        """:obj:`bool`
        Shortcut for ``MSCX.parsed.has_annotations``.
        Is True as long as at least one label is attached to the current XML."""
        return self.parsed.has_annotations

    @has_annotations.setter
    def has_annotations(self, val):
        self.parsed.has_annotations = val


    @property
    def labels(self):
        """:obj:`pandas.DataFrame`
        DataFrame representing all <Harmony> tags in the score as returned by calling :py:meth:`~ms3.annotations.Annotations.get_labels`
        on the object at :obj:`._annotations` with the current :obj:`._labels_cfg`."""
        if self._annotations is None:
            self.logger.info("The score does not contain any annotations.")
            return None
        return self._annotations.get_labels(**self.labels_cfg)


    @property
    def measures(self):
        """:obj:`pandas.DataFrame`
        DataFrame representing the measures of the MuseScore file (which can be incomplete measures).
        The potentially incomplete measure units are numbered starting from one, which corresponds to the
        "bar count" displayed in MuseScore 3's status bar. This numbering is represented in the column :ref:`mc<mc>`.
        (measure count). The columns represent for every MC its :ref:`actual duration<act_dur>`, its
        :ref:`time signature<timesig>`, how it is to be considered when computing measure numbers (:ref:`mn<mn>`),
        and which other MCs can "come :ref:`next`" according to the score's repeat structure."""
        return self._parsed.ml


    @property
    def metadata(self):
        """:obj:`dict`
        Shortcut for ``MSCX.parsed.metadata``.
        Metadata from and about the MuseScore file."""
        return self.parsed.metadata


    @property
    def notes(self):
        """:obj:`pandas.DataFrame`
        DataFrame representing all <Note> tags within the score."""
        return self._parsed.nl


    @property
    def notes_and_rests(self):
        """:obj:`pandas.DataFrame`
        The union of :obj:`.notes` and :obj:`.rests`."""
        return self._parsed.notes_and_rests


    @property
    def parsed(self):
        """{:obj:`_MSCX_bs4`}
        Standard way of accessing the object exposed by the current parser. :obj:`MSCX` uses this object's
        interface for requesting manipulations of and information from the source XML."""
        if self._parsed is None:
            self.logger.error("Score has not been parsed yet.")
            return None
        return self._parsed


    @property
    def rests(self):
        """:obj:`pandas.DataFrame`
        DataFrame representing all <Rest> tags."""
        return self._parsed.rl


    @property
    def staff_ids(self):
        """:obj:`list` of :obj:`int`
        The staff IDs contained in the score, usually just a list of increasing numbers starting at 1."""
        return self._parsed.staff_ids


    @property
    def version(self):
        """:obj:`str`
        MuseScore version that the file was created with."""
        return self._parsed.version


    def add_labels(self, df, label='label', mc='mc', mc_onset='mc_onset', staff='staff', voice='voice', **kwargs):
        """ Receives the labels from an :py:class:`~ms3.annotations.Annotations` object and adds them to the XML structure
        representing the MuseScore file that might be written to a file afterwards.

        Parameters
        ----------
        df : :obj:`pandas.DataFrame`
            DataFrame with labels to be added.
        label, mc, mc_onset, staff, voice : :obj:`str`
            Names of the DataFrame columns for the five required parameters.
        kwargs:
            label_type, root, base, leftParen, rightParen, offset_x, offset_y, nashville
                For these parameters, the standard column names are used automatically if the columns are present.
                If the column names have changed, pass them as kwargs, e.g. ``base='name_of_the_base_column'``

        Returns
        -------
        :obj:`int`
            Number of actually added labels.

        """
        if len(df) == 0:
            self.logger.info("Nothing to add.")
            return
        cols = dict(
            label_type='label_type',
            root='root',
            base='base',
            leftParen='leftParen',
            rightParen='rightParen',
            offset_x='offset:x',
            offset_y='offset:y',
            nashville='nashville',
            decoded='decoded'
        )
        missing_additional = {k: v for k, v in kwargs.items() if v not in df.columns}
        if len(missing_additional) > 0:
            self.logger.warning(f"The following specified columns could not be found:\n{missing_additional}.")
        main_params = ['label', 'mc', 'mc_onset', 'staff', 'voice']
        l = locals()
        missing_main = {k: l[k] for k in main_params if l[k] not in df.columns}
        assert len(
            missing_main) == 0, f"The specified columns for the following main parameters are missing:\n{missing_main}"
        main_cols = {k: l[k] for k in main_params}
        cols.update(kwargs)
        if cols['decoded'] not in df.columns:
            df[cols['decoded']] = decode_harmonies(df, label_col=label, return_series=True)
        add_cols = {k: v for k, v in cols.items() if v in df.columns}
        param2cols = {**main_cols, **add_cols}
        parameters = list(param2cols.keys())
        columns = list(param2cols.values())
        self.logger.debug(f"add_label() will be called with this param2col mapping:\n{param2cols}")
        tups = tuple(df[columns].itertuples(index=False, name=None))
        params = [{a: b for a, b in zip(parameters, t)} for t in tups]
        res = [self._parsed.add_label(**p) for p in params]
        changes = sum(res)
        # changes = sum(self.parsed.add_label(**{a: b for a, b in zip(parameters, t)})
        #               for t
        #               in df[columns].itertuples(index=False, name=None)
        #               )
        if changes > 0:
            self.changed = True
            self._parsed.parse_measures()
            logger_cfg = self.logger_cfg.copy()
            logger_cfg['name'] += ':annotations'
            self._annotations = Annotations(df=self.get_raw_labels(), read_only=True, mscx_obj=self,
                                            logger_cfg=logger_cfg)
            self.logger.debug(f"{changes}/{len(df)} labels successfully added to score.")
        return changes


    def change_labels_cfg(self, labels_cfg={}, staff=None, voice=None, label_type=None, positioning=None, decode=None):
        """ Update :obj:`MSCX.labels_cfg`.

        Parameters
        ----------
        labels_cfg : :obj:`dict`
            Using an entire dictionary or, to change only particular options, choose from:
        staff, voice, label_type, positioning, decode
            Arguments as they will be passed to :py:meth:`~ms3.annotations.Annotations.get_labels`
        """
        for k in self.labels_cfg.keys():
            val = locals()[k]
            if val is not None:
                labels_cfg[k] = val
        self.labels_cfg.update(update_labels_cfg(labels_cfg), logger=self.logger)



    def delete_labels(self, df):
        """ Delete a set of labels from the current XML.

        Parameters
        ----------
        df : :obj:`pandas.DataFrame`
            A DataFrame with the columns ['mc', 'mc_onset', 'staff', 'voice']
        """
        changed = pd.Series([self._parsed.delete_label(mc, staff, voice, mc_onset)
                             for mc, staff, voice, mc_onset
                             in reversed(
                list(df[['mc', 'staff', 'voice', 'mc_onset']].itertuples(name=None, index=False)))],
                            index=df.index)
        changes = changed.sum()
        if changes > 0:
            self.changed = True
            self._parsed.parse_measures()
            target = len(df)
            self.logger.debug(f"{changes}/{target} labels successfully deleted.")
            if changes < target:
                self.logger.warning(f"{target - changes} labels have not been deleted:\n{df.loc[~changed]}")


    def get_chords(self, staff=None, voice=None, mode='auto', lyrics=False, staff_text=False, dynamics=False,
                   articulation=False, spanners=False, **kwargs):
        """ Retrieve a customized chord list, e.g. one including less of the processed features or additional,
        unprocessed ones compared to the standard chord list.

        Parameters
        ----------
        staff : :obj:`int`
            Get information from a particular staff only (1 = upper staff)
        voice : :obj:`int`
            Get information from a particular voice only (1 = only the first layer of every staff)
        mode : {'auto', 'all', 'strict'}, optional
            * 'auto' (default), meaning that those aspects are automatically included that occur in the score; the resulting
              DataFrame has no empty columns except for those parameters that are set to True.
            * 'all': Columns for all aspects are created, even if they don't occur in the score (e.g. lyrics).
            * 'strict': Create columns for exactly those parameters that are set to True, regardless which aspects occur in the score.
        lyrics : :obj:`bool`, optional
            Include lyrics.
        staff_text : :obj:`bool`, optional
            Include staff text such as tempo markings.
        dynamics : :obj:`bool`, optional
            Include dynamic markings such as f or p.
        articulation : :obj:`bool`, optional
            Include articulation such as arpeggios.
        spanners : :obj:`bool`, optional
            Include spanners such as slurs, 8va lines, pedal lines etc.
        **kwargs : :obj:`bool`, optional
            Set a particular keyword to True in order to include all columns from the _events DataFrame
            whose names include that keyword. Column names include the tag names from the MSCX source code.

        Returns
        -------
        :obj:`pandas.DataFrame`
            DataFrame representing all <Chord> tags in the score with the selected features.
        """
        return self.parsed.get_chords(staff=staff, voice=voice, mode=mode, lyrics=lyrics, staff_text=staff_text,
                                      dynamics=dynamics,
                                      articulation=articulation, spanners=spanners, **kwargs)


    def get_raw_labels(self):
        """Shortcut for ``MSCX.parsed.get_raw_labels()``.
        Retrieve a "raw" list of labels, meaning that label types reflect only those defined within <Harmony> tags
        which can be 1 (Nashville), 2 (MuseScore's Roman Numeral display) or undefined (in the case of 'normal'
        chord labels, defaulting to 0).

        Returns
        -------
        :obj:`pandas.DataFrame`
            DataFrame with raw label features (i.e. as encoded in XML)
        """
        return self.parsed.get_raw_labels()


    def infer_mc(self, mn, mn_onset=0, volta=None):
        """ Shortcut for ``MSCX.parsed.infer_mc()``.
        Tries to convert a ``(mn, mn_onset)`` into a ``(mc, mc_onset)`` tuple on the basis of this MuseScore file.
        In other words, a human readable score position such as "measure number 32b (i.e., a second ending), beat
        3" needs to be converted to ``(32, 1/2, 2)`` if "beat" has length 1/4, or--if the meter is, say 9/8 and "beat"
        has a length of 3/8-- to ``(32, 6/8, 2)``. The resulting ``(mc, mc_onset)`` tuples are required for attaching
        a label to a score. This is only necessary for labels that were not originally extracted by ms3.

        Parameters
        ----------
        mn : :obj:`int` or :obj:`str`
            Measure number as in a reference print edition.
        mn_onset : :obj:`fractions.Fraction`, optional
            Distance of the requested position from beat 1 of the complete measure (MN), expressed as
            fraction of a whole note. Defaults to 0, i.e. the position of beat 1.
        volta : :obj:`int`, optional
            In the case of first and second endings, which bear the same measure number, a MN might have to be
            disambiguated by passing 1 for first ending, 2 for second, and so on. Alternatively, the MN
            can be disambiguated traditionally by passing it as string with a letter attached. In other words,
            ``infer_mc(mn=32, volta=1)`` is equivalent to ``infer_mc(mn='32a')``.

        Returns
        -------
        :obj:`int`
            Measure count (MC), denoting particular <Measure> tags in the score.
        :obj:`fractions.Fraction`

        """
        return self.parsed.infer_mc(mn=mn, mn_onset=mn_onset, volta=volta)

    def parse_mscx(self):
        implemented_parsers = ['bs4']
        if self.parser in implemented_parsers:
            try:
                self._parsed = _MSCX_bs4(self.mscx_src, read_only=self.read_only, logger_cfg=self.logger_cfg)
            except:
                self.logger.error(f"Failed parsing {self.mscx_src}.")
                raise
        else:
            raise NotImplementedError(f"Only the following parsers are available: {', '.join(implemented_parsers)}")


        if self._parsed.has_annotations:
            logger_cfg = self.logger_cfg.copy()
            logger_cfg['name'] += ':annotations'
            self._annotations = Annotations(df=self.get_raw_labels(), read_only=True, mscx_obj=self,
                                            logger_cfg=logger_cfg)


    def store_mscx(self, filepath):
        """Shortcut for ``MSCX.parsed.store_mscx()``.
        Store the current XML structure as uncompressed MuseScore file.

        Parameters
        ----------
        filepath : :obj:`str`
            Path of the newly created MuseScore file, including the file name ending on '.mscx'.
            Uncompressed files ('.mscz') are not supported.

        Returns
        -------
        :obj:`bool`
            Whether the file was successfully created.
        """
        return self.parsed.store_mscx(filepath=filepath)

    def store_list(self, what='all', folder=None, suffix=None, **kwargs):
        """
        Store one or several several lists as TSV files(s).
        
        Parameters
        ----------
        what : :obj:`str` or :obj:`Collection`, optional
            Defaults to 'all' but could instead be one or several strings out of {'notes', 'rests', 'notes_and_rests', 'measures', 'events', 'labels', 'chords', 'expanded'}
        folder : :obj:`str`, optional
            Where to store. Defaults to the directory of the parsed MSCX file.
        suffix : :obj:`str` or :obj:`Collection`, optional
            Suffix appended to the file name of the parsed MSCX file to create a new file name.
            Defaults to None, meaning that standard suffixes based on ``what`` are attached.
            Number of suffixes needs to be equal to the number of ``what``.
        **kwargs:
            Keyword arguments for :py:meth:`pandas.DataFrame.to_csv`. Defaults to ``{'sep': '\\t', 'index': False}``.
            If 'sep' is changed to a different separator, the file extension(s) will be changed to '.csv' rather than '.tsv'.

        Returns
        -------
        None

        """
        folder = resolve_dir(folder)
        mscx_path, file = os.path.split(self.mscx_src)
        fname, _ = os.path.splitext(file)
        if folder is None:
            folder = mscx_path
        if not os.path.isdir(folder):
            if input(folder + ' does not exist. Create? (y|n)') == "y":
                os.mkdir(d)
            else:
                return
        what, suffix = self._treat_storing_params(what, suffix)
        self.logger.debug(f"Parameters normalized to what={what}, suffix={suffix}.")
        if what is None:
            self.logger.error("Tell me 'what' to store.")
            return
        if 'sep' not in kwargs:
            kwargs['sep'] = '\t'
        if 'index' not in kwargs:
            kwargs['index'] = False
        ext = '.tsv' if kwargs['sep'] == '\t' else '.csv'

        for w, s in zip(what, suffix):
            df = self.__getattribute__(w)
            if len(df.index) > 0:
                new_name = f"{fname}{s}{ext}"
                full_path = os.path.join(folder, new_name)
                no_collections_no_booleans(df, logger=self.logger).to_csv(full_path, **kwargs)
                self.logger.info(f"{w} written to {full_path}")
            else:
                self.logger.debug(f"{w} empty, no file written.")

    def _treat_storing_params(self, what, suffix):
        tables = ['notes', 'rests', 'notes_and_rests', 'measures', 'events', 'labels', 'chords', 'expanded']
        if what == 'all':
            if suffix is None:
                return tables, [f"_{t}" for t in tables]
            elif len(suffix) < len(tables) or isinstance(suffix, str):
                self.logger.error(
                    f"If what='all', suffix needs to be None or include one suffix each for {tables}.\nInstead, {suffix} was passed.")
                return None, None
            elif len(suffix) > len(tables):
                suffix = suffix[:len(tables)]
            return tables, [str(s) for s in suffix]

        if isinstance(what, str):
            what = [what]
        if isinstance(suffix, str):
            suffix = [suffix]

        correct = [(i, w) for i, w in enumerate(what) if w in tables]
        if suffix is None:
            suffix = [f"_{w}" for _, w in correct]
        if len(correct) < len(what):
            if len(correct) == 0:
                self.logger.error(f"The value for what can only be out of {['all'] + tables}, not {what}.")
                return None, None
            else:
                incorrect = [w for w in what if w not in tables]
                self.logger.warning(f"The following values are not accepted as parameters for 'what': {incorrect}")
                suffix = [suffix[i] for i, _ in correct]
        if len(correct) < len(suffix):
            self.logger.error(f"Only {len(suffix)} suffixes were passed for storing {len(correct)} tables.")
            return None, None
        elif len(suffix) > len(correct):
            suffix = suffix[:len(correct)]
        return what, [str(s) for s in suffix]
