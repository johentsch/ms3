import os, re
from contextlib import contextmanager
from tempfile import NamedTemporaryFile as Temp

import pandas as pd

from .utils import check_labels, color2rgba, convert, DCML_DOUBLE_REGEX, decode_harmonies,\
    get_ms_version, get_musescore, no_collections_no_booleans,\
    resolve_dir, rgba2params, unpack_mscz, update_labels_cfg
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


    NASHVILLE_REGEX = r"^(b*|#*)(\d).*$"
    """:obj:`str`
    Class variable with a regular expression that
    recognizes labels representing a Nashville numeral, which MuseScore is able to encode.
    """

    RN_REGEX = r"^$"
    """:obj:`str`
    Class variable with a regular expression for Roman numerals that
    momentarily matches nothing because ms3 tries interpreting Roman Numerals
    als DCML harmony annotations.
    """

    native_formats = ('mscx', 'mscz')
    """:obj:`tuple`
    Formats that MS3 reads without having to convert.
    """

    convertible_formats = ('cap', 'capx', 'midi', 'mid', 'musicxml', 'mxl', 'xml', )
    """:obj:`tuple`
    Formats that have to be converted before parsing.
    """

    parseable_formats = native_formats + convertible_formats
    """:obj:`tuple`
    Formats that ms3 can parse.
    """

    def __init__(self, musescore_file=None, infer_label_types=['dcml'], read_only=False, labels_cfg={}, logger_cfg={},
                 parser='bs4', ms=None):
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
            Store a configuration dictionary to determine the output format of the :py:class:`~.annotations.Annotations`
            object representing the currently attached annotations. See :py:attr:`MSCX.labels_cfg`.
        logger_cfg : :obj:`dict`, optional
            The following options are available:
            'name': LOGGER_NAME -> by default the logger name is based on the parsed file(s)
            'level': {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}
            'file': PATH_TO_LOGFILE to store all log messages under the given path.
        parser : 'bs4', optional
            The only XML parser currently implemented is BeautifulSoup 4.
        ms : :obj:`str`, optional
            If you want to parse musicXML files or MuseScore 2 files by temporarily converting them, pass the path or command
            of your local MuseScore 3 installation. If you're using the standard path, you may try 'auto', or 'win' for
            Windows, 'mac' for MacOS, or 'mscore' for Linux.
        """
        super().__init__(subclass='Score', logger_cfg=logger_cfg)

        self.full_paths = {}
        """:obj:`dict`
        ``{KEY: {i: full_path}}`` dictionary holding the full paths of all parsed MuseScore and TSV files,
        including file names. Handled internally by :py:meth:`~.Score._handle_path`.
        """

        self.paths = {}
        """:obj:`dict`
        ``{KEY: {i: file path}}`` dictionary holding the paths of all parsed MuseScore and TSV files,
        excluding file names. Handled internally by :py:meth:`~.Score._handle_path`.
        """

        self.files = {}
        """:obj:`dict`
        ``{KEY: {i: file name with extension}}`` dictionary holding the complete file name  of each parsed file,
        including the extension. Handled internally by :py:meth:`~.Score._handle_path`.
        """

        self.fnames = {}
        """:obj:`dict`
        ``{KEY: {i: file name without extension}}`` dictionary holding the file name  of each parsed file,
        without its extension. Handled internally by :py:meth:`~.Score._handle_path`.
        """

        self.fexts = {}
        """:obj:`dict`
        ``{KEY: {i: file extension}}`` dictionary holding the file extension of each parsed file.
        Handled internally by :py:meth:`~.Score._handle_path`.
        """

        self.ms = get_musescore(ms, logger=self.logger)
        """:obj:`str`
        Path or command of the local MuseScore 3 installation if specified by the user."""

        self._mscx = None
        """:obj:`MSCX`
        The object representing the parsed MuseScore file.
        """

        self._detached_annotations = {}
        """:obj:`dict`
        ``{(key, i): Annotations object}`` dictionary for accessing all detached :py:class:`~.annotations.Annotations` objects.
        """

        self._types_to_infer = []
        """:obj:`list`
        Current order in which types are being recognized."""

        self._label_types = {
            0: "Simple string (does not begin with a note name, otherwise MS3 will turn it into type 3; prevent through leading dot)",
            1: "MuseScore's Roman Numeral Annotation format",
            2: "MuseScore's Nashville Number format",
            3: "Absolute chord encoded by MuseScore",
            'dcml': "Latest version of the DCML harmonic annotation standard.",
        }
        """:obj:`dict`
        Mapping label types to their descriptions.
        0: "Simple string (does not begin with a note name, otherwise MS3 will turn it into type 3; prevent through leading dot)",
        1: "MuseScore's Roman Numeral Annotation format",
        2: "MuseScore's Nashville Number format",
        3: "Absolute chord encoded by MuseScore",
        'dcml': "Latest version of the DCML harmonic annotation standard.",
        """

        self._label_regex = {
            1: self.RN_REGEX,
            2: self.NASHVILLE_REGEX,
            3: self.ABS_REGEX,
            'dcml': DCML_DOUBLE_REGEX,
        }
        """:obj:`dict`
        Mapping label types to their corresponding regex. Managed via the property :py:meth:`infer_label_types`.
        1: self.RN_REGEX,
        2: self.NASHVILLE_REGEX,
        3: self.ABS_REGEX,
        'dcml': utils.DCML_REGEX,
        """

        self.labels_cfg = {
            'staff': None,
            'voice': None,
            'label_type': None,
            'positioning': True,
            'decode': False,
            'column_name': 'label',
            'color_format': None,
        }
        """:obj:`dict`
        Configuration dictionary to determine the output format of the :py:class:`~.annotations.Annotations`
        objects contained in the current object, especially when calling :py:attr:`Score.mscx.labels<.MSCX.labels>`.
        The default options correspond to the default parameters of
        :py:meth:`Annotations.get_labels()<.annotations.Annotations.get_labels>`.
        """
        self.labels_cfg.update(update_labels_cfg(labels_cfg, logger=self.logger))

        self.parser = parser
        """{'bs4'}
        Currently only one XML parser has been implemented which uses BeautifulSoup 4.
        """

        self.infer_label_types = infer_label_types
        if musescore_file is not None:
            self._parse_mscx(musescore_file, read_only=read_only, labels_cfg=self.labels_cfg)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END of __init__() %%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    @property
    def infer_label_types(self):
        """:obj:`list` or :obj:`dict`, optional
        The order in which label types are to be inferred.
        Assigning a new value results in a call to :py:meth:`~.annotations.Annotations.infer_types`.
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
            for key in self:
                self[key].infer_types(after_reg)

    @property
    def has_detached_annotations(self):
        """:obj:`bool`
        Is True as long as the score contains :py:class:`~.annotations.Annotations` objects, that are not attached to the :obj:`MSCX` object.
        """
        return len(self._detached_annotations) > 0

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


    def attach_labels(self, key, staff=None, voice=None, label_type=None, check_for_clashes=True, remove_detached=True):
        """ Insert detached labels ``key`` into this score's :obj:`MSCX` object.

        Parameters
        ----------
        key : :obj:`str`
            Key of the detached labels you want to insert into the score.
        staff, voice : :obj:`int`, optional
            Pass one or both of these arguments to change the original annotation layer or if there was none.
        label_type : :obj:`int`, optional
            By default, the labels are written into the staff's layer for absolute ('guitar') chords, meaning that when
            opened next time, MuseScore will split and encode those beginning with a note name (internal label_type 3).
            In order to influence how MuseScore treats the labels pass one of these values:
            1: Roman Numeral Analysis
            2: Nashville Numbers
        check_for_clashes : :obj:`bool`, optional
            Defaults to True, meaning that the positions where the labels will be inserted will be checked for existing
            labels.
        remove_detached : :obj:`bool`, optional
            By default, the detached :py:class:`~.annotations.Annotations` object is removed after successfully attaching it.
            Pass False to have it remain in detached state.

        Returns
        -------
        :obj:`int`
            Number of newly attached labels.
        :obj:`int`
            Number of labels that were to be attached.
        """
        assert self._mscx is not None, "No score has been parsed yet."
        assert key != 'annotations', "Labels with key 'annotations' are already attached."
        if key not in self._detached_annotations:
            self.mscx.logger.info(f"""Key '{key}' doesn't correspond to a detached set of annotations.
Use one of the existing keys or load a new set with the method load_annotations().\nExisting keys: {list(self._detached_annotations.keys())}""")
            return 0, 0

        annotations = self._detached_annotations[key]
        goal = len(annotations.df)
        if goal == 0:
            self.mscx.logger.warning(f"The Annotation object '{key}' does not contain any labels.")
            return 0, 0
        df = annotations.prepare_for_attaching(staff=staff, voice=voice, label_type=label_type, check_for_clashes=check_for_clashes)
        reached = len(df)
        if reached == 0:
            self.mscx.logger.error(f"No labels from '{key}' have been attached due to aforementioned errors.")
            return reached, goal

        prepared_annotations = Annotations(df=df, cols=annotations.cols, infer_types=annotations.regex_dict)
        reached = self.mscx.add_labels(prepared_annotations)
        if remove_detached:
            if reached == goal:
                del(self._detached_annotations[key])
                self.mscx.logger.debug(f"Detached annotations '{key}' successfully attached and removed.")
            else:
                self.mscx.logger.info(f"Only {reached} of the {goal} targeted labels could be attached, so '{key}' was not removed.")
        return reached, goal

    def change_labels_cfg(self, labels_cfg={}, staff=None, voice=None, label_type=None, positioning=None, decode=None,
                          column_name=None, color_format=None):
        """ Update :py:attr:`.Score.labels_cfg` and :py:attr:`.MSCX.labels_cfg`.

        Parameters
        ----------
        labels_cfg : :obj:`dict`
            Using an entire dictionary or, to change only particular options, choose from:
        staff, voice, label_type, positioning, decode
            Arguments as they will be passed to :py:meth:`~.annotations.Annotations.get_labels`
        """
        keys = ['staff', 'voice', 'label_type', 'positioning', 'decode', 'column_name', 'color_format']
        for k in keys:
            val = locals()[k]
            if val is not None:
                labels_cfg[k] = val
        updated = update_labels_cfg(labels_cfg, logger=self.logger)
        self.labels_cfg.update(updated)
        self.mscx.labels_cfg.update(updated)


    def check_labels(self, keys='annotations', regex=None, label_type='dcml', **kwargs):
        """ Tries to match the labels ``keys`` against the given ``regex`` or the one of the registered ``label_type``.
        Returns wrong labels.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`Collection`, optional
            The key(s) of the Annotation objects you want to check. Defaults to 'annotations', the attached labels.
        regex : :obj:`str`, optional
            Pass a regular expression against which to check the labels if you don't want to use the one of an existing
            ``label_type`` or in order to register a new one on the fly by passing the new name as ``label_type``.
        label_type : :obj:`str`, optional
            To use the regular expression of a registered type, pass its name, defaults to 'dcml'. Pass a new name and
            a ``regex`` to register a new label type on the fly.
        kwargs :
            Parameters passed to :py:func:`~.utils.check_labels`.

        Returns
        -------
        :obj:`pandas.DataFrame`
            Labels not matching the regex.
        """
        if keys == 'annotations' and not self.mscx.has_annotations:
            self.mscx.logger.debug("Score contains no Annotations.")
            return
        if regex is None:
            if label_type in self._label_regex:
                regex = self._label_regex[label_type]
            else:
                self.logger.warning(f"Type {label_type} has not been registered. Pass a regular expression for it as argument 'regex'.")
                return
        else:
            if regex.__class__ != re.compile('').__class__:
                regex = re.compile(regex, re.VERBOSE)
            if label_type not in self._label_regex:
                self._label_regex[label_type] = regex
        if isinstance(keys, str):
            keys = [keys]
        existing, missing = [], []
        for k in keys:
            (existing if k in self else missing).append(k)
        if len(missing) > 0:
            self.logger.warning(f"The keys {missing} are not among the Annotations objects, which are: {list(self)}")
        if len(existing) == 0:
            return pd.DataFrame()
        labels_cfg = self.labels_cfg.copy()
        labels_cfg['decode'] = True
        checks = [check_labels(self[k].get_labels(**labels_cfg), regex=regex, **kwargs) for k in existing]
        if len(keys) > 1:
            return pd.concat(checks, keys=existing)
        else:
            return checks[0]



    def compare_labels(self, detached_key, new_color='ms3_darkgreen', old_color='ms3_darkred', detached_is_newer=False, add_to_rna=True):
        """ Compare detached labels ``key`` to the ones attached to the Score.
        By default, the attached labels are considered as the reviewed version and changes are colored in green;
        Changes with respect to the detached labels are attached to the Score in red.

        Parameters
        ----------
        detached_key : :obj:`str`
            Key of the detached labels you want to compare to the ones in the score.
        new_color, old_color : :obj:`str` or :obj:`tuple`, optional
            The colors by which new and old labels are differentiated. Identical labels remain unchanged.
        detached_is_newer : :obj:`bool`, optional
            Pass True if the detached labels are to be added with ``new_color`` whereas the attached changed labels
            will turn ``old_color``, as opposed to the default.
        add_to_rna : :obj:`bool`, optional
            By default, new labels are attached to the Roman Numeral layer. Pass false to attach them to the chord layer instead.
        """
        assert detached_key != 'annotations', "Pass a key of detached labels, not 'annotations'."
        if not self.mscx.has_annotations:
            self.logger.info(f"This score has no annotations attached.")
            return
        if detached_key not in self._detached_annotations:
            self.logger.info(f"""Key '{detached_key}' doesn't correspond to a detached set of annotations.
Use one of the existing keys or load a new set with the method load_annotations().\nExisting keys: {list(self._detached_annotations.keys())}""")
            return

        old_obj = self._detached_annotations[detached_key]
        new_obj = self.mscx._annotations
        compare_cols = ['mc', 'mc_onset', 'staff', 'voice', 'label']
        old_cols = [old_obj.cols[c] for c in compare_cols]
        new_cols = [new_obj.cols[c] for c in compare_cols]
        old = decode_harmonies(old_obj.df, label_col=old_obj.cols['label'])
        new = decode_harmonies(new_obj.df, label_col=old_obj.cols['label'])
        assert all(c in old.columns for c in old_cols), f"DataFrame needs to have columns {old_cols} but has only {old.columns}"
        assert all(c in new.columns for c in new_cols), f"DataFrame needs to have columns {new_cols} but has only {new.columns}"
        old_vals = set(old[old_cols].itertuples(index=False, name=None))
        new_vals = set(new[new_cols].itertuples(index=False, name=None))
        unchanged = old_vals.intersection(new_vals)
        changes_old = old_vals - unchanged
        changes_new = new_vals - unchanged
        if len(changes_new) == 0 and len(changes_old) == 0:
            self.mscx.logger.info(f"Comparison yielded no changes.")
            return False

        new_rgba =  color2rgba(new_color)
        new_color_params = rgba2params(new_rgba)
        old_rgba = color2rgba(old_color)
        old_color_params = rgba2params(old_rgba)

        if detached_is_newer:
            change_to = old_color
            change_to_params = old_color_params
            added_color = new_color
            added_color_params = new_color_params
        else:
            change_to = new_color
            change_to_params = new_color_params
            added_color = old_color
            added_color_params = old_color_params

        color_changes = sum(self.mscx.change_label_color(*t, **change_to_params) for t in changes_new)
        df = pd.DataFrame(changes_old, columns=compare_cols)
        for k, v in added_color_params.items():
            df[k] = v
        if add_to_rna:
            df['label_type'] = 1
            anno = Annotations(df=df)
            anno.remove_initial_dots()
        else:
            df['label_type'] = 0
            anno = Annotations(df=df)
            anno.add_initial_dots()
        added_changes = self.mscx.add_labels(anno)
        if added_changes > 0 or color_changes > 0:
            self.mscx.changed = True
            self.mscx.parsed.parse_measures()
            self.mscx._update_annotations()
            self.mscx.logger.info(f"{color_changes} attached labels changed to {change_to}, {added_changes} labels added in {added_color}.")
            return True
        return False



    def detach_labels(self, key, staff=None, voice=None, label_type=None, delete=True):
        """ Detach all annotations labels from this score's :obj:`MSCX` object or just a selection of them.
        The extracted labels are stored as a new :py:class:`~.annotations.Annotations` object that is accessible via ``Score.{key}``.
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
        if not self.mscx.has_annotations:
            self.mscx.logger.info("No annotations present in score.")
            return
        assert key not in dir(self) + ['annotations'], f"The key {key} is reserved, please choose a different one."
        if not key.isidentifier():
            self.logger.warning(
                f"'{key}' cannot be used as an identifier. The extracted labels need to be accessed via self._detached_annotations['{key}']")
        df = self.annotations.get_labels(staff=staff, voice=voice, label_type=label_type, drop=delete)
        if len(df) == 0:
            self.mscx.logger.info(f"No labels found for staff {staff}, voice {voice}, label_type {label_type}.")
            return
        logger_cfg = self.logger_cfg.copy()
        logger_cfg['name'] += f"{self.mscx.logger.logger.name}:{key}"
        if self.logger.logger.file_handler is not None:
            logger_cfg['file'] = self.logger.logger.file_handler.baseFilename
        self._detached_annotations[key] = Annotations(df=df, infer_types=self.get_infer_regex(), mscx_obj=self._mscx,
                                                      logger_cfg=logger_cfg)
        if delete:
            self._mscx.delete_labels(df)
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
            By default, the labels of all :py:class:`~.annotations.Annotations` objects are matched against the new type.
            Pass False to not change any label's type.
        """
        assert name not in self._label_types, f"'{name}' already added to types: {self._label_types[name]}"
        self._label_types[name] = description
        self._label_regex[name] = regex
        if infer:
            self._types_to_infer.insert(0, name)
            for key in self:
                self[key].infer_types(self.get_infer_regex())

    def load_annotations(self, tsv_path=None, anno_obj=None, key='tsv', cols={}, infer=True):
        """ Attach an :py:class:`~.annotations.Annotations` object to the score and make it available as ``Score.{key}``.
        It can be an existing object or one newly created from the TSV file ``tsv_path``.

        Parameters
        ----------
        tsv_path : :obj:`str`
            If you want to create a new :py:class:`~.annotations.Annotations` object from a TSV file, pass its path.
        anno_obj : :py:class:`~.annotations.Annotations`
            Instead, you can pass an existing object.
        key : :obj:`str`, defaults to 'tsv'
            Specify a new key for accessing the set of annotations. The string needs to be usable
            as an identifier, e.g. not start with a number, not contain special characters etc. In return you
            may use it as a property: For example, passing ``'chords'`` lets you access the :py:class:`~.annotations.Annotations` as
            ``Score.chords``. The key 'annotations' is reserved for all annotations attached to the score.
        cols : :obj:`dict`, optional
            If the columns in the specified TSV file diverge from the :ref:`standard column names<column_names>`,
            pass a {standard name: custom name} dictionary.
        infer : :obj:`bool`, optional
            By default, the label types are inferred in the currently configured order (see :py:attr:`infer_label_types`).
            Pass False to not add and not change any label types.
        """
        assert key != 'annotations', "The key 'annotations' is reserved, please choose a different one."
        assert key is not None, "Key cannot be None."
        assert sum(True for arg in [tsv_path, anno_obj] if arg is not None) == 1, "Pass either tsv_path or anno_obj."
        inf_dict = self.get_infer_regex() if infer else {}
        mscx = None if self._mscx is None else self._mscx
        if tsv_path is not None:
            key = self._handle_path(tsv_path, key)
            logger_cfg = self.logger_cfg.copy()
            logger_cfg['name'] = f"{self.logger_names[key]}"
            self._detached_annotations[key] = Annotations(tsv_path=tsv_path, infer_types=inf_dict, cols=cols, mscx_obj=mscx,
                                                          logger_cfg=logger_cfg)
        else:
            anno_obj.mscx_obj = mscx
            self._detached_annotations[key] = anno_obj

    def store_annotations(self, key='annotations', tsv_path=None, **kwargs):
        """ Save a set of annotations as TSV file. While ``store_list`` stores attached labels only, this method
        can also store detached labels by passing a ``key``.

        Parameters
        ----------
        key : :obj:`str`, optional
            Key of the :py:class:`~.annotations.Annotations` object which you want to output as TSV file.
            By default, the annotations attached to the score (key='annotations') are stored.
        tsv_path : :obj:`str`, optional
            Path of the newly created TSV file including the file name.
            By default, the TSV file is stored next to t
        kwargs
            Additional keyword arguments will be passed to the function :py:meth:`pandas.DataFrame.to_csv` to
            customise the format of the created file (e.g. to change the separator to commas instead of tabs,
            you would pass ``sep=','``).
        """
        assert key in self, f"Key '{key}' not found. Available keys: {list(self)}"
        if tsv_path is None:
            if 'mscx' in self.paths:
                path = self.paths['mscx']
                fname = self.fnames['mscx']
                tsv_path = os.path.join(path, fname + '_labels.tsv')
            else:
                self.logger.warning(f"No tsv_path has been specified and no MuseScore file has been parsed to infer one.")
                return
        if self[key].store_tsv(tsv_path, **kwargs):
            new_key = self._handle_path(tsv_path, key=key)
            if key != 'annotations':
                self[key].update_logger_cfg({'name': self.logger_names[new_key]})


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
            raise ValueError(f"Path not found: {path}.")
            #self.logger.error("No file found at this path: " + full_path)
            return None

    @staticmethod
    def _make_extension_regex(native=True, convertible=True, tsv=False):
        assert sum((native, convertible)) > 0, "Select at least one type of extensions."
        exts = []
        if native:
            exts.extend(Score.native_formats)
        if convertible:
            exts.extend(Score.convertible_formats)
        if tsv:
            exts.append('tsv')
        dot = r'\.'
        regex = f"({'|'.join(dot + e for e in exts)})$"
        return re.compile(regex, re.IGNORECASE)


    def _parse_mscx(self, musescore_file, read_only=False, parser=None, labels_cfg={}):
        """ 
        This method is called by :py:meth:`.__init__` to parse the score. It checks the file extension
        and in the case of a compressed MuseScore file (.mscz), a temporary uncompressed file is generated
        which is removed after the parsing process.
        Essentially, parsing means to initiate a :obj:`MSCX` object and to make it available as ``Score.mscx``
        and, if the score includes annotations, to initiate an :py:class:`~.annotations.Annotations` object that
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
            Store a configuration dictionary to determine the output format of the :py:class:`~.annotations.Annotations`
            object representing the currently attached annotations. See :py:attr:`.MSCX.labels_cfg`.
        """
        if parser is not None:
            self.parser = parser

        permitted_extensions = self.native_formats + self.convertible_formats
        _, ext = os.path.splitext(musescore_file)
        ext = ext[1:]
        if ext.lower() not in permitted_extensions:
            raise ValueError(f"The extension of a score should be one of {permitted_extensions} not {ext}.")
        if ext.lower() in self.convertible_formats and self.ms is None:
            raise ValueError(f"To open a {ext} file, use 'ms3 convert' command or pass parameter 'ms' to Score to temporally convert.")
        extension = self._handle_path(musescore_file)
        logger_cfg = self.logger_cfg.copy()
        logger_cfg['name'] = self.logger_names[extension]
        musescore_file = resolve_dir(musescore_file)

        if extension in self.convertible_formats +  ('mscz', ):
            ctxt_mgr = unpack_mscz if extension == 'mscz' else self._tmp_convert
            with ctxt_mgr(musescore_file) as tmp_mscx:
                self.logger.debug(f"Using temporary file {os.path.basename(tmp_mscx)} in order to parse {musescore_file}.")
                self._mscx = MSCX(tmp_mscx, read_only=read_only, labels_cfg=labels_cfg, parser=self.parser,
                                  logger_cfg=logger_cfg, parent_score=self)
                self.mscx.mscx_src = (musescore_file)
        else:
            self._mscx = MSCX(musescore_file, read_only=read_only, labels_cfg=labels_cfg, parser=self.parser,
                              logger_cfg=logger_cfg, parent_score=self)
        if self.mscx.has_annotations:
            self.mscx._annotations.infer_types(self.get_infer_regex())

    @contextmanager
    def _tmp_convert(self, file, dir=None):
        if dir is None:
            dir = os.path.dirname(file)
        try:
            tmp_file = Temp(suffix='.mscx', prefix='.', dir=dir, delete=False)
            convert(file, tmp_file.name, self.ms, logger=self.logger)
            yield tmp_file.name
        except:
            self.logger.error(f"Error while dealing with the temporarily converted {os.path.basename(file)}")
            raise
        finally:
            os.remove(tmp_file.name)


    def __repr__(self):
        msg = ''
        if any(ext in self.full_paths for ext in ('mscx', 'mscz')):
            if 'mscx' in self.full_paths:
                path = self.full_paths['mscx']
                msg = f"Uncompressed MuseScore file"
            else:
                path = self.full_paths['mscz']
                msg = f"ZIP compressed MuseScore file"
            if self._mscx.changed:
                msg += " (CHANGED!!!)"
        else:
            frst = list(self.full_paths.keys())[0]
            path = self.full_paths[frst]
            msg = f"Temporarily converted {frst.upper()} file"
        n_chars = len(msg)
        if self._mscx.changed:
            msg += '\n' + (n_chars-12) * '-' + 12 * '!'
        else:
            msg += '\n' + n_chars * '-'
        msg += f"\n\n{path}\n\n"
        if self.mscx.has_annotations:
            msg += f"Attached annotations\n--------------------\n\n{self.annotations}\n\n"
        else:
            msg += "No annotations attached.\n\n"
        if self.has_detached_annotations:
            msg += "Detached annotations\n--------------------\n\n"
            for key, obj in self._detached_annotations.items():
                key_info = key + f" (stored as {self.files[key]})" if key in self.files else key
                msg += f"{key_info} -> {obj}\n\n"
        if self.mscx.n_form_labels > 0:
            msg += f"Score contains {self.mscx.n_form_labels} form labels."
        return msg



    def __getattr__(self, item):
        if item == 'annotations':
            return self.mscx._annotations
        try:
            return self._detached_annotations[item]
        except:
            raise AttributeError(item)

    def __getitem__(self, item):
        if item == 'annotations':
            return self.mscx._annotations
        try:
            return self._detached_annotations[item]
        except:
            raise AttributeError(item)

    def __iter__(self):
        """ Iterate keys of Annotation objects. """
        attached = ['annotations'] if self._mscx is not None and self._mscx.has_annotations else []
        yield from attached + list(self._detached_annotations.keys())
    # def __setattr__(self, key, value):
    #     assert key != 'annotations', "The key 'annotations' is managed automatically, please pick a different one."
    #     assert key.isidentifier(), "Please use an alphanumeric key without special characters."
    #     if key in self.__dict__:
    #         self.__dict__[key] = value
    #     else:
    #         self._annotations[key] = value


########################################################################################################################
########################################################################################################################
################################################ End of Score() ########################################################
########################################################################################################################
########################################################################################################################





class MSCX(LoggedClass):
    """ Object for interacting with the XML structure of a MuseScore 3 file. Is usually attached to a
    :obj:`Score` object and exposed as ``Score.mscx``.
    An object is only created if a score was successfully parsed.
    """

    def __init__(self, mscx_src, read_only=False, parser='bs4', labels_cfg={}, logger_cfg={}, level=None, parent_score=None):
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
            Store a configuration dictionary to determine the output format of the :py:class:`~.annotations.Annotations`
            object representing the currently attached annotations. See :py:attr:`.MSCX.labels_cfg`.
        logger_cfg : :obj:`dict`, optional
            The following options are available:
            'name': LOGGER_NAME -> by default the logger name is based on the parsed file(s)
            'level': {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}
            'file': PATH_TO_LOGFILE to store all log messages under the given path.
        level : :obj:`str` or :obj:`int`
            Quick way to change the logging level which defaults to the one of the parent :obj:`Score`.
        parent_score : :obj:`Score`, optional
            Store the Score object to which this MSCX object is attached.
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


        self.changed = False
        """:obj:`bool`
        Switches to True as soon as the original XML structure is changed. Does not automatically switch back to False.
        """

        self.read_only = read_only
        """:obj:`bool`, optional
        Shortcut for ``MSCX.parsed.read_only``.
        Defaults to ``False``, meaning that the parsing is slower and uses more memory in order to allow for manipulations
        of the score, such as adding and deleting labels. Set to ``True`` if you're only extracting information."""

        self._annotations = None
        """:py:class:`~.annotations.Annotations` or None
        If the score contains at least one <Harmony> tag, this attribute points to the object representing all
        annotations, otherwise it is None."""

        self.parent_score = parent_score
        """:obj:`Score`
        The Score object to which this MSCX object is attached."""

        self.parser = parser
        """{'bs4'}
        The currently used parser."""

        self._parsed = None
        """{:obj:`_MSCX_bs4`}
        Holds the MSCX score parsed by the selected parser (currently only BeautifulSoup 4 available)."""


        self.labels_cfg = labels_cfg
        """:obj:`dict`
        Configuration dictionary to determine the output format of the :py:class:`~.annotations.Annotations`
        object representing the labels that are attached to a score (stored as :py:attr:`._annotations``). 
        The options correspond to the parameters of
        :py:meth:`Annotations.get_labels()<.annotations.Annotations.get_labels>`.
        """

        ms_version = get_ms_version(self.mscx_src)
        if ms_version is None:
            raise ValueError(f"MuseScore version could not be read from {self.mscx_src}")
        if ms_version[0] == '3':
            self.parse_mscx()
        else:
            if self.parent_score.ms is None:
                raise ValueError(f"""In order to parse a version {ms_version} file,
use 'ms3 convert' command or pass parameter 'ms' to Score to temporally convert.""")
            with self.parent_score._tmp_convert(self.mscx_src) as tmp:
                self.logger.debug(f"Using temporally converted file {os.path.basename(tmp)} for parsing the version {ms_version} file.")
                self.mscx_src = tmp
                self.parse_mscx()
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END of __init__() %%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


    @property
    def cadences(self):
        """:obj:`pandas.DataFrame`
        DataFrame representing all cadence annotations in the score.
        """
        exp = self.expanded
        if exp is None or 'cadence' not in exp.columns:
            return None
        return exp[exp.cadence.notna()]


    @property
    def chords(self):
        """:obj:`pandas.DataFrame`
        DataFrame representing all <Chord> tags in the score. A chord in that sense is a grouping of all
        synchronous notes occurring in the same notational layer of the same staff. The DataFrame contains
        all kinds of score markup that is not attached to particular notes but to a <Chord>, such as
        slurs, lyrics, staff text, ottava lines etc.
        """
        return self.parsed.chords


    @property
    def events(self):
        """:obj:`pandas.DataFrame`
        DataFrame representating a raw skeleton of the score's XML structure and contains all score events,
        i.e. <Chord>, <Rest>, <Harmony> and markup tags such as <Beam> together with, in the columns the values of their
        XML properties and children. It serves as master for computing :obj:`.chords`, :obj:`rests`, and :obj:`labels`
        (and therefore :obj:`.expanded`, too)."""
        return self.parsed.events


    @property
    def expanded(self):
        """:obj:`pandas.DataFrame`
        DataFrame of labels that have been split into various features using a regular expression."""
        if self._annotations is None:
            return None
        #labels_cfg = self.labels_cfg.copy()
        #labels_cfg['decode'] = False
        return self._annotations.expand_dcml(**self.labels_cfg)


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
    def n_form_labels(self):
        """:obj:`int`
        Shortcut for ``MSCX.parsed.n_form_labels``.
        Is True if at least one StaffText seems to constitute a form label."""
        return self.parsed.n_form_labels


    @property
    def form_labels(self):
        """:obj:`pandas.DataFrame`
        DataFrame representing a filtered event list containing only StaffTexts that include the regular expression
        :py:const:`.utils.FORM_DETECTION_REGEX`
        """
        return self.parsed.fl


    @property
    def labels(self):
        """:obj:`pandas.DataFrame`
        DataFrame representing all <Harmony> tags in the score as returned by calling :py:meth:`~.annotations.Annotations.get_labels`
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
        return self.parsed.ml


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
        return self.parsed.nl


    @property
    def notes_and_rests(self):
        """:obj:`pandas.DataFrame`
        The union of :obj:`.notes` and :obj:`.rests`."""
        return self.parsed.notes_and_rests


    @property
    def parsed(self):
        """:obj:`~._MSCX_bs4`
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
        return self.parsed.rl


    @property
    def staff_ids(self):
        """:obj:`list` of :obj:`int`
        The staff IDs contained in the score, usually just a list of increasing numbers starting at 1."""
        return self.parsed.staff_ids

    @property
    def style(self):
        """:obj:`Style`
        Can be used like a dictionary to change the information within the score's <Style> tag."""
        return self.parsed.style


    @property
    def version(self):
        """:obj:`str`
        MuseScore version that the file was created with."""
        return self.parsed.version

    @property
    def volta_structure(self):
        """:obj:`dict`
        {first_mc -> {volta_number -> [mc1, mc2...]} } dictionary."""
        return self.parsed.volta_structure


    def add_labels(self, annotations_object):
        """ Receives the labels from an :py:class:`~.annotations.Annotations` object and adds them to the XML structure
        representing the MuseScore file that might be written to a file afterwards.

        Parameters
        ----------
        annotations_object : :py:class:`~.annotations.Annotations`
            Object of labels to be added.

        Returns
        -------
        :obj:`int`
            Number of actually added labels.

        """
        df = annotations_object.df
        if len(df) == 0:
            self.logger.info("Nothing to add.")
            return 0
        main_cols = Annotations.main_cols
        columns = annotations_object.cols
        missing_main = {c for  c in main_cols if columns[c] not in df.columns}
        assert len(missing_main) == 0, f"The specified columns for the following main parameters are missing:\n{missing_main}"
        if columns['decoded'] not in df.columns:
            df[columns['decoded']] = decode_harmonies(df, label_col=columns['label'], return_series=True)
        #df = df[df[columns['label']].notna()]
        existing_cols = {k: v for k, v in columns.items() if v in df.columns}
        param2cols = {**existing_cols}
        parameters = list(param2cols.keys())
        clmns = list(param2cols.values())
        self.logger.debug(f"add_label() will be called with this param2col mapping:\n{param2cols}")
        tups = tuple(df[clmns].itertuples(index=False, name=None))
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
            self._update_annotations()
            self.logger.debug(f"{changes}/{len(df)} labels successfully added to score.")
        return changes


    def change_label_color(self, mc, mc_onset, staff, voice, label, color_name=None, color_html=None, color_r=None,
                           color_g=None, color_b=None, color_a=None):
        """  Shortcut for :py:meth:``MSCX.parsed.change_label_color``

        Parameters
        ----------
        mc : :obj:`int`
            Measure count of the label
        mc_onset : :obj:`fractions.Fraction`
            Onset position to which the label is attached.
        staff : :obj:`int`
            Staff to which the label is attached.
        voice : :obj:`int`
            Notational layer to which the label is attached.
        label : :obj:`str`
            (Decoded) label.
        color_name, color_html : :obj:`str`, optional
            Two ways of specifying the color.
        color_r, color_g, color_b, color_a : :obj:`int` or :obj:`str`, optional
            To specify a RGB color instead, pass at least, the first three. ``color_a`` (alpha = opacity) defaults
            to 255.
        """
        return self.parsed.change_label_color(mc=mc, mc_onset=mc_onset, staff=staff, voice=voice, label=label,
                                              color_name=color_name, color_html=color_html, color_r=color_r,
                                              color_g=color_g, color_b=color_b, color_a=color_a)

    def change_labels_cfg(self, labels_cfg={}, staff=None, voice=None, label_type=None, positioning=None, decode=None,
                          column_name=None, color_format=None):
        """ Update :py:attr:`.MSCX.labels_cfg`.

        Parameters
        ----------
        labels_cfg : :obj:`dict`
            Using an entire dictionary or, to change only particular options, choose from:
        staff, voice, label_type, positioning, decode
            Arguments as they will be passed to :py:meth:`~.annotations.Annotations.get_labels`
        """
        keys = ['staff', 'voice', 'label_type', 'positioning', 'decode', 'column_name', 'color_format']
        for k in keys:
            val = locals()[k]
            if val is not None:
                labels_cfg[k] = val
        updated = update_labels_cfg(labels_cfg, logger=self.logger)
        self.labels_cfg.update(updated)



    def delete_labels(self, df):
        """ Delete a set of labels from the current XML.

        Parameters
        ----------
        df : :obj:`pandas.DataFrame`
            A DataFrame with the columns ['mc', 'mc_onset', 'staff', 'voice']
        """
        cols = ['mc', 'staff', 'voice', 'mc_onset']
        positions = set(df[cols].itertuples(name=None, index=False))
        changed = {ix: self._parsed.delete_label(mc, staff, voice, mc_onset)
                   for ix, mc, staff, voice, mc_onset
                   in reversed(
                        list(df[cols].drop_duplicates().itertuples(name=None, index=True)))}
        changed = pd.Series(changed, index=df.index).fillna(method='ffill')
        changes = changed.sum()
        if changes > 0:
            self.changed = True
            self._parsed.parse_measures()
            self._update_annotations()
            target = len(df)
            self.logger.debug(f"{changes}/{target} labels successfully deleted.")
            if changes < target:
                self.logger.warning(f"{target - changes} labels could not be deleted:\n{df.loc[~changed]}")


    def replace_labels(self, annotations_object):
        """

        Parameters
        ----------
        annotations_object : :py:class:`~.annotations.Annotations`
            Object of labels to be added.

        Returns
        -------

        """
        self.delete_labels(annotations_object.df)
        self.add_labels(annotations_object)



    def delete_empty_labels(self):
        """ Remove all empty labels from the attached annotations. """
        if self._annotations is None:
            self.logger.info("No annotations attached.")
            return
        df = self._annotations.get_labels(decode=True)
        label_col = self._annotations.cols['label']
        sel = df[label_col] == 'empty_harmony'
        if sel.sum() == 0:
            self.logger.info("Score contains no empty labels.")
            return
        cols = ['mc', 'staff', 'voice', 'mc_onset']
        changed = [self._parsed.delete_label(mc, staff, voice, mc_onset, empty_only=True)
                   for mc, staff, voice, mc_onset
                   in df.loc[sel, cols].itertuples(name=None, index=False)]
        if sum(changed) > 0:
            self.changed = True
            self._parsed.parse_measures()
            self._update_annotations()
            self.logger.info(f"Successfully deleted {sum(changed)} empty labels.")
        else:
            self.logger.info("No empty labels were deleted.")




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
        which can be 1 (MuseScore's Roman Numeral display), 2 (Nashville) or undefined (in the case of 'normal'
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

        self._update_annotations()


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
            Defaults to 'all' but could instead be one or several strings out of {'notes', 'rests', 'notes_and_rests', 'measures', 'events', 'labels', 'chords', 'expanded', 'form_labels'}
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
                os.makedirs(folder)
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
        tables = ['notes', 'rests', 'notes_and_rests', 'measures', 'events', 'labels', 'chords', 'expanded', 'form_labels']
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


    def _update_annotations(self, infer_types={}):
        if len(infer_types) == 0 and self._annotations is not None:
            infer_types = self._annotations.regex_dict
        if self._parsed.has_annotations:
            self.has_annotations = True
            logger_cfg = self.logger_cfg.copy()
            logger_cfg['name'] += ':annotations'
            self._annotations = Annotations(df=self.get_raw_labels(), read_only=True, mscx_obj=self, infer_types=infer_types,
                                            logger_cfg=logger_cfg)
        else:
            self._annotations = None
