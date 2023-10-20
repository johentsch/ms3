"""
.. |act_dur| replace:: :ref:`act_dur <act_dur>`
.. |alt_label| replace:: :ref:`alt_label <alt_label>`
.. |added_tones| replace:: :ref:`added_tones <chord_tones>`
.. |articulation| replace:: :ref:`articulation <articulation>`
.. |bass_note| replace:: :ref:`bass_note <bass_note>`
.. |barline| replace:: :ref:`barline <barline>`
.. |breaks| replace:: :ref:`breaks <breaks>`
.. |cadence| replace:: :ref:`cadence <cadence>`
.. |changes| replace:: :ref:`changes <changes>`
.. |chord| replace:: :ref:`chord <chord>`
.. |chord_id| replace:: :ref:`chord_id <chord_id>`
.. |chord_tones| replace:: :ref:`chord_tones <chord_tones>`
.. |chord_type| replace:: :ref:`chord_type <chord_type>`
.. |crescendo_hairpin| replace:: :ref:`crescendo_hairpin <hairpins>`
.. |crescendo_line| replace:: :ref:`crescendo_line <cresc_lines>`
.. |decrescendo_hairpin| replace:: :ref:`decrescendo_hairpin <hairpins>`
.. |diminuendo_line| replace:: :ref:`diminuendo_line <cresc_lines>`
.. |dont_count| replace:: :ref:`dont_count <dont_count>`
.. |duration| replace:: :ref:`duration <duration>`
.. |duration_qb| replace:: :ref:`duration_qb <duration_qb>`
.. |dynamics| replace:: :ref:`dynamics <dynamics>`
.. |figbass| replace:: :ref:`figbass <figbass>`
.. |form| replace:: :ref:`form <form>`
.. |globalkey| replace:: :ref:`globalkey <globalkey>`
.. |globalkey_is_minor| replace:: :ref:`globalkey_is_minor <globalkey_is_minor>`
.. |gracenote| replace:: :ref:`gracenote <gracenote>`
.. |harmony_layer| replace:: :ref:`harmony_layer <harmony_layer>`
.. |keysig| replace:: :ref:`keysig <keysig>`
.. |label| replace:: :ref:`label <label>`
.. |label_type| replace:: :ref:`label_type <label_type>`
.. |localkey| replace:: :ref:`localkey <localkey>`
.. |localkey_is_minor| replace:: :ref:`localkey_is_minor <localkey_is_minor>`
.. |lyrics:1| replace:: :ref:`lyrics:1 <lyrics_1>`
.. |mc| replace:: :ref:`mc <mc>`
.. |mc_offset| replace:: :ref:`mc_offset <mc_offset>`
.. |mc_onset| replace:: :ref:`mc_onset <mc_onset>`
.. |midi| replace:: :ref:`midi <midi>`
.. |mn| replace:: :ref:`mn <mn>`
.. |mn_onset| replace:: :ref:`mn_onset <mn_onset>`
.. |next| replace:: :ref:`next <next>`
.. |nominal_duration| replace:: :ref:`nominal_duration <nominal_duration>`
.. |numbering_offset| replace:: :ref:`numbering_offset <numbering_offset>`
.. |numeral| replace:: :ref:`numeral <numeral>`
.. |offset_x| replace:: :ref:`offset_x <offset>`
.. |offset_y| replace:: :ref:`offset_y <offset>`
.. |Ottava:15mb| replace:: :ref:`Ottava:15mb <ottava>`
.. |Ottava:8va| replace:: :ref:`Ottava:8va <ottava>`
.. |Ottava:8vb| replace:: :ref:`Ottava:8vb <ottava>`
.. |pedal| replace:: :ref:`pedal <pedal>`
.. |phraseend| replace:: :ref:`phraseend <phraseend>`
.. |qpm| replace:: :ref:`qpm <qpm>`
.. |quarterbeats| replace:: :ref:`quarterbeats <quarterbeats>`
.. |quarterbeats_all_endings| replace:: :ref:`quarterbeats_all_endings <quarterbeats_all_endings>`
.. |relativeroot| replace:: :ref:`relativeroot <relativeroot>`
.. |regex_match| replace:: :ref:`regex_match <regex_match>`
.. |repeats| replace:: :ref:`repeats <repeats>`
.. |root| replace:: :ref:`root <root>`
.. |scalar| replace:: :ref:`scalar <scalar>`
.. |slur| replace:: :ref:`slur <slur>`
.. |staff| replace:: :ref:`staff <staff>`
.. |staff_text| replace:: :ref:`staff_text <staff_text>`
.. |system_text| replace:: :ref:`system_text <system_text>`
.. |tempo| replace:: :ref:`tempo <tempo>`
.. |TextLine| replace:: :ref:`TextLine <textline>`
.. |tied| replace:: :ref:`tied <tied>`
.. |timesig| replace:: :ref:`timesig <timesig>`
.. |tpc| replace:: :ref:`tpc <tpc>`
.. |tremolo| replace:: :ref:`tremolo <tremolo>`
.. |volta| replace:: :ref:`volta <volta>`
.. |voice| replace:: :ref:`voice <voice>`
"""
import os
import re
from contextlib import contextmanager
from fractions import Fraction
from tempfile import NamedTemporaryFile as Temp
from typing import IO, Collection, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .annotations import Annotations
from .bs4_parser import _MSCX_bs4, get_row_at_quarterbeat
from .logger import LoggedClass, get_log_capture_handler, get_logger
from .transformations import add_quarterbeats_col
from .utils import (
    assert_dfs_equal,
    check_labels,
    check_phrase_annotations,
    color2rgba,
    convert,
    decode_harmonies,
    expand_form_labels,
    get_ms_version,
    get_musescore,
    replace_index_by_intervals,
    resolve_dir,
    rgba2params,
    unpack_mscz,
    update_labels_cfg,
)
from .utils.constants import DCML_DOUBLE_REGEX, FORM_DETECTION_REGEX

module_logger = get_logger(__name__)


class MSCX(LoggedClass):
    """Object for interacting with the XML structure of a MuseScore 3 file. Is usually attached to a
    :obj:`Score` object and exposed as ``Score.mscx``.
    An object is only created if a score was successfully parsed.
    """

    _deprecated_elements = ["output_mscx"]

    def __init__(
        self,
        mscx_src,
        read_only=False,
        parser="bs4",
        labels_cfg={},
        parent_score=None,
        **logger_cfg,
    ):
        """Object for interacting with the XML structure of a MuseScore 3 file.

        Parameters
        ----------
        mscx_src : :obj:`str`
            Uncompressed MuseScore 3 file to parse.
        read_only : :obj:`bool`, optional
            Defaults to ``False``, meaning that the parsing is slower and uses more memory in order to allow for
            manipulations of the score, such as adding and deleting labels. Set to ``True`` if you're only extracting
            information.
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
        super().__init__(subclass="MSCX", logger_cfg=logger_cfg)
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
        Defaults to ``False``, meaning that the parsing is slower and uses more memory in order to allow for
        manipulations of the score, such as adding and deleting labels. Set to ``True`` if you're only extracting
        information."""

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

        self._parsed: _MSCX_bs4 = None
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
            raise ValueError(
                f"MuseScore version could not be read from {self.mscx_src}"
            )
        major_version = int(ms_version[0])
        if major_version == 3:
            self.parse_mscx()
        elif major_version == 4:
            self.logger.info(
                f"This file has been created with MuseScore {ms_version} which I deal with experimentally for now."
            )
            self.parse_mscx()
        else:
            if self.parent_score.ms is None:
                raise ValueError(
                    f"In order to parse a version {ms_version} file, "
                    f"use 'ms3 convert' command or pass parameter 'ms' to Score to temporally convert."
                )
            with self.parent_score._tmp_convert(self.mscx_src) as tmp:
                self.logger.debug(
                    f"Using temporally converted file {os.path.basename(tmp)} for parsing the version {ms_version} "
                    f"file."
                )
                self.mscx_src = tmp
                self.parse_mscx()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%% END of __init__() %%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    def cadences(
        self, interval_index: bool = False, unfold: bool = False
    ) -> Optional[pd.DataFrame]:
        """:obj:`pandas.DataFrame`
        DataFrame representing all cadence annotations in the score.
        """
        exp = self.expanded(interval_index=interval_index, unfold=unfold)
        if exp is None or "cadence" not in exp.columns:
            return None
        cadences = exp[exp.cadence.notna()]
        if len(cadences) == 0:
            return
        return cadences

    def chords(
        self,
        mode: Literal["auto", "strict"] = "auto",
        interval_index: bool = False,
        unfold: bool = False,
    ) -> Optional[pd.DataFrame]:
        """DataFrame of :ref:`chords` representing all <Chord> tags contained in the MuseScore file
        (all <note> tags come within one) and attached score information and performance maerks, e.g.
        lyrics, dynamics, articulations, slurs (see the explanation for the ``mode`` parameter for more details).
        Comes with the columns |quarterbeats|, |duration_qb|, |mc|, |mn|, |mc_onset|, |mn_onset|, |timesig|, |staff|,
        |voice|, |duration|, |gracenote|, |tremolo|, |nominal_duration|, |scalar|, |volta|, |chord_id|, |dynamics|,
        |articulation|, |staff_text|, |slur|, |Ottava:8va|, |Ottava:8vb|, |pedal|, |TextLine|, |decrescendo_hairpin|,
        |diminuendo_line|, |crescendo_line|, |crescendo_hairpin|, |tempo|, |qpm|, |lyrics:1|, |Ottava:15mb|

        Args:
          mode:
              Defaults to 'auto', meaning that additional performance markers available in the score are to be included,
              namely lyrics, dynamics, fermatas, articulations, slurs, staff_text, system_text, tempo, and spanners
              (e.g. slurs, 8va lines, pedal lines). This results in NaN values in the column 'chord_id' for those
              markers that are not part of a <Chord> tag, e.g. <Dynamic>, <StaffText>, or <Tempo>. To prevent that, pass
              'strict', meaning that only <Chords> are included, i.e. the column 'chord_id' will have no empty values.
          interval_index:
              Pass True to replace the default :obj:`~pandas.RangeIndex` by an
              :obj:`~pandas.IntervalIndex`.

        Returns:
          DataFrame of :ref:`chords` representing all <Chord> tags contained in the MuseScore file.
        """
        return self.parsed.chords(
            mode=mode, interval_index=interval_index, unfold=unfold
        )

    def events(
        self, interval_index: bool = False, unfold: bool = False
    ) -> Optional[pd.DataFrame]:
        """DataFrame representing a raw skeleton of the score's XML structure and contains all :ref:`events`
        contained in it. It is the original tabular representation of the MuseScore file’s source code from which
        all other tables, except ``measures`` are generated.

        Args:
          interval_index: Pass True to replace the default :obj:`~pandas.RangeIndex` by an :obj:`~pandas.IntervalIndex`.

        Returns:
          DataFrame containing the original tabular representation of all :ref:`events` encoded in the MuseScore file.
        """
        return self.parsed.events(interval_index=interval_index, unfold=unfold)

    def expanded(
        self, interval_index: bool = False, unfold: bool = False
    ) -> Optional[pd.DataFrame]:
        """DataFrame representing :ref:`expanded` labels, i.e., all annotations encoded in <Harmony> tags which could
        be matched against one of the registered regular expressions and split into feature columns. Currently this
        method is hard-coded to return expanded DCML harmony labels only but it takes into account the current
        :attr:`._labels_cfg`. Comes with the columns |quarterbeats|, |duration_qb|, |mc|, |mn|, |mc_onset|, |mn_onset|,
        |timesig|, |staff|, |voice|, |volta|, |label|, |alt_label|, |offset_x|, |offset_y|, |regex_match|, |globalkey|,
        |localkey|, |pedal|, |chord|, |numeral|, |form|, |figbass|, |changes|, |relativeroot|, |cadence|, |phraseend|,
        |chord_type|, |globalkey_is_minor|, |localkey_is_minor|, |chord_tones|, |added_tones|, |root|, |bass_note|

        Args:
          interval_index: Pass True to replace the default :obj:`~pandas.RangeIndex` by an :obj:`~pandas.IntervalIndex`.

        Returns:
          DataFrame representing all :ref:`labels`, i.e., all <Harmony> tags in the score.
        """
        # TODO: Retrieving expanded labels for custom regEx (registered with _.new_type())
        if self._annotations is None:
            return None
        expanded = self._annotations.expand_dcml(**self.labels_cfg)
        if expanded is None:
            labels = self.labels()
            if "regex_match" in labels and "dcml" in labels.regex_match.unique():
                self.logger.warning(
                    "Annotations are present but expansion failed due to errors.",
                    extra={"message_id": (17,)},
                )
            return None

        unfolded_expanded = self.parsed.unfold_facet_df(
            expanded, "expanded DCML labels"
        )
        if unfolded_expanded is not None:
            check_phrase_annotations(unfolded_expanded, "phraseend", logger=self.logger)
        else:
            self.logger.debug(
                "Since unfolding failed, I'll check the phrase annotations as is."
            )
            if "volta" in expanded:
                check_phrase_annotations(
                    expanded[expanded.volta.fillna(2) == 2],
                    "phraseend",
                    logger=self.logger,
                )
            else:
                check_phrase_annotations(expanded, "phraseend", logger=self.logger)

        if unfold:
            if unfolded_expanded is None:
                return
            expanded = unfolded_expanded

        has_chord = expanded.chord.notna()
        offset_dict_all_endings = self.offset_dict(all_endings=True)
        if not has_chord.all():
            # Compute duration_qb for chord spans without interruption by other labels, such as phrase and
            # cadence labels, which are considered to have duration 0 and not interrupt the prevailing chord
            offset_dict = self.offset_dict(unfold=unfold)
            with_chord = add_quarterbeats_col(
                expanded[has_chord],
                offset_dict=offset_dict,
                offset_dict_all_endings=offset_dict_all_endings,
                logger=self.logger,
            )
            without_chord = add_quarterbeats_col(
                expanded[~has_chord],
                offset_dict=offset_dict,
                offset_dict_all_endings=offset_dict_all_endings,
                logger=self.logger,
            )
            without_chord.duration_qb = 0.0
            expanded = pd.concat([with_chord, without_chord]).sort_index()
            if interval_index:
                expanded = replace_index_by_intervals(expanded, logger=self.logger)
        else:
            expanded = add_quarterbeats_col(
                expanded,
                offset_dict=self.offset_dict(unfold=unfold),
                offset_dict_all_endings=self.offset_dict(all_endings=True),
                interval_index=interval_index,
                logger=self.logger,
            )
        return expanded

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

    def form_labels(
        self,
        detection_regex: str = None,
        exclude_harmony_layer: bool = False,
        interval_index: bool = False,
        expand: bool = True,
        unfold: bool = False,
    ) -> Optional[pd.DataFrame]:
        """DataFrame representing :ref:`form labels <form_labels>` (or other) that have been encoded as <StaffText>s
        rather than in the <Harmony> layer. This function essentially filters all StaffTexts matching the
        ``detection_regex`` and adds the standard position columns.

        Args:
          detection_regex:
              By default, detects all labels starting with one or two digits followed by a column
              (see :const:`the regex <~.utils.FORM_DETECTION_REGEX>`). Pass another regex to retrieve only StaffTexts
              matching this one.
          exclude_harmony_layer:
              By default, form labels are detected even if they have been encoded as Harmony labels (rather than as
              StaffText). Pass True in order to retrieve only StaffText form labels.
          interval_index: Pass True to replace the default :obj:`~pandas.RangeIndex` by an :obj:`~pandas.IntervalIndex`.

        Returns:
          DataFrame containing all StaffTexts matching the ``detection_regex``
        """
        form = self.parsed.form_labels(
            detection_regex=detection_regex,
            exclude_harmony_layer=exclude_harmony_layer,
            interval_index=interval_index,
        )
        if form is None:
            self.logger.info("The score does not contain any form labels.")
            return
        if unfold:
            form = self.parsed.unfold_facet_df(form, "form labels")
            if form is None:
                return
        if expand:
            if unfold:
                self.logger.warning(
                    "Expanding unfolded form labels has not been tested."
                )
            form = expand_form_labels(form, logger=self.logger)
        return form

    def labels(
        self, interval_index: bool = False, unfold: bool = False
    ) -> Optional[pd.DataFrame]:
        """DataFrame representing all :ref:`labels`, i.e., all <Harmony> tags in the score, as returned by calling
        :meth:`~.annotations.Annotations.get_labels` on the object at :attr:`._annotations` with the current
        :attr:`._labels_cfg`.
        Comes with the columns |quarterbeats|, |duration_qb|, |mc|, |mn|, |mc_onset|, |mn_onset|, |timesig|, |staff|,
        |voice|, |volta|, |harmony_layer|, |label|, |offset_x|, |offset_y|, |regex_match|


        Args:
          interval_index: Pass True to replace the default :obj:`~pandas.RangeIndex` by an :obj:`~pandas.IntervalIndex`.

        Returns:
          DataFrame representing all :ref:`labels`, i.e., all <Harmony> tags in the score.
        """
        if self._annotations is None:
            self.logger.info("The score does not contain any annotations.")
            return None
        labels = self._annotations.get_labels(**self.labels_cfg)
        if unfold:
            labels = self.parsed.unfold_facet_df(labels, "harmony labels")
            if labels is None:
                return
        labels = add_quarterbeats_col(
            labels,
            offset_dict=self.offset_dict(unfold=unfold),
            offset_dict_all_endings=self.offset_dict(all_endings=True),
            interval_index=interval_index,
            logger=self.logger,
        )
        return labels

    def measures(
        self, interval_index: bool = False, unfold: bool = False
    ) -> Optional[pd.DataFrame]:
        """DataFrame representing the :ref:`measures` of the MuseScore file (which can be incomplete measures).
        Comes with the columns |mc|, |mn|, |quarterbeats|, |duration_qb|, |keysig|, |timesig|, |act_dur|,
        |mc_offset|, |volta|, |numbering_offset|, |dont_count|, |barline|, |breaks|, |repeats|, |next|

        Args:
          interval_index: Pass True to replace the default :obj:`~pandas.RangeIndex` by an :obj:`~pandas.IntervalIndex`.

        Returns:
           DataFrame representing the :ref:`measures <measures>` of the MuseScore file
           (which can be incomplete measures).
        """
        return self.parsed.measures(interval_index=interval_index, unfold=unfold)

    def offset_dict(
        self,
        all_endings: bool = False,
        unfold: bool = False,
    ) -> dict:
        """{mc -> offset} dictionary measuring each MC's distance from the piece's beginning (0) in quarter notes."""
        return self.parsed.offset_dict(
            all_endings=all_endings,
            unfold=unfold,
        )

    @property
    def metadata(self):
        """:obj:`dict`
        Shortcut for ``MSCX.parsed.metadata``.
        Metadata from and about the MuseScore file."""
        return self.parsed.metadata

    def notes(
        self, interval_index: bool = False, unfold: bool = False
    ) -> Optional[pd.DataFrame]:
        """DataFrame representing the :ref:`notes` of the MuseScore file. Comes with the columns
        |quarterbeats|, |duration_qb|, |mc|, |mn|, |mc_onset|, |mn_onset|, |timesig|, |staff|, |voice|, |duration|,
        |gracenote|, |tremolo|, |nominal_duration|, |scalar|, |tied|, |tpc|, |midi|, |volta|, |chord_id|


        Args:
          interval_index: Pass True to replace the default :obj:`~pandas.RangeIndex` by an :obj:`~pandas.IntervalIndex`.

        Returns:
          DataFrame representing the :ref:`notes` of the MuseScore file.
        """
        return self.parsed.notes(interval_index=interval_index, unfold=unfold)

    def notes_and_rests(
        self, interval_index: bool = False, unfold: bool = False
    ) -> Optional[pd.DataFrame]:
        """DataFrame representing the :ref:`notes_and_rests` of the MuseScore file. Comes with the columns
        |quarterbeats|, |duration_qb|, |mc|, |mn|, |mc_onset|, |mn_onset|, |timesig|, |staff|, |voice|, |duration|,
        |gracenote|, |tremolo|, |nominal_duration|, |scalar|, |tied|, |tpc|, |midi|, |volta|, |chord_id|

        Args:
          interval_index: Pass True to replace the default :obj:`~pandas.RangeIndex` by an :obj:`~pandas.IntervalIndex`.

        Returns:
          DataFrame representing the :ref:`notes_and_rests` of the MuseScore file.
        """
        return self.parsed.notes_and_rests(interval_index=interval_index, unfold=unfold)

    @property
    def parsed(self) -> _MSCX_bs4:
        """:obj:`~._MSCX_bs4`
        Standard way of accessing the object exposed by the current parser. :obj:`MSCX` uses this object's
        interface for requesting manipulations of and information from the source XML.
        """
        if self._parsed is None:
            self.logger.error("Score has not been parsed yet.")
            return None
        return self._parsed

    def rests(
        self, interval_index: bool = False, unfold: bool = False
    ) -> Optional[pd.DataFrame]:
        """DataFrame representing the :ref:`rests` of the MuseScore file. Comes with the columns
        |quarterbeats|, |duration_qb|, |mc|, |mn|, |mc_onset|, |mn_onset|, |timesig|, |staff|, |voice|, |duration|,
        |nominal_duration|, |scalar|, |volta|

        Args:
          interval_index: Pass True to replace the default :obj:`~pandas.RangeIndex` by an :obj:`~pandas.IntervalIndex`.

        Returns:
          DataFrame representing the :ref:`rests` of the MuseScore file.
        """
        return self.parsed.rests(interval_index=interval_index, unfold=unfold)

    @property
    def staff_ids(self):
        """:obj:`list` of :obj:`int`
        The staff IDs contained in the score, usually just a list of increasing numbers starting at 1.
        """
        return self.parsed.staff_ids

    @property
    def style(self):
        """:obj:`Style`
        Can be used like a dictionary to change the information within the score's <Style> tag.
        """
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
        """Receives the labels from an :py:class:`~.annotations.Annotations` object and adds them to the XML structure
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
        del columns["regex_match"]
        missing_main = {c for c in main_cols if columns[c] not in df.columns}
        assert (
            len(missing_main) == 0
        ), f"The specified columns for the following main parameters are missing:\n{missing_main}"
        if columns["decoded"] not in df.columns:
            df[columns["decoded"]] = decode_harmonies(
                df, label_col=columns["label"], return_series=True, logger=self.logger
            )
        # df = df[df[columns['label']].notna()]
        param2cols = {k: v for k, v in columns.items() if v in df.columns}
        parameters = list(param2cols.keys())
        clmns = list(param2cols.values())
        self.logger.debug(
            f"add_label() will be called with this param2col mapping:\n{param2cols}"
        )
        value_tuples = tuple(df[clmns].itertuples(index=False, name=None))
        params = [dict(zip(parameters, t)) for t in value_tuples]
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
            self.logger.debug(
                f"{changes}/{len(df)} labels successfully added to score."
            )
        return changes

    def change_label_color(
        self,
        mc,
        mc_onset,
        staff,
        voice,
        label,
        color_name=None,
        color_html=None,
        color_r=None,
        color_g=None,
        color_b=None,
        color_a=None,
    ):
        """Shortcut for :py:meth:``MSCX.parsed.change_label_color``

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
        return self.parsed.change_label_color(
            mc=mc,
            mc_onset=mc_onset,
            staff=staff,
            voice=voice,
            label=label,
            color_name=color_name,
            color_html=color_html,
            color_r=color_r,
            color_g=color_g,
            color_b=color_b,
            color_a=color_a,
        )

    def change_labels_cfg(
        self,
        labels_cfg={},
        staff=None,
        voice=None,
        harmony_layer=None,
        positioning=None,
        decode=None,
        column_name=None,
        color_format=None,
    ):
        """Update :py:attr:`.MSCX.labels_cfg`.

        Parameters
        ----------
        labels_cfg : :obj:`dict`
            Using an entire dictionary or, to change only particular options, choose from:
        staff, voice, harmony_layer, positioning, decode
            Arguments as they will be passed to :py:meth:`~.annotations.Annotations.get_labels`
        """
        keys = [
            "staff",
            "voice",
            "harmony_layer",
            "positioning",
            "decode",
            "column_name",
            "color_format",
        ]
        for k in keys:
            val = locals()[k]
            if val is not None:
                labels_cfg[k] = val
        updated = update_labels_cfg(labels_cfg, logger=self.logger)
        self.labels_cfg.update(updated)

    def color_non_chord_tones(
        self,
        df: pd.DataFrame,
        color_name: str = "red",
        chord_tone_cols: Collection[str] = ["chord_tones", "added_tones"],
        color_nan: bool = True,
    ) -> pd.DataFrame:
        """Iterates backwards through the rows of the given DataFrame, interpreting each row as a score segment,
        and colors all notes that do not correspond to one of the tonal pitch classes (TPC) indicated in one of the
        tuples contained in the ``chord_tone_cols``. The columns 'mc' and 'mc_onset' are taken to indicate each score
        segment's start, which reaches to the subsequent one (the last segment reaching to the end of the score). Only
        notes whose onsets lie within the respective segment are colored, meaning that those whose durations reach
        into a segment are not taken into account.

        Args:
          df: A DataFrame with the columns ['mc', 'mc_onset'] + ``chord_tone_cols``
          color_name:
              Name the color that the non-chord tones should get, defaults to 'red'. Name can be a CSS color or
              a MuseScore color (see :py:attr:`utils.MS3_COLORS`).
          chord_tone_cols:
            Names of the columns containing tuples of chord tones, expressed as TPC. Not that in the expanded tables
            extracted by default, these columns correspond to intervals relative to the local tonic. The absolute
            representation required here can be obtained using :attr:`.Annotations.expand_dcml` with ``absolute=True``.
          color_nan:
              By default, if all of the ``chord_tone_cols`` contain a NaN value, all notes in the segment
              will be colored. Pass False to add the segment to the previous one instead.

        Returns:
          A coloring report which is the original ``df`` with the appended columns 'n_colored', 'n_untouched',
          'count_ratio', 'dur_colored', 'dur_untouched', 'dur_ratio'. They contain the counts and durations of the
          colored vs. untouched notes as well the ratio of each pair. Note that the report does not take into account
          notes that reach into a segment, nor does it correct the duration of notes that reach into the subsequent
          segment.
        """
        if self.read_only:
            self.parsed.make_writeable()
        if df is None or len(df) == 0:
            return df
        for col in chord_tone_cols:
            assert (
                col in df.columns
            ), f"DataFrame does not come with a '{col}' column. Specify the parameter chord_tone_cols."
        # iterating backwards through the DataFrame; the first (=last) segment spans to the end of the score
        to_mc, to_mc_onset = None, None
        expand_segment = (
            False  # Flag allowing to add NaN segments to their preceding ones
        )
        results = []
        for row_tuple in df[::-1].itertuples(index=False):
            mc, mc_onset = row_tuple.mc, row_tuple.mc_onset
            chord_tone_tuples = [
                row_tuple.__getattribute__(col) for col in chord_tone_cols
            ]
            if all(pd.isnull(ctt) for ctt in chord_tone_tuples):
                if color_nan:
                    colored_durs, untouched_durs = self.parsed.color_notes(
                        from_mc=mc,
                        from_mc_onset=mc_onset,
                        to_mc=to_mc,
                        to_mc_onset=to_mc_onset,
                        color_name=color_name,
                    )
                else:
                    colored_durs, untouched_durs = [], []
                    expand_segment = True
            else:
                chord_tones = tuple(
                    [
                        ct
                        for ctt in chord_tone_tuples
                        for ct in ctt
                        if not pd.isnull(ctt)
                    ]
                )
                colored_durs, untouched_durs = self.parsed.color_notes(
                    from_mc=mc,
                    from_mc_onset=mc_onset,
                    to_mc=to_mc,
                    to_mc_onset=to_mc_onset,
                    color_name=color_name,
                    tpc=chord_tones,
                    inverse=True,
                )
            n_colored, n_untouched = len(colored_durs), len(untouched_durs)
            if n_colored + n_untouched == 0:
                self.logger.debug(
                    f"MC {mc}, onset {mc_onset}: NaN segment to be merged with the preceding one."
                )
                results.append(())
            else:
                count_ratio = n_colored / (n_colored + n_untouched)
                dur_colored, dur_untouched = float(sum(colored_durs)), float(
                    sum(untouched_durs)
                )
                dur_ratio = dur_colored / (dur_colored + dur_untouched)
                self.logger.debug(
                    f"MC {mc}, onset {mc_onset}: {count_ratio:.1%} of all notes have been coloured, making up for "
                    f"{dur_ratio:.1%} of the summed durations."
                )
                results.append(
                    (
                        n_colored,
                        n_untouched,
                        count_ratio,
                        dur_colored,
                        dur_untouched,
                        dur_ratio,
                    )
                )
            if expand_segment:
                expand_segment = False
            else:
                to_mc, to_mc_onset = mc, mc_onset
        stats = pd.DataFrame(
            reversed(results),
            columns=[
                "n_colored",
                "n_untouched",
                "count_ratio",
                "dur_colored",
                "dur_untouched",
                "dur_ratio",
            ],
            index=df.index,
        )
        stats = stats.astype(dict(n_colored="Int64", n_untouched="Int64"))
        if (stats.n_colored > 0).any():
            self.parsed.parse_measures()
            self.changed = True
        return pd.concat([df, stats], axis=1)

    def delete_labels(self, df):
        """Delete a set of labels from the current XML.

        Parameters
        ----------
        df : :obj:`pandas.DataFrame`
            A DataFrame with the columns ['mc', 'mc_onset', 'staff', 'voice']
        """
        cols = ["mc", "staff", "voice", "mc_onset"]
        # positions = set(df[cols].itertuples(name=None, index=False))
        changed = {
            ix: self._parsed.delete_label(mc, staff, voice, mc_onset)
            for ix, mc, staff, voice, mc_onset in reversed(
                list(df[cols].drop_duplicates().itertuples(name=None, index=True))
            )
        }
        changed = pd.Series(changed, index=df.index).ffill()
        changes = changed.sum()
        if changes > 0:
            self.changed = True
            self._parsed.parse_measures()
            self._update_annotations()
            target = len(df)
            self.logger.debug(f"{changes}/{target} labels successfully deleted.")
            if changes < target:
                self.logger.warning(
                    f"{target - changes} labels could not be deleted:\n{df.loc[~changed]}"
                )

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
        """Remove all empty labels from the attached annotations."""
        if self._annotations is None:
            self.logger.info("No annotations attached.")
            return
        df = self._annotations.get_labels(decode=True)
        label_col = self._annotations.cols["label"]
        sel = df[label_col] == "empty_harmony"
        if sel.sum() == 0:
            self.logger.info("Score contains no empty labels.")
            return
        cols = ["mc", "staff", "voice", "mc_onset"]
        changed = [
            self._parsed.delete_label(mc, staff, voice, mc_onset, empty_only=True)
            for mc, staff, voice, mc_onset in df.loc[sel, cols].itertuples(
                name=None, index=False
            )
        ]
        if sum(changed) > 0:
            self.changed = True
            self._parsed.parse_measures()
            self._update_annotations()
            self.logger.info(f"Successfully deleted {sum(changed)} empty labels.")
        else:
            self.logger.info("No empty labels were deleted.")

    def get_chords(
        self,
        staff=None,
        voice=None,
        mode="auto",
        lyrics=False,
        staff_text=False,
        dynamics=False,
        articulation=False,
        spanners=False,
        **kwargs,
    ):
        """Retrieve a customized chord list, e.g. one including less of the processed features or additional,
        unprocessed ones compared to the standard chord list.

        Parameters
        ----------
        staff : :obj:`int`
            Get information from a particular staff only (1 = upper staff)
        voice : :obj:`int`
            Get information from a particular voice only (1 = only the first layer of every staff)
        mode : {'auto', 'all', 'strict'}, optional
            * 'auto' (default), meaning that those aspects are automatically included that occur in the score; the
              resulting DataFrame has no empty columns except for those parameters that are set to True.
            * 'all': Columns for all aspects are created, even if they don't occur in the score (e.g. lyrics).
            * 'strict': Create columns for exactly those parameters that are set to True, regardless which aspects occur
              in the score.
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
        return self.parsed.get_chords(
            staff=staff,
            voice=voice,
            mode=mode,
            lyrics=lyrics,
            staff_text=staff_text,
            dynamics=dynamics,
            articulation=articulation,
            spanners=spanners,
            **kwargs,
        )

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
        """Shortcut for ``MSCX.parsed.infer_mc()``.
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
        implemented_parsers = ["bs4"]
        if self.parser in implemented_parsers:
            try:
                self._parsed = _MSCX_bs4.from_filepath(
                    self.mscx_src, read_only=self.read_only, logger_cfg=self.logger_cfg
                )
            except Exception:
                self.logger.error(f"Failed parsing {self.mscx_src}.")
                raise
        else:
            raise NotImplementedError(
                f"Only the following parsers are available: {', '.join(implemented_parsers)}"
            )

        self._update_annotations()

    def get_playthrough_mcs(self) -> Optional[pd.Series]:
        return self.parsed.get_playthrough_mcs()

    def write_score_to_handler(self, file_handler: IO) -> bool:
        """Shortcut for ``MSCX.parsed.write_score_to_handler()``.
        Write the current XML structure to a file handler.

        Args:
            file_handler: File handler to write to.

        Returns:
            Whether the file was successfully created.
        """
        if self.read_only:
            self.parsed.make_writeable()
        return self.parsed.write_score_to_handler(file_handler=file_handler)

    def store_score(self, filepath: str) -> bool:
        """Shortcut for ``MSCX.parsed.store_scores()``.
        Store the current XML structure as uncompressed MuseScore file.

        Args:
            filepath:
                Path of the newly created MuseScore file, including the file name ending on '.mscx'.
                Uncompressed files ('.mscz') are not supported.

        Returns:
            Whether the file was successfully created.
        """
        if self.read_only:
            self.parsed.make_writeable()
        return self.parsed.store_score(filepath=filepath)

    def _update_annotations(self, infer_types={}):
        if len(infer_types) == 0 and self._annotations is not None:
            infer_types = self._annotations.regex_dict
        if self._parsed.has_annotations:
            self.has_annotations = True
            logger_cfg = self.logger_cfg.copy()
            # logger_cfg['name'] += '.annotations'
            self._annotations = Annotations(
                df=self.get_raw_labels(),
                read_only=True,
                mscx_obj=self,
                infer_types=infer_types,
                **logger_cfg,
            )
        else:
            self._annotations = None

    def store_excerpt(
        self,
        start_mc: Optional[int] = None,
        start_mn: Optional[int] = None,
        start_mc_onset: Optional[Fraction | float] = None,
        end_mc: Optional[int] = None,
        end_mn: Optional[int] = None,
        end_mc_onset: Optional[Fraction | float] = None,
        exclude_start: Optional[bool] = False,
        exclude_end: Optional[bool] = False,
        metronome_tempo: Optional[float] = None,
        metronome_beat_unit: Optional[Fraction] = Fraction(1 / 4),
        directory: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """Store an excerpt of the current score as a new .mscx file by defining start and end measure. If no end
        measure is specified, the excerpt will include everything following the start measure.
        The original score header and metadata are kept. Start and end measure both can be specified either as MC
        (the number in MuseScore's status bar) or as MN (the number as displayed in the score).

        Args:
            start_mc:
                Measure count of the first measure to be included in the excerpt.
                If ``start_mc`` is given, ``start_mn`` must be None.
            start_mn:
                Measure number of the first measure to be included in the excerpt.
                If ``start_mn`` is given, ``start_mc`` must be None.
            start_mc_onset:
                The starting onset value in the first measure. Every note with onset value strictly smaller than
                ``start_mc_onset`` will be removed from the excerpt.
            end_mc:
                Measure count of the last measure to be included in the excerpt.
                If ``end_mc`` is given, ``end_mn`` must be None.
            end_mn:
                Measure number of the last measure to be included in the excerpt.
                If ``end_mn`` is given, ``end_mc`` must be None.
            end_mc_onset:
                The ending onset value in the last measure. Every not with onset value strictly greate than
                ``end_mc_onset`` will be removed from the excerpt.
            exclude_start:
                If set to True, the note corresponding to ``start_mc_onset`` will be removed as well.
            exclude_end:
                If set to True, the note corresponding to ``end_mc_onset`` will be removed as well.
            metronome_tempo: Optional[float], optional
                Setting this value will override the tempo at the beginning of the excerpt which, otherwise, is created
                automatically according to the tempo in vigour at that moment in the score. This is achieved by
                inserting a hidden metronome marking with a value that depends on the specified "beats per minute",
                where "beat" depends on the value of the ``metronome_beat_unit`` parameter.
            metronome_beat_unit: Optional[Fraction | float], optional
                Defaults to 1/4, which stands for a quarter note. Please note that for now,
                the combination of beat unit and tempo is converted and expressed as quarter notes per
                minute in the (invisible) metronome marking. For example, specifying 1/8=100 will effectively result
                in 1/4=50 (which is equivalent).
            directory:
                Path to the folder where the excerpts are to be stored.
            suffix:
                String to be inserted in the excerpts filename[suffix]_[start_mc]-[end_mc]

        Returns:
            Optional[None]: if it was impossible to find a quarterbeat value for the given start measure.
                            In this case the function will not produce an excerpt.

        """
        assert (start_mc is None) + (
            start_mn is None
        ) == 1, "Exactly one of start_mc or start_mn must be provided."

        if end_mc is not None and end_mn is not None:
            raise ValueError(
                "Exactly one of end_mc or end_mn must be provided or None."
            )

        for arg, arg_val in zip(
            ("start_mc", "start_mn", "end_mc", "end_mn"),
            (start_mc, start_mn, end_mc, end_mn),
        ):
            if arg_val is not None and not isinstance(arg_val, int):
                raise TypeError(
                    f"{arg} must be an integer. Got {arg_val!r} ({type(arg_val)!r})."
                )

        if suffix is None:
            suffix = ""

        measures = self.measures()
        mc = measures["mc"]
        mn = measures["mn"]

        # Setting ending mc value
        if end_mc is None:
            if end_mn is None:
                end = mc.max()
            elif end_mn not in mn.values:
                raise ValueError(
                    f"Score has no measure number {end_mn} to end an excerpt on."
                )
            else:
                end = measures.loc[mn == end_mn, "mc"].iloc[-1]
        else:
            end = end_mc

        # Setting starting mc value
        if start_mc is None:
            if start_mn not in mn.values:
                raise ValueError(
                    f"Score has no measure number {start_mn} to start an excerpt from."
                )
            start = measures.loc[mn == start_mn, "mc"].iloc[0]
        else:
            start = start_mc

        global_key, local_key = None, None
        dcml_labels = self.expanded()
        if dcml_labels is not None and len(dcml_labels) > 0:
            # try to infer global key and local key from the annotations
            mc_measures = measures.set_index("mc")
            quarterbeat_start = mc_measures.loc[start, "quarterbeats"]
            if pd.isnull(quarterbeat_start):
                self.logger.error(
                    f"The given start MC {start} has no quarterbeat value and no globalkey and localkey "
                    f"could be inferred. Probably it is a first ending."
                )
            else:
                row = get_row_at_quarterbeat(
                    df=dcml_labels, quarterbeat=quarterbeat_start
                )

                if row is not None:
                    global_key = row["globalkey"]
                    local_key = row["localkey"]

        included_mcs = tuple(range(start, end + 1))

        self.logger.debug(
            f"Start: {start}, End: {end}. Total number of measures: {len(included_mcs)}"
        )
        self.logger.debug(f"Global key: {global_key}, Local key: {local_key}")

        excerpt = self.parsed.make_excerpt(
            included_mcs=included_mcs,
            start_mc_onset=start_mc_onset,
            end_mc_onset=end_mc_onset,
            exclude_start=exclude_start,
            exclude_end=exclude_end,
            metronome_tempo=metronome_tempo,
            metronome_beat_unit=metronome_beat_unit,
            globalkey=global_key,
            localkey=local_key,
        )

        original_directory, original_filename = os.path.split(excerpt.filepath)
        original_file_name = os.path.splitext(original_filename)[0]
        new_file_name = original_file_name + f"{suffix}_{start}-{end}" + ".mscx"
        if directory is None:
            excerpt_filepath = os.path.join(original_directory, new_file_name)
        else:
            excerpt_filepath = os.path.join(directory, new_file_name)
        excerpt.store_score(excerpt_filepath)
        self.logger.info(f"Excerpt for MCs {start}-{end} stored at {excerpt_filepath}.")

    def get_phrase_boundaries(self):
        """This method uses the `expanded` and `unfolded` labels to find all the phrase boundaries where a beginning
        is defined by an opening bracket `{` and the end is defined by a cadence. This cadence can either come with a
        closing bracket `}` or after the end of a phrase and before the beginning of the next one. The start and end
        point are also associated with onset values to precisely know the position of the labels within the measure
        in order to be able to trim `"unrelated"` notes later on.

        Returns:
            a list of all unique maps that identify all phrases in the score. Each map has three keys: `"mcs"`,
            `"start_onset"`, `"end_onset"`. The first one corresponds to a tuple containing all the measure counts
            included in the phrase, the second is onset value of the starting label and the last key is the onset
            value for the ending label.
        """

        expanded = self.expanded(unfold=True)
        if expanded is None or len(expanded) == 0:
            self.logger.info("No DCML labels found to extract phrase information from.")
            return

        phrase_label_mask = expanded["label"].str.contains("{|}|\\|")
        if not phrase_label_mask.any():
            self.logger.info(
                "DCML labels do not contain phrase labels with curly brackets {}"
            )
            return

        df = expanded[phrase_label_mask][["mc_onset", "label", "phraseend"]]

        filter1 = df["phraseend"].isna() & (df["phraseend"].shift(1).isin(["{", "}{"]))
        df = df.drop(df[filter1].index)

        filter2 = (df["phraseend"] == "}") & (df["phraseend"].shift(-1).isna())
        if df.iloc[-1]["phraseend"] == "}":
            filter2 &= df.index != df.index[-1]
        final = df.drop(df[filter2].index)

        start_mask = final["phraseend"].isin(["{", "}{"])
        end_mask = final["phraseend"].isin(["}", "}{"]) | final["phraseend"].isna()

        start_indexes = final.index[start_mask].tolist()
        end_indexes = final.index[end_mask].tolist()

        if len(start_indexes) != len(end_indexes):
            self.logger.warning("Incoherent labels, aborting...")

        phrases = []
        unique_mcs = []
        for start, end in zip(start_indexes, end_indexes):
            phrase = {
                "mcs": tuple(
                    expanded.iloc[start : end + 1]["mc"].drop_duplicates().values
                ),
                "start_onset": expanded.iloc[start]["mc_onset"],
                "end_onset": expanded.iloc[end]["mc_onset"],
            }
            if phrase["mcs"] not in unique_mcs:
                phrases.append(phrase)
                unique_mcs.append(phrase["mcs"])

        return phrases

    def store_phrase_excerpts(
        self,
        metronome_tempo: Optional[float] = None,
        metronome_beat_unit: Optional[Fraction | float] = Fraction(1 / 4),
        directory: Optional[str] = None,
        suffix: Optional[str] = "phrase",
        random_skip: Optional[bool] = False,
    ):
        """Store excerpts based on the phrase annotations contained in the score, if any. For this purpose,
        the self.find_phrases() method is called; for each pair of start and end MC an excerpt will be stored.
        The resulting excerpts will be named ``[original_filename]_phrase_[start_mc]-[end_mc].mscx`` by default or
        ``[original_filename]_[suffix]_[start_mc]-[end_mc].mscx`` if ``suffix`` is specified.

        Args:
            metronome_tempo: Optional[float], optional
                Setting this value will override the tempo at the beginning of the excerpt which, otherwise, is created
                automatically according to the tempo in vigour at that moment in the score. This is achieved by
                inserting a hidden metronome marking with a value that depends on the specified "beats per minute",
                where "beat" depends on the value of the ``metronome_beat_unit`` parameter.

            metronome_beat_unit: Optional[Fraction | float], optional
                Defaults to 1/4, which stands for a quarter note. Please note that for now,
                the combination of beat unit and tempo is converted and expressed as quarter notes per
                minute in the (invisible) metronome marking. For example, specifying 1/8=100 will effectively result
                in 1/4=50 (which is equivalent).

            directory: Optional[str], optional
                name of the directory you want the excerpt saved to, by default None

            suffix: Optional[str], optional
                It is the string "category identifier" of your excerpts. For instance the name of the output files will
                in general be ``[original_filename]_[suffix]_[start_mc]-[end_mc].mscx``

            random_skip: Optional[bool], optional
                This boolean value, if True, will make the method randomly skip extracted
                excerpts and don't generate them. This parameter is set by default to False.
        """

        phrases = self.get_phrase_boundaries()

        for phrase in phrases:
            if random_skip and np.random.choice([True, False]):
                continue
            self.store_measures(
                included_mcs=phrase["mcs"],
                start_mc_onset=phrase["start_onset"],
                end_mc_onset=phrase["end_onset"],
                exclude_end=False,
                metronome_tempo=metronome_tempo,
                metronome_beat_unit=metronome_beat_unit,
                directory=directory,
                suffix=suffix,
            )

        self.logger.info(f"Extracted {len(phrases)} phrases.\n" f"Phrases: {phrases}")

    def store_measures(
        self,
        included_mcs: Tuple[int, ...],
        start_mc_onset: Optional[Fraction | float] = None,
        end_mc_onset: Optional[Fraction | float] = None,
        exclude_start: Optional[bool] = False,
        exclude_end: Optional[bool] = False,
        metronome_tempo: Optional[float] = None,
        metronome_beat_unit: Optional[Fraction | float] = Fraction(1 / 4),
        directory: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """This method takes a tuple containing the number of the measures that contained in the excerpt to be
        stored. The method will infer the active global and local keys, relative to the excerpt, from the annotations.
        It will then store the excerpt in the given (or default) directory with the name
        ``[original_filename]_[suffix]_[start_mc]-[end_mc].mscx``.

        Args:
            included_mcs: Tuple[int]
                The mc values of the measures to be included in the excerpt

            start_mc_onset: Optional[Fraction | float], optional
                The value of the chosen onset for the true start of the excerpt. If onset is ``None`` or ``0``, then the
                excerpt will normally begin on the onset of the first included measure. In the case where this value
                should be different, for example ``1/2`` or ``.5``, then all the notes with onset strictly smaller than
                this value will be removed from the first measure.

            end_mc_onset: Optional[Fraction | float], optional
                This has the same behaviour as the previous parameter. This means that if is set to None or to the value
                of the last onset in the measure, then the excerpt will normally finish at the end of the last included
                measure. In the cse where this value should be different, for example ``1/2`` or ``.5``,
                then all notes with onset strictly greater than this value will be removed from the last measure.

            exclude_start: Optional[bool], optional
                If set to True the note (in first measure) with onset value equal to ``start_mc_onset`` will also be
                removed thus excluding the first onset (i.e. the end)

            exclude_end: Optional[bool], optional
                If set to True the note (in last measure) with onset value equal to ``end_mc_onset`` will also be
                removed thus excluding the last onset (i.e. the end)

            metronome_tempo: Optional[float], optional
                Setting this value will override the tempo at the beginning of the excerpt which, otherwise, is created
                automatically according to the tempo in vigour at that moment in the score. This is achieved by
                inserting a hidden metronome marking with a value that depends on the specified "beats per minute",
                where "beat" depends on the value of the ``metronome_beat_unit`` parameter.

            metronome_beat_unit: Optional[Fraction | float], optional
                Defaults to 1/4, which stands for a quarter note. Please note that for now,
                the combination of beat unit and tempo is converted and expressed as quarter notes per
                minute in the (invisible) metronome marking. For example, specifying 1/8=100 will effectively result
                in 1/4=50 (which is equivalent).

            directory: Optional[str], optional
                name of the directory you want the excerpt saved to, by default None

            suffix: Optional[str], optional
                It is the string "category identifier" of your excerpts. For instance the name of the output files will
                in general be ``[original_filename]_[suffix]_[start_mc]-[end_mc].mscx``
        """
        measures = self.measures()

        if len(included_mcs) == 0:
            self.logger.warning(
                "Tuple passed as argument was found empty. Nothing to do."
            )
        else:
            start = included_mcs[0]
            end = included_mcs[-1]

            global_key, local_key = None, None
            dcml_labels = self.expanded()
            if dcml_labels is not None and len(dcml_labels) > 0:
                # try to infer global key and local key from the annotations
                mc_measures = measures.set_index("mc")
                quarterbeat_start = mc_measures.loc[start, "quarterbeats"]
                if pd.isnull(quarterbeat_start):
                    self.logger.error(
                        f"The given start MC {start} has no quarterbeat value and no globalkey and localkey "
                        f"could be inferred. Probably it is a first ending."
                    )
                else:
                    row = get_row_at_quarterbeat(
                        df=dcml_labels, quarterbeat=quarterbeat_start
                    )

                    if row is not None:
                        global_key = row["globalkey"]
                        local_key = row["localkey"]

            excerpt = self.parsed.make_excerpt(
                included_mcs=included_mcs,
                globalkey=global_key,
                localkey=local_key,
                start_mc_onset=start_mc_onset,
                end_mc_onset=end_mc_onset,
                exclude_start=exclude_start,
                exclude_end=exclude_end,
                metronome_tempo=metronome_tempo,
                metronome_beat_unit=metronome_beat_unit,
                decompose_repeat_tags=True,
            )

            original_directory, original_filename = os.path.split(excerpt.filepath)
            original_file_name = os.path.splitext(original_filename)[0]
            new_file_name = (
                original_file_name
                + f"_{suffix if suffix is not None else ''}_{start}-{end}"
                + ".mscx"
            )
            if directory is None:
                excerpt_filepath = os.path.join(original_directory, new_file_name)
            else:
                resolve_dir(directory)
                os.makedirs(directory, exist_ok=True)
                excerpt_filepath = os.path.join(directory, new_file_name)
            excerpt.store_score(excerpt_filepath)
            self.logger.info(
                f"Excerpt for MCs {start}-{end} stored at {excerpt_filepath}."
            )

    def store_within_phrase_excerpts(
        self,
        metronome_tempo: Optional[float] = None,
        metronome_beat_unit: Optional[Fraction | float] = Fraction(1 / 4),
        directory: Optional[str] = None,
        suffix: Optional[str] = "within_phrase",
        random_skip: Optional[bool] = False,
    ):
        """Extract random snippets from the given score. The snippets have the constraint that they must strictly
        lie within a phrase. This means that within this type of excerpt neither phrase beginnings nor phrase endings
        will be considered. Not even cadences. By default, it extracts all possible
        snippets and stores them at the optional directory path.
        The resulting excerpts will be named ``[original_filename]_within_phrase_[start_mc]-[end_mc].mscx``.

        Args:
            metronome_tempo: Optional[float], optional
                The value that the user wants to set as the tempo of the excerpts. The tag will be added
                to XML tree of the excerpt's file and will have the desired tempo

            metronome_beat_unit: Optional[Fraction | float], optional
                To obtain the correct value for the tempo it is important to specify the beat unit that corresponds
                to the given tempo value. Since MuseScore works in quarter-beats, the convention is that 1 indicates
                that the unit is the quarter beat and all other values are relative to this one (i.e. 1/2 would be the
                eighth note etc.)

            directory: Optional[str], optional
                name of the directory you want the excerpt saved to, by default None

            suffix: Optional[str], optional
                It is the string "category identifier" of your excerpts. For instance the name of the output files will
                in general be ``[original_filename]_[suffix]_[start_mc]-[end_mc].mscx``

            random_skip: Optional[bool], optional
                This boolean value, if True, will make the method randomly skip extracted
                excerpts and don't generate them. This parameter is set by default to False.
        """

        expanded = self.expanded(unfold=True)
        measures = self.measures(unfold=True)

        df = expanded[expanded["label"].str.contains("{|}|\\|")].copy()
        last_phrase_label = ""
        rows_to_drop = []
        for index, row in df.iterrows():
            if pd.notna(row["phraseend"]):
                last_phrase_label = row["phraseend"]
            if (
                "|" in row["label"]
                and last_phrase_label == "}"
                and pd.isna(row["phraseend"])
            ):
                rows_to_drop.append(index)
        df.drop(rows_to_drop, inplace=True)

        available_mcs = [
            tuple(
                measures.iloc[
                    measures.loc[
                        measures["mn_playthrough"] == df.iloc[i]["mn_playthrough"]
                    ].index[0]
                    + 1 : measures.loc[
                        measures["mn_playthrough"] == df.iloc[i + 1]["mn_playthrough"]
                    ].index[0]
                ]["mc"].values
            )
            for i in range(len(df) - 1)
        ]

        available_mcs = [x for x in available_mcs if len(x) >= 2]
        available_mcs = list(set(available_mcs))

        if available_mcs:
            for included_mcs in available_mcs:
                for i in range(len(included_mcs) - 1):
                    if random_skip and np.random.choice([True, False]):
                        continue
                    self.store_measures(
                        included_mcs=(included_mcs[i : i + 2]),
                        start_mc_onset=None,
                        end_mc_onset=None,
                        metronome_tempo=metronome_tempo,
                        metronome_beat_unit=metronome_beat_unit,
                        directory=directory,
                        suffix=suffix,
                    )
        else:
            self.logger.info("No measures to be stored.")

    def store_phrase_endings(
        self,
        metronome_tempo: Optional[float] = None,
        metronome_beat_unit: Optional[Fraction | float] = Fraction(1 / 4),
        max_excerpt_length: Optional[int] = 2,
        directory: Optional[str] = None,
        suffix: Optional[str] = "phrase_end",
        random_skip: Optional[bool] = False,
    ):
        """Calls the self.find_phrase_endings() method to find all phrase endings contained in the score, then stores
        all corresponding excerpts. A phrase ending is specified to finish on a cadence and to start 2 MCs before the
        corresponding closing bracket that indicates the "end" of the phrase. The resulting excerpts will be named
        ``[original_filename]_[suffix]_[start_mc]-[end_mc].mscx``.

        Args:
            metronome_tempo: Optional[float], optional
                The value that the user wants to set as the tempo of the excerpts. The tag will be added
                to XML tree of the excerpt's file and will have the desired tempo

            metronome_beat_unit: Optional[Fraction | float], optional
                To obtain the correct value for the tempo it is important to specify the beat unit that corresponds
                to the given tempo value. Since MuseScore works in quarter-beats, the convention is that 1 indicates
                that the unit is the quarter beat and all other values are relative to this one (i.e. 1/2 would be the
                eighth note etc.)

            max_excerpt_length: Optional[int], optional
                This parameter specifies the maximum number of measures to be included in the excerpt. For example,
                if `max_excerpt_length` is set to `3`, all phrase endings excerpts will contain a max. of 3 measures.

            directory: Optional[str], optional
                name of the directory you want the excerpt saved to, by default None

            suffix: Optional[str], optional
                It is the string "category identifier" of your excerpts. For instance the name of the output files will
                in general be ``[original_filename]_[suffix]_[start_mc]-[end_mc].mscx``

            random_skip: Optional[bool], optional
                This boolean value, if True, will make the method randomly skip extracted
                excerpts and don't generate them. This parameter is set by default to False.
        """

        # phrase_endings = self.find_phrase_endings()
        phrase_boundaries = self.get_phrase_boundaries()

        if phrase_boundaries:
            for phrase in phrase_boundaries:
                if random_skip and np.random.choice([True, False]):
                    continue
                included_mcs = phrase["mcs"][-max_excerpt_length:]
                self.store_measures(
                    included_mcs=included_mcs,
                    start_mc_onset=0,
                    end_mc_onset=phrase["end_onset"],
                    exclude_end=False,
                    metronome_tempo=metronome_tempo,
                    metronome_beat_unit=metronome_beat_unit,
                    directory=directory,
                    suffix=suffix,
                )
        else:
            self.logger.info("No phrases to be stored.")

    def store_random_excerpts(
        self,
        n_excerpts: Optional[int] = None,
        mc_length: Optional[int] = 2,
        metronome_tempo: Optional[float] = None,
        metronome_beat_unit: Optional[Fraction | float] = Fraction(1 / 4),
        directory: Optional[str] = None,
        suffix: Optional[str] = "random",
    ):
        """Method that stores ``n_excerpts`` random excerpts each ``mn_lengths`` measures long. If ``n_excerpts`` is
        not specified then the method will create the maximum possible number of different excerpts containing
        ``mn_length`` measures each.

        Args:
            n_excerpts:
                The number of random excerpts to be created
            mc_length:
                The allowed number of measures for each excerpt
            metronome_tempo:
                The tempo value that the user might specify to overwrite the original piece tempo
            metronome_beat_unit:
                Beat unit value that goes with the specified tempo value. Might be ``1/4`` if the unit is the quarter
                note, ``1/8`` if the unit is the eighth note and so on.
            directory:
                Name of the directory into which the excerpts need to be stored
            suffix:
                Suffix to be added to the name of the generated excerpts
        """
        measures = self.measures()
        mc_values = np.array(measures["mc"].tolist())
        max_mc = int(max(mc_values))

        if len(mc_values) != max_mc:
            self.logger.error("Incoherent measures table. Aborting...")
            return

        if mc_length >= max_mc:
            self.logger.warning(
                "You are either requesting an excerpt longer than the original piece or with the same "
                "length. Aborting..."
            )
            return

        rng = np.random.default_rng()
        available_starts = mc_values[: -mc_length + 1] - 1

        if n_excerpts is None:
            random_excerpts = [
                tuple(mc_values[x : x + mc_length]) for x in available_starts
            ]
        else:
            random_excerpts = [
                tuple(mc_values[x : x + mc_length])
                for x in rng.choice(available_starts, n_excerpts, replace=False)
            ]

        for excerpt in random_excerpts:
            self.store_measures(
                included_mcs=excerpt,
                metronome_tempo=metronome_tempo,
                metronome_beat_unit=metronome_beat_unit,
                directory=directory,
                suffix=suffix,
            )

    def update_metadata(
        self,
        composer: Optional[str] = None,
        workTitle: Optional[str] = None,
        movementNumber: Optional[str] = None,
        movementTitle: Optional[str] = None,
        workNumber: Optional[str] = None,
        poet: Optional[str] = None,
        lyricist: Optional[str] = None,
        arranger: Optional[str] = None,
        copyright: Optional[str] = None,
        creationDate: Optional[str] = None,
        mscVersion: Optional[str] = None,
        platform: Optional[str] = None,
        source: Optional[str] = None,
        translator: Optional[str] = None,
        compared_against: Optional[str] = None,  # resolve
        **kwargs,
    ):
        """Update the metadata tags of the parsed score."""
        metadata_update = {
            key: value
            for key, value in (
                ("composer", composer),
                ("workTitle", workTitle),
                ("movementNumber", movementNumber),
                ("movementTitle", movementTitle),
                ("workNumber", workNumber),
                ("poet", poet),
                ("lyricist", lyricist),
                ("arranger", arranger),
                ("copyright", copyright),
                ("creationDate", creationDate),
                ("mscVersion", mscVersion),
                ("platform", platform),
                ("source", source),
                ("translator", translator),
                ("compared_against", compared_against),
            )
            if value is not None
        }
        metadata_update.update({key: value for key, value in kwargs.items() if value})
        assert (
            len(metadata_update) > 0
        ), "Didn't receive any parameters to update the metadata."
        MSCX = self.parsed
        current_metadata = MSCX.metatags.fields
        current_metadata.update(MSCX.prelims.fields)
        changed = False
        for field, value in metadata_update.items():
            specifier = (
                "New field" if field in field not in current_metadata else "Field"
            )
            self.logger.debug(f"{specifier} {field!r} set to {value}.")
            MSCX.metatags[field] = value
            changed = True
        if changed:
            MSCX.update_metadata()
            self.changed = True
        else:
            self.logger.debug("No metadata updated.")


# ######################################################################################################################
# ######################################################################################################################
# ############################################### End of MSCX() ########################################################
# ######################################################################################################################
# ######################################################################################################################


class Score(LoggedClass):
    """Object representing a score."""

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

    native_formats = ("mscx", "mscz")
    """:obj:`tuple`
    Formats that MS3 reads without having to convert.
    """

    convertible_formats = (
        "cap",
        "capx",
        "midi",
        "mid",
        "musicxml",
        "mxl",
        "xml",
    )
    """:obj:`tuple`
    Formats that have to be converted before parsing.
    """

    parseable_formats = native_formats + convertible_formats
    """:obj:`tuple`
    Formats that ms3 can parse.
    """

    dataframe_types = (
        "measures",
        "notes",
        "rests",
        "notes_and_rests",
        "labels",
        "expanded",
        "form_labels",
        "cadences",
        "events",
        "chords",
    )

    def __init__(
        self,
        musescore_file=None,
        match_regex=["dcml", "form_labels"],
        read_only=False,
        labels_cfg={},
        parser="bs4",
        ms=None,
        **logger_cfg,
    ):
        """

        Parameters
        ----------
        musescore_file : :obj:`str`, optional
            Path to the MuseScore file to be parsed.
        match_regex : :obj:`list` or :obj:`dict`, optional
            Determine which label types are determined automatically. Defaults to ['dcml'].
            Pass ``{'type_name': r"^(regular)(Expression)$"}`` to call :meth:`ms3.Score.new_type`.
        read_only : :obj:`bool`, optional
            Defaults to ``False``, meaning that the parsing is slower and uses more memory in order to allow for
            manipulations of the score, such as adding and deleting labels. Set to ``True`` if you're only extracting
            information.
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
            If you want to parse musicXML files or MuseScore 2 files by temporarily converting them, pass the path or
            command of your local MuseScore 3 installation. If you're using the standard path, you may try 'auto',
            or 'win' for Windows, 'mac' for MacOS, or 'mscore' for Linux.
        """
        super().__init__(subclass="Score", logger_cfg=logger_cfg)

        self.read_only = read_only
        """:obj:`bool`, optional
        Defaults to ``False``, meaning that the parsing is slower and uses more memory in order to allow for
        manipulations of the score, such as adding and deleting labels. Set to ``True`` if you're only extracting
        information."""

        if musescore_file is not None:
            assert os.path.isfile(
                musescore_file
            ), f"File does not exist: {musescore_file}"
        self.musescore_file = musescore_file

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

        self._mscx: MSCX = None
        """ The object representing the parsed MuseScore file."""

        self._detached_annotations = {}
        """:obj:`dict`
        ``{(key, i): Annotations object}`` dictionary for accessing all detached :py:class:`~.annotations.Annotations`
        objects.
        """

        self._types_to_infer = []
        """:obj:`list`
        Current order in which types are being recognized."""

        self._harmony_layer_description = {
            0: "Simple string (does not begin with a note name, otherwise MS3 will turn it into type 3; "
            "prevent through leading dot)",
            1: "MuseScore's Roman Numeral Annotation format",
            2: "MuseScore's Nashville Number format",
            3: "Absolute chord encoded by MuseScore",
        }

        self._harmony_layer_regex = {
            1: self.RN_REGEX,
            2: self.NASHVILLE_REGEX,
            3: self.ABS_REGEX,
        }

        self._regex_name_description = {
            "dcml": "Latest version of the DCML harmonic annotation standard.",
            "form_labels": "Form labels that have been encoded as harmonies rather than as StaffText.",
        }
        """:obj:`dict`
        Mapping regex names to their descriptions.
        """

        self._name2regex = {
            "dcml": DCML_DOUBLE_REGEX,
            "form_labels": FORM_DETECTION_REGEX,
        }
        """:obj:`dict`
        Mapping names to their corresponding regex. Managed via the property :py:attr:`name2regex`.
        'dcml': utils.DCML_REGEX,
        """

        self.labels_cfg = {
            "staff": None,
            "voice": None,
            "harmony_layer": None,
            "positioning": False,
            "decode": True,
            "column_name": "label",
            "color_format": None,
        }
        """:obj:`dict`
        Configuration dictionary to determine the output format of the :py:class:`~.annotations.Annotations`
        objects contained in the current object, especially when calling :meth:`Score.mscx.labels()<.MSCX.labels>`.
        The default options correspond to the default parameters of
        :py:meth:`Annotations.get_labels()<.annotations.Annotations.get_labels>`.
        """
        self.labels_cfg.update(update_labels_cfg(labels_cfg, logger=self.logger))

        self.parser = parser
        """{'bs4'}
        Currently only one XML parser has been implemented which uses BeautifulSoup 4.
        """

        self.review_report = pd.DataFrame()
        """:obj:`pandas.DataFrame`
        After calling :py:meth:`color_non_chord_tones`, this DataFrame contains the expanded chord labels
        plus the six additional columns ['n_colored', 'n_untouched', 'count_ratio', 'dur_colored', 'dur_untouched',
        'dur_ratio'] representing the statistics of chord (untouched) vs. non-chord (colored) notes.
        """

        self.comparison_report = pd.DataFrame()
        """:obj:`pandas.DataFrame`
        DataFrame showing the labels modified ('new') and added ('old') by :py:meth:`compare_labels`.
        """

        self.name2regex = match_regex
        if self.musescore_file is not None:
            self.parse_mscx()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%% END of __init__() %%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    @property
    def name2regex(self):
        """:obj:`list` or :obj:`dict`, optional
        The order in which label types are to be inferred.
        Assigning a new value results in a call to :py:meth:`~.annotations.Annotations.infer_types`.
        Passing a {label type: regex} dictionary is a shortcut to update type regex's or to add new ones.
        The inference will take place in the order in which they appear in the dictionary. To reuse an existing
        regex will updating others, you can refer to them as None, e.g. ``{'dcml': None, 'my_own': r'^(PAC|HC)$'}``.
        """
        return self._types_to_infer

    @name2regex.setter
    def name2regex(self, val):
        if val is None:
            val = []
        before_inf, before_reg = self._types_to_infer, self.get_infer_regex()
        if isinstance(val, list):
            exist = [v for v in val if v in self._name2regex]
            if len(exist) < len(val):
                self.logger.warning(
                    f"The following harmony types have not been added via the new_type() method:\n"
                    f"{[v for v in val if v not in self._name2regex]}"
                )
            self._types_to_infer = exist
        elif isinstance(val, dict):
            for k, v in val.items():
                if k in self._name2regex:
                    if v is None:
                        val[k] = self._name2regex[k]
                    else:
                        self._name2regex[k] = v
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
        Is True as long as the score contains :py:class:`~.annotations.Annotations` objects, that are not attached to
        the :obj:`MSCX` object.
        """
        return len(self._detached_annotations) > 0

    @property
    def mscx(self) -> MSCX:
        """Standard way of accessing the parsed MuseScore file."""
        if self._mscx is None:
            raise LookupError(
                "No XML has been parsed so far. Use the method parse_mscx()."
            )
        return self._mscx

    @property
    def types(self):
        """:obj:`dict`
        Shows the mapping of label types to their descriptions."""
        return self._regex_name_description

    def attach_labels(
        self,
        key,
        staff=None,
        voice=None,
        harmony_layer=None,
        check_for_clashes=True,
        remove_detached=True,
    ):
        """Insert detached labels ``key`` into this score's :obj:`MSCX` object.

        Parameters
        ----------
        key : :obj:`str`
            Key of the detached labels you want to insert into the score.
        staff : :obj:`int`, optional
            By default, labels are added to staves as specified in the TSV or to -1 (lowest).
            Pass an integer to specify a staff.
        voice : :obj:`int`, optional
            By default, labels are added to voices (notational layers) as specified in the TSV or to 1 (main voice).
            Pass an integer to specify a voice.
        harmony_layer : :obj:`int`, optional
            | By default, the labels are written to the layer specified as an integer in the column ``harmony_layer``.
            | Pass an integer to select a particular layer:
            | * 0 to attach them as absolute ('guitar') chords, meaning that when opened next time,
            |   MuseScore will split and encode those beginning with a note name ( resulting in ms3-internal
                harmony_layer 3).
            | * 1 the labels are written into the staff's layer for Roman Numeral Analysis.
            | * 2 to have MuseScore interpret them as Nashville Numbers
        check_for_clashes : :obj:`bool`, optional
            Defaults to True, meaning that the positions where the labels will be inserted will be checked for existing
            labels.
        remove_detached : :obj:`bool`, optional
            By default, the detached :py:class:`~.annotations.Annotations` object is removed after successfully
            attaching it. Pass False to have it remain in detached state.

        Returns
        -------
        :obj:`int`
            Number of newly attached labels.
        :obj:`int`
            Number of labels that were to be attached.
        """
        assert self._mscx is not None, "No score has been parsed yet."
        assert (
            key != "annotations"
        ), "Labels with key 'annotations' are already attached."
        if key not in self._detached_annotations:
            self.mscx.logger.info(
                f"Key {key!r} doesn't correspond to a detached set of annotations. Use one of the existing keys or "
                f"load a new set with the method load_annotations().\nExisting keys:"
                f" {list(self._detached_annotations.keys())}"
            )
            return 0, 0

        annotations = self._detached_annotations[key]
        goal = len(annotations.df)
        if goal == 0:
            self.mscx.logger.warning(
                f"The Annotation object '{key}' does not contain any labels."
            )
            return 0, 0
        df = annotations.prepare_for_attaching(
            staff=staff,
            voice=voice,
            harmony_layer=harmony_layer,
            check_for_clashes=check_for_clashes,
        )
        reached = len(df)
        if reached == 0:
            self.mscx.logger.error(
                f"No labels from '{key}' have been attached due to aforementioned errors."
            )
            return reached, goal

        prepared_annotations = Annotations(
            df=df,
            cols=annotations.cols,
            infer_types=annotations.regex_dict,
            **self.logger_cfg,
        )
        reached = self.mscx.add_labels(prepared_annotations)
        if remove_detached:
            if reached == goal:
                del self._detached_annotations[key]
                self.mscx.logger.debug(
                    f"Detached annotations '{key}' successfully attached and removed."
                )
            else:
                self.mscx.logger.info(
                    f"Only {reached} of the {goal} targeted labels could be attached, so '{key}' was not removed."
                )
        return reached, goal

    def change_labels_cfg(
        self,
        labels_cfg={},
        staff=None,
        voice=None,
        harmony_layer=None,
        positioning=None,
        decode=None,
        column_name=None,
        color_format=None,
    ):
        """Update :py:attr:`.Score.labels_cfg` and :py:attr:`.MSCX.labels_cfg`.

        Parameters
        ----------
        labels_cfg : :obj:`dict`
            Using an entire dictionary or, to change only particular options, choose from:
        staff, voice, harmony_layer, positioning, decode
            Arguments as they will be passed to :py:meth:`~.annotations.Annotations.get_labels`
        """
        keys = [
            "staff",
            "voice",
            "harmony_layer",
            "positioning",
            "decode",
            "column_name",
            "color_format",
        ]
        for k in keys:
            val = locals()[k]
            if val is not None:
                labels_cfg[k] = val
        updated = update_labels_cfg(labels_cfg, logger=self.logger)
        self.labels_cfg.update(updated)
        self.mscx.labels_cfg.update(updated)

    def check_labels(self, keys="annotations", regex=None, regex_name="dcml", **kwargs):
        """Tries to match the labels ``keys`` against the given ``regex`` or the one of the registered ``regex_name``.
        Returns wrong labels.

        Parameters
        ----------
        keys : :obj:`str` or :obj:`Collection`, optional
            The key(s) of the Annotation objects you want to check. Defaults to 'annotations', the attached labels.
        regex : :obj:`str`, optional
            Pass a regular expression against which to check the labels if you don't want to use the one of an existing
            ``regex_name`` or in order to register a new one on the fly by passing the new name as ``regex_name``.
        regex_name : :obj:`str`, optional
            To use the regular expression of a registered type, pass its name, defaults to 'dcml'. Pass a new name and
            a ``regex`` to register a new label type on the fly.
        kwargs :
            Parameters passed to :py:func:`~.utils.check_labels`.

        Returns
        -------
        :obj:`pandas.DataFrame`
            Labels not matching the regex.
        """
        if keys == "annotations" and not self.mscx.has_annotations:
            self.mscx.logger.debug("Score contains no Annotations.")
            return
        if regex is None:
            if regex_name in self._name2regex:
                regex = self._name2regex[regex_name]
            else:
                self.logger.warning(
                    f"Type {regex_name} has not been registered. Pass a regular expression for it as argument 'regex'."
                )
                return
        else:
            if regex.__class__ != re.compile("").__class__:
                regex = re.compile(regex, re.VERBOSE)
            if regex_name not in self._name2regex:
                self._name2regex[regex_name] = regex
        if isinstance(keys, str):
            keys = [keys]
        existing, missing = [], []
        for k in keys:
            (existing if k in self else missing).append(k)
        if len(missing) > 0:
            self.logger.warning(
                f"The keys {missing} are not among the Annotations objects, which are: {list(self)}"
            )
        if len(existing) == 0:
            return pd.DataFrame()
        labels_cfg = self.labels_cfg.copy()
        labels_cfg["decode"] = True
        checks = [
            check_labels(self[k].get_labels(**labels_cfg), regex=regex, **kwargs)
            for k in existing
        ]
        if len(keys) > 1:
            return pd.concat(checks, keys=existing)
        else:
            return checks[0]

    def color_non_chord_tones(self, color_name: str = "red") -> Optional[pd.DataFrame]:
        """Iterates through the attached labels, tries to interpret them as DCML harmony labels,
        colors the notes in the parsed score that are not expressed by the respective label for a score segment,
        and stores a report under :py:attr:`review_report`.

        Args:
          color_name:
              Name the color that the non-chord tones should get, defaults to 'red'. Name can be a CSS color or
              a MuseScore color (see :py:attr:`utils.MS3_COLORS`).

        Returns:
          A coloring report which is the original ``df`` with the appended columns 'n_colored', 'n_untouched',
          'count_ratio', 'dur_colored', 'dur_untouched', 'dur_ratio'. They contain the counts and durations of the
          colored vs. untouched notes as well the ratio of each pair. Note that the report does not take into account
          notes that reach into a segment, nor does it correct the duration of notes that reach into the subsequent
          segment.
        """
        if not self.mscx.has_annotations:
            self.mscx.logger.debug("Score contains no harmony labels.")
            return
        expanded = self.annotations.expand_dcml(drop_others=False, absolute=True)
        if expanded is None:
            self.mscx.logger.debug("Score contains no DCML harmony labels.")
            return
        self.review_report = self.mscx.color_non_chord_tones(
            expanded[expanded.chord.notna()], color_name=color_name
        )
        return self.review_report

    def compare_labels(
        self,
        key: str = "detached",
        new_color: str = "ms3_darkgreen",
        old_color: str = "ms3_darkred",
        detached_is_newer: bool = False,
        add_to_rna: bool = True,
        metadata_update: Optional[dict] = None,
        force_metadata_update: bool = False,
    ) -> Tuple[int, int]:
        """Compare detached labels ``key`` to the ones attached to the Score to create a diff.
        By default, the attached labels are considered as the reviewed version and labels that have changed or been
        added in comparison to the detached labels are colored in green; whereas the previous versions of changed
        labels are attached to the Score in red, just like any deleted label.

        Args:
           key: Key of the detached labels you want to compare to the ones in the score.
           new_color, old_color:
             The colors by which new and old labels are differentiated. Identical labels remain unchanged. Colors can
             be CSS colors or MuseScore colors (see :py:attr:`utils.MS3_COLORS`).
           detached_is_newer:
             Pass True if the detached labels are to be added with ``new_color`` whereas the attached changed labels
             will turn ``old_color``, as opposed to the default.
           add_to_rna:
             By default, new labels are attached to the Roman Numeral layer.
             Pass False to attach them to the chord layer instead.
           metadata_update:
             Dictionary containing metadata that is to be included in the comparison score. Notably, ms3 uses the key
             'compared_against' when the comparison is performed against a given git_revision.
           force_metadata_update:
             By default, the metadata is only updated if the comparison yields at least one difference to avoid
             outputting comparison scores not displaying any changes. Pass True to force the metadata update, which
             results in the properts :attr:`changed` being set to True.


        Returns:
          Number of attached labels that were not present in the old version and whose color has been changed.
          Number of added labels that are not present in the current version any more and which have been added as a
          consequence.
        """
        assert key != "annotations", "Pass a key of detached labels, not 'annotations'."
        if not self.mscx.has_annotations:
            self.logger.info("This score has no annotations attached.")
            return (0, 0)
        if key not in self._detached_annotations:
            self.logger.info(
                f"Key '{key}' doesn't correspond to a detached set of annotations. Use one of the existing keys or "
                f"load a new set with the method load_annotations().\nExisting keys: "
                f"{list(self._detached_annotations.keys())}"
            )
            return (0, 0)

        old_obj = self._detached_annotations[key]
        new_obj = self.mscx._annotations
        compare_cols = ["mc", "mc_onset", "staff", "voice", "label"]
        old_cols = [old_obj.cols[c] for c in compare_cols]
        new_cols = [new_obj.cols[c] for c in compare_cols]
        old = decode_harmonies(
            old_obj.df, label_col=old_obj.cols["label"], logger=self.logger
        )
        new = decode_harmonies(
            new_obj.df, label_col=new_obj.cols["label"], logger=self.logger
        )
        assert all(
            c in old.columns for c in old_cols
        ), f"DataFrame needs to have columns {old_cols} but has only {old.columns}"
        assert all(
            c in new.columns for c in new_cols
        ), f"DataFrame needs to have columns {new_cols} but has only {new.columns}"
        old_vals = set(old[old_cols].itertuples(index=False, name=None))
        new_vals = set(new[new_cols].itertuples(index=False, name=None))
        unchanged = old_vals.intersection(new_vals)
        changes_old = old_vals - unchanged
        changes_new = new_vals - unchanged
        if len(changes_new) == 0 and len(changes_old) == 0:
            self.mscx.logger.info("Comparison yielded no changes.")
            if force_metadata_update and metadata_update:
                self.logger.debug(
                    f"Updating the metadata of the compared score with {metadata_update!r}, even though no differences "
                    f"have been detected."
                )
                self.mscx.update_metadata(**metadata_update)
            return (0, 0)

        new_rgba = color2rgba(new_color)
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

        color_changes = sum(
            self.mscx.change_label_color(*t, **change_to_params) for t in changes_new
        )
        if len(changes_old) == 0:
            added_changes = 0
            old_df = None
        else:
            old_df = pd.DataFrame(changes_old, columns=compare_cols)
            for k, v in added_color_params.items():
                old_df[k] = v
            if add_to_rna:
                old_df["harmony_layer"] = 1
                anno = Annotations(df=old_df, **self.logger_cfg)
                anno.remove_initial_dots()
            else:
                old_df["harmony_layer"] = 0
                anno = Annotations(df=old_df, **self.logger_cfg)
                anno.add_initial_dots()
            added_changes = self.mscx.add_labels(anno)
        if added_changes > 0 or color_changes > 0:
            self.mscx.changed = True
            self.mscx.parsed.parse_measures()
            self.mscx._update_annotations()
            self.mscx.logger.info(
                f"{color_changes} attached labels changed to {change_to}, {added_changes} labels added in "
                f"{added_color}."
            )
        if metadata_update:
            self.logger.debug(
                f"Updating the metadata of the compared score with {metadata_update!r}."
            )
            self.mscx.update_metadata(**metadata_update)
        res = (color_changes, added_changes)
        new_df = pd.DataFrame(changes_new, columns=compare_cols)
        if old_df is None:
            self.comparison_report = pd.concat([new_df], keys=["new"])
        else:
            self.comparison_report = pd.concat([old_df, new_df], keys=["old", "new"])
        return res

    def detach_labels(
        self,
        key,
        staff=None,
        voice=None,
        harmony_layer=None,
        delete=True,
        inverse=False,
        regex=None,
    ):
        """Detach all annotations labels from this score's :obj:`MSCX` object or just a selection of them, without
        taking labels_cfg into account (don't decode the labels).
        The extracted labels are stored as a new :py:class:`~.annotations.Annotations` object that is accessible via
        ``Score.{key}``. By default, ``delete`` is set to True, meaning that if you call :py:meth:`store_scores`
        afterwards, the created MuseScore file will not contain the detached labels.

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
        harmony_layer : :obj:`int` or :obj:`str`, optional
            Select one of the harmony layers {0,1,2,3} to select only these.
        delete : :obj:`bool`, optional
            By default, the labels are removed from the XML structure in :obj:`MSCX`.
            Pass False if you want them to remain. This could be useful if you only want to extract a subset
            of the annotations for storing them separately but without removing the labels from the score.
        """
        if not self.mscx.has_annotations:
            self.mscx.logger.info("No annotations present in score.")
            return
        assert key not in dir(self) + [
            "annotations"
        ], f"The key {key} is reserved, please choose a different one."
        if not key.isidentifier():
            self.logger.warning(
                f"'{key}' cannot be used as an identifier. The extracted labels need to be accessed via "
                f"self._detached_annotations['{key}']"
            )
        df = self.annotations.get_labels(
            staff=staff,
            voice=voice,
            harmony_layer=harmony_layer,
            positioning=True,
            decode=False,
            drop=delete,
            inverse=inverse,
            regex=regex,
        )
        if len(df) == 0:
            self.mscx.logger.info(
                f"No labels found for staff {staff}, voice {voice}, harmony_layer {harmony_layer}."
            )
            return
        logger_cfg = self.logger_cfg.copy()
        logger_cfg["name"] = f"{self.logger.name}.{key}"
        self._detached_annotations[key] = Annotations(
            df=df, infer_types=self.get_infer_regex(), mscx_obj=self.mscx, **logger_cfg
        )
        if delete:
            self.mscx.delete_labels(df)
        return

    def get_infer_regex(self):
        """
        Returns
        -------
        :obj:`dict`
            Mapping of label types to the corresponding regular expressions
            in the order in which they are currently set to be inferred.
        """
        return {t: self._name2regex[t] for t in self._types_to_infer}

    def get_labels(
        self,
        key: Optional[str] = None,
        interval_index: bool = False,
        unfold: bool = False,
    ) -> Optional[pd.DataFrame]:
        """DataFrame representing all :ref:`labels`, i.e., all <Harmony> tags, of the score or another set of
        annotations. Corresponds to calling :meth:`~.annotations.Annotations.get_labels` on the selected object (by
        default, the one representing labels attached to the score) with the current :attr:`._labels_cfg`.
        Comes with the columns |quarterbeats|, |duration_qb|, |mc|, |mn|, |mc_onset|, |mn_onset|, |timesig|, |staff|,
        |voice|, |volta|, |harmony_layer|, |label|,  |offset_x|, |offset_y|, |regex_match|


        Args:
          key:
          interval_index: Pass True to replace the default :obj:`~pandas.RangeIndex` by an :obj:`~pandas.IntervalIndex`.

        Returns:
          DataFrame representing all :ref:`labels`, i.e., all <Harmony> tags in the score.
        """
        detached_annotations = list(self._detached_annotations.keys())
        if key is None:
            if self.mscx._annotations is None:
                msg = "The score does not contain any annotations."
                if len(detached_annotations) > 0:
                    msg += f" Available set of labels: {detached_annotations}"
                self.logger.info(msg)
                return None
            else:
                labels = self.mscx._annotations.get_labels(**self.labels_cfg)
        else:
            assert (
                key in detached_annotations
            ), f"No annotations available for key '{key}': {detached_annotations}"
            labels = self._detached_annotations[key].get_labels(**self.labels_cfg)
        if unfold:
            labels = self.mscx.parsed.unfold_facet_df(labels, "harmony labels")
            if labels is None:
                return
        if "quarterbeats" not in labels.columns:
            if self.mscx is None:
                self.logger.warning(
                    f"Could not add quarterbeats to the detached labels with key '{key}' because no score has been "
                    f"parsed yet."
                )
            else:
                labels = add_quarterbeats_col(
                    labels,
                    offset_dict=self.mscx.offset_dict(unfold=unfold),
                    offset_dict_all_endings=self.mscx.offset_dict(all_endings=True),
                    interval_index=interval_index,
                    logger=self.logger,
                )
        return labels

    def move_labels_to_layer(
        self,
        staff: Optional[int] = None,
        voice: Optional[Literal[1, 2, 3, 4]] = None,
        harmony_layer: Optional[Literal[0, 1, 2, 3]] = None,
        above: bool = False,
        safe: bool = True,
    ) -> bool:
        if not self.mscx.has_annotations:
            self.logger.debug("File has no labels to update.")
            return False
        before = self.annotations
        labels_before = self.annotations.df
        if len(labels_before) == 0:
            self.logger.info("Annotation object does not contain any labels.")
            return False
        if staff is not None:
            staff = int(staff)
            if staff < 1:
                staff = self.mscx.staff_ids[staff]
        need_moving = before.get_labels(
            staff=staff,
            voice=voice,
            harmony_layer=harmony_layer,
            regex=r"^[^\.]",
            inverse=True,
        )
        labels_need_moving = len(need_moving) > 0
        if self.mscx.style["romanNumeralPlacement"] is None:
            if above:
                self.mscx.style["romanNumeralPlacement"] = 0
                self.mscx.changed = True
        else:
            above_target = 0 if above else 1
            if int(self.mscx.style["romanNumeralPlacement"]) != above_target:
                self.mscx.style["romanNumeralPlacement"] = above_target
                self.mscx.changed = True
        if labels_need_moving:
            self.logger.info(
                f"Moving {len(need_moving)} labels to staff={staff}, voice={voice}, harmony_layer={harmony_layer}, "
                f"above={above}."
            )
            self.detach_labels(
                "old",
                staff=staff,
                voice=voice,
                harmony_layer=harmony_layer,
                regex=r"^[^\.]",
                inverse=True,
            )
            self.old.remove_initial_dots()
            self.attach_labels(
                "old", staff=int(staff), voice=voice, harmony_layer=int(harmony_layer)
            )
            if safe:
                labels_after = self.annotations.df
                try:
                    assert_dfs_equal(
                        labels_before,
                        labels_after,
                        exclude=["staff", "voice", "label", "harmony_layer"],
                    )
                except AssertionError as e:
                    self.logger.error(
                        f"Labels were not moved because of the following error:\n{e}"
                    )
                    return False
        else:
            self.logger.info("No labels needed moving.")
        return self.mscx.changed

    def new_type(self, name, regex, description="", infer=True):
        """Declare a custom label type. A type consists of a name, a regular expression and,
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
            By default, the labels of all :py:class:`~.annotations.Annotations` objects are matched against the new
            type. Pass False to not change any label's type.
        """
        # TODO: Registering new regEx type requires unittesting
        assert (
            name not in self._regex_name_description
        ), f"'{name}' already added to types: {self._regex_name_description[name]}"
        self._regex_name_description[name] = description
        self._name2regex[name] = regex
        if infer:
            self._types_to_infer.insert(0, name)
            for key in self:
                self[key].infer_types(self.get_infer_regex())

    def load_annotations(
        self,
        tsv_path: Optional[str] = None,
        anno_obj: Optional[Annotations] = None,
        df: Optional[pd.DataFrame] = None,
        key: str = "detached",
        infer: bool = True,
        **cols,
    ) -> None:
        """Attach an :py:class:`~.annotations.Annotations` object to the score and make it available as ``Score.{key}``.
        It can be an existing object or one newly created from the TSV file ``tsv_path``.

        Args:
          tsv_path:
              If you want to create a new :py:class:`~.annotations.Annotations` object from a TSV file, pass its path.
          anno_obj: Instead, you can pass an existing object.
          df: Or you can automatically create one from a given DataFrame.
          key:
              Specify a new key for accessing the set of annotations. The string needs to be usable
              as an identifier, e.g. not start with a number, not contain special characters etc. In return you
              may use it as a property: For example, passing ``'chords'`` lets you access the
              :py:class:`~.annotations.Annotations` as ``Score.chords``. The key 'annotations' is reserved for all
              annotations attached to the score.
          infer:
              By default, the label types are inferred in the currently configured order (see :py:attr:`name2regex`).
              Pass False to not add and not change any label types.
          **cols:
              If the columns in the specified TSV file diverge from the :ref:`standard column names<column_names>`,
              pass them as standard_name='custom name' keywords.
        """
        assert (
            key != "annotations"
        ), "The key 'annotations' is reserved, please choose a different one."
        assert key is not None, "Key cannot be None."
        assert key.isidentifier(), f"Key '{key}' is not a valid identifier."
        assert (
            sum(arg is not None for arg in (tsv_path, anno_obj, df)) == 1
        ), "Pass either tsv_path or anno_obj or df."
        inf_dict = self.get_infer_regex() if infer else {}
        mscx = None if self._mscx is None else self._mscx
        if tsv_path is not None:
            key = self._handle_path(tsv_path, key)
            logger_cfg = self.logger_cfg.copy()
            logger_cfg["name"] = f"{self.logger_names[key]}"
            anno_obj = Annotations(
                tsv_path=tsv_path,
                infer_types=inf_dict,
                cols=cols,
                mscx_obj=mscx,
                **logger_cfg,
            )
        elif df is not None:
            anno_obj = Annotations(
                df=df, infer_types=inf_dict, cols=cols, mscx_obj=mscx, **self.logger_cfg
            )
        else:
            anno_obj.mscx_obj = mscx
        self._detached_annotations[key] = anno_obj

    def store_annotations(self, key="annotations", tsv_path=None, **kwargs):
        """Save a set of annotations as TSV file. While ``store_list`` stores attached labels only, this method
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
            if "mscx" in self.paths:
                path = self.paths["mscx"]
                fname = self.fnames["mscx"]
                tsv_path = os.path.join(path, fname + "_labels.tsv")
            else:
                self.logger.warning(
                    "No tsv_path has been specified and no MuseScore file has been parsed to infer one."
                )
                return
        if self[key].store_tsv(tsv_path, **kwargs):
            new_key = self._handle_path(tsv_path, key=key)
            if key != "annotations":
                self[key].update_logger_cfg({"name": self.logger_names[new_key]})

    def write_score_to_handler(self, file_handler):
        """Write the current :obj:`MSCX` object to a file handler.
        Just a shortcut for ``Score.mscx.write_score_to_handler()``.

        Args:
            file_handler: File handler to write to.
        """
        return self.mscx.write_score_to_handler(file_handler)

    def store_score(self, filepath):
        """Store the current :obj:`MSCX` object attached to this score as uncompressed MuseScore file.
        Just a shortcut for ``Score.mscx.store_scores()``.

        Args:
            filepath:
                Path of the newly created MuseScore file, including the file name ending on '.mscx'.
                Uncompressed files ('.mscz') are not supported.
        """
        return self.mscx.store_score(filepath)

    def _handle_path(self, path, key=None):
        """Puts the path into ``paths, files, fnames, fexts`` dicts with the given key.

        Parameters
        ----------
        path : :obj:`str`
            Full file path.
        key : :obj:`str`, optional
            The key chosen by the user. By default, the key is automatically assigend to be the file's extension.
        """
        full_path = resolve_dir(path)
        if os.path.isfile(full_path):
            file_path, file = os.path.split(full_path)
            file_name, file_ext = os.path.splitext(file)
            if key is None:
                key = file_ext[1:]
            elif key == "file":
                key = file
            self.full_paths[key] = full_path
            self.paths[key] = file_path
            self.files[key] = file
            self.fnames[key] = file_name
            self.fexts[key] = file_ext
            logger_name = self.logger.name
            if logger_name == "ms3.Score":
                logger_name += "." + file_name.replace(".", "") + file_ext
            self.logger_names[key] = logger_name  # logger
            return key
        else:
            raise ValueError(f"Path not found: {path}.")
            # self.logger.error("No file found at this path: " + full_path)
            return None

    @staticmethod
    def make_extension_regex(native=True, convertible=True, tsv=False):
        assert sum((native, convertible)) > 0, "Select at least one type of extensions."
        exts = []
        if native:
            exts.extend(Score.native_formats)
        if convertible:
            exts.extend(Score.convertible_formats)
        if tsv:
            exts.append("tsv")
        dot = r"\."
        regex = f"({'|'.join(dot + e for e in exts)})$"
        return re.compile(regex, re.IGNORECASE)

    def parse_mscx(
        self, musescore_file=None, read_only=None, parser=None, labels_cfg={}
    ):
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
        musescore_file : :obj:`str`, optional
            Path to the MuseScore file to be parsed.
        read_only : :obj:`bool`, optional
            Defaults to ``False``, meaning that the parsing is slower and uses more memory in order to allow for
            manipulations of the score, such as adding and deleting labels. Set to ``True`` if you're only extracting
            information.
        parser : 'bs4', optional
            The only XML parser currently implemented is BeautifulSoup 4.
        labels_cfg : :obj:`dict`, optional
            Store a configuration dictionary to determine the output format of the :py:class:`~.annotations.Annotations`
            object representing the currently attached annotations. See :py:attr:`.MSCX.labels_cfg`.
        """
        if musescore_file is not None:
            assert os.path.isfile(
                musescore_file
            ), f"File does not exist: {musescore_file}"
            self.musescore_file = musescore_file
        if read_only is not None:
            self.read_only = read_only
        if parser is not None:
            self.parser = parser
        if len(labels_cfg) > 0:
            self.labels_cfg.update(update_labels_cfg(labels_cfg, logger=self.logger))

        permitted_extensions = self.native_formats + self.convertible_formats
        _, ext = os.path.splitext(self.musescore_file)
        ext = ext[1:]
        if ext.lower() not in permitted_extensions:
            raise ValueError(
                f"The extension of a score should be one of {permitted_extensions} not {ext}."
            )
        if ext.lower() in self.convertible_formats and self.ms is None:
            raise ValueError(
                f"To parse a {ext} file, use 'ms3 convert' command or set the attribute 'ms' for temporal on-the-fly "
                f"conversion."
            )
        extension = self._handle_path(self.musescore_file)
        logger_cfg = dict(self.logger_cfg)
        logger_cfg["name"] = self.logger_names[extension]
        musescore_file = resolve_dir(self.musescore_file)
        if extension in self.convertible_formats + ("mscz",):
            ctxt_mgr = unpack_mscz if extension == "mscz" else self._tmp_convert
            with ctxt_mgr(musescore_file) as tmp_mscx:
                self.logger.debug(
                    f"Using temporary file {os.path.basename(tmp_mscx)} in order to parse {musescore_file}."
                )
                self._mscx = MSCX(
                    tmp_mscx,
                    read_only=self.read_only,
                    parser=self.parser,
                    labels_cfg=self.labels_cfg,
                    parent_score=self,
                    **logger_cfg,
                )
                self.mscx.mscx_src = musescore_file
        else:
            self._mscx = MSCX(
                musescore_file,
                read_only=self.read_only,
                parser=self.parser,
                labels_cfg=self.labels_cfg,
                parent_score=self,
                **logger_cfg,
            )
        if self.mscx.has_annotations:
            self.mscx._annotations.infer_types(self.get_infer_regex())

    @contextmanager
    def _tmp_convert(self, file, dir=None):
        if dir is None:
            dir = os.path.dirname(file)
        try:
            tmp_file = Temp(suffix=".mscx", prefix=".", dir=dir, delete=False)
            convert(file, tmp_file.name, self.ms, logger=self.logger)
            yield tmp_file.name
        except Exception:
            self.logger.error(
                f"Error while dealing with the temporarily converted {os.path.basename(file)}"
            )
            raise
        finally:
            os.remove(tmp_file.name)

    def __repr__(self):
        if len(self.full_paths) == 0:
            if self.musescore_file is None:
                return "Empty Score object."
            else:
                return f"Empty Score object ready to parse {self.musescore_file}"
        msg = ""
        if any(ext in self.full_paths for ext in ("mscx", "mscz")):
            if "mscx" in self.full_paths:
                path = self.full_paths["mscx"]
                msg = "Uncompressed MuseScore file"
            else:
                path = self.full_paths["mscz"]
                msg = "ZIP compressed MuseScore file"
            if self._mscx.changed:
                msg += " (CHANGED!!!)"
        else:
            frst = list(self.full_paths.keys())[0]
            path = self.full_paths[frst]
            msg = f"Temporarily converted {frst.upper()} file"
        n_chars = len(msg)
        if self._mscx.changed:
            msg += "\n" + (n_chars - 12) * "-" + 12 * "!"
        else:
            msg += "\n" + n_chars * "-"
        msg += f"\n\n{path}\n\n"
        if self.mscx.has_annotations:
            msg += (
                f"Attached annotations\n--------------------\n\n{self.annotations}\n\n"
            )
        else:
            msg += "No annotations attached.\n\n"
        if self.has_detached_annotations:
            msg += "Detached annotations\n--------------------\n\n"
            for key, obj in self._detached_annotations.items():
                key_info = (
                    key + f" (stored as {self.files[key]})"
                    if key in self.files
                    else key
                )
                msg += f"{key_info} -> {obj}\n\n"
        if self.mscx.n_form_labels > 0:
            msg += f"Score contains {self.mscx.n_form_labels} form labels."
        return msg

    def __getattr__(self, item):
        if item == "annotations":
            return self.mscx._annotations
        try:
            return self._detached_annotations[item]
        except Exception:
            raise AttributeError(item)

    def __getitem__(self, item):
        if item == "annotations":
            return self.mscx._annotations
        try:
            return self._detached_annotations[item]
        except Exception:
            raise AttributeError(item)

    def __getstate__(self):
        """Loggers pose problems when pickling: Remove the reference."""
        log_capture_handler = get_log_capture_handler(self.logger)
        if log_capture_handler is not None:
            self.captured_logs = log_capture_handler.log_queue
        self.logger = None
        return self.__dict__

    def __iter__(self):
        """Iterate keys of Annotation objects."""
        attached = (
            ["annotations"]
            if self._mscx is not None and self._mscx.has_annotations
            else []
        )
        yield from attached + list(self._detached_annotations.keys())

    # def __setattr__(self, key, value):
    #     assert key != 'annotations', "The key 'annotations' is managed automatically, please pick a different one."
    #     assert key.isidentifier(), "Please use an alphanumeric key without special characters."
    #     if key in self.__dict__:
    #         self.__dict__[key] = value
    #     else:
    #         self._annotations[key] = value

    def output_mscx(*args, **kwargs) -> None:
        """Deprecated method. Replaced by :meth:`store_score`."""
        raise DeprecationWarning(
            "Method not in use any more. Use Score.store_score() to write the score to an MSCX file."
        )


# ######################################################################################################################
# ######################################################################################################################
# ############################################## End of Score() ########################################################
# ######################################################################################################################
# ######################################################################################################################


def compare_two_score_objects(
    old_score: Score,
    new_score: Score,
    logger=None,
) -> None:
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    old_path = old_score.mscx.mscx_src
    new_path = new_score.mscx.mscx_src
    dataframe_pairs = {
        "events": (old_score.mscx.events(), new_score.mscx.events()),
        "measures": (old_score.mscx.measures(), new_score.mscx.measures()),
        "labels": (old_score.mscx.labels(), new_score.mscx.labels()),
    }
    for facet, (old_df, new_df) in dataframe_pairs.items():
        n_none = (old_df is None) + (new_df is None)
        if n_none == 2:
            continue
        if n_none == 1:
            logger.warning(f"{facet} BEFORE:\n{old_df}\n{facet} AFTER:\n{new_df}")
            continue
        try:
            assert_dfs_equal(old_df, new_df)
        except AssertionError as e:
            logger.warning(
                f"The {facet} extracted from '{old_path}' and from the updated '{new_path}' do not match:\n{e}"
            )
