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
.. |metronome_base| replace:: :ref:`metronome_base <metronome_base>`
.. |metronome_number| replace:: :ref:`metronome_number <metronome_number>`
.. |tempo_visible| replace:: :ref:`tempo_visible <tempo_visible>`
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

from __future__ import annotations

import difflib
import os
import re
import warnings
from collections import ChainMap, defaultdict  # for merging dictionaries
from copy import copy
from fractions import Fraction
from functools import cache
from itertools import zip_longest
from pprint import pformat
from typing import (
    IO,
    Collection,
    Dict,
    Hashable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    overload,
)

import bs4  # python -m pip install beautifulsoup4 lxml
import numpy as np
import pandas as pd
from bs4 import NavigableString
from ms3._version import __version__
from typing_extensions import Self

from .annotations import Annotations
from .bs4_measures import MeasureList
from .logger import LoggedClass, get_logger, temporarily_suppress_warnings
from .transformations import add_quarterbeats_col, make_note_name_and_octave_columns
from .utils import (
    adjacency_groups,
    color_params2rgba,
    column_order,
    decode_harmonies,
    fifths2name,
    make_continuous_offset_series,
    make_offset_dict_from_measures,
    make_playthrough2mc,
    make_playthrough_info,
    ordinal_suffix,
    replace_index_by_intervals,
    resolve_dir,
    rgb_tuple2format,
    rgba2attrs,
    sort_note_list,
    unfold_measures_table,
    unfold_repeats,
    write_score_to_handler,
)
from .utils.constants import DCML_DOUBLE_REGEX, FORM_DETECTION_REGEX

module_logger = get_logger(__name__)

NOTE_SYMBOL_MAP = {
    "metNoteHalfUp": "𝅗𝅥",
    "metNoteQuarterUp": "𝅘𝅥",
    "metNote8thUp": "𝅘𝅥𝅮",
    "metAugmentationDot": ".",
    "": "𝅝",
    "": "𝅗𝅥",
    "": "𝅘𝅥",
    "": "𝅘𝅥𝅮",
    "": "𝅘𝅥𝅯",
    "": "𝅘𝅥𝅰",
    "": "𝅘𝅥𝅱",
    "": "𝅘𝅥𝅲",
    "": ".",
}


class _MSCX_bs4(LoggedClass):
    """This sister class implements :py:class:`~.score.MSCX`'s methods for a score parsed with beautifulsoup4.

    Attributes
    ----------
    mscx_src : :obj:`str`
        Path to the uncompressed MuseScore 3 file (MSCX) to be parsed.

    """

    durations = {
        "measure": Fraction(1),
        "breve": Fraction(2),  # in theory, of course, they could have length 1.5
        "long": Fraction(4),  # and 3 as well and other values yet
        "whole": Fraction(1),
        "half": Fraction(1, 2),
        "quarter": Fraction(1, 4),
        "eighth": Fraction(1, 8),
        "16th": Fraction(1, 16),
        "32nd": Fraction(1, 32),
        "64th": Fraction(1, 64),
        "128th": Fraction(1, 128),
        "256th": Fraction(1, 256),
        "512th": Fraction(1, 512),
        "1024th": Fraction(1, 1024),
    }

    @classmethod
    def from_filepath(
        cls,
        mscx_src: str,
        read_only: bool = False,
        logger_cfg: Optional[dict] = None,
    ) -> Self:
        with open(mscx_src, "r", encoding="utf-8") as file:
            soup = bs4.BeautifulSoup(file.read(), "xml")
        created_object = cls(soup, read_only=read_only, logger_cfg=logger_cfg)
        created_object.filepath = mscx_src
        return created_object

    def __init__(
        self,
        soup: bs4.BeautifulSoup,
        read_only: bool = False,
        logger_cfg: Optional[dict] = None,
    ):
        """

        Args:
            soup: A beautifulsoup4 object representing the MSCX file.
            read_only:
                If set to True, all references to XML tags will be removed after parsing to allow the object to be
                pickled.
            logger_cfg:
                The following options are available:
                'name': LOGGER_NAME -> by default the logger name is based on the parsed file(s)
                'level': {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}
                'file': PATH_TO_LOGFILE to store all log messages under the given path.
        """
        super().__init__(subclass="_MSCX_bs4", logger_cfg=logger_cfg)
        self.filepath = None  # is set by :meth:`from_filepath`
        self.soup = soup
        self.metadata = None
        self._metatags = None
        self._measures, self._events, self._notes = (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
        self.read_only = read_only
        self.first_mc = 1
        self.measure_nodes = {}
        """{staff -> {MC -> tag} }"""

        self.tags = {}  # only used if not self.read_only
        """ Nested dictionary allowing to access the score's XML elements in a convenient and structured manner:

           {MC ->
             {staff ->
                {voice ->
                  {mc_onset ->
                     [{"name" -> str,
                       "duration" -> Fraction,
                       "tag" -> bs4.Tag
                       },
                       ...
                     ]
                  }
                }
             }
           }
        """

        self.has_annotations = False
        self.n_form_labels = 0
        self._ml = None
        cols = [
            "mc",
            "mc_onset",
            "duration",
            "staff",
            "voice",
            "scalar",
            "nominal_duration",
        ]
        self._nl, self._cl, self._rl, self._nrl, self._fl = (
            pd.DataFrame(),
            pd.DataFrame(columns=cols),
            pd.DataFrame(columns=cols),
            pd.DataFrame(columns=cols),
            pd.DataFrame(columns=cols),
        )
        self._instrumentation: Instrumentation = None
        self._prelims: Prelims = None
        self._style: Style = None
        self.staff2drum_map: Dict[int, pd.DataFrame] = {}
        """For each stuff that is to be treated as drumset score, keep a mapping from MIDI pitch (DataFrame index) to
        note and instrument features.
        The columns typically include ['head', 'line', 'voice', 'name', 'stem', 'shortcut'].
        When creating note tables, the 'name' column will be populated with the names here rather than note names.
        """
        self.parse_soup()
        self.parse_measures()
        self.perform_checks()

    @property
    @cache
    def has_voltas(self) -> bool:
        """
        Return True if the score includes first and second endings. Otherwise, no 'volta' columns will be added to
        facets."""
        measures = self.ml()
        return measures.volta.notna().any()

    @property
    def instrumentation(self):
        if self._instrumentation is None:
            if self.soup is None:
                self.make_writeable()
            self._instrumentation = Instrumentation(self.soup, name=self.logger.name)
        return self._instrumentation

    @property
    def metatags(self):
        if self._metatags is None:
            if self.soup is None:
                self.make_writeable()
            self._metatags = Metatags(self.soup)
        return self._metatags

    @property
    def prelims(self):
        if self._prelims is None:
            if self.soup is None:
                self.make_writeable()
            self._prelims = Prelims(self.soup, name=self.logger.name)
        return self._prelims

    @property
    def staff_ids(self):
        return list(self.measure_nodes.keys())

    @property
    def style(self):
        if self._style is None:
            if self.soup is None:
                self.make_writeable()
            self._style = Style(self.soup)
        return self._style

    @property
    def version(self):
        return str(self.soup.find("programVersion").string)

    @property
    def volta_structure(self) -> Dict[int, Dict[int, List[int]]]:
        """{first_mc -> {volta_number -> [MC] } }"""
        if self._ml is not None:
            return self._ml.volta_structure

    def add_label(self, label, mc, mc_onset, staff=1, voice=1, **kwargs):
        """Adds a single label to the current XML in form of a new
        <Harmony> (and maybe also <location>) tag.

        Parameters
        ----------
        label
        mc
        mc_onset
        staff
        voice
        kwargs

        Returns
        -------

        """
        if pd.isnull(label) and len(kwargs) == 0:
            self.logger.error(f"Label cannot be '{label}'")
            return False
        assert (
            mc_onset >= 0
        ), f"Cannot attach label {label} to negative onset {mc_onset} at MC {mc}, staff {staff}, voice {voice}"
        self.make_writeable()
        if mc not in self.tags:
            self.logger.error(f"MC {mc} not found.")
            return False
        if staff not in self.measure_nodes:
            try:
                # maybe a negative integer?
                staff = list(self.measure_nodes.keys())[staff]
            except Exception:
                self.logger.error(f"Staff {staff} not found.")
                return False
        if voice not in [1, 2, 3, 4]:
            self.logger.error(f"Voice needs to be 1, 2, 3, or 4, not {voice}.")
            return False

        mc_onset = Fraction(mc_onset)
        label_name = kwargs["decoded"] if "decoded" in kwargs else label
        if voice not in self.tags[mc][staff] or len(self.tags[mc][staff][voice]) == 0:
            # Adding label to an unused voice that has to be created
            existing_voices = list(self.measure_nodes[staff][mc].find_all("voice"))
            n = len(existing_voices)
            if voice <= n:
                last = existing_voices[voice - 1]
            else:
                last = existing_voices[-1]
                while voice > n:
                    last = self.new_tag("voice", after=last)
                    n += 1
            remember = self.insert_label(
                label=label,
                loc_before=None if mc_onset == 0 else mc_onset,
                within=last,
                **kwargs,
            )
            self.tags[mc][staff][voice] = defaultdict(list)
            self.tags[mc][staff][voice][mc_onset] = remember
            self.logger.debug(
                f"Added {label_name} to empty {voice}{ordinal_suffix(voice)} voice in MC {mc} at mc_onset "
                f"{mc_onset}."
            )
            return True

        measure = self.tags[mc][staff][voice]
        if mc_onset in measure:
            # There is an event (chord or rest) with the same onset to attach the label to
            elements = measure[mc_onset]
            names = [e["name"] for e in elements]
            _, name = get_duration_event(elements)
            # insert before the first tag that is not in the tags_before_label list
            tags_before_label = [
                "BarLine",
                "Clef",  # MuseScore is inconsistent: If clef is present, the order is Clef-Harmony-Dynamic
                "Dynamic",  # but if not, it's Dynamic-Harmony
                "endTuplet",
                "FiguredBass",
                "KeySig",
                "location",
                "StaffText",
                "Tempo",
                "TimeSig",
            ]
            try:
                ix, before = next(
                    (i, element["tag"])
                    for i, element in enumerate(elements)
                    if element["name"] not in tags_before_label
                )
                remember = self.insert_label(label=label, before=before, **kwargs)
            except Exception:
                self.logger.debug(
                    f"""'{label}' is to be inserted at MC {mc}, onset {mc_onset}, staff {staff}, voice {voice},
where there is no Chord or Rest, just: {elements}."""
                )
                n_elements = len(elements)
                if "FiguredBass" in names:
                    ix, after = next(
                        (i, elements[i]["tag"])
                        for i in range(n_elements)
                        if elements[i]["name"] == "FiguredBass"
                    )
                else:
                    if n_elements > 1 and names[-1] == "location":
                        ix = n_elements - 1
                    else:
                        ix = n_elements
                    after = elements[ix - 1]["tag"]
                try:
                    remember = self.insert_label(label=label, after=after, **kwargs)
                except Exception as e:
                    self.logger.warning(
                        f"Inserting label '{label}' at mc {mc}, onset {mc_onset} failed with '{e}'"
                    )
                    return False
            measure[mc_onset].insert(ix, remember[0])
            old_names = list(names)
            names.insert(ix, "Harmony")
            if name is None:
                self.logger.debug(
                    f"MC {mc}, mc_onset {mc_onset}, staff {staff}, voice {voice} had only these tags (and no <Chord> "
                    f"or <Rest>): {old_names}\nAfter insertion: {names}"
                )
            else:
                self.logger.debug(
                    f"Added {label_name} to {name} in MC {mc}, mc_onset {mc_onset}, staff {staff}, voice {voice}."
                )
            if "Harmony" in old_names:
                self.logger.debug("There had already been a label.")
            return True

        # There is no event at the given onset to attach the label to
        ordered_onsets = list(reversed(sorted(measure)))
        try:
            prv_pos, nxt_pos = next(
                (prv, nxt)
                for prv, nxt in zip(ordered_onsets + [None], [None] + ordered_onsets)
                if prv < mc_onset
            )
        except Exception:
            self.logger.error(
                f"No event occurs before onset {mc_onset} at MC {mc}, staff {staff}, voice {voice}. All elements: "
                f"{ordered_onsets}"
            )
            raise
        prv = measure[prv_pos]
        nxt = None if nxt_pos is None else measure[nxt_pos]
        prv_names = [e["name"] for e in prv]
        prv_ix, prv_name = get_duration_event(prv)
        if nxt is not None:
            nxt_names = [e["name"] for e in nxt]
            _, nxt_name = get_duration_event(nxt)
        prv_name = ", ".join(f"<{e}>" for e in prv_names if e != "location")
        # distinguish six cases: prv can be [event, location], nxt can be [event, location, None]
        if prv_ix is not None:
            # prv is event (chord or rest)
            if nxt is None:
                loc_after = prv_pos + prv[prv_ix]["duration"] - mc_onset
                # i.e. the ending of the last event minus the onset
                remember = self.insert_label(
                    label=label,
                    loc_before=-loc_after,
                    after=prv[prv_ix]["tag"],
                    **kwargs,
                )
                self.logger.debug(
                    f"Added {label_name} at {loc_after} before the ending of MC {mc}'s last {prv_name}."
                )
            elif nxt_name is not None or nxt_names.count("location") == 0:
                # nxt is event (chord or rest) or something at onset 1 (after all sounding events, e.g. <Segment>)
                loc_after = nxt_pos - mc_onset
                remember = self.insert_label(
                    label=label,
                    loc_before=-loc_after,
                    loc_after=loc_after,
                    after=prv[prv_ix]["tag"],
                    **kwargs,
                )
                self.logger.debug(
                    f"MC {mc}: Added {label_name} at {loc_after} before the {nxt_name} at mc_onset {nxt_pos}."
                )
            else:
                # nxt is not a sounding event and has location tag(s)
                loc_ix = nxt_names.index("location")
                loc_dur = nxt[loc_ix]["duration"]
                assert loc_dur <= 0, (
                    f"Positive location tag at MC {mc}, mc_onset {nxt_pos} when trying to insert {label_name} at "
                    f"mc_onset {mc_onset}: {nxt}"
                )
                # nxt_name = ", ".join(f"<{e}>" for e in nxt_names if e != "location")
                # if nxt_pos + loc_dur == mc_onset:
                #     self.logger.info(f"nxt_pos: {nxt_pos}, loc_dur: {loc_dur}, mc_onset: {mc_onset}")
                #     # label to be positioned with the same location
                #     remember = self.insert_label(label=label, after=nxt[-1]['tag'], **kwargs)
                #     self.logger.debug(
                #         f"MC {mc}: Joined {label_name} with the {nxt_name} occuring at {loc_dur} "
                #         f"before the ending of the {prv_name} at mc_onset {prv_pos}.")
                # else:
                loc_before = loc_dur - nxt_pos + mc_onset
                remember = self.insert_label(
                    label=label,
                    loc_before=loc_before,
                    before=nxt[loc_ix]["tag"],
                    **kwargs,
                )
                loc_after = nxt_pos - mc_onset
                nxt[loc_ix]["tag"].fractions.string = str(loc_after)
                nxt[loc_ix]["duration"] = loc_after
                self.logger.debug(
                    f"MC {mc}: Added {label_name} at {-loc_before} before the ending of the {prv_name} at mc_onset "
                    f" {prv_pos} and {loc_after} before the subsequent\n{nxt}."
                )

        else:
            # prv has location tag(s)
            loc_before = mc_onset - prv_pos
            if nxt is None:
                remember = self.insert_label(
                    label=label, loc_before=loc_before, after=prv[-1]["tag"], **kwargs
                )
                self.logger.debug(
                    f"MC {mc}: Added {label_name} at {loc_before} after the previous {prv_name} at mc_onset {prv_pos}."
                )
            else:
                try:
                    loc_ix = next(
                        i
                        for i, name in zip(
                            range(len(prv_names) - 1, -1, -1), reversed(prv_names)
                        )
                        if name == "location"
                    )
                except Exception:
                    self.logger.error(
                        f"Trying to add {label_name} to MC {mc}, staff {staff}, voice {voice}, onset {mc_onset}: "
                        f"The tags of mc_onset {prv_pos} should include a <location> tag but don't:\n{prv}"
                    )
                    raise
                prv[loc_ix]["tag"].fractions.string = str(loc_before)
                prv[loc_ix]["duration"] = loc_before
                loc_after = nxt_pos - mc_onset
                remember = self.insert_label(
                    label=label, loc_after=loc_after, after=prv[loc_ix]["tag"], **kwargs
                )
                if nxt_name is None:
                    nxt_name = ", ".join(f"<{e}>" for e in nxt_names if e != "location")
                self.logger.debug(
                    f"""MC {mc}: Added {label_name} at {loc_before} after the previous {prv_name} at mc_onset {prv_pos}
and {loc_after} before the subsequent {nxt_name}."""
                )

        # if remember[0]['name'] == 'location':
        #     measure[prv_pos].append(remember[0])
        #     measure[mc_onset] = remember[1:]
        # else:
        measure[mc_onset] = remember
        return True

    def add_standard_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures that the DataFrame's first columns are ['mc', 'mn', ('volta'), 'timesig', 'mc_offset']"""
        ml_columns = ["mn", "timesig", "mc_offset"]
        if self.has_voltas:
            ml_columns.insert(1, "volta")
        add_cols = ["mc"] + [c for c in ml_columns if c not in df.columns]
        df = df.merge(self.ml()[add_cols], on="mc", how="left")
        df["mn_onset"] = df.mc_onset + df.mc_offset
        return df[[col for col in df.columns if not col == "mc_offset"]]

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
        """Change the color of an existing label.

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
        if label == "empty_harmony":
            self.logger.debug(
                "Empty harmony was skipped because the color wouldn't change anything."
            )
            return True
        params = [color_name, color_html, color_r, color_g, color_b, color_a]
        rgba = color_params2rgba(*params)
        if rgba is None:
            given_params = [p for p in params if p is not None]
            self.logger.warning(
                f"Parameters could not be turned into a RGBA color: {given_params}"
            )
            return False
        self.make_writeable()
        if mc not in self.tags:
            self.logger.error(f"MC {mc} not found.")
            return False
        if staff not in self.tags[mc]:
            self.logger.error(f"Staff {staff} not found.")
            return False
        if voice not in [1, 2, 3, 4]:
            self.logger.error(f"Voice needs to be 1, 2, 3, or 4, not {voice}.")
            return False
        if voice not in self.tags[mc][staff]:
            self.logger.error(f"Staff {staff}, MC {mc} has no voice {voice}.")
            return False
        measure = self.tags[mc][staff][voice]
        mc_onset = Fraction(mc_onset)
        if mc_onset not in measure:
            self.logger.error(
                f"Staff {staff}, MC {mc}, voice {voice} has no event on mc_onset {mc_onset}."
            )
            return False
        elements = measure[mc_onset]
        harmony_tags = [e["tag"] for e in elements if e["name"] == "Harmony"]
        n_labels = len(harmony_tags)
        if n_labels == 0:
            self.logger.error(
                f"Staff {staff}, MC {mc}, voice {voice}, mc_onset {mc_onset} has no labels."
            )
            return False
        labels = [decode_harmony_tag(t) for t in harmony_tags]
        try:
            ix = labels.index(label)
        except Exception:
            self.logger.error(
                f"Staff {staff}, MC {mc}, voice {voice}, mc_onset {mc_onset} has no label '{label}'."
            )
            return False
        tag = harmony_tags[ix]
        attrs = rgba2attrs(rgba)
        if tag.color is None:
            tag_order = [
                "absolute_base",
                "function",
                "name",
                "rootCase",
                "absolute_root",
            ]
            after = next(tag.find(t) for t in tag_order if tag.find(t) is not None)
            self.new_tag("color", attributes=attrs, after=after)
        else:
            for k, v in attrs.items():
                tag.color[k] = v
        return True

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
        |diminuendo_line|, |crescendo_line|, |crescendo_hairpin|, |tempo|, |qpm|, |metronome_base|, |metronome_number|,
        |tempo_visible|, |lyrics:1|, |Ottava:15mb|

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
        if mode == "strict":
            chords = self.cl()
        else:
            chords = self.get_chords(mode=mode)
        if unfold:
            chords = self.unfold_facet_df(chords, "chords")
            if chords is None:
                return
        chords = add_quarterbeats_col(
            chords,
            offset_dict=self.offset_dict(unfold=unfold),
            offset_dict_all_endings=self.offset_dict(all_endings=True),
            interval_index=interval_index,
            logger=self.logger,
        )
        return chords

    def cl(self, recompute: bool = False) -> pd.DataFrame:
        """Get the raw :ref:`chords` without adding quarterbeat columns."""
        if recompute or len(self._cl) == 0:
            self._cl = self.get_chords(mode="strict")
        return self._cl.copy()

    def color_notes(
        self,
        from_mc: int,
        from_mc_onset: Fraction,
        to_mc: Optional[int] = None,
        to_mc_onset: Optional[Fraction] = None,
        midi: List[int] = [],
        tpc: List[int] = [],
        inverse: bool = False,
        color_name: Optional[str] = None,
        color_html: Optional[str] = None,
        color_r: Optional[int] = None,
        color_g: Optional[int] = None,
        color_b: Optional[int] = None,
        color_a: Optional[int] = None,
    ) -> Tuple[List[Fraction], List[Fraction]]:
        """Colors all notes occurring in a particular score segment in one particular color, or
        only those (not) pertaining to a collection of MIDI pitches or Tonal Pitch Classes (TPC).

        Args:
          from_mc: MC in which the score segment starts.
          from_mc_onset: mc_onset where the score segment starts.
          to_mc: MC in which the score segment ends. If not specified, the segment ends at the end of the score.
          to_mc_onset: If ``to_mc`` is defined, the mc_onset where the score segment ends.
          midi: Collection of MIDI numbers to use as a filter or an inverse filter (depending on ``inverse``).
          tpc:
              Collection of Tonal Pitch Classes (C=0, G=1, F=-1 etc.) to use as a filter or an inverse filter
              (depending on ``inverse``).
          inverse:
              By default, only notes where all specified filters (midi and/or tpc) apply are colored.
              Set to True to color only those notes where none of the specified filters match.
          color_name:
              Specify the color either as a name, or as HTML color, or as RGB(A). Name can be a CSS color or
              a MuseScore color (see :py:attr:`utils.MS3_COLORS`).
          color_html:
              Specify the color either as a name, or as HTML color, or as RGB(A). An HTML color
              needs to be string of length 6.
          color_r: If you specify the color as RGB(A), you also need to specify color_g and color_b.
          color_g: If you specify the color as RGB(A), you also need to specify color_r and color_b.
          color_b: If you specify the color as RGB(A), you also need to specify color_r and color_g.
          color_a: If you have specified an RGB color, the alpha value defaults to 255 unless specified otherwise.

        Returns:
          List of durations (in fractions) of all notes that have been colored.
          List of durations (in fractions) of all notes that have not been colored.
        """
        if len(self.tags) == 0:
            if self.read_only:
                self.logger.error("Score is read_only.")
            else:
                self.logger.error("Score does not include any parsed tags.")
            return

        rgba = color_params2rgba(
            color_name, color_html, color_r, color_g, color_b, color_a
        )
        if rgba is None:
            self.logger.error("Pass a valid color value.")
            return
        if color_name is None:
            color_name = rgb_tuple2format(rgba[:3], format="name")
        color_attrs = rgba2attrs(rgba)

        str_midi = [str(m) for m in midi]
        # MuseScore's TPCs are shifted such that C = 14:
        ms_tpc = [str(t + 14) for t in tpc]

        until_end = pd.isnull(to_mc)
        negation = " not" if inverse else ""
        colored_durations, untouched_durations = [], []
        for mc, staves in self.tags.items():
            if mc < from_mc or (not until_end and mc > to_mc):
                continue
            for staff, voices in staves.items():
                for voice, onsets in voices.items():
                    for onset, tag_dicts in onsets.items():
                        if mc == from_mc and onset < from_mc_onset:
                            continue
                        if not until_end and mc == to_mc and onset >= to_mc_onset:
                            continue
                        for tag_dict in tag_dicts:
                            if tag_dict["name"] != "Chord":
                                continue
                            duration = tag_dict["duration"]
                            for note_tag in tag_dict["tag"].find_all("Note"):
                                reason = ""
                                if len(midi) > 0:
                                    midi_val = note_tag.pitch.string
                                    if inverse and midi_val in str_midi:
                                        untouched_durations.append(duration)
                                        continue
                                    if not inverse and midi_val not in str_midi:
                                        untouched_durations.append(duration)
                                        continue
                                    reason = (
                                        f"MIDI pitch {midi_val} is{negation} in {midi}"
                                    )
                                if len(ms_tpc) > 0:
                                    tpc_val = note_tag.tpc.string
                                    if inverse and tpc_val in ms_tpc:
                                        untouched_durations.append(duration)
                                        continue
                                    if not inverse and tpc_val not in ms_tpc:
                                        untouched_durations.append(duration)
                                        continue
                                    if reason != "":
                                        reason += " and "
                                    reason += (
                                        f"TPC {int(tpc_val) - 14} is{negation} in {tpc}"
                                    )
                                if reason == "":
                                    reason = " because no filters were specified."
                                else:
                                    reason = " because " + reason
                                first_inside = note_tag.find()
                                _ = self.new_tag(
                                    "color", attributes=color_attrs, before=first_inside
                                )
                                colored_durations.append(duration)
                                self.logger.debug(
                                    f"MC {mc}, onset {onset}, staff {staff}, voice {voice}: Changed note color to "
                                    f"{color_name}{reason}."
                                )
        return colored_durations, untouched_durations

    def delete_label(self, mc, staff, voice, mc_onset, empty_only=False):
        """Delete a label from a particular position (if there is one).

        Parameters
        ----------
        mc : :obj:`int`
            Measure count.
        staff, voice
            Notational layer in which to delete the label.
        mc_onset : :obj:`fractions.Fraction`
            mc_onset
        empty_only : :obj:`bool`, optional
            Set to True if you want to delete only empty harmonies. Since normally all labels at the defined position
            are deleted, this flag is needed to prevent deleting non-empty <Harmony> tags.

        Returns
        -------
        :obj:`bool`
            Whether a label was deleted or not.
        """
        self.make_writeable()
        measure = self.tags[mc][staff][voice]
        if mc_onset not in measure:
            self.logger.warning(
                f"Nothing to delete for MC {mc} mc_onset {mc_onset} in staff {staff}, voice {voice}."
            )
            return False
        elements = measure[mc_onset]
        element_names = [e["name"] for e in elements]
        if "Harmony" not in element_names:
            self.logger.warning(
                f"No harmony found at MC {mc}, mc_onset {mc_onset}, staff {staff}, voice {voice}."
            )
            return False
        if "Chord" in element_names and "location" in element_names:
            NotImplementedError(
                f"Check MC {mc}, mc_onset {mc_onset}, staff {staff}, voice {voice}:\n{elements}"
            )
        onsets = sorted(measure)
        ix = onsets.index(mc_onset)
        is_first = ix == 0
        is_last = ix == len(onsets) - 1
        # delete_locations = True

        _, name = get_duration_event(elements)
        if name is None:
            # this label is not attached to a chord or rest and depends on <location> tags, i.e. <location> tags on
            # previous and subsequent onsets might have to be adapted
            n_locs = element_names.count("location")
            if is_first:
                all_dur_ev = sum(
                    True
                    for os, tag_list in measure.items()
                    if get_duration_event(tag_list)[0] is not None
                )
                if all_dur_ev > 0:
                    assert (
                        n_locs > 0
                    ), f"""The label on MC {mc}, mc_onset {mc_onset}, staff {staff}, voice {voice} is the first onset
    in a measure with subsequent durational events but has no <location> tag"""
                prv_n_locs = 0
                # if not is_last:
                #     delete_locations = False
            else:
                prv_onset = onsets[ix - 1]
                prv_elements = measure[prv_onset]
                prv_names = [e["name"] for e in prv_elements]
                prv_n_locs = prv_names.count("location")

            if n_locs == 0:
                # The current onset has no <location> tag. This presumes that it is the last onset in the measure.
                if not is_last:
                    raise NotImplementedError(
                        f"The label on MC {mc}, mc_onset {mc_onset}, staff {staff}, voice {voice} is not on the last "
                        f"onset but has no <location> tag."
                    )
                if prv_n_locs > 0 and len(element_names) == 1:
                    # this harmony is the only event on the last onset, therefore the previous <location> tag can be
                    # deleted
                    if prv_names[-1] != "location":
                        raise NotImplementedError(
                            f"Location tag is not the last element in MC {mc}, mc_onset {onsets[ix - 1]}, staff "
                            f"{staff}, voice {voice}."
                        )
                    prv_elements[-1]["tag"].decompose()
                    del measure[prv_onset][-1]
                    if len(measure[prv_onset]) == 0:
                        del measure[prv_onset]
                    self.logger.debug(
                        f"""Removed <location> tag in MC {mc}, mc_onset {prv_onset}, staff {staff}, voice {voice}
    because it precedes the label to be deleted which is the voice's last onset, {mc_onset}."""
                    )

            elif n_locs == 1:
                if not is_last and not is_first:
                    # This presumes that the previous onset has at least one <location> tag which needs to be adapted
                    # assert prv_n_locs > 0, (f"The label on MC {mc}, mc_onset {mc_onset}, staff {staff}, voice {voice}"
                    #                         f"locs forward but the previous onset {prv_onset} has no <location> tag.")
                    # if prv_names[-1] != 'location':
                    #     raise NotImplementedError(f"Location tag is not the last element in MC {mc}, mc_onset "
                    #                               f"{prv_onset}, staff {staff}, voice {voice}.")
                    if prv_n_locs > 0:
                        cur_loc_dur = Fraction(
                            elements[element_names.index("location")]["duration"]
                        )
                        prv_loc_dur = Fraction(prv_elements[-1]["duration"])
                        prv_loc_tag = prv_elements[-1]["tag"]
                        new_loc_dur = prv_loc_dur + cur_loc_dur
                        prv_loc_tag.fractions.string = str(new_loc_dur)
                        measure[prv_onset][-1]["duration"] = new_loc_dur
                    else:
                        self.logger.debug(
                            f"""The label on MC {mc}, mc_onset {mc_onset}, staff {staff}, voice {voice} locs forward
    # but the previous onset {prv_onset} has no <location> tag:\n{prv_elements}"""
                        )
                # else: proceed with deletion

            elif n_locs == 2:
                # this onset has two <location> tags meaning that if the next onset has a <location> tag, too, a second
                # one needs to be added
                assert (
                    prv_n_locs == 0
                ), f"""The label on MC {mc}, mc_onset {mc_onset}, staff {staff}, voice {voice} has two
    <location> tags but the previous onset {prv_onset} has one, too."""
                if not is_last:
                    nxt_onset = onsets[ix + 1]
                    nxt_elements = measure[nxt_onset]
                    nxt_names = [e["name"] for e in nxt_elements]
                    nxt_n_locs = nxt_names.count("location")
                    _, nxt_name = get_duration_event(nxt_elements)
                    if nxt_name is None:
                        # The next onset is neither a chord nor a rest and therefore it needs to have exactly one
                        # location tag and a second one needs to be added based on the first one being deleted
                        nxt_is_last = ix + 1 == len(onsets) - 1
                        if not nxt_is_last:
                            assert (
                                nxt_n_locs == 1
                            ), f"""The label on MC {mc}, mc_onset {mc_onset}, staff {staff}, voice {voice} has two
    <location> tags but the next onset {nxt_onset} has
    {nxt_n_locs if nxt_n_locs > 1 else "none although it's neither a chord nor a rest, nor the last onset,"}."""
                            if nxt_names[-1] != "location":
                                raise NotImplementedError(
                                    f"Location tag is not the last element in MC {mc}, mc_onset {nxt_onset}, "
                                    f"staff {staff}, voice {voice}."
                                )
                        if element_names[-1] != "location":
                            raise NotImplementedError(
                                f"Location tag is not the last element in MC {mc}, mc_onset {mc_onset}, "
                                f"staff {staff}, voice {voice}."
                            )
                        neg_loc_dur = Fraction(
                            elements[element_names.index("location")]["duration"]
                        )
                        assert (
                            neg_loc_dur < 0
                        ), f"""Location tag in MC {mc}, mc_onset {nxt_onset}, staff {staff}, voice {voice}
    should be negative but is {neg_loc_dur}."""
                        pos_loc_dur = Fraction(elements[-1]["duration"])
                        new_loc_value = neg_loc_dur + pos_loc_dur
                        new_tag = self.new_location(new_loc_value)
                        nxt_elements[0]["tag"].insert_before(new_tag)
                        remember = {
                            "name": "location",
                            "duration": new_loc_value,
                            "tag": new_tag,
                        }
                        measure[nxt_onset].insert(0, remember)
                        self.logger.debug(
                            f"""Added a new negative <location> tag to the subsequent mc_onset {nxt_onset} in
    order to prepare the label deletion on MC {mc}, mc_onset {mc_onset}, staff {staff}, voice {voice}."""
                        )
                # else: proceed with deletions because it has no effect on a subsequent onset
            else:
                raise NotImplementedError(
                    f"Too many location tags in MC {mc}, mc_onset {prv_onset}, staff {staff}, voice {voice}."
                )
        # else: proceed with deletions because the <Harmony> is attached to a durational event (Rest or Chord)

        # Here the actual removal takes place.
        deletions = []
        delete_location = False
        if name is None and "location" in element_names:
            other_elements = sum(
                e not in ("Harmony", "location") for e in element_names
            )
            delete_location = is_last or (mc_onset > 0 and other_elements == 0)
        labels = [e for e in elements if e["name"] == "Harmony"]
        if empty_only:
            empty = [
                e
                for e in labels
                if e["tag"].find("name") is None or e["tag"].find("name").string is None
            ]
            if len(empty) == 0:
                self.logger.info(
                    f"No empty label to delete at MC {mc}, mc_onset {mc_onset}, staff {staff}, voice {voice}."
                )
            elif len(empty) < len(labels):
                # if there are additional non-empty labels, delete nothing but the empty ones
                elements = empty

        for i, e in enumerate(elements):
            if e["name"] == "Harmony" or (e["name"] == "location" and delete_location):
                e["tag"].decompose()
                deletions.append(i)
                self.logger.debug(
                    f"<{e['name']}>-tag deleted in MC {mc}, mc_onset {mc_onset}, staff {staff}, voice {voice}."
                )
        for i in reversed(deletions):
            del measure[mc_onset][i]
        if len(measure[mc_onset]) == 0:
            del measure[mc_onset]
        self.remove_empty_voices(mc, staff)
        return len(deletions) > 0

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
        events = column_order(self.add_standard_cols(self._events))
        if unfold:
            events = self.unfold_facet_df(events, "chords")
            if events is None:
                return
        events = add_quarterbeats_col(
            events,
            offset_dict=self.offset_dict(unfold=unfold),
            offset_dict_all_endings=self.offset_dict(all_endings=True),
            interval_index=interval_index,
            logger=self.logger,
        )
        return events

    def form_labels(
        self,
        detection_regex: str = None,
        exclude_harmony_layer: bool = False,
        interval_index: bool = False,
        unfold: bool = False,
    ) -> Optional[pd.DataFrame]:
        """DataFrame representing :ref:`form labels <form_labels>` (or other) that have been encoded as <StaffText>s
        rather than in the <Harmony> layer (see argument ``exclude_harmony_layer``).
        This function essentially filters all StaffTexts matching the ``detection_regex`` and adds the standard position
        columns.

        Args:
          detection_regex:
              By default, detects all labels starting with one or two digits followed by a column
              (see :const:`the regex <~.utils.FORM_DETECTION_REGEX>`). Pass another regex to retrieve only StaffTexts
              matching this one.
          exclude_harmony_layer:
              By default, form labels are detected even if they have been encoded as Harmony labels (rather than as
              StaffText).
              Pass True in order to retrieve only StaffText form labels.
          interval_index: Pass True to replace the default :obj:`~pandas.RangeIndex` by an :obj:`~pandas.IntervalIndex`.
          unfold: Pass True to retrieve a Dat

        Returns:
          DataFrame containing all StaffTexts matching the ``detection_regex``
        """
        form = self.fl(
            detection_regex=detection_regex, exclude_harmony_layer=exclude_harmony_layer
        )
        if form is None:
            return
        if unfold:
            form = self.unfold_facet_df(form, "chords")
            if form is None:
                return
        form = add_quarterbeats_col(
            form,
            offset_dict=self.offset_dict(unfold=unfold),
            offset_dict_all_endings=self.offset_dict(all_endings=True),
            interval_index=interval_index,
            logger=self.logger,
        )
        return form

    def fl(
        self, detection_regex: str = None, exclude_harmony_layer=False
    ) -> pd.DataFrame:
        """Get the raw :ref:`form_labels` (or other) that match the ``detection_regex``, but without adding quarterbeat
        columns.

        {ref}`$1`
            detection_regex:
                By default, detects all labels starting with one or two digits followed by a column
                (see :const:`the regex <~.utils.FORM_DETECTION_REGEX>`). Pass another regex to retrieve only StaffTexts
                matching this one.

        Returns:
            DataFrame containing all StaffTexts matching the ``detection_regex`` or None
        """
        stafftext_col = "StaffText/text"
        harmony_col = "Harmony/name"
        has_stafftext = stafftext_col in self._events.columns
        has_harmony_layer = (
            harmony_col in self._events.columns and not exclude_harmony_layer
        )
        if has_stafftext or has_harmony_layer:
            if detection_regex is None:
                detection_regex = FORM_DETECTION_REGEX
            form_label_column = pd.Series(
                pd.NA, index=self._events.index, dtype="string", name="form_label"
            )
            if has_stafftext:
                stafftext_selector = (
                    self._events[stafftext_col]
                    .str.contains(detection_regex)
                    .fillna(False)
                )
                if stafftext_selector.sum() > 0:
                    form_label_column.loc[stafftext_selector] = self._events.loc[
                        stafftext_selector, stafftext_col
                    ]
            if has_harmony_layer:
                harmony_selector = (
                    self._events[harmony_col]
                    .str.contains(detection_regex)
                    .fillna(False)
                )
                if harmony_selector.sum() > 0:
                    form_label_column.loc[harmony_selector] = self._events.loc[
                        harmony_selector, harmony_col
                    ]
            detected_form_labels = form_label_column.notna()
            if detected_form_labels.sum() == 0:
                self.logger.debug("No form labels found.")
                return
            events_with_form = pd.concat([self._events, form_label_column], axis=1)
            form_labels = events_with_form[detected_form_labels]
            cols = [
                "mc",
                "mn",
                "mc_onset",
                "mn_onset",
                "staff",
                "voice",
                "timesig",
                "form_label",
            ]
            if self.has_voltas:
                cols.insert(2, "volta")
            self._fl = self.add_standard_cols(form_labels)[cols].sort_values(
                ["mc", "mc_onset"]
            )
            return self._fl
        return

    def get_chords(
        self,
        staff: Optional[int] = None,
        voice: Optional[Literal[1, 2, 3, 4]] = None,
        mode: Literal["auto", "strict"] = "auto",
        lyrics: bool = False,
        dynamics: bool = False,
        articulation: bool = False,
        staff_text: bool = False,
        system_text: bool = False,
        tempo: bool = False,
        spanners: bool = False,
        thoroughbass: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Retrieve a customized chord lists, e.g. one including less of the processed features or additional,
        unprocessed ones.

        Args:
          staff: Get information from a particular staff only (1 = upper staff)
          voice: Get information from a particular voice only (1 = only the first layer of every staff)
          mode:
              | Defaults to 'auto', meaning that those aspects are automatically included that occur in the score;
                the resulting DataFrame has no empty columns except for those parameters that are set to True.
              | 'strict': Create columns for exactly those parameters that are set to True, regardless whether they
                occur in the score or not (in which case the column will be empty).
          lyrics: Include lyrics.
          dynamics: Include dynamic markings such as f or p.
          articulation: Include articulation such as arpeggios.
          staff_text: Include expression text such as 'dolce' and free-hand staff text such as 'div.'.
          system_text: Include system text such as movement titles.
          tempo: Include tempo markings.
          spanners: Include spanners such as slurs, 8va lines, pedal lines etc.
          thoroughbass: Include thoroughbass figures' levels and durations.
          **kwargs:

        Returns:
          DataFrame representing all <Chord> tags in the score with the selected features.
        """
        cols = {
            "nominal_duration": "Chord/durationType",
            "lyrics": "Chord/Lyrics/text",
            "articulation": "Chord/Articulation/subtype",
            "dynamics": "Dynamic/subtype",
            "system_text": "SystemText_text",
            "staff_text": "StaffText_text",
            "tremolo": "Chord/Tremolo/subtype",
        }
        main_cols = [
            "mc",
            "mn",
            "mc_onset",
            "mn_onset",
            "event",
            "timesig",
            "staff",
            "voice",
            "duration",
            "gracenote",
            "tremolo",
            "nominal_duration",
            "scalar",
            "chord_id",
        ]
        if self.has_voltas:
            main_cols.insert(2, "volta")
        selector = self._events.event == "Chord"
        aspects = [
            "lyrics",
            "dynamics",
            "articulation",
            "staff_text",
            "system_text",
            "tempo",
            "spanners",
            "thoroughbass",
        ]
        if mode == "all":
            params = {p: True for p in aspects}
        else:
            lcls = locals()
            params = {p: lcls[p] for p in aspects}
        # map parameter to values to select from the event table's 'event' column
        param2event = {
            "dynamics": "Dynamic",
            "spanners": "Spanner",
            "staff_text": "StaffText",
            "system_text": "SystemText",
            "tempo": "Tempo",
            "thoroughbass": "FiguredBass",
        }
        selectors = {
            param: self._events.event == event_name
            for param, event_name in param2event.items()
        }
        if mode == "auto":
            for param, boolean_mask in selectors.items():
                if not params[param] and boolean_mask.any():
                    params[param] = True
        for param, boolean_mask in selectors.items():
            if params[param]:
                selector |= boolean_mask
        if staff:
            selector &= self._events.staff == staff
        if voice:
            selector &= self._events.voice == voice
        df = self.add_standard_cols(self._events[selector])
        if "chord_id" in df.columns:
            df = df.astype({"chord_id": "Int64"})
        df.rename(
            columns={v: k for k, v in cols.items() if v in df.columns}, inplace=True
        )

        if mode == "auto":
            if "lyrics" in df.columns:
                params["lyrics"] = True
            if "articulation" in df.columns:
                params["articulation"] = True
            if any(c in df.columns for c in ("Spanner:type", "Chord/Spanner:type")):
                params["spanners"] = True
        if "nominal_duration" in df.columns:
            df.loc[:, "nominal_duration"] = df.nominal_duration.map(
                self.durations
            )  # replace string values by fractions
        new_cols = {}
        if params["lyrics"]:
            column_pattern = r"(lyrics_(\d+))"
            if df.columns.str.match(column_pattern).any():
                column_names: pd.DataFrame = df.columns.str.extract(column_pattern)
                column_names = column_names.dropna()
                column_names = column_names.sort_values(1)
                column_names = column_names[0].to_list()
                main_cols.extend(column_names)
            else:
                main_cols.append("lyrics_1")
        if params["dynamics"]:
            main_cols.append("dynamics")
        if params["articulation"]:
            main_cols.append("articulation")
        if params["staff_text"]:
            main_cols.append("staff_text")
        if params["system_text"]:
            main_cols.append("system_text")
        if params["tempo"]:
            main_cols.extend(
                [
                    "tempo",
                    "qpm",
                    "metronome_base",
                    "metronome_number",
                    "tempo_visible",
                ]
            )
        if params["thoroughbass"]:
            if "thoroughbass_level_1" in df.columns:
                tb_level_columns = [
                    col for col in df.columns if col.startswith("thoroughbass_level")
                ]
                if "thoroughbass_duration" in df.columns:
                    tb_columns = ["thoroughbass_duration"] + tb_level_columns
                else:
                    tb_columns = tb_level_columns
            else:
                tb_columns = ["thoroughbass_duration", "thoroughbass_level_1"]
            main_cols.extend(tb_columns)
        for col in main_cols:
            if (col not in df.columns) and (col not in new_cols):
                new_cols[col] = pd.Series(index=df.index, dtype="object")
        df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
        additional_cols = []
        if params["spanners"]:
            spanner_ids = make_spanner_cols(df, logger=self.logger)
            if len(spanner_ids.columns) > 0:
                additional_cols.extend(spanner_ids.columns.to_list())
                df = pd.concat([df, spanner_ids], axis=1)
        for feature in kwargs.keys():
            additional_cols.extend(
                [c for c in df.columns if feature in c and c not in main_cols]
            )
        result = df[main_cols + additional_cols]
        if mode == "auto":
            return result.dropna(axis=1, how="all")
        return result.copy()

    @cache
    def get_playthrough_mcs(self) -> Optional[pd.Series]:
        measures = self.ml()  # measures table without quarterbeats
        playthrough_mcs = make_playthrough2mc(measures, logger=self.logger)
        if len(playthrough_mcs) == 0:
            self.logger.warning(
                f"Error in the repeat structure: Did not reach the stopping value -1 in measures.next:\n"
                f"{measures.set_index('mc').next}"
            )
            playthrough_mcs = None
        else:
            self.logger.debug("Repeat structure successfully unfolded.")
        return playthrough_mcs

    def get_raw_labels(self):
        """Returns a list of <harmony> tags from the parsed score.

        Returns
        -------
        :obj:`pandas.DataFrame`

        """
        cols = {
            "harmony_layer": "Harmony/harmonyType",
            "label": "Harmony/name",
            "nashville": "Harmony/function",
            "absolute_root": "Harmony/root",
            "absolute_base": "Harmony/base",
            "leftParen": "Harmony/leftParen",
            "rightParen": "Harmony/rightParen",
            "offset_x": "Harmony/offset:x",
            "offset_y": "Harmony/offset:y",
            "color_r": "Harmony/color:r",
            "color_g": "Harmony/color:g",
            "color_b": "Harmony/color:b",
            "color_a": "Harmony/color:a",
        }
        std_cols = [
            "mc",
            "mn",
            "mc_onset",
            "mn_onset",
            "timesig",
            "staff",
            "voice",
            "label",
        ]
        main_cols = std_cols + Annotations.additional_cols
        sel = self._events.event == "Harmony"
        df = self.add_standard_cols(self._events[sel]).dropna(axis=1, how="all")
        if len(df.index) == 0:
            return pd.DataFrame(columns=std_cols)
        df.rename(
            columns={v: k for k, v in cols.items() if v in df.columns}, inplace=True
        )
        if "harmony_layer" in df.columns:
            df.harmony_layer = df.harmony_layer.fillna(0)
        columns = [c for c in main_cols if c in df.columns]
        additional_cols = {
            c: c[8:]
            for c in df.columns
            if c[:8] == "Harmony/" and c not in cols.values()
        }
        df.rename(columns=additional_cols, inplace=True)
        columns += list(additional_cols.values())
        return df[columns]

    def get_texts(self, only_header: bool = True) -> Dict[str, str]:
        """Process <Text> nodes (normally attached to <Staff id="1">)."""
        texts = defaultdict(set)
        tags = self.soup.find_all("Text")
        for t in tags:
            txt, style = tag2text(t)
            if style == "Title":
                style = "title_text"
            elif style == "Subtitle":
                style = "subtitle_text"
            elif style == "Composer":
                style = "composer_text"
            elif style == "Lyricist":
                style = "lyricist_text"
            elif style == "Instrument Name (Part)":
                style = "part_name_text"
            else:
                if only_header:
                    continue
                style = "text"
            texts[style].add(txt)
        return {st: "; ".join(txt) for st, txt in texts.items()}

    def _get_metadata(self):
        """


        Returns
        -------
        :obj:`dict`
        """
        assert self.soup is not None, (
            "The file's XML needs to be loaded. Get metadata from the 'metadata' property or use the method "
            "make_writeable()"
        )

        def nav_str2str(s):
            return "" if s is None else str(s)

        data = {
            tag["name"]: nav_str2str(tag.string)
            for tag in self.soup.find_all("metaTag")
        }
        data.update(self.get_texts())
        if "reviewer" in data:
            if "reviewers" in data:
                self.logger.warning(
                    "Score properties contain a superfluous key called 'reviewer'. "
                    "Please merge with the value for 'reviewers' and delete."
                )
            else:
                self.logger.info(
                    "The key 'reviewer' contained in the Score properties was automatically "
                    "renamed to 'reviewers' when extracting metadata."
                )
                data["reviewers"] = data["reviewer"]
                del data["reviewer"]
        if "annotator" in data:
            if "annotators" in data:
                self.logger.warning(
                    "Score properties contain a superfluous key called 'annotator'. "
                    "Please merge with the value for 'annotators' and delete."
                )
            else:
                self.logger.info(
                    "The key 'annotator' contained in the Score properties was automatically "
                    "renamed to 'annotators' when extracting metadata."
                )
                data["annotators"] = data["annotator"]
                del data["annotator"]
        for name, value in data.items():
            # check for columns with same name but different capitalization
            name_lwr = name.lower()
            if name == name_lwr:
                continue
            if name_lwr in data:
                self.logger.warning(
                    f"Metadata contain the fields {name} and {name_lwr}. Please merge."
                )
            elif name_lwr in ("harmony_version", "annotators", "reviewers"):
                data[name_lwr] = value
                del data[name]
                self.logger.warning(
                    f"Wrongly spelled metadata field {name} read as {name_lwr}."
                )
        # measures properties
        measures = self.measures()
        # time signatures
        ts_groups, _ = adjacency_groups(measures.timesig)
        mc_ts = measures.groupby(ts_groups)[["mc", "timesig"]].head(1)
        timesigs = dict(mc_ts.values)
        data["TimeSig"] = timesigs
        # key signatures
        ks_groups, _ = adjacency_groups(measures.keysig)
        mc_ks = measures.groupby(ks_groups)[["mc", "keysig"]].head(1)
        keysigs = {int(k): int(v) for k, v in mc_ks.values}
        data["KeySig"] = keysigs
        # last measure counts & numbers, total duration in quarters
        last_measure = measures.iloc[-1]
        data["last_mc"] = int(last_measure.mc)
        data["last_mn"] = int(last_measure.mn)
        data["length_qb"] = round(measures.duration_qb.sum(), 2)
        # the same unfolded
        unfolded_measures = self.measures(unfold=True)
        if unfolded_measures is None:
            for aspect in (
                "last_mc_unfolded",
                "last_mn_unfolded",
                "length_qb_unfolded",
            ):
                data[aspect] = None
        else:
            data["last_mc_unfolded"] = int(max(unfolded_measures.mc_playthrough))
            if "mn_playthrough" in unfolded_measures.columns:
                unfolded_mn = unfolded_measures.mn_playthrough.nunique()
                if measures.iloc[0].mn == 0:
                    unfolded_mn -= 1
                data["last_mn_unfolded"] = unfolded_mn
            else:
                data["last_mn_unfolded"] = None
            data["length_qb_unfolded"] = round(unfolded_measures.duration_qb.sum(), 2)
        if self.has_voltas:
            data["volta_mcs"] = list(
                list(list(mcs) for mcs in group.values())
                for group in self.volta_structure.values()
            )

        # labels
        all_labels = self.get_raw_labels()
        if len(all_labels) > 0:
            decoded_labels = decode_harmonies(
                all_labels, return_series=True, logger=self.logger
            )
            matches_dcml = decoded_labels[decoded_labels.notna()].str.match(
                DCML_DOUBLE_REGEX
            )
            n_dcml = int(matches_dcml.sum())
            data["guitar_chord_count"] = len(all_labels) - n_dcml
            data["label_count"] = n_dcml
        else:
            data["guitar_chord_count"] = 0
            data["label_count"] = 0
        data["form_label_count"] = self.n_form_labels
        annotated_key = None
        for harmony_tag in self.soup.find_all("Harmony"):
            label = harmony_tag.find("name")
            if label is not None and label.string is not None:
                m = re.match(r"^\.?([A-Ga-g](#+|b+)?)\.", label.string)
                if m is not None:
                    annotated_key = m.group(1)
                    break
        if annotated_key is not None:
            data["annotated_key"] = annotated_key

        data["musescore"] = self.version
        data["ms3_version"] = __version__

        # notes
        notes = self.nl()
        if len(notes.index) == 0:
            data["all_notes_qb"] = 0.0
            data["n_onsets"] = 0
            return data
        has_drumset = len(self.staff2drum_map) > 0
        data["has_drumset"] = has_drumset
        data["all_notes_qb"] = round((notes.duration * 4.0).sum(), 2)
        not_tied = ~notes.tied.isin((0, -1))
        data["n_onsets"] = int(sum(not_tied))
        data["n_onset_positions"] = (
            notes[not_tied].groupby(["mc", "mc_onset"]).size().shape[0]
        )
        staff_groups = notes.groupby("staff").midi
        ambitus = {}
        for staff, min_tpc, min_midi in notes.loc[
            staff_groups.idxmin(),
            [
                "staff",
                "tpc",
                "midi",
            ],
        ].itertuples(name=None, index=False):
            if staff in self.staff2drum_map:
                continue
            ambitus[staff] = {
                "min_midi": int(min_midi),
                "min_name": fifths2name(min_tpc, min_midi, logger=self.logger),
            }
        for staff, max_tpc, max_midi in notes.loc[
            staff_groups.idxmax(),
            [
                "staff",
                "tpc",
                "midi",
            ],
        ].itertuples(name=None, index=False):
            if staff in self.staff2drum_map:
                # no ambitus for drum parts
                continue
            ambitus[staff]["max_midi"] = int(max_midi)
            ambitus[staff]["max_name"] = fifths2name(
                max_tpc, max_midi, logger=self.logger
            )
        data["parts"] = {
            f"part_{i}": get_part_info(part)
            for i, part in enumerate(self.soup.find_all("Part"), 1)
        }
        # for including the metadata as one line in metadata.tsv the function utils.metadata2series() is used
        # which updates `data` with the items of all part dictionaries, removing they key 'parts' afterwards
        for part, part_dict in data["parts"].items():
            for id in part_dict["staves"]:
                part_dict[f"staff_{id}_ambitus"] = ambitus[id] if id in ambitus else {}
        ambitus_tuples = [
            tuple(amb_dict.values()) for amb_dict in ambitus.values() if amb_dict != {}
        ]
        if len(ambitus_tuples) == 0:
            self.logger.info(
                "The score does not seem to contain any pitched events. No indication of ambitus possible."
            )
            data["ambitus"] = {}
        else:
            # computing global ambitus
            mimi, mina, mami, mana = zip(*ambitus_tuples)
            min_midi, max_midi = min(mimi), max(mami)
            data["ambitus"] = {
                "min_midi": min_midi,
                "min_name": mina[mimi.index(min_midi)],
                "max_midi": max_midi,
                "max_name": mana[mami.index(max_midi)],
            }
        return data

    def get_instrumentation(self) -> Dict[str, str]:
        """Returns a {staff_<i>_instrument -> instrument_name} dict."""
        return {
            staff: instrument["trackName"]
            for staff, instrument in self.instrumentation.fields.items()
        }

    def infer_mc(self, mn, mn_onset=0, volta=None):
        """mn_onset and needs to be converted to mc_onset"""
        try:
            mn = int(mn)
        except Exception:
            # Check if MN has volta information, e.g. '16a' for first volta, or '16b' for second etc.
            m = re.match(r"^(\d+)([a-e])$", str(mn))
            if m is None:
                self.logger.error(f"MN {mn} is not a valid measure number.")
                raise
            mn = int(m.group(1))
            volta = ord(m.group(2)) - 96  # turn 'a' into 1, 'b' into 2 etc.
        try:
            mn_onset = Fraction(mn_onset)
        except Exception:
            self.logger.error(
                f"The mn_onset {mn_onset} could not be interpreted as a fraction."
            )
            raise
        measures = self.ml()
        candidates = measures[measures["mn"] == mn]
        if len(candidates) == 0:
            self.logger.error(
                f"MN {mn} does not occur in measure list, which ends at MN {measures['mn'].max()}."
            )
            return
        if len(candidates) == 1:
            mc = candidates.iloc[0].mc
            self.logger.debug(f"MN {mn} has unique match with MC {mc}.")
            return mc, mn_onset
        if candidates.volta.notna().any():
            if volta is None:
                mc = candidates.iloc[0].mc
                self.logger.warning(
                    f"MN {mn} is ambiguous because it is a measure with first and second endings, but volta has not "
                    f"been specified. The first ending MC {mc} is being used. Suppress this warning by using "
                    f"disambiguating endings such as '16a' for first or '16b' for second. "
                    f"{candidates[['mc', 'mn', 'mc_offset', 'volta']]}"
                )
                return mc, mn_onset
            candidates = candidates[candidates.volta == volta]
        if len(candidates) == 1:
            mc = candidates.iloc[0].mc
            self.logger.debug(f"MN {mn}, volta {volta} has unique match with MC {mc}.")
            return mc, mn_onset
        if len(candidates) == 0:
            self.logger.error("Volta selection failed")
            return None, None
        if mn_onset == 0:
            mc = candidates.iloc[0].mc
            return mc, mn_onset
        right_boundaries = candidates.act_dur + candidates.act_dur.shift().fillna(0)
        left_boundary = 0
        for i, right_boundary in enumerate(sorted(right_boundaries)):
            j = i
            if mn_onset < right_boundary:
                mc_onset = mn_onset - left_boundary
                break
            left_boundary = right_boundary
        mc = candidates.iloc[j].mc
        if left_boundary == right_boundary:
            self.logger.warning(
                f"The onset {mn_onset} is bigger than the last possible onset of MN {mn} which is {right_boundary}"
            )
        return mc, mc_onset

    def insert_label(
        self,
        label,
        loc_before=None,
        before=None,
        loc_after=None,
        after=None,
        within=None,
        **kwargs,
    ):
        tag = self.new_label(label, before=before, after=after, within=within, **kwargs)
        remember = [dict(name="Harmony", duration=Fraction(0), tag=tag)]
        if loc_before is not None:
            location = self.new_location(loc_before)
            tag.insert_before(location)
            remember.insert(0, dict(name="location", duration=loc_before, tag=location))
        if loc_after is not None:
            location = self.new_location(loc_after)
            tag.insert_after(location)
            remember.append(dict(name="location", duration=loc_after, tag=location))
        return remember

    @cache
    def make_excerpt(
        self,
        included_mcs: Tuple[int] | int,
        globalkey: Optional[str] = None,
        localkey: Optional[str] = None,
        start_mc_onset: Optional[Fraction | float] = None,
        end_mc_onset: Optional[Fraction | float] = None,
        exclude_start: Optional[bool] = False,
        exclude_end: Optional[bool] = False,
        metronome_tempo: Optional[float] = None,
        metronome_beat_unit: Optional[Fraction] = Fraction(1 / 4),
        decompose_repeat_tags: Optional[bool] = True,
    ) -> Excerpt:
        """Create an excerpt by removing all <Measure> tags that are not selected in ``included_mcs``. The order of
        the given integers is inconsequential because measures are always printed in the order in which they appear in
        the score. Also, it is assumed that the MCs are consecutive, i.e. there are no gaps between them; otherwise
        the excerpt will not show correct measure numbers and might be incoherent in terms of missing key and time
        signatures.

        Args:
            included_mcs:
                List of measure counts to be included in the excerpt. Pass a single integer to get an excerpt from
                that MC to the end of the piece.
            globalkey:
                If the excerpt has chord labels, make sure the first label starts with the given global key, e.g.
                'F#' for F sharp major or 'ab' for A flat minor.
            localkey:
                If the excerpt has chord labels, make sure the first label starts with the given local key, e.g.
                'I' for the major tonic key or '#iv' for the raised subdominant minor key or 'bVII' for the lowered
                subtonic major key.
            start_mc_onset:
                Onset value (either Fraction or float) specified as the "true" start of the first measure. Every note
                with strictly smaller onset value will be "removed" (i.e. mutated into rest)
            end_mc_onset:
                Onset value (either Fraction or float) specified as the "true" end of the last measure. Every note
                with strictly greater onset value will be "removed" (i.e. mutated into rest)
            exclude_start:
                If set to True, the first note corresponding to ``start_mc_onset`` will also be "removed"
            exclude_end:
                If set to True, the last note corresponding to ``end_mc_onset`` will also be "removed"
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
            decompose_repeat_tags:
                If set to true, the XML tree will be cleansed from all tags referring to repeat-like structures to
                avoid possible "broken" structures within the excerpt.

        """
        measures = self.measures()
        available_mcs = measures.mc.to_list()
        last_mc = max(available_mcs)
        if isinstance(included_mcs, int):
            assert (
                included_mcs in available_mcs
            ), f"Score has no measure count {included_mcs} (available: 1 - {last_mc})"
            excluded_mcs = set(range(1, included_mcs))
            first_mc = included_mcs
            final_barline = True
        else:
            not_available = [mc for mc in included_mcs if mc not in available_mcs]
            assert (
                len(not_available) == 0
            ), f"Score has no measure counts {not_available} (available: 1 - {last_mc})"
            excluded_mcs = set(mc for mc in available_mcs if mc not in included_mcs)
            first_mc = min(included_mcs)
            final_barline = max(included_mcs) == last_mc
        assert excluded_mcs != available_mcs, (
            f"Cannot create an excerpt not containing no measures, which would be the result for included_mcs="
            f"{included_mcs}."
        )
        if self.soup is None:
            self.make_writeable()
        soup = copy(self.soup)
        part_tag = soup.find("Part")
        if part_tag is None:
            staff_tag_iterator = soup.find_all("Staff")
        else:
            staff_tag_iterator = part_tag.find_next_siblings("Staff")

        tempo_tags = []
        for staff_tag in staff_tag_iterator:
            for mc, measure_tag in enumerate(staff_tag.find_all("Measure"), 1):
                if mc <= min(included_mcs):
                    tempo_tag = measure_tag.find("Tempo")
                    if tempo_tag is not None:
                        tempo_tags.append(copy(tempo_tag))

        for staff_tag in staff_tag_iterator:
            for mc, measure_tag in enumerate(staff_tag.find_all("Measure"), 1):
                if mc in excluded_mcs:
                    measure_tag.decompose()
        mc_measures = measures.set_index("mc")
        first_selected = mc_measures.loc[first_mc]
        first_mn = first_selected.mn
        first_timesig = first_selected.timesig
        first_keysig = first_selected.keysig
        first_quarterbeat = first_selected.quarterbeats
        events = self.events()
        clefs = events[events.event == "Clef"]
        staff2clef = {}
        for staff, clefs_df in clefs.groupby("staff"):
            active_clef_row = get_row_at_quarterbeat(clefs_df, first_quarterbeat)
            if active_clef_row is not None:
                clef_values = {
                    k[5:]: v
                    for k, v in active_clef_row.items()
                    if k.startswith("Clef/")
                }
                staff2clef[staff] = clef_values
        harmony_selector = events.event == "Harmony"
        first_harmony_values = None
        if harmony_selector.any():
            harmonies = events[harmony_selector].sort_values("quarterbeats")
            if first_quarterbeat not in harmonies.quarterbeats.values:
                # harmony labels are present but not on beat 1 of the excerpt, so we will insert the one that's active
                active_harmony_row = get_row_at_quarterbeat(
                    harmonies, first_quarterbeat
                )
                if active_harmony_row is not None:
                    first_harmony_values = {
                        k[8:]: v
                        for k, v in active_harmony_row.items()
                        if k.startswith("Harmony/")
                    }
        if tempo_tags:
            first_tempo_tag = tempo_tags[-1]
        else:
            first_tempo_tag = None

        excerpt = Excerpt(
            soup,
            measures=included_mcs,
            read_only=False,
            logger_cfg=self.logger_cfg,
            first_mn=first_mn,
            first_timesig=first_timesig,
            first_keysig=first_keysig,
            first_harmony_values=first_harmony_values,
            first_tempo_tag=first_tempo_tag,
            staff2clef=staff2clef,
            final_barline=final_barline,
            globalkey=globalkey,
            localkey=localkey,
            start_mc_onset=start_mc_onset,
            end_mc_onset=end_mc_onset,
            exclude_start=exclude_start,
            exclude_end=exclude_end,
            metronome_tempo=metronome_tempo,
            metronome_beat_unit=metronome_beat_unit,
            decompose_repeat_tags=decompose_repeat_tags,
        )

        excerpt.filepath = self.filepath
        return excerpt

    def _make_measure_list(self, sections=True, secure=True, reset_index=True):
        """Regenerate the measure list from the parsed score with advanced options."""
        logger_cfg = self.logger_cfg.copy()
        return MeasureList(
            self._measures,
            sections=sections,
            secure=secure,
            reset_index=reset_index,
            logger_cfg=logger_cfg,
        )

    def make_standard_chordlist(self):
        """Stores the result of self.get_chords(mode='strict')"""
        self._cl = self.get_chords(mode="strict")

    def make_standard_restlist(self):
        self._rl = self.add_standard_cols(self._events[self._events.event == "Rest"])
        if len(self._rl) == 0:
            return
        self._rl = self._rl.rename(columns={"Rest/durationType": "nominal_duration"})
        self._rl.loc[:, "nominal_duration"] = self._rl.nominal_duration.map(
            self.durations
        )  # replace string values by fractions
        cols = [
            "mc",
            "mn",
            "mc_onset",
            "mn_onset",
            "timesig",
            "staff",
            "voice",
            "duration",
            "nominal_duration",
            "scalar",
        ]
        if self.has_voltas:
            cols.insert(2, "volta")
        self._rl = self._rl[cols].reset_index(drop=True)

    def make_standard_notelist(self):
        cols = {
            "midi": "Note/pitch",
            "tpc": "Note/tpc",
        }
        nl_cols = [
            "mc",
            "mn",
            "mc_onset",
            "mn_onset",
            "timesig",
            "staff",
            "voice",
            "duration",
            "gracenote",
            "nominal_duration",
            "scalar",
            "tied",
            "tpc",
            "midi",
            "name",
            "octave",
            "tuning",
            "chord_id",
        ]
        if self.has_voltas:
            nl_cols.insert(2, "volta")
        if len(self._notes.index) == 0:
            self._nl = pd.DataFrame(columns=nl_cols)
            return
        if "tremolo" in self._notes.columns:
            nl_cols.insert(9, "tremolo")
        self._nl = self.add_standard_cols(self._notes)
        self._nl.rename(columns={v: k for k, v in cols.items()}, inplace=True)
        self._nl = self._nl.merge(
            self.cl()[["chord_id", "nominal_duration", "scalar"]], on="chord_id"
        )
        tie_cols = [
            "Note/Spanner:type",
            "Note/Spanner/next/location",
            "Note/Spanner/prev/location",
        ]
        tied = make_tied_col(self._notes, *tie_cols)
        pitch_info = self._nl[["midi", "tpc"]].apply(pd.to_numeric).astype("Int64")
        pitch_info.tpc -= 14
        names, octaves = make_note_name_and_octave_columns(
            pd.concat([pitch_info, self._nl.staff], axis=1),
            staff2drums=self.staff2drum_map,
        )
        append_cols = [pitch_info, tied, names, octaves]
        if "Note/tuning" in self._notes.columns:
            detuned_notes = self._notes["Note/tuning"].rename("tuning")
            detuned_notes = pd.to_numeric(detuned_notes, downcast="float")
            append_cols.append(detuned_notes)
        self._nl = pd.concat(
            [self._nl.drop(columns=["midi", "tpc"])] + append_cols, axis=1
        )
        final_cols = [col for col in nl_cols if col in self._nl.columns]
        self._nl = sort_note_list(self._nl[final_cols])

    def make_writeable(self):
        if self.read_only:
            if not self.filepath:
                raise RuntimeError(
                    "Cannot be made writeable because no filepath is stored. Has the object been "
                    "created directly from BeautifulSoup?"
                )
            with open(self.filepath, "r", encoding="utf-8") as file:
                self.soup = bs4.BeautifulSoup(file.read(), "xml")
            self.read_only = False
            with temporarily_suppress_warnings(self) as self:
                # This is an automatic re-parse which does not have to be logged again
                self.parse_soup()
                self.parse_measures()

    def measures(
        self, interval_index: bool = False, unfold: bool = False
    ) -> Optional[pd.DataFrame]:
        """DataFrame representing the :ref:`measures` of the MuseScore file (which can be incomplete measures). Comes
        with the columns |mc|, |mn|, |quarterbeats|, |duration_qb|, |keysig|, |timesig|, |act_dur|, |mc_offset|,
        |volta|, |numbering_offset|, |dont_count|, |barline|, |breaks|, |repeats|, |next|

        Args:
          interval_index: Pass True to replace the default :obj:`~pandas.RangeIndex` by an :obj:`~pandas.IntervalIndex`.

        Returns:
          DataFrame representing the :ref:`measures <measures>` of the MuseScore file (which can be incomplete
          measures).
        """
        measures = self.ml()
        duration_qb = (measures.act_dur * 4).astype(float)
        measures.insert(2, "duration_qb", duration_qb)
        # add quarterbeats column
        if unfold:
            measures = self.unfold_facet_df(measures, "measures")
            if measures is None:
                return
        # functionality adapted from utils.make_continuous_offset()
        qb_column_name = (
            "quarterbeats_all_endings"
            if self.has_voltas and not unfold
            else "quarterbeats"
        )
        quarterbeats_col = (measures.act_dur.cumsum() * 4).shift(fill_value=0)
        insert_after = next(
            col
            for col in ("mn_playthrough", "mc_playthrough", "mn", "mc")
            if col in measures.columns
        )
        self.logger.debug(f"Inserting {qb_column_name} after '{insert_after}'")
        insert_position = measures.columns.get_loc(insert_after) + 1
        measures.insert(insert_position, qb_column_name, quarterbeats_col)
        if self.has_voltas and not unfold:
            self.logger.debug(
                "No quarterbeats are assigned to first endings. Pass unfold=True to "
                "compute quarterbeats for a full playthrough."
            )
            if 3 in measures.volta.values:
                self.logger.info(
                    "Piece contains third endings; please note that only second endings are taken into account for "
                    "quarterbeats."
                )
            quarterbeats_col = (
                measures.loc[measures.volta.fillna(2) == 2, "act_dur"]
                .cumsum()
                .shift(fill_value=0)
                .reindex(measures.index)
            )
            measures.insert(insert_position, "quarterbeats", quarterbeats_col * 4)
            self.logger.debug(f"Inserting 'quarterbeats' after '{insert_after}'")
        elif not self.has_voltas:
            measures.drop(columns="volta", inplace=True)
        if interval_index:
            # ToDo: same quarterbeats columns as for all other facets, i.e. always add quarterbeats_all_endings,
            # for unfolded, rename quarterbeats to quarterbeats_playthrough
            if unfold:
                position_col = "quarterbeats_playthrough"
            else:
                position_col = "quarterbeats_all_endings"
            if all(c in measures.columns for c in (position_col, "duration_qb")):
                measures = replace_index_by_intervals(
                    measures, position_col=position_col, logger=self.logger
                )
                self.logger.debug(
                    f"IntervalIndex created based on the column {position_col!r}."
                )
            else:
                self.logger.warning(
                    f"Cannot create interval index because column {position_col!r} is missing."
                )
        return measures.copy()

    def ml(self, recompute: bool = False) -> pd.DataFrame:
        """Get the raw :ref:`measures` without adding quarterbeat columns.

        Args:
          recompute: By default, the measures are cached. Pass True to enforce recomputing anew.
        """
        if recompute or self._ml is None:
            self._ml = self._make_measure_list()
        return self._ml.ml.copy()

    def new_label(
        self,
        label,
        harmony_layer=None,
        after=None,
        before=None,
        within=None,
        absolute_root=None,
        rootCase=None,
        absolute_base=None,
        leftParen=None,
        rightParen=None,
        offset_x=None,
        offset_y=None,
        nashville=None,
        decoded=None,
        color_name=None,
        color_html=None,
        color_r=None,
        color_g=None,
        color_b=None,
        color_a=None,
        placement=None,
        minDistance=None,
        style=None,
        z=None,
    ):
        tag = self.new_tag("Harmony")
        if not pd.isnull(harmony_layer):
            try:
                harmony_layer = int(harmony_layer)
            except Exception:
                if harmony_layer[0] in ("1", "2"):
                    harmony_layer = int(harmony_layer[0])
            # only include <harmonyType> tag for harmony_layer 1 and 2 (MuseScore's Nashville Numbers and
            # Roman Numerals)
            if harmony_layer in (1, 2):
                _ = self.new_tag("harmonyType", value=harmony_layer, append_within=tag)
        if not pd.isnull(leftParen):
            _ = self.new_tag("leftParen", append_within=tag)
        if not pd.isnull(absolute_root):
            _ = self.new_tag("root", value=absolute_root, append_within=tag)
        if not pd.isnull(rootCase):
            _ = self.new_tag("rootCase", value=rootCase, append_within=tag)
        if not pd.isnull(label):
            if label == "/":
                label = ""
            _ = self.new_tag("name", value=label, append_within=tag)
        else:
            assert not pd.isnull(
                absolute_root
            ), "Either label or root need to be specified."

        if not pd.isnull(z):
            _ = self.new_tag("z", value=z, append_within=tag)
        if not pd.isnull(style):
            _ = self.new_tag("style", value=style, append_within=tag)
        if not pd.isnull(placement):
            _ = self.new_tag("placement", value=placement, append_within=tag)
        if not pd.isnull(minDistance):
            _ = self.new_tag("minDistance", value=minDistance, append_within=tag)
        if not pd.isnull(nashville):
            _ = self.new_tag("function", value=nashville, append_within=tag)
        if not pd.isnull(absolute_base):
            _ = self.new_tag("base", value=absolute_base, append_within=tag)

        rgba = color_params2rgba(
            color_name, color_html, color_r, color_g, color_b, color_a
        )
        if rgba is not None:
            attrs = rgba2attrs(rgba)
            _ = self.new_tag("color", attributes=attrs, append_within=tag)

        if not pd.isnull(offset_x) or not pd.isnull(offset_y):
            if pd.isnull(offset_x):
                offset_x = "0"
            if pd.isnull(offset_y):
                offset_y = "0"
            _ = self.new_tag(
                "offset", attributes={"x": offset_x, "y": offset_y}, append_within=tag
            )
        if not pd.isnull(rightParen):
            _ = self.new_tag("rightParen", append_within=tag)
        if after is not None:
            after.insert_after(tag)
        elif before is not None:
            before.insert_before(tag)
        elif within is not None:
            within.append(tag)
        return tag

    def new_location(self, location):
        tag = self.new_tag("location")
        _ = self.new_tag("fractions", value=str(location), append_within=tag)
        return tag

    def new_tag(
        self,
        name: str,
        value: Optional[str] = None,
        attributes: Optional[dict] = None,
        after: Optional[bs4.Tag] = None,
        before: Optional[bs4.Tag] = None,
        append_within: Optional[bs4.Tag] = None,
        prepend_within: Optional[bs4.Tag] = None,
    ) -> bs4.Tag:
        """Create a new tag with the given name, value and attributes and insert it into the score relative to a
        given tag. Only one of ``after``, ``before``, ``append_within`` and ``prepend_within`` can be specified.

        Args:
            name: <name></name>
            value: <name>value</name> (if specified)
            attributes: <name key=value, ...></name>
            after: Insert the tag as sibling following the given tag.
            before: Insert the tag as sibling preceding the given tag.
            append_within: Insert the tag as last child of the given tag.
            prepend_within: Insert the tag as first child of the given tag.

        Returns:
            The new tag.
        """
        tag = self.soup.new_tag(name)
        if value is not None:
            tag.string = str(value)
        if attributes:
            for k, v in attributes.items():
                tag.attrs[k] = v

        if after is not None:
            after.insert_after(tag)
        elif before is not None:
            before.insert_before(tag)
        elif append_within is not None:
            append_within.append(tag)
        elif prepend_within is not None:
            prepend_within.insert(0, tag)

        return tag

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
        notes = self.nl()
        if unfold:
            notes = self.unfold_facet_df(notes, "notes")
            if notes is None:
                return
        notes = add_quarterbeats_col(
            notes,
            offset_dict=self.offset_dict(unfold=unfold),
            offset_dict_all_endings=self.offset_dict(all_endings=True),
            interval_index=interval_index,
            logger=self.logger,
        )
        return notes

    def nl(self, recompute: bool = False) -> pd.DataFrame:
        """Get the raw :ref:`notes` without adding quarterbeat columns.

        Args:
          recompute:  By default, the notes are cached. Pass True to enforce recomputing anew.
        """
        if recompute or len(self._nl) == 0:
            self.make_standard_notelist()
        return self._nl

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
        nrl = self.nrl()
        if unfold:
            nrl = self.unfold_facet_df(nrl, "notes and rests")
            if nrl is None:
                return
        nrl = add_quarterbeats_col(
            nrl,
            offset_dict=self.offset_dict(unfold=unfold),
            offset_dict_all_endings=self.offset_dict(all_endings=True),
            interval_index=interval_index,
            logger=self.logger,
        )
        return nrl

    def nrl(self, recompute: bool = False) -> pd.DataFrame:
        """Get the raw :ref:`notes_and_rests` without adding quarterbeat columns.

        Args:
          recompute:  By default, the measures are cached. Pass True to enforce recomputing anew.
        """
        if recompute or len(self._nrl) == 0:
            rl = self.rl()
            nl = self.nl()
            if len(rl) == 0:
                self._nrl = nl
            elif len(nl) == 0:
                self._nrl = rl
            else:
                nr = pd.concat([nl, rl]).astype(
                    {col: "Int64" for col in ["tied", "tpc", "midi", "chord_id"]}
                )
                self._nrl = sort_note_list(nr.reset_index(drop=True))
        return self._nrl

    @cache
    def offset_dict(
        self,
        all_endings: bool = False,
        unfold: bool = False,
    ) -> dict:
        """Dictionary mapping MCs (measure counts) to their quarterbeat offset from the piece's beginning.
        Used for computing quarterbeats for other facets.

        Args:
          all_endings:
              If a pieces as alternative endings, by default, only the second ending is taken into account for
              computing quarterbeats in order to make the timeline correspond to a rendition without performing
              repeats. Events in other endings, notably the first, receive value NA so that they can be filtered out.
              For score addressability, one might want to apply a continuous timeline to all measures, in which case
              one would pass True to use the column 'quarterbeats_all_endings' of the measures table if it has one.
              If not, falls back to the default 'quarterbeats'.
          unfold:
              Pass True to compute quarterbeats for a mc_playthrough column resulting from unfolding repeats.
              The parameter ``all_endings`` is ignored in this case because the unfolded version brings each ending in
              its correct place.

        Returns:
          {MC -> quarterbeat_offset}. Offsets are Fractions. If ``all_endings`` is not set to ``True``,
          values for MCs that are part of a first ending (or third or larger) are NA.
        """
        measures = self.measures(unfold=unfold)
        if unfold:
            offset_dict = make_continuous_offset_series(
                measures,
            ).to_dict()
        else:
            offset_dict = make_offset_dict_from_measures(measures, all_endings)
        return offset_dict

    def remove_empty_voices(self, mc, staff):
        voice_tags = self.measure_nodes[staff][mc].find_all("voice")
        dict_keys = sorted(self.tags[mc][staff])
        assert len(dict_keys) == len(
            voice_tags
        ), f"""In MC {mc}, staff {staff}, there are {len(voice_tags)} <voice> tags
but the keys of _MSCX_bs4.tags[{mc}][{staff}] are {dict_keys}."""
        for key, tag in zip(reversed(dict_keys), reversed(voice_tags)):
            if len(self.tags[mc][staff][key]) == 0:
                tag.decompose()
                del self.tags[mc][staff][key]
                self.logger.debug(
                    f"Empty <voice> tag of voice {key} deleted in MC {mc}, staff {staff}."
                )
            else:
                # self.logger.debug(f"No superfluous <voice> tags in MC {mc}, staff {staff}.")
                break

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
        rests = self.rl()
        if len(rests) == 0:
            return None
        if unfold:
            rests = self.unfold_facet_df(rests, "rests")
            if rests is None:
                return
        rests = add_quarterbeats_col(
            rests,
            offset_dict=self.offset_dict(unfold=unfold),
            offset_dict_all_endings=self.offset_dict(all_endings=True),
            interval_index=interval_index,
            logger=self.logger,
        )
        return rests

    def rl(self, recompute: bool = False) -> pd.DataFrame:
        """Get the raw :ref:`rests` without adding quarterbeat columns.

        Args:
          recompute:  By default, the measures are cached. Pass True to enforce recomputing anew.
        """
        if recompute or len(self._rl) == 0:
            self.make_standard_restlist()
        return self._rl

    def parse_soup(self) -> None:
        """First step of parsing the MuseScore source. Involves discovering the <staff> tags and storing the
        <Measure> tags of each in the :attr:`measure_nodes` dictionary.  Also stores the drum_map for each Drumset
        staff.
        """
        if self.version[0] not in ("3", "4"):
            # self.logger.exception(f"Cannot parse MuseScore {self.version} file.")
            raise ValueError(
                f"Cannot parse MuseScore {self.version} file. "
                f"Use 'ms3 convert' command or pass parameter 'ms' to Score to temporally convert."
            )

        root_tag = self.soup.find("museScore")
        if root_tag is None:
            self.logger.error(
                "This does not seem to be a MuseScore file because it lacks the <museScore> tag that "
                "would normally be the root of the XML tree."
            )
            return

        score_tags = root_tag.find_all("Score")
        if len(score_tags) == 0:
            score_tag = root_tag
        else:
            score_tag = score_tags[0]
            if len(score_tags) > 1:
                self.logger.warning(
                    "The file seems to include separately encoded parts, encoded with their own "
                    "<Score> tags. Only the first one will be considered."
                )

        # Check if any of the <Part> tags contains a pitch -> drumset instrument map
        # all_part_tags = self.soup.find_all('Part')
        # if len(all_part_tags) == 0:
        #     self.logger.error(f"Looks like an empty score to me.")
        part_tag = None
        for part_tag in score_tag.find_all("Part", recursive=False):
            drum_tags = part_tag.find_all("Drum")
            staff_tag = part_tag.find("Staff")
            if len(drum_tags) == 0 or staff_tag is None:
                continue
            staff = int(staff_tag["id"])
            drum_map = {}
            for tag in drum_tags:
                pitch = int(tag["pitch"])
                features = {t.name: str(t.string) for t in tag.find_all()}
                drum_map[pitch] = features
            df = pd.DataFrame.from_dict(drum_map, orient="index")
            df.index.rename("pitch", inplace=True)
            self.staff2drum_map[staff] = df

        # Populate measure_nodes with one {mc: <Measure>} dictionary per staff.
        # The <Staff> nodes containing the music are siblings of <Part>
        if part_tag is None:
            staff_iterator = score_tag.find_all("Staff")
        else:
            staff_iterator = part_tag.find_next_siblings("Staff")
        staff = None
        for staff in staff_iterator:
            staff_id = int(staff["id"])
            self.measure_nodes[staff_id] = {}
            for mc, measure in enumerate(
                staff.find_all("Measure"), start=self.first_mc
            ):
                self.measure_nodes[staff_id][mc] = measure
        if staff is None:
            self.logger.error("Looks like an empty score to me.")

    def parse_measures(self):
        """Converts the score into the three DataFrame self._measures, self._events, and self._notes"""
        if self.soup is None:
            raise RuntimeError(
                f"No BeautifulSoup available, the field has value {self.soup!r}"
            )
        grace_tags = [
            "grace4",
            "grace4after",
            "grace8",
            "grace8after",
            "grace16",
            "grace16after",
            "grace32",
            "grace32after",
            "grace64",
            "grace64after",
            "appoggiatura",
            "acciaccatura",
        ]

        measure_list, event_list, note_list = [], [], []
        staff_ids = tuple(self.measure_nodes.keys())
        chord_id = 0
        # For every measure: bundle the <Measure> nodes from every staff
        mc = (
            self.first_mc - 1
        )  # replace the previous enumerate() loop so we can filter out multimeasure rests which seem to be redundant
        # additional tags
        for measure_stack in zip(
            *[
                [measure_node for measure_node in measure_dict.values()]
                for measure_dict in self.measure_nodes.values()
            ]
        ):
            if measure_stack[0].find("multiMeasureRest") is not None:
                self.logger.debug(
                    f"Skipping multimeasure rest that follows MC {mc} in the encoding: {measure_stack}."
                )
                continue
            mc += 1
            if not self.read_only:
                self.tags[mc] = {}
            # iterate through staves and collect information about each <Measure> node
            for staff_id, measure in zip(staff_ids, measure_stack):
                if not self.read_only:
                    self.tags[mc][staff_id] = {}
                measure_info = {"mc": mc, "staff": staff_id}
                measure_info.update(recurse_node(measure, exclude_children=["voice"]))
                # iterate through <voice> tags and run a position counter
                voice_nodes = measure.find_all("voice", recursive=False)
                # measure_info['voices'] = len(voice_nodes)
                for voice_id, voice_node in enumerate(voice_nodes, start=1):
                    if not self.read_only:
                        self.tags[mc][staff_id][voice_id] = defaultdict(list)
                    # (re-)initialize variables for this voice's pass through the <Measure> tag
                    current_position = Fraction(0)
                    duration_multiplier = Fraction(1)
                    multiplier_stack = [Fraction(1)]
                    tremolo_type = None
                    tremolo_component = 0
                    # iterate through children of <voice> which constitute the note level of one notational layer
                    for event_node in voice_node.find_all(recursive=False):
                        event_name = event_node.name

                        event = {
                            "mc": mc,
                            "staff": staff_id,
                            "voice": voice_id,
                            "mc_onset": current_position,
                            "duration": Fraction(0),
                        }

                        if event_name == "Chord":
                            event["chord_id"] = chord_id
                            grace = event_node.find(grace_tags)

                            event_duration, dot_multiplier = bs4_chord_duration(
                                event_node, duration_multiplier
                            )
                            if grace:
                                event["gracenote"] = grace.name
                            else:
                                event["duration"] = event_duration
                            chord_info = dict(event)
                            # chord_info is a copy of the basic properties of the <Chord> that will be copied for each
                            # included <Note> and <Rest>; whereas the event dict will be updated with additional
                            # elements that make it into the "chords" and the "events" table

                            tremolo_tag = event_node.find("Tremolo")
                            if tremolo_tag:
                                if tremolo_component > 0:
                                    raise NotImplementedError(
                                        "Chord with <Tremolo> follows another one with <Tremolo>"
                                    )
                                tremolo_type = tremolo_tag.subtype.string
                                tremolo_duration_node = event_node.find("duration")
                                if tremolo_duration_node:
                                    # the tremolo has two components that factually start sounding
                                    # on the same onset, but are encoded as two subsequent <Chord> tags
                                    tremolo_duration_string = (
                                        tremolo_duration_node.string
                                    )
                                    tremolo_duration_fraction = Fraction(
                                        tremolo_duration_string
                                    )
                                    tremolo_component = 1
                                else:
                                    # the tremolo consists of one <Chord> only
                                    tremolo_duration_string = str(event_duration)
                            elif tremolo_component == 1:
                                # The previous <Chord> was the first component of a tremolo, so this one is marked
                                # as second component in the notes list (expected to have a <duration> tag of the
                                # same length). The pointer is set back by half the tremolo's length, which is the
                                # duration by which the first component had set it forward (see below). This was
                                # necessary to allow for the correct computation of positions encoded via <location>
                                # relative to the first component.
                                tremolo_component = 2
                                current_position -= tremolo_duration_fraction
                                event["mc_onset"] = current_position
                                chord_info["mc_onset"] = current_position
                            if tremolo_type:
                                chord_info["tremolo"] = (
                                    f"{tremolo_duration_string}_{tremolo_type}_{tremolo_component}"
                                )
                                if tremolo_component in (0, 2):
                                    # delete 'tremolo_type' which signals that the <Chord> is part of a tremolo
                                    tremolo_type = None
                                if tremolo_component == 2:
                                    completing_duration_node = event_node.find(
                                        "duration"
                                    )
                                    if completing_duration_node:
                                        duration_to_complete_tremolo = (
                                            completing_duration_node.string
                                        )
                                        if (
                                            duration_to_complete_tremolo
                                            != tremolo_duration_string
                                        ):
                                            self.logger.warning(
                                                "Two components of tremolo have non-matching <duration>"
                                            )
                                    tremolo_component = 0

                            for chord_child in event_node.find_all(recursive=False):
                                if chord_child.name == "Note":
                                    note_event = dict(
                                        chord_info,
                                        **recurse_node(
                                            chord_child, prepend=chord_child.name
                                        ),
                                    )
                                    note_list.append(note_event)
                                else:
                                    event.update(
                                        recurse_node(
                                            chord_child,
                                            prepend="Chord/" + chord_child.name,
                                        )
                                    )
                            chord_id += 1
                        elif event_name == "Rest":
                            event["duration"], dot_multiplier = bs4_rest_duration(
                                event_node, duration_multiplier
                            )
                        elif (
                            event_name == "location"
                        ):  # <location> tags move the position counter
                            event["duration"] = Fraction(event_node.fractions.string)
                        elif event_name == "Tuplet":
                            multiplier_stack.append(duration_multiplier)
                            duration_multiplier = duration_multiplier * Fraction(
                                int(event_node.normalNotes.string),
                                int(event_node.actualNotes.string),
                            )
                        elif event_name == "endTuplet":
                            duration_multiplier = multiplier_stack.pop()

                        # These nodes describe the entire measure and go into measure_list
                        # All others go into event_list
                        if event_name in ["TimeSig", "KeySig", "BarLine"] or (
                            event_name == "Spanner"
                            and "type" in event_node.attrs
                            and event_node.attrs["type"] == "Volta"
                        ):
                            measure_info.update(
                                recurse_node(event_node, prepend=f"voice/{event_name}")
                            )
                        else:
                            event.update({"event": event_name})
                            if event_name == "Chord":
                                event["scalar"] = duration_multiplier * dot_multiplier
                                for attr, value in event_node.attrs.items():
                                    event[f"Chord:{attr}"] = value
                            elif event_name == "Rest":
                                event["scalar"] = duration_multiplier * dot_multiplier
                                event.update(
                                    recurse_node(event_node, prepend=event_name)
                                )
                            else:
                                event.update(
                                    recurse_node(event_node, prepend=event_name)
                                )
                            if event_name == "FiguredBass":
                                components, duration = process_thoroughbass(event_node)
                                if len(components) > 0:
                                    thoroughbass_cols = {
                                        f"thoroughbass_level_{i}": comp
                                        for i, comp in enumerate(components, 1)
                                    }
                                    event.update(thoroughbass_cols)
                                    if duration is not None:
                                        event["thoroughbass_duration"] = duration

                            def safe_update_event(key, value):
                                """Update event dict unless key is already present."""
                                nonlocal event, current_position, staff_id, event_name, text_including_html
                                if key and key in event:
                                    self.logger.warning(
                                        f"MC {mc}@{current_position}, staff {staff_id}, {event_name!r} already "
                                        f"contained a '{key}': {event[key]} "
                                        f"so I did not overwrite it with {value!r}."
                                    )
                                else:
                                    event[key] = value

                            for text_tag in event_node.find_all("text"):
                                column_name = None  # the key to be written to the `event` row dict after the if-else
                                # block
                                parent_name = text_tag.parent.name
                                text_including_html = text_tag2str(text_tag)
                                text_excluding_html = text_tag2str_recursive(
                                    text_tag, join_char=" "
                                )
                                if parent_name == "Fingering":
                                    # fingerings occur within <Note> tags, if they are to be extracted, they should go
                                    # into the notes table
                                    continue
                                if parent_name == "Tempo":
                                    tempo_tag = text_tag.parent
                                    quarters_per_second = float(tempo_tag.tempo.string)
                                    safe_update_event(
                                        "qpm", round(quarters_per_second * 60)
                                    )
                                    safe_update_event("tempo", text_excluding_html)
                                    metronome_match = re.match(
                                        r"^(.+)=(([0-9]+(?:\.[0-9]*)?))$",
                                        text_excluding_html,
                                    )
                                    if metronome_match:
                                        base = metronome_match.group(1)
                                        value = metronome_match.group(2)
                                        safe_update_event("metronome_base", base)
                                        safe_update_event(
                                            "metronome_number", float(value)
                                        )
                                    try:
                                        tempo_visible = int(tempo_tag.visible.string)
                                    except AttributeError:
                                        tempo_visible = 1
                                    safe_update_event("tempo_visible", tempo_visible)
                                elif parent_name == "Lyrics":
                                    lyrics_tag = text_tag.parent
                                    no_tag = lyrics_tag.find("no")
                                    if no_tag is None:
                                        verse = 1
                                    else:
                                        verse_string = no_tag.string
                                        verse = int(verse_string) + 1
                                    column_name = f"lyrics_{verse}"
                                    syllabic_tag = lyrics_tag.find("syllabic")
                                    if syllabic_tag is not None:
                                        match syllabic_tag.string:
                                            case "begin":
                                                text_including_html = (
                                                    text_including_html + "-"
                                                )
                                            case "middle":
                                                text_including_html = (
                                                    "-" + text_including_html + "-"
                                                )
                                            case "end":
                                                text_including_html = (
                                                    "-" + text_including_html
                                                )
                                            case _:
                                                self.logger.warning(
                                                    f"<syllabic> tag came with the value '{syllabic_tag.string}', not "
                                                    f"begin|middle|end."
                                                )
                                    safe_update_event(column_name, text_including_html)
                                else:
                                    self.logger.debug(
                                        f"MC {mc}@{current_position}, staff {staff_id}, {event_name!r} contained a "
                                        f"<text> tag within a <{parent_name}> tag, "
                                        f"which I did not know how to handle. I stored it in the column "
                                        f"{parent_name}_text."
                                    )
                                    safe_update_event(
                                        parent_name + "_text", text_including_html
                                    )
                            event_list.append(event)

                        if not self.read_only:
                            remember = {
                                "name": event_name,
                                "duration": event["duration"],
                                "tag": event_node,
                            }
                            position = event["mc_onset"]
                            if event_name == "location" and event["duration"] < 0:
                                # this is a backwards pointer: store it where it points to for easy deletion
                                position += event["duration"]
                            self.tags[mc][staff_id][voice_id][position].append(remember)

                        if tremolo_component == 1 and event_name == "Chord":
                            # In case a tremolo appears in the score as two subsequent events of equal length,
                            # (rather than a single tremolo event), the first <Chord> contains a <Tremolo> tag and
                            # MuseScore assigns a <duration> of half the note value to both <Chord> components.
                            # The parser, instead, assigns the actual note value and the same position to both the
                            # <Chord> with the <Tremolo> tag and the following one. The current_position pointer,
                            # however, needs to move forward as the <duration> of the first component specifies in
                            # order to handle <location> tags correctly that might occur between the two tremolo
                            # components (e.g., the first harmonx in liszt_pelerinage/160.06_Vallee_dObermann, m. 121).
                            # This is achieved by moving the pointer forward by half the length of the tremolo after
                            # the first component (which is happening right here), and then substracting it again
                            # before adding the second component (see above in the code).
                            current_position += tremolo_duration_fraction
                        else:
                            current_position += event["duration"]

                measure_list.append(measure_info)
        self._measures = column_order(pd.DataFrame(measure_list))
        self._events = column_order(pd.DataFrame(event_list))
        if "chord_id" in self._events.columns:
            self._events.chord_id = self._events.chord_id.astype("Int64")
        self._notes = column_order(pd.DataFrame(note_list))
        if len(self._events) == 0:
            self.logger.warning("Score does not seem to contain any events.")
        else:
            self.has_annotations = "Harmony" in self._events.event.values
            if "StaffText/text" in self._events.columns:
                form_labels = (
                    self._events["StaffText/text"]
                    .str.contains(FORM_DETECTION_REGEX)
                    .fillna(False)
                )
                if form_labels.any():
                    self.n_form_labels = sum(form_labels)
        self.update_metadata()

    def perform_checks(self):
        """Perform a series of checks after parsing and emit warnings registered by the ms3 check command (and,
        by extension, by ms3 review, too)."""
        # check if the first measure includes a metronome mark
        events = self._events
        first_two_mcs_event_types = events.loc[events.mc.isin((1, 2)), "event"]
        metronome_mark_missing = True
        if "Tempo" in first_two_mcs_event_types.values:
            metronome_mark_missing = False
        # here we could insert logic for treating incipit measure groups differently
        if metronome_mark_missing:
            msg = "No metronome mark found in the first measure"
            tempo_selector = (events.event == "Tempo").fillna(False)
            if tempo_selector.sum() == 0:
                msg += " nor anywhere else in the score."
            else:
                all_tempo_mark_mcs = events.loc[
                    tempo_selector,
                    [
                        "mc",
                        "staff",
                        "voice",
                        "tempo",
                    ],
                ]
                msg += ". Later in the score:\n" + all_tempo_mark_mcs.to_string(
                    index=False
                )
            warn_msg = msg + (
                "\n* Please add one at the very beginning and hide it if it's not from the original "
                "print edition."
                "\n* Make sure to choose the rhythmic unit that corresponds to beats in this piece and to set "
                "another mark wherever that unit changes."
                "\n* The tempo marks can be rough estimates, maybe cross-checked with a recording."
            )
            with warnings.catch_warnings():
                warnings.simplefilter("once")
                warnings.warn(warn_msg, UserWarning)
            self.logger.warning(msg, extra=dict(message_id=(29,)))

    def store_score(self, filepath: str) -> bool:
        """Store the score as an MSCX file."""
        with open(resolve_dir(filepath), "w", encoding="utf-8") as file:
            result = self.write_score_to_handler(file)
        if result:
            self.logger.info(f"Score written to {filepath}.")
        return result

    def unfold_facet_df(
        self, facet_df: pd.DataFrame, facet: str
    ) -> Optional[pd.DataFrame]:
        if facet == "measures":
            return unfold_measures_table(facet_df, logger=self.logger)
        playthrough_info = make_playthrough_info(self.ml(), logger=self.logger)
        if playthrough_info is None:
            self.logger.warning(
                f"Unfolding '{facet}' unsuccessful. Check warnings concerning repeat structure and fix."
            )
            return
        facet_df = unfold_repeats(facet_df, playthrough_info, logger=self.logger)
        self.logger.debug(f"{facet} successfully unfolded.")
        return facet_df

    def update_metadata(self):
        self.metadata = self._get_metadata()

    def write_score_to_handler(self, file_handler: IO) -> bool:
        return write_score_to_handler(self.soup, file_handler, logger=self.logger)

    def __getstate__(self):
        """When pickling, make object read-only, i.e. delete the BeautifulSoup object and all references to tags."""
        super().__getstate__()
        self.soup = None
        self.tags = {}
        self.measure_nodes = {k: None for k in self.measure_nodes.keys()}
        self.read_only = True
        return self.__dict__


# ##########################################################################
# ###################### END OF _MSCX_bs4 DEFINITION #######################
# ##########################################################################


def replace_chord_tag_with_rest(target_tag):
    """This functions takes as a parameter a given chord tag from the XML tree and mutates it
    into a rest tag of the same exact notation. This functionality is useful to `trim` excerpts to have more
    control over the actual musical elements that are extracted. It also gives the advantage of not changing
    the relative positions of notes from the original score.

    Args:
        target_tag: bs4.Tag
            The chord tag that needs to be mutated into a rest tag of the same duration

    """
    grace_tags = [
        "grace4",
        "grace4after",
        "grace8",
        "grace8after",
        "grace16",
        "grace16after",
        "grace32",
        "grace32after",
        "grace64",
        "grace64after",
        "appoggiatura",
        "acciaccatura",
    ]
    for _ in target_tag.find_all(grace_tags):
        target_tag.decompose()
        return
    duration = copy(target_tag.find("durationType"))
    dots_tag = copy(target_tag.find("dots"))
    target_tag.clear()
    target_tag.name = "Rest"
    if dots_tag is not None:
        target_tag.append(dots_tag)
    target_tag.append(duration)


class Excerpt(_MSCX_bs4):
    """Takes a copy of :attr:`_MSCX_bs4.soup` and eliminates all <Measure> tags that do not correspond to the given
    list of MCs.
    """

    def __init__(
        self,
        soup: bs4.BeautifulSoup,
        measures: Tuple[int] | int,
        read_only: bool = False,
        logger_cfg: Optional[dict] = None,
        first_mn: Optional[int] = None,
        first_timesig: Optional[str] = None,
        first_keysig: Optional[int] = None,
        first_harmony_values: Optional[Dict[str, str]] = None,
        first_tempo_tag: Optional[bs4.Tag] = None,
        staff2clef: Optional[Dict[int, Dict[str, str]]] = None,
        final_barline: bool = False,
        globalkey: Optional[str] = None,
        localkey: Optional[str] = None,
        start_mc_onset: Optional[Fraction] = None,
        end_mc_onset: Optional[Fraction] = None,
        exclude_start: Optional[bool] = False,
        exclude_end: Optional[bool] = False,
        metronome_tempo: Optional[float] = None,
        metronome_beat_unit: Optional[Fraction] = Fraction(1 / 1),
        decompose_repeat_tags: Optional[bool] = True,
    ):
        """
        Args:
            soup: A beautifulsoup4 object representing the MSCX file.
            measures:
                The tuple containing the MC values of the included measures
            read_only:
                If set to True, all references to XML tags will be removed after parsing to allow the object to be
                pickled.
            logger_cfg:
                The following options are available:
                'name': LOGGER_NAME -> by default the logger name is based on the parsed file(s)
                'level': {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}
                'file': PATH_TO_LOGFILE to store all log messages under the given path.
            first_mn:
                Measure number to be displayed at the beginning of the excerpt.
            first_timesig:
                Time signature to be displayed at the beginning of the excerpt.
            first_keysig:
                Key signature to be displayed at the beginning of the excerpt.
            first_harmony_values:
                If a harmony is to be inserted at the beginning, pass the {tag -> value} dictionary specifying the
                tags to be appended as children of the <Harmony> tag. If mc_onset 0 already has a <Harmony> tag,
                it will be replaced in whatever (staff, voice) layer it occurs. Otherwise, the new tag will be inserted
                in the first voice of the lowest staff.
            staff2clef:
                A {staff -> {tag -> value}} dictionary specifying one dictionary for each staff at the beginning of
                which a <Clef> tag is to be created, containing the tags specified in the corresponding dict.
                Tag names containing a '/' are ignored for now.
            final_barline:
                By default, the last barline is prevented from being displayed as ending barline. Pass True if the
                excerpt's last measure is the final measure.
            globalkey:
                If the excerpt has chord labels, make sure the first label starts with the given global key, e.g.
                'F#' for F sharp major or 'ab' for A flat minor.
            localkey:
                If the excerpt has chord labels, make sure the first label starts with the given local key, e.g.
                'I' for the major tonic key or '#iv' for the raised subdominant minor key or 'bVII' for the lowered
                subtonic major key.
            start_mc_onset:
                Onset value (either Fraction or float) specified as the "true" start of the first measure. Every note
                with strictly smaller onset value will be "removed" (i.e. mutated into rest)
            end_mc_onset:
                Onset value (either Fraction or float) specified as the "true" end of the last measure. Every note
                with strictly greater onset value will be "removed" (i.e. mutated into rest)
            exclude_start:
                If set to True, the note first note corresponding to ``start_mc_onset`` will also be "removed"
            exclude_end:
                If set to True, the note last note corresponding to ``end_mc_onset`` will also be "removed"
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
            decompose_repeat_tags:
                If set to true, the XML tree will be cleansed from all tags referring to repeat-like structures to
                avoid possible "broken" structures within the excerpt.
        """
        super().__init__(soup=soup, read_only=read_only, logger_cfg=logger_cfg)

        # to prepend within first <Measure>
        if first_mn:  # doesn't call if first_mn == 0
            self.set_first_mn(first_mn)

        # # to prepend within first <voice> (in that order)
        if first_harmony_values:
            self.replace_first_harmony(first_harmony_values)
        if first_timesig:
            self.set_first_timesig(first_timesig)
        if first_keysig:  # doesn't call if first_keysig == 0 (no accidentals)
            self.set_first_keysig(first_keysig)
        if staff2clef:
            self.set_clefs(staff2clef)

        # to append within last <Measure>
        if not final_barline:
            self.remove_final_barline()

        # sanitize values in case NaN was passed
        if pd.isnull(globalkey):
            globalkey = None
        if pd.isnull(localkey):
            localkey = None
        # amend first label to indicate global and/or local key
        if globalkey or localkey:
            self.amend_first_harmony_keys(globalkey, localkey)

        # fine trimming with onset values
        if start_mc_onset is not None or end_mc_onset is not None:
            self.trim(start_mc_onset, end_mc_onset, exclude_start, exclude_end)

        # enforcing user-set tempo or amending last active metronome mark
        self.set_tempo(first_tempo_tag, metronome_tempo, metronome_beat_unit)

        # cleaning tree from repeat-structure tags
        if decompose_repeat_tags:
            self.decompose_repeat_tags()

    def set_tempo(self, first_tempo_tag, metronome_tempo, metronome_beat_unit):
        """This method handles the enforcing of the tempo at the beginning of the excerpt. If a metronome mark
        was found in the piece from which the excerpt was taken, and was still active, and no tempo was specified by the
        user, then it will be set again in the first measure of the excerpt. Otherwise, if the user indeed specified
        a tempo along with a beat unit, a custom metronome mark will be added to the beginning of the excerpt
        overwriting any possible pre-existing metronome mark that could've been there.

        Args:
            first_tempo_tag:
                The last active metronome mark found in the original piece (if any was found)
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
        """
        if metronome_tempo is not None:
            if first_tempo_tag is not None:
                self.logger.info("You are overwriting an existing active tempo")
            self.enforce_tempo(
                metronome_tempo=metronome_tempo,
                metronome_beat_unit=metronome_beat_unit,
                user_call=True,
            )
        elif first_tempo_tag is not None:
            self.enforce_tempo(piece_tempo_tag=first_tempo_tag, user_call=False)

    def trim(
        self,
        start_mc_onset: Optional[Fraction] = None,
        end_mc_onset: Optional[Fraction] = None,
        exclude_start: Optional[bool] = False,
        exclude_end: Optional[bool] = False,
    ):
        """This method handles the trimming of the excerpt where notes outside of the set onset boundaries are
        mutated into rests (to not change the relative positions of the notes in the whole excerpt).

        Args:
            start_mc_onset:
                The onset value before which we want to mutate all other notes (associated with first measure)
            end_mc_onset:
                The onset value after which we want to mutate all other notes (associated with last measure)
            exclude_start:
                If set to `True`, the note corresponding to the `start_mc_onset` in the first measure will also be
                removed
            exclude_end:
                If set to `True`, the note corresponding to the `end_mc_onset` in the last measure will also be removed
        """
        assert not (
            start_mc_onset is None and end_mc_onset is None
        ), "At least one onset value (for either the start or the end) must be defined."

        self.replace_chords_with_rests(
            start_onset=start_mc_onset,
            end_onset=end_mc_onset,
            exclude_start=exclude_start,
            exclude_end=exclude_end,
        )

    def amend_first_harmony_keys(
        self,
        globalkey: Optional[str] = None,
        localkey: Optional[str] = None,
    ):
        if globalkey is None and localkey is None:
            return
        harmony_tag = self.get_onset_zero_harmony(return_layer=False)
        if not harmony_tag:
            self.logger.warning(
                "Could not find <Harmony> tag at mc_onset 0 to amend keys."
            )
            return
        name_tag, current_label = find_tag_get_string(harmony_tag, "name")
        if name_tag is None:
            self.logger.warning(
                "Could not find <name> tag in <Harmony> tag at mc_onset 0 to amend keys."
            )
            return
        keys_regex = re.compile(
            r"""
        ^(\.?
            ((?P<globalkey>[a-gA-G](b*|\#*))\.)?
            ((?P<localkey>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)+)\.)?
            (?P<label>.+)
        )$""",
            re.VERBOSE,
        )
        match = keys_regex.match(current_label)
        if not match:
            self.logger.warning(
                f"Current label {current_label!r} does not match the expected format."
            )
            return
        current_values = match.groupdict()
        if globalkey:
            current_values["globalkey"] = globalkey
        if localkey:
            current_values["localkey"] = localkey
        new_label = ".".join(value for value in current_values.values() if value)
        name_tag.string = new_label
        self.logger.debug(f"First label {current_label!r} amended to {new_label!r}.")

    def iter_first_measures(self) -> Iterator[bs4.Tag]:
        for measure_dict in self.measure_nodes.values():
            yield measure_dict[1]

    def iter_last_measures(self) -> Iterator[bs4.Tag]:
        first_staff_measure_dict = self.measure_nodes[1]
        last_mc = max(first_staff_measure_dict.keys())
        for measure_dict in self.measure_nodes.values():
            yield measure_dict[last_mc]

    def remove_final_barline(self):
        for measure_tag in self.iter_last_measures():
            first_voice_tag = measure_tag.find("voice")
            self.new_tag("BarLine", append_within=first_voice_tag)

    def replace_first_harmony(self, first_harmony_values: Dict[str, str]):
        harmony_tag, staff, voice = self.get_onset_zero_harmony(return_layer=True)
        if harmony_tag is not None:
            self.delete_label(mc=1, staff=staff, voice=voice, mc_onset=0)
        else:
            staff = -1
            voice = 1
        label = first_harmony_values.pop("name", None)
        harmony_layer = first_harmony_values.pop("harmonyType", None)
        self.add_label(
            label=label,
            mc=1,
            mc_onset=0,
            staff=staff,
            voice=voice,
            harmony_layer=harmony_layer,
            **first_harmony_values,
        )

    @overload
    def get_onset_zero_harmony(
        self, return_layer: Literal[False]
    ) -> Optional[bs4.Tag]: ...

    @overload
    def get_onset_zero_harmony(
        self, return_layer: Literal[True]
    ) -> Tuple[Optional[bs4.Tag], int, int]: ...

    def get_onset_zero_harmony(self, return_layer: bool = False) -> Optional[bs4.Tag]:
        """Iterate through all tags at mc_onset 0 for all notational (staff, voice) layers and return the first
        <Harmony> tag or None."""
        for staff, voices_dict in self.tags[1].items():
            # iterate through staves of MC 1
            for voice, onset2tags in voices_dict.items():
                # iterate through voices of current staff
                if 0 not in onset2tags:
                    continue
                for tag_info in onset2tags[0]:
                    # iterate through all tags at mc_onset 0
                    if tag_info["name"] == "Harmony":
                        if return_layer:
                            return tag_info["tag"], staff, voice
                        else:
                            return tag_info["tag"]
        if return_layer:
            return None, None, None
        else:
            return None

    def set_clefs(self, staff2clef: Dict[int, Dict[str, str]]):
        """Set the initial clefs for the given staves."""
        for staff, tag_value_dict in staff2clef.items():
            first_measure = self.measure_nodes[staff][1]
            first_voice = first_measure.find("voice")
            clef_tag = self.new_tag("Clef", prepend_within=first_voice)
            for tag, value in tag_value_dict.items():
                if pd.isnull(value):
                    continue
                if "/" in tag:
                    self.logger.debug(
                        f"Haven't learned how to deal with secondary Clef tags such as Clef/{tag}. "
                        f"Igoring."
                    )
                elif ":" in tag:
                    self.logger.debug(
                        f"Inclusion of tag attributes (such as {tag}) not yet implemented."
                    )
                else:
                    _ = self.new_tag(tag, value=value, append_within=clef_tag)

    def set_first_keysig(self, first_keysig: int):
        """Set the key signature of the first measure to the given value."""
        if first_keysig == 0:
            self.logger.debug("first_keysig == 0, so I won't set a key signature.")
            return
        for measure_tag in self.iter_first_measures():
            first_voice_tag = measure_tag.find("voice")
            keysig_tag = measure_tag.find("KeySig")
            if keysig_tag is None:
                keysig_tag = self.new_tag("KeySig", prepend_within=first_voice_tag)
                _ = self.new_tag(
                    "accidental", value=first_keysig, append_within=keysig_tag
                )

    def set_first_mn(self, first_mn: int):
        """Set the measure number of the first measure to the given value."""
        for i, measure_tag in enumerate(self.iter_first_measures()):
            # <irregular> tags need to ensure that the first measure has number 1
            irregular_tag = measure_tag.find("irregular")
            if irregular_tag:
                irregular_tag.decompose()
            if i == 0:
                # the measure number offset is encoded only in the first staff
                # the offset is first_mn - 1 because the first measure has number 1 by default
                _ = self.new_tag(
                    "noOffset", value=first_mn - 1, prepend_within=measure_tag
                )

    def set_first_timesig(self, first_timesig: str):
        sigN, sigD = first_timesig.split("/")
        for measure_tag in self.iter_first_measures():
            first_voice_tag = measure_tag.find("voice")
            timesig_tag = measure_tag.find("TimeSig")
            if timesig_tag is None:
                timesig_tag = self.new_tag("TimeSig", prepend_within=first_voice_tag)
                _ = self.new_tag("sigN", value=sigN, append_within=timesig_tag)
                _ = self.new_tag("sigD", value=sigD, append_within=timesig_tag)

    def set_first_tempo(self, active_tempo_tag: bs4.Tag):
        self.enforce_tempo(piece_tempo_tag=active_tempo_tag, user_call=False)

    def replace_chords_with_rests(
        self,
        start_onset: Optional[Fraction | float] = None,
        end_onset: Optional[Fraction | float] = None,
        exclude_start: Optional[bool] = False,
        exclude_end: Optional[bool] = False,
    ):
        """The method that given the specific onset and measure values, will handle the silencing of all notes that
        are not withing the onset bounds. More specifically, notes that appear before the ``start_onset`` in the
        ``start_mc`` will be mutated to rests (i.e. silenced). Same thing goes for the ``end_mc``. All notes found
        after the ``end_onset`` will also be mutated to rests.

        Args:
            start_onset:
                onset value set for the first measure. Everything before this will be silenced
            end_onset:
                onset value set for the last measure. Everything after this will be silenced
            exclude_start:
                If set to ``True``, the note corresponding to ``start_onset`` in the first measure will also be silenced
            exclude_end:
                If set to ``True``, the note corresponding to ``end_onset`` in the last measure will also be silenced
        """
        if start_onset is not None:
            staves = self.tags[1]
            for staff, voices in staves.items():
                for voice, onsets in voices.items():
                    for onset, tag_dicts in onsets.items():
                        if onset == start_onset and not exclude_start:
                            continue
                        elif onset > start_onset:
                            continue
                        for tag_dict in tag_dicts:
                            if tag_dict["name"] != "Chord":
                                continue
                            replace_chord_tag_with_rest(tag_dict["tag"])
        else:
            self.logger.warning(
                "Both the starting MC value and the onset need to be specified for trimming"
            )

        end = max(self.tags.keys())

        if end_onset is not None:
            staves = self.tags[end]
            for staff, voices in staves.items():
                for voice, onsets in voices.items():
                    for onset, tag_dicts in onsets.items():
                        if onset == end_onset and not exclude_end:
                            continue
                        elif onset < end_onset:
                            continue
                        for tag_dict in tag_dicts:
                            if tag_dict["name"] != "Chord":
                                continue
                            replace_chord_tag_with_rest(tag_dict["tag"])
        else:
            self.logger.warning(
                "Both the ending MC value and the onset need to be specified for trimming"
            )

    def enforce_tempo(
        self,
        piece_tempo_tag: Optional[bs4.Tag] = None,
        metronome_tempo: Optional[float] = None,
        metronome_beat_unit: Optional[Fraction | float] = Fraction(1 / 4),
        user_call: Optional[bool] = True,
    ):
        """Creates the artificial hidden metronome mark that either comes from the last active metronome mark of the
        original piece or from some specified tempo and beat unit values specified by the user.


        Args:
            piece_tempo_tag:
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
            user_call:

        Returns:

        """
        for measure_tag in self.iter_first_measures():
            tempo_tag = measure_tag.find("Tempo")
            timesig_tag = measure_tag.find("TimeSig")
            if not user_call and piece_tempo_tag is not None and tempo_tag is None:
                # Copying active tempo tag from "parent" piece
                _ = self.new_tag(
                    name="visible",
                    value=str(0),
                    append_within=piece_tempo_tag,
                )
                timesig_tag.insert_after(piece_tempo_tag)
                return
            elif user_call and tempo_tag is not None:
                relative_tempo = compute_relative_tempo(
                    metronome_tempo=metronome_tempo,
                    metronome_beat_unit=metronome_beat_unit,
                )
                tempo_tag.clear()
                _ = self.new_tag(
                    name="tempo",
                    value=str(relative_tempo),
                    append_within=tempo_tag,
                )
                # Make marking hidden
                _ = self.new_tag(
                    name="visible",
                    value=str(0),
                    append_within=tempo_tag,
                )
                return
            elif user_call and tempo_tag is None:
                relative_tempo = compute_relative_tempo(
                    metronome_tempo=metronome_tempo,
                    metronome_beat_unit=metronome_beat_unit,
                )
                tempo_tag = self.new_tag(name="Tempo", after=timesig_tag)
                _ = self.new_tag(
                    name="tempo",
                    value=str(relative_tempo),
                    append_within=tempo_tag,
                )
                # Make marking hidden
                _ = self.new_tag(
                    name="visible",
                    value=str(0),
                    append_within=tempo_tag,
                )
                return
            elif piece_tempo_tag is None and not user_call:
                self.logger.warning(
                    "No active tempo was found and none was set by the user."
                )
                return

    def decompose_repeat_tags(self):
        """Decomposes all tags that refer to repeat structures of any king in the XML tree of the excerpt.
        This is a safety measure to avoid ending up with broken repeat structures that would alter the proper "timeline"
        of the excerpt itself."""
        soup = self.soup
        tags = [
            {"name": "endRepeat"},
            {"name": "startRepeat"},
            {"name": "noOffset"},
            {"name": "Jump"},
            {"name": "Marker"},
        ]

        for tag in tags:
            for _ in soup.find_all(name=tag["name"]):
                _.decompose()

        # not in the list because has an attribute. Easier this way
        for _ in soup.find_all("Spanner", type="Volta"):
            _.decompose()


def compute_relative_tempo(
    metronome_tempo: float,
    metronome_beat_unit: Optional[Fraction] = Fraction(1 / 4),
):
    unit = Fraction(metronome_beat_unit).limit_denominator(32)
    return np.round((metronome_tempo / 60) * unit * 4, 3)


class ParsedParts(LoggedClass):
    """
    Storing found parts object from a BeautifulSoup file

    Args:
        soup: bs4.BeautifulSoup,
            BeautifulSoup object to parse
        **logger_cfg:obj:`dict`, optional
            The following options are available:
            'name': LOGGER_NAME -> by default the logger name is based on the parsed file(s)
            'level': {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}
            'file': PATH_TO_LOGFILE to store all log messages under the given path.
    """

    def __init__(self, soup: bs4.BeautifulSoup, **logger_cfg):
        super().__init__("ParsedParts", logger_cfg)
        self.parts_data: Dict[str, bs4.Tag] = {
            f"part_{i}": part for i, part in enumerate(soup.find_all("Part"), 1)
        }

    @property
    def staff2part(self) -> dict[list, str]:
        """
        Allows users to determine the corresponding part based on the staff number

        Example:
            Returns {[2, 3]: 'part_1'} for staves 2 and 3 of part 1

        Returns:
            dict[list, str]: the dictionary mapping parts to staves

        """
        staff2part = {}
        for key_part, part in self.parts_data.items():
            staves = [f"staff_{staff['id']}" for staff in part.find_all("Staff")]
            staff2part.update(dict.fromkeys(staves, key_part))
        return staff2part

    def __repr__(self):
        return pformat(self.parts_data, sort_dicts=False)


"""Instrument Defaults is a csv file that includes all possible instruments and their properties: 'id', 'longName',
    'shortName', 'trackName', 'instrumentId', 'part_trackName', 'ChannelName', 'ChannelValue'"""
INSTRUMENT_DEFAULTS = pd.read_csv(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "instrument_defaults.csv"
    ),
    index_col=0,
)
INSTRUMENT_DEFAULTS[["controllers", "ChannelName", "ChannelValue"]] = (
    INSTRUMENT_DEFAULTS[["controllers", "ChannelName", "ChannelValue"]].apply(
        lambda k: list(map(lambda j: eval(j) if j is not None else None, k))
    )
)
for int_column in ["keysig", "useDrumset"]:
    INSTRUMENT_DEFAULTS[int_column] = INSTRUMENT_DEFAULTS[int_column].astype("Int64")
INSTRUMENT_DEFAULTS.replace({np.nan: None}, inplace=True)


def get_enlarged_default_dict() -> Dict[str, dict]:
    """
    Allows users to point to an instrument not only with a 'trackName', but also with 'id', 'longName', 'shortName',
       'instrumentId', 'part_trackName'

    Returns:
        Dict[str, dict]: dictionary mapping any of the possible fields ('id', 'longName', 'shortName', trackName',
       'instrumentId', 'part_trackName') corresponding to an instrument into complete information about the instrument
       ('id',  'longName', 'shortName', 'trackName', 'instrumentId', 'part_trackName', 'ChannelName', 'ChannelValue')
    """
    enlarged_dict = dict(INSTRUMENT_DEFAULTS.T)
    # we drop "ChannelName", "ChannelValue" etc because they are not unique for the instrument, so they can't be keys
    for cur_key, cur_value in (
        INSTRUMENT_DEFAULTS.T.drop(
            [
                "ChannelName",
                "ChannelValue",
                "useDrumset",
                "clef",
                "group",
                "staff_type_name",
                "defaultClef",
                "controllers",
                "keysig",
            ]
        )
        .to_dict()
        .items()
    ):
        added_value = INSTRUMENT_DEFAULTS.T[cur_key]
        # additional_key takes values from 'id', 'longName', 'shortName', 'trackName', 'instrumentId', 'part_trackName'
        for additional_key in cur_value.values():
            if additional_key is not None:
                if type(additional_key) is str:
                    additional_key = additional_key.lower().strip(".")
                if additional_key in enlarged_dict:
                    continue
                enlarged_dict[additional_key] = added_value
    return enlarged_dict


class Instrumentation(LoggedClass):
    """Easy way to read and write the instrumentation of a score, that is
    'id', 'longName', 'shortName', 'trackName', 'instrumentId', 'part_trackName',
                                       'ChannelName', 'ChannelValue'."""

    key2default_instrumentation = get_enlarged_default_dict()

    def __init__(self, soup: bs4.BeautifulSoup, **logger_cfg):
        super().__init__("Instrumentation", logger_cfg)
        self.part_tracknames = INSTRUMENT_DEFAULTS["part_trackName"]
        self.soup = soup
        self.instrumentation_fields = [
            "id",
            "longName",
            "shortName",
            "trackName",
            "instrumentId",
            "part_trackName",
            "keysig",
            "ChannelName",
            "ChannelValue",
            "useDrumset",
            "clef",
            "group",
            "staff_type_name",
            "defaultClef",
            "controllers",
        ]
        self.only_drumset_features = ["staff_type_name", "defaultClef"]
        self.parsed_parts = ParsedParts(soup)
        self.soup_references_data = (
            self.soup_references()
        )  # store references to XML tags
        self.updated = defaultdict()

    def soup_references(self) -> dict[str, dict[str, bs4.Tag]]:
        """
        Stores tags references for each staff

        Returns: the dictionary in the format {'staff_1': {'id': None, 'longName': None, 'shortName': None,
        'trackName': None, 'instrumentId': None, 'part_trackName': None, 'ChannelName', 'ChannelValue'},
        'staff_2': {...}, ...} containing the BeautifulSoup tags

        """
        tag_dict = {}
        for key_part, part in self.parsed_parts.parts_data.items():
            instrument_tag = part.Instrument
            staves_list = part.find_all("Staff")
            staves = [f"staff_{(staff['id'])}" for staff in staves_list]
            staves_dict = {}
            for key_staff, data_staff in zip(staves, staves_list):
                staff_type = data_staff.StaffType
                staves_dict[key_staff] = {
                    "group": staff_type["group"],
                    "staff_type_name": staff_type.find("name"),
                    "keysig": staff_type.find("keysig"),
                    "defaultClef": data_staff.find("defaultClef"),
                }
            channel_info = part.find_all("Channel")
            cur_dict = {
                "id": instrument_tag.get("id"),
                "ChannelName": [],
                "ChannelValue": [],
                "controllers": [],
            }
            for elem in channel_info:
                channel_name = None if "name" not in elem.attrs.keys() else elem["name"]
                cur_dict["ChannelName"].append(channel_name)
                cur_dict["ChannelValue"].append(elem.program)
                cur_dict["controllers"].append(
                    [
                        {"ctrl": elem["ctrl"], "value": elem["value"]}
                        for elem in elem.find_all("controller")
                    ]
                )
            cur_dict.update(staves_dict[staves[0]])
            for name in self.instrumentation_fields:
                if name not in cur_dict.keys():
                    if name == "part_trackName":
                        tag = part.trackName
                    else:
                        tag = instrument_tag.find(name)
                    if name == "trackName" and (
                        tag is None or tag.get_text() == ""
                    ):  # this corresponds to the current behaviour of bs4_parser.get_part_info
                        part_trackName = part.trackName.string
                        instrument_tag.trackName.string = part_trackName if part_trackName else ""
                        tag = instrument_tag.find(name)
                    cur_dict[name] = tag
            tag_dict.update(
                {key_staff: cur_dict | staves_dict[key_staff] for key_staff in staves}
            )
        return tag_dict

    @property
    def fields(self):
        """
        Extracts information from the tag and stores it for each staff

        Returns: the dictionary in the format {'staff_1': {'id': None, 'longName': None, 'shortName': None,
        'trackName': None, 'instrumentId': None, 'part_trackName': None, 'ChannelName', 'ChannelValue'},
        'staff_2': {...}, ...} containing the information extracted from tags

        """
        result = {}
        for key, instr_data in self.soup_references().items():
            result[key] = {}
            for key_instr_data, tag in instr_data.items():
                if (
                    type(tag) in [bs4.element.Tag, list]
                    and tag is not None
                    and tag != [None]
                ):
                    if key_instr_data == "ChannelValue":
                        value = [int(elem["value"]) for elem in tag]
                    elif key_instr_data in ["useDrumset", "keysig"]:
                        value = int(tag.get_text())
                    elif key_instr_data == "controllers":
                        value = [
                            [
                                {"ctrl": elem["ctrl"], "value": elem["value"]}
                                for elem in channel_elem
                            ]
                            for channel_elem in tag
                        ]
                    elif key_instr_data == "ChannelName":
                        value = [elem for elem in tag]
                    else:
                        value = tag.get_text()
                else:
                    value = tag
                result[key][key_instr_data] = value
        return result

    def get_instrument_name(self, staff_name: Union[str, int]):
        """
        Allows users accessing the instrument trackname attributed to the staff staff_name
        Args:
            staff_name: a number or a string in the format 'staff_1' defining the staff of interest

        Returns:
            str: trackName extracted from tag for the staff staff_name

        """
        if isinstance(staff_name, int):
            staff_name = f"staff_{staff_name}"
        fields_data = self.fields
        if (
            staff_name not in self.parsed_parts.staff2part.keys()
            or staff_name not in fields_data
        ):
            raise KeyError(f"No data for staff '{staff_name}'")
        else:
            return fields_data[staff_name]["trackName"]

    def add_suffix(self, new_values, suffix):
        """
        Adds suffix of the instrument
        Args:
            new_values: the dictionary of fields to update
            suffix: the string containing version

        Returns:
            the dictionary with updated names with versions
        """
        update_dict = new_values.copy()
        for version_key in ["trackName", "longName", "shortName"]:
            version_value = new_values[version_key]
            if version_value is not None:
                update_dict[version_key] = f"{version_value} {suffix}"
        return update_dict

    def modify_drumset_tags(self, staff_type, value, changed_part, field_to_change):
        """
        Sets tags specific for Drumset instruments
        Args:
            staff_type: the tags containing info of the field
            value: new value of the field
            changed_part: the index of part to update
            field_to_change: the name of field to update

        """
        for elem in staff_type:
            tag = elem.find(field_to_change)
            if value is not None:
                if tag is not None:
                    tag.string = value
                else:
                    new_tag = self.soup.new_tag(field_to_change)
                    new_tag.string = str(value)
                    elem.append(new_tag)
                    self.logger.debug(
                        f"Added new {new_tag} with value {value!r} to part {changed_part}"
                    )
            else:
                if tag is not None:
                    tag.extract()

    def modify_list_tags(self, changed_part, found, value):
        """
        Sets instruments if there is alist of values to update
        :param changed_part: number of part of soup file where to find and update in the original file
        :param found: parts of soup containing channel info in the original file
        :param value: new values to set
        :return: corrected list of parts of the same length as value list
        """
        l_found, l_value = 1 if found is None else len(found), (
            1 if value is None else len(value)
        )
        if l_found < l_value:
            for i in range(l_value - l_found):
                new_tag = self.soup.new_tag("Channel")
                new_tag.string = str(value[i + len(found) - 1])
                new_tag.append(self.soup.new_tag("program"))
                self.parsed_parts.parts_data[changed_part].append(new_tag)
                self.logger.debug(
                    f"Added new {new_tag} with value {value!r} to part {changed_part}"
                )
        elif l_found > l_value:
            for elem in found[l_value:]:
                elem.extract()
        return self.parsed_parts.parts_data[changed_part].find_all("Channel"), value

    def set_instrument(self, staff_id: Union[str, int], trackname):
        """
        Modifies the instrument and all its corresponding information in the soup source file

        Args:
            staff_id: an integer number i or a string in the format 'staff_i' defining the staff of interest
            trackname:
                key defining the new value of the instrument, can be one of ('id', 'longName', 'shortName',
                trackName', 'instrumentId', 'part_trackName')

        """
        # preprocessing and verification of correctness of staff_id
        available_staves = list(self.parsed_parts.staff2part.keys())
        if not isinstance(staff_id, str):
            try:
                staff_id = int(staff_id)
            except Exception:
                raise ValueError(
                    f"{staff_id!r} cannot be interpreted as staff ID which needs to be int or str, not "
                    f"{type(staff_id)}. Use one of {available_staves}."
                )
            staff_id = f"staff_{staff_id}"
        if staff_id not in available_staves:
            raise KeyError(
                f"Don't recognize key '{staff_id}'. Use one of {available_staves}."
            )
        changed_part = self.parsed_parts.staff2part[staff_id]
        self.logger.debug(
            f"References to tags before the instrument was changed: {self.soup_references()}"
        )

        # checking that the current changes will not affect other staves
        staves_within_part = np.array(
            [
                staff_key
                for staff_key, part_value in self.parsed_parts.staff2part.items()
                if part_value == changed_part and staff_key != staff_id
            ]
        )  # which staves share this part

        # preprocessing and verification of correctness of trackname
        trackname_norm = trackname.lower().strip(".")
        if trackname_norm not in self.key2default_instrumentation:
            # add splitting by suffix and then adapt other names to it
            split_trackname = trackname.split()
            trackname_without_suffix = " ".join(split_trackname[:-1]).lower().strip(".")
            if trackname_without_suffix in self.key2default_instrumentation:
                suffix = split_trackname[-1]
                new_values = self.add_suffix(
                    self.key2default_instrumentation[trackname_without_suffix], suffix
                )
                self.updated.update({staff_id: new_values["id"]})
            else:
                # if there is no data for the trackname to update
                fuzzy_matches = difflib.get_close_matches(
                    trackname_norm, list(self.key2default_instrumentation.keys()), n=1
                )
                if len(fuzzy_matches) == 0:
                    suggestion = (
                        "and no default name was found via fuzzy string matching."
                    )
                else:
                    suggestion = f". Did you mean {fuzzy_matches[0]}?"
                trackname_old = self.fields[staff_id]["instrumentId"].lower().strip(".")
                self.logger.warning(
                    f"Don't recognize trackName '{trackname}'{suggestion} Instrumentation of "
                    f"staves {np.append(staves_within_part, staff_id)} is left unchanged with instrument:"
                    f" {trackname_old}",
                    extra=dict(message_id=(30,)),
                )
                if trackname_old not in self.key2default_instrumentation:
                    trackname_old = (
                        self.fields[staff_id]["part_trackName"].lower().strip(".")
                    )
                new_values = self.key2default_instrumentation[trackname_old]
        else:
            new_values = self.key2default_instrumentation[trackname_norm]
            self.updated.update({staff_id: new_values["id"]})

        # if no drumset updates we drop redundant features
        if (
            new_values.useDrumset is None
            and self.fields[staff_id]["useDrumset"] is None
        ):
            for elem in self.only_drumset_features:
                if elem in self.instrumentation_fields:
                    self.instrumentation_fields.remove(elem)
        else:
            self.instrumentation_fields.extend(self.only_drumset_features)
            self.instrumentation_fields = list(set(self.instrumentation_fields))
        if len(staves_within_part) > 0:
            damaged_upd_staves = [
                staff_key
                for staff_key in set(staves_within_part) & self.updated.keys()
                if staff_key and new_values["id"] != self.updated[staff_key]
            ]
            if len(damaged_upd_staves) > 0:
                damaged_dict = {elem: self.updated[elem] for elem in damaged_upd_staves}
                damaged_dict[staff_id] = new_values["id"]
                self.logger.warning(
                    f"You are trying to assign instruments {pformat(damaged_dict, width=1)} but they are belonging to "
                    f"the same part. In order to assign two different instruments, you would have to split them in two "
                    f"parts in MuseScore. For now, I'm assigning {new_values['id']!r} to all of them.",
                    extra=dict(message_id=(31,)),
                )
            else:
                different_values_set = np.where(
                    [
                        new_values["id"] != self.fields[staff_key]["id"]
                        for staff_key in staves_within_part
                    ]
                )[
                    0
                ]  # staves of the same part with different instruments
                if len(different_values_set) > 0:
                    damaged_staves = staves_within_part[different_values_set]
                    damaged_dict = {
                        elem: self.fields[elem]["id"] for elem in damaged_staves
                    }
                    self.logger.warning(
                        f"The change of {staff_id} to {new_values['id']} will also affect staves {damaged_staves} with "
                        f"instruments: \n {pformat(damaged_dict, width=1)}",
                        extra=dict(message_id=(31,)),
                    )
        # modification of fields
        staff_data = self.parsed_parts.parts_data[changed_part].find_all("Staff")
        staff_type = [elem.StaffType for elem in staff_data]
        channel_data = self.parsed_parts.parts_data[changed_part].find_all("Channel")
        for field_to_change in self.instrumentation_fields:
            value = new_values[field_to_change]
            self.logger.debug(
                f"field {field_to_change!r} to be updated from {self.soup_references_data[staff_id][field_to_change]} "
                f"to {value!r}"
            )
            if field_to_change == "id":
                self.parsed_parts.parts_data[changed_part].Instrument[
                    field_to_change
                ] = value
            elif field_to_change == "ChannelName":
                channel_data, value = self.modify_list_tags(
                    changed_part, channel_data, value
                )
                if value is not None:
                    for idx_channel, found_channel in enumerate(channel_data):
                        cur_value = value[idx_channel]
                        if cur_value is not None:
                            found_channel["name"] = cur_value
            elif field_to_change == "controllers":
                channel_data, value = self.modify_list_tags(
                    changed_part, channel_data, value
                )
                for idx_channel, found_channel in enumerate(channel_data):
                    cur_value = value[idx_channel]
                    found = found_channel.find_all("controller")
                    for idx, elem in enumerate(cur_value):
                        if idx >= len(found) - 1:
                            new_tag = self.soup.new_tag("controller")
                            new_tag["ctrl"] = cur_value[idx]["ctrl"]
                            new_tag["value"] = cur_value[idx]["value"]
                            found_channel.append(new_tag)
                        else:
                            found[idx]["ctrl"] = elem["ctrl"]
                            found[idx]["value"] = elem["value"]
                    if len(found) > len(cur_value):
                        for i in range(len(cur_value) - len(found)):
                            found[i + len(found) - 1].extract()
            elif field_to_change == "ChannelValue":
                channel_data, value = self.modify_list_tags(
                    changed_part, channel_data, value
                )
                for idx_channel, found_channel in enumerate(channel_data):
                    cur_value = value[idx_channel]
                    if cur_value is not None:
                        found_channel.program["value"] = cur_value
            elif field_to_change == "group":
                for elem in staff_type:
                    elem["group"] = value
            elif field_to_change == "staff_type_name":
                self.modify_drumset_tags(staff_type, value, changed_part, "name")
            elif field_to_change == "defaultClef":
                self.modify_drumset_tags(
                    staff_data, value, changed_part, field_to_change
                )
            elif field_to_change == "keysig":
                self.modify_drumset_tags(
                    staff_type, value, changed_part, field_to_change
                )
            elif (
                field_to_change in ["clef", "useDrumset", "keysig"]
                and self.soup_references_data[staff_id][field_to_change] is not None
                and value is None
            ):
                self.soup_references_data[staff_id][field_to_change].extract()
            else:
                if self.soup_references_data[staff_id][field_to_change] is not None:
                    self.soup_references_data[staff_id][field_to_change].string = value
                    self.logger.debug(
                        f"Updated {field_to_change!r} to {value!r} in part {changed_part}"
                    )
                elif value is not None:
                    new_tag = self.soup.new_tag(field_to_change)
                    new_tag.string = str(value)
                    self.parsed_parts.parts_data[changed_part].Instrument.append(
                        new_tag
                    )
                    self.logger.debug(
                        f"Added new {new_tag} with value {value!r} to part {changed_part}"
                    )
            self.soup_references_data = self.soup_references()  # update references
        self.logger.debug(
            f"References to tags after the instrument was changed: {self.soup_references()}"
        )

    def __repr__(self):
        return pformat(self.fields, sort_dicts=False)


class Metatags:
    """Easy way to read and write any style information in a parsed MSCX score."""

    def __init__(self, soup):
        self.soup = soup

    @property
    def tags(self) -> Dict[str, bs4.Tag]:
        return {tag["name"]: tag for tag in self.soup.find_all("metaTag")}

    @property
    def fields(self):
        return {
            name: "" if tag.string is None else str(tag.string)
            for name, tag in self.tags.items()
        }

    def remove(self, tag_name) -> bool:
        tag = self.get_tag(tag_name)
        if tag is None:
            return False
        tag.decompose()
        return True

    def __getitem__(self, attr) -> Optional[str]:
        """Retrieve value of metadata tag."""
        tags = self.tags
        if attr in tags:
            val = tags[attr].string
            return "" if val is None else str(val)
        return None

    def get_tag(self, attr) -> Optional[bs4.Tag]:
        tags = self.tags
        return tags.get(attr)

    def __setitem__(self, attr, val):
        tags = self.tags
        if attr in tags:
            tags[attr].string = str(val)
        else:
            new_tag = self.soup.new_tag("metaTag")
            new_tag.attrs["name"] = attr
            new_tag.string = str(val)
            for insert_here in tags.keys():
                if insert_here > attr:
                    break
            tags[insert_here].insert_before(new_tag)

    def __repr__(self):
        return "\n".join(str(t) for t in self.tags.values())


class Style:
    """Easy way to read and write any style information in a parsed MSCX score."""

    def __init__(self, soup):
        self.soup = soup
        self.style = self.soup.find("Style")
        assert self.style is not None, "No <Style> tag found."

    def __getitem__(self, attr):
        tag = self.style.find(attr)
        if tag is None:
            return None
        val = tag.string
        return "" if val is None else str(val)

    def __setitem__(self, attr, val):
        if attr in self:
            tag = self.style.find(attr)
            tag.string = str(val)
        else:
            new_tag = self.soup.new_tag(attr)
            new_tag.string = str(val)
            self.style.append(new_tag)

    def __iter__(self):
        tags = self.style.find_all()
        return (t.name for t in tags)

    def __repr__(self):
        tags = self.style.find_all()
        return ", ".join(t.name for t in tags)


class Prelims(LoggedClass):
    """Easy way to read and write the preliminaries of a score, that is
    Title, Subtitle, Composer, Lyricist, and 'Instrument Name (Part)'."""

    styles = ("Title", "Subtitle", "Composer", "Lyricist", "Instrument Name (Part)")
    keys = (
        "title_text",
        "subtitle_text",
        "composer_text",
        "lyricist_text",
        "part_name_text",
    )  # == utils.MUSESCORE_HEADER_FIELDS
    key2style = dict(zip(keys, styles))
    style2key = dict(zip(styles, keys))

    def __init__(self, soup: bs4.BeautifulSoup, **logger_cfg):
        super().__init__("Prelims", logger_cfg)
        self.soup = soup
        vbox_tag = get_vbox(soup, self.logger)
        if vbox_tag is None:
            self.vbox = self.soup.new_tag("VBox")
            part = soup.find("Part")
            first_staff = part.find_next_sibling("Staff")
            first_staff.insert(0, self.vbox)
            self.logger.debug("Inserted <VBox> at the beginning of the first staff.")
        else:
            self.vbox = vbox_tag

    @property
    def text_tags(self) -> Dict[str, bs4.Tag]:
        """Returns a {key->tag} dict reflecting the <Text> tags currently present in the first <VBox>."""
        tag_dict = {}
        for text_tag in self.vbox.find_all("Text"):
            style = text_tag.find("style")
            if style is not None:
                identifier = str(style.string)
                if identifier in self.style2key:
                    key = self.style2key[identifier]
                    tag_dict[key] = text_tag
                else:
                    self.logger.info(
                        f"Score contains a non-default text field '{identifier}' in the header that "
                        f"can only be amended or removed manually."
                    )
        return tag_dict

    @property
    def fields(self) -> Dict[str, str]:
        """Returns a {key->value} dict reflecting the currently set <text> values."""
        result = {}
        for key, tag in self.text_tags.items():
            value, _ = tag2text(tag)
            result[key] = value
        return result

    def __getitem__(self, key) -> Optional[str]:
        if key not in self.keys:
            raise KeyError(f"Don't recognize key '{key}'")
        fields = self.fields
        if key in fields:
            return fields[key]
        return

    def __setitem__(self, key, val: str):
        if key not in self.keys:
            raise KeyError(f"Don't recognize key '{key}'")
        existing_value = self[key]
        new_value = str(val)
        if existing_value is not None and existing_value == new_value:
            self.logger.debug(
                f"The {key} was already '{existing_value}' and doesn't need changing."
            )
            return
        clean_tag = self.soup.new_tag("Text")
        style_tag = self.soup.new_tag("style")
        style_tag.string = self.key2style[key]
        clean_tag.append(style_tag)
        text_tag = self.soup.new_tag("text")
        # turn the new value into child nodes of an HTML <p> tag (in case it contains HTML markup)
        new_value_as_html_body = bs4.BeautifulSoup(new_value, "lxml").find("body")
        new_value_as_p_tag = new_value_as_html_body.find("p")
        if new_value_as_p_tag is None:
            # if the created HTML contains a <p> tag, the new value (with tags or without) has been wrapped
            iter_contents = new_value_as_html_body.contents
        else:
            iter_contents = new_value_as_p_tag.contents
        for tag_or_string in iter_contents:
            text_tag.append(copy(tag_or_string))
        clean_tag.append(text_tag)
        text_tags = self.text_tags
        if existing_value is None:
            following_key_index = self.keys.index(key) + 1
            try:
                following_present_key = next(
                    k for k in self.keys[following_key_index:] if k in text_tags
                )
                following_tag = text_tags[following_present_key]
                following_tag.insert_before(clean_tag)
                self.logger.info(
                    f"Inserted {key} before existing {self.keys[following_key_index]}."
                )
            except StopIteration:
                self.vbox.append(clean_tag)
                self.logger.info(
                    f"Appended {key} as last tag of the VBox (after {text_tags.keys()})."
                )
        else:
            existing_tag = text_tags[key]
            existing_tag.replace_with(clean_tag)
            self.logger.info(f"Replaced {key} '{existing_value}' with '{new_value}'.")


def get_duration_event(elements):
    """Receives a list of dicts representing the events for a given mc_onset and returns the index and name of
    the first event that has a duration, so either a Chord or a Rest."""
    names = [e["name"] for e in elements]
    if "Chord" in names or "Rest" in names:
        if "Rest" in names:
            ix = names.index("Rest")
            name = "<Rest>"
        else:
            ix = next(
                i
                for i, d in enumerate(elements)
                if d["name"] == "Chord" and d["duration"] > 0
            )
            name = "<Chord>"
        return ix, name
    return (None, None)


def get_vbox(soup: bs4.BeautifulSoup, logger=None) -> Optional[bs4.Tag]:
    """
    Returns the first <VBox> tag contained in the first staff, if any, which usually corresponds to the vertical
    box at the top of a MuseScore file which contains the prelims (title, composer, etc.)
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    part = soup.find("Part")
    first_staff = part.find_next_sibling("Staff")
    vbox_nodes = first_staff.find_all("VBox")
    if len(vbox_nodes) == 0:
        return
    result = vbox_nodes[0]
    if len(vbox_nodes) > 1:
        logger.warning(
            "First staff starts off with more than one VBox. Picked the first one."
        )
    return result


def get_part_info(part_tag):
    """Instrument names come in different forms in different places. This function extracts the information from a
    <Part> tag and returns it as a dictionary."""
    res = {}
    res["staves"] = [int(staff["id"]) for staff in part_tag.find_all("Staff")]
    if part_tag.trackName is not None and part_tag.trackName.string is not None:
        res["trackName"] = part_tag.trackName.string.strip()
    else:
        res["trackName"] = ""
    if part_tag.Instrument is not None:
        instr = part_tag.Instrument
        if instr.longName is not None and instr.longName.string is not None:
            res["longName"] = instr.longName.string.strip()
        if instr.shortName is not None and instr.shortName.string is not None:
            res["shortName"] = instr.shortName.string.strip()
        if instr.trackName is not None and instr.trackName.string is not None:
            res["instrument"] = instr.trackName.string.strip()
        else:
            res["instrument"] = res["trackName"]
    return res


def make_spanner_cols(
    df: pd.DataFrame, spanner_types: Optional[Collection[str]] = None, logger=None
) -> pd.DataFrame:
    """From a raw chord list as returned by ``get_chords(spanners=True)``
        create a DataFrame with Spanner IDs for all chords for all spanner
        types they are associated with.

    Args:
        spanner_types
            If this parameter is passed, only the enlisted
            spanner types ['Slur', 'HairPin', 'Pedal', 'Ottava'] are included.

    History of this algorithm
    -------------------------

    At first, spanner IDs were written to Chords of the same layer until a prev/location was found. At first this
    caused some spanners to continue until the end of the piece because endings were missing when selecting based
    on the subtype column (endings don't specify subtype). After fixing this, there were still mistakes,
    particularly for slurs, because:
    1. endings can be missing, 2. endings can occur in a different voice than they should, 3. endings can be
    expressed with different values than the beginning (all three cases found in
    ms3/tests/test_local_files/MS3/stabat_03_coloured.mscx)
    Therefore, the new algorithm ends spanners simply after their given duration.
    """

    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    cols = {
        "nxt_m": "Spanner/next/location/measures",
        "nxt_f": "Spanner/next/location/fractions",
        # 'prv_m': 'Spanner/prev/location/measures',
        # 'prv_f': 'Spanner/prev/location/fractions',
        "type": "Spanner:type",
    }
    # nxt = beginning of spanner & indication of its duration
    # (prv = ending of spanner & negative duration supposed to match nxt)

    def get_spanner_ids(
        spanner_type: str, subtype: Optional[str] = None
    ) -> Dict[str, List[Union[str | Literal[pd.NA]]]]:
        """

        Args:
          spanner_type: Create one or several columns expressing all <Spanner type=``spanner_type``> tags.
          subtype:
              Defaults to None. If at least one spanner includes a <subtype> tag, the function will call itself
              for every subtype and create column names of the form spanner_type:subtype

        Returns:
          {column_name -> [IDs]} dictionary. IDs start at 0 and appear in every row that falls within the respective
          spanner's span. In the case of Slurs, however, this is true only for rows with events occurring in the
          same voice as the spanner.
        """
        nonlocal df
        if spanner_type == "Slur":
            spanner_duration_cols = [
                "Chord/" + cols[c] for c in ["nxt_m", "nxt_f"]
            ]  # , 'prv_m', 'prv_f']]
            type_col = "Chord/" + cols["type"]
        else:
            spanner_duration_cols = [
                cols[c] for c in ["nxt_m", "nxt_f"]
            ]  # , 'prv_m', 'prv_f']]
            type_col = cols["type"]

        subtype_col = f"Spanner/{spanner_type}/subtype"
        if subtype is None and subtype_col in df:
            # automatically generate one column per available subtype
            subtypes = set(df.loc[df[subtype_col].notna(), subtype_col])
            results = [get_spanner_ids(spanner_type, st) for st in subtypes]
            return dict(ChainMap(*results))

        # select rows corresponding to spanner_type
        boolean_selector = df[type_col] == spanner_type
        # then select only beginnings
        existing = [col for col in spanner_duration_cols if col in df.columns]
        boolean_selector &= df[existing].notna().any(axis=1)
        if subtype is not None:
            boolean_selector &= df[subtype_col] == subtype
        duration_df = pd.DataFrame(index=df.index, columns=spanner_duration_cols)
        duration_df.loc[boolean_selector, existing] = df.loc[boolean_selector, existing]
        with warnings.catch_warnings():
            # Setting values in-place is fine, ignore the warning in Pandas >= 1.5.0
            # This can be removed, if Pandas 1.5.0 does not need to be supported any longer.
            # See also: https://stackoverflow.com/q/74057367/859591
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=(
                    ".*will attempt to set the values inplace instead of always setting a new array. "
                    "To retain the old behavior, use either.*"
                ),
            )
            duration_df.iloc[:, 0] = (
                duration_df.iloc[:, 0].fillna(0).astype(int).abs()
            )  # nxt_m
            duration_df.iloc[:, 1] = (
                duration_df.iloc[:, 1].fillna(0).map(Fraction)
            )  # nxt_f
        custom_text_col = f"Spanner/{spanner_type}/beginText"
        add_custom_text_cols = (
            custom_text_col in df and df[custom_text_col].notna().any()
        )
        if add_custom_text_cols:
            custom_texts = df[custom_text_col]
            concat_this = [df[["mc", "mc_onset", "staff"]], duration_df, custom_texts]
            custom_text2ids = {text: [] for text in custom_texts.dropna().unique()}
        else:
            concat_this = [df[["mc", "mc_onset", "staff"]], duration_df]
        time_and_duration_df = pd.concat(concat_this, axis=1)

        current_id = -1
        column_name = spanner_type
        if subtype:
            column_name += ":" + subtype
        distinguish_voices = spanner_type in ["Slur", "Trill"]
        if distinguish_voices:
            # slurs need to be ended by the same voice, there can be several going on in parallel in different voices
            time_and_duration_df.insert(3, "voice", df.voice)
            one_stack_per_layer = {
                (i, v): {} for i in df.staff.unique() for v in range(1, 5)
            }
        else:
            # For all other spanners, endings can be encoded in any of the 4 voices
            one_stack_per_layer = {i: {} for i in df.staff.unique()}
        # one_stack_per_layer contains for every possible layer a dictionary  {ID -> (end_mc, end_f)};
        # going through chords chronologically, output all "open" IDs for the current layer until they are closed, i.e.
        # removed from the stack

        def row2active_ids(row) -> Union[str | Literal[pd.NA]]:
            nonlocal one_stack_per_layer, current_id, distinguish_voices, custom_text2ids
            if distinguish_voices:
                if add_custom_text_cols:
                    mc, mc_onset, staff, voice, nxt_m, nxt_f, custom_text = row
                else:
                    mc, mc_onset, staff, voice, nxt_m, nxt_f = row
                layer = (staff, voice)
            else:
                if add_custom_text_cols:
                    mc, mc_onset, staff, nxt_m, nxt_f, custom_text = row
                else:
                    mc, mc_onset, staff, nxt_m, nxt_f = row
                layer = staff

            beginning = nxt_m > 0 or nxt_f != 0
            if beginning:
                current_id += 1
                one_stack_per_layer[layer][current_id] = (mc + nxt_m, mc_onset + nxt_f)
                if add_custom_text_cols and not pd.isnull(custom_text):
                    custom_text2ids[custom_text].append(str(current_id))
            for active_id, (end_mc, end_f) in tuple(one_stack_per_layer[layer].items()):
                if end_mc < mc or (end_mc == mc and end_f <= mc_onset):
                    del one_stack_per_layer[layer][active_id]
            val = ", ".join(str(i) for i in one_stack_per_layer[layer].keys())
            return val if val != "" else pd.NA

        # create the ID column for the currently selected spanner (sub)type
        res = {
            column_name: [row2active_ids(row) for row in time_and_duration_df.values]
        }
        # ## With the new algorithm, remaining 'open' spanners result from no further event occurring in the
        # ## respective layer after the end of the last spanner.
        # open_ids = {layer: d for layer, d in one_stack_per_layer.items() if len(d) > 0}
        # if len(open_ids) > 0:
        #     logger.warning(f"At least one of the spanners of type {spanner_type}"
        #                    f"{'' if subtype is None else ', subtype: ' + subtype} "
        #                    f"has not been closed: {open_ids}")
        if not add_custom_text_cols:
            return res
        if not any(len(ids) > 0 for ids in custom_text2ids.values()):
            logger.warning(
                f"None of the {column_name} IDs have been attributed to one of the custom texts "
                f"{list(custom_text2ids.keys())}."
            )
            return res
        split_ids = [
            [] if pd.isnull(value) else value.split(", ") for value in res[column_name]
        ]
        for text, relevant_ids in custom_text2ids.items():
            custom_column_name = f"{column_name}_{text}"
            subselected_ids = [
                [ID for ID in relevant_ids if ID in ids] for ids in split_ids
            ]
            custom_column = [
                pd.NA if len(ids) == 0 else ", ".join(ids) for ids in subselected_ids
            ]
            res[custom_column_name] = custom_column
        return res

    type_col = cols["type"]
    types = (
        list(set(df.loc[df[type_col].notna(), type_col]))
        if type_col in df.columns
        else []
    )
    if "Chord/" + type_col in df.columns:
        types += ["Slur"]
    if spanner_types is not None:
        types = [t for t in types if t in spanner_types]
    list_of_dicts = [get_spanner_ids(t) for t in types]
    merged_dict = dict(ChainMap(*list_of_dicts))
    renaming = {
        "HairPin:0": "crescendo_hairpin",
        "HairPin:1": "decrescendo_hairpin",
        "HairPin:2": "crescendo_line",
        "HairPin:3": "diminuendo_line",
        "Slur": "slur",
        "Pedal": "pedal",
    }
    return pd.DataFrame(merged_dict, index=df.index).rename(columns=renaming)


def make_tied_col(df, tie_col, next_col, prev_col):
    new_col = pd.Series(pd.NA, index=df.index, name="tied")
    if tie_col not in df.columns:
        return new_col
    has_tie = df[tie_col].fillna("").str.contains("Tie")
    if has_tie.sum() == 0:
        return new_col
    # merge all columns whose names start with `next_col` and `prev_col` respectively
    next_cols = [col for col in df.columns if col[: len(next_col)] == next_col]
    nxt = df[next_cols].notna().any(axis=1)
    prev_cols = [col for col in df.columns if col[: len(prev_col)] == prev_col]
    prv = df[prev_cols].notna().any(axis=1)
    new_col = new_col.where(~has_tie, 0).astype("Int64")
    tie_starts = has_tie & nxt
    tie_ends = has_tie & prv
    new_col.loc[tie_ends] -= 1
    new_col.loc[tie_starts] += 1
    return new_col


def safe_update(old, new):
    """Update dict without replacing values."""
    existing = [k for k in new.keys() if k in old]
    if len(existing) > 0:
        new = dict(new)
        for ex in existing:
            old[ex] = f"{old[ex]} & {new[ex]}"
            del new[ex]
    old.update(new)


def recurse_node(node, prepend=None, exclude_children=None):
    """The heart of the XML -> DataFrame conversion. Changes may have ample repercussions!

    Returns
    -------
    :obj:`dict`
        Keys are combinations of tag (& attribute) names, values are value strings.
    """

    def tag_or_string(c, ignore_empty=False):
        nonlocal info, name
        if isinstance(c, bs4.element.Tag):
            if c.name not in exclude_children:
                safe_update(
                    info,
                    {
                        child_prepend + k: v
                        for k, v in recurse_node(c, prepend=c.name).items()
                    },
                )
        elif c not in ["\n", None]:
            info[name] = str(c)
        elif not ignore_empty:
            if c == "\n":
                info[name] = "∅"
            elif c is None:
                info[name] = "/"

    info = {}
    if exclude_children is None:
        exclude_children = []
    name = node.name if prepend is None else prepend
    attr_prepend = name + ":"
    child_prepend = "" if prepend is None else prepend + "/"
    for attr, value in node.attrs.items():
        info[attr_prepend + attr] = value
    children = tuple(node.children)
    if len(children) > 1:
        for c in children:
            tag_or_string(c, ignore_empty=True)
    elif len(children) == 1:
        tag_or_string(children[0], ignore_empty=False)
    else:
        info[name] = "/"
    return info


def bs4_chord_duration(
    node: bs4.Tag, duration_multiplier: Fraction = Fraction(1)
) -> Tuple[Fraction, Fraction]:
    duration_type_tag = node.find("durationType")
    if duration_type_tag is None:
        return Fraction(0), Fraction(0)
    durationtype = duration_type_tag.string
    if durationtype == "measure" and node.find("duration"):
        nominal_duration = Fraction(node.find("duration").string)
    else:
        nominal_duration = _MSCX_bs4.durations[durationtype]
    dots = node.find("dots")
    dotmultiplier = (
        sum([Fraction(1, 2) ** i for i in range(int(dots.string) + 1)])
        if dots
        else Fraction(1)
    )
    return nominal_duration * duration_multiplier * dotmultiplier, dotmultiplier


def bs4_rest_duration(node, duration_multiplier=Fraction(1)):
    return bs4_chord_duration(node, duration_multiplier)


def decode_harmony_tag(tag):
    """Decode a <Harmony> tag into a string."""
    label = ""
    if tag.function is not None:
        label = str(tag.function.string)
    if tag.leftParen is not None:
        label = "("
    if tag.root is not None:
        root = fifths2name(tag.root.string, ms=True)
        if str(tag.rootCase) == "1":
            root = root.lower()
        label += root
    name = tag.find("name")
    if name is not None:
        label += str(name.string)
    if tag.base is not None:
        label += "/" + str(tag.base.string)
    if tag.rightParen is not None:
        label += ")"
    return label


def text_tag2str(tag: bs4.Tag) -> str:
    """Transforms a <text> tag into a string that potentially includes written-out HTML tags."""
    components = []
    for c in tag.contents:
        if isinstance(c, NavigableString):
            components.append(c)
        elif c.name == "sym":
            sym = c.string
            if sym in NOTE_SYMBOL_MAP:
                components.append(NOTE_SYMBOL_MAP[sym])
        else:
            # <i></i> or other text markup within the string
            components.append(str(c))
    txt = "".join(components)
    return txt


def text_tag2str_components(tag: bs4.Tag) -> List[str]:
    """Recursively traverses a <text> tag and returns all string components, effectively removing all HTML markup."""
    components = []
    for c in tag.contents:
        if isinstance(c, str):
            s = c.replace(" ", "")
            for symbol, replacement in NOTE_SYMBOL_MAP.items():
                s = s.replace(symbol, replacement)
            components.append(s)
        else:
            # <i></i> or <sym></sym> other text markup within the string
            components.extend(text_tag2str_components(c))
    return components


def text_tag2str_recursive(tag: bs4.Tag, join_char: str = "") -> str:
    """Gets all string components from a <text> tag and joins them with join_char."""
    components = text_tag2str_components(tag)
    return join_char.join(components)


def tag2text(tag: bs4.Tag) -> Tuple[str, str]:
    """Takes the <Text> from a MuseScore file's header and returns its style and string."""
    sty_tag = tag.find("style")
    txt_tag = tag.find("text")
    style = sty_tag.string if sty_tag is not None else ""
    if txt_tag is None:
        txt = ""
    else:
        txt = text_tag2str(txt_tag)
    return txt, style


DEFAULT_THOROUGHBASS_SYMBOLS = {
    "0": "",
    "1": "bb",
    "2": "b",
    "3": "h",
    "4": "#",
    "5": "##",
    "6": "+",
    "7": "\\",
    "8": "/",
    "9": "",
    "10": "(",
    "11": ")",
    "12": "[",
    "13": "]",
    "14": "0",
    "15": "0+",
}

DEFAULT_THOROUGHBASS_BRACKETS = {
    "0": "",
    "1": "(",
    "2": ")",
    "3": "[",
    "4": "]",
    "5": "0",
    "6": "0+",
    "7": "0+",
    "8": "?",
    "9": "1",
    "10": "1+",
    "11": "1+",
}


@overload
def find_tag_get_string(
    parent_tag: bs4.Tag, tag_to_find: str, fallback: Literal[None]
) -> Tuple[Optional[bs4.Tag], Optional[str]]: ...


@overload
def find_tag_get_string(
    parent_tag: bs4.Tag, tag_to_find: str, fallback: Hashable
) -> Tuple[Optional[bs4.Tag], Optional[Hashable]]: ...


def find_tag_get_string(
    parent_tag: bs4.Tag, tag_to_find: str, fallback: Optional[Hashable] = None
) -> Tuple[Optional[bs4.Tag], Optional[Union[str, Hashable]]]:
    found = parent_tag.find(tag_to_find)
    if found is None:
        return None, fallback
    return found, str(found.string)


def get_thoroughbass_symbols(item_tag: bs4.Tag) -> Tuple[str, str]:
    """Returns the prefix and suffix of a <FiguredBassItem> tag if present, empty strings otherwise."""
    symbol_map = DEFAULT_THOROUGHBASS_SYMBOLS  # possibly allow for other mappings if need comes up
    prefix_tag, prefix = find_tag_get_string(item_tag, "prefix", fallback="")
    if prefix != "":
        prefix = symbol_map[prefix]
    suffix_tag, suffix = find_tag_get_string(item_tag, "suffix", fallback="")
    if suffix != "":
        suffix = symbol_map[suffix]
    return prefix, suffix


def thoroughbass_item(item_tag: bs4.Tag) -> str:
    """Turns a <FiguredBassItem> tag into a string by concatenating brackets, prefix, digit and suffix."""
    digit_tag, digit = find_tag_get_string(item_tag, "digit", fallback="")
    prefix, suffix = get_thoroughbass_symbols(item_tag)
    bracket_symbol_map = DEFAULT_THOROUGHBASS_BRACKETS  # possibly allow for other mappings if need comes up
    brackets_tag = item_tag.find("brackets")
    if brackets_tag:
        result = ""
        bracket_attributes = (
            "b0",
            "b1",
            "b2",
            "b3",
            "b4",
        )  # {'before_prefix', 'before_digit', 'after_digit', 'after_suffix', 'after_b3')
        components = (prefix, digit, suffix)
        for bracket_attribute, component in zip_longest(
            bracket_attributes, components, fillvalue=""
        ):
            bracket_code = brackets_tag[bracket_attribute]
            result += bracket_symbol_map[bracket_code] + component
    else:
        result = prefix + digit + suffix
    cont_tag, cont_value = find_tag_get_string(item_tag, "continuationLine", 0)
    continuation_line = (
        min(int(cont_value), 2) * "_"
    )  # more than two underscores result in the same behaviour as 2
    return result + continuation_line


def process_thoroughbass(
    thoroughbass_tag: bs4.Tag,
) -> Tuple[List[str], Optional[Fraction]]:
    """Turns a <FiguredBass> tag into a list of components strings, one per level, and duration."""
    ticks_tag = thoroughbass_tag.find("ticks")
    if ticks_tag is None:
        duration = None
    else:
        duration = Fraction(ticks_tag.string)
    components = []
    for item_tag in thoroughbass_tag.find_all("FiguredBassItem"):
        components.append(thoroughbass_item(item_tag))
    if len(components) == 0:
        text_tag, text = find_tag_get_string(thoroughbass_tag, "text")
        if text is not None:
            components = text.split("\n")
            # for level in text.split('\n'):
            #     begin, end = re.search('(_*)$', level).span()
            #     continuation_line_length = end - begin
            #     cont = 2 if continuation_line_length > 2 else continuation_line_length
            #     components.append((level, cont))
    return components, duration


@overload
def get_row_at_quarterbeat(
    df: pd.DataFrame, quarterbeat: Literal[None]
) -> pd.DataFrame: ...


@overload
def get_row_at_quarterbeat(
    df: pd.DataFrame, quarterbeat: float
) -> Optional[pd.Series]: ...


def get_row_at_quarterbeat(
    df: pd.DataFrame, quarterbeat: Optional[float] = None
) -> Optional[pd.Series]:
    """Returns the row of a DataFrame that is active at a given quarterbeat by interpreting subsequent intervals of
     the given dataframe's "quarterbeat" column as activation intervals. That is, the rows are interpreted as
     consecutive, non-overlapping events and the ``duration_qb`` column is not taken into account for computing the
     activation intervals. The last interval's right boundary is np.inf, so that all values higher than the latest
     event resolve to the latest event without needing to know the end of the piece.

    Args:
        df: DataFrame in which the column "quarterbeat" is monotonically increasing.
        quarterbeat:
            The position the active row for which will be returned. If the position does not exist because it's
            before the first event, None is returned.
            If None is passed (default), the whole dataframe is returned.

    Returns:
        The row of the dataframe.
    """
    df = df[df.quarterbeats.notna()].sort_values("quarterbeats")
    # ToDo Systematically use quarterbeats_all_endings for excerpt creation
    df.duration_qb = (
        (df.quarterbeats.shift(-1) - df.quarterbeats).astype(float).fillna(np.inf)
    )
    df = replace_index_by_intervals(df)
    if quarterbeat is None:
        return df
    try:
        result = df.loc[quarterbeat]
    except KeyError:
        return
    if isinstance(result, pd.DataFrame) and len(result) > 1:
        raise ValueError(
            f"More than one row active at quarterbeat {quarterbeat}:\n{result}"
        )
    return result
