import re
import logging
from fractions import Fraction as frac
from collections import defaultdict, ChainMap # for merging dictionaries

import bs4  # python -m pip install beautifulsoup4 lxml
import pandas as pd
import numpy as np

from .bs4_measures import MeasureList
from .logger import get_logger, function_logger
from .utils import fifths2name, ordinal_suffix, resolve_dir


class _MSCX_bs4:
    """ This sister class implements MSCX's methods for a score parsed with beautifulsoup4.

    Attributes
    ----------
    mscx_src : :obj:`str`
        Path to the uncompressed MuseScore 3 file (MSCX) to be parsed.
    logger_name : :obj:`str`, optional
        If you have defined a logger, pass its name.
    level : {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}, optional
        Pass a level name for which (and above which) you want to see log records.

    """

    durations = {"measure": frac(1),
                 "breve": frac(2),  # in theory, of course, they could have length 1.5
                 "long": frac(4),   # and 3 as well and other values yet
                 "whole": frac(1),
                 "half": frac(1 / 2),
                 "quarter": frac(1 / 4),
                 "eighth": frac(1 / 8),
                 "16th": frac(1 / 16),
                 "32nd": frac(1 / 32),
                 "64th": frac(1 / 64),
                 "128th": frac(1 / 128),
                 "256th": frac(1 / 256),
                 "512th": frac(1 / 512), }

    def __init__(self, mscx_src, read_only=False, logger_name='_MSCX_bs4', level=None):
        self.logger = get_logger(logger_name, level=level)
        self.soup = None
        self.metadata = None
        self._measures, self._events, self._notes = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.mscx_src = mscx_src
        self.read_only = read_only
        self.first_mc = 1
        self.measure_nodes = {}
        self.tags = {} # only used if not self.read_only
        self.has_annotations = False
        self._ml = None
        cols = ['mc', 'onset', 'duration', 'staff', 'voice', 'scalar', 'nominal_duration']
        self._nl, self._cl, self._rl, self._nrl = pd.DataFrame(), pd.DataFrame(columns=cols), pd.DataFrame(
            columns=cols), pd.DataFrame(columns=cols)

        self.parse_measures()



    def parse_mscx(self):
        """ Load the XML structure from the score in self.mscx_src and store references to staves and measures.
        """
        assert self.mscx_src is not None, "No MSCX file specified." \
                                          ""
        with open(self.mscx_src, 'r') as file:
            self.soup = bs4.BeautifulSoup(file.read(), 'xml')

        if self.version[0] != '3':
            # self.logger.exception(f"Cannot parse MuseScore {self.version} file.")
            raise ValueError(f"Cannot parse MuseScore {self.version} file.")

        # Populate measure_nodes with one {mc: <Measure>} dictionary per staff.
        # The <Staff> nodes containing the music are siblings of <Part>
        # <Part> contains <Staff> nodes with staff information which is being ignored for now
        for staff in self.soup.find('Part').find_next_siblings('Staff'):
            staff_id = int(staff['id'])
            self.measure_nodes[staff_id] = {}
            for mc, measure in enumerate(staff.find_all('Measure'), start=self.first_mc):
                self.measure_nodes[staff_id][mc] = measure


    def parse_measures(self):
        """ Converts the score into the three DataFrame self._measures, self._events, and self._notes
        """
        if self.soup is None:
            self.parse_mscx()
        grace_tags = ['grace4', 'grace4after', 'grace8', 'grace8after', 'grace16', 'grace16after', 'grace32',
                      'grace32after', 'grace64', 'grace64after', 'appoggiatura', 'acciaccatura']

        measure_list, event_list, note_list = [], [], []
        staff_ids = tuple(self.measure_nodes.keys())
        chord_id = 0
        # For every measure: bundle the <Measure> nodes from every staff
        for mc, measure_stack in enumerate(
                zip(
                    *[[measure_node for measure_node in measure_dict.values()] for measure_dict in
                      self.measure_nodes.values()]
                ),
                start=self.first_mc):
            if not self.read_only:
                self.tags[mc] = {}
            # iterate through staves and collect information about each <Measure> node
            for staff_id, measure in zip(staff_ids, measure_stack):
                if not self.read_only:
                    self.tags[mc][staff_id] = {}
                measure_info = {'mc': mc, 'staff': staff_id}
                measure_info.update(recurse_node(measure, exclude_children=['voice']))
                # iterate through <voice> tags and run a position counter
                voice_nodes = measure.find_all('voice', recursive=False)
                # measure_info['voices'] = len(voice_nodes)
                for voice_id, voice_node in enumerate(voice_nodes, start=1):
                    if not self.read_only:
                        self.tags[mc][staff_id][voice_id] = defaultdict(list)
                    current_position = frac(0)
                    duration_multiplier = 1
                    multiplier_stack = [1]
                    # iterate through children of <voice> which constitute the note level of one notational layer
                    for event_node in voice_node.find_all(recursive=False):
                        event_name = event_node.name

                        event = {
                            'mc': mc,
                            'staff': staff_id,
                            'voice': voice_id,
                            'onset': current_position,
                            'duration': frac(0)}

                        if event_name == 'Chord':
                            event['chord_id'] = chord_id
                            grace = event_node.find(grace_tags)
                            dur, dot_multiplier = bs4_chord_duration(event_node, duration_multiplier)
                            if grace:
                                event['gracenote'] = grace.name
                            else:
                                event['duration'] = dur
                            chord_info = dict(event)
                            note_event = dict(chord_info)
                            for chord_child in event_node.find_all(recursive=False):
                                if chord_child.name == 'Note':
                                    note_event.update(recurse_node(chord_child, prepend=chord_child.name))
                                    note_list.append(note_event)
                                    note_event = dict(chord_info)
                                else:
                                    event.update(recurse_node(chord_child, prepend='Chord/' + chord_child.name))
                            chord_id += 1
                        elif event_name == 'Rest':
                            event['duration'], dot_multiplier = bs4_rest_duration(event_node, duration_multiplier)
                        elif event_name == 'location':  # <location> tags move the position counter
                            event['duration'] = frac(event_node.fractions.string)
                        elif event_name == 'Tuplet':
                            multiplier_stack.append(duration_multiplier)
                            duration_multiplier = duration_multiplier * frac(int(event_node.normalNotes.string),
                                                                             int(event_node.actualNotes.string))
                        elif event_name == 'endTuplet':
                            duration_multiplier = multiplier_stack.pop()

                        # These nodes describe the entire measure and go into measure_list
                        # All others go into event_list
                        if event_name in ['TimeSig', 'KeySig', 'BarLine'] or (
                                event_name == 'Spanner' and 'type' in event_node.attrs and event_node.attrs[
                            'type'] == 'Volta'):
                            measure_info.update(recurse_node(event_node, prepend=f"voice/{event_name}"))
                        else:
                            event.update({'event': event_name})
                            if event_name == 'Chord':
                                event['scalar'] = duration_multiplier * dot_multiplier
                                for attr, value in event_node.attrs.items():
                                    event[f"Chord:{attr}"] = value
                            elif event_name == 'Rest':
                                event['scalar'] = duration_multiplier * dot_multiplier
                                event.update(recurse_node(event_node, prepend=event_name))
                            else:
                                event.update(recurse_node(event_node, prepend=event_name))
                            event_list.append(event)

                        if not self.read_only:
                            remember = {'name': event_name,
                                        'duration': event['duration'],
                                        'tag': event_node, }
                            position = event['onset']
                            if event_name == 'location' and event['duration'] < 0:
                                # this is a backwards pointer: store it where it points to for easy deletion
                                position += event['duration']
                            self.tags[mc][staff_id][voice_id][position].append(remember)

                        current_position += event['duration']

                measure_list.append(measure_info)
        col_order = ['mc', 'onset', 'event', 'duration', 'staff', 'voice', 'chord_id', 'gracenote', 'scalar', 'tpc',
                     'pitch']
        self._measures = sort_cols(pd.DataFrame(measure_list), col_order)
        self._events = sort_cols(pd.DataFrame(event_list), col_order)
        if 'chord_id' in self._events.columns:
            self._events.chord_id = self._events.chord_id.astype('Int64')
        self._notes = sort_cols(pd.DataFrame(note_list), col_order)
        if len(self._events) == 0:
            self.logger.warning("Empty score?")
        elif 'Harmony' in self._events.event.values:
            self.has_annotations = True
        self.metadata = self._get_metadata()




    def store_mscx(self, filepath):

        with open(resolve_dir(filepath), 'w') as file:
            file.write(bs4_to_mscx(self.soup))
        self.logger.info(f"Score written to {filepath}.")
        return True

    def _make_measure_list(self, section_breaks=True, secure=False, reset_index=True, logger_name=None):
        """ Regenerate the measure list from the parsed score with advanced options."""
        ln = self.logger.name if logger_name is None else logger_name
        return MeasureList(self._measures, section_breaks=section_breaks, secure=secure, reset_index=reset_index, logger_name=ln)

    @property
    def measures(self):
        """ Retrieve a standard measure list from the parsed score.
        """
        self._ml = self._make_measure_list()
        return self._ml.ml


    @property
    def ml(self):
        """Like property `measures` but without recomputing."""
        if self._ml is None:
            return self.measures
        return self._ml.ml

    @property
    def chords(self):
        """A list of <chord> tags (all <note> tags come within one) and attached score information such as
            lyrics, dynamics, articulations, slurs, etc."""
        return self.get_chords()

    @property
    def cl(self):
        """Getting self._cl but without recomputing."""
        if len(self._cl) == 0:
            self.make_standard_chordlist()
        return self._cl

    @property
    def notes(self):
        """A list of all notes with their features."""
        self.make_standard_notelist()
        return self._nl

    @property
    def nl(self):
        """Like property `notes` but without recomputing."""
        if len(self._nl) == 0:
            return self.notes
        return self._nl

    @property
    def rests(self):
        """A list of all rests with their features."""
        self.make_standard_restlist()
        return self._rl

    @property
    def rl(self):
        """Like property `rests` but without recomputing."""
        if len(self._rl) == 0:
            return self.rests
        return self._rl

    @property
    def notes_and_rests(self):
        """Get a combination of properties `notes` and `rests`"""
        if len(self._nrl) == 0:
            nr = pd.concat([self.nl, self.rl]).astype({col: 'Int64' for col in ['tied', 'tpc', 'midi', 'chord_id']})
            self._nrl = sort_note_list(nr.reset_index(drop=True))
        return self._nrl


    def make_standard_chordlist(self):
        """ This chord list has chords only as opposed to the one yielded by selr.get_chords()"""
        self._cl = self.add_standard_cols(self._events[self._events.event == 'Chord'])
        self._cl = self._cl.astype({'chord_id': int})
        self._cl.rename(columns={'Chord/durationType': 'nominal_duration'}, inplace=True)
        self._cl.loc[:, 'nominal_duration'] = self._cl.nominal_duration.map(self.durations)
        cols = ['mc', 'mn', 'timesig', 'onset', 'staff', 'voice', 'duration', 'gracenote', 'nominal_duration', 'scalar', 'volta', 'chord_id']
        for col in cols:
            if not col in self._cl.columns:
                self._cl[col] = np.nan
        self._cl = self._cl[cols]



    def make_standard_restlist(self):
        self._rl = self.add_standard_cols(self._events[self._events.event == 'Rest'])
        if len(self._rl) == 0:
             return
        self._rl = self._rl.rename(columns={'Rest/durationType': 'nominal_duration'})
        self._rl.loc[:, 'nominal_duration'] = self._rl.nominal_duration.map(self.durations)
        cols = ['mc', 'mn', 'timesig', 'onset', 'staff', 'voice', 'duration', 'nominal_duration', 'scalar', 'volta']
        self._rl = self._rl[cols].reset_index(drop=True)


    def make_standard_notelist(self):
        cols = {'midi': 'Note/pitch',
                'tpc': 'Note/tpc',
                }
        self._nl = self.add_standard_cols(self._notes)
        self._nl.rename(columns={v: k for k, v in cols.items()}, inplace=True)
        self._nl = self._nl.astype({'midi': int, 'tpc': int})
        self._nl.tpc -= 14
        self._nl = self._nl.merge(self.cl[['chord_id', 'nominal_duration', 'scalar']], on='chord_id')
        tie_cols = ['Note/Spanner:type', 'Note/Spanner/next/location', 'Note/Spanner/prev/location']
        self._nl['tied'] = make_tied_col(self._notes, *tie_cols)

        final_cols = [col for col in ['mc', 'mn', 'timesig', 'onset', 'staff', 'voice', 'duration', 'gracenote', 'nominal_duration',
                                'scalar', 'tied', 'tpc', 'midi', 'volta', 'chord_id'] if col in self._nl.columns]
        self._nl = sort_note_list(self._nl[final_cols])



    def get_chords(self, staff=None, voice=None, mode='auto', lyrics=False, staff_text=False, dynamics=False, articulation=False, spanners=False, **kwargs):
        """ Returns a DataFrame with the score's chords (groups of simultaneous notes in the same layer).
            Such a list is needed for extracting certain types of information which is attached to chords rather than notes.

        Parameters
        ----------
        staff : :obj:`int`
            Get information from a particular staff only (1 = upper staff)
        voice : :obj:`int`
            Get information from a particular voice only (1 = only the first layer of every staff)
        mode : {'auto', 'all', 'strict'}, optional
            Defaults to 'auto', meaning that those aspects are automatically included that occur in the score; the resulting
                DataFrame has no empty columns except for those parameters that are set to True.
            'all': Columns for all aspects are created, even if they don't occur in the score (e.g. lyrics).
            'strict': Create columns for exactly those parameters that are set to True, regardless which aspects occur in the score.
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

        """
        cols = {'nominal_duration': 'Chord/durationType',
                'lyrics': 'Chord/Lyrics/text',
                'syllabic': 'Chord/Lyrics/syllabic',
                'articulation': 'Chord/Articulation/subtype',
                'dynamics': 'Dynamic/subtype'}
        sel = self._events.event == 'Chord'
        aspects = ['lyrics', 'staff_text', 'dynamics', 'articulation', 'spanners']
        if mode == 'all':
            params = {p: True for p in aspects}
        else:
            l = locals()
            params = {p: l[p] for p in aspects}
        spanner_sel = self._events.event == 'Spanner'
        staff_text_sel = self._events.event == 'StaffText'
        dynamics_sel = self._events.event == 'Dynamic'
        if mode == 'auto':
            if not params['spanners'] and spanner_sel.any():
                params['spanners'] = True
            if not params['staff_text'] and staff_text_sel.any():
                params['staff_text'] = True
            if not params['dynamics'] and dynamics_sel.any():
                params['dynamics'] = True
        if params['spanners']:
            sel = sel | spanner_sel
        if params['staff_text']:
            sel = sel | staff_text_sel
        if params['dynamics']:
            sel = sel | dynamics_sel
        if staff:
            sel = sel & (self._events.staff == staff)
        if voice:
            sel = sel & self._events.voice == voice
        df = self.add_standard_cols(self._events[sel])
        df = df.astype({'chord_id': 'Int64' if df.chord_id.isna().any() else int})
        df.rename(columns={v: k for k, v in cols.items() if v in df.columns}, inplace=True)
        if mode == 'auto':
            if 'lyrics' in df.columns:
                params['lyrics'] = True
            if 'articulation' in df.columns:
                params['articulation'] = True
            if any(c in df.columns for c in ('Spanner:type', 'Chord/Spanner:type')):
                params['spanners'] = True
        df.loc[:, 'nominal_duration'] = df.nominal_duration.map(self.durations)
        main_cols = ['mc', 'mn', 'timesig', 'onset', 'staff', 'voice', 'duration', 'gracenote', 'nominal_duration', 'scalar',
                'volta', 'chord_id']
        if params['staff_text']:
            main_cols.append('staff_text')
            text_cols = ['StaffText/text', 'StaffText/text/b', 'StaffText/text/i']
            existing_cols = [c for c in text_cols if c in df.columns]
            if len(existing_cols) > 0:
                df.loc[:, 'staff_text'] = df[existing_cols].fillna('').sum(axis=1).replace('', np.nan)
            else:
                df.loc[:, 'staff_text'] = np.nan
        if params['lyrics']:
            main_cols.append('lyrics')
            if 'syllabic' in df:
                # turn the 'syllabic' column into the typical dashs
                sy = df.syllabic
                empty = pd.Series(np.nan, index=df.index)
                syl_start, syl_mid, syl_end = [empty.where(sy != which, '-').fillna('') for which in
                                               ['begin', 'middle', 'end']]
                lyrics_col = syl_end + syl_mid + df.lyrics + syl_mid + syl_start
            elif 'lyrics' in df:
                lyrics_col = df.lyrics
            else:
                lyrics_col = pd.Series(np.nan, index=df.index)
            df.loc[:, 'lyrics'] = lyrics_col
        if params['articulation']:
            main_cols.append('articulation')
        if params['dynamics']:
            main_cols.append('dynamics')
        for col in main_cols:
            if not col in df.columns:
                df[col] = np.nan
        additional_cols = []
        if params['spanners']:
            spanner_ids = make_spanner_cols(df, logger=self.logger)
            if len(spanner_ids.columns) > 0:
                additional_cols.extend(spanner_ids.columns.to_list())
                df = pd.concat([df, spanner_ids], axis=1)
        for feature in kwargs.keys():
            additional_cols.extend([c for c in df.columns if feature in c and c not in main_cols])
        return df[main_cols + additional_cols]



    def get_annotations(self):
        """ Returns a list of harmony tags from the parsed score.

        Returns
        -------

        """
        cols = {'label_type': 'Harmony/harmonyType',
                'label': 'Harmony/name',
                'nashville': 'Harmony/function',
                'root': 'Harmony/root',
                'base': 'Harmony/base',
                'leftParen': 'Harmony/leftParen',
                'rightParen': 'Harmony/rightParen'}
        std_cols = ['mc', 'mn', 'timesig', 'onset', 'staff', 'voice', 'label',]
        main_cols = std_cols + ['nashville', 'root', 'base', 'leftParen', 'rightParen', 'label_type']
        sel = self._events.event == 'Harmony'
        df = self.add_standard_cols(self._events[sel]).dropna(axis=1, how='all')
        if len(df.index) == 0:
            return pd.DataFrame(columns=std_cols)
        df.rename(columns={v: k for k, v in cols.items() if v in df.columns}, inplace=True)
        if 'label_type' in df.columns:
            df.label_type.fillna(0, inplace=True)
        columns = [c for c in main_cols if c in df.columns]
        additional_cols = {c: c[8:] for c in df.columns if c[:8] == 'Harmony/' if c[8:] not in main_cols}
        df.rename(columns=additional_cols, inplace=True)
        columns += list(additional_cols.values())
        return df[columns]


    def _get_metadata(self):
        assert self.soup is not None, "The file's XML needs to be loaded. Get metadata from the 'metadata' property or use the method make_writeable()"
        nav_str2str = lambda s: '' if s is None else str(s)
        data = {tag['name']: nav_str2str(tag.string) for tag in self.soup.find_all('metaTag')}
        last_measure = self.ml.iloc[-1]
        data['last_mc'] = int(last_measure.mc)
        data['last_mn'] = int(last_measure.mn)
        data['label_count'] = len(self.get_annotations())
        data['TimeSig'] = dict(self.ml.loc[self.ml.timesig != self.ml.timesig.shift(), ['mc', 'timesig']].itertuples(index=False, name=None))
        data['KeySig']  = dict(self.ml.loc[self.ml.keysig != self.ml.keysig.shift(), ['mc', 'keysig']].itertuples(index=False, name=None))
        first_label =  self.soup.find('Harmony')
        first_label_name = first_label.find('name') if first_label is not None else None
        if first_label_name is not None:
            m = re.match(r"^\.?([A-Ga-g](#+|b+)?)", first_label_name.string)
            if m is not None:
                data['annotated_key'] = m.group(1)
        staff_groups = self.nl.groupby('staff').midi
        ambitus = {t.staff: {'min_midi': t.midi, 'min_name': fifths2name(t.tpc, t.midi)}
                        for t in self.nl.loc[staff_groups.idxmin(), ['staff', 'tpc', 'midi', ]].itertuples(index=False)}
        for t in self.nl.loc[staff_groups.idxmax(), ['staff', 'tpc', 'midi', ]].itertuples(index=False):
            ambitus[t.staff]['max_midi'] = t.midi
            ambitus[t.staff]['max_name'] = fifths2name(t.tpc, t.midi)
        data['parts'] = {
            f"part_{i}" if part.trackName.string is None else str(part.trackName.string): {int(staff['id']): ambitus[int(staff['id'])] if int(staff['id']) in ambitus else {} for staff in
                                    part.find_all('Staff')} for i, part in enumerate(self.soup.find_all('Part'), 1)}
        ambitus_tuples = [tuple(amb_dict.values()) for amb_dict in ambitus.values()]
        mimi, mina, mami, mana = zip(*ambitus_tuples)
        min_midi, max_midi = min(mimi), max(mami)
        data['ambitus'] = {
                            'min_midi': min_midi,
                            'min_name': mina[mimi.index(min_midi)],
                            'max_midi': max_midi,
                            'max_name': mana[mami.index(max_midi)],
                          }
        data['musescore'] = self.version
        return data

    @property
    def version(self):
        return str(self.soup.find('programVersion').string)

    def add_standard_cols(self, df):
        df =  df.merge(self.ml[['mc', 'mn', 'timesig', 'mc_offset', 'volta']], on='mc', how='left')
        # df.onset += df.mc_offset
        return df[[col for col in df.columns if not col == 'mc_offset']]


    def delete_label(self, mc, staff, voice, onset):
        self.make_writeable()
        measure = self.tags[mc][staff][voice]
        if onset not in measure:
            self.logger.warning(f"MC {mc} has no onset {onset} in staff {staff}, voice {voice} where a harmony could be deleted.")
            return False
        elements = measure[onset]
        element_names = [e['name'] for e in elements]
        if not 'Harmony' in element_names:
            self.logger.warning(f"No harmony found at MC {mc}, onset {onset}, staff {staff}, voice {voice}.")
            return False
        if 'Chord' in element_names and 'location' in element_names:
            NotImplementedError(f"Check MC {mc}, onset {onset}, staff {staff}, voice {voice}:\n{elements}")
        onsets = sorted(measure)
        ix = onsets.index(onset)
        is_first = ix == 0
        is_last = ix == len(onsets) - 1
        delete_locations = True

        _, name = get_duration_event(elements)
        if name is None:
            # this label is not attached to a chord or rest and depends on <location> tags, i.e. <location> tags on
            # previous and subsequent onsets might have to be adapted
            n_locs = element_names.count('location')
            if is_first:
                all_dur_ev = sum(True for os, tag_list in measure.items() if get_duration_event(tag_list)[0] is not None)
                if all_dur_ev > 0:
                    assert n_locs > 0, f"""The label on MC {mc}, onset {onset}, staff {staff}, voice {voice} is the first onset
in a measure with subsequent durational events but has no <location> tag"""
                prv_n_locs = 0
                if not is_last:
                    delete_locations = False
            else:
                prv_onset = onsets[ix - 1]
                prv_elements = measure[prv_onset]
                prv_names = [e['name'] for e in prv_elements]
                prv_n_locs = prv_names.count('location')

            if n_locs == 0:
                # The current onset has no <location> tag. This presumes that it is the last onset in the measure.
                if not is_last:
                    raise NotImplementedError(
f"The label on MC {mc}, onset {onset}, staff {staff}, voice {voice} is not on the last onset but has no <location> tag.")
                if prv_n_locs > 0 and len(element_names) == 1:
                    # this harmony is the only event on the last onset, therefore the previous <location> tag can be deleted
                    if prv_names[-1] != 'location':
                        raise NotImplementedError(
f"Location tag is not the last element in MC {mc}, onset {onsets[ix-1]}, staff {staff}, voice {voice}.")
                    prv_elements[-1]['tag'].decompose()
                    del(measure[prv_onset][-1])
                    if len(measure[prv_onset]) == 0:
                        del(measure[prv_onset])
                    self.logger.debug(f"""Removed <location> tag in MC {mc}, onset {prv_onset}, staff {staff}, voice {voice}  
because it precedes the label to be deleted which is the voice's last onset, {onset}.""")

            elif n_locs == 1:
                if not is_last and not is_first:
                    # This presumes that the previous onset has at least one <location> tag which needs to be adapted
                    assert prv_n_locs > 0, f"""The label on MC {mc}, onset {onset}, staff {staff}, voice {voice} locs forward 
but the previous onset {prv_onset} has no <location> tag."""
                    if prv_names[-1] != 'location':
                        raise NotImplementedError(
    f"Location tag is not the last element in MC {mc}, onset {prv_onset}, staff {staff}, voice {voice}.")
                    cur_loc_dur = frac(elements[element_names.index('location')]['duration'])
                    prv_loc_dur = frac(prv_elements[-1]['duration'])
                    prv_loc_tag = prv_elements[-1]['tag']
                    new_loc_dur = prv_loc_dur + cur_loc_dur
                    prv_loc_tag.fractions.string = str(new_loc_dur)
                    measure[prv_onset][-1]['duration'] = new_loc_dur
                # else: proceed with deletion

            elif n_locs == 2:
                # this onset has two <location> tags meaning that if the next onset has a <location> tag, too, a second
                # one needs to be added
                assert prv_n_locs == 0, f"""The label on MC {mc}, onset {onset}, staff {staff}, voice {voice} has two 
<location> tags but the previous onset {prv_onset} has one, too."""
                if not is_last:
                    nxt_onset = onsets[ix + 1]
                    nxt_elements = measure[nxt_onset]
                    nxt_names = [e['name'] for e in nxt_elements]
                    nxt_n_locs = nxt_names.count('location')
                    _, nxt_name = get_duration_event(nxt_elements)
                    if nxt_name is None:
                        # The next onset is neither a chord nor a rest and therefore it needs to have exactly one
                        # location tag and a second one needs to be added based on the first one being deleted
                        nxt_is_last = ix + 1 == len(onsets) - 1
                        if not nxt_is_last:
                            assert nxt_n_locs == 1, f"""The label on MC {mc}, onset {onset}, staff {staff}, voice {voice} has two 
<location> tags but the next onset {nxt_onset} has {nxt_n_locs if nxt_n_locs > 1 else 
"none although it's neither a chord nor a rest, nor the last onset,"}."""
                            if nxt_names[-1] != 'location':
                                raise NotImplementedError(
f"Location tag is not the last element in MC {mc}, onset {nxt_onset}, staff {staff}, voice {voice}.")
                        if element_names[-1] != 'location':
                            raise NotImplementedError(
f"Location tag is not the last element in MC {mc}, onset {onset}, staff {staff}, voice {voice}.")
                        neg_loc_dur = frac(elements[element_names.index('location')]['duration'])
                        assert neg_loc_dur < 0, f"""Location tag in MC {mc}, onset {nxt_onset}, staff {staff}, voice {voice}
should be negative but is {neg_loc_dur}."""
                        pos_loc_dur = frac(elements[-1]['duration'])
                        new_loc_value = neg_loc_dur + pos_loc_dur
                        new_tag = self.new_location(new_loc_value)
                        nxt_elements[0]['tag'].insert_before(new_tag)
                        remember = {
                            'name': 'location',
                            'duration': new_loc_value,
                            'tag': new_tag
                        }
                        measure[nxt_onset].insert(0, remember)
                        self.logger.debug(f"""Added a new negative <location> tag to the subsequent onset {nxt_onset} in 
order to prepare the label deletion on MC {mc}, onset {onset}, staff {staff}, voice {voice}.""")
                # else: proceed with deletions because it has no effect on a subsequent onset
            else:
                raise NotImplementedError(
f"Too many location tags in MC {mc}, onset {prv_onset}, staff {staff}, voice {voice}.")
        # else: proceed with deletions because the <Harmony> is attached to a durational event (Rest or Chord)

        ##### Here the actual removal takes place.
        deletions = []
        delete_location = not (onset == 0 and not is_last)
        for i, e in enumerate(elements):
            if e['name'] == 'Harmony' or (e['name']  == 'location' and delete_location):
                e['tag'].decompose()
                deletions.append(i)
                self.logger.debug(f"<{e['name']}>-tag deleted in MC {mc}, onset {onset}, staff {staff}, voice {voice}.")
        for i in reversed(deletions):
            del(measure[onset][i])
        if len(measure[onset]) == 0:
            del(measure[onset])
        self.remove_empty_voices(mc, staff)
        return len(deletions) > 0


    def remove_empty_voices(self, mc, staff):
        voice_tags = self.measure_nodes[staff][mc].find_all('voice')
        dict_keys = sorted(self.tags[mc][staff])
        assert len(dict_keys) == len(voice_tags), f"""In MC {mc}, staff {staff}, there are {len(voice_tags)} <voice> tags
but the keys of _MSCX_bs4.tags[{mc}][{staff}] are {dict_keys}."""
        for key, tag in zip(reversed(dict_keys), reversed(voice_tags)):
            if len(self.tags[mc][staff][key]) == 0:
                tag.decompose()
                del(self.tags[mc][staff][key])
                self.logger.debug(f"Empty <voice> tag of voice {key} deleted in MC {mc}, staff {staff}.")
            else:
                # self.logger.debug(f"No superfluous <voice> tags in MC {mc}, staff {staff}.")
                break

    def make_writeable(self):
        if self.read_only:
            self.read_only = False
            prev_level = self.logger.getEffectiveLevel()
            self.logger.setLevel(logging.CRITICAL)
            # This is an automatic re-parse which does not have to be logged again
            self.parse_measures()
            self.logger.setLevel(prev_level)


    def add_label(self, label, mc, onset, staff=1, voice=1, **kwargs):
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

        onset = frac(onset)
        label_name = kwargs['decoded'] if 'decoded' in kwargs else label
        if voice not in self.tags[mc][staff]:
            # Adding label to an unused voice that has to be created
            existing_voices = self.measure_nodes[staff][mc].find_all('voice')
            n = len(existing_voices)
            if not voice <= n:
                last = existing_voices[-1]
                while voice > n:
                    last = self.new_tag('voice', after=last)
                    n += 1
                remember = self.insert_label(label=label, loc_before=None if onset == 0 else onset, within=last, **kwargs)
                self.tags[mc][staff][voice] = defaultdict(list)
                self.tags[mc][staff][voice][onset] = remember
                self.logger.debug(f"Added {label_name} to empty {voice}{ordinal_suffix(voice)} voice in MC {mc} at onset {onset}.")
                return True

        measure = self.tags[mc][staff][voice]
        if onset in measure:
            # There is an event (chord or rest) with the same onset to attach the label to
            elements = measure[onset]
            names = [e['name'] for e in elements]
            _, name = get_duration_event(elements)
            # insert before the first tag that is not in the tags_before_label list
            tags_before_label = ['BarLine', 'Dynamic', 'endTuplet', 'FiguredBass', 'KeySig', 'location', 'StaffText', 'Tempo', 'TimeSig']
            ix, before = next((i, elements[i]['tag']) for i in range(len(elements)) if elements[i]['name'] not in
                          tags_before_label )
            remember = self.insert_label(label=label, before=before, **kwargs)
            measure[onset].insert(ix, remember[0])
            old_names = list(names)
            names.insert(ix, 'Harmony')
            if name is None:
                self.logger.debug(f"""MC {mc}, onset {onset}, staff {staff}, voice {voice} had only these tags:
{old_names}\nAfter insertion: {names}""")
            else:
                self.logger.debug(f"Added {label_name} to {name} in MC {mc}, onset {onset}, staff {staff}, voice {voice}.")
            if 'Harmony' in old_names:
                self.logger.warning(
                    f"The chord in MC {mc}, onset {onset}, staff {staff}, voice {voice} was already carrying a label.")
            return True


        # There is no event to attach the label to
        ordered = list(reversed(sorted(measure)))
        prv_pos, nxt_pos = next((prv, nxt)
                                for prv, nxt
                                in zip(ordered + [None], [None] + ordered)
                                if prv < onset)
        assert prv_pos is not None, f"MC {mc} empty in staff {staff}, voice {voice}?"
        prv = measure[prv_pos]
        nxt = None if nxt_pos is None else measure[nxt_pos]
        prv_names = [e['name'] for e in prv]
        prv_ix, prv_name = get_duration_event(prv)
        if nxt is not None:
            nxt_names = [e['name'] for e in nxt]
            _, nxt_name = get_duration_event(nxt)
        # distinguish six cases: prv can be [event, location], nxt can be [event, location, None]
        if prv_ix is not None:
            # prv is event (chord or rest)
            if nxt is None:
                loc_after = prv_pos + prv[prv_ix]['duration'] - onset
                # i.e. the ending of the last event minus the onset
                remember = self.insert_label(label=label, loc_before= -loc_after, after=prv[prv_ix]['tag'], **kwargs)
                self.logger.debug(f"Added {label_name} at {loc_after} before the ending of MC {mc}'s last {prv_name}.")
            elif nxt_name is not None or nxt_names.count('location') == 0:
                # nxt is event (chord or rest) or something at onset 1 (after all sounding events, e.g. <Segment>)
                loc_after = nxt_pos - onset
                remember = self.insert_label(label=label, loc_before=-loc_after, loc_after=loc_after,
                                             after=prv[prv_ix]['tag'], **kwargs)
                self.logger.debug(f"Added {label_name} at {loc_after} before the {nxt_name} at onset {nxt_pos}.")
            else:
                # nxt has location tag(s)
                loc_ix = nxt_names.index('location')
                loc_dur = nxt[loc_ix]['duration']
                assert loc_dur < 0, f"Positive location tag at MC {mc}, when trying to insert {label_name} at onset {onset}: {nxt}"
                loc_before = loc_dur - nxt_pos + onset
                remember = self.insert_label(label=label, loc_before=loc_before, before=nxt[loc_ix]['tag'], **kwargs)
                loc_after = nxt_pos - onset
                nxt[loc_ix]['tag'].fractions.string = str(loc_after)
                nxt[loc_ix]['duration'] = loc_after
                nxt_name = ', '.join(f"<{e}>" for e in nxt_names if e != 'location')
                self.logger.debug(f"""Added {label_name} at {-loc_before} before the ending of the {prv_name} at onset {prv_pos}
and {loc_after} before the subsequent {nxt_name}.""")

        else:
            # prv has location tag(s)
            prv_name = ', '.join(f"<{e}>" for e in prv_names if e != 'location')
            loc_before = onset - prv_pos
            if nxt is None:
                remember = self.insert_label(label=label, loc_before=loc_before, after=prv[-1]['tag'], **kwargs)
                self.logger.debug(f"MC {mc}: Added {label_name} at {loc_before} after the previous {prv_name} at onset {prv_pos}.")
            else:
                try:
                    loc_ix = next(i for i, name in zip(range(len(prv_names) - 1, -1, -1), reversed(prv_names)) if name == 'location')
                except:
                    self.logger.error(f"MC {mc}, staff {staff}, voice {voice}: The tags of onset {prv_pos} should include a <location> tag.")
                    raise
                prv[loc_ix]['tag'].fractions.string = str(loc_before)
                prv[loc_ix]['duration'] = loc_before
                loc_after = nxt_pos - onset
                remember = self.insert_label(label=label, loc_after=loc_after, after=prv[loc_ix]['tag'], **kwargs)
                if nxt_name is None:
                    nxt_name = ', '.join(f"<{e}>" for e in nxt_names if e != 'location')
                self.logger.debug(f"""MC {mc}: Added {label_name} at {loc_before} after the previous {prv_name} at onset {prv_pos}
and {loc_after} before the subsequent {nxt_name}.""")

        if remember[0]['name'] == 'location':
            measure[prv_pos].append(remember[0])
            measure[onset] = remember[1:]
        else:
            measure[onset] = remember
        return True






    def insert_label(self, label, loc_before=None, before=None, loc_after=None, after=None, within=None, **kwargs):
        tag = self.new_label(label, before=before, after=after, within=within, **kwargs)
        remember = [dict(
                        name = 'Harmony',
                        duration = frac(0),
                        tag = tag
                    )]
        if loc_before is not None:
            location = self.new_location(loc_before)
            tag.insert_before(location)
            remember.insert(0, dict(
                        name = 'location',
                        duration =loc_before,
                        tag = location
                    ))
        if loc_after is not None:
            location = self.new_location(loc_after)
            tag.insert_after(location)
            remember.append(dict(
                        name = 'location',
                        duration =loc_after,
                        tag =location
                    ))
        return remember


    def new_label(self, label, label_type=None, after=None, before=None, within=None, root=None, base=None, leftParen=None, rightParen=None,  offset_x=None, offset_y=None, nashville=None, decoded=None):
        tag = self.new_tag('Harmony')
        if not pd.isnull(label_type):
            # only include <harmonyType> tag for label_type 1 and 2 (MuseScore's Nashville Numbers and Roman Numerals)
            if label_type in [1, 2, '1', '2']:
                _ = self.new_tag('harmonyType', value=label_type, within=tag)
        if not pd.isnull(leftParen):
            _ = self.new_tag('leftParen', within=tag)
        if not pd.isnull(root):
            _ = self.new_tag('root', value=root, within=tag)
        if not pd.isnull(label):
            _ = self.new_tag('name', value=label, within=tag)
        else:
            assert not pd.isnull(root), "Either label or root need to be specified."
        if not pd.isnull(nashville):
            _ = self.new_tag('function', value=nashville, within=tag)
        if not pd.isnull(base):
            _ = self.new_tag('base', value=base, within=tag)
        if not pd.isnull(offset_x) or not pd.isnull(offset_y):
            if pd.isnull(offset_x):
                offset_x = '0'
            if pd.isnull(offset_y):
                offset_y = '0'
            _ = self.new_tag('offset', attributes={'x': offset_x, 'y': offset_y}, within=tag)
        if not pd.isnull(rightParen):
            _ = self.new_tag('rightParen', within=tag)
        if after is not None:
            after.insert_after(tag)
        elif before is not None:
            before.insert_before(tag)
        elif within is not None:
            within.append(tag)
        return tag


    def new_location(self, location):
        tag = self.new_tag('location')
        _ = self.new_tag('fractions', value=str(location), within=tag)
        return tag



    def new_tag(self, name, value=None, attributes={}, after=None, before=None, within=None):
        tag = self.soup.new_tag(name)
        if value is not None:
            tag.string = str(value)
        for k, v in attributes.items():
            tag.attrs[k] = v

        if after is not None:
            after.insert_after(tag)
        elif before is not None:
            before.insert_before(tag)
        elif within is not None:
            within.append(tag)

        return tag


    def __getstate__(self):
        """When pickling, make object read-only, i.e. delete the BeautifulSoup object and all references to tags."""
        self.soup = None
        self.tags, self.measure_nodes = {}, {}
        self.read_only = True
        return self.__dict__




####################### END OF CLASS DEFINITION #######################


def get_duration_event(elements):
    """ Receives a list of dicts representing the events for a given onset and returns the index and name of
    the first event that has a duration, so either a Chord or a Rest."""
    names = [e['name'] for e in elements]
    if 'Chord' in names or 'Rest' in names:
        if 'Rest' in names:
            ix = names.index('Rest')
            name = '<Rest>'
        else:
            ix = next(i for i, d in enumerate(elements) if d['name'] == 'Chord' and d['duration'] > 0)
            name = '<Chord>'
        return ix, name
    return (None, None)


@function_logger
def make_spanner_cols(df, spanner_types=None):
    """ From a raw chord list as returned by ``get_chords(spanners=True)``
        create a DataFrame with Spanner IDs for all chords for all spanner
        types they are associated with.

    Parameters
    ----------
    spanner_types : :obj:`collection`
        If this parameter is passed, only the enlisted
        spanner types (e.g. ``Slur`` or ``Pedal``) are included.

    """

    cols = {
        'nxt_m': 'Spanner/next/location/measures',
        'nxt_f': 'Spanner/next/location/fractions',
        'prv_m': 'Spanner/prev/location/measures',
        'prv_f': 'Spanner/prev/location/fractions',
        'type':  'Spanner:type',
        }

    def get_spanner_ids(spanner_type, subtype=None):

        if spanner_type == 'Slur':
            f_cols = ['Chord/' + cols[c] for c in ['nxt_m', 'nxt_f', 'prv_m', 'prv_f']]
            type_col = 'Chord/' + cols['type']
        else:
            f_cols = [cols[c] for c in ['nxt_m', 'nxt_f', 'prv_m', 'prv_f']]
            type_col = cols['type']
        sel = df[type_col] == spanner_type
        subtype_col = f"Spanner/{spanner_type}/subtype"
        if subtype is None and subtype_col in df:
            subtypes = set(df.loc[df[subtype_col].notna(), subtype_col])
            results = [get_spanner_ids(spanner_type, st) for st in subtypes]
            return dict(ChainMap(*results))
        elif subtype:
            sel = sel & (df[subtype_col] == subtype)
        existing = [c for c in f_cols if c in df.columns]
        features = pd.DataFrame('', index=df.index, columns=f_cols)
        features.loc[sel, existing] = df.loc[sel, existing]
        features = features.apply(lambda col: col.fillna('').str.replace('-', ''))
        features.insert(0, 'staff', df.staff)

        current_id = -1
        column_name = spanner_type
        if subtype:
            column_name += ':' + subtype
        if spanner_type != 'Slur':
            staff_stacks = {i: {} for i in df.staff.unique()}
        else:
            features.insert(1, 'voice', df.voice)
            staff_stacks = {(i, v): {} for i in df.staff.unique() for v in range(1, 5)}

        def spanner_ids(row, distinguish_voices=False):
            nonlocal staff_stacks, current_id
            if distinguish_voices:
                staff, voice, nxt_m, nxt_f, prv_m, prv_f = row
                layer = (staff, voice)
            else:
                staff, nxt_m, nxt_f, prv_m, prv_f = row
                layer = staff
            if nxt_m != '' or nxt_f != '':
                current_id += 1
                staff_stacks[layer][(nxt_m, nxt_f)] = current_id
                return ', '.join(str(i) for i in staff_stacks[layer].values())

            val = ', '.join(str(i) for i in staff_stacks[layer].values())
            if prv_m != '' or prv_f != '':
                if len(staff_stacks[layer]) == 0 or (prv_m, prv_f) not in staff_stacks[layer]:
                    logger.warning(f"Spanner ending (type {spanner_type}{'' if subtype is None else ', subtype: ' + subtype }) could not be matched with a beginning at id {current_id}.")
                    return 'err'
                del(staff_stacks[layer][(prv_m, prv_f)])
            return val if val != '' else np.nan

        return {column_name: [spanner_ids(row, distinguish_voices=(spanner_type == 'Slur')) for row in features.values]}

    type_col = cols['type']
    types = list(set(df.loc[df[type_col].notna(), type_col])) if type_col in df.columns else []
    if 'Chord/' + type_col in df.columns:
        types += ['Slur']
    if spanner_types is not None:
        types = [t for t in types if t in spanner_types]
    list_of_dicts = [get_spanner_ids(t) for t in types]
    merged_dict = dict(ChainMap(*list_of_dicts))
    renaming = {
        'HairPin:1': 'decrescendo',
        'HairPin:3': 'diminuendo',
    }
    return pd.DataFrame(merged_dict, index=df.index).rename(columns=renaming)



def sort_note_list(df, mc_col='mc', onset_col='onset', midi_col='midi', duration_col='duration'):
    """Sort every measure (MC) by ['onset', 'midi', 'duration'] while leaving gracenotes' order (duration=0) intact"""
    is_grace = df[duration_col] == 0
    grace_ix = {k: v.to_numpy() for k, v in df[is_grace].groupby([mc_col, onset_col]).groups.items()}
    has_nan = df[midi_col].isna().any()
    if has_nan:
        df.loc[:, midi_col] = df[midi_col].fillna(1000)
    normal_ix = df.loc[~is_grace, [mc_col, onset_col, midi_col, duration_col]].groupby([mc_col, onset_col]).apply(
        lambda gr: gr.index[np.lexsort((gr.values[:, 3], gr.values[:, 2]))].to_numpy())
    sorted_ixs = [np.concatenate((grace_ix[mc_onset], ix)) if mc_onset in grace_ix else ix for mc_onset, ix in
                  normal_ix.iteritems()]
    df = df.reindex(np.concatenate(sorted_ixs)).reset_index(drop=True)
    if has_nan:
        df.loc[:, midi_col] = df[midi_col].replace({1000: np.nan}).astype('Int64')
    return df

def make_tied_col(df, tie_col, next_col, prev_col):
    new_col = pd.Series(np.nan, index=df.index, name='tied')
    if tie_col not in df.columns:
        return new_col
    has_tie = df[tie_col].fillna('').str.contains('Tie')
    if has_tie.sum() == 0:
        return new_col
    # merge all columns whose names start with `next_col` and `prev_col` respectively
    next_cols = [col for col in df.columns if col[:len(next_col)] == next_col]
    nxt = df[next_cols].notna().any(axis=1)
    prev_cols = [col for col in df.columns if col[:len(prev_col)] == prev_col]
    prv = df[prev_cols].notna().any(axis=1)
    new_col = new_col.where(~has_tie, 0).astype('Int64')
    tie_starts = has_tie & nxt
    tie_ends = has_tie & prv
    new_col.loc[tie_ends] -= 1
    new_col.loc[tie_starts] += 1
    return new_col


def safe_update(old, new):
    """ Update dict without replacing values.
    """
    existing = [k for k in new.keys() if k in old]
    if len(existing) > 0:
        new = dict(new)
        for ex in existing:
            old[ex] = f"{old[ex]} & {new[ex]}"
            del (new[ex])
    old.update(new)


def recurse_node(node, prepend=None, exclude_children=None):

    def tag_or_string(c, ignore_empty=False):
        nonlocal info, name
        if isinstance(c, bs4.element.Tag):
            if c.name not in exclude_children:
                safe_update(info, {child_prepend + k: v for k, v in recurse_node(c, prepend=c.name).items()})
        elif c not in ['\n', None]:
            info[name] = str(c)
        elif not ignore_empty:
            if c == '\n':
                info[name] = '∅'
            elif c is None:
                info[name] = '/'


    info = {}
    if exclude_children is None:
        exclude_children = []
    name = node.name if prepend is None else prepend
    attr_prepend = name + ':'
    child_prepend = '' if prepend is None else prepend + '/'
    for attr, value in node.attrs.items():
        info[attr_prepend + attr] = value
    children = tuple(node.children)
    if len(children) > 1:
        for c in children:
            tag_or_string(c, ignore_empty=True)
    elif len(children) == 1:
        tag_or_string(children[0], ignore_empty=False)
    else:
        info[name] = '/'
    return info


def sort_cols(df, first_cols=None):
    if first_cols is None:
        first_cols = []
    cols = df.columns
    column_order = [col for col in first_cols if col in cols] + sorted([col for col in cols if col not in first_cols])
    return df[column_order]


def bs4_chord_duration(node, duration_multiplier=1):

    durationtype = node.find('durationType').string
    if durationtype == 'measure' and node.find('duration'):
        nominal_duration = frac(node.find('duration').string)
    else:
        nominal_duration = _MSCX_bs4.durations[durationtype]
    dots = node.find('dots')
    dotmultiplier = sum([frac(1 / 2) ** i for i in range(int(dots.string) + 1)]) if dots else 1
    return nominal_duration * duration_multiplier * dotmultiplier, dotmultiplier


def bs4_rest_duration(node, duration_multiplier=1):
    return bs4_chord_duration(node, duration_multiplier)


def opening_tag(node, closed=False):
    closing = '/' if closed else ''
    result = f"<{node.name}"
    attributes = node.attrs.items()
    if len(attributes) > 0:
        result += ' ' + ' '.join(f'{attr}="{value}"' for attr, value in attributes)
    return f"{result}{closing}>"


def closing_tag(node_name):
    return f"</{node_name}>"


def make_oneliner(node):
    result = opening_tag(node)
    for c in node.children:
        if isinstance(c, bs4.element.Tag):
            result += make_oneliner(c)
        else:
            result += str(c).replace('"', '&quot;')\
                            .replace('<', '&lt;')\
                            .replace('>', '&gt;')\
                            .replace('&', '&amp;')
    result += closing_tag(node.name)
    return result


def bs4_to_mscx(soup):
    def format_node(node, indent):
        nxt_indent = indent + 2
        space = indent * ' '
        node_name = node.name
        # The following tags are exceptionally not abbreviated when empty,
        # so for instance you get <metaTag></metaTag> and not <metaTag/>
        if node_name in ['continueAt', 'endText', 'text', 'LayerTag', 'metaTag', 'trackName']:
            return f"{space}{make_oneliner(node)}\n"
        children = node.find_all(recursive=False)
        if len(children) > 0:
            result = f"{space}{opening_tag(node)}\n"
            result += ''.join(format_node(child, nxt_indent) for child in children)
            result += f"{nxt_indent * ' '}{closing_tag(node_name)}\n"
            return result
        if node.string == '\n':
            return f"{space}{opening_tag(node)}\n{nxt_indent * ' '}{closing_tag(node_name)}\n"
        if node.string is None:
            return f"{space}{opening_tag(node, closed=True)}\n"
        return f"{space}{make_oneliner(node)}\n"

    initial_tag = """<?xml version="1.0" encoding="UTF-8"?>\n"""
    first_tag = soup.find()
    return initial_tag + format_node(first_tag, indent=0)

