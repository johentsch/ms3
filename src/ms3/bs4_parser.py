from fractions import Fraction as frac
from collections import ChainMap # for merging dictionaries

import bs4  # python -m pip install beautifulsoup4 lxml
import pandas as pd
import numpy as np

from .bs4_measures import MeasureList
from .logger import get_logger

class _MSCX_bs4:
    """ This sister class implements MSCX's methods for a score parsed with beautifulsoup4.

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

    def __init__(self, mscx_src, logger_name='_MSCX_bs4', level=None):
        self.logger = get_logger(logger_name, level=level)
        self._measures, self._events, self._notes = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.mscx_src = mscx_src
        self.first_mc = 1
        self.measure_nodes = {}
        self._ml = None
        cols = ['mc', 'onset', 'duration', 'staff', 'voice', 'scalar', 'nominal_duration']
        self._nl, self._cl, self._rl = pd.DataFrame(), pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)


        with open(mscx_src, 'r') as file:
            self.soup = bs4.BeautifulSoup(file.read(), 'xml')

        self.version = self.soup.find('programVersion').string

        # Populate measure_nodes with one {mc: <Measure>} dictionary per staff.
        # The <Staff> nodes containing the music are siblings of <Part>
        # <Part> contains <Staff> nodes with staff information which is being ignored for now
        for staff in self.soup.find('Part').find_next_siblings('Staff'):
            staff_id = int(staff['id'])
            self.measure_nodes[staff_id] = {}
            for mc, measure in enumerate(staff.find_all('Measure'), start=self.first_mc):
                self.measure_nodes[staff_id][mc] = measure

        self.parse_measures()

    def parse_measures(self):
        """ Converts the score into the three DataFrame self._measures, self._events, and self._notes
        """
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
            # iterate through staves and collect information about each <Measure> node
            for staff_id, measure in zip(staff_ids, measure_stack):
                measure_info = {'mc': mc, 'staff': staff_id}
                measure_info.update(recurse_node(measure, exclude_children=['voice']))
                # iterate through <voice> tags and run a position counter
                voice_nodes = measure.find_all('voice', recursive=False)
                # measure_info['voices'] = len(voice_nodes)
                for voice_id, voice_node in enumerate(voice_nodes, start=1):
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

                        current_position += event['duration']

                measure_list.append(measure_info)
        col_order = ['mc', 'onset', 'event', 'duration', 'staff', 'voice', 'chord_id', 'gracenote', 'scalar', 'tpc',
                     'pitch']
        self._measures = sort_cols(pd.DataFrame(measure_list), col_order)
        self._events = sort_cols(pd.DataFrame(event_list), col_order)
        self._notes = sort_cols(pd.DataFrame(note_list), col_order)



    def output_mscx(self, filepath):

        with open(filepath, 'w') as file:
            file.write(bs4_to_mscx(self.soup))

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
        """A list of <chord> tags (all <note> tags come within one)."""
        self.make_standard_chordlist()
        return self._cl

    @property
    def cl(self):
        """Like property `chords` but without recomputing."""
        if len(self._cl) == 0:
            return self.chords
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
        nr = pd.concat([self.nl, self.rl]).astype({col: 'Int64' for col in ['tied', 'tpc', 'midi', 'chord_id']})
        return sort_note_list(nr.reset_index(drop=True))


    def make_standard_chordlist(self):
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



    def get_chords(self, staff=None, voice=None, lyrics=False, articulation=False, spanners=False, **kwargs):
        cols = {'nominal_duration': 'Chord/durationType',
                'lyrics': 'Chord/Lyrics/text',
                'syllabic': 'Chord/Lyrics/syllabic',
                'articulation': 'Chord/Articulation/subtype'}
        sel = self._events.event == 'Chord'
        if spanners:
            sel = sel | (self._events.event == 'Spanner')
        if staff:
            sel = sel & (self._events.staff == staff)
        if voice:
            sel = sel & self._events.voice == voice
        df = self.add_standard_cols(self._events[sel])
        df = df.astype({'chord_id': 'Int64' if spanners else int})
        df.rename(columns={v: k for k, v in cols.items() if v in df.columns}, inplace=True)
        df.loc[:, 'nominal_duration'] = df.nominal_duration.map(self.durations)
        main_cols = ['mc', 'mn', 'timesig', 'onset', 'staff', 'voice', 'duration', 'gracenote', 'nominal_duration', 'scalar',
                'volta', 'chord_id']
        if lyrics:
            main_cols.append('lyrics')
            if 'syllabic' in df:
                # turn the 'syllabic' column into the typical dashs
                sy = df.syllabic
                empty = pd.Series(np.nan, index=df.index)
                syl_start, syl_mid, syl_end = [empty.where(sy != which, '-').fillna('') for which in
                                               ['begin', 'middle', 'end']]
                lyrics_col = syl_end + syl_mid + df.lyrics + syl_mid + syl_start
            else:
                lyrics_col = df.lyrics
            df.loc[:, 'lyrics'] = lyrics_col
        if articulation:
            main_cols.append('articulation')
        for col in main_cols:
            if not col in df.columns:
                df[col] = np.nan
        additional_cols = []
        if spanners:
            spanner_ids = make_spanner_cols(df)
            if len(spanner_ids.columns) > 0:
                additional_cols.extend(spanner_ids.columns.to_list())
                df = pd.concat([df, spanner_ids], axis=1)
        for feature in kwargs.keys():
            additional_cols.extend([c for c in df.columns if feature in c])
        return df[main_cols + additional_cols]



    def get_harmonies(self, staff=None, harmony_type=None, positioning=False):
        """ Returns a list of harmony tags from the parsed score.

        Parameters
        ----------
        staff : :obj:`int`, optional
            Select harmonies from a given staff only. Pass `staff=1` for the upper staff.
        harmony_type : {0, 1, 2}, optional
            If MuseScore's harmony feature has been used, you can filter harmony types by passing
                0 for 'normal' chord labels only
                1 for Roman Numeral Analysis
                2 for Nashville Numbers
        positioning : :obj:`bool`, optional
            Set to True if you want to include information about how labels have been manually positioned.

        Returns
        -------

        """
        cols = {'harmony_type': 'Harmony/harmonyType',
                'label': 'Harmony/name',
                'nashville': 'Harmony/function',
                'root': 'Harmony/root',
                'base': 'Harmony/base',
                'leftParen': 'Harmony/leftParen',
                'rightParen': 'Harmony/rightParen'}
        main_cols = ['mc', 'mn', 'timesig', 'onset', 'staff', 'voice', 'root', 'label', 'base', 'leftParen', 'rightParen', 'harmony_type']
        sel = self._events.event == 'Harmony'
        if staff:
            sel = sel & self._events.staff == staff
        if harmony_type:
            if harmony_type == 0:
                sel = sel & self._events[cols['harmony_type']].isna()
            else:
                sel = sel & (pd.to_numeric(self._events[cols['harmony_type']]).astype('Int64') == harmony_type).fillna(False)
        df = self.add_standard_cols(self._events[sel]).dropna(axis=1, how='all')
        if len(df.index) == 0:
            return pd.DataFrame(columns=main_cols)
        df.rename(columns={v: k for k, v in cols.items() if v in df.columns}, inplace=True)
        if 'nashville' in df.columns:
            sel = df.nashville.notna()
            df.loc[sel, 'label'] = df.loc[sel, 'nashville'] + df.loc[sel, 'label'].replace('/', '')
            df.drop(columns='nashville', inplace=True)
        columns = [c for c in main_cols if c in df.columns]
        if positioning:
            additional_cols = {c: c[8:] for c in df.columns if c[:8] == 'Harmony/' if c[8:] not in main_cols}
            df.rename(columns=additional_cols, inplace=True)
            columns += list(additional_cols.values())
        return df[columns]


    def get_metadata(self):

        data = {}

        for tag in self.soup.find_all('metaTag'):
            tag_type = tag['name']
            tag_str = tag.string
            data[tag_type] = tag_str
        data['label_count'] = len(self.get_harmonies())
        data['TimeSig'] = dict(self.ml.loc[self.ml.timesig != self.ml.timesig.shift(), ['mc', 'timesig']].itertuples(index=False, name=None))
        data['KeySig']  = dict(self.ml.loc[self.ml.keysig != self.ml.keysig.shift(), ['mc', 'keysig']].itertuples(index=False, name=None))
        data['parts']   = {part.trackName.string: [staff['id'] for staff in part.find_all('Staff')] for part in self.soup.find_all('Part')}
        data['musescore'] = self.soup.find('programVersion').string
        return data






    def add_standard_cols(self, df):
        df =  df.merge(self.ml[['mc', 'mn', 'timesig', 'offset', 'volta']], on='mc', how='left')
        df.onset += df.offset
        return df[[col for col in df.columns if not col == 'offset']]


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
                    print(f"Spanner ending (type {spanner_type}{'' if subtype is None else ', subtype: ' + subtype }) could not be matched with a beginning.")
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
    has_tie = df[tie_col].fillna('').str.contains('Tie')
    new_col = pd.Series(np.nan, index=df.index, name='tied')
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
                            .replace('>', '&gt;')
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

