import os
import logging
from fractions import Fraction as frac

from bs4 import BeautifulSoup as bs4  # python -m pip install beautifulsoup4 lxml
import pandas as pd


class Score:
    """ Object representing a score.

    Attributes
    ----------

    """
    mscx_src: str

    def __init__(self):
        self.xml = None
        self.mscx_src = ''
        self.staff_nodes: {}

    def parse_mscx(self, mscx, parser='bs4'):
        self.xml = MSCX(mscx, parser)


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
    version :
    """

    def __init__(self, mscx_src=None, parser='bs4'):

        self.mscx_src = mscx_src
        if parser is not None:
            self.parser = parser

        assert os.path.isfile(self.mscx_src), f"{self.mscx_src} does not exist."

        implemented_parsers = ['bs4']
        if self.parser == 'bs4':
            self.parsed = _MSCX_bs4(self.mscx_src)
        else:
            raise NotImplementedError(f"Only the following parsers are available: {', '.join(implemented_parsers)}")

    @property
    def measures(self):
        return self.parsed.measures

    @property
    def events(self):
        return self.parsed.events

    @property
    def notes(self):
        return self.parsed.notes

    @property
    def version(self):
        return self.parsed.version


class _MSCX_bs4:
    """ This sister class implements MSCX's methods for a score parsed with beautifulsoup4.

    """

    def __init__(self, mscx_src, first_mc=1):
        self.events = pd.DataFrame()
        self.mscx_src = mscx_src
        self.first_mc = first_mc
        self.measure_nodes = {}

        with open(mscx_src, 'r') as file:
            self.soup = bs4(file.read(), 'xml')

        self.version = self.soup.find('programVersion').string

        # Populate measure_nodes with one {mc: <Measure>} dictionary per staff.
        # The <Staff> nodes containing the music are siblings of <Part>
        # <Part> contains <Staff> nodes with staff information which is being ignored for now
        for staff in self.soup.find('Part').find_next_siblings('Staff'):
            staff_id = int(staff['id'])
            self.measure_nodes[staff_id] = {}
            for mc, measure in enumerate(staff.find_all('Measure'), start=first_mc):
                self.measure_nodes[staff_id][mc] = measure

        self.parse_measures()


    def parse_measures(self):

        def recurse_node(node, prepend=None, exclude_children=None):
            info = {}
            if exclude_children is None:
                exclude_children = []
            name = node.name if prepend is None else prepend
            attr_prepend = name + ':'
            child_prepend = '' if prepend is None else prepend + '/'
            for attr, value in node.attrs.items():
                info[attr_prepend+attr] = value
            sub_nodes = {sn.name: sn for sn in node.find_all(recursive=False) if not sn.name in exclude_children}
            if len(sub_nodes) > 0:
                for sn_name, sn in sub_nodes.items():
                    info.update({child_prepend+k: v for k, v in  recurse_node(sn, prepend=sn_name).items()})
            else:
                value = node.string
                info[name] = '∅' if value in ['', '\n'] else str(value)
            return info

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
                for voice_id, voice_node in enumerate(measure.find_all('voice', recursive=False), start=1):
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
                            if grace:
                                event['gracenote'] = grace.name
                            else:
                                event['duration'] = bs4_chord_duration(event_node, duration_multiplier)
                            chord_info = dict(event)
                            note_event = dict(chord_info)
                            for chord_child in event_node.find_all(recursive=False):
                                child_name = chord_child.name
                                # These nodes describe the entire chord and go into event_list
                                # All others go into note_list
                                if child_name in grace_tags + ['dots', 'BeamMode', 'durationType']:
                                    event.update(recurse_node(chord_child, prepend='Chord/'+child_name))
                                else:
                                    if child_name != 'Note':
                                        note_event.update(recurse_node(chord_child, prepend=child_name))
                                        continue
                                    note_event.update(recurse_node(chord_child))
                                    note_list.append(note_event)
                                    note_event = dict(chord_info)
                            chord_id += 1
                        elif event_name == 'Rest':
                            event['duration'] = bs4_rest_duration(event_node, duration_multiplier)
                        elif event_name == 'location': # <location> tags move the position counter
                            event['duration'] = frac(event_node.fractions.string)
                        elif event_name == 'Tuplet':
                            multiplier_stack.append(duration_multiplier)
                            duration_multiplier = duration_multiplier * frac(int(event_node.normalNotes.string),
                                                                             int(event_node.actualNotes.string))
                        elif event_name == 'endTuplet':
                                duration_multiplier = multiplier_stack.pop()

                        # These nodes describe the entire measure and go into measure_list
                        # All others go into event_list
                        if event_name in ['TimeSig', 'KeySig']:
                            measure_info.update(recurse_node(event_node, prepend=f"voice/{event_name}"))
                        else:
                            event.update({'event': event_name})
                            if event_name == 'Chord': # <Chord> children are stored as note_events
                                event['scalar'] = duration_multiplier
                                for attr, value in event_node.attrs.items():
                                    event[f"Chord:{attr}"] = value
                            else:
                                event.update(recurse_node(event_node, prepend=event_name))
                            event_list.append(event)

                        current_position += event['duration']

                measure_list.append(measure_info)
        col_order = ['mc', 'onset', 'event', 'duration', 'scalar', 'staff', 'voice', 'gracenote']
        self.measures = sort_cols(pd.DataFrame(measure_list), col_order)
        self.events = sort_cols(pd.DataFrame(event_list), col_order)
        self.notes = sort_cols(pd.DataFrame(note_list), col_order)


def sort_cols(df, first_cols=None):
    if first_cols is None:
        first_cols = []
    cols = df.columns
    column_order = [col for col in first_cols if col in cols] + sorted([col for col in cols if not col in first_cols])
    return df[column_order]


def bs4_chord_duration(node, duration_multiplier=1):
    durations = {"measure": frac(1),
                 "breve": frac(2),  # in theory, of course, they could have length 1.5
                 "long": frac(4),  # and 3 as well and other values yet
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
    durationtype = node.find('durationType').string
    if durationtype == 'measure' and node.find('duration'):
        nominal_duration = frac(node.find('duration').string)
    else:
        nominal_duration = durations[durationtype]
    dots = node.find('dots')
    dotmultiplier = sum(
        [frac(1 / 2) ** i for i in range(int(dots.string) + 1)]) * duration_multiplier if dots else duration_multiplier
    return nominal_duration * dotmultiplier


def bs4_rest_duration(node, duration_multiplier=1):
    return bs4_chord_duration(node, duration_multiplier)
