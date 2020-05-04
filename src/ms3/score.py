import os
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
    def events(self):
        return self.parsed.events

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
            for i, measure in enumerate(staff.find_all('Measure'), start=first_mc):
                self.measure_nodes[staff_id][i] = measure

        self.parse_measures()

    def parse_measures(self):

        def recurse_node(node, prepend=None):
            info = {}
            name = node.name if prepend is None else f"{prepend}/{node.name}"
            for attr, value in node.attrs.items():
                info[f"{name}:{attr}"] = value
            sub_nodes = node.find_all(recursive=False)
            if len(sub_nodes) > 0:
                for sn in sub_nodes:
                    info.update({f"{name}/{k}": v for k, v in recurse_node(sn).items()})
            else:
                value = node.string
                info[name] = '∅' if value in ['', '\n'] else value
            return info

        events = []
        staff_ids = tuple(self.measure_nodes.keys())
        for mc, measure_stack in enumerate(
                                    zip(
                                        *[[measure_node for measure_node in measure_dict.values()] for measure_dict in self.measure_nodes.values()]
                                    ),
                                 start=self.first_mc):
            for staff_id, measure in zip(staff_ids, measure_stack):
                for voice_id, voice_node in enumerate(measure.find_all('voice', recursive=False), start=1):
                    current_position = frac(0)
                    duration_multiplier = 1
                    multiplier_stack = [1]

                    for event_node in voice_node.find_all(recursive=False):
                        event_name = event_node.name

                        if event_name == 'endTuplet':
                            if len(multiplier_stack) > 0:
                                duration_multiplier = multiplier_stack.pop()
                            else:
                                logging.warning("Error in the scalar_stack.")
                            continue

                        if event_name == 'Tuplet':
                            multiplier_stack.append(duration_multiplier)
                            duration_multiplier = duration_multiplier * frac(int(event_node.normalNotes.string), int(event_node.actualNotes.string))

                        event = {
                            'mc': mc,
                            'onset': current_position,
                            'event': event_name,
                            'duration': frac(0),
                            'scalar': duration_multiplier,
                            'staff': staff_id,
                            'voice': voice_id, }
                        for attr, value in event_node.attrs.items():
                            event[f"{event_name}:{attr}"] = value

                        if event_name == 'Chord':
                            grace = event_node.find(
                                ['grace4', 'grace4after', 'grace8', 'grace8after', 'grace16', 'grace16after', 'grace32',
                                 'grace32after', 'grace64', 'grace64after', 'appoggiatura', 'acciaccatura'])
                            if grace:
                                event['gracenote'] = grace.name
                            else:
                                event['duration'] = bs4_chord_duration(event_node, duration_multiplier)

                            for event_child in event_node.find_all(recursive=False):
                                child_name = event_child.name
                                if child_name != 'Note':
                                    event.update(recurse_node(event_child, prepend='Chord'))
                                    continue
                                note_event = dict(event)
                                note_event['event'] = 'Note'
                                for note_info in event_child.find_all(recursive=False):
                                    note_event.update(recurse_node(note_info, prepend='Note'))
                                events.append(note_event)
                        else:
                            for event_child in event_node.find_all(recursive=False):
                                event.update(recurse_node(event_child, prepend=event_name))

                            if event_name == 'Rest':
                                event['duration'] = bs4_rest_duration(event_node, duration_multiplier)
                            elif event_name == 'location':
                                event['duration'] = frac(event_node.fractions.string)

                            events.append(event)

                        current_position += event['duration']

        self.events = sort_cols(pd.DataFrame(events), ['mc', 'onset', 'event', 'duration', 'scalar', 'staff', 'voice', 'gracenote'])




def sort_cols(df, first_cols=[]):
    cols = df.columns
    column_order = [col for col in first_cols if col in cols] + sorted([col for col in cols if not col in first_cols])
    return df[column_order]


def bs4_chord_duration(node, duration_multiplier=1):
    DURATIONS = {"measure": frac(1),
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
        nominal_duration = DURATIONS[durationtype]
    dots = node.find('dots')
    dotmultiplier = sum([frac(1 / 2) ** i for i in range(int(dots.string) + 1)]) * duration_multiplier if dots else duration_multiplier
    return nominal_duration * dotmultiplier

def bs4_rest_duration(node, duration_multiplier=1):
    return bs4_chord_duration(node, duration_multiplier)