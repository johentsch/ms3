from fractions import Fraction as frac

import bs4  # python -m pip install beautifulsoup4 lxml
import pandas as pd

from .bs4_measures import MeasureList
from .logger import get_logger

class _MSCX_bs4:
    """ This sister class implements MSCX's methods for a score parsed with beautifulsoup4.

    """

    def __init__(self, mscx_src, logger_name='_MSCX_bs4', level=None):
        self.logger = get_logger(logger_name, level=level)
        self._measures, self._events, self._notes = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.mscx_src = mscx_src
        self.first_mc = 1
        self.measure_nodes = {}


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
        """ Converts the score into the three DataFrame self.measures, self.events, and self.notes
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
                            if grace:
                                event['gracenote'] = grace.name
                            else:
                                event['duration'] = bs4_chord_duration(event_node, duration_multiplier)
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
                            event['duration'] = bs4_rest_duration(event_node, duration_multiplier)
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
                            if event_name == 'Chord':  # <Chord> children are stored as note_events
                                event['scalar'] = duration_multiplier
                                for attr, value in event_node.attrs.items():
                                    event[f"Chord:{attr}"] = value
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
        self.ml = self._make_measure_list()
        return self.ml.ml


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
        elif not ignore_empty:
            if c == '\n':
                info[name] = '∅'
            elif c is None:
                info[name] = '/'
            else:
                info[name] = c

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
            result += str(c).replace('"', '&quot;')
    result += closing_tag(node.name)
    return result


def bs4_to_mscx(soup):
    def format_node(node, indent):
        nxt_indent = indent + 2
        space = indent * ' '
        node_name = node.name
        # The following tags are exceptionally not abbreviated when empty,
        # so for instance you get <metaTag></metaTag> and not <metaTag/>
        if node_name in ['text', 'LayerTag', 'metaTag', 'trackName']:
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

