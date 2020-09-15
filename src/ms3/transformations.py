from collections.abc import Iterable

import pandas as pd



def fifths2acc(fifths):
    """ Returns accidentals for a stack of fifths that can be combined with a
        basic representation of the seven steps."""
    return abs(fifths // 7) * 'b' if fifths < 0 else fifths // 7 * '#'


def fifths2name(fifths, midi=None, ms=False):
    """ Return note name of a stack of fifths such that
       0 = C, -1 = F, -2 = Bb, 1 = G etc.
       Uses: map2elements(), fifths2str()

    Parameters
    ----------
    fifths : :obj:`int`
        Tonal pitch class to turn into a note name.
    midi : :obj:`int`
        In order to include the octave into the note name,
        pass the corresponding MIDI pitch.
    ms : :obj:`bool`, optional
        Pass True if ``fifths`` is a MuseScore TPC, i.e. C = 14
    """
    try:
        fifths = int(fifths)
    except:
        if isinstance(fifths, Iterable):
            return map2elements(fifths, fifths2name, ms=ms)
        return fifths

    if ms:
        fifths -= 14
    note_names = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
    name = fifths2str(fifths, note_names, inverted=True)
    if midi is not None:
        octave = midi2octave(midi, fifths)
        return f"{name}{octave}"
    return name


def fifths2pc(fifths):
    """ Turn a stack of fifths into a chromatic pitch class.
        Uses: map2elements()
    """
    try:
        fifths = int(fifths)
    except:
        if isinstance(fifths, Iterable):
            return map2elements(fifths, fifths2pc)
        return fifths

    return int(7 * fifths % 12)

 

def fifths2str(fifths, steps, inverted=False):
    """ Boiler plate used by fifths2-functions.
    """
    fifths += 1
    acc = fifths2acc(fifths)
    if inverted:
        return steps[fifths % 7] + acc
    return acc + steps[fifths % 7]


def map2elements(e, f, *args, **kwargs):
    """ If `e` is an iterable, `f` is applied to all elements.
    """
    if isinstance(e, Iterable) and not isinstance(e, str):
        return e.__class__(map2elements(x, f, *args, **kwargs) for x in e)
    return f(e, *args, **kwargs)


def midi2octave(midi, fifths=None):
    """ For a given MIDI pitch, calculate the octave. Middle octave = 4
        Uses: fifths2pc(), map2elements()

    Parameters
    ----------
    midi : :obj:`int`
        MIDI pitch (positive integer)
    tpc : :obj:`int`, optional
        To be precise, for some Tonal Pitch Classes, the octave deviates
        from the simple formula ``MIDI // 12 - 1``, e.g. for B# or Cb.
    """
    try:
        midi = int(midi)
    except:
        if isinstance(midi, Iterable):
            return map2elements(midi, midi2octave)
        return midi
    i = -1
    if fifths is not None:
        pc = fifths2pc(fifths)
        if midi % 12 != pc:
            logging.debug(f"midi2octave(): The Tonal Pitch Class {fifths} cannot be MIDI pitch {midi} ")
        if fifths in [
                    12, # B#
                    19, # B##
                    26, # B###
                    24, # A###
                  ]:
            i -= 1
        elif fifths in [
                    -7,  # Cb
                    -14, # Cbb
                    -21, # Cbbb
                    -19, # Dbbb
                  ]:
            i += 1
    return midi // 12 + i