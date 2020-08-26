from collections.abc import Iterable

import pandas as pd

def fifths2name(fifths):
    """ Return note name of a stack of fifths such that
       0 = C, -1 = F, -2 = Bb, 1 = G etc.
    """
    if pd.isnull(fifths):
        return fifths
    if isinstance(fifths, Iterable):
        return map2elements(fifths, fifths2name)
    note_names = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
    return fifths2str(fifths, note_names)


def map2elements(e, f, *args, **kwargs):
    """ If `e` is an iterable, `f` is applied to all elements.
    """
    if isinstance(e, Iterable):
        return e.__class__(map2elements(x, f, *args, **kwargs) for x in e)
    return f(e, *args, **kwargs)