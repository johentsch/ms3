from .logger import get_logger, function_logger


class NoteList:
    """ Turns a _MSCX_bs4._notes DataFrame into a note list.

    Attributes
    ----------
    df : :obj:`pandas.DataFrame`
        The input DataFrame from _MSCX_bs4.raw_measures
    section_breaks : :obj:`bool`, default True
        By default, section breaks allow for several anacrusis measures within the piece (relevant for `offset` column)
        and make it possible to omit a repeat sign in the following bar (relevant for `next` column).
        Set to False if you want to ignore section breaks.
    secure : :obj:`bool`, default False
        By default, measure information from lower staves is considered to contain only redundant information.
        Set to True if you want to be warned about additional measure information from lower staves that is not taken into account.
    reset_index : :obj:`bool`, default True
        By default, the original index of `df` is replaced. Pass False to keep original index values.
    logger_name : :obj:`str`, optional
        If you have defined a logger, pass its name.
    level : {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}, optional
        Pass a level name for which (and above which) you want to see log records.

    cols : :obj:`dict`
        Dictionary of the relevant columns in `df` as present after the parse.
    ml : :obj:`pandas.DataFrame`
        The measure list in the making; the final result.
    volta_structure : :obj:`dict`
        Keys are first MCs of volta groups, values are dictionaries of {volta_no: [mc1, mc2 ...]}

    """

    def __init__(self, df, section_breaks=True, secure=False, reset_index=True, logger_name='MeasureList', level=None):
        self.logger = get_logger(logger_name, level=level)
        self.df = df

        self.make_nl()

    nl = b.merge(ml[['mc', 'timesig', 'offset']], on='mc')
    nl.rename(columns={'Note/pitch': 'midi', 'Note/tpc': 'tpc'}, inplace=True)
    nl.onset += nl.offset