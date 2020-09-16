from .utils import load_tsv
from .logger import get_logger

class Annotations:

    def __init__(self, tsv_path=None, df=None, index_col=None, sep='\t', logger_name='Harmonies', level=None):
        self.logger = get_logger(logger_name, level)
        if df is not None:
            self.df = df
        else:
            assert tsv_path is not None, "Name a TSV file to be loaded."
            self.df = load_tsv(tsv_path, index_col=index_col, sep=sep)


    def __repr__(self):
        return f"{len(self.df)} labels"