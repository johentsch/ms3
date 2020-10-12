import sys, re

import pandas as pd

from .utils import decode_harmonies, load_tsv, resolve_dir
from .logger import get_logger
from .expand_dcml import expand_labels

class Annotations:
    """
    Class for storing, converting and manipulating annotation labels.
    """
    dcml_double_re = re.compile(r"""
                                    ^(?P<first>
                                      (\.?
                                        ((?P<globalkey>[a-gA-G](b*|\#*))\.)?
                                        ((?P<localkey>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\.)?
                                        ((?P<pedal>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\[)?
                                        (?P<chord>
                                            (?P<numeral>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))
                                            (?P<form>(%|o|\+|M|\+M))?
                                            (?P<figbass>(7|65|43|42|2|64|6))?
                                            (\((?P<changes>((\+|-|\^)?(b*|\#*)\d)+)\))?
                                            (/(?P<relativeroot>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)*))?
                                        )
                                        (?P<pedalend>\])?
                                      )?
                                      (?P<phraseend>(\\\\|\{|\}|\}\{)
                                      )?
                                     )
                                     (?P<second>
                                      (-
                                        ((?P<globalkey2>[a-gA-G](b*|\#*))\.)?
                                        ((?P<localkey2>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\.)?
                                        ((?P<pedal2>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\[)?
                                        (?P<chord2>
                                            (?P<numeral2>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))
                                            (?P<form2>(%|o|\+|M|\+M))?
                                            (?P<figbass2>(7|65|43|42|2|64|6))?
                                            (\((?P<changes2>((\+|-|\^)?(b*|\#*)\d)+)\))?
                                            (/(?P<relativeroot2>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)*))?
                                        )
                                        (?P<pedalend2>\])?
                                      )?
                                      (?P<phraseend2>(\\\\|\{|\}|\}\{)
                                      )?
                                     )?
                                    $
                                    """,
                            re.VERBOSE)

    dcml_re = re.compile(r"""^(\.?
                                ((?P<globalkey>[a-gA-G](b*|\#*))\.)?
                                ((?P<localkey>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\.)?
                                ((?P<pedal>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\[)?
                                (?P<chord>
                                    (?P<numeral>(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))
                                    (?P<form>(%|o|\+|M|\+M))?
                                    (?P<figbass>(7|65|43|42|2|64|6))?
                                    (\((?P<changes>((\+|-|\^)?(b*|\#*)\d)+)\))?
                                    (/(?P<relativeroot>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)*))?
                                )
                                (?P<pedalend>\])?
                              )?
                              (?P<phraseend>(\\\\|\{|\}|\}\{)
                              )?$
                            """,
                    re.VERBOSE)

    def __init__(self, tsv_path=None, df=None, index_col=None, sep='\t', infer_types={}, logger_name='Annotations', level=None, **kwargs):
        self.logger = get_logger(logger_name, level)
        self.regex_dict = infer_types
        self._expanded = None
        if df is not None:
            self.df = df.copy()
        else:
            assert tsv_path is not None, "Name a TSV file to be loaded."
            self.df = load_tsv(tsv_path, index_col=index_col, sep=sep, **kwargs)
        if 'offset' in self.df.columns:
            self.df.drop(columns='offset', inplace=True)
        self.infer_types()


    def n_labels(self):
        return len(self.df)


    @property
    def label_types(self):
        """ Returns the counts of the label_types as dict.
        """
        if 'label_type' in self.df.columns:
            return self.df.label_type.value_counts(dropna=False).to_dict()
        else:
            return {None: len(self.df)}


    @property
    def annotation_layers(self):
        layers = [col for col in ['staff', 'voice', 'label_type'] if col in self.df.columns]
        return self.n_labels(), self.df.groupby(layers).size()

    def __repr__(self):
        n, layers = self.annotation_layers
        return f"{n} labels:\n{layers.to_string()}"

    def get_labels(self, staff=None, voice=None, label_type=None, positioning=True, decode=False, drop=False, warnings=True):
        """ Returns an annotation label .

        Parameters
        ----------
        staff : :obj:`int`, optional
            Select harmonies from a given staff only. Pass `staff=1` for the upper staff.
        label_type : {0, 1, 2, 3, 'dcml', ...}, optional
            If MuseScore's harmony feature has been used, you can filter harmony types by passing
                0 for unrecognized strings
                1 for Roman Numeral Analysis
                2 for Nashville Numbers
                3 for encoded absolute chords
                'dcml' for labels from the DCML harmonic annotation standard
                ... self-defined types that have been added to self.regex_dict through the use of self.infer_types()
        positioning : :obj:`bool`, optional
            Set to True if you want to include information about how labels have been manually positioned.
        decode : :obj:`bool`, optional
            Set to True if you don't want to keep labels in their original form as encoded by MuseScore (with root and
            bass as TPC (tonal pitch class) where C = 14).
        drop : :obj:`bool`, optional
            Set to True to delete the returned labels from this object.
        warnings : :obj:`bool`, optional
            Set to False to suppress warnings about non-existent label_types.

        Returns
        -------

        """
        sel = pd.Series(True, index=self.df.index)
        if staff is not None:
            sel = sel & (self.df.staff == staff)
        if voice is not None:
            sel = sel & (self.df.voice == voice)
        if label_type is not None and 'label_type' in self.df.columns:
            label_type = self._treat_label_type_param(label_type, warnings=warnings)
            sel = sel & self.df.label_type.isin(label_type)
            # if the column contains strings and NaN:
            # (pd.to_numeric(self.df['label_type']).astype('Int64') == label_type).fillna(False)
        res = self.df[sel].copy()
        if not positioning:
            pos_cols = [c for c in ['minDistance',  'offset', 'offset:x', 'offset:y'] if c in res.columns]
            res.drop(columns=pos_cols, inplace=True)
        if drop:
            self.df = self.df[~sel]
        if decode:
            res = decode_harmonies(res)
        return res


    def expand_dcml(self, drop_others=True, warn_about_others=True):
        """ Expands all labels where the label_type has been inferred as 'dcml' and stores the DataFrame in self._expanded.

        Parameters
        ----------
        drop_others : :obj:`bool`, optional
            Set to False if you want to keep labels in the expanded DataFrame which have not label_type 'dcml'.
        warn_about_others : :obj:`bool`, optional
            Set to False to suppress warnings about labels that have not label_type 'dcml'.
            Is automatically set to False if ``drop_others`` is set to False.

        Returns
        -------
        :obj:`pandas.DataFrame`
            Expanded DCML labels
        """
        if 'dcml' not in self.regex_dict:
            self.regex_dict = dict(dcml=self.dcml_double_re, **self.regex_dict)
            self.infer_types()
        sel = self.df.label_type == 'dcml'
        if not sel.any():
            self.logger.warning(f"Score does not contain any DCML harmonic annotations.")
            return
        if not drop_others:
            warn_about_others = False
        if warn_about_others and (~sel).any():
            self.logger.warning(f"Score contains {(~sel).sum()} labels that don't (and {sel.sum()} that do) match the DCML standard:\n{decode_harmonies(self.df[~sel])[['mc', 'mn', 'label', 'label_type']].to_string()}")
        df = self.df[sel]
        try:
            exp = expand_labels(df, column='label', regex=self.dcml_re, chord_tones=True, logger=self.logger)
            if drop_others:
                self._expanded = exp
            else:
                df = self.df.copy()
                df.loc[sel, exp.df.columns] = exp
        except:
            self.logger.error(f"Expanding labels failed with the following error:\n{sys.exc_info()[1]}")
        return self._expanded


    @property
    def expanded(self):
        if self._expanded is None:
            return self.expand_dcml()
        return self._expanded



    def infer_types(self, regex_dict=None):
        if regex_dict is not None:
            self.regex_dict = regex_dict
        if 'label_type' in self.df.columns:
            self.df.label_type.fillna(0, inplace=True)
            self.df.loc[~self.df.label_type.isin([0, 1, 2, 3, '0', '1', '2', '3']), 'label_type'] = 0
        else:
            self.df['label_type'] = pd.Series(0, index=self.df.index, dtype='object')
        if 'nashville' in self.df.columns:
            self.df.loc[self.df.nashville.notna(), 'label_type'] = 2
        if 'root' in self.df.columns:
            self.df.loc[self.df.root.notna(), 'label_type'] = 3
        for name, regex in self.regex_dict.items():
            sel = self.df.label_type == 0
            mtch = self.df[sel].label.str.match(regex)
            self.df.loc[sel & mtch, 'label_type'] = name


    def store_tsv(self, tsv_path, staff=None, voice=None, label_type=None, positioning=True, decode=False, sep='\t', index=False, **kwargs):
        df = self.get_labels(staff=staff, voice=voice, label_type=label_type, positioning=positioning, decode=decode)
        if decode and 'label_type' in df.columns:
            df.drop(columns='label_type', inplace=True)
        df.to_csv(resolve_dir(tsv_path), sep=sep, index=index, **kwargs)
        self.logger.info(f"{len(df)} labels written to {tsv_path}.")
        return True


    def _treat_label_type_param(self, label_type, warnings=True):
        if label_type is None:
            return None
        all_types = {str(k): k for k in self.label_types.keys()}
        if isinstance(label_type, int) or isinstance(label_type, str):
            label_type = [label_type]
        lt = [str(t) for t in label_type]
        if warnings:
            not_found = [t for t in lt if t not in all_types]
            if len(not_found) > 0:
                plural = len(not_found) > 1
                plural_s = 's' if plural else ''
                self.logger.warning(
                    f"No labels found with {'these' if plural else 'this'} label{plural_s} label_type{plural_s}: {', '.join(not_found)}")
        return [all_types[t] for t in lt if t in all_types]