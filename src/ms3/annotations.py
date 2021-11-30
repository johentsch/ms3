import sys, re

import pandas as pd

from .utils import DCML_REGEX, DCML_DOUBLE_REGEX, decode_harmonies, is_any_row_equal, html2format, load_tsv, \
    map_dict, name2format, resolve_dir, rgb2format, column_order, update_cfg
from .logger import LoggedClass
from .expand_dcml import expand_labels

class Annotations(LoggedClass):
    """
    Class for storing, converting and manipulating annotation labels.
    """

    main_cols = ['label', 'mc', 'mc_onset', 'staff', 'voice']
    additional_cols = ['label_type', 'absolute_root', 'rootCase', 'absolute_base', 'leftParen', 'rightParen', 'offset_x', 'offset_y',
                       'nashville', 'decoded', 'color_name', 'color_html', 'color_r', 'color_g', 'color_b', 'color_a']

    def __init__(self, tsv_path=None, df=None, cols={}, index_col=None, sep='\t', mscx_obj=None, infer_types={}, read_only=False, logger_cfg={}, **kwargs):
        """

        Parameters
        ----------
        tsv_path
        df
        cols : :obj:`dict`, optional
            If one or several column names differ, pass a {NAME -> ACTUAL_NAME} dictionary, where NAME can be
            {'mc', 'mn', 'mc_onset', 'label', 'staff', 'voice', 'volta'}
        cols : :obj:`dict`
            If your columns don't have standard names, pass a {NAME -> ACTUAL_NAME} dictionary.
            Required columns: label, mc, mc_onset, staff, voice
            Additional columns: label_type, root, base, leftParen, rightParen, offset_x, offset_y, nashville, color_name,
            color_html, color_r, color_g, color_b, color_a
        index_col
        sep
        mscx_obj
        infer_types
        read_only
        logger_cfg : :obj:`dict`, optional
            The following options are available:
            'name': LOGGER_NAME -> by default the logger name is based on the parsed file(s)
            'level': {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}
            'file': PATH_TO_LOGFILE to store all log messages under the given path.
        kwargs :
        """
        super().__init__(subclass='Annotations', logger_cfg=logger_cfg)
        self.regex_dict = infer_types
        self._expanded = None
        self.changed = False
        self.read_only = read_only
        self.mscx_obj = mscx_obj
        self.volta_structure = None if mscx_obj is None else mscx_obj.volta_structure

        columns = self.main_cols + self.additional_cols
        self.cols = {c: c for c in columns}
        self.cols.update(update_cfg(cols, self.cols.keys(), logger=self.logger))


        if df is not None:
            self.df = df.copy()
        else:
            assert tsv_path is not None, "Name a TSV file to be loaded."
            self.df = load_tsv(tsv_path, index_col=index_col, sep=sep, **kwargs)
        sorting_cols = ['mc', 'mn', 'mc_onset', 'staff']
        sorting_cols = [self.cols[c] if c in self.cols else c for c in sorting_cols]
        sorting_cols = [c for c in sorting_cols if c in self.df.columns]
        self.df.sort_values(sorting_cols, inplace=True)
#         for col in ['label']:
#             assert self.cols[col] in self.df.columns, f"""The DataFrame has no column named '{self.cols[col]}'. Pass the column name as col={{'{col}'=col_name}}.
# Present column names are:\n{self.df.columns.to_list()}."""
        # if 'offset' in self.df.columns:
        #     self.df.drop(columns='offset', inplace=True)
        #self.cols = {k: v for k, v in self.cols.items() if k in self.main_cols + ['label_type'] or v in df.columns}
        self.infer_types()

    def add_initial_dots(self):
        if self.read_only:
            self.logger.warning(f"Cannot change labels attached to a score. Detach them first.")
            return
        label_col = self.cols['label']
        notes = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}
        add_dots = lambda s: '.' + s if s[0].lower() in notes else s
        self.df[label_col] = self.df[label_col].map(add_dots)


    def prepare_for_attaching(self, staff=None, voice=None, label_type=None, check_for_clashes=True):
        if self.mscx_obj is None:
            self.logger.warning(f"Annotations object not aware to which MSCX object it is attached.")
            return pd.DataFrame()
        df = self.df.copy()
        cols = list(df.columns)
        error = False
        staff_col = self.cols['staff']
        if staff_col not in cols:
            if staff is None:
                self.logger.warning(f"""Annotations don't have staff information. Pass the argument staff to decide where to attach them.
Available staves are {self.mscx_obj.staff_ids}""")
                error = True
        if staff is not None:
            df[staff_col] = staff
        if staff_col in cols and df[staff_col].isna().any():
            self.logger.warning(f"The following labels don't have staff information: {df[df.staff.isna()]}")
            error = True

        voice_col = self.cols['voice']
        if voice_col not in cols:
            if voice is None:
                self.logger.warning(f"""Annotations don't have voice information. Pass the argument voice to decide where to attach them.
Possible values are {{1, 2, 3, 4}}.""")
                error = True
        if voice is not None:
            df[voice_col] = voice
        if voice_col in cols and df[voice_col].isna().any():
            self.logger.warning(f"The following labels don't have voice information: {df[df.voice.isna()]}")
            error = True
        if error:
            return pd.DataFrame()

        if label_type is not None:
            df[self.cols['label_type']] = label_type

        if self.cols['mc'] not in cols:
            mn_col = self.cols['mn'] if 'mn' in self.cols else 'mn'
            if mn_col not in cols:
                self.logger.warning(f"Annotations are lacking 'mn' and 'mc' columns.")
                error = True
            else:
                inferred_positions = self.infer_mc_from_mn()
                if inferred_positions.isna().any().any():
                    self.logger.error(f"Measure counts and corresponding mc_onsets could not be successfully inferred.")
                    error = True
                else:
                    self.logger.info(f"Measure counts and corresponding mc_onsets successfully inferred.")
                    df.insert(df.columns.get_loc('mn'), 'mc', inferred_positions['mc'])
                    df.loc[:, 'mc_onset'] = inferred_positions['mc_onset']
                    cols.extend(['mc', 'mc_onset'])

        if self.cols['mc_onset' ] not in cols:
            self.logger.info("No 'mc_onset' column found. All labels will be inserted at mc_onset 0.")

        position_cols = ['mc', 'mc_onset', 'staff', 'voice']
        new_pos_cols = [self.cols[c] for c in position_cols]
        if all(c in df.columns for c in new_pos_cols):
            if check_for_clashes and self.mscx_obj.has_annotations:
                existing = self.mscx_obj.get_raw_labels()[position_cols]
                to_be_attached = df[new_pos_cols]
                clashes = is_any_row_equal(existing, to_be_attached)
                has_clashes = len(clashes) > 0
                if has_clashes:
                    self.logger.error(f"The following positions already have labels:\n{pd.DataFrame(clashes, columns=position_cols)}")
                    error = True
        elif check_for_clashes:
            self.logger.error(f"Check for clashes could not be performed because there are columns missing.")

        if error:
            return pd.DataFrame()
        return df


    def count(self):
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
        df = self.df.copy()
        layers = ['staff', 'voice', 'label_type']
        for c in layers:
            if self.cols[c] not in df.columns:
                df[c] = None
        color_cols = ['color_name', 'color_html', 'color_r']
        if any(True for c in color_cols if self.cols[c] in df):
            color_name = self.cols['color_name']
            if color_name in df.columns:
                pass
            elif self.cols['color_html'] in df.columns:
                df[color_name] = html2format(df, 'name')
            elif self.cols['color_r'] in df.columns:
                df[color_name] = rgb2format(df, 'name')
            df[color_name] = df[color_name].fillna('default')
            layers.append(color_name)
        else:
            df['color_name'] = 'default'
            layers.append('color_name')
        type2name = map_dict({
            0: '0 (Plain Text)',
            1: '1 (Nashville)',
            2: '2 (Roman Numeral)',
            3: '3 (Absolute Chord)',
            'dcml': 'dcml',
        })
        df.label_type = df.label_type.map(type2name)
        return self.count(), df.groupby(layers, dropna=False).size()

    def __repr__(self):
        n, layers = self.annotation_layers
        return f"{n} labels:\n{layers.to_string()}"

    def get_labels(self, staff=None, voice=None, label_type=None, positioning=True, decode=False, drop=False, warnings=True, column_name=None, color_format=None):
        """ Returns a DataFrame of annotation labels.

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
        column_name : :obj:`str`, optional
            Can be used to rename the columns holding the labels.
        color_format : {'html', 'rgb', 'rgba', 'name', None}
            If label colors are encoded, determine how they are displayed.

        Returns
        -------

        """
        sel = pd.Series(True, index=self.df.index)

        if staff is not None:
            sel = sel & (self.df[self.cols['staff']] == staff)
        if voice is not None:
            sel = sel & (self.df[self.cols['voice']] == voice)
        if label_type is not None and 'label_type' in self.df.columns:
            label_type = self._treat_label_type_param(label_type, warnings=warnings)
            sel = sel & self.df.label_type.isin(label_type)
            # if the column contains strings and NaN:
            # (pd.to_numeric(self.df['label_type']).astype('Int64') == label_type).fillna(False)
        res = self.df[sel].copy()
        if not positioning:
            pos_cols = [c for c in ['minDistance',  'offset', 'offset_x', 'offset_y'] if c in res.columns]
            res.drop(columns=pos_cols, inplace=True)
        if drop:
            self.df = self.df[~sel]
        label_col = self.cols['label']
        if decode:
            res = decode_harmonies(res, label_col=label_col)
        if column_name is not None and column_name != label_col:
            res = res.rename(columns={label_col: column_name})
        color_cols = ['color_html', 'color_r', 'color_g', 'color_b', 'color_a', 'color_name']
        rgb_cols = ['color_r', 'color_g', 'color_b']
        present_cols = [c for c in color_cols if c in res.columns]
        if color_format is not None and len(present_cols) > 0:
            res['color'] = pd.NA
            has_html = 'color_html' in res.columns
            has_name = 'color_name' in res.columns
            has_rgb  = all(col in res.columns for col in rgb_cols)
            has_rgba = has_rgb and 'color_a' in res.columns

            def tuple_or_na(row):
                if row.isna().all():
                    return pd.NA
                return tuple(row)

            if color_format == 'html' and has_html:
                res.color = res.color_html
            elif color_format == 'name' and has_name:
                res.color = res.color_name
            elif color_format == 'rgb' and has_rgb:
                res.color = res[rgb_cols].apply(tuple_or_na, axis=1)
            elif color_format == 'rgba' and has_rgba:
                res.color = res[rgb_cols + ['color_a']].apply(tuple_or_na, axis=1)
            elif has_html:
                res.color = html2format(res, color_format)
            elif has_name:
                res.color = name2format(res, color_format)
            elif has_rgb:
                res.color = rgb2format(res, color_format)
            else:
                logger.warning(f"Color format '{color_format}' could not be computed from columns {present_cols}.")
            res.drop(columns=present_cols, inplace=True)

        if self.mscx_obj is not None:
            res = column_order(self.mscx_obj.parsed.add_standard_cols(res))
        return res


    def expand_dcml(self, drop_others=True, warn_about_others=True, drop_empty_cols=False, **kwargs):
        """ Expands all labels where the label_type has been inferred as 'dcml' and stores the DataFrame in self._expanded.

        Parameters
        ----------
        drop_others : :obj:`bool`, optional
            Set to False if you want to keep labels in the expanded DataFrame which have not label_type 'dcml'.
        warn_about_others : :obj:`bool`, optional
            Set to False to suppress warnings about labels that have not label_type 'dcml'.
            Is automatically set to False if ``drop_others`` is set to False.
        drop_empty_cols : :obj:`bool`, optional
            Return without unused columns
        kwargs
            Additional arguments are passed to :py:meth:`.get_labels` to define the original representation.

        Returns
        -------
        :obj:`pandas.DataFrame`
            Expanded DCML labels
        """
        if 'dcml' not in self.regex_dict:
            self.regex_dict = dict(dcml=DCML_DOUBLE_REGEX, **self.regex_dict)
            self.infer_types()
        df = self.get_labels(**kwargs)
        sel = df.label_type.astype(str).str.contains('dcml').fillna(False)
        if not sel.any():
            self.logger.info(f"Score does not contain any DCML harmonic annotations.")
            return
        if not drop_others:
            warn_about_others = False
        if warn_about_others and (~sel).any():
            self.logger.warning(f"Score contains {(~sel).sum()} labels that don't (and {sel.sum()} that do) match the DCML standard:\n{decode_harmonies(df[~sel], keep_type=True)[['mc', 'mn', 'label', 'label_type']].to_string()}")
        df = df[sel]
        try:
            exp = expand_labels(df, column='label', regex=DCML_REGEX, volta_structure=self.volta_structure, chord_tones=True, logger=self.logger)
            if drop_others:
                self._expanded = exp
            else:
                df = self.df.copy()
                df.loc[sel, exp.df.columns] = exp
                self._expanded = df
            if 'label_type' in self._expanded.columns:
                self._expanded.drop(columns='label_type', inplace=True)
        except:
            self.logger.error(f"Expanding labels failed with the following error:\n{sys.exc_info()[1]}")

        if drop_empty_cols:
            return self._expanded.dropna(axis=1, how='all')
        return self._expanded



    def infer_mc_from_mn(self, mscx_obj=None):
        if mscx_obj is None and self.mscx_obj is None:
            self.logger.error(f"Either pass an MSCX object or load this Annotations object to a score using load_annotations().")
            return False

        mscx = mscx_obj if mscx_obj is not None else self.mscx_obj
        column_names = [self.cols[c] if c in self.cols else c for c in ['mn', 'mn_onset', 'volta']]
        cols = [c for c in column_names if c in self.df.columns]
        inferred_positions = [mscx.infer_mc(**dict(zip(cols, t))) for t in self.df[cols].values]
        return pd.DataFrame(inferred_positions, index=self.df.index, columns=['mc', 'mc_onset'])




    def infer_types(self, regex_dict=None):
        recognized = [0, 1, 2, 3, '0', '1', '2', '3']
        if regex_dict is not None:
            self.regex_dict = regex_dict
        if 'label_type' in self.df.columns:
            self.df.label_type.fillna(0, inplace=True)
            self.df.loc[~self.df.label_type.isin(recognized), 'label_type'] = 0
        else:
            self.df['label_type'] = pd.Series(0, index=self.df.index, dtype='object')
        if 'nashville' in self.df.columns:
            self.df.loc[self.df.nashville.notna(), 'label_type'] = 2
        if 'absolute_root' in self.df.columns:
            self.df.loc[self.df.absolute_root.notna(), 'label_type'] = 3
        if len(self.regex_dict) > 0:
            decoded = decode_harmonies(self.df, label_col=self.cols['label'], return_series=True)
            for name, regex in self.regex_dict.items():
                sel = self.df.label_type.isin(recognized)
                #mtch = self.df.loc[sel, self.cols['label']].str.match(regex)
                mtch = decoded[sel].str.match(regex)
                self.df.loc[sel & mtch, 'label_type'] = self.df.loc[sel & mtch, 'label_type'].astype(str) + f" ({name})"


    def remove_initial_dots(self):
        if self.read_only:
            self.logger.warning(f"Cannot change labels attached to a score. Detach them first.")
            return
        label_col = self.cols['label']
        rem_dots = lambda s: s[1:] if s[0] == '.' else s
        no_nan = self.df[label_col].notna()
        self.df.loc[no_nan, label_col] = self.df.loc[no_nan, label_col].map(rem_dots)


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