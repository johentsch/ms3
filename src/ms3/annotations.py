import sys
import warnings
from functools import cache
from inspect import stack
from typing import Literal, Optional

import pandas as pd

from .expand_dcml import expand_labels
from .logger import LoggedClass
from .utils import (
    column_order,
    decode_harmonies,
    html2format,
    is_any_row_equal,
    load_tsv,
    name2format,
    resolve_dir,
    rgb2format,
    update_cfg,
)
from .utils.constants import DCML_DOUBLE_REGEX, DCML_REGEX, FORM_DETECTION_REGEX


class Annotations(LoggedClass):
    """
    Class for storing, converting and manipulating annotation labels.
    """

    main_cols = ["label", "mc", "mc_onset", "staff", "voice"]
    additional_cols = [
        "harmony_layer",
        "regex_match",
        "absolute_root",
        "rootCase",
        "absolute_base",
        "leftParen",
        "rightParen",
        "offset_x",
        "offset_y",
        "nashville",
        "decoded",
        "color_name",
        "color_html",
        "color_r",
        "color_g",
        "color_b",
        "color_a",
        "placement",
        "minDistance",
        "style",
        "z",
    ]

    def __init__(
        self,
        tsv_path=None,
        df=None,
        cols={},
        index_col=None,
        sep="\t",
        mscx_obj=None,
        infer_types=None,
        read_only=False,
        **logger_cfg,
    ):
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
            Additional columns: harmony_layer, regex_match, absolute_root, rootCase, absolute_base, leftParen,
            rightParen, offset_x, offset_y, nashville, decoded, color_name,
            color_html, color_r, color_g, color_b, color_a, placement, minDistance, style, z
        index_col
        sep
        mscx_obj
        infer_types : :obj:`dict`, optional
            If you want to check all labels against one or several regular expressions, pass them as a
            {label_type -> regEx} dictionary.
            The column regex_match will display the label_type of the last matched regEx. If you pass None, the
            default behaviour is detecting labels of the DCML harmony annotation standard's current version.
        read_only
        logger_cfg : :obj:`dict`, optional
            The following options are available:
            'name': LOGGER_NAME -> by default the logger name is based on the parsed file(s)
            'level': {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}
            'file': PATH_TO_LOGFILE to store all log messages under the given path.
        kwargs :
        """
        super().__init__(subclass="Annotations", logger_cfg=logger_cfg)
        if infer_types is None:
            self.regex_dict = {
                "dcml": DCML_DOUBLE_REGEX,
                "form_labels": FORM_DETECTION_REGEX,
            }
        else:
            self.regex_dict = infer_types
        self._expanded = None
        self.changed = False
        self.read_only = read_only
        self.mscx_obj = mscx_obj
        self.volta_structure = None if mscx_obj is None else mscx_obj.volta_structure

        columns = self.main_cols + self.additional_cols
        self.cols = {c: c for c in columns}
        cols_update, incorrect = update_cfg(cols, self.cols.keys())
        if len(incorrect) > 0:
            last_5 = ", ".join(f"-{i}: {stack()[i].function}()" for i in range(1, 6))
            plural = "These mappings do" if len(incorrect) > 1 else "This mapping does"
            self.logger.warning(
                f"{plural} not pertain to standard columns: {incorrect}\nLast 5 function calls leading here: {last_5}"
            )
        self.cols.update(cols_update)

        if df is not None:
            self.df = df.copy()
        else:
            assert (
                tsv_path is not None
            ), "Name a TSV file to be loaded or pass a DataFrame."
            self.df = load_tsv(tsv_path, index_col=index_col, sep=sep)
        sorting_cols = ["mc", "mn", "mc_onset", "staff"]
        sorting_cols = [self.cols[c] if c in self.cols else c for c in sorting_cols]
        sorting_cols = [c for c in sorting_cols if c in self.df.columns]
        self.df.sort_values(sorting_cols, inplace=True)
        self.infer_types()

    def add_initial_dots(self):
        if self.read_only:
            self.logger.warning(
                "Cannot change labels attached to a score. Detach them first."
            )
            return
        label_col = self.cols["label"]
        notes = {"a", "b", "c", "d", "e", "f", "g", "h"}

        def add_dots(s):
            if s[0].lower() in notes:
                return "." + s
            return s

        self.df[label_col] = self.df[label_col].map(add_dots)

    def prepare_for_attaching(
        self,
        staff: Optional[int] = None,
        voice: Optional[Literal[1, 2, 3, 4]] = None,
        harmony_layer: Optional[Literal[0, 1, 2, 3]] = 1,
        check_for_clashes: bool = True,
    ) -> pd.DataFrame:
        if self.mscx_obj is None:
            self.logger.warning(
                "Annotations object not aware to which MSCX object it is attached."
            )
            return pd.DataFrame()
        df = self.df.copy()
        cols = list(df.columns)
        staff_col = self.cols["staff"]
        if staff_col not in cols:
            if staff is None:
                self.logger.info(
                    "Annotations don't have staff information. Using the default -1 (lowest staff)."
                )
                staff = -1
            df[staff_col] = staff
        else:
            if staff is None:
                if df[staff_col].isna().any():
                    staff = -1
                    self.logger.info(
                        f"Some labels don't have staff information. Assigned staff {staff}."
                    )
                    df[staff_col].fillna(staff, inplace=True)
            else:
                df[staff_col] = staff

        voice_col = self.cols["voice"]
        if voice_col not in cols:
            if voice is None:
                self.logger.info(
                    "Annotations don't have voice information. Attaching to the default, voice 1."
                )
                voice = 1
            df[voice_col] = voice
        else:
            if voice is None:
                if df[voice_col].isna().any():
                    voice = 1
                    self.logger.info(
                        "Some labels don't have voice information. Attaching to the default, voice 1."
                    )
                    df[voice_col].fillna(voice, inplace=True)
            else:
                df[voice_col] = voice

        layer_col = self.cols["harmony_layer"]
        if layer_col not in cols:
            if harmony_layer is None:
                self.logger.info(
                    "Annotations don't have harmony_layer information. Using the default, 1 (Roman numerals)."
                )
                harmony_layer = 1
            else:
                harmony_layer = int(harmony_layer)
            df[layer_col] = harmony_layer
        else:
            if harmony_layer is None:
                if df[layer_col].isna().any():
                    harmony_layer = 1
                    self.logger.info(
                        "Some labels don't have harmony_layer information. Using the default, 1 (Roman numerals)."
                    )
                    df[layer_col].fillna(harmony_layer, inplace=True)
            else:
                df[layer_col] = int(harmony_layer)

        error = False
        if self.cols["mc"] not in cols:
            mn_col = self.cols["mn"] if "mn" in self.cols else "mn"
            if mn_col not in cols:
                self.logger.error(
                    "Annotations need to have at least one column named 'mn' or 'mc'."
                )
                error = True
            else:
                inferred_positions = self.infer_mc_from_mn()
                if inferred_positions.isna().any().any():
                    self.logger.error(
                        "Measure counts and corresponding mc_onsets could not be successfully inferred."
                    )
                    error = True
                else:
                    if "mn_onset" not in self.cols:
                        self.logger.info(
                            "Measure counts successfully inferred. Since there is no 'mn_onset' column, all "
                            "mc_onsets have been set to 0."
                        )
                    else:
                        self.logger.info(
                            "Measure counts and corresponding mc_onsets successfully inferred."
                        )
                    df.insert(df.columns.get_loc("mn"), "mc", inferred_positions["mc"])
                    df.loc[:, "mc_onset"] = inferred_positions["mc_onset"]
                    cols.extend(["mc", "mc_onset"])

        mc_onset_col = self.cols["mc_onset"]
        if mc_onset_col not in cols:
            self.logger.info(
                "No 'mc_onset' column found. All labels will be inserted at mc_onset 0."
            )
            new_col = pd.Series([0] * len(df), index=df.index, name="mc_onset")
            df = pd.concat([new_col, df], axis=1)

        position_cols = ["mc", "mc_onset", "staff", "voice"]
        new_pos_cols = [self.cols[c] for c in position_cols]
        if all(c in df.columns for c in new_pos_cols):
            if check_for_clashes and self.mscx_obj.has_annotations:
                existing = self.mscx_obj.get_raw_labels()[position_cols]
                to_be_attached = df[new_pos_cols]
                clashes = is_any_row_equal(existing, to_be_attached)
                has_clashes = len(clashes) > 0
                if has_clashes:
                    self.logger.error(
                        f"The following positions already have labels:\n{pd.DataFrame(clashes, columns=position_cols)}"
                    )
                    error = True
        elif check_for_clashes:
            self.logger.error(
                "Check for clashes could not be performed because there are columns missing."
            )

        if error:
            return pd.DataFrame()
        return df

    def count(self):
        return len(self.df)

    @property
    def harmony_layer_counts(self):
        """Returns the counts of the harmony_layers as dict."""
        if "harmony_layer" in self.df.columns:
            return self.df.harmony_layer.value_counts(dropna=False).to_dict()
        else:
            return {None: len(self.df)}

    @property
    def annotation_layers(self):
        df = self.df.copy()
        layers = ["staff", "voice", "harmony_layer"]
        for c in layers:
            if self.cols[c] not in df.columns:
                df[c] = None
        color_cols = ["color_name", "color_html", "color_r"]
        if any(True for c in color_cols if self.cols[c] in df):
            color_name = self.cols["color_name"]
            if color_name in df.columns:
                pass
            elif self.cols["color_html"] in df.columns:
                df[color_name] = html2format(df, "name")
            elif self.cols["color_r"] in df.columns:
                df[color_name] = rgb2format(df, "name")
            df[color_name] = df[color_name].fillna("default")
            layers.append(color_name)
        else:
            df["color_name"] = "default"
            layers.append("color_name")
        if "regex_match" in df.columns:
            df.harmony_layer = df.harmony_layer.astype(str) + (
                " (" + df.regex_match + ")"
            ).fillna("")
        return self.count(), df.groupby(layers, dropna=False).size()

    def __repr__(self):
        n, layers = self.annotation_layers
        return f"{n} labels:\n{layers.to_string()}"

    def get_labels(
        self,
        staff: Optional[int] = None,
        voice: Optional[Literal[1, 2, 3, 4]] = None,
        harmony_layer: Optional[Literal[0, 1, 2, 3]] = None,
        positioning: bool = False,
        decode: bool = True,
        drop: bool = False,
        inverse: bool = False,
        column_name: Optional[str] = None,
        color_format: Optional[Literal["html", "rgb", "rgba", "name"]] = None,
        regex=None,
    ):
        """Returns a DataFrame of annotation labels.

        Parameters
        ----------
        staff : :obj:`int`, optional
            Select harmonies from a given staff only. Pass `staff=1` for the upper staff.
        harmony_layer : {0, 1, 2, 3, 'dcml', ...}, optional
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
            Set to False if you want to keep labels in harmony_layer 0, 2, and 3 labels in their original form
            as encoded by MuseScore (e.g., with root and bass as TPC (tonal pitch class) where C = 14 for layer 0).
        drop : :obj:`bool`, optional
            Set to True to delete the returned labels from this object.
        column_name : :obj:`str`, optional
            Can be used to rename the columns holding the labels.
        color_format : {'html', 'rgb', 'rgba', 'name', None}
            If label colors are encoded, determine how they are displayed.

        Returns
        -------

        """
        sel = pd.Series(True, index=self.df.index)

        if staff is not None:
            sel = sel & (self.df[self.cols["staff"]] == staff)
        if voice is not None:
            sel = sel & (self.df[self.cols["voice"]] == voice)
        if harmony_layer is not None and "harmony_layer" in self.df.columns:
            # TODO: account for the split into harmony_layer and regex_match
            # harmony_layer = self._treat_harmony_layer_param(harmony_layer, warnings=warnings)
            sel = sel & (self.df.harmony_layer == str(harmony_layer))
        if regex is not None:
            sel = sel & self.df[self.cols["label"]].str.match(regex).fillna(False)
        if inverse:
            sel = ~sel
        res = self.df[sel].copy()
        if positioning:
            pos_cols = [c for c in ("offset",) if c in res.columns]
        else:
            pos_cols = [
                c
                for c in ("minDistance", "offset", "offset_x", "offset_y")
                if c in res.columns
            ]
        res.drop(columns=pos_cols, inplace=True)
        if drop:
            self.df = self.df[~sel]
        label_col = self.cols["label"]
        if decode:
            res = decode_harmonies(res, label_col=label_col, logger=self.logger)
        if column_name is not None and column_name != label_col:
            res = res.rename(columns={label_col: column_name})
        color_cols = [
            "color_html",
            "color_r",
            "color_g",
            "color_b",
            "color_a",
            "color_name",
        ]
        rgb_cols = ["color_r", "color_g", "color_b"]
        present_cols = [c for c in color_cols if c in res.columns]
        if color_format is not None and len(present_cols) > 0:
            res["color"] = pd.NA
            has_html = "color_html" in res.columns
            has_name = "color_name" in res.columns
            has_rgb = all(col in res.columns for col in rgb_cols)
            has_rgba = has_rgb and "color_a" in res.columns

            def tuple_or_na(row):
                if row.isna().all():
                    return pd.NA
                return tuple(row)

            if color_format == "html" and has_html:
                res.color = res.color_html
            elif color_format == "name" and has_name:
                res.color = res.color_name
            elif color_format == "rgb" and has_rgb:
                res.color = res[rgb_cols].apply(tuple_or_na, axis=1)
            elif color_format == "rgba" and has_rgba:
                res.color = res[rgb_cols + ["color_a"]].apply(tuple_or_na, axis=1)
            elif has_html:
                res.color = html2format(res, color_format)
            elif has_name:
                res.color = name2format(res, color_format)
            elif has_rgb:
                res.color = rgb2format(res, color_format)
            else:
                self.logger.warning(
                    f"Color format '{color_format}' could not be computed from columns {present_cols}."
                )
            res.drop(columns=present_cols, inplace=True)

        if self.mscx_obj is not None:
            res = column_order(self.mscx_obj.parsed.add_standard_cols(res))
        return res

    @cache
    def expand_dcml(
        self,
        drop_others=True,
        warn_about_others=True,
        drop_empty_cols=False,
        chord_tones=True,
        relative_to_global=False,
        absolute=False,
        all_in_c=False,
        **kwargs,
    ):
        """
        Expands all labels where the regex_match has been inferred as 'dcml' and stores the DataFrame in self._expanded.

        Parameters
        ----------
        drop_others : :obj:`bool`, optional
            Set to False if you want to keep labels in the expanded DataFrame which have not regex_match 'dcml'.
        warn_about_others : :obj:`bool`, optional
            Set to False to suppress warnings about labels that have not regex_match 'dcml'.
            Is automatically set to False if ``drop_others`` is set to False.
        drop_empty_cols : :obj:`bool`, optional
            Return without unused columns
        chord_tones : :obj:`bool`, optional
            Pass True if you want to add four columns that contain information about each label's
            chord, added, root, and bass tones. The pitches are expressed as intervals
            relative to the respective chord's local key or, if ``relative_to_global=True``,
            to the globalkey. The intervals are represented as integers that represent
            stacks of fifths over the tonic, such that 0 = tonic, 1 = dominant, -1 = subdominant,
            2 = supertonic etc.
        relative_to_global : :obj:`bool`, optional
            Pass True if you want all labels expressed with respect to the global key.
            This levels and eliminates the features `localkey` and `relativeroot`.
        absolute : :obj:`bool`, optional
            Pass True if you want to transpose the relative `chord_tones` to the global
            key, which makes them absolute so they can be expressed as actual note names.
            This implies prior conversion of the chord_tones (but not of the labels) to
            the global tonic.
        all_in_c : :obj:`bool`, optional
            Pass True to transpose `chord_tones` to C major/minor. This performs the same
            transposition of chord tones as `relative_to_global` but without transposing
            the labels, too. This option clashes with `absolute=True`.
        kwargs
            Additional arguments are passed to :py:meth:`.get_labels` to define the original representation.

        Returns
        -------
        :obj:`pandas.DataFrame`
            Expanded DCML labels
        """
        if "dcml" not in self.regex_dict:
            self.regex_dict = dict(dcml=DCML_DOUBLE_REGEX, **self.regex_dict)
            self.infer_types()
        df = self.get_labels(**kwargs)
        select_dcml = (df.regex_match == "dcml").fillna(False)
        if not select_dcml.any():
            self.logger.info("Score does not contain any DCML harmonic annotations.")
            return
        if not drop_others:
            warn_about_others = False
        if warn_about_others and (~select_dcml).any():
            syntax_errors = decode_harmonies(
                df[~select_dcml], keep_layer=True, logger=self.logger
            )[["mc", "mn", "label", "harmony_layer"]].to_string()
            self.logger.warning(
                f"Score contains {(~select_dcml).sum()} labels that don't (and {select_dcml.sum()} that do) match the "
                f"DCML standard:\n{syntax_errors}",
                extra={"message_id": (15,)},
            )
        df = df[select_dcml]
        try:
            exp = expand_labels(
                df,
                column="label",
                regex=DCML_REGEX,
                volta_structure=self.volta_structure,
                chord_tones=chord_tones,
                relative_to_global=relative_to_global,
                absolute=absolute,
                all_in_c=all_in_c,
                logger=self.logger,
            )
            if drop_others:
                self._expanded = exp
            else:
                df = self.df.copy()
                key_cols = [
                    "globalkey",
                    "localkey",
                    "globalkey_is_minor",
                    "localkey_is_minor",
                ]
                with warnings.catch_warnings():
                    # Setting values in-place is fine, ignore the warning in Pandas >= 1.5.0
                    # This can be removed, if Pandas 1.5.0 does not need to be supported any longer.
                    # See also: https://stackoverflow.com/q/74057367/859591
                    # addition: pandas 2.1.0 throws "FutureWarning: Setting an item of incompatible dtype is
                    # deprecated and will raise in a future error of pandas." because all new columns are interpreted
                    # seem to default to dtype float64.
                    warnings.filterwarnings(
                        "ignore",
                        category=FutureWarning,
                    )
                    df.loc[select_dcml, exp.columns] = exp
                    df.loc[:, key_cols] = df[key_cols].ffill()
                self._expanded = df
            drop_cols = [
                col for col in ("harmony_layer", "regex_match") if col in df.columns
            ]
            if len(drop_cols) > 0:
                self._expanded.drop(columns=drop_cols, inplace=True)
        except Exception:
            self.logger.error(
                f"Expanding labels failed with the following error:\n{sys.exc_info()[1]}"
            )

        if drop_empty_cols:
            return self._expanded.dropna(axis=1, how="all")
        return self._expanded

    def infer_mc_from_mn(self, mscx_obj=None):
        if mscx_obj is None and self.mscx_obj is None:
            self.logger.error(
                "Either pass an MSCX object or load this Annotations object to a score using load_annotations()."
            )
            return False

        mscx = mscx_obj if mscx_obj is not None else self.mscx_obj
        column_names = [
            self.cols[c] if c in self.cols else c for c in ["mn", "mn_onset", "volta"]
        ]
        cols = [c for c in column_names if c in self.df.columns]
        inferred_positions = [
            mscx.infer_mc(**dict(zip(cols, t))) for t in self.df[cols].values
        ]
        return pd.DataFrame(
            inferred_positions, index=self.df.index, columns=["mc", "mc_onset"]
        )

    def infer_types(self, regex_dict=None):
        if "harmony_layer" not in self.df.columns:
            harmony_layer_col = pd.Series(
                0, index=self.df.index, dtype="object", name="harmony_layer"
            )
            self.df = pd.concat([self.df, harmony_layer_col], axis=1)
        if "nashville" in self.df.columns:
            self.df.loc[self.df.nashville.notna(), "harmony_layer"] = 2
        if "absolute_root" in self.df.columns:
            self.df.loc[self.df.absolute_root.notna(), "harmony_layer"] = 3

        if regex_dict is None:
            regex_dict = self.regex_dict
        if len(regex_dict) > 0:
            decoded = decode_harmonies(
                self.df,
                label_col=self.cols["label"],
                return_series=True,
                logger=self.logger,
            )
            sel = decoded.notna()
            if not sel.any():
                self.logger.debug(f"No labels present: {self.df}")
                return
            if "regex_match" not in self.df.columns and sel.any():
                regex_col = pd.Series(index=self.df.index, dtype="object")
                column_position = self.df.columns.get_loc("harmony_layer") + 1
                self.df.insert(column_position, "regex_match", regex_col)
            for name, regex in regex_dict.items():
                # TODO Check if in the loop, previously matched regex names are being overwritten by those matched after
                try:
                    mtch = decoded[sel].str.match(regex)
                except AttributeError:
                    self.logger.warning(
                        f"Couldn't match regex against these labels: {decoded[sel]}"
                    )
                    raise
                self.df.loc[sel & mtch, "regex_match"] = name

    def remove_initial_dots(self):
        if self.read_only:
            self.logger.warning(
                "Cannot change labels attached to a score. Detach them first."
            )
            return
        label_col = self.cols["label"]
        starts_with_dot = self.df[label_col].str[0] == "."
        self.df.loc[starts_with_dot, label_col] = self.df.loc[
            starts_with_dot, label_col
        ].str[1:]

    def store_tsv(
        self,
        tsv_path,
        staff=None,
        voice=None,
        harmony_layer=None,
        positioning=True,
        decode=False,
        sep="\t",
        index=False,
        **kwargs,
    ):
        df = self.get_labels(
            staff=staff,
            voice=voice,
            harmony_layer=harmony_layer,
            positioning=positioning,
            decode=decode,
        )
        if decode and "harmony_layer" in df.columns:
            df.drop(columns="harmony_layer", inplace=True)
        df.to_csv(resolve_dir(tsv_path), sep=sep, index=index, **kwargs)
        self.logger.info(f"{len(df)} labels written to {tsv_path}.")
        return True

    def _treat_harmony_layer_param(self, harmony_layer, warnings=True):
        if harmony_layer is None:
            return None
        all_types = {str(k): k for k in self.harmony_layer_counts.keys()}
        if isinstance(harmony_layer, int) or isinstance(harmony_layer, str):
            harmony_layer = [harmony_layer]
        lt = [str(t) for t in harmony_layer]
        if warnings:
            not_found = [t for t in lt if t not in all_types]
            if len(not_found) > 0:
                plural = len(not_found) > 1
                plural_s = "s" if plural else ""
                self.logger.warning(
                    f"No labels found with {'these' if plural else 'this'} label{plural_s} harmony_layer{plural_s}: "
                    f"{', '.join(not_found)}"
                )
        return [all_types[t] for t in lt if t in all_types]
