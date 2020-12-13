from collections import defaultdict, namedtuple
from fractions import Fraction as frac

import pandas as pd
import numpy as np

from .logger import LoggedClass, function_logger
from .utils import next2sequence

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

Subsection = namedtuple('subsection', ['endRepeat', 'voltas', 'jumps'], defaults=[[], [], []])


class MeasureList(LoggedClass):
    """ Turns a _MSCX_bs4._measures DataFrame into a measure list and performs a couple of consistency checks on the score.

    Attributes
    ----------
    df : :obj:`pandas.DataFrame`
        The input DataFrame from _MSCX_bs4.raw_measures
    sections : :obj:`bool`, default True
        By default, section breaks allow for several anacrusis measures within the piece (relevant for `mc_offset` column)
        and make it possible to omit a repeat sign in the following bar (relevant for `next` column).
        Set to False if you want to ignore section breaks.
    secure : :obj:`bool`, default False
        By default, measure information from lower staves is considered to contain only redundant information.
        Set to True if you want to be warned about additional measure information from lower staves that is not taken into account.
    reset_index : :obj:`bool`, default True
        By default, the original index of `df` is replaced. Pass False to keep original index values.

    cols : :obj:`dict`
        Dictionary of the relevant columns in `df` as present after the parse.
    ml : :obj:`pandas.DataFrame`
        The measure list in the making; the final result.
    volta_structure : :obj:`dict`
        Keys are first MCs of volta groups, values are dictionaries of {volta_no: [mc1, mc2 ...]}

    """

    def __init__(self, df, sections=True, secure=True, reset_index=True, columns={}, logger_cfg={}):
        """

        Parameters
        ----------
        df
        sections : :obj:`bool`, optional
            By default, pieces containing section breaks (where counting MNs restarts) receive two more columns in the measures
            list, namely ``section`` and ``ambiguous_mn`` to grant access to MNs as shown in MuseScore. Pass False to not
            add such columns.
        secure
        reset_index
        logger_cfg : :obj:`dict`, optional
            The following options are available:
            'name': LOGGER_NAME -> by default the logger name is based on the parsed file(s)
            'level': {'W', 'D', 'I', 'E', 'C', 'WARNING', 'DEBUG', 'INFO', 'ERROR', 'CRITICAL'}
            'file': PATH_TO_LOGFILE to store all log messages under the given path.
        """
        super().__init__(subclass='MeasureList', logger_cfg=logger_cfg)
        self.df = df
        self.add_section_col = sections
        self.sections = []
        self.secure = secure
        self.reset_index = reset_index
        #self.section_breaks = pd.Series(dtype='object')
        self.has_sections = False
        self.unfolded = []

        # create standard columns
        self.cols = {'barline': 'voice/BarLine/subtype',
                     'breaks': 'LayoutBreak/subtype',
                     'dont_count': 'irregular',
                     'endRepeat': 'endRepeat',
                     'jump_bwd': 'Jump/jumpTo',
                     'jump_fwd': 'Jump/continueAt',
                     'keysig_col': 'voice/KeySig/accidental',
                     'len_col': 'Measure:len',
                     'markers': 'Marker/label',
                     'mc': 'mc',
                     'numbering_offset': 'noOffset',
                     'play_repeats': 'Jump/playRepeats',
                     'play_until': 'Jump/playUntil',
                     'sigN_col': 'voice/TimeSig/sigN',
                     'sigD_col': 'voice/TimeSig/sigD',
                     'staff': 'staff',
                     'startRepeat': 'startRepeat',
                     'volta_start': 'voice/Spanner/Volta/endings',
                     'volta_length': 'voice/Spanner/next/location/measures',
                     'volta_frac': 'voice/Spanner/next/location/fractions'}
        col_names = list(self.cols.keys())
        if any(True for c in columns if c not in col_names):
            wrong = [c for c in columns if c not in col_names]
            plural_s = 's' if len(wrong) > 1 else ''
            logger.warning(f"Wrong column name{plural_s} passed: {wrong}. Only {col_names} implemented.")
            columns = {k: v for k, v in columns.items() if k in col_names}
        self.cols.update(columns)
        self.ml = self.get_unique_measure_list().set_index('mc', drop=False)
        renaming = {v: k for k, v in self.cols.items() if v in self.ml.columns}
        self.ml.rename(columns=renaming, inplace=True)
        info_cols = ['barline', 'breaks', 'dont_count', 'endRepeat', 'jump_bwd', 'jump_fwd', 'len_col', 'markers',
                     'numbering_offset',
                     'play_repeats', 'play_until', 'startRepeat', 'volta_start', 'volta_length']
        for col in info_cols:
            if not col in self.ml.columns:
                self.ml[col] = np.nan
        if self.ml.jump_fwd.notna().any():
            self.ml.jump_fwd = self.ml.jump_fwd.replace({'/': None})

        self.volta_groups = get_volta_groups(self.ml)
        jump_sel = self.ml.jump_bwd.notna()
        self.jumps = dict(self.ml.loc[jump_sel, ['mc', 'jump_bwd']].values)
        self.ml.loc[jump_sel, 'play_repeats'] = self.ml.loc[jump_sel, 'play_repeats'].fillna(0).astype(bool)
        self.markers = get_markers(self.ml)

        # store measure counts (MC)
        self.mc = self.ml.mc  # Series
        self.ml = self.ml
        if self.mc.isna().any():
            self.logger.warning(f"MC column contains NaN which will lead to an incorrect 'next' column.")
        self.last_mc = df.iloc[-1].mc
        assert self.last_mc == self.mc.max(), f"The last MC is different from the highest MC number:\n{self.mc.iloc[self.mc.idxmax():]}"
        self.end_mc = self.last_mc

        self.make_ml()

    def make_ml(self):
        self.ml['repeats'] = make_repeat_col(self.ml)
        self.ml['volta'] = make_volta_col(self.ml, self.volta_groups)
        self.make_sections()
        self.ml['next'] = self.next_col
        self.ml['mn'] = make_mn_col(self.ml)
        self.ml['keysig'] = make_keysig_col(self.ml)
        self.ml['timesig'] = make_timesig_col(self.ml, logger=self.logger)
        self.ml['act_dur'] = make_actdur_col(self.ml)
        self.ml['mc_offset'] = make_offset_col(self.ml, logger=self.logger)

        if self.reset_index:
            self.ml.reset_index(drop=True, inplace=True)
        rn = {self.cols[col]: col for col in ['barline', 'dont_count', 'numbering_offset']}
        self.ml.rename(columns=rn, inplace=True)
        ml_cols = ['mc', 'mn', 'keysig', 'timesig', 'act_dur', 'mc_offset', 'volta', 'numbering_offset', 'dont_count',
                   'barline', 'breaks', 'repeats']
        remove_if_empty = ['markers', 'jump_bwd', 'jump_fwd', 'play_repeats', 'play_until']
        if self.ml[remove_if_empty].isna().all().all():
            ml_cols += ['next']
        else:
            ml_cols += remove_if_empty + ['next']
        self.ml = self.ml[ml_cols]
        self.ml[['numbering_offset', 'dont_count']] = self.ml[['numbering_offset', 'dont_count']].apply(
            pd.to_numeric).astype('Int64')
        #self.check_measure_numbers()


    def get_marker_mc(self, m, sec):
        self.logger.debug(f"Getting marker '{m}' from section {sec}: {dict(self.sections[sec].markers)}")
        if m == 'start':
            return self.mc2section.index[sec].left
        mcs = self.sections[sec].markers[m]
        if len(mcs) == 0:
            self.logger.debug(f"No '{m}'-marker was not found in section {sec}. Checking other sections...")
            mcs = self.markers[m]
        if len(mcs) == 0:
            self.logger.warning(f"Marker '{m}' not found in score.")
            return
        if len(mcs) > 1:
            self.logger.warning(
                f"MC {mc} is supposed to jump to label {m} but there are {len(mcs)} of them: {mcs}. Picking the first one.")
        return mcs[0]

    def add_jump(self, mc):
        jump_bwd, play_until, play_repeats, jump_fwd = self.bwd_jumps.loc[mc]
        this_sec = self.mc2section[mc]
        fro = self.get_marker_mc(jump_bwd, this_sec)
        fro_sec = self.mc2section[fro]
        if pd.isnull(play_until) or play_until == 'end':
            to = self.mc2section.index[fro_sec].right - 1
        else:
            to = self.get_marker_mc(play_until, fro_sec)
        to_sec = self.mc2section[to]
        assert fro_sec == to_sec, "Backward jumps that are to continue over section breaks not implemented"
        self.sections[this_sec].dd_amend_last(mc, fro)
        if not pd.isnull(jump_fwd):
            nxt_mc = self.get_marker_mc(jump_fwd, this_sec)
        else:
            nxt_mc = self.sections[this_sec].nxt[mc].default_factory()
        repetition, _ = self.unfold(fro, to + 1, fro_sec, repeats=True)
        self.sections[fro_sec].dd_amend_last(to, nxt_mc)
        return repetition, nxt_mc

    def make_sections(self):
        section_breaks = (self.ml.loc[self.ml.breaks == 'section', 'mc'] + 1).to_list()
        section_breaks = [1] + section_breaks + [self.last_mc + 1]
        ix = pd.IntervalIndex.from_breaks(section_breaks, closed='left', name='boundaries')
        self.has_sections = len(ix) > 1
        self.mc2section = pd.Series(range(len(ix)), index=ix)
        self.bwd_jumps = self.ml.loc[self.ml.jump_bwd.notna(), ['jump_bwd', 'play_until', 'play_repeats', 'jump_fwd']]

        for i in ix:
            l, r = i.left, i.right - 1
            if pd.isnull(self.ml.repeats.loc[l]):
                self.ml.loc[l, 'repeats'] = 'firstSectionMeasure'
            if pd.isnull(self.ml.repeats.loc[r]):
                self.ml.loc[r, 'repeats'] = 'lastSectionMeasure'
            self.sections.append(Section(self.ml.loc[l:r], parent=self))





        for iv, sec in self.mc2section.iteritems():
            l, r = iv.left, iv.right
            mcs, nxt_mc = self.unfold(l, r, sec)
            self.logger.info(f"after section: {r}, after unfold: {nxt_mc}")
            self.unfolded.extend(mcs)
            self.logger.debug(f"Section {sec} unfolded: {mcs}")


        fines = self.markers['fine']
        if len(fines) > 0:
            last_mc = fines[-1]
        else:
            last_mc = self.last_mc
        sec = self.mc2section[last_mc]
        self.sections[sec].dd_amend_last(last_mc, -1)

        check_nxt_col = next2sequence(self.next_col)
        if check_nxt_col != self.unfolded:
            self.logger.error(f"""The playthrough does not match the unfolded 'next' column:
playthrough: {self.unfolded}
unfoldednxt: {check_nxt_col}""")
        else:
            self.logger.debug("Playthrough and unfolded 'next' column match.")

    def unfold(self, fro, to, sec=None, repeats=True, continue_with=None):
        self.logger.info(f"Unfolding from {fro} to {to}")
        res = []
        if sec is None:
            sec = self.mc2section.loc[fro]
        mc = fro
        while mc < to:
            res.append(mc)
            nxt_mc = self.sections[sec].dd_append(mc)
            if repeats and mc in self.sections[sec].repeats:
                    repeated_subsection = self.sections[sec].repeats[mc]
                    del(self.sections[sec].repeats[mc])
                    rep_fro, rep_to = repeated_subsection.left, repeated_subsection.right
                    self.sections[sec].dd_amend_last(mc, rep_fro)
                    repetition, following_mc = self.unfold(rep_fro, rep_to, sec, repeats=False)
                    self.logger.debug(f"nxt_mc: {nxt_mc}, mc following repeat: {following_mc}")
                    res.extend(repetition)
            if mc in self.jumps:
                del (self.jumps[mc])
                repetition, following_mc = self.add_jump(mc)
                self.logger.debug(f"nxt_mc: {nxt_mc}, mc following jump: {following_mc}")
                res.extend(repetition)
            mc = nxt_mc
        return res, mc

    @property
    def next_col(self):
        return pd.Series({mc: tuple(nxt.values()) for sec in self.sections for mc, nxt in sec.nxt.items()})

    def get_unique_measure_list(self, secure=None, **kwargs):
        """ Keep only the measure information from the first staff.
        Uses: keep_one_row_each()

        Parameters
        ----------
        secure : :obj:`bool`
            If the dropped rows contain additional information, set `secure` to True to
            be informed about the information being lost by the function keep_one_row_each().
        **kwargs: Additional parameter passed on to keep_one_row_each(). Ignored if `secure=False`.
        """
        if secure is None:
            secure = self.secure
        if not secure:
            return self.df.drop_duplicates(subset=self.cols['mc']).drop(columns=self.cols['staff'])
        return keep_one_row_each(self.df, compress_col=self.cols['mc'], differentiating_col=self.cols['staff'],
                                 logger=self.logger, **kwargs)

    @property
    def subsections(self):
        sub = [sec.subsection_df for sec in self.sections]
        return pd.concat(sub)

    def __repr__(self):
        res = [str(sec) for sec in self.sections]
        n = len(res)
        if n > 1:
            return '\n'.join(f"{i:<6}{r}" for i, r in enumerate(res, 1))
        return res[0]


class Section(LoggedClass):

    def __init__(self, ml, parent=None, logger_cfg={}):
        super().__init__(subclass='Section', logger_cfg=logger_cfg)
        self.parent=parent
        self.ml = ml
        self.first_mc = self.ml.mc.min()
        self.last_mc = self.ml.mc.max()
        self.repeat_structure = self.get_repeat_structure()
        # self.endRepeats = self.ml.repeats.fillna('').str.contains('end')
        self.repeats = {mc: self.repeat_structure.subsections.index[self.repeat_structure.subsections.index.get_loc(mc)] for mc, _ in self.ml.repeats[self.ml.repeats.fillna('').str.contains('end')].iteritems()}
        self.markers = get_markers(self.ml)
        self.nxt = {mc: defaultdict(default_int(mc + 1)) for mc in
                    range(self.first_mc, self.last_mc + 1)}
        self.played = {mc: 0 for mc in range(self.first_mc, self.last_mc + 1)}
        self.volta_groups = get_volta_groups(self.ml)
        self.amend_voltas()

    def amend_voltas(self):
        """amend measures before volta groups by iterating over lists"""
        for iv, group in self.volta_groups.iteritems():
            l, r = iv.left, iv.right
            nxt = {}
            for i, volta in enumerate(group, 1):
                # if i > 1:
                #     for mc in volta:
                #         self.nxt[mc][i] = self.nxt[mc].default_factory()
                #         del (self.nxt[mc][1])
                frst = volta[0]
                nxt[i] = frst
                last = volta[-1]
                self.nxt[last].default_factory = lambda: r
            new_default = defaultdict(lambda: frst)
            new_default.update(nxt)
            before_first_volta = l - 1
            self.nxt[before_first_volta] = new_default




    def infer_repeats(self):
        """ amend measures at the end of repeated sections and jumps (e.g. dal segno)"""
        for iv, sub in self.repeat_structure.subsections.iteritems():
            fro, to = iv.left, iv.right
            for mc in range(fro, to):
                _ = self.dd_append(mc)
            pointers = [(end, 'endRepeat') for end in sub.endRepeat] + [(j, 'jump') for j in sub.jumps]
            pointers = list(sorted(pointers))
            for mc, pointer in pointers:
                if pointer == 'endRepeat':
                    self.add_repeat(fro, mc)
                elif pointer == 'jump':
                    self.add_jump(mc)

    def add_jump(self, mc):
        self.parent.add_jump(mc)

    def add_repeat(self, fro, to, nxt=None):
        if nxt is None:
            self.dd_amend_last(to, fro)
        else:
            _ = self.dd_append(to, nxt)
        mc = fro
        while mc <= to:
            mc = self.dd_append(mc)

    def dd_amend_last(self, mc, to):
        last = self.played[mc]
        self.nxt[mc][last] = to
        self.logger.debug(f"MC {mc} now jumps to {to} at play {last}")

    def dd_append(self, mc, to=None):
        nxt = self.played[mc] + 1
        self.played[mc] = nxt
        if nxt not in self.nxt[mc]:
            if to is None:
                to = self.nxt[mc].default_factory()
            self.nxt[mc][nxt] = to
        self.logger.debug(f"Playing MC {mc} for the {ordinal(nxt)} time, followed by MC {self.nxt[mc][nxt]}")
        return self.nxt[mc][nxt]

    # def add_repeat(self, fro, to, nxt=None):
    #     previous_plays = max(self.nxt[fro].keys())
    #     pp = max(self.nxt[to].keys())
    #     assert pp == previous_plays, f"Before applying repeat, MC {fro} had been played {previous_plays}, but the end MC {to} {pp} times."
    #     i = previous_plays + 1
    #     self.logger.debug(f"Playing MCs {fro}-{to} for the {ordinal(i)} time. nxt[{to}]: {dict(self.nxt[to])}")
    #     if nxt is None:
    #         # self.logger.debug(
    #         #     f"Changed continuation of MC {to} at {ordinal(previous_plays)} time from {self.nxt[to][previous_plays]} to {fro}")
    #         self.nxt[to][previous_plays] = fro
    #         #self.dd_amend_last(to, fro)
    #     else:
    #         self.logger.debug(f"Changed continuation of MC {to} at {ordinal(i)} time from {self.nxt[to][i]} to {nxt}")
    #         self.nxt[to][i] = nxt
    #
    #     mc = fro
    #     while mc < to:
    #         if not i in self.nxt[mc]:
    #             self.logger.debug(
    #                 f"Adding a {ordinal(i)} play to MC {mc}, continuing with the default MC {self.nxt[mc].default_factory()}")
    #             self.nxt[mc][i] = self.nxt[mc].default_factory()
    #         else:
    #             self.logger.debug(f"MC {mc} already has continuation at {ordinal(i)} play: {self.nxt[mc][i]}")
    #         mc = self.nxt[mc][i]

    @property
    def raw(self):
        return {mc: dict(dd, default=dd.default_factory()) for mc, dd in self.nxt.items()}

        # self.check_volta_repeats = {}
        # self.wasp_nest = {}
        # for first_mc, group in volta_structure.items():
        #     firsts = []
        #     lasts = []
        #     last_volta = max(group)
        #     mc_after_voltas = max(group[last_volta]) + 1
        #     if mc_after_voltas not in self.next:
        #         mc_after_voltas = None
        #     for volta, mcs in group.items():
        #         # every volta except the last needs will have the `next` value replaced either by the startRepeat MC or
        #         # by the first MC after the last volta
        #         if volta < last_volta:
        #             lasts.append(mcs[-1])
        #         # the bar before the first volta will have first bar of every volta as `next`
        #         firsts.append(mcs[0])
        #     self.next[first_mc - 1] = firsts + self.next[first_mc - 1][1:]
        #     # check_volta_repeats keys are last MCs of all voltas except last voltas, values are all False at the beginning
        #     # and they are set to True if
        #     lasts_with_repeat = [l for l in lasts if self.repeats[l] == 'end']
        #     for l in lasts:
        #         if not pd.isnull(self.repeats[l]):
        #             if self.repeats[l] == 'end':
        #                 self.check_volta_repeats[l] = False
        #                 self.wasp_nest[l] = lasts_with_repeat
        #             else:
        #                 self.logger.warning(f"MC {l}, which is the last MC of a volta, has a different repeat sign than 'end': {self.repeats[l]}")
        #         elif mc_after_voltas is None:
        #             if l not in bwd_jumps.mc.values:
        #                 self.logger.warning(f"MC {l} is the last MC of a volta but has neither a repeat sign or jump, nor is there a MC after the volta group where to continue.")
        #         else:
        #             self.next[l] = [mc_after_voltas]

    def get_repeat_structure(self):
        col_names = ['mc', 'repeats', 'breaks']
        sel = self.ml[col_names[1:]].notna().any(axis=1)
        r = RepeatStructure(self.ml)
        for mc, repeats, breaks in self.ml.loc[sel, col_names].itertuples(index=False):
            r.treat_input(mc, repeats, breaks == 'section')
        return r

    @property
    def subsection_df(self):
        return self.repeat_structure.subsection_df

    def __repr__(self):
        res = []
        for iv, sub in self.repeat_structure.subsections.iteritems():
            groups = self.repeat_structure.volta_groups[self.repeat_structure.volta_groups.index.overlaps(iv)].to_list()
            res.append(subsection2string(iv.left, iv.right - 1, sub.endRepeat, groups))
        return '  '.join(res)


class RepeatStructure(LoggedClass):

    def __init__(self, df, logger_cfg={}):
        super().__init__(subclass='RepeatStructure', logger_cfg=logger_cfg)
        self.mc = df.mc  # Series
        if self.mc.isna().any():
            self.logger.warning(f"MC column contains NaN which will lead to an incorrect 'next' column.")
        self.first_mc = self.mc.min()
        self.last_mc = self.mc.max()
        self.ix = pd.IntervalIndex.from_tuples([(self.first_mc, self.last_mc + 1)], closed='left', name='boundaries')
        self.subsections = pd.Series([Subsection()], index=self.ix)
        self.volta_groups = get_volta_groups(df)
        self.repeats = dict(df[['mc', 'repeats']].values)
        self.jumps = dict(df.loc[df.jump_bwd.notna(), ['mc', 'jump_bwd']].values)
        self.start = None
        self.potential_start = None
        self.potential_ending = None
        self.last_added_iv = pd.Interval(0, 0)

    def start_section(self, mc):
        if self.start is not None:
            if self.potential_ending is None:
                self.logger.warning(f"""The startRepeat in MC {self.start} is missing its endRepeat.
For correction, MC {mc - 1} is interpreted as such because it precedes the next startRepeat.""")
                self.end_section(mc - 1)
            else:
                ending, reason = self.potential_ending
                self.logger.warning(f"""The startRepeat in MC {self.start} is missing its endRepeat.
For correction, MC {ending} is interpreted as such because it {reason}.""")
                self.end_section(ending)
        self.start = mc
        self.potential_start = None
        self.potential_ending = None

    def end_section(self, mc):
        if self.start is not None:
            start = self.start
        elif self.potential_start is not None:
            start, reason = self.potential_start
            if reason == 'firstMeasure':
                self.logger.debug(
                    f"MC {start} has been inferred as startRepeat for the endRepeat in MC {mc} because it is the first bar of the piece.")
            elif reason == 'firstSectionMeasure':
                self.logger.debug(
                    f"MC {start} has been inferred as startRepeat for the endRepeat in MC {mc} because it is the first bar of the section.")
            else:
                msg = f"""The endRepeat in MC {mc} is missing its startRepeat.
For correction, MC {start} is interpreted as such because it {reason}."""
                if "section break" in msg:
                    self.logger.debug(msg)
                else:
                    self.logger.warning(msg)
        else:
            start = None

        if start is None:
            self.logger.error(
                f"No starting point for the repeatEnd in MC {mc} could be determined. It is being ignored.")
        else:
            # self.next[mc] = [start] + self.next[mc]
            try:
                mc_volta_group = self.volta_groups.index.get_loc(mc)
                following = self.volta_groups.index[mc_volta_group].right
                previous = 'volta group'
            except:
                following = mc + 1
                previous = 'endRepeat'

            try:
                pd_iv = pd.Interval(start, following, closed='left')
            except:
                self.logger.error(f"""Probably a mistake in the MSCX source code concerning voltas:
MC: {mc}, volta_groups: {self.volta_groups}. The error is raised when trying to create an interval from these values:
start: {start}, following: {following}""")
                raise
            volta_groups = self.volta_groups[self.volta_groups.index.overlaps(pd_iv)].to_list()
            volta_mcs = [mc for group in volta_groups for mcs in group for mc in mcs]
            end_repeats = [mc for mc in volta_mcs if mc in self.repeats and 'end' in str(self.repeats[mc])]
            jumps = [mc for mc in range(start, following) if mc in self.jumps]
            self.add_section(pd_iv, endRepeat=end_repeats, voltas=volta_groups, jumps=jumps)
            if self.potential_start is not None:
                pot_mc, reason = self.potential_start
                if pot_mc == following:
                    self.potential_start = (pot_mc, reason + f' and the previous {previous}')
                else:
                    self.potential_start = (following, f'is the first bar after the previous {previous}')
            else:
                self.potential_start = (following, f'is the first bar after the previous {previous}')
            self.start = None

    def add_section(self, iv, endRepeat, voltas=[], jumps=[]):
        self.subsections = insert_subsection(self.subsections, iv, Subsection(endRepeat=endRepeat, voltas=voltas, jumps=jumps))
        self.last_added_iv = iv

    def treat_input(self, mc, repeat, section_break=False):
        if pd.isnull(repeat):
            return
        if mc in self.last_added_iv:
            self.logger.debug(f"'{repeat}' in MC {mc} occurs within existing subsection {self.last_added_iv} and is skipped.")
            return
        if repeat == 'firstMeasure':
            self.potential_start = (mc, 'firstMeasure')
        elif repeat == 'firstSectionMeasure':
            self.potential_start = (mc, 'firstSectionMeasure')
        elif repeat == 'start':
            self.start_section(mc)
        elif repeat == 'startend':
            self.start_section(mc)
            self.end_section(mc)
        elif repeat == 'end':
            self.end_section(mc)
        elif repeat == 'lastMeasure':
            if self.start is not None:
                self.potential_ending = (mc, 'is the last bar of the piece.')
                self.start_section(mc + 1)
                self.start = None
        elif repeat == 'lastSectionMeasure':
            if self.start is not None:
                self.potential_ending = (mc, 'is the last bar of the section.')
                self.start_section(mc + 1)
                self.start = None

    @property
    def subsection_df(self):
        cols = Subsection._fields
        s = self.subsections
        # return pd.DataFrame({col: vals for col, vals in zip(cols, zip(*s.values))}, index=s.index)
        return pd.DataFrame.from_records(s.values, columns=cols, index=s.index)

    def __repr__(self):
        return self.subsection_df.to_string()


def default_int(i):
    return lambda: i


def get_markers(df):
    markers = defaultdict(list)
    for t in df.loc[df.markers.notna(), ['mc', 'markers']].itertuples(index=False):
        for marker in t.markers.split(' & '):
            markers[marker].append(t.mc)
    return markers


def get_volta_groups(df):
    """
        Uses: treat_group()
    """

    def treat_group(mc, group):
        n = group.shape[0]

        mcs, numbers, lengths = group.T
        # # check volta numbers
        # expected = np.arange(1, n + 1)
        # if (numbers != expected).any():
        #     logger.warning(f"Volta group of MC {mc} should have voltas {expected.tolist()} but has {numbers.tolist()}")
        # # check volta lengths
        # frst = lengths[0]
        # if (lengths != frst).any():
        #     logger.warning(
        #         f"Volta group of MC {mc} contains voltas with different lengths: {lengths.tolist()} Check for correct computation of MNs.")
        # # check for overlaps and holes
        boundaries = np.append(mcs, mcs[-1] + group[-1, 2])
        correct = {i: np.arange(fro, to).tolist() for i, (fro, to) in
                   enumerate(zip(boundaries[:-1], boundaries[1:]), start=1)}
        # in_score = {i: [row[0] + i for i in range(row[2])] for i, row in enumerate(group, start=1)}
        # if in_score != correct:
        #     logger.warning(
        #         f"The incorrect structure {in_score} of the volta group of MC {mc} has been corrected to {correct}.")
        return list(correct.values())

    cols = ['mc', 'volta_start', 'volta_length', ]
    volta_frac = 'volta_frac' in df.columns
    sel_cols = cols + (['volta_frac'] if volta_frac else [])
    sel = df['volta_start'].notna()
    voltas = df.loc[sel, sel_cols]
    if len(voltas) == 0:
        ix = pd.IntervalIndex([], closed='left', name='boundaries')
        return pd.Series(index=ix, dtype=bool)
    if volta_frac:
        voltas['volta_frac'] = voltas['volta_frac'].notna()
    voltas = voltas.fillna(0).astype(int)
    if volta_frac:
        voltas['volta_length'] += voltas['volta_frac']
    # if voltas['volta_length'].isna().any():
        # rows = voltas[voltas['volta_length'].isna()]
        # logger.debug(f"The volta in MC {rows['mc'].values} has no length: A standard length of 1 is supposed.")
        # voltas['volta_length'] = voltas['volta_length'].fillna(0).astype(int)

    voltas.loc[voltas['volta_start'] == 1, 'group'] = 1
    voltas.group = voltas.group.fillna(0).astype(int).cumsum()
    # groups = {v['mc'].iloc[0]: v[cols].to_numpy() for _, v in voltas.groupby('group')}
    # res = {mc: treat_group(mc, group, logger=logger) for mc, group in groups.items()}
    groups = [treat_group(mc, v[cols].to_numpy()) for mc, v in voltas.groupby('group')]
    print(groups)
    # ivls = [(v.min(), v.max() + 1) for _, v in voltas.groupby('group').mc]
    ivls = [(min(min(mcs) for mcs in group), max(max(mcs) for mcs in group) + 1) for group in groups]
    print(ivls)
    ix = pd.IntervalIndex.from_tuples(ivls, closed='left', name='boundaries')
    res = pd.Series(groups, index=ix)
    return res


def insert_interval(ix, iv):
    l, r = iv.left, iv.right
    if not ix.contains(l).any() or not ix.contains(r).any:
        print("Warning")
        return ix
    breaks = ix.left.to_list() + [ix.right.max()]
    if l not in breaks:
        breaks.append(l)
    if r not in breaks:
        breaks.append(r)
    return pd.IntervalIndex.from_breaks(sorted(breaks), closed='left', name='boundaries')


def insert_subsection(subsections, iv, sub):
    ix = subsections.index
    new_ix = insert_interval(ix, iv)
    l, r = iv.left, iv.right
    old_iv = ix[ix.get_loc(l)]
    old_l, old_r = old_iv.left, old_iv.right
    assert r <= old_r, "The new interval spans an existing interval."
    if l == old_l:
        if r == old_r:
            res = subsections.copy()
            res.loc[l] = sub
            return res
        else:
            values = subsections.loc[:l - 1].to_list() + [sub] + subsections.loc[l:].to_list()
    elif r == old_r:
        values = subsections.loc[:l].to_list() + [sub] + subsections.loc[r + 1:].to_list()
    else:
        values = subsections.loc[:l].to_list() + [sub] + subsections.loc[r:].to_list()
    try:
        res = pd.Series(values, index=new_ix)
    except:
        print(f"old_l: {old_l}, old_r: {old_r}, l: {l}, r: {r}")
        print(f"before: {subsections}\nvalues: {values}\nindex: {new_ix}")
        raise
    return res


@function_logger
def keep_one_row_each(df, compress_col, differentiating_col, differentiating_val=None, ignore_columns=None, fillna=True,
                      drop_differentiating=True):
    """ Eliminates duplicates in `compress_col` but warns about values within the
        dropped rows which diverge from those of the remaining rows. The `differentiating_col`
        serves to identify places where information gets lost during the process.

    The result of this function is the same as `df.drop_duplicates(subset=[compress_col])`
    if `differentiating_val` is None, and `df[df[compress_col] == differentiating_val]` otherwise
    but with the difference that only adjacent duplicates are eliminated.

    Parameters
    ----------
    compress_col : :obj:`str`
        Column with duplicates (e.g. measure counts).
    differentiating_col : :obj:`str`
        Column that differentiates duplicates (e.g. staff IDs).
    differentiating_val : value, optional
        If you want to keep rows where ``compress_col`` has a particular `differentiating_col` value rather than just the
        first one, pass that value (e.g. a certain staff).
    ignore_columns : :obj:`Iterable`, optional
        These columns are not checked.
    fillna : :obj:`bool`, optional
        By default, missing values of kept rows are filled if the dropped rows contain
        one unique value in that particular column. Pass False to keep rows as they are.
    drop_differentiating : :obj:`bool`, optional
        By default, the column that differentiates the `compress_col` is dropped.
        Pass False to prevent that.
    """
    if ignore_columns is None:
        ignore_columns = [differentiating_col]
    else:
        ignore_columns.append(differentiating_col)
    consider_for_notna = [col for col in df.columns if not col in ignore_columns + [compress_col]]
    consider_for_duplicated = consider_for_notna + [compress_col]
    empty_rows = df[consider_for_duplicated].isnull().all(axis=1)
    if differentiating_val is not None:
        keep_rows = df[differentiating_col] == differentiating_val
    else:
        keep_rows = df[compress_col] != df[compress_col].shift()
    drop_rows = ~keep_rows & empty_rows
    result = df.drop(df[drop_rows].index)

    def compress(df):
        if len(df) == 1:
            return df.iloc[0]
        if differentiating_val is None:
            keep_row = df.iloc[[0]].copy()
            remaining = df.iloc[1:].drop_duplicates(subset=consider_for_duplicated)
        else:
            sel = df[differentiating_col] == differentiating_val
            keep_row = df[sel].copy()
            assert len(keep_row) == 1, "The column designated by `differentiating_col` needs to be unique."
            remaining = df[~sel].drop_duplicates(subset=consider_for_duplicated)
        if len(remaining) == 1:
            return keep_row
        which = keep_row[compress_col]
        for val, (col_name, col) in zip(*keep_row[consider_for_notna].itertuples(index=False, name=None),
                                        remaining[consider_for_notna].items()):
            if col.isna().all():
                continue
            vals = col[col.notna()].unique()
            if len(vals) == 1:
                if vals[0] == val:
                    continue
                new_val = vals[0]
                if pd.isnull(val) and fillna:
                    keep_row[col_name] = new_val
                    logger.debug(
                        f"{compress_col} {which}: The missing value in '{col_name}' was replaced by '{new_val}', present in {differentiating_col} {remaining.loc[remaining[col_name] == new_val, differentiating_col].values}.")
                    continue
                logger.warning(
                    f"{compress_col} {which}: The value '{new_val}' in '{col_name}' of {differentiating_col} {remaining.loc[remaining[col_name] == new_val, differentiating_col].values} is lost.")
                continue
            logger.warning(
                f"{compress_col} {which}: The values {vals} in '{col_name}' of {differentiating_col} {remaining.loc[col.notna(), differentiating_col].values} are lost.")
        return keep_row

    result = result.groupby(compress_col, group_keys=False).apply(compress)
    return result.drop(columns=differentiating_col) if drop_differentiating else result


def make_actdur_col(df):
    actdur = df['len_col']
    actdur = actdur.fillna(df['timesig'])
    try:
        return actdur.map(frac).rename('act_dur')
    except:
        print(df.to_dict())
        raise


def make_keysig_col(df):
    if 'keysig_col' in df.columns:
        return df['keysig_col'].fillna(method='ffill').fillna(0).astype(int).rename('keysig')
    return pd.Series(0, index=df.index).rename('keysig')


def make_mn_col(df):
    """ Compute measure numbers where one or two columns can influence the counting.

    Parameters
    ----------
    df : :obj:`pd.DataFrame`
        If no other parameters are given, every row is counted, starting from 1.
    numbering_offset : :obj:`str`, optional
        This column has values of the MuseScore option "Add to bar number", which adds
        notna() values to this and all subsequent measures.
    """
    if 'dont_count' not in df.columns:
        mn = pd.Series(range(1, len(df) + 1), index=df.index)
    else:
        excluded = df['dont_count'].fillna(0).astype(bool)
        mn = (~excluded).cumsum()
    if 'numbering_offset' in df.columns:
        offset = df['numbering_offset']
        if offset.notna().any():
            offset = offset.fillna(0).astype(int).cumsum()
            mn += offset
    return mn.rename('mn')


@function_logger
def make_offset_col(df, breaks=None, ):
    """ If one MN is composed of two MCs, the resulting column indicates the second MC's offset from the MN's beginning.

    Parameters
    ----------
    breaks : :obj:`str`, optional
        If you pass the name of a column, the string 'section' is taken into account
        as ending a section and therefore potentially ending a repeated part even when
        the repeat sign is missing.
    """
    nom_dur = df['timesig'].map(frac)
    sel = df['act_dur'] < nom_dur
    if sel.sum() == 0:
        return pd.Series(0, index=df.index, name='mc_offset')

    if breaks is not None and len(df[df[breaks].fillna('') == 'section']) == 0:
        breaks = None
    cols = ['mc', 'next']
    if breaks is not None:
        cols.append(breaks)
        last_mc = df['mc'].max()
        offsets = {m: 0 for m in df[df[breaks].fillna('') == 'section'].mc + 1 if m <= last_mc}
        # offset == 0 is a neutral value but the presence of mc in offsets indicates that it could potentially be an
        # (incomplete) pickup measure which can be offset even if the previous measure is complete
    else:
        offsets = {}
    nom_durs = dict(df[['mc']].join(nom_dur).itertuples(index=False))
    act_durs = dict(df[['mc', 'act_dur']].itertuples(index=False))

    def missing(mc):
        return nom_durs[mc] - act_durs[mc]

    def add_offset(mc, val=None):
        if val is None:
            val = missing(mc)
        offsets[mc] = val

    irregular = df.loc[sel, cols]
    if irregular['mc'].iloc[0] == 1:
        # Check whether first MC is an anacrusis and mark accordingly
        if len(irregular) > 1 and irregular['mc'].iloc[1] == 2:
            if not missing(1) + act_durs[2] == nom_durs[1]:
                add_offset(1)
            else:
                # regular divided measure, no anacrusis
                pass
        else:
            # is anacrusis
            add_offset(1)
    for t in irregular.itertuples(index=False):
        if breaks:
            mc, nx, sec = t
            if sec == 'section':
                nxt = [i for i in nx if i <= mc]
                if len(nxt) == 0:
                    logger.debug(f"MC {mc} ends a section with an incomplete measure.")
            else:
                nxt = [i for i in nx]
        else:
            mc, nx = t
            nxt = [i for i in nx]
        if mc not in offsets:
            completions = {m: act_durs[m] for m in nxt if m > -1}
            expected = missing(mc)
            errors = sum(True for c in completions.values() if c != expected)
            if errors > 0:
                logger.warning(
                    f"The incomplete MC {mc} (timesig {nom_durs[mc]}, act_dur {act_durs[mc]}) is completed by {errors} incorrect duration{'s' if errors > 1 else ''} (expected: {expected}):\n{completions}")
            for compl in completions.keys():
                add_offset(compl)
        elif offsets[mc] == 0:
            add_offset(mc)
    mc2ix = {m: ix for ix, m in df.mc.iteritems()}
    result = {mc2ix[m]: offset for m, offset in offsets.items()}
    return pd.Series(result, name='mc_offset').reindex(df.index, fill_value=0)


def make_repeat_col(df):
    repeats = df['startRepeat'].copy()
    ends = df['endRepeat']
    sel = dict(
        start=repeats.notna() & ends.isna(),
        startend=repeats.notna() & ends.notna(),
        end=repeats.isna() & ends.notna()
    )
    for case, arr in sel.items():
        repeats.loc[arr] = case
    if pd.isnull(repeats.iloc[0]):
        repeats.iloc[0] = 'firstMeasure'
    if pd.isnull(repeats.iloc[-1]):
        repeats.iloc[-1] = 'lastMeasure'
    return repeats.rename('repeats')


@function_logger
def make_timesig_col(df):
    if pd.isnull(df['sigN_col'].iloc[0]):
        logger.warning("No time signature defined in MC 1: Wild-guessing it's 4/4")
        sigN_pos, sigD_pos = df.columns.get_loc('sigN_col'), df.columns.get_loc('sigD_col')
        df.iloc[0, [sigN_pos, sigD_pos]] = '4'
    n = pd.to_numeric(df['sigN_col'].fillna(method='ffill')).astype(str)
    d = pd.to_numeric(df['sigD_col'].fillna(method='ffill')).astype(str)
    return (n + '/' + d).rename('timesig')


def make_volta_col(df, volta_groups):
    """ Create the input for `volta_structure` using get_volta_groups()
    """
    mc2volta = {mc: volta for group in volta_groups for volta, mcs in enumerate(group, 1) for mc in mcs}
    return df.mc.map(mc2volta).astype('Int64').rename('volta')


def ordinal(i):
    if i == 1:
        return '1st'
    elif i == 2:
        return '2nd'
    elif i == 3:
        return '3rd'
    return f'{i}th'


def subsection2string(start, end, endRepeat=[], volta_groups=[]):
    repeated = endRepeat != []
    if len(volta_groups) == 0:
        assert len(endRepeat) < 2, f"Subsection MCs {start}-{end} have more than one endRepeat: {endRepeat}"
        if repeated:
            left, right = '|:', ':|'
        else:
            left, right = '|', '|'
        return f"{left}{start} {end}{right}"

    left = '|:' if repeated else '|'
    right = ':|' if repeated else '|'
    last_volta_group = volta_groups[-1]
    volta_mcs = [mc for mcs in last_volta_group for mc in mcs]
    volta_last_mcs = [mcs[-1] for mcs in last_volta_group]
    res = f"{left}{start} "

    def volta_group2string(gr):
        return ''.join(f"({', '.join(str(mc) for mc in mcs)})" for mcs in gr)

    if repeated:
        prev_pos = 0
        res += ' '.join(volta_group2string(gr) for gr in volta_groups[:-1]) + ' '
        for rep in endRepeat:
            assert rep in volta_last_mcs, f"The endRepeat in MC {rep} does not occur at the end of the volta."
            pos = volta_last_mcs.index(rep) + 1
            res += volta_group2string(last_volta_group[prev_pos:pos])
            res += right
            prev_pos = pos
        res += volta_group2string(last_volta_group[pos:]) + '|'
    else:
        res += ' '.join(volta_group2string(gr) for gr in volta_groups)
        if end in volta_mcs:
            assert end == max(volta_mcs), f"Subsection end in MC {mc} is not the last measure of the volta group {last_volta_group}."
            res += right
        else:
            res += f" {end}{right}"
    return res


@function_logger
def treat_group(mc, group):
    """ Helper function for make_volta_col()
        Input example: array([[93,  1,  1], [94,  2,  2], [96,  3,  1]])
        where columns are (MC, volta number, volta length).
    """
    n = group.shape[0]
    mcs, numbers, lengths = group.T
    # check volta numbers
    expected = np.arange(1, n + 1)
    if (numbers != expected).any():
        logger.warning(f"Volta group of MC {mc} should have voltas {expected.tolist()} but has {numbers.tolist()}")
    # check volta lengths
    frst = lengths[0]
    if (lengths != frst).any():
        logger.warning(
            f"Volta group of MC {mc} contains voltas with different lengths: {lengths.tolist()} Check for correct computation of MNs.")
    # check for overlaps and holes
    boundaries = np.append(mcs, mcs[-1] + group[-1, 2])
    correct = {i: np.arange(fro, to) for i, (fro, to) in
               enumerate(zip(boundaries[:-1], boundaries[1:]), start=1)}
    in_score = {i: [row[0] + i for i in range(row[2])] for i, row in enumerate(group, start=1)}
    if in_score != correct:
        logger.warning(
            f"The incorrect structure {in_score} of the volta group of MC {mc} has been corrected to {correct}.")
    return correct
