import os
from typing import Literal, Optional, Tuple, Dict, List, Union

from ms3 import Parse, Corpus
from ms3._typing import AnnotationsFacet
from ms3.utils import capture_parse_logs, LATEST_MUSESCORE_VERSION, pretty_dict, check_argument_against_literal_type, compute_path_from_file, write_tsv, tpc2scale_degree, fifths2name
from ms3.logger import get_logger, temporarily_suppress_warnings, function_logger, get_ignored_warning_ids


def insert_labels_into_score(ms3_object: Union[Parse, Corpus],
                             facet: AnnotationsFacet,
                             ask_for_input: bool = True,
                             replace: bool = True,
                             staff: int = None,
                             voice: Optional[Literal[1, 2, 3, 4]] = None,
                             harmony_layer: Optional[Literal[0, 1, 2]] = None,
                             check_for_clashes: bool = True,
                             print_info: bool = True,
                             ) -> None:
    """ Write labels into the <Harmony> tags of the corresponding MuseScore files.

    Args:
      ms3_object: A Corpus or Parse object including the corresponding files.
      facet: Which kind of labels to pick ('labels', 'expanded', or 'unknown').
      ask_for_input:
          What to do if more than one TSV or MuseScore file is detected for a particular fname. By default, the user is asked for input.
          Pass False to prevent that and pick the files with the shortest relative paths instead.
      replace: By default, any existing labels are removed from the scores. Pass False to leave them in, which may lead to clashes.
      staff
          If you pass a staff ID, the labels will be attached to that staff where 1 is the upper stuff.
          By default, the staves indicated in the 'staff' column of :obj:`ms3.annotations.Annotations.df`
          will be used, or, if such a column is not present, labels will be inserted under the lowest staff -1.
      voice
          If you pass the ID of a notational layer (where 1 is the upper voice, blue in MuseScore),
          the labels will be attached to that one.
          By default, the notational layers indicated in the 'voice' column of
          :obj:`ms3.annotations.Annotations.df` will be used,
          or, if such a column is not present, labels will be inserted for voice 1.
      harmony_layer
          | By default, the labels are written to the layer specified as an integer in the column ``harmony_layer``.
          | Pass an integer to select a particular layer:
          | * 0 to attach them as absolute ('guitar') chords, meaning that when opened next time,
          |   MuseScore will split and encode those beginning with a note name ( resulting in ms3-internal harmony_layer 3).
          | * 1 the labels are written into the staff's layer for Roman Numeral Analysis.
          | * 2 to have MuseScore interpret them as Nashville Numbers
      check_for_clashes
          By default, warnings are thrown when there already exists a label at a position (and in a notational
          layer) where a new one is attached. Pass False to deactivate these warnings.
      print_info:
          By default, the ms3_object is displayed before and after parsing. Pass False to prevent this,
          for example when the object has many, many files.
    """
    logger = get_logger('ms3.add')
    facet = check_argument_against_literal_type(facet, AnnotationsFacet, logger=logger)
    ms3_object.view.include('facets', 'scores', f"^{facet}$")
    ms3_object.disambiguate_facet(facet, ask_for_input=ask_for_input)
    ms3_object.disambiguate_facet('scores', ask_for_input=ask_for_input)
    ms3_object.view.fnames_with_incomplete_facets = False
    obj_name = type(ms3_object).__name__.upper()
    if print_info:
        print(f"VIEW ON THE {obj_name} BEFORE PARSING:")
        ms3_object.info()
    print(f"PARSING SCORES...")
    ms3_object.parse(parallel=False)
    if replace:
        print("REMOVING LABELS FROM PARSED SCORES...")
        ms3_object.detach_labels()
    print("INSERTING LABELS INTO SCORES...")
    ms3_object.load_facet_into_scores(facet)
    ms3_object.insert_detached_labels(staff=staff, voice=voice, harmony_layer=harmony_layer, check_for_clashes=check_for_clashes)
    if print_info:
        print(f"{obj_name} OBJECT AFTER THE OPERATION:")
        ms3_object.info()
    print("DONE INSERTING.")

def extract(parse_obj: Parse,
            root_dir: Optional[str] = None,
            notes_folder: Optional[str] = None,
            rests_folder: Optional[str] = None,
            notes_and_rests_folder: Optional[str] = None,
            measures_folder: Optional[str] = None,
            events_folder: Optional[str] = None,
            labels_folder: Optional[str] = None,
            chords_folder: Optional[str] = None,
            expanded_folder: Optional[str] = None,
            cadences_folder: Optional[str] = None,
            form_labels_folder: Optional[str] = None,
            metadata_suffix: Optional[str] = None,
            markdown: bool = True,
            simulate: bool = False,
            parallel: bool = True,
            unfold: bool = False,
            interval_index: bool = False,
            silence_label_warnings: bool = False,
            **suffixes):
    parse_obj.parse_scores(parallel=parallel)
    parse_obj.store_extracted_facets(root_dir=root_dir, notes_folder=notes_folder, rests_folder=rests_folder, notes_and_rests_folder=notes_and_rests_folder,
                                     measures_folder=measures_folder, events_folder=events_folder, labels_folder=labels_folder, chords_folder=chords_folder,
                                     expanded_folder=expanded_folder, cadences_folder=cadences_folder, form_labels_folder=form_labels_folder, metadata_suffix=metadata_suffix,
                                     markdown=markdown, simulate=simulate, unfold=unfold, interval_index=interval_index, silence_label_warnings=silence_label_warnings, **suffixes)



def check(parse_obj: Parse,
          ignore_labels: bool = False,
          ignore_scores: bool = False,
          assertion: bool = False,
          parallel: bool = True) -> List[str]:
    assert ignore_labels + ignore_scores < 2, "Activate either ignore_labels or ignore_scores, not both."
    all_warnings = []
    check_logger = get_logger("ms3.check", level=parse_obj.logger.getEffectiveLevel())
    if not ignore_scores:
        with capture_parse_logs(parse_obj.logger) as captured_warnings:
            parse_obj.parse_scores(parallel=parallel, only_new=False)
            warnings = captured_warnings.content_list
        if len(warnings) > 0:
            all_warnings.extend(warnings)
            check_logger.warning("Warnings detected while parsing scores (see above).")
    else:
        with temporarily_suppress_warnings(parse_obj) as parse_obj:
            parse_obj.parse_scores(parallel=parallel)
    if not ignore_labels:
        with capture_parse_logs(parse_obj.logger) as captured_warnings:
            expanded = parse_obj.get_dataframes(expanded=True)
            warnings = captured_warnings.content_list
        if len(expanded) == 0:
            parse_obj.logger.info(f"No DCML labels to check.")
        elif len(warnings) > 0:
            all_warnings.extend(warnings)
            check_logger.warning("Warnings detected while checking DCML labels (see above).")
    if assertion:
        assert len(all_warnings) == 0, "Encountered warnings, check failed."
    if len(all_warnings) == 0:
        if ignore_labels:
            msg = 'All checked scores alright.'
        elif ignore_scores:
            msg = 'All checked labels alright.'
        else:
            msg = 'All checked scores and labels alright.'
        check_logger.info(msg)
    return all_warnings

@function_logger
def compare(parse_obj: Parse,
            facet: AnnotationsFacet,
            ask: bool = False,
            revision_specifier: Optional[str] = None,
            flip=False) -> Tuple[int, int]:
    """

    Args:
      parse_obj:
      facet:
      ask:
      revision_specifier:
          If None, no comparison is undertaken. Passing an empty string will result in a comparison with the parsed
          TSV files included in the current view (if any). Specifying a git revision will result in a comparison
          with the TSV files at that commit.
      flip:

    Returns:

    """
    parse_obj.parse(parallel=False)
    if parse_obj.n_parsed_scores == 0:
        parse_obj.logger.warning(f"Parse object does not include any scores.")
        return
    choose = 'ask' if ask else 'auto'
    if revision_specifier is None:
        key = f"previous_{facet}"
        logger.info(f"Comparing annotations to those contained in the current '{facet}' TSV files...")
    else:
        key = revision_specifier
        logger.info(f"Comparing annotations to those contained in the '{facet}' TSV files @ git revision {revision_specifier}...")
    if not key.isidentifier():
        key = "old"
    comparisons_per_corpus = parse_obj.load_facet_into_scores(facet=facet,
                                                              choose=choose,
                                                              git_revision=revision_specifier,
                                                              key=key)
    logger.info(f"Comparisons to be performed:\n{pretty_dict(comparisons_per_corpus, 'Corpus', 'Comparisons')}")
    return parse_obj.compare_labels(key=key,
                             detached_is_newer=flip)


def store_scores(ms3_object: Union[Parse, Corpus],
                 only_changed: bool = True,
                 root_dir: Optional[str] = None,
                 folder: str = 'reviewed',
                 suffix: str = '_reviewed',
                 overwrite: bool = True,
                 simulate=False) -> Dict[str, List[str]]:
    return ms3_object.store_parsed_scores(only_changed=only_changed,
                                          root_dir=root_dir,
                                          folder=folder,
                                          suffix=suffix,
                                          overwrite=overwrite,
                                          simulate=simulate)


def update(parse_obj: Parse,
           root_dir: Optional[str] = None,
           folder: str = '.',
           suffix: str = '',
           overwrite: bool = False,
           staff: int = -1,
           voice: Literal[1, 2, 3, 4] = 1,
           harmony_layer: Literal[0, 1, 2, 3] = 1,
           above: bool = False,
           safe: bool = True,
           parallel: bool = True):
    parse_obj.parse_scores(parallel=parallel)
    for corpus_name, corpus in parse_obj.iter_corpora():
        need_update = []
        latest_version = LATEST_MUSESCORE_VERSION.split('.')
        for fname, piece in corpus.iter_pieces():
            for file, score in piece.iter_parsed('scores'):
                score_version = score.mscx.metadata['musescore'].split('.')
                need_update.append(score_version < latest_version)
        if any(need_update):
            if corpus.ms is None:
                n_need_update = sum(need_update)
                print(f"No MuseScore 3 executable was specified, so none of the {n_need_update} outdated scores "
                      f"have been updated.")
            else:
                up2date_paths = corpus.update_scores(root_dir=root_dir,
                                                     folder=folder,
                                                     suffix=suffix,
                                                     overwrite=overwrite)
                filtered_view = corpus.view.copy()
                filtered_view.update_config(file_paths=up2date_paths)
                corpus.set_view(filtered_view)
                corpus.info()
                corpus.parse_scores()
        scores_with_updated_labels = corpus.update_labels(staff=staff,
                                                          voice=voice,
                                                          harmony_layer=harmony_layer,
                                                          above=above,
                                                          safe=safe)
        corpus.logger.info(f"Labels updated in {len(scores_with_updated_labels)}")
        file_paths = corpus.store_parsed_scores(overwrite=overwrite, only_changed=True)
        return file_paths


def make_coloring_reports_and_warnings(parse_obj: Parse,
                                       out_dir: Optional[str] = None,
                                       threshold: float = 0.6) -> bool:
    """Performs the note coloring, stores the reports as TSV files in the reviewed folder, and logs warnings about
    those chord label segments where the ratio of out-of-label chords is greater than the given threshold.


    Args:
        parse_obj:
            Parse object with parsed scores containing labels. Coloring will be performed in the XML structure in the
            memory and scores have to be written to disk to see the result.
        out_dir: By default, reports are written to <CORPUS_PATH>/reviewed unless another path is specified here.
        threshold: Above which ratio of out-of-label tones a warning is to be issued.

    Returns:
        False if at least one label went beyond the threshold, True otherwise.
    """
    review_reports = parse_obj.color_non_chord_tones()
    test_passes = True
    for (corpus_name, fname), file_df_pairs in review_reports.items():
        piece_logger = get_logger(parse_obj[corpus_name].logger_names[fname])
        ignored_warning_ids = get_ignored_warning_ids(piece_logger)
        is_first = True
        for file, report in file_df_pairs:
            report_path = compute_path_from_file(file, root_dir=out_dir, folder='reviewed')
            os.makedirs(report_path, exist_ok=True)
            report_file = os.path.join(report_path, file.fname + '_reviewed.tsv')
            if not is_first and os.path.isfile(report_file):
                get_logger('ms3.review').warning(f"This coloring report has been overwritten because several scores have the same fname:\n{report_file}")
            write_tsv(report, report_file)
            is_first = False
            warning_selection = (report.count_ratio > threshold) & report.chord_tones.notna()
            for t in report[warning_selection].itertuples():
                message_id = (19, t.mc, str(t.mc_onset), t.label)
                if message_id in ignored_warning_ids:
                    continue
                test_passes = False
                if len(t.added_tones) > 0:
                    added = f" plus the added {tpc2scale_degree(t.added_tones, t.localkey, t.globalkey)} [{fifths2name(t.added_tones)}]"
                else:
                    added = ""
                msg = f"""The label '{t.label}' in m. {t.mn}, onset {t.mn_onset} (MC {t.mc}, onset {t.mc_onset}) seems not to correspond well to the score (which does not necessarily mean it is wrong).
In the context of {t.globalkey}.{t.localkey}, it expresses the scale degrees {tpc2scale_degree(t.chord_tones, t.localkey, t.globalkey)} [{fifths2name(t.chord_tones)}]{added}.
The corresponding score segment has {t.n_untouched} within-label and {t.n_colored} out-of-label note onsets, a ratio of {t.count_ratio} > {threshold} (the current, arbitrary, threshold).
If it turns out the label is correct, please add the header of this warning to the IGNORED_WARNINGS, ideally followed by a free-text comment in subsequent lines starting with a space or tab."""
                piece_logger.warning(msg, extra={'message_id': message_id})
    return test_passes
