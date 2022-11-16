import os.path
from typing import Literal, Optional, Tuple, Dict, List

import pandas as pd

from ms3 import Parse
from ms3._typing import AnnotationsFacet
from ms3.score import Score, compare_two_score_objects
from ms3.utils import capture_parse_logs, LATEST_MUSESCORE_VERSION, make_file_path, assert_dfs_equal, convert
from ms3.logger import get_logger, temporarily_suppress_warnings, function_logger
from ms3.view import create_view_from_parameters


def extract(parse_obj,
            root_dir=None,
            notes_folder=None,
            rests_folder=None,
            notes_and_rests_folder=None,
            measures_folder=None,
            events_folder=None,
            labels_folder=None,
            chords_folder=None,
            expanded_folder=None,
            cadences_folder=None,
            form_labels_folder=None,
            metadata_suffix=None,
            markdown=True,
            simulate=None,
            parallel=True,
            unfold=False,
            quarterbeats=False,
            silence_label_warnings=False,
            **suffixes):
    parse_obj.parse_scores(parallel=parallel)
    parse_obj.store_extracted_facets(root_dir=root_dir, notes_folder=notes_folder, rests_folder=rests_folder, notes_and_rests_folder=notes_and_rests_folder,
                                     measures_folder=measures_folder, events_folder=events_folder, labels_folder=labels_folder, chords_folder=chords_folder,
                                     expanded_folder=expanded_folder, cadences_folder=cadences_folder, form_labels_folder=form_labels_folder, metadata_suffix=metadata_suffix,
                                     markdown=markdown, simulate=simulate, unfold=unfold, interval_index=quarterbeats, silence_label_warnings=silence_label_warnings, **suffixes)



def check(parse_obj, scores_only=False, labels_only=False, assertion=False, parallel=True):
    assert sum((scores_only, labels_only)) < 2, "Activate either scores_only or labels_only, not both."
    all_warnings = []
    check_logger = get_logger("ms3.check", level=parse_obj.logger.getEffectiveLevel())
    if not labels_only:
        with capture_parse_logs(parse_obj.logger) as captured_warnings:
            parse_obj.parse_scores(parallel=parallel, only_new=False)
            warnings = captured_warnings.content_list
        if len(warnings) > 0:
            all_warnings.extend(warnings)
            check_logger.warning("Warnings detected while parsing scores (see above).")
    else:
        with temporarily_suppress_warnings(parse_obj) as parse_obj:
            parse_obj.parse_scores(parallel=parallel)
    if not scores_only:
        with capture_parse_logs(parse_obj.logger) as captured_warnings:
            expanded = parse_obj.get_dataframes(expanded=True)
            warnings = captured_warnings.content_list
        if len(expanded) == 0:
            parse_obj.logger.info(f"No DCML labels could be detected.")
        elif len(warnings) > 0:
            all_warnings.extend(warnings)
            check_logger.warning("Warnings detected while checking DCML labels (see above).")
    if assertion:
        assert len(all_warnings) == 0, "Encountered warnings, check failed."
    if len(warnings) == 0:
        if scores_only:
            msg = 'All checked scores alright.'
        elif labels_only:
            msg = 'All checked labels alright.'
        else:
            msg = 'All checked scores and labels alright.'
        check_logger.info(msg)
        return True
    else:
        return False


def compare(parse_obj: Parse,
            facet: AnnotationsFacet,
            ask: bool = False,
            revision_specifier=None,
            flip=False,
            root_dir=None,
            folder='reviewed',
            suffix='_compared',
            simulate=False) -> Dict[str, List[str]]:
    parse_obj.parse(parallel=False)
    if parse_obj.n_parsed_scores == 0:
        parse_obj.logger.warning(f"Parse object does not include any scores.")
        return
    choose = 'ask' if ask else 'auto'
    key = f"previous_{facet}" if revision_specifier is None else revision_specifier
    if not key.isidentifier():
        key = "old"
    parse_obj.load_facet_into_scores(facet=facet,
                                     choose=choose,
                                     git_revision=revision_specifier,
                                     key=key,
                                     )
    parse_obj.compare_labels(key=key,
                             detached_is_newer=flip)
    return parse_obj.store_parsed_scores(only_changed=True, root_dir=root_dir, folder=folder, suffix=suffix, simulate=simulate)

def review(parse_obj: Parse,
           commit: Optional[str] = None,
           root_dir: Optional[str] = None,
           reviewed_folder: str = 'reviewed',
           reviewed_suffix: str = '_reviewed',
           ignore_score_warnings: bool = False,
           ignore_labels_warnings: bool = False,
           n_colored_ratio_threshold: Optional[float] = 0.5,
           assertion: bool = False,
           ):
    review_logger = get_logger('ms3.review')
    parse_obj.parse_scores(parallel=False)
    test_passes = True
    if ignore_labels_warnings + ignore_labels_warnings < 2:
        test_passes = check(parse_obj,
                            scores_only = not ignore_score_warnings,
                            labels_only = not ignore_labels_warnings)
    extract_cmd(args, parse_obj)
    review_reports = parse_obj.color_non_chord_tones()
    filtered_report = pd.concat(review_reports.values(), keys=[(key, parse_obj.files[key][i]) for (key, i) in review_reports.keys()])
    print(filtered_report[filtered_report.dur_ratio > n_colored_ratio_threshold])
    comparison = compare(parse_obj, use=args.use, revision_specifier=commit)
    parse_obj.store_parsed_scores(only_changed=True, root_dir=root_dir, folder=reviewed_folder, suffix=reviewed_suffix)
    if test_passes:
        review_logger.info(f"Parsed scores passed all tests.")
    else:
        msg = "Not all tests have passed."
        if assertion:
            assert test_passes, msg
        else:
            review_logger.info(msg)


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
                filtered_view.update_config(paths=up2date_paths)
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
