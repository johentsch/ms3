from typing import Literal, Optional, Tuple, Dict, List

from ms3 import Parse
from ms3._typing import AnnotationsFacet
from ms3.utils import capture_parse_logs, LATEST_MUSESCORE_VERSION, pretty_dict, check_argument_against_literal_type
from ms3.logger import get_logger, temporarily_suppress_warnings, function_logger

def insert_labels_into_score(parse_obj: Parse,
                             facet: AnnotationsFacet,
                             ask_for_input: bool = True,
                             replace: bool = True,
                             ) -> None:
    logger = get_logger('ms3.add')
    facet = check_argument_against_literal_type(facet, AnnotationsFacet, logger=logger)
    parse_obj.view.include('facets', 'scores', facet)
    parse_obj.disambiguate_facet(facet, ask_for_input=ask_for_input)
    parse_obj.disambiguate_facet('scores', ask_for_input=ask_for_input)
    parse_obj.view.fnames_with_incomplete_facets = False
    print("PARSING...")
    parse_obj.parse(parallel=False)
    if replace:
        print("REMOVING LABELS FROM PARSED SCORES...")
        parse_obj.detach_labels()
    print("INSERTING LABELS INTO SCORES...")
    parse_obj.load_facet_into_scores(facet)
    parse_obj.insert_detached_labels()
    print(f"PARSE OBJECT AFTER THE OPERATION:")
    parse_obj.info()

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
          parallel: bool = True,
          warnings_file: Optional[str] = None) -> bool:
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
    if warnings_file is not None and len(all_warnings) > 0:
        with open(warnings_file, 'a', encoding='utf-8') as f:
            f.writelines(w + '\n' for w in all_warnings)
        parse_obj.logger.info(f"Added captured warnings to {warnings_file}")
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
        return True
    else:
        return False

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


def store_scores(parse_obj: Parse,
                 only_changed: bool = True,
                 root_dir: Optional[str] = None,
                 folder: str = 'reviewed',
                 suffix: str = '_reviewed',
                 overwrite: bool = True,
                 simulate=False) -> Dict[str, List[str]]:
    return parse_obj.store_parsed_scores(only_changed=only_changed,
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
