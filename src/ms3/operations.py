"""This module contains the functions called by the ms3 commandline interface, which is why they may use
print() instead of log messages from time to time.
"""

import os
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

import pandas as pd
from ms3._typing import AnnotationsFacet, TSVtype, TSVtypes
from ms3.corpus import Corpus
from ms3.logger import (
    MessageType,
    get_ignored_warning_ids,
    get_logger,
    temporarily_suppress_warnings,
)
from ms3.parse import Parse
from ms3.utils import (
    capture_parse_logs,
    check_argument_against_literal_type,
    compute_path_from_file,
    fifths2name,
    make_valid_frictionless_name,
    pretty_dict,
    resolve_facets_param,
    store_dataframe_resource,
    store_dataframes_package,
    tpc2scale_degree,
    update_relative_paths_with_corpus_dirs,
    write_tsv,
)
from ms3.utils.constants import LATEST_MUSESCORE_VERSION

module_logger = get_logger(__name__)


def insert_labels_into_score(
    ms3_object: Union[Parse, Corpus],
    facet: AnnotationsFacet,
    ask_for_input: bool = True,
    replace: bool = True,
    staff: int = None,
    voice: Optional[Literal[1, 2, 3, 4]] = None,
    harmony_layer: Optional[Literal[0, 1, 2]] = None,
    check_for_clashes: bool = True,
    print_info: bool = True,
) -> None:
    """Write labels into the <Harmony> tags of the corresponding MuseScore files.

    Args:
      ms3_object: A Corpus or Parse object including the corresponding files.
      facet: Which kind of labels to pick ('labels', 'expanded', or 'unknown').
      ask_for_input:
          What to do if more than one TSV or MuseScore file is detected for a particular piece. By default,
          the user is asked for input.
          Pass False to prevent that and pick the files with the shortest relative paths instead.
      replace: By default, any existing labels are removed from the scores. Pass False to leave them in,
      which may lead to clashes.
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
          |   MuseScore will split and encode those beginning with a note name ( resulting in ms3-internal
          harmony_layer 3).
          | * 1 the labels are written into the staff's layer for Roman Numeral Analysis.
          | * 2 to have MuseScore interpret them as Nashville Numbers
      check_for_clashes
          By default, warnings are thrown when there already exists a label at a position (and in a notational
          layer) where a new one is attached. Pass False to deactivate these warnings.
      print_info:
          By default, the ms3_object is displayed before and after parsing. Pass False to prevent this,
          for example when the object has many, many files.
    """
    logger = get_logger("ms3.add")
    facet = check_argument_against_literal_type(facet, AnnotationsFacet, logger=logger)
    ms3_object.view.include("facets", "scores", f"^{facet}$")
    ms3_object.disambiguate_facet(facet, ask_for_input=ask_for_input)
    ms3_object.disambiguate_facet("scores", ask_for_input=ask_for_input)
    ms3_object.view.pieces_with_incomplete_facets = False
    obj_name = type(ms3_object).__name__.upper()
    if print_info:
        print(f"VIEW ON THE {obj_name} BEFORE PARSING:")
        ms3_object.info()
    print("PARSING SCORES...")
    ms3_object.parse(parallel=False)
    if replace:
        print("REMOVING LABELS FROM PARSED SCORES...")
        ms3_object.detach_labels()
    print("INSERTING LABELS INTO SCORES...")
    ms3_object.load_facet_into_scores(facet)
    ms3_object.insert_detached_labels(
        staff=staff,
        voice=voice,
        harmony_layer=harmony_layer,
        check_for_clashes=check_for_clashes,
    )
    if print_info:
        print(f"{obj_name} OBJECT AFTER THE OPERATION:")
        ms3_object.info()
    print("DONE INSERTING.")


def extract(
    parse_obj: Parse,
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
    corpuswise: bool = False,
):
    mode = "IN PARALLEL" if parallel else "ONE AFTER THE OTHER"
    if not corpuswise:
        print(f"PARSING SCORES {mode}...")
        parse_obj.parse_scores(parallel=parallel)
        parse_obj.store_extracted_facets(
            root_dir=root_dir,
            notes_folder=notes_folder,
            rests_folder=rests_folder,
            notes_and_rests_folder=notes_and_rests_folder,
            measures_folder=measures_folder,
            events_folder=events_folder,
            labels_folder=labels_folder,
            chords_folder=chords_folder,
            expanded_folder=expanded_folder,
            cadences_folder=cadences_folder,
            form_labels_folder=form_labels_folder,
            metadata_suffix=metadata_suffix,
            markdown=markdown,
            simulate=simulate,
            unfold=unfold,
            interval_index=interval_index,
        )
        return
    for (
        corpus_name,
        corpus,
    ) in parse_obj.iter_independent_corpora():
        print(f"PARSING SCORES FOR CORPUS '{corpus_name}' {mode}...")
        corpus.parse_scores(parallel=parallel)
        corpus.store_extracted_facets(
            root_dir=root_dir,
            notes_folder=notes_folder,
            rests_folder=rests_folder,
            notes_and_rests_folder=notes_and_rests_folder,
            measures_folder=measures_folder,
            events_folder=events_folder,
            labels_folder=labels_folder,
            chords_folder=chords_folder,
            expanded_folder=expanded_folder,
            cadences_folder=cadences_folder,
            form_labels_folder=form_labels_folder,
            metadata_suffix=metadata_suffix,
            markdown=markdown,
            simulate=simulate,
            unfold=unfold,
            interval_index=interval_index,
        )


def check(
    parse_obj: Parse,
    ignore_labels: bool = False,
    ignore_scores: bool = False,
    assertion: bool = False,
    parallel: bool = True,
    ignore_metronome: bool = False,
) -> List[str]:
    assert (
        ignore_labels + ignore_scores < 2
    ), "Activate either ignore_labels or ignore_scores, not both."
    all_warnings = []
    check_logger = get_logger("ms3.check", level=parse_obj.logger.getEffectiveLevel())
    first_bar_warning_str = MessageType(29).name
    if not ignore_scores:
        with capture_parse_logs(parse_obj.logger) as captured_warnings:
            parse_obj.parse_scores(parallel=parallel, only_new=False)
            warning_strings = captured_warnings.content_list
        n_score_warnings = len(warning_strings)
        if n_score_warnings > 0:
            if ignore_metronome:
                # filter out warning messages about missing metronome mark in the first bar, if any
                warning_strings = [
                    w
                    for w in warning_strings
                    if not first_bar_warning_str.startswith(first_bar_warning_str)
                ]
                if len(warning_strings) < n_score_warnings:
                    check_logger.debug(
                        f"Ignored {n_score_warnings - len(warning_strings)} warnings about missing metronome mark in "
                        f"the first bar."
                    )
            if warning_strings:
                all_warnings.extend(warning_strings)
                check_logger.warning(
                    "Warnings detected while parsing scores (see above)."
                )
    else:
        with temporarily_suppress_warnings(parse_obj) as parse_obj:
            parse_obj.parse_scores(parallel=parallel)
    if not ignore_labels:
        with capture_parse_logs(parse_obj.logger) as captured_warnings:
            expanded = parse_obj.get_dataframes(expanded=True)
            warning_strings = captured_warnings.content_list
        if len(expanded) == 0:
            parse_obj.logger.info("No DCML labels to check.")
        elif len(warning_strings) > 0:
            all_warnings.extend(warning_strings)
            check_logger.warning(
                "Warnings detected while checking DCML labels (see above)."
            )
    n_warnings = len(all_warnings)
    if assertion:
        if n_warnings > 0 and all(
            w.startswith(first_bar_warning_str) for w in all_warnings
        ):
            plural = (
                "some of the checked scores do"
                if n_warnings > 1
                else "one of the checked score does"
            )
            assert_msg = (
                f"Check failed only because the {plural} not have a metronome mark in the first bar. "
                f"It would therefore pass if you were to add the --ignore_metronome flag."
            )
        else:
            assert_msg = "Encountered warnings, check failed."
        assert n_warnings == 0, assert_msg
    if n_warnings == 0:
        if ignore_labels:
            msg = "All checked scores alright."
        elif ignore_scores:
            msg = "All checked labels alright."
        else:
            msg = "All checked scores and labels alright."
        check_logger.info(msg)
    return all_warnings


def compare(
    parse_obj: Parse,
    facet: AnnotationsFacet,
    ask: bool = False,
    revision_specifier: Optional[str] = None,
    flip: bool = False,
    force: bool = False,
    logger=None,
) -> Tuple[int, int]:
    """

    Args:
      parse_obj:
      facet:
      ask:
      revision_specifier:
          If None, no comparison is undertaken. Passing an empty string will result in a comparison with the parsed
          TSV files included in the current view (if any). Specifying a git revision will result in a comparison
          with the TSV files at that commit. "LATEST_VERSION" is accepted as a revision specifier and will result in
          a comparison with the TSV files at the tag with the highest version number (falling back to HEAD if no tags
          have been assigned to the repository).
      flip:
      force: Output comparison files even when no differences are found.

    Returns:

    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    parse_obj.parse(parallel=False)
    if parse_obj.n_parsed_scores == 0:
        parse_obj.logger.warning("Parse object does not include any scores.")
        return
    choose = "ask" if ask else "auto"
    if revision_specifier is None:
        key = f"previous_{facet}"
        metadata_update = None
        logger.info(
            f"Comparing annotations to those contained in the current '{facet}' TSV files..."
        )
    else:
        key = revision_specifier
        metadata_update = dict(compared_against=revision_specifier)
        logger.info(
            f"Comparing annotations to those contained in the '{facet}' TSV files @ git revision "
            f"{revision_specifier}..."
        )
    if not key.isidentifier():
        key = "old"
    comparisons_per_corpus = parse_obj.load_facet_into_scores(
        facet=facet, choose=choose, git_revision=revision_specifier, key=key
    )
    logger.info(
        f"Comparisons to be performed:\n{pretty_dict(comparisons_per_corpus, 'Corpus', 'Comparisons')}"
    )
    print(f"METADATA_UPDATE: {metadata_update}")
    return parse_obj.compare_labels(
        key=key,
        detached_is_newer=flip,
        metadata_update=metadata_update,
        force_metadata_update=force,
    )


def store_scores(
    ms3_object: Union[Parse, Corpus],
    only_changed: bool = True,
    root_dir: Optional[str] = None,
    folder: str = "reviewed",
    suffix: str = "_reviewed",
    overwrite: bool = True,
    simulate=False,
) -> Dict[str, List[str]]:
    return ms3_object.store_parsed_scores(
        only_changed=only_changed,
        root_dir=root_dir,
        folder=folder,
        suffix=suffix,
        overwrite=overwrite,
        simulate=simulate,
    )


def _transform(
    ms3_object: Parse | Corpus,
    facets: TSVtypes,
    filename: str,
    output_folder: Optional[str] = None,
    choose: Literal["all", "auto", "ask"] = "auto",
    interval_index: bool = False,
    unfold: bool = False,
    test: bool = False,
    zipped: bool = False,
    overwrite: bool = True,
    log_level=None,
) -> Iterator[Tuple[pd.DataFrame, TSVtype, str, str, str]]:
    """Helper generator to iterate over each selected facets--if a DataFrame is to be written for it either as
    standalone TSV or into a ZIP file--concatenated for the entire Parse or Corpus object.

    Args:
        ms3_object: The object to iterate over.
        facets: Selected facets. Determines the number of iterations.
        filename:
            If non-empty, this will be used for filenames as in f"{filename.{facet}.tsv" and, if ``zipped``,
            f"{filename}.zip". Otherwise, filename will be set to :attr:`Corpus.name` for corpus objects, and for
            Parse objects to the name of the ``output_folder``, if specified, or to "ms3_parse" otherwise.
        output_folder:
            The path where the file is to be created. The generator itself does not create any files, but the argument
            is needed to determine if a file is actually to be written, or not, depending on ``overwrite``. The
            argument is yielded as the third element.
        choose:
            Whether to automatically pick a single facet DataFrame per piece based on the shortest relative path (
            default), or all available, or ask for user input when multiple versions of the same facet have been
            detected.
        interval_index:
            Set to True if you want to replace the RangeIndex level with an IntervalIndex indicating the temporal
            dimensions of each row in terms of quarterbeats.
        unfold:
            Set to True if repeats are to be unfolded prior to concatenation of the facet DataFrames.
        test: Set to True to log what would have happened without yielding anything.
        zipped:
            If False (default), each concatenated facet is to be written to a separate TSV file and these are the
            paths that generator checks for existence, skipping existing files if ``overwrite`` has been set to False.
            If True, all facets are to be written into the same ZIP file; if it exists and ``overwrite`` has been set
            to False, the generator aborts (returns None).
        overwrite:
            True by default, meaning that existing files are to be overwritten. If ``zipped`` has been set to True and
            the target ZIP file exists, it will be removed before beginning the iteration.
        log_level:

    Yields:
        The DataFrame holding the concatenated facet for the entire Corpus or Parse object.
        The facet name.
        ``output_folder``.
        The filename prefix depending on the ``filename`` argument as explained above.
        The log message to emit when writing the file (depends on ``test`` and ``zipped``).
    """
    logger = get_logger("ms3.transform", level=log_level)
    if len(facets) == 0:
        print(
            "Pass at least one of the following arguments: -M (measures), -N (notes), -R (rests), -L (labels), "
            "-X (expanded), -F (form_labels), -E (events), -C (chords), -D (metadata)"
        )
        return
    facets = resolve_facets_param(facets, TSVtype, none_means_all=False)
    obj_is_corpus = isinstance(ms3_object, Corpus)
    if filename:
        prefix = filename
    elif obj_is_corpus:
        prefix = ms3_object.name
    elif output_folder:
        prefix = os.path.basename(output_folder)
    else:
        prefix = "ms3_parse"
    prefix = make_valid_frictionless_name(prefix)
    if zipped:
        # The variable 'path' will be the ZIP file to write to, or, if not zipped, one TSV path per facet (loop below).
        zip_name = f"{prefix}.zip"
        path = (
            zip_name if output_folder is None else os.path.join(output_folder, zip_name)
        )
        if os.path.isfile(path):
            if overwrite:
                if test:
                    logger.info(f"Would have overwritten file {path}")
                    return
                os.remove(path)
                logger.info(f"Removed existing file {path} to overwrite it.")
            else:
                logger.info(
                    f"File {path} already exists and is not to be overwritten. Aborting..."
                )
                return
    for facet in facets:
        facet_filename = make_valid_frictionless_name(f"{prefix}.{facet}")
        tsv_name = f"{facet_filename}.tsv"
        overwriting_tsv = False
        if not zipped:
            path = (
                tsv_name
                if output_folder is None
                else os.path.join(output_folder, tsv_name)
            )
            if os.path.isfile(path):
                if overwrite:
                    if test:
                        logger.info(f"Would have overwritten file {path}")
                        continue
                    overwriting_tsv = True
                else:
                    logger.info(
                        f"File {path} already exists and is not to be overwritten. Skipping..."
                    )
                    continue

        # get concatenated dataframe:
        if facet == "metadata":
            df = ms3_object.metadata()
            update_relative_paths_with_corpus_dirs(df)
        else:
            if obj_is_corpus:
                df = ms3_object.get_facet(
                    facet,
                    choose=choose,
                    interval_index=interval_index,
                    unfold=unfold,
                )
            else:
                # noinspection PyArgumentList
                df = ms3_object.get_facet(
                    facet,
                    choose=choose,
                    interval_index=interval_index,
                    unfold=unfold,
                )
        if df is None or len(df.index) == 0:
            logger.info(
                f"No {facet} data found. Maybe you still need to run ms3 extract?"
            )
            continue
        if test:
            logger.info(f"Would have written {path}.")
            continue
        if zipped:
            msg = f"{tsv_name} written to {path}."
        else:
            msg = f"{path} overwritten." if overwriting_tsv else f"{path} written."
        yield df, facet, output_folder, prefix, msg


def transform_to_resources(
    ms3_object: Parse | Corpus,
    facets: TSVtypes,
    filename: str,
    output_folder: Optional[str] = None,
    choose: Literal["all", "auto", "ask"] = "auto",
    interval_index: bool = False,
    unfold: bool = False,
    test: bool = False,
    zipped: bool = False,
    overwrite: bool = True,
    raise_exception: bool = False,
    write_or_remove_errors_file: bool = True,
    log_level="i",
    custom_metadata: Optional[dict] = None,
):
    logger = get_logger("ms3.transform", level=log_level)
    for df, facet, output_folder, prefix, msg in _transform(
        ms3_object=ms3_object,
        facets=facets,
        filename=filename,
        output_folder=output_folder,
        choose=choose,
        interval_index=interval_index,
        unfold=unfold,
        test=test,
        zipped=zipped,
        overwrite=overwrite,
        log_level=log_level,
    ):
        _ = store_dataframe_resource(
            df=df,
            directory=output_folder,
            piece_name=prefix,
            facet=facet,
            pre_process=True,
            zipped=zipped,
            frictionless=True,
            descriptor_extension="json",
            raise_exception=raise_exception,
            write_or_remove_errors_file=write_or_remove_errors_file,
            custom_metadata=custom_metadata,
        )
        logger.info(msg)


def transform_to_package(
    ms3_object: Parse | Corpus,
    facets: TSVtypes,
    filename: str,
    output_folder: Optional[str] = None,
    choose: Literal["all", "auto", "ask"] = "auto",
    interval_index: bool = False,
    unfold: bool = False,
    test: bool = False,
    zipped: bool = True,
    overwrite: bool = True,
    raise_exception: bool = False,
    write_or_remove_errors_file: bool = True,
    log_level="i",
    custom_metadata: Optional[dict] = None,
):
    logger = get_logger("ms3.transform", level=log_level)
    dfs, returned_facets = [], []
    for df, facet, output_folder, prefix, _ in _transform(
        ms3_object=ms3_object,
        facets=facets,
        filename=filename,
        output_folder=output_folder,
        choose=choose,
        interval_index=interval_index,
        unfold=unfold,
        test=test,
        zipped=zipped,
        overwrite=overwrite,
        log_level=log_level,
    ):
        dfs.append(df)
        returned_facets.append(facet)
    if len(dfs) == 0:
        logger.info("No data to be written.")
        return
    store_dataframes_package(
        dataframes=dfs,
        facets=returned_facets,
        directory=output_folder,
        piece_name=prefix,
        pre_process=True,
        zipped=zipped,
        frictionless=True,
        descriptor_extension="json",
        raise_exception=raise_exception,
        write_or_remove_errors_file=write_or_remove_errors_file,
        logger=logger,
        custom_metadata=custom_metadata,
    )


def corpus2default_package(
    corpus_obj: Corpus,
    raise_exception: bool = False,
    write_or_remove_errors_file: bool = True,
    log_level="i",
):
    """Convenience function for creating a frictionless datapackage for and in a Corpus repository,
    with the repository name,
    including exactly those facets that have been previously extracted as TSV files (and not excluded from the
    current view).
    """
    logger = get_logger("ms3.corpus2default_package", level=log_level)
    corpus_obj.parse_tsv()
    tsv_facets = [
        facet
        for facet in corpus_obj.count_files(detected=False).columns
        if facet not in ("scores", "unknown")
    ]
    if len(tsv_facets) == 0:
        logger.error(
            "No TSV files found. Consider using ms3 extract to create TSV files first."
        )
        return
    facets = ["metadata"] + tsv_facets
    logger.info(f"Creating package containing the facets {facets}...")
    transform_to_package(
        ms3_object=corpus_obj,
        facets=facets,
        filename=corpus_obj.name,
        output_folder=corpus_obj.corpus_path,
        raise_exception=raise_exception,
        write_or_remove_errors_file=write_or_remove_errors_file,
        log_level=log_level,
    )


def update(
    parse_obj: Parse,
    root_dir: Optional[str] = None,
    folder: str = ".",
    suffix: str = "",
    overwrite: bool = False,
    staff: int = -1,
    voice: Literal[1, 2, 3, 4] = 1,
    harmony_layer: Optional[Literal[0, 1, 2, 3]] = 1,
    above: bool = False,
    safe: bool = True,
    parallel: bool = True,
):
    parse_obj.parse_scores(parallel=parallel)
    for corpus_name, corpus in parse_obj.iter_corpora():
        need_update = []
        latest_version = LATEST_MUSESCORE_VERSION.split(".")
        for piece, piece_obj in corpus.iter_pieces():
            for file, score in piece_obj.iter_parsed("scores"):
                score_version = score.mscx.metadata["musescore"].split(".")
                need_update.append(score_version < latest_version)
        if any(need_update):
            if corpus.ms is None:
                n_need_update = sum(need_update)
                print(
                    f"No MuseScore 3 executable was specified, so none of the {n_need_update} outdated scores "
                    f"have been updated."
                )
            else:
                up2date_paths = corpus.update_scores(
                    root_dir=root_dir, folder=folder, suffix=suffix, overwrite=overwrite
                )
                filtered_view = corpus.view.copy()
                filtered_view.update_config(file_paths=up2date_paths)
                corpus.set_view(filtered_view)
                corpus.info()
                corpus.parse_scores()
        scores_with_updated_labels = corpus.update_labels(
            staff=staff,
            voice=voice,
            harmony_layer=harmony_layer,
            above=above,
            safe=safe,
        )
        corpus.logger.info(f"Labels updated in {len(scores_with_updated_labels)}")
        file_paths = corpus.store_parsed_scores(overwrite=overwrite, only_changed=True)
        return file_paths


def make_coloring_reports_and_warnings(
    parse_obj: Parse, out_dir: Optional[str] = None, threshold: float = 0.6
) -> bool:
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
    for (corpus_name, piece), file_df_pairs in review_reports.items():
        try:
            piece_logger = get_logger(parse_obj[corpus_name].logger_names[piece])
        except KeyError:
            piece_logger = parse_obj.logger
            piece_logger.warning(
                f"Piece name {piece!r} not found in the logger dict of {corpus_name!r}: "
                f"{list(parse_obj[corpus_name].logger_names.keys())}"
            )
        ignored_warning_ids = get_ignored_warning_ids(piece_logger)
        is_first = True
        for file, report in file_df_pairs:
            report_path = compute_path_from_file(
                file, root_dir=out_dir, folder="reviewed"
            )
            os.makedirs(report_path, exist_ok=True)
            report_file = os.path.join(report_path, file.piece + "_reviewed.tsv")
            if not is_first and os.path.isfile(report_file):
                get_logger("ms3.review").warning(
                    f"This coloring report has been overwritten because several scores have the same piece:\n"
                    f"{report_file}"
                )
            write_tsv(report, report_file)
            is_first = False
            warning_selection = (
                report.count_ratio > threshold
            ) & report.chord_tones.notna()
            for t in report[warning_selection].itertuples():
                message_id = (19, t.mc, str(t.mc_onset), t.label)
                if message_id in ignored_warning_ids:
                    continue
                test_passes = False
                if pd.isnull(t.localkey) or pd.isnull(t.globalkey):
                    context_msg = "The local and/or global key could not be determined."
                else:
                    if len(t.added_tones) > 0:
                        added = (
                            f" plus the added {tpc2scale_degree(t.added_tones, t.localkey, t.globalkey)} ["
                            f"{fifths2name(t.added_tones)}]"
                        )
                    else:
                        added = ""
                    context_msg = f"""In the context of {t.globalkey}.{t.localkey}, it expresses the scale degrees
{tpc2scale_degree(t.chord_tones, t.localkey, t.globalkey)} [{fifths2name(t.chord_tones)}]{added}.
The corresponding score segment has {t.n_untouched} within-label and {t.n_colored} out-of-label note onsets,
a ratio of {t.count_ratio} > {threshold} (the current, arbitrary, threshold).
If it turns out the label is correct, please add the header of this warning to the IGNORED_WARNINGS, ideally followed
by a free-text comment in subsequent lines starting with a space or tab."""
                msg = (
                    f"The label '{t.label}' in m. {t.mn}, onset {t.mn_onset} (MC {t.mc}, onset {t.mc_onset}) seems "
                    f"not to correspond well to the score (which does not necessarily mean it is wrong).\n"
                )
                piece_logger.warning(
                    msg + context_msg, extra={"message_id": message_id}
                )
    return test_passes
