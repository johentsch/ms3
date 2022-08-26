from ms3.utils import capture_parse_logs
from ms3.logger import get_logger, temporarily_suppress_warnings


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
            metadata_path=None, markdown=True,
            simulate=None,
            unfold=False,
            quarterbeats=False,
            silence_label_warnings=False,
            **suffixes):
    parse_obj.parse_mscx(simulate=simulate)
    parse_obj.output_dataframes(root_dir=root_dir,
                        notes_folder=notes_folder,
                        notes_and_rests_folder=notes_and_rests_folder,
                        labels_folder=labels_folder,
                        measures_folder=measures_folder,
                        rests_folder=rests_folder,
                        events_folder=events_folder,
                        chords_folder=chords_folder,
                        expanded_folder=expanded_folder,
                        cadences_folder=cadences_folder,
                        metadata_path=metadata_path,
                        markdown=markdown,
                        form_labels_folder=form_labels_folder,
                        simulate=simulate,
                        unfold=unfold,
                        quarterbeats=quarterbeats,
                        silence_label_warnings=silence_label_warnings,
                        **suffixes)



def check(parse_obj, scores_only=False, labels_only=False, assertion=False, parallel=True):
    assert sum((scores_only, labels_only)) < 2, "Activate either scores_only or labels_only, not both."
    all_warnings = []
    check_logger = get_logger("ms3.check", level=parse_obj.logger.getEffectiveLevel())
    if not labels_only:
        with capture_parse_logs(parse_obj.logger) as captured_warnings:
            parse_obj.parse_mscx(parallel=parallel, only_new=False)
            warnings = captured_warnings.content_list
        if len(warnings) > 0:
            all_warnings.extend(warnings)
            check_logger.warning("Warnings detected while parsing scores (see above).")
    else:
        with temporarily_suppress_warnings(parse_obj) as parse_obj:
            parse_obj.parse_mscx(parallel=parallel)
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
        check_logger.info(f"All good.")
        return True
    else:
        return False
