=========
Changelog
=========

Version 1.2.10
==============

* merges old_tests with new_tests
* correct handling of ``labels_cfg``
* refrains from calling ``logging.basicConfig()``
* unknown TSV types now default to ``labels``
* ``conti`` now recognized as abbreviation for "continuation idea"
* suppresses warnings about multiple "Fingering_text" values


Version 1.2.9
=============

* when updating ``README.md``:

  * make 2nd-level heading ``## Overview`` (instead of first-level)
  * don't output ms3 version (to avoid merge conflicts)

* small bugfixes in ``ms3 review`` command

Version 1.2.8
=============

* operations.insert_labels_into_score() filters pieces exactly one facet to be inserted (e.g. ``labels``),
  not a fuzzy regex (e.g., which would include ``form_labels`` in the filter)

Version 1.2.7
=============

* warning files omit system-dependend information from warning headers (6764476)
* bugfixes

Version 1.2.6
=============


* changes the behaviour of the ``ms3 review`` command
  * after coloring out-of-label notes, issue one warning per dubious label
  * rather than one `warnings.log` file per corpus, create one `<fname>.warnings` file per piece in the `reviewed` folder
* makes ``ms3 empty`` work under the new CLI (d8f661a)

Version 1.2.5
=============

* :obj:`~ms3.Corpus` and :obj:`~ms3.Piece` come with the new method ``count_pieces()``
* ``ms3 transform -D`` to concatenate only metadata works
* ``View.fnames_with_incomplete_facets = False`` enforces selected facets if some have been excluded

Version 1.2.4
=============

* segment_by_criterion warns if not IntervalIndex is present d2602617
* adds missing arguments 'unfold' and 'interval_index' to Piece.get_parsed() 71f8c3e4
* when iterating through pieces, skip fnames that don't have any files under the current view fdce948f


Version 1.2.3
=============

**ms3 requires Python 3.10**

* Piece.get_facet() gets parameter 'force' which defaults to False (analogous to the other methods),
  in order to avoid unsolicited score parsing.
* improves ``ms3 transform``:
  * parse only facets to be concatenated (rather than all TSV files)
  * do not accidentally output metadata if not requested
* prevents including 'volta_mcs' in metadata of pieces that don't have voltas

Version 1.2.2
=============

**ms3 requires Python 3.10**

* removes deprecated elements from tab completion
* enables view settings when adding new corpora to Parse object
* small stuff

Version 1.2.1
=============

**ms3 requires Python 3.10**

* enables hiding the info prints in operations.insert_labels_into_score()
* adds `filter_other_fnames` argument to Corpus.add_dir()

Version 1.2.0
=============

**ms3 requires Python 3.10**

Extraction of all lyrics
------------------------

This version enables the extraction of lyrics with all verses. Previously, only the last verse's syllable for any
given position was extracted. The lyrics now can be found in `lyrics_[verse]` columns in the chords facet,
where `lyrics_1` corresponds to the first or only verse.

Extraction of figured bass
--------------------------

Figured bass labels can now be found in the chords facet tables. Score that include at least one figure will have a
``thoroughbass_duration`` column and each layer of figures comes in a separate ``thoroughbass_layer_#`` column.
For example, if all stacks of figures have only layer, there will be only the column ``thoroughbass_layer_1``.

Extraction of custom-named spanners
-----------------------------------

Spanners with adjusted "Begin text" property get their own columns in the chords tables, containing the relevant subselection of IDs.
For example, if a score contains normal ``8va`` spanners and others where the "Begin text" has been set to ``custom``,
all IDs will be combined in the column ``Ottava:8va`` as before, but the subset pertaining to the custom spanners is
additionally shown in the column ``Ottava:8va_custom``.

Including and excluding paths
-----------------------------

It is now possible to specify specific directories to be included or excluded from a view, not only folder names.

New methods and properties
--------------------------

* ``Parse.get_facet()`` (singular)
* ``Corpus.fnames``
* ``Corpus.add_dir()``
* first version of ``utils.merge_chords_and_notes()``


Version 1.1.2
=============

**ms3 requires Python 3.10**

* Refines the new "writing score headers" functionality and makes it non-default. User needs to
  set ``ms3 metadata --prelims`` which replaces the flag ``--ignore`` that had been introduced in 1.1.1.
* A couple of bug fixes, including a very important one regarding conversion of fifths introduced with b0ce8a1d

Version 1.1.1
=============

**ms3 requires Python 3.10**

* enables updating score headers from the respective `metadata.tsv` columns
  (`title_text`, `subtitle_text`, `composer_text`, `lyricist_text`, and `part_name_text`)
* Parse, Corpus, and Piece now come with the method keys()

Version 1.1.0
=============

**ms3 requires Python 3.10**

This version does not throw errors when trying to parse files created by MuseScore 4. Parsing these files has not
sufficiently been tested but so far it was looking good. The fact that MuseScore 3 is able to read such files shows
that not much has changed in the file format itself.

The command ``ms3 convert`` has been updated to support MuseScore 4 executables. With the current MuseScore 4.0.0 this
is not quite straightforward because conversion to ``.mscz`` via the commandline isn't currently working and conversion
to ``.mscx``, if it works at all, deletes the contents of the target directory
(`issue #15367 <https://github.com/musescore/MuseScore/issues/15367#issuecomment-1369783686>`__). The new function
``utils.convert_to_ms4()`` offers a workaround that creates temporary directories to store the
"Uncompressed MuseScore folder" and then copies the ``.mscx`` file to the target directory (default) or zips the
temporary directory into an ``.mscz`` file (parameter ``--format mscz``). For all other target formats, the output
will correspond to what the MuseScore 4 executable yields.

Version 1.0.4
=============

**ms3 requires Python 3.10**

ms3 has gotten a makeover and does not quite like it did before. The major changes are:

* The library is now optimized for one particular folder structure, namely ``[meta-corpus ->] corpus -> piece``.
* ms3 now comes with a full-fledged "views" feature which lets you subselect files in manifold ways.
* The TSV outputs have gained additional columns. In particular, all TSV files now come with the column ``quarterbeats``
  reflecting each event's offset from the piece's beginning.
* Warnings concerning irregularities, e.g. wrong measure numbering due to a cadenza, can now be sanctioned by copying
  them into an IGNORED_WARNINGS file.


New features
------------

* Each object that the user interacts with, ``Parse, Corpus, and Piece``, comes with at least two views, called "default"
  and "all". The "default" view disregards review files, scores in convertible formats, and scores that are not listed
  in the top-level ``metadata.tsv`` file.
* ``metadata.tsv`` files, by the virtue of their first column ``fname``, now serve as authority on what is
  included in the corpus and what belongs together. This column is always unique and supposed to be used as index.
* Suffixed ``metadata_<suffix>.tsv`` files are loaded as available views based on the column ``fname`` (other columns
  are disregarded).
* The Parse object now detects if the passed directory contains individual corpora or if it is a corpus itself.
* Parse objects perform operations by iterating over Corpus objects.
* Corpus objects perform operations by iterating over Piece objects.
* Corpus objects reflect exactly one folder, the ``corpus_path``, and always discover all present files (which can be
  filtered before the actual parsing). Default output paths are derived from it.
* Piece objects unite the various files pertaining to the same ``fname`` and are able to keep multiple versions of the
  same type apart (e.g., scores or annotation files) and pick one automatically, if necessary, or ask for
  user input.
* The command ``ms3 review`` combines the functionalities of ``ms3 check``, ``ms3 extract``, and ``ms3 compare``, and is
  now the only command used in the new ``dcml_corpus_workflow`` action. For each score that has DCML harmony labels,
  it stores another score and TSV file with the suffix ``_reviewed``, in the folder ``reviewed``.

  * The score has all out-of-label tones colored in red and
  * the TSV file contains a report on this coloring procedure. Both files are stored in the folder
    ``reviewed`` on the top level of the corpus.
  * **(1.0.2)** In addition, if any warnings pop up, they are stored in the top-level ``warnings.log`` file.

* Inserting labels into scores is accomplished using the new method ``load_facet_into_scores()`` which comes with the
  optional parameter ``git_revision`` which allows loading TSVs from a specific commit.
* Therefore, ``ms3 compare`` (and hence, ``ms3 review``) is now able to compare the labels in a score with those in a TSV file
  from an older git revision.
* ``ms3 extract -F`` extracts form labels and expands them into a tree-like view in the output TSV.

Changes to the interface
------------------------

* Many things have been renamed for the benefit of a more homogeneous user interface.

  * Methods previously beginning with ``output_`` were renamed to ``store_``.
  * Parse.parse_mscx() => Parse.parse_scores()

* The properties for retrieving DataFrames from ``Score`` objects:

  * are now methods and accept the parameters ``unfold`` and ``interval_index``.
  * return None when a facet is not available.

* Parsed scores and dataframes are always returned with File object that identifies the parsed file in question. This
  is particularly relevant when using the ``get_facet()`` methods that may return facets from parsed TSV files or
  extract them from the scores, according to availability.
* Gets rid of the argument ``simulate`` except for writing files.
* logger_cfg now as **kwargs
* **(1.0.3)** Currently the ``-d/--dir`` argument to ``ms3`` commands accepts only one directory, not several.



Changes to the outputs
----------------------

* **(1.0.1)** When unfolding repeats, add the column ``mn_playthrough`` with disambiguated measure Numbers ('1a', '12b', etc.).
* The column ``label_type`` has been replaced and disambiguated into ``harmony_layer`` (0-3, text, Roman numeral, Nashville,
  guitar chord) and ``regex_match`` (containing the name of the regular expression that matched first).
* Notes tables now come with the two additional columns ``name`` (e.g. "E#4") and ``octave``. For unpitched instruments,
  such as drumset, the column ``name`` displays the designated instrument name (which the user can modify in MuseScore),
  and have no value in the ``octave`` columns.
* For pieces that don't have first and second endings, the TSVs come without a ``volta`` column.
* Extracted metadata

  * **(1.0.1)** come with the new columns last_mc_unfolded, last_mn_unfolded, volta_mcs, guitar_chord_count,
    form_label_count, ms3_version, has_drumset
  * uses the column ``fname`` as index
  * comes with a modified column order
  * renames the previous column ``rel_paths`` to subdir, whereas the new column ``rel_path`` contains
  * include the text fields included in a score. Columns are ``composer_text``, ``title_text``,
    ``subtitle_text``, ``lyricist_text``, ``part_name_text``.

* Upon a full parse (i.e. if the view has default settings), each facet folder gets a ``csv-metadata.json`` file following
  the CSVW standard. This file indicates the version of ms3 that was used to extract the facets. The version is also
  included in the last row of the README.

Other changes
-------------

Many, many bugs have died on the way. Also:

* Most functions and methods now come with type hints.
* New unittest suite that makes use of the DCMLab/unittest_metacorpus repo and enforces it to be at the correct commit.
* The parser is now more robust against user-induced strangeness in MuseScore files.
* **(1.0.1)** Repetitions are unfolded for checking the integrity of DCML phrase annotations in order to deal with voltas correctly.
* **(1.0.3)** Pedal notes that have multiple (volta) endings, although still not being correctly propagated into each ending,
  get propagated into the first ending, and don't cause propagation nor the integrity check to fail anymore


Version 1.0.3
=============

See above, version 1.0.4

Version 1.0.2
=============

See above, version 1.0.4

Version 1.0.1
=============

See above, version 1.0.4


Version 1.0.0
=============

See above, version 1.0.4


Version 0.5.3
=============

* recognizes metadata fields ``reviewers`` and ``annotators`` also in their singular forms
* adds column ``n_onset_positions`` to metadata.tsv
* interval index levels renamed from 'iv' => 'interval'
* gets rid of pandas deprecation warnings
* bug fixes & log messages

Version 0.5.2
=============

* the ``View`` on a ``Parse`` object can now be subscripted with a filename to obtain a ``Piece`` object, allowing
  for better access to the various files belonging to the same piece (based on their file names). These new objects
  facilitate access to the information which previously was available in one row of tge ``View.pieces()`` DataFrame.
* adds command ``ms3 empty`` to remove harmony annotations from scores
* adds command ``ms3 add`` to add harmony annotations from TSV files to scores
* re-factored ``ms3 compare`` to use new methods added to ``View`` objects
* methods based on ``View.iter()`` now accept the parameter ``fnames`` to filter out file names not included in the list
* while adding labels, use fallback values ``staff=-1`` and ``voice=1`` if not specified

Version 0.5.1
=============

* changes to ``iter`` methods for iterating through DataFrames and metadata belonging together:

  * supressed the second item: instead of ``(metadata, paths, df1, df2...)`` yield ``(metadata, df1, df2...)`` where the
    metadata dict contains the paths
  * added methods ``iter_transformed()`` and ``iter_notes()`` to ``Parse`` and ``View`` objects

* added command ``ms3 transform``

  * used to concatenate all parsed TSVs of a certain type into one file including the option to unfold and add quarterbeats
  * stores them with prefix ``concatenated_``; ms3 now ignores all files beginning with this prefix

* changes in default TSV columns

  * ``metadata.tsv`` includes the new columns

    * ``length_qb``: a scores length in quarterbeats (including all voltas)
    * ``length_qb_unfolded``: the same but with unfolded repeats, if any
    * ``all_notes_qb``: the sum of all note durations in quarterbeats
    * ``n_onsets``: the number of all onsets

  * no empty ``volta`` columns are included (except for measures) when no voltas are present

Version 0.5.0
=============

* considerable changes to ``Parse`` objects (bugs might still be abundant, please report them)
    * abolished custom DataFrame indices
    * behaviour shaped towards ms3's standard corpus structure
        * automatic detection of corpora and generation of keys
        * this enables better matching of files that belong together through ``View`` objects (access via ``p['key']``)
        * new method ``iter()`` for iterating through metadata and files that belong together
    * all JSON files passed under the ``paths`` argument are now scanned for a contained list of file paths to be extracted
      (as opposed to before where the JSON file had to be passed as a single path)
    * new iterator ``p.annotation_objects()``
* new module ``transformations``
    * just as ``utils``, members can be imported directly via ``from ms3 import``
    * includes a couple of functions that were previously part of ``utils`` or ``expand_dcml``
    * includes a couple of new functions:
        * get_chord_sequences()
        * group_annotations_by_features()
        * make_gantt_data()
        * transform_annotations()
        * transform_multiple()
* handling hierarchical localkeys and pedals (i.e. we can modulate to the key of ``V/III``)
* Renamed column 'durations_quarterbeats' to 'duration_qb'
* You can now set ``interval_index = True`` to add quarterbeat columns **and** an index with quarterbeat intervals
* New behaviour of the ``folder_re`` argument: It now gets to all paths matching the regEx rather than stopping at a
  higher level that doesn't match. Effectively, this allows, for example, to do ``Parse(path, folder_re='notes')`` to
  select all files from folders called notes.
* bug fixes (e.g. failing less on incoherent repeat structures)

Version 0.4.10
==============

* Enabled extraction of score labels.
* Made the use of ``labels_cfg`` more consistent.
* improved chord lists:
    * include system and tempo texts
    * new algorithm for correct spanner IDs (i.e. for Slurs, Pedal, HairPins, Ottava)
    * lyrics: still extracts only the last verse but now in the corresponding column, e.g. ``lyrics:3`` for verse 3.
* new feature (still in beta): extraction of form labels
    * ``Score.mscx.form_labels``
    * ``Parse.form_labels()``
    * added ``form_labels`` -related parameters to ``Parse.get_lists()`` and ``Parse.store_lists()``
    * added ``utils.expand_form_labels()`` for hierarchical display of form labels

Version 0.4.9
=============


* enabled ``import from ms3`` for all utils
* new command ``ms3 update`` for converting files and moving annotations to the Roman Numeral Analysis layer
* new command ``ms3 metadata`` for writing manually changed information from ``metadata.tsv`` to the metadata fields of the corresponding MuseScore files
* improved the ``ms3 extract`` command:
    * added option ``-D`` for extracting and updating ``metadata.tsv`` and ``README.md``
    * added option ``-q`` for adding 'quarterbeats' and 'durations_quarterbeats' columns
    * included default paths for the capital-letter parameters
* improved the ``ms3 compare`` command:
    * now works with 'expanded' TSVs, too (not only with 'labels')
    * allows 'label' column to include NaN values
* improvements to Parse() objects:
    * attempts to parse scores that need file conversion (e.g. XML, MIDI)
    * ``get_lists()`` method now allows for adding the columns ``quarterbeats`` and ``durations_quarterbeats``, even without unfolding repeats
    * adding 'quarterbeats' without unfolding repeats excludes voltas
    * new method ``get_tsvs()`` for retrieving and concatenating parsed TSV files
    * Parse() now recognizes ``metadata.tsv`` files, expanded TSVs, and TSVs containing cadence labels only
    * parsed ``metadata.tsv`` files can be retrieved/included via the method ``metadata()``
    * new method ``update_metadata()`` for the new ``ms3 metadata`` command
    * decided on standard index levels ``rel_paths`` and ``fnames``
    * improved matching of corresponding score and TSV files
* improvements to Score() objects:
    * new property Score.mscx.volta_structure for retrieving information on first and second endings
* improvements to Annotations() objects:
    * correct propagation of ``localkey`` for voltas
* improvements to commandline interface:
    * added parameter ``-o`` for specifying output directory
    * harmonized the interface of the ``ms3 convert`` command
    * parameter ``exclude_re`` now also filters paths passed via ``-f``
* changed logging behaviours:
    * write only WARNINGs to log file
    * combine loggers for filenames independently of file extensions
* improved extraction of instrument names for metadata
* improved ``ms3 compare`` functionality
* restructured code architecture
* renamed master branch to 'main'
* many bug fixes

Version 0.4.8
=============

* now reads DCML labels with cadence annotations
* unified command-line interface file options and included ``-f file.json``
* Parse got more options for creating DataFrame index levels
* Parse.measures property for convenience
* bug fixes for better GitHub workflows

Version 0.4.7
=============

* Labels can be attached to MuseScore's Roman Numeral Analysis (RNA) layer
    * parameter `label_type=1` in both `Score.attach_labels()` and `Parse.attach_labels()`
    * `Annotations.remove_initial_dots()` before inserting into the RNA layer
    * `Annotations.add_initial_dots()` before inserting into the absolute chord layer
* interpret all `#vii` in major contexts as `vii` when computing chord tones
* code cosmetics and bug fixes

Version 0.4.6
=============

* ms3 extract and Parse.store_lists() now have the option unfold to account for repeats
* minor bug fixes

Version 0.4.5
=============

* added 'ms3 compare' command
* support for parsing cap, capx, midi, musicxml, mxl, and xml files through temporary conversion
* support for parsing MuseScore 2 files through temporary conversion

Version 0.4.3
=============

* added 'ms3 check' command
* support of coloured labels
* write coloured labels to score comparing attached and detached labels to each other
* better interface for defining log file paths (more options, now conforming to the Parse.store_lists() interface)
* fixed erroneous separation of alternative labels


Version 0.4.2
=============

* small bug fixes
* correct computation of chord tones for new DCML syntax elements ``+M``, ``-``, ``^``, and ``v``

Version 0.4.1
=============

* ms3 0.4.1 supports parsing (but not storing) compressed MuseScore files (.mscz)
* Installs "ms3 convert" command to your system for batch conversion using your local MuseScore installation
* "ms3 extract" command now supports creation of log files
* take ``labels_cfg`` into account when creating expanded chord tables

Version 0.4.0
=============

* The standard column 'onset' has been renamed to 'mc_onset' and 'mn_onset' has been added as an additional standard column.
* Parse TSV files as Annotations objects
* Parse.attach_labels() for inserting annotations into MuseScore files
* Prepare detached labels so that they can actually be attached
* Install "ms3 extract" command to the system
* Including da capo, dal segno, fine, and coda for calculating 'next' column in measures tables (for correct unfolding of repeats)
* Simulate parsing and table extraction
* Passing labels_cfg to Score/Parse to control the format of annotation lists
* Easy access to individual parsed files through Parse[ID] or Parse[ix]
* parse annotation files with diverging column names

Version 0.3.0
=============

* Parse.detach_levels() for emptying all parsed scores from annotations
* Parse.store_mscx() for storing altered (e.g. emptied) score objects as MuseScore files
* Parse.metadata() to return a DataFrame with all parsed pieces' metadata
* Parse.get_labels() to retrieve labels of a particular kind
* Parse.info() has improved the information that objects return about themselves
* Parse.key for a quick overview of the files of a given key
* Parse can be used with a custom index instead of IDs [an ID is an (key, i) tuple]
* Score.store_list() for easily storing TSVs
* renamed Score.output_mscx() to store_mscx() for consistency.
* improved expansion of DCML harmony labels

Version 0.2.0
=============

Beta stage:

* attaching and detaching labels
* parsing multiple pieces at once
* extraction of metadata from scores
* inclusion of staff text, dynamics and articulation in chord lists, added 'auto' mode
* conversion of MuseScore's encoding of absolute chords
* first version of docs

Version 0.1.3
=============

At this stage, the library can parse MuseScore 3 files to different types of lists:

* measures
* chords (= groups of notes)
    * including slurs and spanners such as pedal, 8va or hairpin markings
    * including lyrics
* notes
* harmonies

and also some basic metadata.

Version 0.1.0
=============

- Basic parser implemented
- Logging
- Measure lists
