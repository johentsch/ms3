# Changelog

## [2.6.0](https://github.com/johentsch/ms3/compare/v2.5.4...v2.6.0) (2025-03-11)


### Features

* adds the --dirty parameter to ms3 transform which allows one to use the command on a dirty repository ([6a33678](https://github.com/johentsch/ms3/commit/6a3367829165f686f57ecfb3a80f2d2274dc88fd))
* adds the helper check_and_create_unresolved and uses it to not resolve the CLI --out parameter ([214bc52](https://github.com/johentsch/ms3/commit/214bc52ae022324fe31d7c4a815eae77fddcf4dd))
* applies updates relative file paths in ms3 transform -D (i.e., concatenating metadata) ([4398a25](https://github.com/johentsch/ms3/commit/4398a25e9c05091f485e4f2b69bebf0bf2af40d5))
* ms3 convert without -a now looks for metadata to convert only the files indicated there ([994d206](https://github.com/johentsch/ms3/commit/994d2063394c68d664349eb256f5f1edf086daaa))
* the ms3 convert command now differentiates between a relative and an absolute target directory ([0546bdc](https://github.com/johentsch/ms3/commit/0546bdcc6866dedf6fa4cee10904a3b5914ce86e))


### Bug Fixes

* catches bug caused by empty &lt;trackName&gt; tags in the instrumentation header of a MuseScore file ([aa0979a](https://github.com/johentsch/ms3/commit/aa0979aaa7a2929ae839ad1566921e9382cd85af))
* do not remove spaces from tempo markings (in the chords tables) ([9294db8](https://github.com/johentsch/ms3/commit/9294db81746f25fade1e73846c84bbaa345dc21c))
* fixes bug that caused add_quarterbeats_col() to file in the special case where column "mc" is missing yet "quarterbeats" is present ([7441cab](https://github.com/johentsch/ms3/commit/7441cab4b4e4cd3f0702644704aeb488a18e4eaf))
* fixes bug that caused Piece.get_parsed_tsv() to fail ([7c268fd](https://github.com/johentsch/ms3/commit/7c268fd0d2faea71b01522e461031316d4a27162))

## [2.5.4](https://github.com/johentsch/ms3/compare/v2.5.3...v2.5.4) (2025-02-28)


### Bug Fixes

* adds "quarterbeats_playthrough" to the correctly parsed columns ([77c05ac](https://github.com/johentsch/ms3/commit/77c05acde4123220c4814b1a247ac23a7aa4f5e8))
* adds concat_metadata.py ([a2465be](https://github.com/johentsch/ms3/commit/a2465bee6d56b0f6d1e2a9f9e02a01276825f401))
* fixes a small bug re: missing columns and prepares the code for integration into utils ([4b8c740](https://github.com/johentsch/ms3/commit/4b8c74007ac13a4f9762d1edddad8c6863065bc5))

## [2.5.3](https://github.com/johentsch/ms3/compare/v2.5.2...v2.5.3) (2024-09-26)


### Bug Fixes

* pins numpy&lt;2.0.0 ([b75e671](https://github.com/johentsch/ms3/commit/b75e67104d1ac0e0d8118b04696cf92acc96593d))
* replaces webcolors constant with the local one introduced in b680f452 ([06dad93](https://github.com/johentsch/ms3/commit/06dad933ceaba08aa49c5181b1024103c4749a3c))
* update syntax of inplace operations ([b34a323](https://github.com/johentsch/ms3/commit/b34a323b63d8432850d67122200c7814aeb19a01))

## [2.5.2](https://github.com/johentsch/ms3/compare/v2.5.1...v2.5.2) (2024-08-06)


### Bug Fixes

* hardcodes CSS_COLORS rather than creating it from a webcolors constant (fixes [#115](https://github.com/johentsch/ms3/issues/115)) ([b680f45](https://github.com/johentsch/ms3/commit/b680f452e77d421f4b8124d46ad6f1b5ee15d7a3))

## [2.5.1](https://github.com/johentsch/ms3/compare/v2.5.0...v2.5.1) (2024-05-23)


### Bug Fixes

* enables ignoring the COMPETING_MEASURE_INFO_WARNING ([5c727dd](https://github.com/johentsch/ms3/commit/5c727ddb652f5b2a0f6739663cdcd61b9f6801a4))
* resolves [#112](https://github.com/johentsch/ms3/issues/112) ([2f03b79](https://github.com/johentsch/ms3/commit/2f03b79efad442efeaaeb25baee92c4f0a9542ea))

## [2.5.0](https://github.com/johentsch/ms3/compare/v2.4.4...v2.5.0) (2024-05-23)


### Features

* adds helper function write_soup_to_mscx_file() ([48fa73c](https://github.com/johentsch/ms3/commit/48fa73cd6e0a0fa5c57741c603a9781a15b0adac))


### Bug Fixes

* for scores that encode multiple parts, take into account only the main score ([6115ea6](https://github.com/johentsch/ms3/commit/6115ea67ab47fa37cfd6d251830d4adac35b3daa))


### Documentation

* updates CONTRIBUTING.rst ([4689897](https://github.com/johentsch/ms3/commit/4689897ce5c32f5775fc1a8f60961a31349b8682))

## [2.4.4](https://github.com/johentsch/ms3/compare/v2.4.3...v2.4.4) (2024-01-21)


### Documentation

* updates changelog ([70fd82e](https://github.com/johentsch/ms3/commit/70fd82ec744e13d2b718f3d637b5fac44eaf9115))

## [2.4.3](https://github.com/johentsch/ms3/compare/v2.4.2...v2.4.3) (2024-01-16)


### Bug Fixes

* makes utility functions more robust against missing data ([f408de3](https://github.com/johentsch/ms3/commit/f408de319177a62aeff2a77c9fec95d807429682))


### Documentation

* converts CHANGELOG rst =&gt; md and adds the release-please workflow ([04e26cb](https://github.com/johentsch/ms3/commit/04e26cbfd8f93313fd57720efe7b5851e625846f))
* updates changelog ([306cea7](https://github.com/johentsch/ms3/commit/306cea7ea85d850a7bd2c3a398826a7b8d8bc7e4))

## Version 2.4.2

**Full Changelog**: https://github.com/johentsch/ms3/compare/v2.4.1...v2.4.2

## Version 2.4.1

**Full Changelog**: https://github.com/johentsch/ms3/compare/v2.4.0...v2.4.1

## Version 2.4.0

-   adds `git_revision` and `git_tag` to frictionless JSON descriptors
    whenever the git repo is clean (5b76a815)
    -   This includes the new property `Corpus.repo` that makes the
        `git.Repo` object available if applicable.
-   adds `--force` to `ms3 compare` and `ms3 review` commands, allowing
    to output comparison files (potentially including the
    `compared_against` metadata key) even if no differences were found
    (5b76a815)

## Version 2.3.1

Ignoring pandas 2.1.0 FutureWarning which cause some intimidating output
when using the ms3 precommit hook.

## Version 2.3.0

-   Adds 'ms3 precommit' and makes the repo usable as a hook by
    \@johentsch in <https://github.com/johentsch/ms3/pull/106>
    -   The new `ms3 precommit` command is simply a wrapper around
        `ms3 review` that accepts the `--files` arguments as positional
        arguments. This is required for the command to be useable as an
        entry point for a [Git pre-commit](https://pre-commit.com/),
        which passes the paths of modified or added files as positional
        arguments. In addition, the command executes `git add -A` after
        the review so that all changed files are included.
    -   This is to work in the first version of the new, localized, DCML
        annotation workflow that runs on the annotator's machine before
        committing, rather than on a GitHub runner after pushing. Things
        that might be changed in the future:
        -   The `ms3 precommit` command could convert the positional
            arguments into a regular expression to be passed to
            `-i/--include` instead of using the deprecated `--files`.
        -   At some point a mechanism might be needed that makes it
            possible for the hook to ignore warnings that were already
            there, i.e., which are not caused/added by the current
            commit. Currently one would have to remove `--fail` from the
            repo's args configuration but that would let all warnings
            pass and would be besides the point.
    -   New method `score.mscx.update_metadata()` to facilitate (manual)
        updating of the key-value pairs.
    -   Comparison files come with the metadata key
        `compared_against=<commit hash>` when the comparison has been
        performed against a particular git revision.
    -   `"LATEST_VERSION"` is now accepted as argument to `git_revision`
        and resolves to the latest version tag (falling back to the
        current HEAD if the repo has no tags)
-   Extended excerpting functionality by \@leobruneau in
    <https://github.com/johentsch/ms3/pull/105>
    -   It is now possible to replace head and tail of an excerpt with
        rests. This does not look pretty but it is an easy way to create
        audio excerpts starting and ending at the given points in time.
    -   It is now possible to set an arbitrary tempo by inserting an
        invisible metronome mark at the beginning of excerpts.
    -   `score.mscx.store_phrase_excerpts()` makes use of this to omit
        notes before and after the actual phrase
    -   new methods accessible via `score.mscx`:
        -   `store_measures()`
        -   `store_within_phrase_excerpts()`
        -   `store_phrase_endings()`
        -   `store_random_excerpts()`

**Full Changelog**:
<https://github.com/johentsch/ms3/compare/v2.2.2>\...v2.3.0

## Version 2.2.1

-   Form label columns by \@johentsch in
    [https://github.com/johentsch/ms3/pull/98]{.title-ref}\_\_
    -   catches exception when resource descriptor cannot be generated
    -   creates IntervalIndex based on the `quarterbeats_all_endings`
        column
    -   always stores form_labels with a single column level instead of
        MultiIndex

**Full Changelog**:
<https://github.com/johentsch/ms3/compare/v2.2.0>\...v2.2.1

## Version 2.2.0

### What's Changed

-   Changing the instrumentation to "Drumset" by \@arinaLozhkina in
    [https://github.com/johentsch/ms3/pull/84]{.title-ref}\_\_
    -   More robust updating of score instrumentation by modifying
        `metadata.tsv` and calling `ms3 metadata --instrumentation`
    -   Ensures playback with the correct MIDI instrument
    -   Handles a change to `Drumset` correctly in terms of changing
        clef, removing key signature, and playback
-   MSCX class API for creating score excerpts by \@leobruneau in
    [https://github.com/johentsch/ms3/pull/91]{.title-ref}\_\_
    -   score excerpts can now be stored using
        `score.mscx.store_excerpt()`
    -   batch excerpt creation via `MSCX.store_phrase_excerpts()` and
        `MSCX.store_random_excerpts()`
-   Improved algorithm for computing `mc_offset` including bugfix by
    \@johentsch in
    [https://github.com/johentsch/ms3/pull/95]{.title-ref}\_\_
-   Updated schema mechanism following the first trial by \@johentsch in
    [https://github.com/johentsch/ms3/pull/97]{.title-ref}\_\_
    -   `quarterbeats_all_endings` column now added to all facet
        dataframes by default
    -   schema URLs now use the dedicated
        [DCMLab/frictionless_schemas](https://github.com/DCMLab/frictionless_schemas/)
        and the amount of required schemas was reduced drastically
        -   no schemas are stored for `events` anymore
        -   `rests` and `notes_and_rests` do not include non-sensical
            empty columns anymore
        -   `chords` is the only remaining facet where the abundance of
            schemas due to high combinatoriality of column names is
            (somewhat) justified.
    -   Renamed columns:
        -   in unfolded dataframes, `quarterbeats` is now called
            `quarterbeats_playthrough`
        -   in the `chords` facet, `metronome_visible` is now called
            `tempo_visible`
-   documentation now hosted at
    [https://ms3.readthedocs.io/]{.title-ref}\_\_

### New Contributors

-   \@leobruneau made their first contribution in
    [https://github.com/johentsch/ms3/pull/91]{.title-ref}\_\_

**Full Changelog**:
[https://github.com/johentsch/ms3/compare/v2.1.1...v2.2.0]{.title-ref}\_\_

## Version 2.1.1

-   Message headers in `.warnings` files now come without trailing `--`.
-   This version is able to correctly IGNORED_WARNINGS even if the
    header ends on a trailing `--` (copied from a `.warnings` file
    generated by an older version of ms3).
-   adds the low-level function
    `ms3.bs4_parser._MSCX_bs4.make_excerpt`{.interpreted-text
    role="meth"} that returns the new object type
    `ms3.bs4_parser.Excerpt`. High-level API in preparation (#91).

## Version 2.1.0

This update includes a few minor bug fixes but some heavy updating of
the code internals:

-   pandas\>=2.0.0 is now supported
-   the `@function_logger` decorator has been removed and replaced with
    a function argument that defaults to the `module_logger`
-   all modules which have seen a commit since the previous tag have
    been fully linted using pre-commit hooks
-   the filelock problem that made a couple of test fail under Windows
    since the early days has been resolved (by using pytest\'s
    `tmp_path` fixture instad of `NamedTemporaryFile`).
-   `make_ml()` (responsible for creating measure tables) was refactored
    and should be much more legible (and easier to adapt and extend in
    the future)

**Full Changelog**:
[https://github.com/johentsch/ms3/compare/v2.0.1\...v2.1.0]{.title-ref}\_\_

## Version 2.0.1

-   Allow metronome mark to appear in MC 2

## Version 2.0.0

### Breaking changes

-   Renamed MultiIndex levels:
    -   The column `fname` has been renamed to `piece`. This concerns
        especially `metadata.tsv` where it is used as index, but also
        the MultiIndex of concatenated facets such as those output by
        `Parse.get_facet()` or `ms3 transform`.
    -   The last (right-most) index level, which used to be called
        `<facet>_i` in some cases, is now consistently called `i`.
-   When extracting TSV files:
    -   The possibility to assign custom suffixes to the extracted
        facets has been replaced by default suffixes separated by a full
        stop. For example, the notes for the MuseScore file
        `MS3/filename.mscx` will be extracted to
        `notes/filename.notes.tsv` by default.
    -   Every extracted TSV file comes with a JSON descriptor file
        following the [frictionless
        specification](https://specs.frictionlessdata.io/) for metadata.
        This replaces the `csv-metadata.json` files that were following
        the [CSV on the Web](https://csvw.org/) specification.
    -   The frictionless schemas used in the JSON descriptor files are
        stored in the `schemas` folder of the ms3 package in YAML
        format. Their filenames are truncated hashes computed from the
        included column/field names and they are stored in a folder
        pertaining to the facet in question. This comes with the
        advantage that schemas do not have to be written out in every
        descriptor: Instead, the `schema` field contains the URL of the
        schema file, allowing to update the schema specifications at a
        later point, e.g. with added or more elaborate descriptions.
    -   Validation errors are written into `.errors` files stored next
        to the resource descriptor in question.
-   The command `ms3 transform`, by default, outputs the concatenated
    facets as a single ZIP file that comes with a [frictionless
    DataPackage descriptor](https://specs.frictionlessdata.io/) (for the
    parameters added to the command, see below). The concatenated files
    are now named `<corpus_name>.<facet>.tsv` (previously
    `concatenated_<facet>.tsv`).

### New features

-   It is now possible to batch-edit the instrumentation in many scores
    at once by changing the relevant column(s) in `metadata.tsv` and
    calling `ms3 metadata --instrumentation`.
-   Since `ms3 transform` now outputs zipped [frictionless
    DataPackages](https://specs.frictionlessdata.io/) by default
    (meaning that all concatenated facets are described in the same
    package descriptor JSON file), it comes with additional parameters:
    -   `--unzipped` to output the package as uncompressed TSV files
        rather than as single ZIP file.
    -   `--resources` to create a frictionless resource descriptor per
        concatenated facet instead of a package descriptor.
    -   `--safe` to prevent overwriting existing files.
-   The `ms3 extract` command now has a `--corpuswise` option allowing
    to parse and extract one corpus after the other, avoiding the need
    to parse all scores at once and keep them in memory before beginning
    the extraction.
-   The parser throws a warning if a score does not have a metronome
    mark at the beginning (which can be hidden). This is to encourage
    the inclusion of information on the basic beat unit (in 6/8 meter,
    e.g., the metronome unit is typically a dotted quarter) and pace to
    every score for better comparability.

### Bugfixes

-   For the `IGNORED_WARNINGS` file.
-   For the `--threshold` argument of the `ms3 review` command.
-   Writing and reading the `volta_mcs` column of `metadata.tsv`.
-   #60, #63, #78, #79

### Internal changes

-   `utils.py` has been turned into a Python package containing the
    mocules `constants`, `functions`, and `frictionless`.
-   Not using the `frac` alias for `fractions.Fraction` anymore.
-   The version number is not manually stored as a constant, instead it
    is automatically written into `_version.py` upon initialization.

### Other

This version contains the final version of the paper *A parser for
MuseScore 3 files and data factory for annotated music corpora* for
publication in the Journal of Open Source Software (JOSS).

## Version 1.2.12

This last version of ms3 1.x uses the \_version.py file introduced in
8f40b16.

## Version 1.2.11

-   stops writing the version of ms3 into the [.warnings]{.title-ref}
    files to avoid merge conflicts
-   bugfixes for
    -   handling IGNORED_WARNINGS
    -   ms3 review command
    -   overview table written to README

## Version 1.2.10

-   merges old_tests with new_tests
-   correct handling of `labels_cfg`
-   refrains from calling `logging.basicConfig()`
-   unknown TSV types now default to `labels`
-   `conti` now recognized as abbreviation for \"continuation idea\"
-   suppresses warnings about multiple \"Fingering_text\" values

## Version 1.2.9

-   when updating `README.md`:
    -   make 2nd-level heading `## Overview` (instead of first-level)
    -   don\'t output ms3 version (to avoid merge conflicts)
-   small bugfixes in `ms3 review` command

## Version 1.2.8

-   operations.insert_labels_into_score() filters pieces exactly one
    facet to be inserted (e.g. `labels`), not a fuzzy regex (e.g., which
    would include `form_labels` in the filter)

## Version 1.2.7

-   warning files omit system-dependend information from warning headers
    (6764476)
-   bugfixes

## Version 1.2.6

-   changes the behaviour of the `ms3 review` command
    -   after coloring out-of-label notes, issue one warning per dubious
        label
    -   rather than one [warnings.log]{.title-ref} file per corpus,
        create one [\<fname\>.warnings]{.title-ref} file per piece in
        the [reviewed]{.title-ref} folder
-   makes `ms3 empty` work under the new CLI (d8f661a)

## Version 1.2.5

-   `~ms3.Corpus`{.interpreted-text role="obj"} and
    `~ms3.Piece`{.interpreted-text role="obj"} come with the new method
    `count_pieces()`
-   `ms3 transform -D` to concatenate only metadata works
-   `View.fnames_with_incomplete_facets = False` enforces selected
    facets if some have been excluded

## Version 1.2.4

-   segment_by_criterion warns if not IntervalIndex is present d2602617
-   adds missing arguments \'unfold\' and \'interval_index\' to
    Piece.get_parsed() 71f8c3e4
-   when iterating through pieces, skip fnames that don\'t have any
    files under the current view fdce948f

## Version 1.2.3

**ms3 requires Python 3.10**

-   Piece.get_facet() gets parameter \'force\' which defaults to False
    (analogous to the other methods), in order to avoid unsolicited
    score parsing.
-   improves `ms3 transform`:
    -   parse only facets to be concatenated (rather than all TSV files)
    -   do not accidentally output metadata if not requested
-   prevents including \'volta_mcs\' in metadata of pieces that don\'t
    have voltas

## Version 1.2.2

**ms3 requires Python 3.10**

-   removes deprecated elements from tab completion
-   enables view settings when adding new corpora to Parse object
-   small stuff

## Version 1.2.1

**ms3 requires Python 3.10**

-   enables hiding the info prints in
    operations.insert_labels_into_score()
-   adds [filter_other_fnames]{.title-ref} argument to Corpus.add_dir()

## Version 1.2.0

**ms3 requires Python 3.10**

### Extraction of all lyrics

This version enables the extraction of lyrics with all verses.
Previously, only the last verse\'s syllable for any given position was
extracted. The lyrics now can be found in
[lyrics\_\[verse\]]{.title-ref} columns in the chords facet, where
[lyrics_1]{.title-ref} corresponds to the first or only verse.

### Extraction of figured bass

Figured bass labels can now be found in the chords facet tables. Score
that include at least one figure will have a `thoroughbass_duration`
column and each layer of figures comes in a separate
`thoroughbass_layer_#` column. For example, if all stacks of figures
have only layer, there will be only the column `thoroughbass_layer_1`.

### Extraction of custom-named spanners

Spanners with adjusted \"Begin text\" property get their own columns in
the chords tables, containing the relevant subselection of IDs. For
example, if a score contains normal `8va` spanners and others where the
\"Begin text\" has been set to `custom`, all IDs will be combined in the
column `Ottava:8va` as before, but the subset pertaining to the custom
spanners is additionally shown in the column `Ottava:8va_custom`.

### Including and excluding paths

It is now possible to specify specific directories to be included or
excluded from a view, not only folder names.

### New methods and properties

-   `Parse.get_facet()` (singular)
-   `Corpus.fnames`
-   `Corpus.add_dir()`
-   first version of `utils.merge_chords_and_notes()`

## Version 1.1.2

**ms3 requires Python 3.10**

-   Refines the new \"writing score headers\" functionality and makes it
    non-default. User needs to set `ms3 metadata --prelims` which
    replaces the flag `--ignore` that had been introduced in 1.1.1.
-   A couple of bug fixes, including a very important one regarding
    conversion of fifths introduced with b0ce8a1d

## Version 1.1.1

**ms3 requires Python 3.10**

-   enables updating score headers from the respective
    [metadata.tsv]{.title-ref} columns ([title_text]{.title-ref},
    [subtitle_text]{.title-ref}, [composer_text]{.title-ref},
    [lyricist_text]{.title-ref}, and [part_name_text]{.title-ref})
-   Parse, Corpus, and Piece now come with the method keys()

## Version 1.1.0

**ms3 requires Python 3.10**

This version does not throw errors when trying to parse files created by
MuseScore 4. Parsing these files has not sufficiently been tested but so
far it was looking good. The fact that MuseScore 3 is able to read such
files shows that not much has changed in the file format itself.

The command `ms3 convert` has been updated to support MuseScore 4
executables. With the current MuseScore 4.0.0 this is not quite
straightforward because conversion to `.mscz` via the commandline isn\'t
currently working and conversion to `.mscx`, if it works at all, deletes
the contents of the target directory ([issue
#15367](https://github.com/musescore/MuseScore/issues/15367#issuecomment-1369783686)).
The new function `utils.convert_to_ms4()` offers a workaround that
creates temporary directories to store the \"Uncompressed MuseScore
folder\" and then copies the `.mscx` file to the target directory
(default) or zips the temporary directory into an `.mscz` file
(parameter `--format mscz`). For all other target formats, the output
will correspond to what the MuseScore 4 executable yields.

## Version 1.0.4

**ms3 requires Python 3.10**

ms3 has gotten a makeover and does not quite like it did before. The
major changes are:

-   The library is now optimized for one particular folder structure,
    namely `[meta-corpus ->] corpus -> piece`.
-   ms3 now comes with a full-fledged \"views\" feature which lets you
    subselect files in manifold ways.
-   The TSV outputs have gained additional columns. In particular, all
    TSV files now come with the column `quarterbeats` reflecting each
    event\'s offset from the piece\'s beginning.
-   Warnings concerning irregularities, e.g. wrong measure numbering due
    to a cadenza, can now be sanctioned by copying them into an
    IGNORED_WARNINGS file.

### New features

-   Each object that the user interacts with,
    `Parse, Corpus, and Piece`, comes with at least two views, called
    \"default\" and \"all\". The \"default\" view disregards review
    files, scores in convertible formats, and scores that are not listed
    in the top-level `metadata.tsv` file.
-   `metadata.tsv` files, by the virtue of their first column `fname`,
    now serve as authority on what is included in the corpus and what
    belongs together. This column is always unique and supposed to be
    used as index.
-   Suffixed `metadata_<suffix>.tsv` files are loaded as available views
    based on the column `fname` (other columns are disregarded).
-   The Parse object now detects if the passed directory contains
    individual corpora or if it is a corpus itself.
-   Parse objects perform operations by iterating over Corpus objects.
-   Corpus objects perform operations by iterating over Piece objects.
-   Corpus objects reflect exactly one folder, the `corpus_path`, and
    always discover all present files (which can be filtered before the
    actual parsing). Default output paths are derived from it.
-   Piece objects unite the various files pertaining to the same `fname`
    and are able to keep multiple versions of the same type apart (e.g.,
    scores or annotation files) and pick one automatically, if
    necessary, or ask for user input.
-   The command `ms3 review` combines the functionalities of
    `ms3 check`, `ms3 extract`, and `ms3 compare`, and is now the only
    command used in the new `dcml_corpus_workflow` action. For each
    score that has DCML harmony labels, it stores another score and TSV
    file with the suffix `_reviewed`, in the folder `reviewed`.
    -   The score has all out-of-label tones colored in red and
    -   the TSV file contains a report on this coloring procedure. Both
        files are stored in the folder `reviewed` on the top level of
        the corpus.
    -   **(1.0.2)** In addition, if any warnings pop up, they are stored
        in the top-level `warnings.log` file.
-   Inserting labels into scores is accomplished using the new method
    `load_facet_into_scores()` which comes with the optional parameter
    `git_revision` which allows loading TSVs from a specific commit.
-   Therefore, `ms3 compare` (and hence, `ms3 review`) is now able to
    compare the labels in a score with those in a TSV file from an older
    git revision.
-   `ms3 extract -F` extracts form labels and expands them into a
    tree-like view in the output TSV.

### Changes to the interface

-   Many things have been renamed for the benefit of a more homogeneous
    user interface.
    -   Methods previously beginning with `output_` were renamed to
        `store_`.
    -   Parse.parse_mscx() =\> Parse.parse_scores()
-   The properties for retrieving DataFrames from `Score` objects:
    -   are now methods and accept the parameters `unfold` and
        `interval_index`.
    -   return None when a facet is not available.
-   Parsed scores and dataframes are always returned with File object
    that identifies the parsed file in question. This is particularly
    relevant when using the `get_facet()` methods that may return facets
    from parsed TSV files or extract them from the scores, according to
    availability.
-   Gets rid of the argument `simulate` except for writing files.
-   logger_cfg now as \*\*kwargs
-   **(1.0.3)** Currently the `-d/--dir` argument to `ms3` commands
    accepts only one directory, not several.

### Changes to the outputs

-   **(1.0.1)** When unfolding repeats, add the column `mn_playthrough`
    with disambiguated measure Numbers (\'1a\', \'12b\', etc.).
-   The column `label_type` has been replaced and disambiguated into
    `harmony_layer` (0-3, text, Roman numeral, Nashville, guitar chord)
    and `regex_match` (containing the name of the regular expression
    that matched first).
-   Notes tables now come with the two additional columns `name` (e.g.
    \"E#4\") and `octave`. For unpitched instruments, such as drumset,
    the column `name` displays the designated instrument name (which the
    user can modify in MuseScore), and have no value in the `octave`
    columns.
-   For pieces that don\'t have first and second endings, the TSVs come
    without a `volta` column.
-   Extracted metadata
    -   **(1.0.1)** come with the new columns last_mc_unfolded,
        last_mn_unfolded, volta_mcs, guitar_chord_count,
        form_label_count, ms3_version, has_drumset
    -   uses the column `fname` as index
    -   comes with a modified column order
    -   renames the previous column `rel_paths` to subdir, whereas the
        new column `rel_path` contains
    -   include the text fields included in a score. Columns are
        `composer_text`, `title_text`, `subtitle_text`, `lyricist_text`,
        `part_name_text`.
-   Upon a full parse (i.e. if the view has default settings), each
    facet folder gets a `csv-metadata.json` file following the CSVW
    standard. This file indicates the version of ms3 that was used to
    extract the facets. The version is also included in the last row of
    the README.

### Other changes

Many, many bugs have died on the way. Also:

-   Most functions and methods now come with type hints.
-   New unittest suite that makes use of the DCMLab/unittest_metacorpus
    repo and enforces it to be at the correct commit.
-   The parser is now more robust against user-induced strangeness in
    MuseScore files.
-   **(1.0.1)** Repetitions are unfolded for checking the integrity of
    DCML phrase annotations in order to deal with voltas correctly.
-   **(1.0.3)** Pedal notes that have multiple (volta) endings, although
    still not being correctly propagated into each ending, get
    propagated into the first ending, and don\'t cause propagation nor
    the integrity check to fail anymore

## Version 1.0.3

See above, version 1.0.4

## Version 1.0.2

See above, version 1.0.4

## Version 1.0.1

See above, version 1.0.4

## Version 1.0.0

See above, version 1.0.4

## Version 0.5.3

-   recognizes metadata fields `reviewers` and `annotators` also in
    their singular forms
-   adds column `n_onset_positions` to metadata.tsv
-   interval index levels renamed from \'iv\' =\> \'interval\'
-   gets rid of pandas deprecation warnings
-   bug fixes & log messages

## Version 0.5.2

-   the `View` on a `Parse` object can now be subscripted with a
    filename to obtain a `Piece` object, allowing for better access to
    the various files belonging to the same piece (based on their file
    names). These new objects facilitate access to the information which
    previously was available in one row of tge `View.pieces()`
    DataFrame.
-   adds command `ms3 empty` to remove harmony annotations from scores
-   adds command `ms3 add` to add harmony annotations from TSV files to
    scores
-   re-factored `ms3 compare` to use new methods added to `View` objects
-   methods based on `View.iter()` now accept the parameter `fnames` to
    filter out file names not included in the list
-   while adding labels, use fallback values `staff=-1` and `voice=1` if
    not specified

## Version 0.5.1

-   changes to `iter` methods for iterating through DataFrames and
    metadata belonging together:
    -   supressed the second item: instead of
        `(metadata, paths, df1, df2...)` yield `(metadata, df1, df2...)`
        where the metadata dict contains the paths
    -   added methods `iter_transformed()` and `iter_notes()` to `Parse`
        and `View` objects
-   added command `ms3 transform`
    -   used to concatenate all parsed TSVs of a certain type into one
        file including the option to unfold and add quarterbeats
    -   stores them with prefix `concatenated_`; ms3 now ignores all
        files beginning with this prefix
-   changes in default TSV columns
    -   `metadata.tsv` includes the new columns
        -   `length_qb`: a scores length in quarterbeats (including all
            voltas)
        -   `length_qb_unfolded`: the same but with unfolded repeats, if
            any
        -   `all_notes_qb`: the sum of all note durations in
            quarterbeats
        -   `n_onsets`: the number of all onsets
    -   no empty `volta` columns are included (except for measures) when
        no voltas are present

## Version 0.5.0

-

    considerable changes to `Parse` objects (bugs might still be abundant, please report them)

    :   -   abolished custom DataFrame indices

        -

            behaviour shaped towards ms3\'s standard corpus structure

            :   -   automatic detection of corpora and generation of
                    keys
                -   this enables better matching of files that belong
                    together through `View` objects (access via
                    `p['key']`)
                -   new method `iter()` for iterating through metadata
                    and files that belong together

        -   all JSON files passed under the `paths` argument are now
            scanned for a contained list of file paths to be extracted
            (as opposed to before where the JSON file had to be passed
            as a single path)

        -   new iterator `p.annotation_objects()`

-

    new module `transformations`

    :   -   just as `utils`, members can be imported directly via
            `from ms3 import`

        -   includes a couple of functions that were previously part of
            `utils` or `expand_dcml`

        -

            includes a couple of new functions:

            :   -   get_chord_sequences()
                -   group_annotations_by_features()
                -   make_gantt_data()
                -   transform_annotations()
                -   transform_multiple()

-   handling hierarchical localkeys and pedals (i.e. we can modulate to
    the key of `V/III`)

-   Renamed column \'durations_quarterbeats\' to \'duration_qb\'

-   You can now set `interval_index = True` to add quarterbeat columns
    **and** an index with quarterbeat intervals

-   New behaviour of the `folder_re` argument: It now gets to all paths
    matching the regEx rather than stopping at a higher level that
    doesn\'t match. Effectively, this allows, for example, to do
    `Parse(path, folder_re='notes')` to select all files from folders
    called notes.

-   bug fixes (e.g. failing less on incoherent repeat structures)

## Version 0.4.10

-   Enabled extraction of score labels.

-   Made the use of `labels_cfg` more consistent.

-

    improved chord lists:

    :   -   include system and tempo texts
        -   new algorithm for correct spanner IDs (i.e. for Slurs,
            Pedal, HairPins, Ottava)
        -   lyrics: still extracts only the last verse but now in the
            corresponding column, e.g. `lyrics:3` for verse 3.

-

    new feature (still in beta): extraction of form labels

    :   -   `Score.mscx.form_labels`
        -   `Parse.form_labels()`
        -   added `form_labels` -related parameters to
            `Parse.get_lists()` and `Parse.store_lists()`
        -   added `utils.expand_form_labels()` for hierarchical display
            of form labels

## Version 0.4.9

-   enabled `import from ms3` for all utils

-   new command `ms3 update` for converting files and moving annotations
    to the Roman Numeral Analysis layer

-   new command `ms3 metadata` for writing manually changed information
    from `metadata.tsv` to the metadata fields of the corresponding
    MuseScore files

-

    improved the `ms3 extract` command:

    :   -   added option `-D` for extracting and updating `metadata.tsv`
            and `README.md`
        -   added option `-q` for adding \'quarterbeats\' and
            \'durations_quarterbeats\' columns
        -   included default paths for the capital-letter parameters

-

    improved the `ms3 compare` command:

    :   -   now works with \'expanded\' TSVs, too (not only with
            \'labels\')
        -   allows \'label\' column to include NaN values

-

    improvements to Parse() objects:

    :   -   attempts to parse scores that need file conversion (e.g.
            XML, MIDI)
        -   `get_lists()` method now allows for adding the columns
            `quarterbeats` and `durations_quarterbeats`, even without
            unfolding repeats
        -   adding \'quarterbeats\' without unfolding repeats excludes
            voltas
        -   new method `get_tsvs()` for retrieving and concatenating
            parsed TSV files
        -   Parse() now recognizes `metadata.tsv` files, expanded TSVs,
            and TSVs containing cadence labels only
        -   parsed `metadata.tsv` files can be retrieved/included via
            the method `metadata()`
        -   new method `update_metadata()` for the new `ms3 metadata`
            command
        -   decided on standard index levels `rel_paths` and `fnames`
        -   improved matching of corresponding score and TSV files

-

    improvements to Score() objects:

    :   -   new property Score.mscx.volta_structure for retrieving
            information on first and second endings

-

    improvements to Annotations() objects:

    :   -   correct propagation of `localkey` for voltas

-

    improvements to commandline interface:

    :   -   added parameter `-o` for specifying output directory
        -   harmonized the interface of the `ms3 convert` command
        -   parameter `exclude_re` now also filters paths passed via
            `-f`

-

    changed logging behaviours:

    :   -   write only WARNINGs to log file
        -   combine loggers for filenames independently of file
            extensions

-   improved extraction of instrument names for metadata

-   improved `ms3 compare` functionality

-   restructured code architecture

-   renamed master branch to \'main\'

-   many bug fixes

## Version 0.4.8

-   now reads DCML labels with cadence annotations
-   unified command-line interface file options and included
    `-f file.json`
-   Parse got more options for creating DataFrame index levels
-   Parse.measures property for convenience
-   bug fixes for better GitHub workflows

## Version 0.4.7

-

    Labels can be attached to MuseScore\'s Roman Numeral Analysis (RNA) layer

    :   -   parameter [label_type=1]{.title-ref} in both
            [Score.attach_labels()]{.title-ref} and
            [Parse.attach_labels()]{.title-ref}
        -   [Annotations.remove_initial_dots()]{.title-ref} before
            inserting into the RNA layer
        -   [Annotations.add_initial_dots()]{.title-ref} before
            inserting into the absolute chord layer

-   interpret all [#vii]{.title-ref} in major contexts as
    [vii]{.title-ref} when computing chord tones

-   code cosmetics and bug fixes

## Version 0.4.6

-   ms3 extract and Parse.store_lists() now have the option unfold to
    account for repeats
-   minor bug fixes

## Version 0.4.5

-   added \'ms3 compare\' command
-   support for parsing cap, capx, midi, musicxml, mxl, and xml files
    through temporary conversion
-   support for parsing MuseScore 2 files through temporary conversion

## Version 0.4.3

-   added \'ms3 check\' command
-   support of coloured labels
-   write coloured labels to score comparing attached and detached
    labels to each other
-   better interface for defining log file paths (more options, now
    conforming to the Parse.store_lists() interface)
-   fixed erroneous separation of alternative labels

## Version 0.4.2

-   small bug fixes
-   correct computation of chord tones for new DCML syntax elements
    `+M`, `-`, `^`, and `v`

## Version 0.4.1

-   ms3 0.4.1 supports parsing (but not storing) compressed MuseScore
    files (.mscz)
-   Installs \"ms3 convert\" command to your system for batch conversion
    using your local MuseScore installation
-   \"ms3 extract\" command now supports creation of log files
-   take `labels_cfg` into account when creating expanded chord tables

## Version 0.4.0

-   The standard column \'onset\' has been renamed to \'mc_onset\' and
    \'mn_onset\' has been added as an additional standard column.
-   Parse TSV files as Annotations objects
-   Parse.attach_labels() for inserting annotations into MuseScore files
-   Prepare detached labels so that they can actually be attached
-   Install \"ms3 extract\" command to the system
-   Including da capo, dal segno, fine, and coda for calculating
    \'next\' column in measures tables (for correct unfolding of
    repeats)
-   Simulate parsing and table extraction
-   Passing labels_cfg to Score/Parse to control the format of
    annotation lists
-   Easy access to individual parsed files through Parse\[ID\] or
    Parse\[ix\]
-   parse annotation files with diverging column names

## Version 0.3.0

-   Parse.detach_levels() for emptying all parsed scores from
    annotations
-   Parse.store_mscx() for storing altered (e.g. emptied) score objects
    as MuseScore files
-   Parse.metadata() to return a DataFrame with all parsed pieces\'
    metadata
-   Parse.get_labels() to retrieve labels of a particular kind
-   Parse.info() has improved the information that objects return about
    themselves
-   Parse.key for a quick overview of the files of a given key
-   Parse can be used with a custom index instead of IDs \[an ID is an
    (key, i) tuple\]
-   Score.store_list() for easily storing TSVs
-   renamed Score.output_mscx() to store_mscx() for consistency.
-   improved expansion of DCML harmony labels

## Version 0.2.0

Beta stage:

-   attaching and detaching labels
-   parsing multiple pieces at once
-   extraction of metadata from scores
-   inclusion of staff text, dynamics and articulation in chord lists,
    added \'auto\' mode
-   conversion of MuseScore\'s encoding of absolute chords
-   first version of docs

## Version 0.1.3

At this stage, the library can parse MuseScore 3 files to different
types of lists:

-   measures

-

    chords (= groups of notes)

    :   -   including slurs and spanners such as pedal, 8va or hairpin
            markings
        -   including lyrics

-   notes

-   harmonies

and also some basic metadata.

## Version 0.1.0

-   Basic parser implemented
-   Logging
-   Measure lists
