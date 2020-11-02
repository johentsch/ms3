======
Manual
======

This page is a detailed guide for using ms3 for different tasks. It supposes you are working in an interactive Python
interpreter such as IPython, Jupyter, Google Colab, or just the console.


Good to know
============

Terminology
-----------

.. _mc_vs_mn:

Measure counts (MC) vs. measure numbers (MN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measure counts are strictly increasing numbers for all <measure> nodes in the score, regardless of their length. This
information is crucial for correctly addressing positions in a MuseScore file and are shown in the software's status
bar. The first measure is always counted as 1 (following MuseScore's convention), even if it is an anacrusis.

Measure numbers are the traditional way by which humans refer to positions in a score. They follow a couple of
conventions which can be summarised as counting complete bars. Quite often, a complete bar (MN) can be made up of
two <measure> nodes (MC). In the context of this library, score addressability needs to be maintained for humans and
computers, therefore a mapping MC -> MN is preserved in the score information DataFrames.

.. _quarter_beats:

Quarter Beats
^^^^^^^^^^^^^

A quarter beat always has the length of a quarter note. It is used as a standard unit to express positions and durations
independently of the beat size suggested by the :ref:`time signature <timesig>` (e.g. three eighths), and can be
:ref:`converted to a different beat size  <converting_quarter_beats>`.

If the guidelines say *"xy is expressed as/in quarter beats"*,
it **actually** means "as fractions of a whole note". So the duration of a half note, for example, is expressed
as ``1/2``, and not as ``2`` (which but be the multiplier of quarter beats, or understanding quarter beats as unit).
This is simply a terminological convention to speak consistently of beat sizes.

Functionality
-------------

.. _converting_quarter_beats:

Converting :ref:`quarter_beats`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*TODO*

.. _read_only:

Read-only mode
^^^^^^^^^^^^^^

For parsing faster using less memory.

Using the library
=================

Parsing a single score
----------------------

.. rst-class:: bignums

1. Import the library.

    To parse a single score, we will use the class ``ms3.Score``. We could import the whole library:

    .. code-block:: python

        >>> import ms3
        >>> s = ms3.Score()

    or simply import the class:

    .. code-block:: python

        >>> from ms3 import Score
        >>> s = Score()


2. Locate the `MuseScore 3 <https://musescore.org/en/download>`__ score you want to parse.

    Make sure it is uncompressed. i.e. it has the extension ``.mscx`` and not ``.mscz``.

    .. tip::

        MSCZ files are ZIP files containing the uncompressed MSCX. A later version of ms3 will be able to deal with MSCZ, too.


    In the examples, we parse the annotated first page of Giovanni
    Battista Pergolesi's influential *Stabat Mater*. The file is called ``stabat.mscx`` and can be downloaded from
    `here <https://raw.githubusercontent.com/johentsch/ms3/master/docs/stabat.mscx>`__ (open link and key ``Ctrl + S`` to save the file
    or right-click on the link to ``Save link as...``).

3. Create a ``ms3.Score`` object.

    In the example, the MuseScore 3 file is located at ``~/ms3/docs/stabat.mscx`` so we can simply create the object
    and bind it to the variable ``s`` like so:

    .. code-block:: python

        >>> from ms3 import Score
        >>> s = Score('~/ms3/docs/stabat.mscx')

4. Inspect the object.

    To have a look at the created object we can simply evoke its variable:

    .. code-block:: python

        >>> s
        MuseScore file
        --------------

        ~/ms3/docs/stabat.mscx

        Attached annotations
        --------------------

        48 labels:
        staff  voice  label_type
        3      2      dcml          48


Parsing options
^^^^^^^^^^^^^^^

.. automethod:: ms3.score.Score.__init__
    :noindex:

Parsing multiple scores
-----------------------

.. rst-class:: bignums

1. Import the library.

    To parse multiple scores, we will use the class ``ms3.Parse``. We could import the whole library:

    .. code-block:: python

        >>> import ms3
        >>> p = ms3.Parse()

    or simply import the class:

    .. code-block:: python

        >>> from ms3 import Parse
        >>> p = Parse()


2. Locate the folder containing MuseScore files.

    In this example, we are going to parse all files included in
    `this older version of ms3's Git repo <https://github.com/johentsch/ms3/tree/e25ec55c3bbf07664436a73db956f4855890516c>`__ which has been
    `cloned <https://www.atlassian.com/git/tutorials/setting-up-a-repository/git-clone>`__
    into the home directory and therefore has the path ``~/ms3``.

3. Create a ``ms3.Parse`` object

    The object is created by calling it with the directory to scan, and bound
    to the variable ``p``. By default, scores are grouped by the subdirectories
    they are in and one key is automatically created for each of them to access
    the files separately.

    .. code-block:: python

        >>> from ms3 import Parse
        >>> p = Parse('~/ms3')
        >>> p
        10 files.
        KEY       -> EXTENSIONS
        -----------------------
        docs      -> {'.mscx': 4}
        tests/MS3 -> {'.mscx': 6}

        No mscx files have been parsed.

    As long as you always want to perform actions on all files, it may be convenient
    to assign a simple key. This might be also useful if you want to add several
    directories to the object using ``p.add_dir()``:

    .. code-block:: python

        >>> p = Parse('~/ms3', key='test')
        >>> p.add_dir('~/other_folder', key='other')
        >>> p
        237 files.
        KEY   -> EXTENSIONS
        -------------------
        test  -> {'.mscx': 10}
        other -> {'.mscx': 227}

        No mscx files have been parsed.

    Note that the same 10 files that were distributed over two keys in the previous
    example are now grouped under the key 'test'.

4. Parse the scores.

    In order to simply parse all registered MuseScore files, call ``p.parse_mscx()``.
    Instead, you can pass the argument ``key`` to parse only one (or several)
    selected group(s) to save time. The argument ``level`` controls how many
    log messages you see; here, it is set to 'critical' or 'c' to suppress all
    warnings:

    .. code-block:: python

        >>> p.parse_mscx(keys='test', level='c')
        >>> p
        KEY   -> EXTENSIONS
        -------------------
        test  -> {'.mscx': 10}
        other -> {'.mscx': 227}

        10/237 MSCX files have been parsed.
        7 of them have annotations attached.
        KEY  -> ANNOTATION LAYERS
        -------------------------
        test -> staff  voice  label_type
             -> 2      1      dcml          167
             -> 3      1      dcml          26
             ->        2      dcml          48
             -> 1      1      0             7
             ->               3             166
             ->               dcml          568

    As we can see, only the 10 files with the key 'test' were parsed and the
    table shows an overview of the counts of the included label types in the
    different notational layers (i.e. staff & voice). For example, the 7 files
    that include labels, have in their respective upper layers (staff 1, voice 1),
    568 DCML harmony labels, 166 absolute chord labels (type 3) and 7 random
    strings (type 0) overall.



Extracting score information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each of the :ref:`DataFrames holding score information<tabular_info>` can be
automatically stored for every score. To select one or several aspects out of
``{notes, measures, rests, notes_and_rests, events, labels, chords, expanded}``,
it is enough to pass the respective ``_folder`` parameter to
:py:meth:`~ms3.parse.Parsed.store_lists` distinguishing where to store the TSV
files. Additionally, the method accepts one ``_suffix`` parameter per aspect,
i.e. a slug added to the respective filenames. If the parameter
``simulate=True`` is passed, no files are written but the file paths to be
created are returned. Since corpora might have quite diverse directory structures,
ms3 gives you various ways of specifying folders which will be explained in detail
in the following section.

Briefly, the rules for specifying the folders are as follows:

* absolute folder (e.g. ``~/labels``): Store all files in this particular folder without creating subfolders.
* relative folder starting with ``./`` or ``../``: relative folders are created
  "at the end" of the original subdirectory structure, i.e. relative to the MuseScore
  files.
* relative folder not starting with ``./`` or ``../`` (e.g. ``rests``): relative
  folders are created at the top level (of the original directory or the specified
  ``root_dir``) and the original subdirectory structure is replicated
  in each of them.

To see examples for the three possibilities, see the following section.

.. _specifying_folders:

Specifying folders
^^^^^^^^^^^^^^^^^^

Consider a two-level folder structure contained in the root directory ``.``
which is the one passed to :obj:`Parse`:

.. code-block:: console

  .
  ├── docs
  │   ├── cujus.mscx
  │   ├── o_quam.mscx
  │   ├── quae.mscx
  │   └── stabat.mscx
  └── tests
      └── MS3
          ├── 05_symph_fant.mscx
          ├── 76CASM34A33UM.mscx
          ├── BWV_0815.mscx
          ├── D973deutscher01.mscx
          ├── Did03M-Son_regina-1762-Sarti.mscx
          └── K281-3.mscx

The first level contains the subdirectories `docs` (4 files) and `tests`
(6 files in the subdirectory `MS3`). Now we look at the three different ways to specify folders for storing notes and
measures.

Absolute Folders
""""""""""""""""

When we specify absolute paths, all files are stored in the specified directories.
In this example, the measures and notes are stored in the two specified subfolders
of the home directory `~`, regardless of the original subdirectory structure.

.. code-block:: python

  >>> p.store_lists(notes_folder='~/notes', measures_folder='~/measures')

.. code-block:: console

  ~
  ├── measures
  │   ├── 05_symph_fant.tsv
  │   ├── 76CASM34A33UM.tsv
  │   ├── BWV_0815.tsv
  │   ├── cujus.tsv
  │   ├── D973deutscher01.tsv
  │   ├── Did03M-Son_regina-1762-Sarti.tsv
  │   ├── K281-3.tsv
  │   ├── o_quam.tsv
  │   ├── quae.tsv
  │   └── stabat.tsv
  └── notes
      ├── 05_symph_fant.tsv
      ├── 76CASM34A33UM.tsv
      ├── BWV_0815.tsv
      ├── cujus.tsv
      ├── D973deutscher01.tsv
      ├── Did03M-Son_regina-1762-Sarti.tsv
      ├── K281-3.tsv
      ├── o_quam.tsv
      ├── quae.tsv
      └── stabat.tsv

Relative Folders
""""""""""""""""

In contrast, specifying relative folders recreates the original subdirectory structure.
There are two different possibilities for that. The first possibility is naming
relative folder names, meaning that the subdirectory structure (``docs`` and ``tests``)
is recreated in each of the folders:

.. code-block:: python

    >>> p.store_lists(root_dir='~/tsv', notes_folder='notes', measures_folder='measures')

.. code-block:: console

    ~/tsv
    ├── measures
    │   ├── docs
    │   │   ├── cujus.tsv
    │   │   ├── o_quam.tsv
    │   │   ├── quae.tsv
    │   │   └── stabat.tsv
    │   └── tests
    │       └── MS3
    │           ├── 05_symph_fant.tsv
    │           ├── 76CASM34A33UM.tsv
    │           ├── BWV_0815.tsv
    │           ├── D973deutscher01.tsv
    │           ├── Did03M-Son_regina-1762-Sarti.tsv
    │           └── K281-3.tsv
    └── notes
        ├── docs
        │   ├── cujus.tsv
        │   ├── o_quam.tsv
        │   ├── quae.tsv
        │   └── stabat.tsv
        └── tests
            └── MS3
                ├── 05_symph_fant.tsv
                ├── 76CASM34A33UM.tsv
                ├── BWV_0815.tsv
                ├── D973deutscher01.tsv
                ├── Did03M-Son_regina-1762-Sarti.tsv
                └── K281-3.tsv

Note that in this example, we have specified a ``root_dir``. Leaving this argument
out will create the same structure in the directory from which the :obj:`Parse`
object was created, i.e. the folder structure would be:

.. code-block:: console

    .
    ├── docs
    ├── measures
    │   ├── docs
    │   └── tests
    │       └── MS3
    ├── notes
    │   ├── docs
    │   └── tests
    │       └── MS3
    └── tests
        └── MS3

If, instead, you want to create the specified relative folders relative to each
MuseScore file's location, specify them with an initial dot. ``./`` means
"relative to the original path" and ``../`` one level up from the original path.
To exemplify both:

.. code-block:: python

    >>> p.store_lists(root_dir='~/tsv', notes_folder='./notes', measures_folder='../measures')

.. code-block:: console

    ~/tsv
    ├── docs
    │   └── notes
    │       ├── cujus.tsv
    │       ├── o_quam.tsv
    │       ├── quae.tsv
    │       └── stabat.tsv
    ├── measures
    │   ├── cujus.tsv
    │   ├── o_quam.tsv
    │   ├── quae.tsv
    │   └── stabat.tsv
    └── tests
        ├── measures
        │   ├── 05_symph_fant.tsv
        │   ├── 76CASM34A33UM.tsv
        │   ├── BWV_0815.tsv
        │   ├── D973deutscher01.tsv
        │   ├── Did03M-Son_regina-1762-Sarti.tsv
        │   └── K281-3.tsv
        └── MS3
            └── notes
                ├── 05_symph_fant.tsv
                ├── 76CASM34A33UM.tsv
                ├── BWV_0815.tsv
                ├── D973deutscher01.tsv
                ├── Did03M-Son_regina-1762-Sarti.tsv
                └── K281-3.tsv

The ``notes`` folders are created in directories where MuseScore files are located,
and the ``measures`` folders one directory above, respectively. Leaving out the
``root_dir`` argument would lead to the same folder structure but in the directory
from which the :obj:`Parse` object has been created. In a similar manner,
the arguments ``p.store_lists(notes_folder='.', measures_folder='.')`` would create
the TSV files just next to the MuseScore files. However, this would lead to warnings
such as

.. warning::

    The notes at ~/ms3/docs/cujus.tsv have been overwritten with measures.

In such a case we need to specify a suffix for at least one of both aspects:

.. code-block:: python

    p.store_lists(notes_folder='.', notes_suffix='_notes',
                  measures_folder='.', measures_suffix='_measures')

Examples
""""""""

Before you are sure to have picked the right parameters for your desired output,
you can simply use the ``simulate=True`` argument which lets you view the paths
without actually creating any files. In this variant, all aspects are stored each
in individual folders but with identical filenames:

.. code-block:: python

    >>> p = Parse('~/ms3/docs', key='pergo')
    >>> p.parse_mscx()
    >>> p.store_lists(  notes_folder='./notes',
                        rests_folder='./rests',
                        notes_and_rests_folder='./notes_and_rests',
                        simulate=True
                        )
    ['~/ms3/docs/notes/cujus.tsv',
     '~/ms3/docs/rests/cujus.tsv',
     '~/ms3/docs/notes_and_rests/cujus.tsv',
     '~/ms3/docs/notes/o_quam.tsv',
     '~/ms3/docs/rests/o_quam.tsv',
     '~/ms3/docs/notes_and_rests/o_quam.tsv',
     '~/ms3/docs/notes/quae.tsv',
     '~/ms3/docs/rests/quae.tsv',
     '~/ms3/docs/notes_and_rests/quae.tsv',
     '~/ms3/docs/notes/stabat.tsv',
     '~/ms3/docs/rests/stabat.tsv',
     '~/ms3/docs/notes_and_rests/stabat.tsv']


In this variant, the different ways of specifying folders are exemplified. To demonstrate all subtleties we parse the
same four files but this time from the perspective of ``~/ms3``:

.. code-block:: python

    >>> p = Parse('~/ms3', folder_re='docs', key='pergo')
    >>> p.parse_mscx()
    >>> p.store_lists(  notes_folder='./notes',            # relative to ms3/docs
                        measures_folder='../measures',     # one level up from ms3/docs
                        rests_folder='rests',              # relative to the parsed directory
                        labels_folder='~/labels',          # absolute folder
                        expanded_folder='~/labels', expanded_suffix='_exp',
                        simulate = True
                        )
    ['~/ms3/docs/notes/cujus.tsv',
     '~/ms3/rests/docs/cujus.tsv',
     '~/ms3/measures/cujus.tsv',
     '~/labels/cujus.tsv',
     '~/labels/cujus_exp.tsv',
     '~/ms3/docs/notes/o_quam.tsv',
     '~/ms3/rests/docs/o_quam.tsv',
     '~/ms3/measures/o_quam.tsv',
     '~/labels/o_quam.tsv',
     '~/labels/o_quam_exp.tsv',
     '~/ms3/docs/notes/quae.tsv',
     '~/ms3/rests/docs/quae.tsv',
     '~/ms3/measures/quae.tsv',
     '~/labels/quae.tsv',
     '~/labels/quae_exp.tsv',
     '~/ms3/docs/notes/stabat.tsv',
     '~/ms3/rests/docs/stabat.tsv',
     '~/ms3/measures/stabat.tsv',
     '~/labels/stabat.tsv',
     '~/labels/stabat_exp.tsv']

.. _column_names:

Column Names
============

General Columns
---------------

.. _mc:

**mc** Measure Counts
^^^^^^^^^^^^^^^^^^^^^

Measure count, identifier for the measure units in the XML encoding.
Always starts with 1 for correspondence to MuseScore's status bar.

.. _mn:

**mn** Measure Numbers
^^^^^^^^^^^^^^^^^^^^^^

Measure number, continuous count of complete measures as used in printed editions.
Starts with 1 except for pieces beginning with a pickup measure, numbered as 0.

.. _onset:

**onsets**
^^^^^^^^^^
The value for ``onset`` represents, expressed as :ref:`quarter beats <quarter_beats>`, a position in a measure where ``0``
corresponds to the earliest possible position (in most cases beat 1), and some other fraction corresponds to an onset's offset from ``0``.
:ref:`Quarter beats <quarter_beats>` can be :ref:`converted to beats <converting_quarter_beats>`, e.g. to half beats or dotted eighth beats;
However, the operation may rely on the value of :ref:`mc_offset <mc_offset>`.

.. tip::

    When loading a table from a file, it is recommended to parse the text of this
    column with :obj:`fractions.Fraction` to be able to calculate with the values.
    MS3 does this automatically.

Measures
--------

.. _act_dur:

**act_dur** Actual duration of a measure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The value of ``act_dur`` in most cases equals the time signature, expressed as a fraction; meaning for example that
a "normal" measure in 6/8 has ``act_dur = 3/4``. If the measure has an irregular length, for example a pickup measure
of length 1/8, would have ``act_dur = 1/8``.

The value of ``act_dur`` plays an important part in inferring :ref:`MNs <mn>`
from :ref:`MCs <mc>`. See also the columns :ref:`dont_count <dont_count>` and :ref:`numbering_offset <numbering_offset>`.

.. _barline:

**barline**
^^^^^^^^^^^

The column ``barline`` encodes information about the measure's final bar line.

.. _breaks:

**breaks**
^^^^^^^^^^

The column ``breaks`` may include three different values: ``{'line', 'page', 'section'}`` which represent the different
breaks types. In the case of section breaks, MuseScore

.. _dont_count:

**dont_count** Measures excluded from bar count
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a binary value that corresponds to MuseScore's setting ``Exclude from bar count`` from the ``Bar Properties`` menu.
The value is ``1`` for pickup bars, second :ref:`MCs <mc>` of divided :ref:`MNs <mn>` and some volta measures,
and ``NaN`` otherwise.

.. _keysig:

**keysig** Key Signatures
^^^^^^^^^^^^^^^^^^^^^^^^^

The feature ``keysig`` represents the key signature of a particular measure.
It is an integer which, if positive, represents the number of sharps, and if
negative, the number of flats. E.g.: ``3``: three sharps, ``-2``: two flats,
``0``: no accidentals.

.. _mc_offset:

**mc_offset** Offset of a MC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The column ``mc_offset`` , in most cases, has the value ``0`` because it expresses the deviation of this MC's
:ref:`onset <onset>` ``0`` (beginning of the MC)
from beat 1 of the corresponding MN. If the value is a fraction > 0, it means that this MC is part of a MN which is
composed of at least two MCs, and it expresses the current MC's offset in terms of the duration of all (usually 1) preceding MCs
which are also part of the corresponding MN. In the standard case that one MN would be split in two MCs, the first MC
would have mc_offset = ``0`` , and the second one mc_offset = ``the previous MC's`` :ref:`act_dur <act_dur>` .

.. _next:

**next**
^^^^^^^^

Every cell in this column has at least one integer, namely the MC of the subsequent bar, or ``-1`` in the cast of the last.
In the case of repetitions, measures can have more than one subsequent MCs, in which case the integers are separated by
``', '`` .

The column is used for checking whether :ref:`irregular measure lengths <act_dur>` even themselves out because otherwise
the inferred MNs might be wrong. Also, it is needed for MS3's unfold repeats functionality (TODO).

.. topic:: Developers

    Within MS3, the ``next`` column holds tuples, which MS3 should normally store as strings without parenthesis. For
    example, the tuple ``(17, 1)`` is stored as ``'17, 1'``. However, users might have extracted and stored a raw DataFrame
    from a :obj:`Score` object and MS3 needs to handle both formats.

.. _numbering_offset:

**numbering_offset** Offsetting MNs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MuseScore's measure number counter can be reset at a given MC by using the ``Add to bar number`` setting from the
``Bar Properties`` menu. If ``numbering_offset`` ≠ 0, the counting offset is added to the current MN and all subsequent
MNs are inferred accordingly.

Scores which include several pieces (e.g. in variations or a suite),
sometimes, instead of using section :ref:`breaks <breaks>`, use ``numbering_offset`` to simulate a restart for counting
:ref:`MNs <mn>` at every new section. This leads to ambiguous MNs.



.. _repeats:

**repeats**
^^^^^^^^^^^

The column ``repeats`` indicates the presence of repeat signs and can have the values
``{'start', 'end', 'startend', 'firstMeasure', 'lastMeasure'}``. MS3 performs a test on the
repeat signs' plausibility and throws warnings when some inference is required for this.

The ``repeats`` column needs to have the correct repeat sign structure in order to have a correct :ref:`next <next>`
column which, in return, is required for MS3's unfolding repetitions functionality. (TODO)


.. _timesig:

**timesig** Time Signatures
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The time signature ``timesig`` of a particular measure is expressed as a string, e.g. ``'2/2'``.
The :ref:`actual duration <act_dur>` of a measure can deviate from the time signature for notational reasons: For example,
a pickup bar could have an actual duration of ``1/4``  but still be part of a ``'3/8'`` meter, which usually
has an actual duration of ``3/8``.

.. tip::

    When loading a table from a file, time signatures are not parsed as fractions because then both
    ``'2/2'`` and ``'4/4'``, for example, would become ``1``.

.. _volta:

**volta**
^^^^^^^^^

In the case of first and second (third etc.) endings, this column holds the number of every "bracket", "house", or *volta*,
which should increase from 1. This is required for MS3's unfold repeats function (TODO) to work.

The MNs for all voltas except those of the first one need to be amended to match those of the
first volta. In the case where these voltas have only one measure each, the :ref:`dont_count <dont_count>` option suffices. If
the voltas have more than one measure, the :ref:`numbering_offset <numbering_offset>` setting needs to be used.
