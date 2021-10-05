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


.. _onsets:

Onset positions
^^^^^^^^^^^^^^^

Onsets express positions of events in a score as their distance from the beginning of the corresponding
:ref:`MC or MN <mc_vs_mn>`. The distances are expressed as fractions of a whole note. In other words, beat 1 has
onset ``0``, an event on beat 2 of a 4/4 meter has onset ``1/4`` and so on.

Since there are two ways of referencing measures (MC and MN), there are also two ways of expressing onsets:

* ``mc_onset`` expresses the distance from the corresponding MC
* ``mn_onset`` expresses the distance from the corresponding MN

In most cases, the two values value will be identical, but take as an example the case where a 4/4 measure with MN 8
is divided into MC 9 of length 3/4 and MC 10 of length 1/4 because of a repeat sign or a double bar line. Since MC 9
corresponds to the first part of MN 8, the two onset values are identical. But for the anacrusis on beat 4, the values
differ: ``mc_onset`` is ``0`` but ``mn_onset`` is ``3/4`` because this is the distance from MN 8.

.. _read_only:

Read-only mode
^^^^^^^^^^^^^^

For parsing faster using less memory. Scores parsed in read-only mode cannot be changed because the original
XML structure is not kept in memory.

Parsing
=======

This chapter explains how to

* parse a single score to access and manipulate the contained information using a :py:class:`~ms3.score.Score` object
* parse a group of scores to access and manipulate the contained information using a :py:class:`~ms3.parse.Parse` object.



Parsing a single score
----------------------

.. rst-class:: bignums

1. Import the library.

    To parse a single score, we will use the class :py:class:`~ms3.score.Score`. We could import the whole library:

    .. code-block:: python

        >>> import ms3
        >>> s = ms3.Score()

    or simply import the class:

    .. code-block:: python

        >>> from ms3 import Score
        >>> s = Score()


2. Locate the `MuseScore 3 <https://musescore.org/en/download>`__ score you want to parse.

    .. tip::

        MSCZ files are ZIP files containing the uncompressed MSCX. In order to trace the score's version history,
        it is recommended to always work with MSCX files.


    In the examples, we parse the annotated first page of Giovanni
    Battista Pergolesi's influential *Stabat Mater*. The file is called ``stabat.mscx`` and can be downloaded from
    `here <https://raw.githubusercontent.com/johentsch/ms3/master/docs/stabat.mscx>`__ (open link and key ``Ctrl + S`` to save the file
    or right-click on the link to ``Save link as...``).

3. Create a :py:class:`~ms3.score.Score` object.

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
        staff  voice  label_type  color_name
        3      2      0 (dcml)    default       48

    .. .. program-output:: python examples/parse_single_score.py



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

    In this example, we are going to parse all files included in the `ms3 repository <https://github.com/johentsch/ms3>`__ which has been
    `cloned <https://www.atlassian.com/git/tutorials/setting-up-a-repository/git-clone>`__
    into the home directory and therefore has the path ``~/ms3``.

3. Create a :py:class:`~ms3.parse.Parse` object

    The object is created by calling it with the directory to scan, and bound
    to the variable ``p``. By default, scores are grouped by the subdirectories
    they are in and one key is automatically created for each of them to access
    the files separately.

    .. code-block:: python

        >>> from ms3 import Parse
        >>> p = Parse('~/ms3')
        >>> p

    .. program-output:: python examples/parse_directory.py

    By default, present TSV files are detected and can be parsed as well, allowing one to access already extracted
    information without parsing the scores anew. In order to select only particular files, a regular expression
    can be passed to the parameter ``file_re``. In the following example, only files ending on ``mscx`` are collected
    in the object (``$`` stands for the end of the filename, without it, files including the string 'mscx' anywhere
    in their names would be selected, too):

    .. code-block:: python

        >>> from ms3 import Parse
        >>> p = Parse('~/ms3', file_re='mscx$', key='ms3')
        >>> p

    .. program-output:: python examples/parse_directory_mscx.py

    In this example, we assigned the key ``'ms3'``. Note that the same MSCX files that were distributed over several keys
    in the previous example are now grouped together. Keys allow operations to be performed on a particular group of
    selected files. For example, we could add MSCX files from another folder using the method
    :py:meth:`~ms3.parse.Parse.add_dir` and the key ``'other'``:

    .. code-block:: python

        >>> p.add_dir('~/other_folder', file_re='mscx$', key='other')
        >>> p

    .. program-output:: python examples/parse_other_directory.py

    Most methods of the :py:class:`~ms3.parse.Parse` object have a ``keys`` parameter to perform an operation of a particular group of files.

4. Parse the scores.

    In order to simply parse all registered MuseScore files, call the method :py:meth:`~ms3.parse.Parse.parse_mscx`.
    Instead, you can pass the argument ``keys`` to parse only one (or several)
    selected group(s) to save time. The argument ``level`` controls how many
    log messages you see; here, it is set to 'critical' or 'c' to suppress all
    warnings:

    .. code-block:: python

        >>> p.parse_mscx(keys='ms3', level='c')
        >>> p

    .. program-output:: python examples/parse_key.py

    As we can see, only the files with the key 'ms3' were parsed and the
    table shows an overview of the counts of the included label types in the
    different notational layers (i.e. staff & voice), grouped by their colours.

Parsing options
^^^^^^^^^^^^^^^

.. automethod:: ms3.parse.Parse.__init__
    :noindex:


Extracting score information
============================

One of ms3's main functionalities is storing the information contained in parsed scores as tabular files (TSV format).
More information on the generated files is summarized :ref:`here <tabular_info>`

Using the commandline
---------------------

The most convenient way to achieve this is the command ``ms3 extract`` and its capital-letter parameters summarize
the available tables:

.. code-block:: console

    -M [folder], --measures [folder]
                          Folder where to store TSV files with measure information needed for tasks such as unfolding repetitions.
    -N [folder], --notes [folder]
                          Folder where to store TSV files with information on all notes.
    -R [folder], --rests [folder]
                          Folder where to store TSV files with information on all rests.
    -L [folder], --labels [folder]
                          Folder where to store TSV files with information on all annotation labels.
    -X [folder], --expanded [folder]
                          Folder where to store TSV files with expanded DCML labels.
    -E [folder], --events [folder]
                          Folder where to store TSV files with all events (notes, rests, articulation, etc.) without further processing.
    -C [folder], --chords [folder]
                          Folder where to store TSV files with <chord> tags, i.e. groups of notes in the same voice with identical onset and duration. The tables include lyrics, slurs, and other markup.
    -D [path], --metadata [path]
                          Directory or full path for storing one TSV file with metadata. If no filename is included in the path, it is called metadata.tsv

The typical way to use this command for a corpus of scores is to keep the MuseScore files in a subfolder (called,
for example, ``MS3``) and to use the parameters' default values, effectively creating additional subfolders for each
extracted aspect next to each folder containing MuseScore files. For example if we take the folder structure of
the `ms3 repository <https://github.com/johentsch/ms3>`__:

.. code-block:: console

    ms3
    ├── docs
    │   ├── cujus.mscx
    │   ├── o_quam.mscx
    │   ├── quae.mscx
    │   └── stabat.mscx
    └── tests
        ├── MS3
        │   ├── 05_symph_fant.mscx
        │   ├── 76CASM34A33UM.mscx
        │   ├── BWV_0815.mscx
        │   ├── D973deutscher01.mscx
        │   ├── Did03M-Son_regina-1762-Sarti.mscx
        │   ├── K281-3.mscx
        │   └── stabat_03_coloured.mscx
        └── repeat_dummies
            ├── repeats0.mscx
            ├── repeats1.mscx
            └── repeats2.mscx

Upon calling ``ms3 extract -N``, two new ``notes`` folders containing note lists are created:

.. code-block:: console

    ms3
    ├── docs
    │   ├── cujus.mscx
    │   ├── o_quam.mscx
    │   ├── quae.mscx
    │   └── stabat.mscx
    ├── notes
    │   ├── cujus.tsv
    │   ├── o_quam.tsv
    │   ├── quae.tsv
    │   └── stabat.tsv
    └── tests
        ├── MS3
        │   ├── 05_symph_fant.mscx
        │   ├── 76CASM34A33UM.mscx
        │   ├── BWV_0815.mscx
        │   ├── D973deutscher01.mscx
        │   ├── Did03M-Son_regina-1762-Sarti.mscx
        │   ├── K281-3.mscx
        │   └── stabat_03_coloured.mscx
        ├── notes
        │   ├── 05_symph_fant.tsv
        │   ├── 76CASM34A33UM.tsv
        │   ├── BWV_0815.tsv
        │   ├── D973deutscher01.tsv
        │   ├── Did03M-Son_regina-1762-Sarti.tsv
        │   ├── K281-3.tsv
        │   ├── repeats0.tsv
        │   ├── repeats1.tsv
        │   ├── repeats2.tsv
        │   └── stabat_03_coloured.tsv
        └── repeat_dummies
            ├── repeats0.mscx
            ├── repeats1.mscx
            └── repeats2.mscx

We witness this behaviour because the default value is ``../notes``, interpreted as relative path in relation to
each MuseScore file. Alternatively, a **relative path** can be specified **without** initial ``./`` or ``../``,
e.g. ``ms3 extract -N notes``, to store the note lists in a recreated sub-directory structure:

.. code-block:: console

    ms3
    ├── docs
    ├── notes
    │   ├── docs
    │   └── tests
    │       ├── MS3
    │       └── repeat_dummies
    └── tests
        ├── MS3
        └── repeat_dummies

A third option consists in specifying an **absolute path** which causes all note lists to be stored in the specified
folder, e.g. ``ms3 extract -N ~/notes``:

.. code-block:: console

    ~/notes
    ├── 05_symph_fant.tsv
    ├── 76CASM34A33UM.tsv
    ├── BWV_0815.tsv
    ├── cujus.tsv
    ├── D973deutscher01.tsv
    ├── Did03M-Son_regina-1762-Sarti.tsv
    ├── K281-3.tsv
    ├── o_quam.tsv
    ├── quae.tsv
    ├── repeats0.tsv
    ├── repeats1.tsv
    ├── repeats2.tsv
    ├── stabat_03_coloured.tsv
    └── stabat.tsv

Note that this leads to problems if MuseScore files from different subdirectories have identical filenames.
In any case it is good practice to not use nested folders to allow for easier file access. For example, a typical
`DCML corpus <https://github.com/DCMLab/dcml_corpora>`__ will store all MuseScore files in the ``MS3`` folder and
include at least the folders created by ``ms3 extract -N -M -X``:

.. code-block:: console

    .
    ├── harmonies
    ├── measures
    ├── MS3
    └── notes


Extracting score information manually
-------------------------------------

What ``ms3 extract`` effectively does is creating a :py:class:`~ms3.parse.Parse` object, calling its method
:py:meth:`~ms3.parse.Parse.parse_mscx` and then :py:meth:`~ms3.parse.Parse.store_lists`. In addition to the
command, the method allows for storing two additional aspects, namely ``notes_and_rests`` and ``cadences`` (if
the score contains cadence labels). For each of the available aspects,
``{notes, measures, rests, notes_and_rests, events, labels, chords, cadences, expanded}``,
the method provides two parameters, namely ``_folder`` (where to store TSVs) and ``_suffix``,
i.e. a slug appended to the respective filenames. If the parameter
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
which is the one passed to :py:class:`~ms3.parse.Parse`:

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
out will create the same structure in the directory from which the :py:class:`~ms3.parse.Parse`
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
from which the :py:class:`~ms3.parse.Parse` object has been created. In a similar manner,
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
Always starts with 1 for correspondence to MuseScore's status bar. For more detailed information, please refer to
:ref:`mc_vs_mn`.

.. _mn:

**mn** Measure Numbers
^^^^^^^^^^^^^^^^^^^^^^

Measure number, continuous count of complete measures as used in printed editions.
Starts with 1 except for pieces beginning with a pickup measure, numbered as 0. MNs are identical for first and
second endings! For more detailed information, please refer to :ref:`mc_vs_mn`.

.. _mc_onset:

**mc_onset**
^^^^^^^^^^^^
The value for ``mc_onset`` represents, expressed as fraction of a whole note, a position in a measure where ``0``
corresponds to the earliest possible position (in most cases beat 1). For more detailed information, please
refer to :ref:`onsets`.

.. tip::

    When loading a table from a TSV file, it is recommended to parse the text of this
    column with :obj:`fractions.Fraction` to be able to calculate with the values.
    MS3 does this automatically.

.. _mn_onset:

**mn_onset**
^^^^^^^^^^^^
The value for ``mn_onset`` represents, expressed as fraction of a whole note, a position in a measure where ``0``
corresponds to the earliest possible position of the corresponding measure number (MN). For more detailed information,
please refer to :ref:`onsets`.

.. _quarterbeats:

quarterbeats
^^^^^^^^^^^^

This column expresses positions, otherwise accessible only as a tuple ``(mc, mc_onset)``, as a running count of
quarter notes from the piece's beginning (quarterbeat = 0). If second endings are present in the score, only the
last ending is counted in order to give authentic values to such a score, as if played without repetitions. If
repetitions are unfolded, i.e. the table corresponds to a full play-through of the score, all endings are taken into
account correctly.

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
:ref:`mc_onset` ``0`` (beginning of the MC)
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
    from a :py:class:`~ms3.score.Score` object and MS3 needs to handle both formats.

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
