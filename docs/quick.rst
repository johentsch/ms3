===============
Quick Reference
===============

This page is a quick guide for using ms3 for different tasks. It supposes you are working in an interactive Python
interpreter such as IPython, Jupyter, Google Colab, or just the console.

Parsing a single score
======================

The example score is called ``stabat.mscx`` and can be downloaded from
`here <https://raw.githubusercontent.com/johentsch/ms3/master/docs/stabat.mscx>`__.

.. code-block:: python

    >>> from ms3 import Score
    >>> s = Score('~/ms3/docs/stabat.mscx')
    >>> s
        MuseScore file
        --------------

        ~/ms3/docs/stabat.mscx

        Attached annotations
        --------------------

        48 labels:
        staff  voice  label_type
        3      2      dcml          48


Storing the labels
------------------

The annotations contained in a score are stored in an :py:class:`~ms3.annotations.Annotations` object and can be accessed
and stored as a tab-separated file (TSV) like this:

.. code-block:: python

    >>> s.annotations
    48 labels:
    staff  voice  label_type
    3      2      0             48

    >>> s.annotations.output_tsv('~/stabat_chords.tsv')
    True

.. _detaching:

Removing annotation labels
--------------------------

The annotations will be stored with a keyword that you choose. It needs to be different from ``'annotations'``.

.. code-block:: python

    >>> s.detach_labels(key='chords')
    >>> s
    MuseScore file (CHANGED!!!)
    ---------------!!!!!!!!!!!!

    ~/ms3/docs/stabat.mscx

    No annotations attached.

    Detached annotations
    --------------------

    chords -> 48 labels:
    staff  voice  label_type
    3      2      dcml          48

Upon inspecting the object we see that the 48 labels are not attached to the score anymore. They are stored in a new
:py:class:`~ms3.annotations.Annotations` object which can be accessed via ``s.chords``, i.e. the key we've chosen.
The warning ``CHANGED!!!`` does not mean that the file on disc has been changed, only the inner representation. Overwriting
the original file could mean a loss of the labels unless they are stored separately.


Storing the changed score
-------------------------

To output the changed score without the labels, choose a different path unless you really want to overwrite the annotated file.

.. code-block:: python

    >>> s.output_mscx('~/stabat_empty.mscx')
    True


Adding labels to score
----------------------

The method :py:meth:`~ms3.score.Score.attach_labels` can be used to re-attach a set of labels that has been
:ref:`detached<detaching>`. Similarly we can load the empty score and the stored labels to reunite them:

.. code-block:: python

    >>> e = Score('~/stabat_empty.mscx')
    >>> e.load_annotations('~/stabat_chords.tsv', key='tsv_chords')
    >>> e
    MuseScore file
    --------------

    ~/stabat_empty.mscx

    No annotations attached.

    Detached annotations
    --------------------

    tsv_chords (stored as stabat_chords.tsv) -> 48 labels:
    staff  voice  label_type
    3      2      0             48

    >>> e.attach_labels(key='tsv_chords', voice=1)
    >>> e
    MuseScore file (CHANGED!!!)
    ---------------!!!!!!!!!!!!

    ~/stabat_empty.mscx

    Attached annotations
    --------------------

    48 labels:
    staff  voice  label_type
    3      1      0             48

    Detached annotations
    --------------------

    tsv_chords (stored as stabat_chords.tsv) -> 48 labels:
    staff  voice  label_type
    3      2      0             48

As we can see, the parameter ``voice=1`` has been used to insert the labels in the first layer (coloured blue in MuseScore)
of staff 3 when originally they had been attached to layer two (coloured in green in the software).


Accessing score information
---------------------------

After parsing a score, all contained information is accessible in structured formats. Most information is returned as
:obj:`pandas.DataFrame`, whereas a given set of metadata is accessible as dictionary.

Since this information is attached to the parsed MSCX file (and not, say, to loaded annotations), it is accessible
via ``s.mscx``.

Metadata
~~~~~~~~

The metadata contains the data that can be accessed and altered in MuseScore 3 through the menu ``File -> Score Properties``
as well as information computed from the score, such as the names and ambitus of the contained staves. Note that the
ambitus here pertain to the first page only.

.. code-block:: python

    >>> s.mscx.get_metadata()
    {'arranger': None,
     'composer': 'Giovanni Battista Pergolesi',
     'copyright': 'Editions FREDIPI',
     'creationDate': '2019-07-23',
     'lyricist': None,
     'movementNumber': '1',
     'movementTitle': 'Stabat Mater dolorosa',
     'platform': 'Microsoft Windows',
     'poet': None,
     'source': 'http://musescore.com/user/1630246/scores/5653570',
     'translator': 'fredipi',
     'workNumber': None,
     'workTitle': 'Stabat Mater',   #  <- Score Properties until here
     'last_mc': 13,                 #  <- computed information from here
     'last_mn': 13,
     'label_count': 48,
     'TimeSig': {1: '4/4'},
     'KeySig': {1: -4},
     'annotated_key': 'f',
     'parts':  {'Soprano': {1:  {'min_midi': 65,
                                'min_name': 'F4',
                                'max_midi': 70,
                                'max_name': 'Bb4'}
                              },
                  'Alto':  {2:  {'min_midi': 64,
                                'min_name': 'E4',
                                'max_midi': 68,
                                'max_name': 'Ab4'}
                              },
                  'Piano': {3: {'min_midi': 56,
                                'min_name': 'Ab3',
                                'max_midi': 85,
                                'max_name': 'Db6'},
                            4: {'min_midi': 44,
                                'min_name': 'Ab2',
                                'max_midi': 70,
                                'max_name': 'Bb4'}
                              }
                  },
     'musescore': '3.5.0'}


The computed information contains the following:

* ``last_mc/last_mn``: Last measure number and measure count (see :ref:`here<mc_vs_mn>` to learn the difference).
* ``TimeSig/KeySig``: Time signatures and key signatures, each given as a dictionary with measure counts as keys.
* ``annotated_key``: Only included if the first annotation label in the score starts with a key such as ``Ab`` or ``f#``.
* ``parts``: contain several inner dictionaries: parts -> partname -> staves -> ambitus. For example, the dictionary
    for the piano part contains staves 3 and for, one for the right hand (Ab3-Db6) and one for the left hand (Ab2-Bb4).
* ``musescore``: The MuseScore version with which the files has been saved.

.. _tabular_info:

Tabular information
~~~~~~~~~~~~~~~~~~~

The accessible DataFrames with score information are:

* ``measures``: A list of all measures together with the strictly increasing **measure counts (MC)** mapped to the actual
  **measure numbers (MN)**. Read more on the difference in the :ref:`manual<mc_vs_mn>`.
* ``notes``: A list of all notes contained in the score together with their respective features.
* ``chords``: Not to confound with labels or chord annotations, a chord is a notational unit in which all included
  notes are part of the same notational layer and have the same onset. Every chord has a ``chord_id`` and every note
  is part of a chord. These tables are used to convey score information that is not attached to a particular note,
  such as lyrics, staff text, dynamics and other markup.
* ``rests``: A list of rests.
* ``events``: For sake of completeness, a raw version of the score information for debugging purposes.

.. code-block:: python

    >>> s.mscx.measures

.. |act_dur| replace:: :ref:`act_dur <act_dur>`
.. |barline| replace:: :ref:`barline <barline>`
.. |breaks| replace:: :ref:`breaks <breaks>`
.. |dont_count| replace:: :ref:`dont_count <dont_count>`
.. |keysig| replace:: :ref:`keysig <keysig>`
.. |mc| replace:: :ref:`mc <mc>`
.. |mc_offset| replace:: :ref:`mc_offset <mc_offset>`
.. |mn| replace:: :ref:`mn <mn>`
.. |next| replace:: :ref:`next <next>`
.. |numbering_offset| replace:: :ref:`numbering_offset <numbering_offset>`
.. |timesig| replace:: :ref:`timesig <timesig>`
.. |repeats| replace:: :ref:`repeats <repeats>`
.. |volta| replace:: :ref:`volta <volta>`


+------+------+----------+-----------+-----------+-------------+----------+--------------+---------+-----------+--------------------+--------------+--------+
| |mc| | |mn| | |keysig| | |timesig| | |act_dur| | |mc_offset| | |breaks| | |repeats|    | |volta| | |barline| | |numbering_offset| | |dont_count| | |next| |
+------+------+----------+-----------+-----------+-------------+----------+--------------+---------+-----------+--------------------+--------------+--------+
| 1    | 1    | -4       | 4/4       | 1         | 0           | NaN      | firstMeasure | <NA>    | NaN       | <NA>               | <NA>         | (2,)   |
+------+------+----------+-----------+-----------+-------------+----------+--------------+---------+-----------+--------------------+--------------+--------+
| 2    | 2    | -4       | 4/4       | 1         | 0           | NaN      | NaN          | <NA>    | NaN       | <NA>               | <NA>         | (3,)   |
+------+------+----------+-----------+-----------+-------------+----------+--------------+---------+-----------+--------------------+--------------+--------+

.. code-block:: python

    >>> s.mscx.notes

+----+----+---------+-------+-------+-------+----------+-----------+------------------+--------+------+-----+------+-------+----------+
| mc | mn | timesig | onset | staff | voice | duration | gracenote | nominal_duration | scalar | tied | tpc | midi | volta | chord_id |
+====+====+=========+=======+=======+=======+==========+===========+==================+========+======+=====+======+=======+==========+
| 1  | 1  | 4/4     | 0     | 4     | 2     | 1/8      | NaN       | 1/8              | 1      | <NA> | -1  | 53   | <NA>  | 4        |
+----+----+---------+-------+-------+-------+----------+-----------+------------------+--------+------+-----+------+-------+----------+
| 1  | 1  | 4/4     | 0     | 3     | 2     | 3/4      | NaN       | 1/2              | 3/2    | <NA> | -1  | 77   | <NA>  | 1        |
+----+----+---------+-------+-------+-------+----------+-----------+------------------+--------+------+-----+------+-------+----------+

.. code-block:: python

    >>> s.mscx.chords

+----+----+---------+-------+-------+-------+----------+-----------+------------------+--------+-------+----------+------------+--------+--------------+----------+------+-------------+------------+
| mc | mn | timesig | onset | staff | voice | duration | gracenote | nominal_duration | scalar | volta | chord_id | staff_text | lyrics | articulation | dynamics | Slur | decrescendo | diminuendo |
+====+====+=========+=======+=======+=======+==========+===========+==================+========+=======+==========+============+========+==============+==========+======+=============+============+
| 1  | 1  | 4/4     | 1/2   | 3     | 1     | 1/2      | NaN       | 1/2              | 1      | <NA>  | 0        | NaN        | NaN    | NaN          | NaN      | NaN  | NaN         | NaN        |
+----+----+---------+-------+-------+-------+----------+-----------+------------------+--------+-------+----------+------------+--------+--------------+----------+------+-------------+------------+
| 1  | 1  | 4/4     | 0     | 3     | 2     | 3/4      | NaN       | 1/2              | 3/2    | <NA>  | 1        | NaN        | NaN    | NaN          | NaN      | 0    | NaN         | NaN        |
+----+----+---------+-------+-------+-------+----------+-----------+------------------+--------+-------+----------+------------+--------+--------------+----------+------+-------------+------------+

.. code-block:: python

    >>> s.mscx.rests

+----+----+---------+-------+-------+-------+----------+------------------+--------+-------+
| mc | mn | timesig | onset | staff | voice | duration | nominal_duration | scalar | volta |
+====+====+=========+=======+=======+=======+==========+==================+========+=======+
| 1  | 1  | 4/4     | 0     | 1     | 1     | 1        | 1                | 1      | <NA>  |
+----+----+---------+-------+-------+-------+----------+------------------+--------+-------+
| 1  | 1  | 4/4     | 0     | 2     | 1     | 1        | 1                | 1      | <NA>  |
+----+----+---------+-------+-------+-------+----------+------------------+--------+-------+


Parsing multiple scores
=======================

Often we want to perform operations on many scores at once, for example extracting the notelist of each and store it as
a tab-separated values file (TSV).

Loading
-------

The first step is to create a :py:class:`~ms3.parse.Parse` object. When passing it
the path of the cloned `Git <https://github.com/johentsch/ms3>`__, it scans it for all MSCX files:

.. code-block:: python

    >>> from ms3 import Parse
    >>> p = Parse('~/ms3')
    >>> p
    10 files.
    KEY       -> EXTENSIONS
    docs      -> {'.mscx': 4}
    tests/MS3 -> {'.mscx': 6}

As we see, different keys have been automatically assigned for the different folders because no key has been specified.
Instead, we could assign all ten files to the same key and then add the 'docs' once more with a different key:

.. code-block:: python

    >>> p = Parse('~/ms3', key='all')
    >>> p.add_dir('~/ms3/docs', key='doubly')
    >>> p
    14 files.
    KEY    -> EXTENSIONS
    all    -> {'.mscx': 10}
    doubly -> {'.mscx': 4}


Parsing
-------

... is as simple as

.. code-block:: python

    >>> p.parse_mscx()
    WARNING Did03M-Son_regina-1762-Sarti -- bs4_measures.py (line 152) check_measure_numbers():
	    MC 94, the 1st measure of a 2nd volta, should have MN 93, not MN 94.

Voilà, parsed in parallel with only one warning where a score has to be corrected. The parsed
:py:class:`~ms3.score.Score` objects (:ref:`read_only` mode) are stored in the dictionary
:py:attr:`~ms3.parse.Parse._parsed`, the state of which can be viewed like this:

.. code-block:: python

    >>> p.parsed
    {('all', 0): '~/ms3/docs/cujus.mscx -> 88 labels',
     ('all', 1): '~/ms3/docs/o_quam.mscx -> 26 labels',
     ('all', 2): '~/ms3/docs/quae.mscx -> 79 labels',
     ('all', 3): '~/ms3/docs/stabat.mscx -> 48 labels',
     ('all', 4): '~/ms3/tests/MS3/05_symph_fant.mscx',
     ('all', 5): '~/ms3/tests/MS3/76CASM34A33UM.mscx -> 173 labels',
     ('all', 6): '~/ms3/tests/MS3/BWV_0815.mscx',
     ('all', 7): '~/ms3/tests/MS3/D973deutscher01.mscx',
     ('all', 8): '~/ms3/tests/MS3/Did03M-Son_regina-1762-Sarti.mscx -> 193 labels',
     ('all', 9): '~/ms3/tests/MS3/K281-3.mscx -> 375 labels',
     ('doubly', 0): '~/ms3/docs/cujus.mscx -> 88 labels',
     ('doubly', 1): '~/ms3/docs/o_quam.mscx -> 26 labels',
     ('doubly', 2): '~/ms3/docs/quae.mscx -> 79 labels',
     ('doubly', 3): '~/ms3/docs/stabat.mscx -> 48 labels'}


Extracting score information
----------------------------

Each of the :ref:`previously discussed DataFrames<tabular_info>` can be automatically stored for every score. To select
one or several aspects from ``[notes, measures, rests, notes_and_rests, events, labels, chords, expanded]``, it is enough
to pass the respective ``_folder`` parameter to :py:meth:`~ms3.parse.Parsed.store_lists` distinguishing where to store
the TSV files. Additionally, the method accepts one ``_suffix`` parameter per aspect, i.e. a slug added to the respective
filenames. If the parameter ``simulate=True`` is passed, no files are written but the file paths to be created are returned.

In this variant, all aspects are stored each in individual folders but with identical filenames:

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


In this variant, the different ways of specifying folder are exemplified. To demonstrate all subtleties we parse the
same four files but this time from the perspective of ``~/ms3``:

.. code-block:: python

    >>> p = Parse('~/ms3', folder_re='docs', key='pergo')
    >>> p.parse_mscx()
    >>> p.store_lists(  notes_folder='./notes',
                        measures_folder='../measures',
                        rests_folder='rests',
                        labels_folder='~/labels',
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

The rules for specifying the folders are as follows:

* absolute folder (e.g. ``~/labels``): Store all files in this particular folder without creating subfolders.
* relative folder starting with ``./`` or ``../`` means that the file is to be placed relative to the location of the
  original MSCX file
* relative folder not starting with ``./`` or ``../`` (e.g. ``rests``) creates the folder under the scan folder and
  places the files into a (newly created) relative folder structure below.
