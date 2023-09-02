|license| |version| |release|  |size|

.. |license| image:: https://img.shields.io/github/license/johentsch/ms3?color=%230000ff
    :alt: GitHub

.. |release| image:: https://img.shields.io/github/release-date/johentsch/ms3
    :alt: GitHub Release Date

.. |size| image:: https://img.shields.io/github/repo-size/johentsch/ms3
    :alt: GitHub repo size

.. .. |tests| image:: https://img.shields.io/github/workflow/status/johentsch/ms3/run_tests/main?label=tests
    :alt: GitHub Workflow Status (branch)

.. |version| image:: https://img.shields.io/pypi/v/ms3?color=%2300
    :alt: PyPI

.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://readthedocs.org/projects/ms3/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://ms3.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/ms3/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/ms3
    .. image:: https://img.shields.io/pypi/v/ms3.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/ms3/
    .. image:: https://pepy.tech/badge/ms3/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/ms3


===============================
ms3 - Parsing MuseScore 3 and 4
===============================

..
    Plan to use
    .. include:: ./docs/intro.rst
    failed because of PyPi


Welcome to **ms3**, a Python library for parsing `MuseScore <https://musescore.org/en/download>`__ files.

Statement of need
=================

Here comes a list of functionalities to help you decide if this library could be useful for you.

* parses MuseScore 3 and 4 files, dispensing with lossy conversion to musicXML. The file formats in question are

  * uncompressed ``*.mscx`` files,
  * compressed ``*.mscz`` files,

* extracts and processes the information contained in one or many scores in the form of
  `DataFrames <https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#dataframe>`__:

  * **notes** (start, duration, pitch etc.) and/or rests,
  * **measures** (time signature, lengths, repeat structure etc.)
  * **labels**, such as

    * guitar/Jazz chord labels
    * arbitrary annotation labels
    * **expanded** harmony labels following the `DCML annotation standard <https://github.com/DCMLab/standards>`__
    * **cadences** (part of the same annotation syntax)
    * **form_labels** (annotation standard currently in press)

  * **chords**, that is, onset positions that have musical markup attached, e.g. dynamics, lyrics, slurs, 8va signs...
  * **metadata** from the respective fields, but also score statistics, such as length, number of notes, etc.

* stores the extracted information in a uniform and interoperable tabular format (``*.tsv``)
* writes information from tabular ``*.tsv`` files into MuseScore files, especially

  * chord and annotation labels
  * metadata
  * header information (title, subtitle, etc.)
  * note coloring

* uses a locally installed or standalone MuseScore executable for

  * batch-converting files to any output format supported by MuseScore (mscz, mscx, mp3, midi, pdf etc.)
  * on-the-fly converting any file that MuseScore can read (including MuseScore 2, cap, capx, midi, and musicxml) to parse it

* offers its functionality via the convenient ``ms3`` commandline interface.

View the `full documentation here <https://ms3.readthedocs.io/>`__.

For a demo video (using an old, pre-1.0.0 version) on YouTube, `click here <https://youtu.be/UBY3wuIS4wc>`__

Installation
============

ms3 requires Python >= 3.10 (type ``python3 --version`` to check). Once you have switched to a virtual environment
that has Python 3.10 installed you can pip-install the library via one of the two commands::

    python3 -m pip install ms3
    pip install ms3

If successful, the installation will make the ``ms3`` commands available in your PATH (try by typing ``ms3``).

Quick demo
==========

Parsing a single score
----------------------

.. code-block:: python

    import ms3
    score = ms3.Score('musescore_file.mscz')

Parsing a corpus
----------------

.. code-block:: python

    import ms3
    corpus = ms3.Corpus('score_directory')
    corpus.parse()

Parsing several corpora
-----------------------

.. code-block:: python

    import ms3
    corpora = ms3.Parse('my_research_corpora')
    corpora.parse()


.. _pyscaffold-notes:

Making Changes & Contributing
=============================

This project uses `pre-commit <https://pre-commit.com/>`__ to ensure code quality. If you are a developer,
please make sure to install it before making any changes::

    cd ms3
    pip install -e ".[dev]" # includes "pip install pre-commit"
    pre-commit install


Acknowledgements
================

Development of this software tool was supported by the Swiss National Science Foundation within the project “Distant
Listening – The Development of Harmony over Three Centuries (1700–2000)” (Grant no. 182811). This project is being
conducted at the Latour Chair in Digital and Cognitive Musicology, generously funded by Mr. Claude Latour.

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/
