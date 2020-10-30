|tests| |license| |version| |release|  |size| |down|

.. |down| image:: https://img.shields.io/github/downloads/johentsch/ms3/total
    :alt: GitHub All Releases

.. |license| image:: https://img.shields.io/github/license/johentsch/ms3?color=%230000ff
    :alt: GitHub

.. |release| image:: https://img.shields.io/github/release-date/johentsch/ms3
    :alt: GitHub Release Date

.. |size| image:: https://img.shields.io/github/repo-size/johentsch/ms3
    :alt: GitHub repo size

.. |tests| image:: https://img.shields.io/github/workflow/status/johentsch/ms3/run_tests/master   
    :alt: GitHub Workflow Status (branch)

.. |version| image:: https://img.shields.io/pypi/v/ms3?color=%2300
    :alt: PyPI


=========================
ms3 - Parsing MuseScore 3
=========================

..
    Plan to use
    .. include:: ./docs/intro.rst
    failed


Welcome to **ms3**, a Python library for parsing annotated `MuseScore 3 <https://musescore.org/en/download>`__ files. It

* parses uncompressed MuseScore 3 files (``*.mscx``),
* stores the contained information in a tabular format (``*.tsv``),
* deletes and writes annotation labels to MuseScores <Harmony> layer,
* parses and transforms labels following the `DCML harmonic annotation standard <https://github.com/DCMLab/standards>`__

View the documentation on `GitHub <https://johentsch.github.io/ms3/>`__.


Note
====

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
