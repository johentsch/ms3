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
