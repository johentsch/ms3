=====================
Developers' Reference
=====================

The library installs a couple of commands to your system. The documentation for these commands can be found
:ref:`below <cli_doc>` or by executing ``ms3 -h``.

When using ms3 as a module, we are dealing with four main object types:

1. :py:class:`~ms3.score.MSCX` objects hold the information of a single
   parsed MuseScore file;
2. :py:class:`~ms3.annotations.Annotations` objects hold a set of annotation labels
   which can be either attached to a score (i.e., contained in its XML structure),
   or detached.
3. Both types of objects are contained within a :py:class:`~ms3.score.Score` object.
   For example, a set of :py:class:`~ms3.annotations.Annotations` read from a TSV
   file can be attached to the XML of an :py:class:`~ms3.score.MSCX` object, which
   can then be output as a MuseScore file.
4. To manipulate many :py:class:`~ms3.score.Score` objects at once, for example
   those of an entire corpus, we use :py:class:`~ms3.parse.Parse` objects.

Since :py:class:`~ms3.score.MSCX` and :py:class:`~ms3.annotations.Annotations`
objects are always attached to a :py:class:`~ms3.score.Score`, the documentation
starts with this central class.

The Parse class
================

.. automodule:: ms3.parse
    :members:
    :special-members:

The Corpus class
================

.. automodule:: ms3.corpus
    :members:

The Piece class
===============

.. automodule:: ms3.piece
    :members:

The View class
==============

.. automodule:: ms3.view
    :members:


The Score class
===============

.. autoclass:: ms3.score.Score
    :members:
    :private-members:

The MSCX class
--------------

This class defines the user interface for accessing score information via :py:attr:`Score.mscx <.score.Score.mscx>`.
It consists mainly of shortcuts for interacting with the parser in use, currently
:ref:`Beautifulsoup exclusively <bs4_parser>`.

.. autoclass:: ms3.score.MSCX
    :members:

The Annotations class
=====================

.. automodule:: ms3.annotations
    :members:

.. _bs4_parser:

The BeautifulSoup parser
========================

.. automodule:: ms3.bs4_parser
    :members:
    :private-members:

The expand_dcml module
======================

.. automodule:: ms3.expand_dcml
    :members:

Utils
=====

.. automodule:: ms3.utils
    :members:

Transformations
===============

.. automodule:: ms3.transformations
    :members:

.. _cli_doc:

The commandline interface
=========================

.. argparse::
    :module: ms3.cli
    :func: get_arg_parser
    :prog: ms3

Unittests
=========

``ms3`` has a test suite that uses the `PyTest <https://docs.pytest.org>`__ library.

Install dependencies
--------------------

Install the library via ``pip install ms3[testing]``.

Configuring the tests
---------------------

In order to run the tests you need to

* clone the `unittest_metacorpus <https://github.com/DCMLab/unittest_metacorpus>`__ including submodules
  (ask for permission)
* in the configuration file ``new_tests/conftest.py``, change the value of ``CORPUS_DIR`` to the path
  containing your clone of the metacorpus (defaults to the user's home directory)
* in the line below, copy the commit SHA of ``TEST_COMMIT``, e.g. ``51e4cb5``, and checkout your metacorpus to that
  commit (e.g., ``git checkout 51e4cb5``).

Running the tests
-----------------

In the commandline, head to your ``ms3`` folder and call ``pytest new_tests``. Alternatively, some IDEs allow
you to right-click on the folder ``new_tests`` and select something like ``Run pytest in new_tests``.