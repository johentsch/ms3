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

The annotations contained in a score are stored in a :py:class:`~ms3.annotations.Annotations` object and can be accessed
and stored as a tab-separated file (TSV) like this:

.. code-block:: python

    >>> s.annotations
    48 labels:
    staff  voice  label_type
    3      2      0             48

    >>> s.annotations.output_tsv('~/stabat_chords.tsv')
    True


Removing annotation labels from score
-------------------------------------

The annotations will be stored with a keyword that you choose. It needs to be different from ``'annotations'``.

.. code-block:: python

    >>> s.detach_labels('chords')
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







