======
Manual
======

This page is a detailed guide for using ms3 for different tasks. It supposes you are working in an interactive Python
interpreter such as IPython, Jupyter, Google Colab, or just the console.

Parsing a single score
======================

.. rst-class:: bignums

1. Locate the `MuseScore 3 <https://musescore.org/en/download>`__ score you want to parse.

    Make sure it is uncompressed. i.e. it has the extension ``.mscx`` and not ``.mscz``.

    .. tip::

        MSCZ files are ZIP files containing the uncompressed MSCX. A later version of ms3 will be able to deal with MSCZ, too.


    In the examples, we parse the annotated first page of Giovanni
    Battista Pergolesi's influential *Stabat Mater*. The file is called ``stabat.mscx`` and can be downloaded from
    `here <https://raw.githubusercontent.com/johentsch/ms3/master/docs/stabat.mscx>`__ (open link and key ``Ctrl + S`` to save the file
    or right-click on the link to ``Save link as...``).

2. Import the library.

    To parse a single score, we will use the class ``ms3.Score``. We could import the whole library:

    .. code-block:: python

        >>> import ms3
        >>> s = ms3.Score()

    or simply import the class:

    .. code-block:: python

        >>> from ms3 import Score
        >>> s = Score()

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
---------------

.. automethod:: ms3.score.Score.__init__
    :noindex:

Bla bla
-------
