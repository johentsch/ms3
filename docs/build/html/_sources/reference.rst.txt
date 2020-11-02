=====================
Developers' Reference
=====================

When using ms3, we are dealing with four main object types:

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

The score module
================

The Score class
---------------

.. autoclass:: ms3.score.Score
    :members:

The MSCX class
--------------

.. autoclass:: ms3.score.MSCX
    :members:

The annotations module
======================

.. automodule:: ms3.annotations
    :members:

The parse module
================

.. automodule:: ms3.parse
    :members:

The expand_dcml module
======================

.. automodule:: ms3.expand_dcml
    :members:

Developing a new parser
=======================

Every new parser needs to fulfil the following interface requirements.

Methods
-------

.. code-block:: python

    def add_label(self, label, mc, mc_onset, staff=1, voice=1, **kwargs):
        """ Adds a single label to the current XML in form of a new
        <Harmony> (and maybe also <location>) tag.
        """

    def delete_label(self, mc, staff, voice, mc_onset):
        """ Delete a label from a particular position (if there is one).

        Parameters
        ----------
        mc : :obj:`int`
            Measure count.
        staff, voice
            Notational layer in which to delete the label.
        mc_onset : :obj:`fractions.Fraction`
            mc_onset

        Returns
        -------
        :obj:`bool`
            Whether a label was deleted or not.
        """

    def get_chords(self, staff=None, voice=None, mode='auto', lyrics=False, staff_text=False, dynamics=False, articulation=False, spanners=False, **kwargs):
        """ Retrieve a customized chord list, e.g. one including less of the processed features or additional,
        unprocessed ones compared to the standard chord list.

        Parameters
        ----------
        staff : :obj:`int`
            Get information from a particular staff only (1 = upper staff)
        voice : :obj:`int`
            Get information from a particular voice only (1 = only the first layer of every staff)
        mode : {'auto', 'all', 'strict'}, optional
            Defaults to 'auto', meaning that those aspects are automatically included that occur in the score; the resulting
                DataFrame has no empty columns except for those parameters that are set to True.
            'all': Columns for all aspects are created, even if they don't occur in the score (e.g. lyrics).
            'strict': Create columns for exactly those parameters that are set to True, regardless which aspects occur in the score.
        lyrics : :obj:`bool`, optional
            Include lyrics.
        staff_text : :obj:`bool`, optional
            Include staff text such as tempo markings.
        dynamics : :obj:`bool`, optional
            Include dynamic markings such as f or p.
        articulation : :obj:`bool`, optional
            Include articulation such as arpeggios.
        spanners : :obj:`bool`, optional
            Include spanners such as slurs, 8va lines, pedal lines etc.
        **kwargs : :obj:`bool`, optional
            Set a particular keyword to True in order to include all columns from the _events DataFrame
            whose names include that keyword. Column names include the tag names from the MSCX source code.

        Returns
        -------
        :obj:`pandas.DataFrame`
            DataFrame representing all <Chord> tags in the score with the selected features.
        """

    def infer_mc(self, mn, mn_onset=0, volta=None):
        """ Shortcut for ``MSCX.parsed.infer_mc()``.
        Tries to convert a ``(mn, mn_onset)`` into a ``(mc, mc_onset)`` tuple on the basis of this MuseScore file.
        In other words, a human readable score position such as "measure number 32b (i.e., a second ending), beat
        3" needs to be converted to ``(32, 1/2, 2)`` if "beat" has length 1/4, or--if the meter is, say 9/8 and "beat"
        has a length of 3/8-- to ``(32, 6/8, 2)``. The resulting ``(mc, mc_onset)`` tuples are required for attaching
        a label to a score. This is only necessary for labels that were not originally extracted by ms3.

        Parameters
        ----------
        mn : :obj:`int` or :obj:`str`
            Measure number as in a reference print edition.
        mn_onset : :obj:`fractions.Fraction`, optional
            Distance of the requested position from beat 1 of the complete measure (MN), expressed as
            fraction of a whole note. Defaults to 0, i.e. the position of beat 1.
        volta : :obj:`int`, optional
            In the case of first and second endings, which bear the same measure number, a MN might have to be
            disambiguated by passing 1 for first ending, 2 for second, and so on. Alternatively, the MN
            can be disambiguated traditionally by passing it as string with a letter attached. In other words,
            ``infer_mc(mn=32, volta=1)`` is equivalent to ``infer_mc(mn='32a')``.

        Returns
        -------
        :obj:`int`
            Measure count (MC), denoting particular <Measure> tags in the score.
        :obj:`fractions.Fraction`

        """

    def parse_measures()
