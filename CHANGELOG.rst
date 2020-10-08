=========
Changelog
=========

Version 0.2.1
=============

* Parse.metadata() to return a DataFrame with all parsed pieces' metadata
* Parse.get_labels() to retrieve labels of a particular kind
* Parse.info() has improved the information that objects return about themselves
* Score.store_list()
* improved expansion of DCML harmony labels

Version 0.2.0
=============

Beta stage:

* attaching and detaching labels
* parsing multiple pieces at once
* extraction of metadata from scores
* inclusion of staff text, dynamics and articulation in chord lists, added 'auto' mode
* conversion of MuseScore's encoding of absolute chords
* first version of docs

Version 0.1.3
=============

At this stage, the library can parse MuseScore 3 files to different types of lists:

* measures
* chords (= groups of notes)
  * including slurs and spanners such as pedal, 8va or hairpin markings
  * including lyrics
* notes
* harmonies

and also some basic metadata.

Version 0.1.0
=============

- Basic parser implemented
- Logging
- Measure lists
