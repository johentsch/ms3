---
title: 'ms3: A parser for MuseScore files, serving as data factory for annotated music corpora.'
tags:

- python
- music
- scores
- corpus
- corpora
- data
- musescore
- tab-separated values
  authors:
- name: Johannes Hentschel
  orcid: 0000-0002-1986-9545
  corresponding: true
  affiliation: 1
- name: Martin Rohrmeier
  orcid: 0000-0002-4323-7257
  affiliation: 1
  affiliations:
- name: École Polytechnique Fédérale de Lausanne, Switzerland
  index: 1
  date: 13 October 2022
  bibliography: paper.bib

---

# Summary

Digital Musicology is a vibrant and quickly growing discipline that addresses traditional and novel music-related
research questions with digital and computational
means [@Huron1999_NewEmpiricismSystematic; @Honing2006_GrowingRoleObservation; @Urberg2017_PastsFuturesDigital].
Research questions and methods often overlap with or draw on those from diverse disciplines such as music theory and
analysis, composition history, mathematics, cognitive psychology, linguistics, anthropology, or computer
science [@Volk2012_MathematicalComputationalApproaches; @Wiggins2012_MusicMindMathematics].
Corpus research, i.e., the computational study of representative collections of texts (in the case of linguistics) or
notated music (in musicology), plays a prominent role in this trans-disciplinary quest to "make sense of music" through
scientific models [@London2013_BuildingRepresentativeCorpus].
`ms3` makes scores (symbolic representations of music) operational for computational approaches by representing their
contents as sets of tabular files.

# Statement of need

<!---
## The importance of scores for musicology

At the heart of corpus-driven methods into tonal music (roughly speaking, Western music since the late 16th century)
lies the score, a symbolic representation of the relations between sound events over
time [@Moss2019_TransitionsTonalityModelBased].
Building on notational conventions that have emerged over the course of circa 1,000 years, they constitute a well-proven
cultural practice of mapping events from the continuous space of physical time and frequency to an idealized, virtual,
discrete space of musical time (perceived temporal bins) and pitch (perceived frequency bins), that bears witness of
categorization as a prevalent principle of human cognition.
Much like texts, scores represent both a means of communicating music to readers and performers (who transduce them into
cognitive processes and sound), and a conventionalized way to reduce, bin, quantize, and protract the complex
information inherent to physical, analog, or digital audio signals and musical thought (inner hearing).
Apart from often being the only surviving trace of music from earlier centuries, scores encode additional semantic
information on sounding events that is unavailable in their physical signal, e.g., the intended or ascribed relation of
a pitch to its surrounding pitches (paradigmatic relations), and to musical time (syntagmatic relations).
Consequently, a score can also be viewed as an abstraction over all possible ways a human can interpret and actuate it,
and hence is an indispensable tool for encoding, communicating, and comparing its various interpretations and recordings.
Understanding a score as an abstract relational model of the composition or musical utterance it represents sheds light
on its essential role for interrelating heterogeneous data from and for performance research, music analysis,
neuroscience, stylometry, music psychology, ethnography, or music information
retrieval [@Cook2005_CompleatMusicologist; @Abdallah2017_DigitalMusicLab].
Whereas text corpora for the long-established discipline of computational linguistics are abundant, the advent of
similarly large curated datasets of symbolically encoded (as opposed to scanned) digital scores yet
awaits [@Shanahan2022_WhatHistoryComputational].

## `ms3` makes scores operational for music research
-->

Music scores represent relations between sounding events by graphical means. Music notation software therefore is very
much concerned with the aesthetically pleasing rendering of symbols in line with the commonplace notational conventions
[@Read1979_MusicNotationManual]; and much less so with the explicit encoding of the musical relations themselves.
For example, the Free and Open Source Software [MuseScore](https://musescore.org/) provides a full-featured yet
intuitive interface for engraving music, but it does not explicitly encode the temporal positions of sounding events in
its native XML format. Hence the need for a parser that extracts this implicit information and stores it in an
interoperable format.

The Python library `ms3` loads the XML tree of a MuseScore file into working memory, computes the temporal positions of
all encoded elements, and transforms those requested by the user
into [DataFrames](https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#dataframe), i.e., feature
matrices. The DataFrames can be used by other Python programs and scripts, or written to Tab-Separated Values (TSV) for
version control and use by other software. The most typical aspects that users extract from a score are notes, measures
(bars), metadata, and chord labels including those resulting from analysis. Moreover, `ms3` allows the user to remove 
and insert chord labels from and into scores and to write back the altered scores. This functionality turns MuseScore
into a convenient score annotation tool by enabling users to graphically insert arbitrary textual labels into a score
and to then have `ms3` extract them together with their temporal positions for further analysis. 

`ms3` has been used for creating several published datasets, namely 


# Acknowledgements

Development of this software tool was supported by the Swiss National Science Foundation within the project “Distant
Listening – The Development of Harmony over Three Centuries (1700–2000)” (Grant no. 182811). This project is being
conducted at the Latour Chair in Digital and Cognitive Musicology, generously funded by Mr. Claude Latour.

# References