---
title: 'ms3: A parser for MuseScore 3 files and data factory for annotated music corpora.'
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

Digital Musicology is a vibrant and quickly growing discipline that addresses traditional and novel music-related research questions with digital and computational means [@Huron1999_NewEmpiricismSystematic; @Honing2006_GrowingRoleObservation; @Urberg2017_PastsFuturesDigital].
Research questions and methods often overlap with or draw on those from diverse disciplines such as music theory and analysis, composition history, mathematics, cognitive psychology, linguistics, anthropology, or computer science [@Volk2012_MathematicalComputationalApproaches; @Wiggins2012_MusicMindMathematics].
Corpus research, i.e., the computational study of representative collections of texts (in the case of linguistics) or compositions (in musicology), plays a prominent role in this trans-disciplinary quest to "make sense of music" through scientific models [@London2013_BuildingRepresentativeCorpus].


At the heart of corpus-driven methods into tonal music (roughly speaking, Western music since the late 16th century) lies the score, a symbolic representation of the relations between sound events over time.
Building on notational conventions that have emerged over the course of circa 1,000 years, they constitute a well-proven cultural practice of mapping events from the continuous space of physical time and frequency to an idealized, virtual, discrete space of musical time (perceived temporal bins) and pitch (perceived frequency bins), that bears witness of categorization as a prevalent principle of human cognition.
Much like texts, scores represent both a means of communicating music to readers and performers (who transduce them into cognitive processes and sound), and a conventionalized way to reduce, bin, quantize, and protract the complex information inherent to physical, analog, or digital audio signals and musical thought (inner hearing).
Apart from often being the only surviving trace of music from earlier centuries, scores encode additional semantic information on sounding events that is unavailable in their physical signal, e.g., the intended or ascribed relation of a pitch to its surrounding pitches (paradigmatic relations), and to musical time (syntagmatic relations).
Consequently, a score can also be viewed as an abstraction over all possible ways a human can interpret and actuate it, and hence is an indispensable tool for encoding, communicating, and comparing its various interpretations and recordings thereof, for instance by means of annotation.  
Understanding a score as an abstract relational model of the composition or musical utterance it represents sheds light on its essential role for interrelating heterogeneous data from and for performance research, music analysis, neuroscience, stylometry, music psychology, ethnography, or music information retrieval [@Cook2005_CompleatMusicologist; @Abdallah2017_DigitalMusicLab].
Whereas text corpora for the long-established discipline of computational linguistics are abundant, the advent of similarly large curated datasets of symbolically encoded (as opposed to scanned) digital scores yet awaits [@Shanahan2022_WhatHistoryComputational].

# Statement of need




# Acknowledgements

Development of this software tool was supported by the Swiss National Science Foundation within the project “Distant Listening – The Development of Harmony over Three Centuries (1700–2000)” (Grant no. 182811). This project is being conducted at the Latour Chair in Digital and Cognitive Musicology, generously funded by Mr. Claude Latour.

# References