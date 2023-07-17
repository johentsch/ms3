---
title: 'ms3: A parser for MuseScore files, serving as data factory for annotated music corpora'
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
date: 17 January 2023
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
scientific models [@London2013_BuildingRepresentativeCorpus; @Moss2019_TransitionsTonalityModelBased; @Shanahan2022_WhatHistoryComputational].
`ms3` makes scores (symbolic representations of music) operational for computational approaches by representing their
contents as sets of tabular files.

# Statement of need

Music scores represent relations between sounding events by graphical means. Music notation software therefore is
often concerned with problems of layout and easy-to-read rendering of symbols in line with the multitude of
notational conventions [@Read1979_MusicNotationManual; @Ross2001_ArtMusicEngraving]; rather than with the explicit
encoding of the musical relations themselves.
For example, the Free and Open Source Software [MuseScore](https://musescore.org/) provides a full-featured yet
intuitive interface for engraving music, but its native XML format does not explicitly encode the temporal positions of
events such as notes and rests. Hence the need for a parser that extracts the implicit information and stores it in an
interoperable format. 

Despite being one of the most widespread score encoding formats, current score parsers 
[e.g., @Cancino-Chacon2022_PartituraPythonPackage; @Cuthbert2010_Music21ToolkitComputerAided; @Pugin2014_VerovioLibraryEngraving], 
do not handle it without first performing a lossy conversion to the musicXML format[^1].
The Python library `ms3` fills this gap. It loads the XML tree of a MuseScore file into working memory, 
computes the temporal positions of all encoded elements, and transforms those requested by the user into DataFrames [@Petersohn2021_DataframeSystemsTheory]. 
The DataFrames can be used by other Python programs and scripts, or written to Tab-Separated Values (TSV) to enable processing with other software
and facilitate version control[^2]. The most typical aspects that users extract from a score are
tables containing notes, measures (bars), metadata, and text labels, in particular those representing analytical annotations.
Moreover, `ms3` allows the user to transform scores by removing analytical labels after their extraction or by (re-)inserting annotations from 
TSV files (whether previously extracted or generated from scratch). 
This functionality turns MuseScore into a convenient score annotation tool enabling users to graphically insert
into a score arbitrary textual labels, to then have `ms3` extract them with their temporal positions for further
analysis. It comes with a command line interface that makes its data extraction, transformation, and validation
functionalities accessible for productive everyday workflows.

[^1]: For example, musicXML's implicit encoding of temporal positions is limited to those where a note or rest event occurs. When converting MuseScore XML to musicXML, all score elements occurring between two such events are misplaced.    
[^2]: Version control is facilitated by the TSV files because, unlike the original XML source, they present score information with timestamps.

`ms3` has already been used for creating several datasets, namely version 2 of the Annotated Beethoven Corpus
[@Neuwirth2018_AnnotatedBeethovenCorpus], the Annotated Mozart Sonatas [@Hentschel2021_AnnotatedMozartSonatas],
and an annotated corpus of 19th century piano music [@Hentschelinpress_AnnotatedCorpusTonal]. It has been successful
in formatting training and validation data for a chord inference algorithm and for inserting its analytical outputs
into the respective scores [@Mcleod2021_ModularSystemHarmonic].
Moreover, the library is at the heart of a semi-automated annotation workflow running on GitHub
[@Hentschel2021_SemiautomatedWorkflowParadigm] and a dependency on the music corpus analysis library
DiMCAT [@Hentschelinpress_IntroducingDiMCATProcessing].

# Acknowledgements

Development of this software tool was supported by the Swiss National Science Foundation within the project “Distant
Listening – The Development of Harmony over Three Centuries (1700–2000)” (Grant no. 182811). This project is being
conducted at the Latour Chair in Digital and Cognitive Musicology, generously funded by Mr Claude Latour.

# References
