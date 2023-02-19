---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Manual

![](ms3_architecture.png)

This page is a detailed guide for using ms3 for different tasks. It
supposes you are working in an interactive Python interpreter such as
IPython, Jupyter, Google Colab, or just the console.

+++

## Good to know

(corpss)=
### Corpss structure 

ms3 extracts various aspects (e.g. notes, chord labels, dynamics) from a
single MuseScore file and stores them in separate TSV files. In order to
jointly evaluate this information (e.g. to combine harmony labels with
the corresponding notes in the score), ms3 associates files by their
names. Therefore it is important to not rename files at a later point
and it is recommended to stick to one organizational principle for all
corpora in use. The two main principles are suffixes and subfolders.

[DCML corpora](https://github.com/DCMLab/dcml_corpora) use the same
subfolder structure: MuseScore files are stored in the `MS3` folder of a
corpus and all other aspects are stored with identical filenames (but
different extension) in sibling folders. This structure results
naturally when using default parameters such as `ms3 extract -N -M -X`:

``` console
corpus1
├── harmonies
├── measures
├── MS3
└── notes
corpus2
├── harmonies
├── measures
├── MS3
└── notes
```

When loading corpora, ms3 looks for the standard folder names (and
suffixes) to indentify individual corpora and assign them keys
automatically (e.g. in the example above, \'corpus1\' and \'corpus2\').
Using the default names will therefore facilitate the use of the library
considerably.

+++

(key_and_id)=
### Keys and IDs

ms3 uses keys for grouping files. The way how these keys are being used
is transitioning in the 0.5.x versions:

-   **\< 0.5** keys were arbitrary and used by some methods to bring
    groups of files together. For example, `Parse.attach_labels()` would
    take a key with a group of scores and, as the parameter
    `annotation_key`, the key with a group of annotations and then use
    `Parse.match_files()` to match the files from the given keys.
-   **\>= 0.6.x** keys are used to address sub-corpora that are assumed
    to have a particular `corpus_structure`{.interpreted-text
    role="ref"}. The method `Parse.attach_labels()` then takes only one
    key and uses the key\'s :py`~.parse.View`{.interpreted-text
    role="class"} object for matching files. The parameter
    `annotation_key` is replaced by `use` that can be used in the case
    that the View object has detected several annotation files for one
    or several pieces.
-   **0.5x** transitioning from the old to the new behaviour.

#### IDs

IDs are `(key, i)` pairs that identify one particular file (not piece)
found by a Parse object. They are used as dictionary keys except for
storing the information on file paths such as
:py`~.parse.full_paths`{.interpreted-text role="attr"} or
:py`~.parse.fnames`{.interpreted-text role="attr"} which are
dictionaries containing lists where only keys are dictionary keys,
whereas `i` is simply the list index of the respective file.

+++

(label_types)=
### Label types 

ms3 recognizes and disambiguates different types of labels, depending on
how they are encoded in MuseScore, see
`harmony_layer <harmony_layer>`{.interpreted-text role="ref"}.

Independent of the type, ms3 will also try to infer whether a label
conforms to the DCML syntax and/or other regular expressions registered
via `ms3.Score.new_type`{.interpreted-text role="meth"}. The column
`regex_match <regex_match>`{.interpreted-text role="ref"} contains for
each label the name of the first regEx that matched. information will
appear with a subtype, e.g. `0 (dcml)`.

See also `~.score.Score.infer_label_types`{.interpreted-text
role="attr"}.

+++

(mc_vs_mn)=
### Measure counts (MC) vs. measure numbers (MN) 

Measure counts are strictly increasing numbers for all \<measure\> nodes
in the score, regardless of their length. This information is crucial
for correctly addressing positions in a MuseScore file and are shown in
the software\'s status bar. The first measure is always counted as 1
(following MuseScore\'s convention), even if it is an anacrusis.

Measure numbers are the traditional way by which humans refer to
positions in a score. They follow a couple of conventions which can be
summarised as counting complete bars. Quite often, a complete bar (MN)
can be made up of two \<measure\> nodes (MC). In the context of this
library, score addressability needs to be maintained for humans and
computers, therefore a mapping MC -\> MN is preserved in the score
information DataFrames.

+++

(onsets)=
### Onset positions 

Onsets express positions of events in a score as their distance from the
beginning of the corresponding `MC or MN <mc_vs_mn>`{.interpreted-text
role="ref"}. The distances are expressed as fractions of a whole note.
In other words, beat 1 has onset `0`, an event on beat 2 of a 4/4 meter
has onset `1/4` and so on.

Since there are two ways of referencing measures (MC and MN), there are
also two ways of expressing onsets:

-   `mc_onset` expresses the distance from the corresponding MC
-   `mn_onset` expresses the distance from the corresponding MN

In most cases, the two values value will be identical, but take as an
example the case where a 4/4 measure with MN 8 is divided into MC 9 of
length 3/4 and MC 10 of length 1/4 because of a repeat sign or a double
bar line. Since MC 9 corresponds to the first part of MN 8, the two
onset values are identical. But for the anacrusis on beat 4, the values
differ: `mc_onset` is `0` but `mn_onset` is `3/4` because this is the
distance from MN 8.

+++

(read_only)=
### Read-only mode 

For parsing faster using less memory. Scores parsed in read-only mode
cannot be changed because the original XML structure is not kept in
memory.

+++

(fifths)=
### Stacks-of-fifths intervals 

In order to express note names (tonal pitch classes,
`tpc <tpc>`{.interpreted-text role="ref"}), and scale degrees, ms3 uses
stacks of fifths (the only way to express these as a single integer).
For note names, `0` corresponds to C, for scale degrees to the local
tonic.

  -------------------------------------------------
  fifths   note name   interval   scale degree
  -------- ----------- ---------- -----------------
  -6       Gb          d5         b5

  -5       Db          m2         b2

  -4       Ab          m6         b6 (6 in minor)

  -3       Eb          m3         b3 (3 in minor)

  -2       Bb          m7         b7 (7 in minor)

  -1       F           P4         4

  0        C           P1         1

  1        G           P5         5

  2        D           M2         2

  3        A           M6         6 (#6 in minor)

  4        E           M3         3 (#3 in minor)

  5        B           M7         7 (#7 in minor)

  6        F#          A4         #4
  -------------------------------------------------

### Voltas

\"Prima/Seconda volta\" is the Italian designation for \"First/Second
time\". Therefore, in the context of ms3, we refer to \'a volta\' as one
of several endings. By convention, all endings should have the same
measure numbers (MN), which are often differentiated by lowercase
letters, e.g. `8a` for the first ending and `8b` for the second ending.
In MuseScore, correct bar numbers can be achieved by excluding `8b` from
the count or, if the endings have more than one bar, by subtracting the
corresponding number from the second ending\'s count. For example, in
order to achieve the correct MNs `[7a 8a][7b 8b]`, you would add `-2` to
7b\'s count which otherwise would come out as 9.

ms3 checks for incorrect MNs and warns you if the score needs
correction. It will also ask you to make all voltas the same length. If
this is not possible for editorial reasons (although often the length of
the second volta is arbitrary), ignore the warning and check in the
`measures <measures>`{.interpreted-text role="ref"} table if the MN are
correct for your purposes.

+++

(score_information)=
## Tables with score information 

This section gives an overview of the various tables that ms3 exposes
after parsing a MuseScore file. Their names, e.g. `measures`, correspond
to the properties of :py`~.score.Score`{.interpreted-text role="class"}
and the methods of :py`~.parse.Parse`{.interpreted-text role="class"}
with which they can be retrieved. They come as
`pandas.DataFrame`{.interpreted-text role="obj"} objects. The available
tables are:

All score information, except the metadata, is contained in the
following two tables:

-   `measures <measures>`{.interpreted-text role="ref"}
-   `notes <notes>`{.interpreted-text role="ref"}
-   `rests <rests>`{.interpreted-text role="ref"}
-   `notes_and_rests <notes_and_rests>`{.interpreted-text role="ref"}
-   `chords <chords>`{.interpreted-text role="ref"}: **Not to be
    confounded with labels or chord annotations**, a chord is a
    notational unit in which all included notes are part of the same
    notational layer and have the same onset and duration. Every chord
    has a `chord_id` and every note is part of a chord. These tables are
    used to convey score information that is not attached to a
    particular note, such as lyrics, staff text, dynamics and other
    markup.
-   `labels <labels>`{.interpreted-text role="ref"}
-   `expanded <expanded>`{.interpreted-text role="ref"}
-   `cadences <cadences>`{.interpreted-text role="ref"}
-   `events <events>`{.interpreted-text role="ref"}

For each of the available tables you will see an example and you can
click on the columns to learn about their meanings.

### Measures

DataFrame representing the measures in the MuseScore file (which can be
incomplete measures, see `mc_vs_mn`{.interpreted-text role="ref"})
together with their respective features. Required for unfolding repeats.

```{code-cell} ipython3
:tags: [remove-input]
import ms3
s = ms3.Score("~/ms3/old_tests/MS3/05_symph_fant.mscx", level='c')
```

```{code-cell} ipython3
:tags: [remove-output]
s.mscx.measures()   # access through a Score object
p.measures()      # access through a Parse object
```

```{code-cell} ipython3
:tags: [remove-input]
s.mscx.measures()
```

### Notes

DataFrame representing the notes in the MuseScore file.

``` python
>>> s.mscx.notes() # access through a Score object
>>> p.notes()      # access through a Parse object
```

<table style="width:91%;">
<colgroup>
<col style="width: 3%" />
<col style="width: 3%" />
<col style="width: 6%" />
<col style="width: 6%" />
<col style="width: 6%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 6%" />
<col style="width: 7%" />
<col style="width: 11%" />
<col style="width: 5%" />
<col style="width: 4%" />
<col style="width: 3%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 6%" />
</colgroup>
<thead>
<tr class="header">
<th><code class="interpreted-text" role="ref">mc &lt;mc&gt;</code></th>
<th><code class="interpreted-text" role="ref">mn &lt;mn&gt;</code></th>
<th><code class="interpreted-text"
role="ref">mc_onset &lt;mc_onset&gt;</code></th>
<th><code class="interpreted-text"
role="ref">mn_onset &lt;mn_onset&gt;</code></th>
<th><code class="interpreted-text"
role="ref">timesig &lt;timesig&gt;</code></th>
<th><code class="interpreted-text"
role="ref">staff &lt;staff&gt;</code></th>
<th><code class="interpreted-text"
role="ref">voice &lt;voice&gt;</code></th>
<th><code class="interpreted-text"
role="ref">duration &lt;duration&gt;</code></th>
<th><code class="interpreted-text"
role="ref">gracenote &lt;gracenote&gt;</code></th>
<th><code class="interpreted-text"
role="ref">nominal_duration &lt;nominal_duration&gt;</code></th>
<th><code class="interpreted-text"
role="ref">scalar &lt;scalar&gt;</code></th>
<th><code class="interpreted-text"
role="ref">tied &lt;tied&gt;</code></th>
<th><code class="interpreted-text"
role="ref">tpc &lt;tpc&gt;</code></th>
<th><code class="interpreted-text"
role="ref">midi &lt;midi&gt;</code></th>
<th><code class="interpreted-text"
role="ref">volta &lt;volta&gt;</code></th>
<th><code class="interpreted-text"
role="ref">chord_id &lt;chord_id&gt;</code></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td>4/4</td>
<td><blockquote>
<p>4</p>
</blockquote></td>
<td><blockquote>
<p>2</p>
</blockquote></td>
<td>1/8</td>
<td>NaN</td>
<td>1/8</td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td>&lt;NA&gt;</td>
<td><blockquote>
<p>-1</p>
</blockquote></td>
<td><blockquote>
<p>53</p>
</blockquote></td>
<td>&lt;NA&gt;</td>
<td><blockquote>
<p>4</p>
</blockquote></td>
</tr>
<tr class="even">
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td>4/4</td>
<td><blockquote>
<p>3</p>
</blockquote></td>
<td><blockquote>
<p>2</p>
</blockquote></td>
<td>3/4</td>
<td>NaN</td>
<td>1/2</td>
<td>3/2</td>
<td>&lt;NA&gt;</td>
<td><blockquote>
<p>-1</p>
</blockquote></td>
<td><blockquote>
<p>77</p>
</blockquote></td>
<td>&lt;NA&gt;</td>
<td><blockquote>
<p>1</p>
</blockquote></td>
</tr>
</tbody>
</table>

### Rests

DataFrame representing the rests in the MuseScore file.

``` python
>>> s.mscx.rests() # access through a Score object
>>> p.rests()      # access through a Parse object
```

<table style="width:92%;">
<colgroup>
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 9%" />
<col style="width: 9%" />
<col style="width: 8%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 9%" />
<col style="width: 16%" />
<col style="width: 7%" />
<col style="width: 7%" />
</colgroup>
<thead>
<tr class="header">
<th><code class="interpreted-text" role="ref">mc &lt;mc&gt;</code></th>
<th><code class="interpreted-text" role="ref">mn &lt;mn&gt;</code></th>
<th><code class="interpreted-text"
role="ref">mc_onset &lt;mc_onset&gt;</code></th>
<th><code class="interpreted-text"
role="ref">mn_onset &lt;mn_onset&gt;</code></th>
<th><code class="interpreted-text"
role="ref">timesig &lt;timesig&gt;</code></th>
<th><code class="interpreted-text"
role="ref">staff &lt;staff&gt;</code></th>
<th><code class="interpreted-text"
role="ref">voice &lt;voice&gt;</code></th>
<th><code class="interpreted-text"
role="ref">duration &lt;duration&gt;</code></th>
<th><code class="interpreted-text"
role="ref">nominal_duration &lt;nominal_duration&gt;</code></th>
<th><code class="interpreted-text"
role="ref">scalar &lt;scalar&gt;</code></th>
<th><code class="interpreted-text"
role="ref">volta &lt;volta&gt;</code></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td>4/4</td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="even">
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td>4/4</td>
<td><blockquote>
<p>2</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td>&lt;NA&gt;</td>
</tr>
</tbody>
</table>

+++

(notes_and_rests)=
### Notes and Rests 

DataFrame combining `notes`{.interpreted-text role="ref"} and
`rests`{.interpreted-text role="ref"}.

``` python
>>> s.mscx.notes_and_rests() # access through a Score object
>>> p.notes_and_rests()      # access through a Parse object
```

<table style="width:91%;">
<colgroup>
<col style="width: 3%" />
<col style="width: 3%" />
<col style="width: 6%" />
<col style="width: 6%" />
<col style="width: 6%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 6%" />
<col style="width: 7%" />
<col style="width: 11%" />
<col style="width: 5%" />
<col style="width: 4%" />
<col style="width: 3%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 6%" />
</colgroup>
<thead>
<tr class="header">
<th><code class="interpreted-text" role="ref">mc &lt;mc&gt;</code></th>
<th><code class="interpreted-text" role="ref">mn &lt;mn&gt;</code></th>
<th><code class="interpreted-text"
role="ref">mc_onset &lt;mc_onset&gt;</code></th>
<th><code class="interpreted-text"
role="ref">mn_onset &lt;mn_onset&gt;</code></th>
<th><code class="interpreted-text"
role="ref">timesig &lt;timesig&gt;</code></th>
<th><code class="interpreted-text"
role="ref">staff &lt;staff&gt;</code></th>
<th><code class="interpreted-text"
role="ref">voice &lt;voice&gt;</code></th>
<th><code class="interpreted-text"
role="ref">duration &lt;duration&gt;</code></th>
<th><code class="interpreted-text"
role="ref">gracenote &lt;gracenote&gt;</code></th>
<th><code class="interpreted-text"
role="ref">nominal_duration &lt;nominal_duration&gt;</code></th>
<th><code class="interpreted-text"
role="ref">scalar &lt;scalar&gt;</code></th>
<th><code class="interpreted-text"
role="ref">tied &lt;tied&gt;</code></th>
<th><code class="interpreted-text"
role="ref">tpc &lt;tpc&gt;</code></th>
<th><code class="interpreted-text"
role="ref">midi &lt;midi&gt;</code></th>
<th><code class="interpreted-text"
role="ref">volta &lt;volta&gt;</code></th>
<th><code class="interpreted-text"
role="ref">chord_id &lt;chord_id&gt;</code></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td>4/4</td>
<td><blockquote>
<p>4</p>
</blockquote></td>
<td><blockquote>
<p>2</p>
</blockquote></td>
<td>1/8</td>
<td>NaN</td>
<td>1/8</td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td>&lt;NA&gt;</td>
<td><blockquote>
<p>-1</p>
</blockquote></td>
<td><blockquote>
<p>53</p>
</blockquote></td>
<td>&lt;NA&gt;</td>
<td><blockquote>
<p>4</p>
</blockquote></td>
</tr>
<tr class="even">
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td>4/4</td>
<td><blockquote>
<p>3</p>
</blockquote></td>
<td><blockquote>
<p>2</p>
</blockquote></td>
<td>3/4</td>
<td>NaN</td>
<td>1/2</td>
<td>3/2</td>
<td>&lt;NA&gt;</td>
<td><blockquote>
<p>-1</p>
</blockquote></td>
<td><blockquote>
<p>77</p>
</blockquote></td>
<td>&lt;NA&gt;</td>
<td><blockquote>
<p>1</p>
</blockquote></td>
</tr>
<tr class="odd">
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td>4/4</td>
<td><blockquote>
<p>3</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td>1/2</td>
<td>NaN</td>
<td>1/2</td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="even">
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td>4/4</td>
<td><blockquote>
<p>4</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td>1/2</td>
<td>NaN</td>
<td>1/2</td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
</tbody>
</table>

### Chords

::: note
::: title
Note
:::

The use of the word chords, here, is very specific because its meaning
stems entirely from the MuseScore XML source code. If you are interested
in chord labels, please refer to `labels`{.interpreted-text role="ref"}
or `expanded`{.interpreted-text role="ref"}.
:::

In a MuseScore file, every note is enclosed by a \<Chord\> tag. One
\<Chord\> tag can enclose several notes, as long as they occur in the
same `staff <staff>`{.interpreted-text role="ref"} and
`voice <voice>`{.interpreted-text role="ref"} (notational layer). As a
consequence, notes belonging to the same \<Chord\> have the same onset
and the same duration.

**Why chord lists?** Most of the markup (such as articulation, lyrics
etc.) in a MuseScore file is attached not to individual notes but
instead to \<Chord\> tags. It might be a matter of interpretation to
what notes exactly the symbols pertain, which is why it is left for the
interested user to link the chord list with the corresponding note list
by joining on the `chord_id <chord_id>`{.interpreted-text role="ref"}
column of each.

#### Standard columns

The output of the analogous commands depends on what markup is available
in the score (`see below <chords_dynamic>`{.interpreted-text
role="ref"}). The columns that are always present in a chord list are
exactly the same as (and correspond to) those of a
`note list <notes>`{.interpreted-text role="ref"} except for
`tied <tied>`{.interpreted-text role="ref"},
`tpc <tpc>`{.interpreted-text role="ref"}, and
`midi <midi>`{.interpreted-text role="ref"}.

<table style="width:92%;">
<colgroup>
<col style="width: 3%" />
<col style="width: 3%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 5%" />
<col style="width: 5%" />
<col style="width: 7%" />
<col style="width: 8%" />
<col style="width: 13%" />
<col style="width: 6%" />
<col style="width: 5%" />
<col style="width: 7%" />
</colgroup>
<thead>
<tr class="header">
<th><code class="interpreted-text" role="ref">mc &lt;mc&gt;</code></th>
<th><code class="interpreted-text" role="ref">mn &lt;mn&gt;</code></th>
<th><code class="interpreted-text"
role="ref">mc_onset &lt;mc_onset&gt;</code></th>
<th><code class="interpreted-text"
role="ref">mn_onset &lt;mn_onset&gt;</code></th>
<th><code class="interpreted-text"
role="ref">timesig &lt;timesig&gt;</code></th>
<th><code class="interpreted-text"
role="ref">staff &lt;staff&gt;</code></th>
<th><code class="interpreted-text"
role="ref">voice &lt;voice&gt;</code></th>
<th><code class="interpreted-text"
role="ref">duration &lt;duration&gt;</code></th>
<th><code class="interpreted-text"
role="ref">gracenote &lt;gracenote&gt;</code></th>
<th><code class="interpreted-text"
role="ref">nominal_duration &lt;nominal_duration&gt;</code></th>
<th><code class="interpreted-text"
role="ref">scalar &lt;scalar&gt;</code></th>
<th><code class="interpreted-text"
role="ref">volta &lt;volta&gt;</code></th>
<th><code class="interpreted-text"
role="ref">chord_id &lt;chord_id&gt;</code></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td>1/2</td>
<td>1/2</td>
<td>4/4</td>
<td><blockquote>
<p>3</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td>1/2</td>
<td>NaN</td>
<td>1/2</td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td>&lt;NA&gt;</td>
<td><blockquote>
<p>0</p>
</blockquote></td>
</tr>
<tr class="even">
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td>0</td>
<td>0</td>
<td>4/4</td>
<td><blockquote>
<p>3</p>
</blockquote></td>
<td><blockquote>
<p>2</p>
</blockquote></td>
<td>3/4</td>
<td>NaN</td>
<td>1/2</td>
<td>3/2</td>
<td>&lt;NA&gt;</td>
<td><blockquote>
<p>1</p>
</blockquote></td>
</tr>
</tbody>
</table>

Such a reduced table can be retrieved using
:py`Score.mscx.parsed.get_chords(mode='strict') <.bs4_parser._MSCX_bs4.get_chords()>`{.interpreted-text
role="meth"}

+++

(chords_dynamic)=
#### Dynamic columns 

Leaving the standard columns aside, the normal interface for accessing
chord lists calls
:py`Score.mscx.parsed.get_chords(mode='auto') <.bs4_parser._MSCX_bs4.get_chords()>`{.interpreted-text
role="meth"} meaning that only columns are included that have at least
one non empty value. The following table shows the first two non-empty
values for each column when parsing all scores included in the [ms3
repository](https://github.com/johentsch/ms3) for demonstration
purposes:

``` python
>>> s.mscx.chords()   # access through a Score object
>>> p.chords()      # access through a Parse object
```

<table style="width:94%;">
<colgroup>
<col style="width: 4%" />
<col style="width: 6%" />
<col style="width: 8%" />
<col style="width: 5%" />
<col style="width: 7%" />
<col style="width: 2%" />
<col style="width: 3%" />
<col style="width: 9%" />
<col style="width: 7%" />
<col style="width: 8%" />
<col style="width: 7%" />
<col style="width: 6%" />
<col style="width: 5%" />
<col style="width: 3%" />
<col style="width: 6%" />
</colgroup>
<thead>
<tr class="header">
<th><code class="interpreted-text"
role="ref">lyrics:1 &lt;lyrics_1&gt;</code></th>
<th><blockquote>
<p><code class="interpreted-text"
role="ref">dynamics &lt;dynamics&gt;</code></p>
</blockquote></th>
<th><blockquote>
<p><code class="interpreted-text"
role="ref">articulation &lt;articulation&gt;</code></p>
</blockquote></th>
<th><code class="interpreted-text"
role="ref">staff_text &lt;staff_text&gt;</code></th>
<th><blockquote>
<p><code class="interpreted-text"
role="ref">tempo &lt;tempo&gt;</code></p>
</blockquote></th>
<th><code class="interpreted-text"
role="ref">qpm &lt;qpm&gt;</code></th>
<th><code class="interpreted-text"
role="ref">slur &lt;slur&gt;</code></th>
<th><code class="interpreted-text"
role="ref">decrescendo_hairpin &lt;hairpins&gt;</code></th>
<th><code class="interpreted-text"
role="ref">diminuendo_line &lt;cresc_lines&gt;</code></th>
<th><code class="interpreted-text"
role="ref">crescendo_hairpin &lt;hairpins&gt;</code></th>
<th><code class="interpreted-text"
role="ref">crescendo_line &lt;cresc_lines&gt;</code></th>
<th><code class="interpreted-text"
role="ref">Ottava:15mb &lt;ottava&gt;</code></th>
<th><code class="interpreted-text"
role="ref">Ottava:8va &lt;ottava&gt;</code></th>
<th><code class="interpreted-text"
role="ref">pedal &lt;pedal&gt;</code></th>
<th><code class="interpreted-text"
role="ref">system_text &lt;system_text&gt;</code></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>Grave</td>
<td><blockquote>
<p>45</p>
</blockquote></td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="even">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>0</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="odd">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>0</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="even">
<td>&lt;NA&gt;</td>
<td>p</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="odd">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>articStaccatoBelow</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>2</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="even">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>articStaccatoBelow</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>2</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="odd">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>simile</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="even">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>espr.</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="odd">
<td>&lt;NA&gt;</td>
<td>other-dynamics</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="even">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>0</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="odd">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>0, 1</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="even">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>0</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="odd">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>0</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="even">
<td>Sta</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="odd">
<td>bat</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="even">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>Andante amoroso</td>
<td>55</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="odd">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>0</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="even">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>0</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="odd">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>0</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="even">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>0</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="odd">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>0</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="even">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>0</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="odd">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>0</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="even">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>0</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="odd">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>0</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="even">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>0</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
</tr>
<tr class="odd">
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>&lt;NA&gt;</td>
<td>Swing</td>
</tr>
</tbody>
</table>

### Labels

DataFrame representing the annotation labels contained in the score. The
output can be controlled by changing the `labels_cfg` configuration.

``` python
>>> s.mscx.labels()   # access through a Score object
>>> p.labels()      # access through a Parse object
```

<table style="width:92%;">
<colgroup>
<col style="width: 5%" />
<col style="width: 5%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 10%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 8%" />
<col style="width: 13%" />
</colgroup>
<thead>
<tr class="header">
<th><code class="interpreted-text" role="ref">mc &lt;mc&gt;</code></th>
<th><code class="interpreted-text" role="ref">mn &lt;mn&gt;</code></th>
<th><code class="interpreted-text"
role="ref">mc_onset &lt;mc_onset&gt;</code></th>
<th><code class="interpreted-text"
role="ref">mn_onset &lt;mn_onset&gt;</code></th>
<th><code class="interpreted-text"
role="ref">timesig &lt;timesig&gt;</code></th>
<th><code class="interpreted-text"
role="ref">staff &lt;staff&gt;</code></th>
<th><code class="interpreted-text"
role="ref">voice &lt;voice&gt;</code></th>
<th><code class="interpreted-text"
role="ref">volta &lt;volta&gt;</code></th>
<th><code class="interpreted-text"
role="ref">label &lt;label&gt;</code></th>
<th><code class="interpreted-text"
role="ref">label_type &lt;label_type&gt;</code></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td>4/4</td>
<td><blockquote>
<p>3</p>
</blockquote></td>
<td><blockquote>
<p>2</p>
</blockquote></td>
<td>&lt;NA&gt;</td>
<td>.f.i</td>
<td>0 (dcml)</td>
</tr>
<tr class="even">
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td>1/4</td>
<td>1/4</td>
<td>4/4</td>
<td><blockquote>
<p>3</p>
</blockquote></td>
<td><blockquote>
<p>2</p>
</blockquote></td>
<td>&lt;NA&gt;</td>
<td>i6</td>
<td>0 (dcml)</td>
</tr>
</tbody>
</table>

### Expanded

If the score contains [DCML harmony
labels](https://github.com/DCMLab/standards), this DataFrames represents
them after splitting them into the encoded features and translating them
into scale degrees.

``` python
>>> s.mscx.expanded()   # access through a Score object
>>> p.expanded()      # access through a Parse object
```

<table style="width:92%;">
<colgroup>
<col style="width: 1%" />
<col style="width: 1%" />
<col style="width: 3%" />
<col style="width: 3%" />
<col style="width: 3%" />
<col style="width: 2%" />
<col style="width: 2%" />
<col style="width: 2%" />
<col style="width: 2%" />
<col style="width: 3%" />
<col style="width: 3%" />
<col style="width: 2%" />
<col style="width: 2%" />
<col style="width: 3%" />
<col style="width: 2%" />
<col style="width: 3%" />
<col style="width: 3%" />
<col style="width: 4%" />
<col style="width: 3%" />
<col style="width: 3%" />
<col style="width: 4%" />
<col style="width: 6%" />
<col style="width: 6%" />
<col style="width: 4%" />
<col style="width: 4%" />
<col style="width: 2%" />
<col style="width: 3%" />
</colgroup>
<thead>
<tr class="header">
<th><code class="interpreted-text" role="ref">mc &lt;mc&gt;</code></th>
<th><code class="interpreted-text" role="ref">mn &lt;mn&gt;</code></th>
<th><code class="interpreted-text"
role="ref">mc_onset &lt;mc_onset&gt;</code></th>
<th><code class="interpreted-text"
role="ref">mn_onset &lt;mn_onset&gt;</code></th>
<th><code class="interpreted-text"
role="ref">timesig &lt;timesig&gt;</code></th>
<th><code class="interpreted-text"
role="ref">staff &lt;staff&gt;</code></th>
<th><code class="interpreted-text"
role="ref">voice &lt;voice&gt;</code></th>
<th><code class="interpreted-text"
role="ref">volta &lt;volta&gt;</code></th>
<th><code class="interpreted-text"
role="ref">label &lt;label&gt;</code></th>
<th><code class="interpreted-text"
role="ref">globalkey &lt;globalkey&gt;</code></th>
<th><code class="interpreted-text"
role="ref">localkey &lt;localkey&gt;</code></th>
<th><code class="interpreted-text"
role="ref">pedal &lt;pedal&gt;</code></th>
<th><code class="interpreted-text"
role="ref">chord &lt;chord&gt;</code></th>
<th><code class="interpreted-text"
role="ref">numeral &lt;numeral&gt;</code></th>
<th><code class="interpreted-text"
role="ref">form &lt;form&gt;</code></th>
<th><code class="interpreted-text"
role="ref">figbass &lt;figbass&gt;</code></th>
<th><code class="interpreted-text"
role="ref">changes &lt;changes&gt;</code></th>
<th><code class="interpreted-text"
role="ref">relativeroot &lt;relativeroot&gt;</code></th>
<th><code class="interpreted-text"
role="ref">cadence &lt;cadence&gt;</code></th>
<th><code class="interpreted-text"
role="ref">phraseend &lt;phraseend&gt;</code></th>
<th><code class="interpreted-text"
role="ref">chord_type &lt;chord_type&gt;</code></th>
<th><code class="interpreted-text"
role="ref">globalkey_is_minor &lt;globalkey_is_minor&gt;</code></th>
<th><code class="interpreted-text"
role="ref">localkey_is_minor &lt;localkey_is_minor&gt;</code></th>
<th><code class="interpreted-text"
role="ref">chord_tones &lt;chord_tones&gt;</code></th>
<th><code class="interpreted-text"
role="ref">added_tones &lt;chord_tones&gt;</code></th>
<th><code class="interpreted-text"
role="ref">root &lt;root&gt;</code></th>
<th><code class="interpreted-text"
role="ref">bass_note &lt;bass_note&gt;</code></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td>4/4</td>
<td><blockquote>
<p>3</p>
</blockquote></td>
<td><blockquote>
<p>2</p>
</blockquote></td>
<td>&lt;NA&gt;</td>
<td>.f.i</td>
<td>f</td>
<td>i</td>
<td>NaN</td>
<td>i</td>
<td>i</td>
<td>NaN</td>
<td><blockquote>
<p>NaN</p>
</blockquote></td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>m</td>
<td>True</td>
<td>True</td>
<td>(0, -3, 1)</td>
<td>()</td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td><blockquote>
<p>0</p>
</blockquote></td>
</tr>
<tr class="even">
<td><blockquote>
<p>1</p>
</blockquote></td>
<td><blockquote>
<p>1</p>
</blockquote></td>
<td>1/4</td>
<td>1/4</td>
<td>4/4</td>
<td><blockquote>
<p>3</p>
</blockquote></td>
<td><blockquote>
<p>2</p>
</blockquote></td>
<td>&lt;NA&gt;</td>
<td>i6</td>
<td>f</td>
<td>i</td>
<td>NaN</td>
<td>i6</td>
<td>i</td>
<td>NaN</td>
<td><blockquote>
<p>6</p>
</blockquote></td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>m</td>
<td>True</td>
<td>True</td>
<td>(-3, 1, 0)</td>
<td>()</td>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td><blockquote>
<p>-3</p>
</blockquote></td>
</tr>
</tbody>
</table>

### Cadences

If DCML harmony labels include cadence labels, return only those. This
table is simply a filter on `expanded <expanded>`{.interpreted-text
role="ref"}. The table has the same columns and contains only rows that
include a cadence label. Just for convenience\...

``` python
>>> s.mscx.cadences   # access through a Score object
>>> p.cadences()      # access through a Parse object
```

+++

(form_labels)=
### Form labels 

``` python
>>> s.mscx.form_labels()  # access through a Score object
>>> p.form_labels()       # access through a Parse object
```

### Events

This DataFrame is the original tabular representation of the MuseScore
file\'s source code from which all other tables, except `measures` are
generated. The nested XML tags are transformed into column names.

The value `'∅'` is used for empty tags. For example, in the column
`Chord/Spanner/Slur` it would correspond to the tag structure
(formatting as in an MSCX file):

``` xml
<Chord>
  <Spanner type="Slur">
    <Slur>
      </Slur>
    </Spanner>
  </Chord>
```

The value `'/'` on the other hand represents a shortcut empty tag. For
example, in the column `Chord/grace16` it would correspond to the tag
structure (formatting as in an MSCX file):

``` xml
<Chord>
  <grace16/>
  </Chord>
```

## Parsing

This chapter explains how to

-   parse a single score to access and manipulate the contained
    information using a :py`~.score.Score`{.interpreted-text
    role="class"} object
-   parse a group of scores to access and manipulate the contained
    information using a :py`~.parse.Parse`{.interpreted-text
    role="class"} object.

### Parsing a single score

::: rst-class
bignums
:::

1.  Import the library.

    > To parse a single score, we will use the class
    > :py`~.score.Score`{.interpreted-text role="class"}. We could
    > import the whole library:
    >
    > ``` python
    > >>> import ms3
    > >>> s = ms3.Score()
    > ```
    >
    > or simply import the class:
    >
    > ``` python
    > >>> from ms3 import Score
    > >>> s = Score()
    > ```

2.  Locate the [MuseScore 3](https://musescore.org/en/download) score
    you want to parse.

    > ::: tip
    > ::: title
    > Tip
    > :::
    >
    > MSCZ files are ZIP files containing the uncompressed MSCX. In
    > order to trace the score\'s version history, it is recommended to
    > always work with MSCX files.
    > :::
    >
    > In the examples, we parse the annotated first page of Giovanni
    > Battista Pergolesi\'s influential *Stabat Mater*. The file is
    > called `stabat.mscx` and can be downloaded from
    > [here](https://raw.githubusercontent.com/johentsch/ms3/master/docs/stabat.mscx)
    > (open link and key `Ctrl + S` to save the file or right-click on
    > the link to `Save link as...`).

3.  Create a :py`~.score.Score`{.interpreted-text role="class"} object.

    > In the example, the MuseScore 3 file is located at
    > `~/ms3/docs/stabat.mscx` so we can simply create the object and
    > bind it to the variable `s` like so:
    >
    > ``` python
    > >>> from ms3 import Score
    > >>> s = Score('~/ms3/docs/stabat.mscx')
    > ```

4.  Inspect the object.

    > To have a look at the created object we can simply evoke its
    > variable:
    >
    > ``` python
    > >>> s
    > MuseScore file
    > --------------
    >
    > ~/ms3/docs/stabat.mscx
    >
    > Attached annotations
    > --------------------
    >
    > 48 labels:
    > staff  voice  label_type  color_name
    > 3      2      0 (dcml)    default       48
    > ```

#### Parsing options

::: {.automethod noindex=""}
ms3.score.Score.\_\_init\_\_
:::

### Parsing multiple scores

::: rst-class
bignums
:::

1.  Import the library.

    > To parse multiple scores, we will use the class
    > :py`ms3.Parse <.parse.Parse>`{.interpreted-text role="class"}. We
    > could import the whole library:
    >
    > ``` python
    > >>> import ms3
    > >>> p = ms3.Parse()
    > ```
    >
    > or simply import the class:
    >
    > ``` python
    > >>> from ms3 import Parse
    > >>> p = Parse()
    > ```

2.  Locate the folder containing MuseScore files.

    > In this example, we are going to parse all files included in the
    > [ms3 repository](https://github.com/johentsch/ms3) which has been
    > [cloned](https://www.atlassian.com/git/tutorials/setting-up-a-repository/git-clone)
    > into the home directory and therefore has the path `~/ms3`.

3.  Create a :py`~.parse.Parse`{.interpreted-text role="class"} object

    > The object is created by calling it with the directory to scan,
    > and bound to the typical variable `p`. ms3 scans the
    > subdirectories for corpora (see
    > `corpus_structure`{.interpreted-text role="ref"}) and assigns keys
    > automatically based on folder names (here \'docs\', and
    > \'tests\'):
    >
    > ``` python
    > >>> from ms3 import Parse
    > >>> p = Parse('~/ms3')
    > >>> p
    > ```
    >
    > ::: program-output
    > python examples/parse_directory.py
    > :::
    >
    > Without any further parameters, ms3 detects only file types that
    > it can potentially parse, i.e. MSCX, MSCZ, and TSV. In the
    > following example, we infer the location of our local MuseScore 3
    > installation (if \'auto\' fails, indicate the path to your
    > executable). As a result, ms3 also shows formats that MuseScore
    > can convert, such as XML, MIDI, or CAP.
    >
    > ``` python
    > >>> from ms3 import Parse
    > >>> p = Parse('~/ms3', ms='auto')
    > >>> p
    > ```
    >
    > ::: program-output
    > python examples/parse_directory_xml.py
    > :::
    >
    > By default, present TSV files are detected and can be parsed as
    > well, allowing one to access already extracted information without
    > parsing the scores anew. In order to select only particular files,
    > a regular expression can be passed to the parameter `file_re`. In
    > the following example, only files ending on `mscx` are collected
    > in the object (`$` stands for the end of the filename, without it,
    > files including the string \'mscx\' anywhere in their names would
    > be selected, too):
    >
    > ::: caution
    > ::: title
    > Caution
    > :::
    >
    > The parameter `key` will be deprecated from version 0.6.0 onwards.
    > See `keys_and_ids`{.interpreted-text role="ref"}.
    > :::
    >
    > ``` python
    > >>> from ms3 import Parse
    > >>> p = Parse('~/ms3', file_re='mscx$', key='ms3')
    > >>> p
    > ```
    >
    > ::: program-output
    > python examples/parse_directory_mscx.py
    > :::
    >
    > In this example, we assigned the key `'ms3'`. Note that the same
    > MSCX files that were distributed over several keys in the previous
    > example are now grouped together. Keys allow operations to be
    > performed on a particular group of selected files. For example, we
    > could add MSCX files from another folder using the method
    > :py`~.parse.Parse.add_dir`{.interpreted-text role="meth"}:
    >
    > ``` python
    > >>> p.add_dir('~/other_folder', file_re='mscx$')
    > >>> p
    > ```
    >
    > ::: program-output
    > python examples/parse_other_directory.py
    > :::

4.  Parse the scores.

    > In order to simply parse all registered MuseScore files, call the
    > method :py`~.parse.Parse.parse_mscx`{.interpreted-text
    > role="meth"}. Instead, you can pass the argument `keys` to parse
    > only one (or several) selected group(s) to save time. The argument
    > `level` controls how many log messages you see; here, it is set to
    > \'critical\' or \'c\' to suppress all warnings:
    >
    > ``` python
    > >>> p.parse_mscx(keys='ms3', level='c')
    > >>> p
    > ```
    >
    > ::: program-output
    > python examples/parse_key.py
    > :::
    >
    > As we can see, only the files with the key \'ms3\' were parsed and
    > the table shows an overview of the counts of the included label
    > types in the different notational layers (i.e. staff & voice),
    > grouped by their colours.

#### Parsing options

::: {.automethod noindex=""}
ms3.parse.Parse.\_\_init\_\_
:::

## Extracting score information

One of ms3\'s main functionalities is storing the information contained
in parsed scores as tabular files (TSV format). More information on the
generated files is summarized `here <tabular_info>`{.interpreted-text
role="ref"}

### Using the commandline

The most convenient way to achieve this is the command `ms3 extract` and
its capital-letter parameters summarize the available tables:

``` console
-M [folder], --measures [folder]
                      Folder where to store TSV files with measure information needed for tasks such as unfolding repetitions.
-N [folder], --notes [folder]
                      Folder where to store TSV files with information on all notes.
-R [folder], --rests [folder]
                      Folder where to store TSV files with information on all rests.
-L [folder], --labels [folder]
                      Folder where to store TSV files with information on all annotation labels.
-X [folder], --expanded [folder]
                      Folder where to store TSV files with expanded DCML labels.
-E [folder], --events [folder]
                      Folder where to store TSV files with all events (notes, rests, articulation, etc.) without further processing.
-C [folder], --chords [folder]
                      Folder where to store TSV files with <chord> tags, i.e. groups of notes in the same voice with identical onset and duration. The tables include lyrics, slurs, and other markup.
-D [path], --metadata [path]
                      Directory or full path for storing one TSV file with metadata. If no filename is included in the path, it is called metadata.tsv
```

The typical way to use this command for a corpus of scores is to keep
the MuseScore files in a subfolder (called, for example, `MS3`) and to
use the parameters\' default values, effectively creating additional
subfolders for each extracted aspect next to each folder containing
MuseScore files. For example if we take the folder structure of the [ms3
repository](https://github.com/johentsch/ms3):

``` console
ms3
├── docs
│   ├── cujus.mscx
│   ├── o_quam.mscx
│   ├── quae.mscx
│   └── stabat.mscx
└── tests
    ├── MS3
    │   ├── 05_symph_fant.mscx
    │   ├── 76CASM34A33UM.mscx
    │   ├── BWV_0815.mscx
    │   ├── D973deutscher01.mscx
    │   ├── Did03M-Son_regina-1762-Sarti.mscx
    │   ├── K281-3.mscx
    │   └── stabat_03_coloured.mscx
    └── repeat_dummies
        ├── repeats0.mscx
        ├── repeats1.mscx
        └── repeats2.mscx
```

Upon calling `ms3 extract -N`, two new `notes` folders containing note
lists are created:

``` console
ms3
├── docs
│   ├── cujus.mscx
│   ├── o_quam.mscx
│   ├── quae.mscx
│   └── stabat.mscx
├── notes
│   ├── cujus.tsv
│   ├── o_quam.tsv
│   ├── quae.tsv
│   └── stabat.tsv
└── tests
    ├── MS3
    │   ├── 05_symph_fant.mscx
    │   ├── 76CASM34A33UM.mscx
    │   ├── BWV_0815.mscx
    │   ├── D973deutscher01.mscx
    │   ├── Did03M-Son_regina-1762-Sarti.mscx
    │   ├── K281-3.mscx
    │   └── stabat_03_coloured.mscx
    ├── notes
    │   ├── 05_symph_fant.tsv
    │   ├── 76CASM34A33UM.tsv
    │   ├── BWV_0815.tsv
    │   ├── D973deutscher01.tsv
    │   ├── Did03M-Son_regina-1762-Sarti.tsv
    │   ├── K281-3.tsv
    │   ├── repeats0.tsv
    │   ├── repeats1.tsv
    │   ├── repeats2.tsv
    │   └── stabat_03_coloured.tsv
    └── repeat_dummies
        ├── repeats0.mscx
        ├── repeats1.mscx
        └── repeats2.mscx
```

We witness this behaviour because the default value is `../notes`,
interpreted as relative path in relation to each MuseScore file.
Alternatively, a **relative path** can be specified **without** initial
`./` or `../`, e.g. `ms3 extract -N notes`, to store the note lists in a
recreated sub-directory structure:

``` console
ms3
├── docs
├── notes
│   ├── docs
│   └── tests
│       ├── MS3
│       └── repeat_dummies
└── tests
    ├── MS3
    └── repeat_dummies
```

A third option consists in specifying an **absolute path** which causes
all note lists to be stored in the specified folder, e.g.
`ms3 extract -N ~/notes`:

``` console
~/notes
├── 05_symph_fant.tsv
├── 76CASM34A33UM.tsv
├── BWV_0815.tsv
├── cujus.tsv
├── D973deutscher01.tsv
├── Did03M-Son_regina-1762-Sarti.tsv
├── K281-3.tsv
├── o_quam.tsv
├── quae.tsv
├── repeats0.tsv
├── repeats1.tsv
├── repeats2.tsv
├── stabat_03_coloured.tsv
└── stabat.tsv
```

Note that this leads to problems if MuseScore files from different
subdirectories have identical filenames. In any case it is good practice
to not use nested folders to allow for easier file access. For example,
a typical [DCML corpus](https://github.com/DCMLab/dcml_corpora) will
store all MuseScore files in the `MS3` folder and include at least the
folders created by `ms3 extract -N -M -X`:

``` console
.
├── harmonies
├── measures
├── MS3
└── notes
```

### Extracting score information manually

What `ms3 extract` effectively does is creating a
:py`~.parse.Parse`{.interpreted-text role="class"} object, calling its
method :py`~.parse.Parse.parse_mscx`{.interpreted-text role="meth"} and
then :py`~.parse.Parse.store_lists`{.interpreted-text role="meth"}. In
addition to the command, the method allows for storing two additional
aspects, namely `notes_and_rests` and `cadences` (if the score contains
cadence labels). For each of the available aspects,
`{notes, measures, rests, notes_and_rests, events, labels, chords, cadences, expanded}`,
the method provides two parameters, namely `_folder` (where to store
TSVs) and `_suffix`, i.e. a slug appended to the respective filenames.
If the parameter `simulate=True` is passed, no files are written but the
file paths to be created are returned. Since corpora might have quite
diverse directory structures, ms3 gives you various ways of specifying
folders which will be explained in detail in the following section.

Briefly, the rules for specifying the folders are as follows:

-   absolute folder (e.g. `~/labels`): Store all files in this
    particular folder without creating subfolders.
-   relative folder starting with `./` or `../`: relative folders are
    created \"at the end\" of the original subdirectory structure, i.e.
    relative to the MuseScore files.
-   relative folder not starting with `./` or `../` (e.g. `rests`):
    relative folders are created at the top level (of the original
    directory or the specified `root_dir`) and the original subdirectory
    structure is replicated in each of them.

To see examples for the three possibilities, see the following section.

+++

(specifying_folders)=
#### Specifying folders 

Consider a two-level folder structure contained in the root directory
`.` which is the one passed to :py`~.parse.Parse`{.interpreted-text
role="class"}:

``` console
.
├── docs
│   ├── cujus.mscx
│   ├── o_quam.mscx
│   ├── quae.mscx
│   └── stabat.mscx
└── tests
    └── MS3
        ├── 05_symph_fant.mscx
        ├── 76CASM34A33UM.mscx
        ├── BWV_0815.mscx
        ├── D973deutscher01.mscx
        ├── Did03M-Son_regina-1762-Sarti.mscx
        └── K281-3.mscx
```

The first level contains the subdirectories [docs]{.title-ref} (4 files)
and [tests]{.title-ref} (6 files in the subdirectory [MS3]{.title-ref}).
Now we look at the three different ways to specify folders for storing
notes and measures.

##### Absolute Folders

When we specify absolute paths, all files are stored in the specified
directories. In this example, the measures and notes are stored in the
two specified subfolders of the home directory [\~]{.title-ref},
regardless of the original subdirectory structure.

``` python
>>> p.store_lists(notes_folder='~/notes', measures_folder='~/measures')
```

``` console
~
├── measures
│   ├── 05_symph_fant.tsv
│   ├── 76CASM34A33UM.tsv
│   ├── BWV_0815.tsv
│   ├── cujus.tsv
│   ├── D973deutscher01.tsv
│   ├── Did03M-Son_regina-1762-Sarti.tsv
│   ├── K281-3.tsv
│   ├── o_quam.tsv
│   ├── quae.tsv
│   └── stabat.tsv
└── notes
    ├── 05_symph_fant.tsv
    ├── 76CASM34A33UM.tsv
    ├── BWV_0815.tsv
    ├── cujus.tsv
    ├── D973deutscher01.tsv
    ├── Did03M-Son_regina-1762-Sarti.tsv
    ├── K281-3.tsv
    ├── o_quam.tsv
    ├── quae.tsv
    └── stabat.tsv
```

##### Relative Folders

In contrast, specifying relative folders recreates the original
subdirectory structure. There are two different possibilities for that.
The first possibility is naming relative folder names, meaning that the
subdirectory structure (`docs` and `tests`) is recreated in each of the
folders:

``` python
>>> p.store_lists(root_dir='~/tsv', notes_folder='notes', measures_folder='measures')
```

``` console
~/tsv
├── measures
│   ├── docs
│   │   ├── cujus.tsv
│   │   ├── o_quam.tsv
│   │   ├── quae.tsv
│   │   └── stabat.tsv
│   └── tests
│       └── MS3
│           ├── 05_symph_fant.tsv
│           ├── 76CASM34A33UM.tsv
│           ├── BWV_0815.tsv
│           ├── D973deutscher01.tsv
│           ├── Did03M-Son_regina-1762-Sarti.tsv
│           └── K281-3.tsv
└── notes
    ├── docs
    │   ├── cujus.tsv
    │   ├── o_quam.tsv
    │   ├── quae.tsv
    │   └── stabat.tsv
    └── tests
        └── MS3
            ├── 05_symph_fant.tsv
            ├── 76CASM34A33UM.tsv
            ├── BWV_0815.tsv
            ├── D973deutscher01.tsv
            ├── Did03M-Son_regina-1762-Sarti.tsv
            └── K281-3.tsv
```

Note that in this example, we have specified a `root_dir`. Leaving this
argument out will create the same structure in the directory from which
the :py`~.parse.Parse`{.interpreted-text role="class"} object was
created, i.e. the folder structure would be:

``` console
.
├── docs
├── measures
│   ├── docs
│   └── tests
│       └── MS3
├── notes
│   ├── docs
│   └── tests
│       └── MS3
└── tests
    └── MS3
```

If, instead, you want to create the specified relative folders relative
to each MuseScore file\'s location, specify them with an initial dot.
`./` means \"relative to the original path\" and `../` one level up from
the original path. To exemplify both:

``` python
>>> p.store_lists(root_dir='~/tsv', notes_folder='./notes', measures_folder='../measures')
```

``` console
~/tsv
├── docs
│   └── notes
│       ├── cujus.tsv
│       ├── o_quam.tsv
│       ├── quae.tsv
│       └── stabat.tsv
├── measures
│   ├── cujus.tsv
│   ├── o_quam.tsv
│   ├── quae.tsv
│   └── stabat.tsv
└── tests
    ├── measures
    │   ├── 05_symph_fant.tsv
    │   ├── 76CASM34A33UM.tsv
    │   ├── BWV_0815.tsv
    │   ├── D973deutscher01.tsv
    │   ├── Did03M-Son_regina-1762-Sarti.tsv
    │   └── K281-3.tsv
    └── MS3
        └── notes
            ├── 05_symph_fant.tsv
            ├── 76CASM34A33UM.tsv
            ├── BWV_0815.tsv
            ├── D973deutscher01.tsv
            ├── Did03M-Son_regina-1762-Sarti.tsv
            └── K281-3.tsv
```

The `notes` folders are created in directories where MuseScore files are
located, and the `measures` folders one directory above, respectively.
Leaving out the `root_dir` argument would lead to the same folder
structure but in the directory from which the
:py`~.parse.Parse`{.interpreted-text role="class"} object has been
created. In a similar manner, the arguments
`p.store_lists(notes_folder='.', measures_folder='.')` would create the
TSV files just next to the MuseScore files. However, this would lead to
warnings such as

::: warning
::: title
Warning
:::

The notes at \~/ms3/docs/cujus.tsv have been overwritten with measures.
:::

In such a case we need to specify a suffix for at least one of both
aspects:

``` python
p.store_lists(notes_folder='.', notes_suffix='_notes',
              measures_folder='.', measures_suffix='_measures')
```

##### Examples

Before you are sure to have picked the right parameters for your desired
output, you can simply use the `simulate=True` argument which lets you
view the paths without actually creating any files. In this variant, all
aspects are stored each in individual folders but with identical
filenames:

::: caution
::: title
Caution
:::

The parameter `key` will be deprecated from version 0.6.0 onwards. See
`keys_and_ids`{.interpreted-text role="ref"}.
:::

``` python
>>> p = Parse('~/ms3/docs', key='pergo')
>>> p.parse_mscx()
>>> p.store_lists(  notes_folder='./notes',
                    rests_folder='./rests',
                    notes_and_rests_folder='./notes_and_rests',
                    simulate=True
                    )
['~/ms3/docs/notes/cujus.tsv',
 '~/ms3/docs/rests/cujus.tsv',
 '~/ms3/docs/notes_and_rests/cujus.tsv',
 '~/ms3/docs/notes/o_quam.tsv',
 '~/ms3/docs/rests/o_quam.tsv',
 '~/ms3/docs/notes_and_rests/o_quam.tsv',
 '~/ms3/docs/notes/quae.tsv',
 '~/ms3/docs/rests/quae.tsv',
 '~/ms3/docs/notes_and_rests/quae.tsv',
 '~/ms3/docs/notes/stabat.tsv',
 '~/ms3/docs/rests/stabat.tsv',
 '~/ms3/docs/notes_and_rests/stabat.tsv']
```

In this variant, the different ways of specifying folders are
exemplified. To demonstrate all subtleties we parse the same four files
but this time from the perspective of `~/ms3`:

``` python
>>> p = Parse('~/ms3', folder_re='docs', key='pergo')
>>> p.parse_mscx()
>>> p.store_lists(  notes_folder='./notes',            # relative to ms3/docs
                    measures_folder='../measures',     # one level up from ms3/docs
                    rests_folder='rests',              # relative to the parsed directory
                    labels_folder='~/labels',          # absolute folder
                    expanded_folder='~/labels', expanded_suffix='_exp',
                    simulate = True
                    )
['~/ms3/docs/notes/cujus.tsv',
 '~/ms3/rests/docs/cujus.tsv',
 '~/ms3/measures/cujus.tsv',
 '~/labels/cujus.tsv',
 '~/labels/cujus_exp.tsv',
 '~/ms3/docs/notes/o_quam.tsv',
 '~/ms3/rests/docs/o_quam.tsv',
 '~/ms3/measures/o_quam.tsv',
 '~/labels/o_quam.tsv',
 '~/labels/o_quam_exp.tsv',
 '~/ms3/docs/notes/quae.tsv',
 '~/ms3/rests/docs/quae.tsv',
 '~/ms3/measures/quae.tsv',
 '~/labels/quae.tsv',
 '~/labels/quae_exp.tsv',
 '~/ms3/docs/notes/stabat.tsv',
 '~/ms3/rests/docs/stabat.tsv',
 '~/ms3/measures/stabat.tsv',
 '~/labels/stabat.tsv',
 '~/labels/stabat_exp.tsv']
```

+++

(column_names)=
## Column Names 

Glossary of the meaning and types of column types. In order to correctly
restore the types when loading TSV files, either use an
:py`~.annotations.Annotations`{.interpreted-text role="class"} object or
the function :py`~.utils.load_tsv`{.interpreted-text role="func"}.

### General Columns

#### **duration**

`fractions.Fraction`{.interpreted-text role="obj"}

Duration of an event expressed in fractions of a whole note. Note that
in note lists, the duration does not take into account if notes are
`tied <tied>`{.interpreted-text role="ref"} together; in other words,
the column expresses no durations that surpass the final bar line.

#### **duration_qb**

`float`{.interpreted-text role="obj"}

Duration expressed in quarter notes. If the column
`duration <duration>`{.interpreted-text role="ref"} is present it
corresponds to that column times four. Otherwise (e.g. for labels) it is
computed from an `~pandas.IntervalIndex`{.interpreted-text role="obj"}
created from the `quarterbeats <quarterbeats>`{.interpreted-text
role="ref"} column.

+++

(keysig)=
#### **keysig** Key Signatures 

`int`{.interpreted-text role="obj"}

The feature `keysig` represents the key signature of a particular
measure. It is an integer which, if positive, represents the number of
sharps, and if negative, the number of flats. E.g.: `3`: three sharps,
`-2`: two flats, `0`: no accidentals.

+++

(mc)=
#### **mc** Measure Counts 

`int`{.interpreted-text role="obj"}

Measure count, identifier for the measure units in the XML encoding.
Always starts with 1 for correspondence to MuseScore\'s status bar. For
more detailed information, please refer to `mc_vs_mn`{.interpreted-text
role="ref"}.

+++

(mn)=
#### **mn** Measure Numbers 

`int`{.interpreted-text role="obj"}

Measure number, continuous count of complete measures as used in printed
editions. Starts with 1 except for pieces beginning with a pickup
measure, numbered as 0. MNs are identical for first and second endings!
For more detailed information, please refer to
`mc_vs_mn`{.interpreted-text role="ref"}.

#### **mc_onset**

`fractions.Fraction`{.interpreted-text role="obj"}

The value for `mc_onset` represents, expressed as fraction of a whole
note, a position in a measure where `0` corresponds to the earliest
possible position (in most cases beat 1). For more detailed information,
please refer to `onsets`{.interpreted-text role="ref"}.

#### **mn_onset**

`fractions.Fraction`{.interpreted-text role="obj"}

The value for `mn_onset` represents, expressed as fraction of a whole
note, a position in a measure where `0` corresponds to the earliest
possible position of the corresponding measure number (MN). For more
detailed information, please refer to `onsets`{.interpreted-text
role="ref"}.

#### **quarterbeats**

`fractions.Fraction`{.interpreted-text role="obj"}

This column expresses positions, otherwise accessible only as a tuple
`(mc, mc_onset)`, as a running count of quarter notes from the piece\'s
beginning (quarterbeat = 0). If second endings are present in the score,
only the second ending is counted in order to give authentic values to
such a score, as if played without repetitions (third endings and more
are also ignored). If repetitions are unfolded, i.e. the table
corresponds to a full play-through of the score, all endings are taken
into account correctly.

For the specific case you need continuous quarterbeats including all
endings, please refer to
`quarterbeats_all_endings <quarterbeats_all_endings>`{.interpreted-text
role="ref"}.

Computation of quarterbeats requires an offset_dict that is computed
from the column `act_dur <act_dur>`{.interpreted-text role="ref"}
contained in every `measures`{.interpreted-text role="ref"} table.
Quarterbeats are based on the cumulative sum of that column, meaning
that they take the length of irregular measures into account.

#### **staff**

`int`{.interpreted-text role="obj"}

In which staff an event occurs. `1` = upper staff.

+++

(timesig)=
#### **timesig** Time Signatures 

`str`{.interpreted-text role="obj"}

The time signature `timesig` of a particular measure is expressed as a
string, e.g. `'2/2'`. The `actual duration <act_dur>`{.interpreted-text
role="ref"} of a measure can deviate from the time signature for
notational reasons: For example, a pickup bar could have an actual
duration of `1/4` but still be part of a `'3/8'` meter, which usually
has an actual duration of `3/8`.

#### **volta**

`int`{.interpreted-text role="obj"}

In the case of first and second (third etc.) endings, this column holds
the number of every \"bracket\", \"house\", or \_[volta](#volta), which
should increase from 1. This is required for MS3\'s unfold repeats
function to work. For more information,
`see here <voltas>`{.interpreted-text role="ref"}.

#### **voice**

`int`{.interpreted-text role="obj"}

In which notational layer an event occurs. Each
`staff`{.interpreted-text role="ref"} has (can have) up to four layers:

-   `1` = upper, default layer (blue)
-   `2` = second layer, downward stems (green)
-   `3` = third layer, upward stems (orange)
-   `4` = fourth layer, downward stems (purple)

### Measures

+++

(act_dur)=
#### **act_dur** Actual duration of a measure 

`fractions.Fraction`{.interpreted-text role="obj"}

The value of `act_dur` in most cases equals the time signature,
expressed as a fraction; meaning for example that a \"normal\" measure
in 6/8 has `act_dur = 3/4`. If the measure has an irregular length, for
example a pickup measure of length 1/8, would have `act_dur = 1/8`.

The value of `act_dur` plays an important part in inferring
`MNs <mn>`{.interpreted-text role="ref"} from
`MCs <mc>`{.interpreted-text role="ref"}. See also the columns
`dont_count <dont_count>`{.interpreted-text role="ref"} and
`numbering_offset <numbering_offset>`{.interpreted-text role="ref"}.

#### **barline**

`str`{.interpreted-text role="obj"}

The column `barline` encodes information about the measure\'s final bar
line.

#### **breaks**

`str`{.interpreted-text role="obj"}

The column `breaks` may include three different values:
`{'line', 'page', 'section'}` which represent the different breaks
types. In the case of section breaks, MuseScore

+++

(dont_count)=
#### **dont_count** Measures excluded from bar count 

`int`{.interpreted-text role="obj"}

This is a binary value that corresponds to MuseScore\'s setting
`Exclude from bar count` from the `Bar Properties` menu. The value is
`1` for pickup bars, second `MCs <mc>`{.interpreted-text role="ref"} of
divided `MNs <mn>`{.interpreted-text role="ref"} and some volta
measures, and `NaN` otherwise.

+++

(mc_offset)=
#### **mc_offset** Offset of a MC 

`fractions.Fraction`{.interpreted-text role="obj"}

The column `mc_offset` , in most cases, has the value `0` because it
expresses the deviation of this MC\'s `mc_onset`{.interpreted-text
role="ref"} `0` (beginning of the MC) from beat 1 of the corresponding
MN. If the value is a fraction \> 0, it means that this MC is part of a
MN which is composed of at least two MCs, and it expresses the current
MC\'s offset in terms of the duration of all (usually 1) preceding MCs
which are also part of the corresponding MN. In the standard case that
one MN would be split in two MCs, the first MC would have mc_offset =
`0` , and the second one mc_offset = `the previous MC's`
`act_dur <act_dur>`{.interpreted-text role="ref"} .

#### **next**

`tuple`{.interpreted-text role="obj"}

Every cell in this column has at least one integer, namely the MC of the
subsequent bar, or `-1` in the cast of the last. In the case of
repetitions, measures can have more than one subsequent MCs, in which
case the integers are separated by `', '` .

The column is used for checking whether
`irregular measure lengths <act_dur>`{.interpreted-text role="ref"} even
themselves out because otherwise the inferred MNs might be wrong. Also,
it is needed for MS3\'s unfold repetitions functionality.

+++

(numbering_offset)=
#### **numbering_offset** Offsetting MNs 

`int`{.interpreted-text role="obj"}

MuseScore\'s measure number counter can be reset at a given MC by using
the `Add to bar number` setting from the `Bar Properties` menu. If
`numbering_offset` ≠ 0, the counting offset is added to the current MN
and all subsequent MNs are inferred accordingly.

Scores which include several pieces (e.g. in variations or a suite),
sometimes, instead of using section `breaks <breaks>`{.interpreted-text
role="ref"}, use `numbering_offset` to simulate a restart for counting
`MNs <mn>`{.interpreted-text role="ref"} at every new section. This
leads to ambiguous MNs.

#### **quarterbeats_all_endings**

`fractions.Fraction`{.interpreted-text role="obj"}

Since the computation of `quarterbeats <quarterbeats>`{.interpreted-text
role="ref"} for pieces including alternative endings
(`voltas <volta>`{.interpreted-text role="ref"}) excludes all but the
second endings, the measures of such pieces get this additional column,
allowing to create an offset_dict for users who need continuous
quarterbeats including all endings. In that case one would call

``` python
from ms3 import add_quarterbeats_col
offset_dict = measures.quarterbeats_all_endings.to_dict()
df_with_gapless_quarterbeats = add_quarterbeats_col(df, offset_dict)
```

#### **repeats**

`str`{.interpreted-text role="obj"}

The column `repeats` indicates the presence of repeat signs and can have
the values
`{'start', 'end', 'startend', 'firstMeasure', 'lastMeasure'}`. MS3
performs a test on the repeat signs\' plausibility and throws warnings
when some inference is required for this.

The `repeats` column needs to have the correct repeat sign structure in
order to have a correct `next <next>`{.interpreted-text role="ref"}
column which, in return, is required for MS3\'s unfolding repetitions
functionality.

### Notes and Rests

#### **chord_id**

`int`{.interpreted-text role="obj"}

Every note keeps the ID of the `<Chord>` tag to which it belongs in the
score. This is necessary because in MuseScore XML, most markup (e.g.
articulation, lyrics etc.) are attached to
`chords <chords>`{.interpreted-text role="ref"} rather than to
individual notes. This column allows for relating markup to notes at a
later point.

#### **gracenote**

`str`{.interpreted-text role="obj"}

For grace notes, type of the grace note as encoded in the MuseScore
source code. They are assigned a `duration <duration>`{.interpreted-text
role="ref"} of 0.

+++

(midi)=
#### **midi** Piano key 

`int`{.interpreted-text role="obj"}

MIDI pitch with `60` = C4, `61` = C#4/Db4/B##3 etc. For the actual note
name, refer to the `tpc <tpc>`{.interpreted-text role="ref"} column.

#### **nominal_duration**

`fractions.Fraction`{.interpreted-text role="obj"}

Note\'s or rest\'s duration without taking into account dots or tuplets.
Multiplying by `scalar <scalar>`{.interpreted-text role="ref"} results
in the actual `duration <duration>`{.interpreted-text role="ref"}.

#### **scalar**

`fractions.Fraction`{.interpreted-text role="obj"}

Value reflecting dots and tuples by which to multiply a note\'s or
rest\'s `nominal_duration <nominal_duration>`{.interpreted-text
role="ref"}.

#### **tied**

`int`{.interpreted-text role="obj"}

Encodes ties on the note\'s left (`-1`), on its right (`1`) or both
(`0`). A tie merges a note with an adjacent one having the same pitch.

  --------------------------------------------------------------------------
  value    explanation
  -------- -----------------------------------------------------------------
  \<NA\>   No ties. This note represents an onset and ends after the given
           duration.

  1        This note is tied to the next one. It represents an onset but not
           a note ending.

  0        This note is being tied to and tied to the next one. It
           represents neither an onset nor a note ending.

  -1       This note is being tied to. That is, it does not represent an
           onset, instead it adds to the duration of a previous note on the
           same pitch and ends it.
  --------------------------------------------------------------------------

+++

(tpc)=
#### **tpc** Tonal pitch class 

`int`{.interpreted-text role="obj"}

Encodes note names by their position on the line of fifth with `0` = C,
`1` = G, `2` = D, `-1` = F, `-2` = Bb etc. The octave is defined by
`midi <midi>`{.interpreted-text role="ref"} DIV 12 - 1

#### **tremolo**

`str`{.interpreted-text role="obj"}

The syntax for this column is `<dur>_<type>_<component>` where `<dur>`
is half the duration of the tremolo, `<type>` is the tremolo type, e.g.
`c32` for 3 beams or `c64` for 4 (values taken from the source code),
and `<component>` is 1 for notes in the first and 2 for notes in the
second \<Chord\>.

Explanation: MuseScore 3 encodes the two components of a tremolo as two
separate \<Chord\> tags with half the duration of the tremolo. This
column serves to keep the information of the two components although
onsets and durations in the `notes`{.interpreted-text role="ref"} are
corrected to represent the fact that all notes are sounding through the
duration of the tremolo.

For example, an octave tremolo with duration of a dotted half note and
tremolo frequency of 32nd notes will appear in the score as a dotted
half on beat 1 and another dotted half 3 eights later. In the note list,
however, both notes have `mc_onset` 0 and `duration` 3/4. The column
`tremolo` has the value `3/8_c32_1` for the first note and `3/8_c32_1`
for the second.

### Chords

The various \<Chord\> tags are identified by increasing integer counts
in the column `chord_id`. Within a note list, a
`column of the same name <chord_id>`{.interpreted-text role="ref"}
specifies which note belongs to which \<Chord\> tag. A chord and all the
notes belonging to it have identical values in the columns
`mc <mc>`{.interpreted-text role="ref"}, `mn <mn>`{.interpreted-text
role="ref"}, `mc_onset <mc_onset>`{.interpreted-text role="ref"},
`mn_onset <mn_onset>`{.interpreted-text role="ref"},
`timesig <timesig>`{.interpreted-text role="ref"},
`staff <staff>`{.interpreted-text role="ref"},
`voice <voice>`{.interpreted-text role="ref"},
`duration <duration>`{.interpreted-text role="ref"},
`gracenote <gracenote>`{.interpreted-text role="ref"},
`nominal_duration <nominal_duration>`{.interpreted-text role="ref"},
`scalar <scalar>`{.interpreted-text role="ref"},
`volta <volta>`{.interpreted-text role="ref"}, and of course
`chord_id <chord_id>`{.interpreted-text role="ref"}.

#### **articulation**

`str`{.interpreted-text role="obj"}

Articulation signs named as in the MuseScore file, e.g.
`articStaccatoBelow`.

#### **dynamics**

`str`{.interpreted-text role="obj"}

Dynamic signs such as `p`, `ff` etc. Other dynamic markings such as
`dolce` are currently displayed as `other-dynamics`. Velocity values are
currently not extracted. These features can easily be implemented [upon
request](https://github.com/johentsch/ms3/issues/).

+++

(lyrics_1)=
#### **lyrics:1** 

`str`{.interpreted-text role="obj"}

When a voice includes only a single verse, all syllables are contained
in the column `lyrics:1`. If it has more than one verse, for each
\<Chord\> the last verse\'s syllable is contained in the respective
column, e.g. `lyrics:3` if the 3rd verse is the last one with a syllable
for this chord. Each syllable has a trailing `-` if it\'s the first
syllable of a word, a leading `-` if it\'s the last syllable of a word,
and both if it\'s in the middle of a word.

+++

(qpm)=
#### **qpm** Quarter notes per minute 

`int`{.interpreted-text role="obj"}

Defined for every `tempo <tempo>`{.interpreted-text role="ref"} mark.
Normalizes the metronome value to quarter notes. For example, `𝅘𝅥. = 112`
gets the value `qbm = 112 * 1.5 = 168`.

#### **staff_text**

`str`{.interpreted-text role="obj"}

Free-form text such as `dolce` or `div.`. Depending on the encoding
standard, this layer may include dynamics such as `cresc.`, articulation
such as `stacc.`, movement titles, and many more. Staff texts are added
in MuseScore via `[C] + T`.

#### **system_text**

Free-form text not attached to a particular staff but to the entire
system. This frequently includes movement names or playing styles such
as `Swing`. System texts are added in MuseScore via `[C] + [S] + T`.

#### **tempo**

Metronome markings and tempo texts. Unfortunately, for tempo texts that
include a metronome mark, e.g. `Larghetto. (𝅘𝅥 = 63)`, the text before
the 𝅘𝅥 symbol is lost. This can be fixed [upon
request](https://github.com/johentsch/ms3/issues/).

#### Spanners

`str`{.interpreted-text role="obj"} (-\> `tuple`{.interpreted-text
role="obj"})

Spanners designate markup that spans several \<Chord\> tags, such as
slurs, hairpins, pedal, trill and ottava lines. The values in a spanner
column are IDs such that all chords with the same ID belong to the same
spanner. Each cell can have more than one ID, separated by commas. For
evaluating spanner columns, the values should be turned into tuples.

Spanners span all chords belonging to the same
`staff <staff>`{.interpreted-text role="ref"}, except for slurs and
trills which span only chords in the same
`voice <voice>`{.interpreted-text role="ref"}. In other words, won\'t
find the ending of a slur that goes from one
`voice <voice>`{.interpreted-text role="ref"} to another.

##### **slur**

`str`{.interpreted-text role="obj"} (-\> `tuple`{.interpreted-text
role="obj"})

Slurs expressing legato and/or phrasing. These
`spanners <spanners>`{.interpreted-text role="ref"} always pertain to a
particular `voice <voice>`{.interpreted-text role="ref"}.

+++

(hairpins)=
##### **(de)crescendo_hairpin** 

`str`{.interpreted-text role="obj"} (-\> `tuple`{.interpreted-text
role="obj"})

`crescendo_hairpin` is a `<` `spanner <spanners>`{.interpreted-text
role="ref"}, `decrescendo_hairpin` a `>`
`spanner <spanners>`{.interpreted-text role="ref"}. These always pertain
to an entire `staff <staff>`{.interpreted-text role="ref"}.

+++

(cresc_lines)=
##### **crescendo_line**, **diminuendo_line** 

`str`{.interpreted-text role="obj"} (-\> `tuple`{.interpreted-text
role="obj"})

These are `spanners <spanners>`{.interpreted-text role="ref"} starting
with a word, by default `cresc.` or `dim.`, followed by a dotted line.
These always pertain to an entire `staff <staff>`{.interpreted-text
role="ref"}.

##### **Ottava**

`str`{.interpreted-text role="obj"} (-\> `tuple`{.interpreted-text
role="obj"})

These `spanners <spanners>`{.interpreted-text role="ref"} are always
specified with a subtype such as `Ottava:8va` or `Ottava:15mb`. They
always pertain to an entire `staff <staff>`{.interpreted-text
role="ref"}

##### **pedal**

`str`{.interpreted-text role="obj"} (-\> `tuple`{.interpreted-text
role="obj"})

Pedal line `spanners <spanners>`{.interpreted-text role="ref"} always
pertain to an entire `staff <staff>`{.interpreted-text role="ref"}.

#### **TextLine**

`str`{.interpreted-text role="obj"} (-\> `tuple`{.interpreted-text
role="obj"})

Custom staff text with a line that can be prolonged at will.

#### **Trill**

`str`{.interpreted-text role="obj"}

Trills `spanners <spanners>`{.interpreted-text role="ref"} can have
different subtypes specified after a colon, e.g. `'Trill:trill'`. They
always pertain to a particular `voice <voice>`{.interpreted-text
role="ref"}.

### Labels

#### **harmony_layer**

`int`{.interpreted-text role="obj"}

This column indicates the harmony layer, or label type, in/as which a
label has been stored. It is an integer within \[0, 3\] that indicates
how it is encoded in MuseScore.

  -------------------------------------------------------------------------------
  harmony_layer   explanation
  --------------- ---------------------------------------------------------------
  0               Label encoded in MuseScore\'s chord layer (Add-\>Text-\>Chord
                  Symbol, or \[C\]+K) that does not start with a note name, i.e.
                  MuseScore did not recognize it as an absolute chord and encoded
                  it as plain text (compare type 3).

  1               Roman Numeral (Add-\>Text-\>Roman Numeral Analysis).

  2               Nashville number (Add-\>Text-\>Nashville Number).

  3               Label encoded in MuseScore\'s chord layer (Add-\>Text-\>Chord
                  Symbol, or \[C\]+K) that does start with a note name, i.e.
                  MuseScore did recognize it as an absolute chord and encoded its
                  root (and bass note) as numerical values.
  -------------------------------------------------------------------------------

#### **label**

`str`{.interpreted-text role="obj"}

Annotation labels from MuseScores \<Harmony\> tags. Depending on the
`label_type <label_type>`{.interpreted-text role="ref"} the column can
include complete strings (decoded) or partial strings (encoded).

#### **regex_match**

`str`{.interpreted-text role="obj"}

Name of the first regular expression that matched a label, e.g.
\'dcml\'.

#### **label_type**

::: warning
::: title
Warning
:::

Deprecated since 0.6.0 where this column has been split and replaced by
`harmony_layer <harmony_layer>`{.interpreted-text role="ref"} and
`regex_match <regex_match>`{.interpreted-text role="ref"}
:::

`str`{.interpreted-text role="obj"}

See `label types <label_types>`{.interpreted-text role="ref"} above.

+++

(offset)=
#### **offset_x** and **offset_y** 

`float`{.interpreted-text role="obj"}

Offset positions for labels whose position has been manually altered. Of
importance mainly for re-inserting labels into a score at the exact same
position.

### Expanded

::: {#alt_label}
`str`{.interpreted-text role="obj"}
:::

Alternative reading to the `label <label>`{.interpreted-text
role="ref"}. Generally considered \"second choice\" compared to the
\"main label\" that has been expanded.

#### **bass_note**

`int`{.interpreted-text role="obj"}

The bass note designated by the label, expressed as
`scale degree <fifths>`{.interpreted-text role="ref"}.

#### **cadence**

`str`{.interpreted-text role="obj"}

Currently allows for the values

  -----------------------------
  value   cadence
  ------- ---------------------
  PAC     perfect authentic

  IAC     imperfect authentic

  HC      half

  DC      deceptive

  EC      evaded

  PC      plagal
  -----------------------------

#### **chord**

`str`{.interpreted-text role="obj"}

This column stands in no relation to the \<Chord\> tags
`discussed above <chords>`{.interpreted-text role="ref"}. Instead, it
holds the substring of the original labels that includes only the actual
chord label, i.e. excluding information about modulations, pedal tones,
phrases, and cadences. In other words, it comprises the features
`numeral <numeral>`{.interpreted-text role="ref"},
`form <form>`{.interpreted-text role="ref"},
`figbass <figbass>`{.interpreted-text role="ref"},
`changes <changes>`{.interpreted-text role="ref"}, and
`relativeroot <relativeroot>`{.interpreted-text role="ref"}.

+++

(chord_tones)=
#### **chord_tones**, **added_tones** 

`str`{.interpreted-text role="obj"} (-\> `tuple`{.interpreted-text
role="obj"})

Chord tones designated by the label, expressed as
`scale degrees <fifths>`{.interpreted-text role="ref"}. Includes 3 scale
degrees for triads, 4 for tetrads, ordered according to the inversion
(i.e. the first value is the `bass_note <bass_note>`{.interpreted-text
role="ref"}). Accounts for chord tone replacement expressed through
intervals \<= 8 within parentheses, without leading +. `added_tones`
reflects only those non-chord tones that were added using, again within
parentheses, intervals preceded by + or/and greater than 8.

#### **chord_type**

`str`{.interpreted-text role="obj"}

A summary of information that otherwise depends on the three columns
`numeral <numeral>`{.interpreted-text role="ref"},
`form <form>`{.interpreted-text role="ref"},
`figbass <figbass>`{.interpreted-text role="ref"}. It can be one of the
wide-spread abbreviations for triads: `M, m, o, +` or for seventh
chords: `o7, %7, +7, +M7` (for diminished, half-diminished and augmented
chords with minor/major seventh), or `Mm7, mm7, MM7, mM7` for all
combinations of a major/minor triad with a minor/major seventh.

+++

(figbass)=
#### **figbass** Inversion 

Figured bass notation of the chord inversion. For triads, this feature
can be `<NA>, '6', '64'`, for seventh chords `'7', '65', '43', '2'`.
This column plays into computing the
`chord_type <chord_type>`{.interpreted-text role="ref"}. This feature is
decisive for
`which chord tone is in the bass <bass_note>`{.interpreted-text
role="ref"}.

#### **form**

`str`{.interpreted-text role="obj"}

This column conveys part of the information what
`chord_type <chord_type>`{.interpreted-text role="ref"} a label
expresses.

  -------------------------------------------------------------------------
  value    chord type
  -------- ----------------------------------------------------------------
  \<NA\>   If `figbass <figbass>`{.interpreted-text role="ref"} is one of
           `<NA>, '6', '64'`, the chord is either a major or minor triad.
           Otherwise, it is either a major or a minor chord with a minor
           seventh.

  o, +     Diminished or augmented chord. Again, it depends on
           `figbass <figbass>`{.interpreted-text role="ref"} whether it is
           a triad or a seventh chord.

  %, M, +M Half diminished or major seventh chord. For the latter, the
           chord form (MM7 or mM7) depends on the
           `numeral <numeral>`{.interpreted-text role="ref"}.
  -------------------------------------------------------------------------

#### **globalkey**

`str`{.interpreted-text role="obj"}

Tonality of the piece, expressed as absolute note name, e.g. `Ab` for A
flat major, or `g#` for G sharp minor.

#### **globalkey_is_minor**

`bool`{.interpreted-text role="obj"}

Auxiliary column which is True if the
`globalkey <globalkey>`{.interpreted-text role="ref"} is a minor key,
False otherwise.

#### **localkey**

`str`{.interpreted-text role="obj"}

Local key expressed as Roman numeral relative to the
`globalkey <globalkey>`{.interpreted-text role="ref"}, e.g. `IV` for the
major key on the 4th scale degree or `#iv` for the minor scale on the
raised 4th scale degree.

#### **localkey_is_minor**

`bool`{.interpreted-text role="obj"}

Auxiliary column which is True if the
`localkey <localkey>`{.interpreted-text role="ref"} is a minor key,
False otherwise.

#### **numeral**

`str`{.interpreted-text role="obj"}

Roman numeral defining the chordal root relative to the local key. An
uppercase numeral stands for a major chordal third, lowercase for a
minor third. The column `root <root>`{.interpreted-text role="ref"}
expresses the same information as
`scale degree <fifths>`{.interpreted-text role="ref"}.

+++

(phraseend)=
#### **phraseend** Phrase annotations 

In versions \< 2.2.0, only phrase endings where annotated, designated by
`\\`. From version 2.2.0 onwards, `{` means beginning and `}` ending of
a phrase. Everything between `}` and the subsequent `{` is to be
considered as part of the previous phrase, a \'codetta\' after the
strong end point.

+++

(relativeroot)=
#### **relativeroot** Tonicized key 

`str`{.interpreted-text role="obj"}

This feature designates a lower-level key to which the current chord
relates. It is expressed relative to the local key. For example, if the
current `numeral <numeral>`{.interpreted-text role="ref"} is a `V` and
it is a secondary dominant, relativeroot is the Roman numeral of the key
that is being tonicized.

#### **root**

`int`{.interpreted-text role="obj"}

The `numeral <numeral>`{.interpreted-text role="ref"} expressed as
`scale degree <fifths>`{.interpreted-text role="ref"}.

### Metadata

If not otherwise specified, metadata fields are of type
`str`{.interpreted-text role="obj"}.

#### **fname**

`str`{.interpreted-text role="obj"}

::: admonition
Metadata category

File information about the score described by this set of metadata.
:::

File name without extension. Serves as ID for linking files that belong
to the same piece although they might have different suffixes and file
extensions. It follows that only files will be detected as belonging to
this score whose file names are at least as long. In other words, the
main score file that is to be considered as the most up-to-date version
of the data should ideally not come with a suffix.

#### **rel_path**

`str`{.interpreted-text role="obj"}

::: admonition
Metadata category

File information about the score described by this set of metadata.
:::

Relative file path of the score, including extension.

Metadata extracted with older versions of ms3 (\<1.0.0) would come
instead with the column `rel_paths` which would include the relative
folder path without the file itself. This value can now be found in the
column `subdirectory`{.interpreted-text role="ref"}.

#### **subdirectory**

`str`{.interpreted-text role="obj"}

::: admonition
Metadata category

File information about the score described by this set of metadata.
:::

Folder where the score is located, relative to the corpus_path.
Equivalent to `rel_path`{.interpreted-text role="ref"} but without the
file.

#### **composer**

`str`{.interpreted-text role="obj"}

::: admonition
Metadata category

Default metadata field in MuseScore\'s *Score Properties*. Can be
updated using the command `ms3 metadata`.
:::

Composer name as it would figure in the English Wikipedia (although
middle names may be dropped).

+++

(workTitle)=
#### **workTitle** 

`str`{.interpreted-text role="obj"}

::: admonition
Metadata category

Default metadata field in MuseScore\'s *Score Properties*. Can be
updated using the command `ms3 metadata`.
:::

Title of the whole composition (cycle), even if the score holds only a
part of it. It should not contain opus or other catalogue numbers, which
go into the `workNumber`{.interpreted-text role="ref"} column/field.

The title of the part included in this score, be it a movement or, for
instance, a song within a song cycle, goes into the
`movementTitle`{.interpreted-text role="ref"} column/field.

+++

(workNumber)=
#### **workNumber** 

::: admonition
Metadata category

Default metadata field in MuseScore\'s *Score Properties*. Can be
updated using the command `ms3 metadata`.
:::

`str`{.interpreted-text role="obj"}

Catalogue number(s), e.g. `op. 30a`.

+++

(movementNumber)=
#### **movementNumber** 

`str`{.interpreted-text role="obj"}

::: admonition
Metadata category

Default metadata field in MuseScore\'s *Score Properties*. Can be
updated using the command `ms3 metadata`.
:::

If applicable, the sequential number of the movement or part of a cycle
contained in this score. In other words, the string should probably be
interpretable as a number; a second movement should have the value `2`,
not `II`.

+++

(movementTitle)=
#### **movementTitle** 

`str`{.interpreted-text role="obj"}

::: admonition
Metadata category

Default metadata field in MuseScore\'s *Score Properties*. Can be
updated using the command `ms3 metadata`.
:::

If applicable, the name of the movement or part of a cycle contained in
this score.

#### **source**

`str`{.interpreted-text role="obj"}

::: admonition
Metadata category

Default metadata field in MuseScore\'s *Score Properties*. Can be
updated using the command `ms3 metadata`.
:::

If applicable, the URL to the online score that this file has been
derived from.

#### **typesetter**

`str`{.interpreted-text role="obj"}

::: admonition
Metadata category

Custom metadata field in MuseScore\'s *Score Properties*. Can be updated
using the command `ms3 metadata`.
:::

Name or user profile URL of the person who first engraved this score.

#### **annotators**

`str`{.interpreted-text role="obj"}

::: admonition
Metadata category

Custom metadata field in MuseScore\'s *Score Properties*. Can be updated
using the command `ms3 metadata`.
:::

Creator(s) of the chord, phrase, cadence, and/or form labels pertaining
to the [DCML harmony annotation
standard](https://github.com/DCMLab/standards).

#### **reviewers**

`str`{.interpreted-text role="obj"}

::: admonition
Metadata category

Custom metadata field in MuseScore\'s *Score Properties*. Can be updated
using the command `ms3 metadata`.
:::

Reviewer(s) of the chord, phrase, cadence, and/or form labels pertaining
to the [DCML harmony annotation
standard](https://github.com/DCMLab/standards).

#### **wikidata**

`str`{.interpreted-text role="obj"}

::: admonition
Metadata category

Custom metadata field in MuseScore\'s *Score Properties*. Can be updated
using the command `ms3 metadata`.
:::

URL of the WikiData item describing the piece that this score
represents.

#### **viaf**

`str`{.interpreted-text role="obj"}

::: admonition
Metadata category

Custom metadata field in MuseScore\'s *Score Properties*. Can be updated
using the command `ms3 metadata`.
:::

URL of the Virtual International Authority File (VIAF) entry identifying
the piece that this score represents.

#### **musicbrainz**

`str`{.interpreted-text role="obj"}

::: admonition
Metadata category

Custom metadata field in MuseScore\'s *Score Properties*. Can be updated
using the command `ms3 metadata`.
:::

MusicBrainz URI identifying the piece that this score represents.

#### **imslp**

`str`{.interpreted-text role="obj"}

::: admonition
Metadata category

Custom metadata field in MuseScore\'s *Score Properties*. Can be updated
using the command `ms3 metadata`.
:::

URL to the wiki page within the International Music Score Library
Project (IMSLP) that identifies this score.

#### **composed_start**

`str`{.interpreted-text role="obj"} of length 4 or `..`

::: admonition
Metadata category

Custom metadata field in MuseScore\'s *Score Properties*. Can be updated
using the command `ms3 metadata`.
:::

Year in which the composing began. If there is evidence that composing
the piece took more than one year but only the
`composed_end`{.interpreted-text role="ref"} of the time span is known,
this value should be `..`. In all other cases the string should be
composed of four integers so that it can be converted to a number.

Collecting `(composed_start, composed_end)` year values was a conscious
decision against more elaborate indications such as the [Extended
Date/Time Format (EDTF)](https://www.loc.gov/standards/datetime/), based
on a trade-off.

#### **composed_end**

`str`{.interpreted-text role="obj"} of length 4 or `..`

::: admonition
Metadata category

Custom metadata field in MuseScore\'s *Score Properties*. Can be updated
using the command `ms3 metadata`.
:::

Year in which the composition was finished, or in which it was published
for the first time. If there is evidence that composing the piece took
more than one year but only the `composed_start`{.interpreted-text
role="ref"} of the time span is known, this value should be `..`. In all
other cases the string should be composed of four integers so that it
can be converted to a number.

Collecting `(composed_start, composed_end)` year values was a conscious
decision against more elaborate indications such as the [Extended
Date/Time Format (EDTF)](https://www.loc.gov/standards/datetime/), based
on a trade-off.

#### **last_mn**

`int`{.interpreted-text role="obj"}

::: admonition
Metadata category

Computed by ms3.
:::

Last `measure number <mn>`{.interpreted-text role="ref"} (i.e., the
length of the score as number of complete measures).

#### **last_mn_unfolded**

`int`{.interpreted-text role="obj"}

::: admonition
Metadata category

Computed by ms3.
:::

Number of measures when playing all repeats.

#### **length_qb**

`float`{.interpreted-text role="obj"}

::: admonition
Metadata category

Computed by ms3.
:::

Length of the piece, measured in quarter notes.

#### **length_qb_unfolded**

`float`{.interpreted-text role="obj"}

::: admonition
Metadata category

Computed by ms3.
:::

Length of the piece when playing all repeats, measured in quarter notes.

#### **volta_mcs**

`tuple`{.interpreted-text role="obj"}

::: admonition
Metadata category

Computed by ms3.
:::

`Measure counts <mc>`{.interpreted-text role="ref"} of first and second
(and further) endings. For example,
`(((16,), (17,)), ((75, 76), (77, 78)))` would stand for two sets of two
brackets, the first one with two endings of length 1 (probably measure
numbers 16a and 16b) and the second one for two endings of length 2,
starting in MC 75.

The name comes from Italian \"prima/seconda volta\" for \"first/second
time\".

#### **all_notes_qb**

`float`{.interpreted-text role="obj"}

::: admonition
Metadata category

Computed by ms3.
:::

Summed up duration of all notes, measured in quarter notes.

#### **n_onsets**

`int`{.interpreted-text role="obj"}

::: admonition
Metadata category

Computed by ms3.
:::

Number of all note onsets. This number is at most the number of rows in
the corresponding notes table which, in return, is the number of all
note *heads*. `n_onsets` does not count tied-to note heads (which do not
represent onsets).

#### **n_onset_positions**

`int`{.interpreted-text role="obj"}

::: admonition
Metadata category

Computed by ms3.
:::

Number of unique note onsets (\"slices\").

#### **guitar_chord_count**

`int`{.interpreted-text role="obj"}

::: admonition
Metadata category

Computed by ms3.
:::

Number of all \<Harmony\> labels that do not match the [DCML harmony
annotation standard](https://github.com/DCMLab/standards). In most
cases, they will be so-called guitar or Jazz chords (\"changes\") as
used in lead sheets, pop and folk songs, etc.

#### **label_count**

`int`{.interpreted-text role="obj"}

::: admonition
Metadata category

Computed by ms3.
:::

Number of chord labels that match the [DCML harmony annotation
standard](https://github.com/DCMLab/standards).

For metadata extracted with older versions of ms3 (\<1.0.0) this value
would represent the number of all \<Harmony\> labels including
`guitar/Jazz chords <guitar_chord_count>`{.interpreted-text role="ref"}.

+++

(key_signatures)=
#### **KeySig** Key signatures 

`str`{.interpreted-text role="obj"}

::: admonition
Metadata category

Computed by ms3.
:::

Key signature(s) (negative = flats, positive = sharps) and their
position(s) in the score. A score in C major would have the value
`1: 0`, i.e. zero accidentals in `MC <mc>`{.interpreted-text role="ref"}
1, the first \<Measure\> tag. A score with the key signatures of C minor
(3 flats), G minor (1 flat) and G major (1 sharp) could have, for
example, `1: -3, 39: -1, 67: 1`. In other words, the values are like
dictionaries without curly braces.

The column name is in CamelCase, other than the
`keysig`{.interpreted-text role="ref"} column found in
`measures`{.interpreted-text role="ref"} tables.m

+++

(time_signatures)=
#### **TimeSig** Time Signatures 

`str`{.interpreted-text role="obj"}

::: admonition
Metadata category

Computed by ms3.
:::

Time signature(s) and their position(s) in the score. A score entirely
in 4/4 would have the value `1: 4/4`, where 1 is the
`MC <mc>`{.interpreted-text role="ref"} of the first \<Measure\> tag. A
score with time signature changes could have, for example,
`1: 4/4, 39: 6/8, 67: 4/4`. In other words, the values are like
dictionaries without curly braces.

The column name is in CamelCase, other than the
`timesig`{.interpreted-text role="ref"} column found in
`measures`{.interpreted-text role="ref"} tables.

#### **musescore**

`str`{.interpreted-text role="obj"}

::: admonition
Metadata category

Computed by ms3.
:::

MuseScore version that has been used to save this score, e.g. `3.6.2`.
