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

# Quick `ms3` reference

## To run this notebook

* install ms3 (`pip install ms3`)
* set the `DATA_PATH` to where you want the folder `dcml_corpora` to be created that contains the data

```{code-cell} ipython3
:tags: []

DATA_PATH = '~'
```

## Setup

```{code-cell} ipython3
import os
import ms3
from git import Repo

corpora_path = os.path.join(os.path.expanduser(DATA_PATH), 'dcml_corpora')
if os.path.isdir(corpora_path):
    repo = Repo(corpora_path)
else:
    repo = Repo.clone_from(url='https://github.com/DCMLab/dcml_corpora.git', 
                to_path=corpora_path, 
                multi_options=['--recurse-submodules', '--shallow-submodules'])
print(f"dcml_corpora @ commit {repo.commit().hexsha}")
```

## Parsing several corpora at once

**We initialize a [Parse](Parse) object which will discover all files that can potentially be parsed:**

```{code-cell} ipython3
corpora = ms3.Parse(corpora_path)
corpora
```

**From here we can use the methods**

* [parse_scores()](Parse.parse_scores()) to parse all detected scores,
* [parse_tsv()](Parse.parse_tsv()) to parse all detected TSV files (previously extracted from scores),
* [parse()](Parse.parse()) to parse everything.

```{code-cell} ipython3
corpora.parse_scores()
corpora
```

**Now we can extract the facets we need from the parsed scores, e.g. all note heads from all scores:**

```{code-cell} ipython3
corpora.get_facets('notes')
```

**Or we iterate through the corpora and print information on the first 10 measures:**

```{code-cell} ipython3
for corpus_name, corpus_object in corpora:
    print(f"First ten measures of {corpus_name}:")
    display(corpus_object.get_facet('measures'))
```

**The available facets are `'measures', 'notes', 'rests', 'notes_and_rests', 'labels', 'expanded', 'form_labels', 'cadences', 'events', 'chords'`.
We can request several at the same time:**

```{code-cell} ipython3
corpora.get_facets(['labels', 'chords'])
```

```{code-cell} ipython3

```
