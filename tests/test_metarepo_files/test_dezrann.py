import json
import os
from collections import Counter
import pytest

from ms3.utils import get_value_profile_mask, load_tsv
from ms3.dezrann import generate_dez

@pytest.fixture(params=[
        'K279-1', 'K279-2', 'K279-3',
        'K280-1', 'K280-2', 'K280-3',
        'K283-1', 'K283-2', 'K283-3',
    ])
def movement(request) -> str:
    return request.param

def test_dcml2dez(mozart_piano_sonatas, movement):
    # first, create .dez file
    measures_path = os.path.join(mozart_piano_sonatas, 'measures', f"{movement}.tsv")
    harmonies_path = os.path.join(mozart_piano_sonatas, 'harmonies', f"{movement}.tsv")
    out_path = os.path.join(mozart_piano_sonatas, f"{movement}.dez")
    generate_dez(path_measures=measures_path,
                 path_labels=harmonies_path,
                 output_path=out_path,
                 cadences=True,
                 harmonies=4,
                 keys=5,
                 phrases=6
                 )
    # then, count the contained labels and compare with the target number (except if score contains voltas because then,
    # the .dez file might contain additional, repeated labels at the beginning of each ending).
    expanded = load_tsv(harmonies_path)
    if 'volta' in expanded and expanded.volta.notna().any():
        return
    with open(out_path, 'r', encoding='utf-8') as f:
        dezrann_file = json.load(f)
    type2column = {
        'Harmony': 'chord',
        'Cadence': 'cadence',
        'Phrase': 'phraseend',
        'Local Key': 'localkey',
    }
    written_labels = dict(Counter(type2column[label['type']] for label in dezrann_file['labels']))
    expected_counts = dict(
        chord = expanded['chord'].notna().sum(),
        cadence = expanded['cadence'].notna().sum(),
        phraseend = expanded['phraseend'].str.contains('{').sum(),
        localkey = get_value_profile_mask(expanded['localkey']).sum(),
    )
    assert written_labels == expected_counts