import os

from ms3.dezrann import generate_dez

MOZART_MOVEMENTS = [
        'K279-1', 'K279-2', 'K279-3',
        'K280-1', 'K280-2', 'K280-3',
        'K283-1', 'K283-2', 'K283-3',
    ]

def test_dcml2dez(mozart_piano_sonatas):
    for fname in MOZART_MOVEMENTS:
        measures_path = os.path.join(mozart_piano_sonatas, 'measures', f"{fname}.tsv")
        harmonies_path = os.path.join(mozart_piano_sonatas, 'harmonies', f"{fname}.tsv")
        out_path = os.path.join(mozart_piano_sonatas, f"{fname}.dez")
        generate_dez(path_measures=measures_path,
                     path_labels=harmonies_path,
                     output_path=out_path)