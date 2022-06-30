import pytest

TEST_NAMES = [
 'multiple-all_paths',
 'multiple-directory',
 'multiple-directory+paths',
 'multiple-mscx_paths',
 'multiple-tsv_paths',
 'single-all_paths',
 'single-directory',
 'single-directory+paths',
 'single-mscx_paths',
 'single-tsv_paths',
]

def node_name2cfg_name(node_name):
    """Takes 'test_function[cfg_name]' and returns 'cfg_name'."""
    start_pos = node_name.index("[") + 1
    cfg_name = node_name[start_pos:-1]
    return cfg_name

CFG_NAMES = []
@pytest.fixture()
def collect_cfg_name(request):
    """Helper for test_collect_cfg_names. Collects names in CFG_NAMES"""
    cfg_name = node_name2cfg_name(request.node.name)
    CFG_NAMES.append(cfg_name)

def test_collect_cfg_names(parse_obj, collect_cfg_name):
    """Run this if the parametrization of Parse has changed and you need to update TEST_NAMES."""
    print('\nTEST_NAMES = [')
    for name in sorted(CFG_NAMES):
        print(f" '{name}',")
    print("]")


################################## Actual tests ############################################

@pytest.fixture()
def expected_keys(request):
    expected = [
        {'mixed_files': {'.mscx': 8,                        # multiple-all_paths
                         '.mscz': 1,
                         '.musicxml': 1,
                         '.mxl': 1,
                         '.tsv': 1,
                         '.xml': 1},
         'ravel_piano': {'.mscx': 5, '.tsv': 14},
         'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
         'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
        {'mixed_files': {'.mscx': 8, '.mscz': 1, '.tsv': 1}, # multiple-directory
         'ravel_piano': {'.mscx': 5, '.tsv': 14},
         'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
         'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
        {'mixed_files': {'.mscx': 8,                        # multiple-directory+paths
                         '.mscz': 1,
                         '.musicxml': 1,
                         '.mxl': 1,
                         '.tsv': 1,
                         '.xml': 1},
         'ravel_piano': {'.mscx': 5, '.tsv': 14},
         'sweelinck_keyboard': {'.mscx': 1, '.tsv': 4},
         'wagner_overtures': {'.mscx': 2, '.tsv': 7}},
        {'mixed_files': {'.mscx': 8},                       # multiple-mscx_paths
         'ravel_piano': {'.mscx': 5},
         'sweelinck_keyboard': {'.mscx': 1},
         'wagner_overtures': {'.mscx': 2}},
        {'mixed_files': {'.tsv': 1},                        # multiple-tsv_paths
         'ravel_piano': {'.tsv': 14},
         'sweelinck_keyboard': {'.tsv': 4},
         'wagner_overtures': {'.tsv': 7}},
        {'pleyel_quartets': {'.mscx': 6, '.tsv': 13}},      # single-all_paths
        {'pleyel_quartets': {'.mscx': 6, '.tsv': 13}},      # single-directory
        {'pleyel_quartets': {'.mscx': 6, '.tsv': 13}},      # single-directory+paths
        {'pleyel_quartets': {'.mscx': 6}},                  # single-mscx_paths
        {'pleyel_quartets': {'.tsv': 13}},                  # single-tsv_paths
    ]
    name2expected = dict(zip(TEST_NAMES, expected))
    name = node_name2cfg_name(request.node.name)
    return name2expected[name]

def test_parse_keys(parse_obj, expected_keys):
    assert parse_obj.count_extensions(per_key=True) == expected_keys

