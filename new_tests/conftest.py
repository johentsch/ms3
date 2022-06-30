import os
import pytest
from ms3 import Parse

# Directory holding your clones of DCMLab/unittest_metacorpus & DCMLab/pleyel_quartets
CORPUS_DIR = "~"


@pytest.fixture(
    scope="session",
    params=[
        "pleyel_quartets",
        "unittest_metacorpus",
    ],
    ids=[
        "single",
        "multiple",
    ],
)
def directory(request):
    """Compose the paths for the test corpora."""
    path = os.path.join(os.path.expanduser(CORPUS_DIR), request.param)
    return path

def make_path_cfg(directory, include_directory, include_paths, path_endswith=None):
    assert sum((include_directory, include_paths)) > 0, "At least one needs to be True"
    path_cfg = {}
    if include_directory:
        path_cfg['directory'] = directory
    if include_paths:
        paths = []
        for dirpath, subdirs, filenames in os.walk(directory):
            subdirs[:] = [sd for sd in subdirs if not sd.startswith('.')]
            for f in filenames:
                if path_endswith is None or f.endswith(path_endswith):
                    paths.append(os.path.join(dirpath, f))
        path_cfg['paths'] = paths
    return path_cfg

@pytest.fixture(
    scope="session",
    params=[
        (False, True, '.mscx'),
        (False, True, '.tsv'),
        (False, True),
        (True, False),
        (True, True)
    ],
    ids=[
        "mscx_paths",
        "tsv_paths",
        "all_paths",
        "directory",
        "directory+paths",
    ],
)
def path_cfg(request):
    return request.param

@pytest.fixture(
    scope="session",
)
def parse_cfg(directory, path_cfg):
    result = make_path_cfg(directory, *path_cfg)
    return result

@pytest.fixture(
    scope="session",
)
def parse_obj(parse_cfg):
    return Parse(**parse_cfg)








