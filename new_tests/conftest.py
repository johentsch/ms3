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
def small_corpora_path(request):
    """Compose the paths for the test corpora."""
    print("Path was requested")
    path = os.path.join(CORPUS_DIR, request.param)
    return path

@pytest.fixture(
    scope="session",
    params=[
        (Corpus, True, False),
        #        (Corpus, False, True),
        #        (Corpus, True, True),
    ],
    ids=[
        "TSV only",
        #        "scores only",
        #        "TSV + scores"
    ],
)
def parse_obj(all_corpora_path, request):
    path = all_corpora_path
    obj, tsv, scores = request.param
    initialized_obj = obj(directory=path, parse_tsv=tsv, parse_scores=scores)
    print(
        f"\nInitialized {type(initialized_obj).__name__}(directory='{path}', "
        f"parse_tsv={tsv}, parse_scores={scores})"
    )
    return initialized_obj