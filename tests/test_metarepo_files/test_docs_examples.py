import os
import subprocess

import pytest

from tests.conftest import DOCS_EXAMPLES_DIR


@pytest.fixture(params=os.listdir(DOCS_EXAMPLES_DIR))
def example_script(request):
    return os.path.join(DOCS_EXAMPLES_DIR, request.param)


def test_examples(example_script):
    print(f"Running {example_script} ...")
    exit_value = subprocess.run(["python", example_script])
    exit_value.check_returncode()
