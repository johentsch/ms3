import subprocess

import os

def test_examples():
    docs_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "docs", "examples")
    for script in os.listdir(docs_folder):
        path = os.path.join(docs_folder, script)
        subprocess.run(["python", path])