import os

from ms3 import Score

stabat_path = os.path.abspath(os.path.join("..", "docs", "stabat.mscx"))
s = Score(stabat_path)
print(s)
