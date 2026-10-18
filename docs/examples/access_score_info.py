import os

from ms3 import Score

stabat_path = os.path.abspath(os.path.join("..", "stabat.mscx"))
s = Score(stabat_path)
print(s.mscx.labels())
