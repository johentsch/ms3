from ms3 import Score
import os
stabat_path = os.path.abspath(os.path.join("..", "..", "docs", "stabat.mscx"))
s = Score(stabat_path)
print(s.mscx.labels())
