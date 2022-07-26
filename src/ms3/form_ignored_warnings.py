import re
import os
import json
from pathlib import Path


home = str(Path.home())
file_path = 'unittest_metacorpus/mixed_files'
with open(os.path.join(home, file_path, 'IGNORED_WARNINGS')) as f:
    file = f.read()

ignored_warnings_dict = {}  # structure of dict: {"file1": [(2, 20), (1, 32)]}
messages = file.split(sep="\n")  # split messages
parsed_annotations = list(map(lambda k: (list(filter(None, re.split("[(, :)]+", k)))), messages))
for message in parsed_annotations:
    if message[1] == "0":  # there is no type of message
        info = (0, )
    else:
        info = (int(message[1]), *list(map(int, message[2:-1])))  # get all info
    if message[-1] in ignored_warnings_dict.keys():
        ignored_warnings_dict[message[-1]].append(info)
    else:
        ignored_warnings_dict[message[-1]] = [info]

print(ignored_warnings_dict)

with open(os.path.join(home, file_path, 'IGNORED_WARNINGS.json'), "w") as f:
    json.dump(ignored_warnings_dict, f)

"""
output:
>>> {'ms3.Parse.mixed_files.Did03M-Son_regina-1762-Sarti.mscx.MeasureList': [(2, 94)], 
'ms3.Parse.MS3.BWV_0815.mscx.MeasureList': [(1, 1, 40, 85, 97, 131, 139)], 
"ms3.Parse.ravel_piano.Ravel_-_Miroirs_III_Une_Barque_sur_l'ocean.mscx.MeasureList": [(3, 52)]}
"""


