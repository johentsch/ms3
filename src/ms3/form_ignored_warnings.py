# for testing we use the example of annotations
test_file = """WARNING  MC_OFFSET (1, 1, 40, 85, 97, 131, 139) ms3.Parse.old_tests.BWV_0815.mscx.MeasureList 
WARNING  NON_EXPECTED_MN (2, 138) ms3.Parse.old_tests.BWV_0815.mscx.MeasureList
WARNING   (0, ) ms3.Parse.old_tests.BWV_0815.mscx.MeasureList
WARNING   (0, ) ms3.Parse.old_tests.BWV_0816.mscx
WARNING   (0, ) ms3.Parse.old_tests.BWV_0815.mscx.MeasureList"""

import re

ignored_warnings_dict = {}  # structure of dict: {"file1": [(2, 20), (1, 32)]}
messages = test_file.split(sep="\n")  # split messages
parsed_annotations = list(map(lambda k: (list(filter(None, re.split("[(, :)]+", k)))), test_file.split(sep="\n")))
for message in parsed_annotations:
    if message[1] == "0":  # there is no type of message
        info = (0, )
    else:
        info = (int(message[2]), list(map(int, message[3:-1])))  # get all info
    if message[-1] in ignored_warnings_dict.keys():
        ignored_warnings_dict[message[-1]].append(info)
    else:
        ignored_warnings_dict[message[-1]] = [info]
print(ignored_warnings_dict)

"""
output:
>>> {'ms3.Parse.old_tests.BWV_0815.mscx.MeasureList': [(1, [1, 40, 85, 97, 131, 139]), (2, [138]), (0,), (0,)], 
'ms3.Parse.old_tests.BWV_0816.mscx': [(0,)]}
"""


