# for testing we use the example of annotations
test_file = """WARNING    ms3.Parse.old_tests.D973deutscher01.mscx.MeasureList
WARNING    ms3.Parse.old_tests.BWV_0815.mscx.MeasureList 
WARNING  MC_OFFSET 1, 40, 85, 97, 131, 139 ms3.Parse.old_tests.BWV_0815.mscx.MeasureList
WARNING  MC_OFFSET 1, 40 ms3.Parse.old_tests.BWV_0815.mscx.MeasureList"""

from logger import MessageType

ignored_warnings_dict = {}  # structure of dict: {"file1": [{"message_type": 1, "info": (20, 20)}]}
parsed_annotations = list(map(lambda elem: elem.split(), test_file.split(sep="\n")))  # from string of sentences to [[word]]
for message in parsed_annotations:
    message_type = MessageType[message[1]].value if len(message) > 2 else 0  # length of parsed message without type message is 2
    info = (message[2], " ".join(message[3:-1])) if len(message) > 2 else () # for messages without type we set info () and type 0
    if message[1] in ignored_warnings_dict.keys():
        ignored_warnings_dict[message[1]].append({"message_type": message_type, "info": info})
    else:
        ignored_warnings_dict[message[1]] = [{"message_type": message_type, "info": info}]
print(ignored_warnings_dict)

"""
output:
>>> {'ms3.Parse.old_tests.D973deutscher01.mscx.MeasureList': [{'message_type': 0, 'info': ()}], 
'ms3.Parse.old_tests.BWV_0815.mscx.MeasureList': [{'message_type': 0, 'info': ()}], 'MC_OFFSET': [{'message_type': 1, 
'info': ('1,', '40, 85, 97, 131, 139')}, {'message_type': 1, 'info': ('1,', '40')}]}
"""


