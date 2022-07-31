import re
import os
import json
from pathlib import Path


class MessageParser(object):
    """
    Gets info from logger messages that have to be ignored and writes it to json file.
    The expected structure of message: warning_type (warning_type_id, label) file
    Example of message: INCORRECT_VOLTA_MN_WARNING (2, 94) ms3.Parse.mixed_files.Did03M-Son_regina-1762-Sarti.mscx.MeasureList
    """
    def __init__(self, filter_path='unittest_metacorpus/mixed_files'):
        self.home_path = str(Path.home())  # path equivalent to ~
        self.filter_path = filter_path  # path to ignored_warnings folder file
        self.message_path = os.path.join(self.home_path, self.filter_path, 'IGNORED_WARNINGS')  # full path

        self.ignored_warnings_dict = {}  # structure of dict: {"file1": [(2, 20), (1, 32)]}

    def read_messages(self):
        """Reads file with messages."""
        with open(self.message_path) as f:
            file = f.read()
        return file

    def split_messages(self, file):
        """Splits every line as a single message and every word as a part of message."""
        messages = file.split(sep="\n")  # split messages
        return list(map(lambda k: (list(filter(None, re.split("[(, :)]+", k)))), messages))  # split parts of message

    def get_info_from_message(self, message):
        """Takes info from every message using the fixed structure."""
        if message[1] == "0":  # there is no type of message
            info = (0,)
        else:
            info = (int(message[1]), *list(map(int, message[2:-1])))

        if message[-1] in self.ignored_warnings_dict.keys():  # check file name in dict
            self.ignored_warnings_dict[message[-1]].append(info)  # append to existing file info
        else:
            self.ignored_warnings_dict[message[-1]] = [info]  # add new file info

    def fill_dict(self, file):
        """Runs for every message of file."""
        for msg in self.split_messages(file):
            self.get_info_from_message(msg)

    def write_json(self, save_path=None):
        """
        Saves as json file.
        Parameters
        ----------
        save_path : str
            Path to the json file."""
        if save_path is None:  # if path is not specified, saves in the same folder as messages file
            save_path = self.message_path + ".json"

        with open(save_path, "w") as f:
            json.dump(self.ignored_warnings_dict, f)

    def run(self):
        """Runs the whole loop and saves result."""
        file = self.read_messages()
        self.fill_dict(file)
        self.write_json()

    def test(self):
        """Test parsing of message with fixed structure."""
        test_file = """INCORRECT_VOLTA_MN_WARNING (1, 6) ms3.Parse.mixed_files.Did03M-Son_regina-1762'Sarti.mscx.MeasureList
        MCS_NOT_EXCLUDED_FROM_BARCOUNT_WARNING (3, 1, 40, 85) ms3.Parse.MS3.BWV_0815.mscx.MeasureList
        INCORRECT_VOLTA_MN_WARNING (1, 2, 4) ms3.Parse.MS3.BWV_0815.mscx.MeasureList"""
        self.fill_dict(test_file)
        assert self.ignored_warnings_dict == {"ms3.Parse.mixed_files.Did03M-Son_regina-1762'Sarti.mscx.MeasureList":
                                    [(1, 6)], 'ms3.Parse.MS3.BWV_0815.mscx.MeasureList': [(3, 1, 40, 85), (1, 2, 4)]}



MessageParser().test()


