import os

from ms3.cli import get_arg_parser, review_cmd

def test_review_cmd(directory):
    parser = get_arg_parser()
    sweelinck = os.path.join(directory, "sweelinck_keyboard")
    args = parser.parse_args(["review", "-d", sweelinck, "-X"])
    review_cmd(args)