import pandas as pd
import pytest
from ms3.utils import dataframe2markdown

# expected outputs were produced by pytablewriter 1.0.0, which dataframe2markdown() used to wrap
CASES = [
    (
        {"file_name": ["a", "bbbbbbb"], "measures": [12, 3], "labels": ["", "x"]},
        "|file_name|measures|labels|\n"
        "|---------|-------:|------|\n"
        "|a        |      12|      |\n"
        "|bbbbbbb  |       3|x     |\n",
    ),
    (
        {"d": [1, 2.0, 3.125], "e": ["1e3", "-4", "0.5"], "s": ["2.3.0", "2.3.0", ""]},
        "|  d  |  e   |  s  |\n"
        "|----:|-----:|-----|\n"
        "|1.000|1000.0|2.3.0|\n"
        "|2.000|  -4.0|2.3.0|\n"
        "|3.125|   0.5|     |\n",
    ),
    (
        {"a": ["001", "1-120", "100", "bwv69.6"], "b": ["bwv69.6", "1", "x", "2"]},
        "|   a   |   b   |\n"
        "|-------|-------|\n"
        "|      1|bwv69.6|\n"
        "|1-120  |1      |\n"
        "|    100|x      |\n"
        "|bwv69.6|2      |\n",
    ),
    (
        {"a": ["abcdef", 1, 2.5, True, "2"], "long_header": [1, 2, 3, 4, 5]},
        "|  a   |long_header|\n"
        "|------|----------:|\n"
        "|abcdef|          1|\n"
        "|     1|          2|\n"
        "|   2.5|          3|\n"
        "|True  |          4|\n"
        "|2     |          5|\n",
    ),
    ({"a": [], "b": []}, "| a | b |\n|---|---|\n"),
]


@pytest.mark.parametrize("data, expected", CASES)
def test_dataframe2markdown(data, expected):
    assert dataframe2markdown(pd.DataFrame(data)) == expected


def test_dataframe2markdown_heading():
    df = pd.DataFrame({"x": ["日本"]}, index=["ignored"])
    assert (
        dataframe2markdown(df, name="Overview")
        == "# Overview\n| x  |\n|----|\n|日本|\n"
    )


def test_dataframe2markdown_escapes_pipes_and_newlines():
    # pytablewriter mis-padded escaped cells
    df = pd.DataFrame({"abcdef": ["x|y", "a\nb"]})
    assert dataframe2markdown(df) == "|abcdef|\n|------|\n|x\\|y  |\n|a b   |\n"
