from ms3 import fifths2name, fifths2rn, fifths2sd


def test_fifths2sd():
    expected_major = [
        acc + sd
        for acc in ("bb", "b", "", "#", "##")
        for sd in ("4", "1", "5", "2", "6", "3", "7")
    ]
    for fifths, exp_maj in zip(range(-15, 16), expected_major):
        assert fifths2sd(fifths) == exp_maj


def test_fifths2rn():
    expected_major = [
        acc + rn
        for acc in ("bb", "b", "", "#", "##")
        for rn in ("IV", "I", "V", "II", "VI", "III", "VII")
    ]
    for fifths, exp_maj in zip(range(-15, 16), expected_major):
        assert fifths2rn(fifths) == exp_maj


def test_fifths2name():
    expected_major = [
        name + acc
        for acc in ("bb", "b", "", "#", "##")
        for name in ("F", "C", "G", "D", "A", "E", "B")
    ]
    for fifths, exp_maj in zip(range(-15, 16), expected_major):
        assert fifths2name(fifths=fifths) == exp_maj
