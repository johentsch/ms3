import os

from ms3 import Parse, first_level_subdirs


class TestEquivalence():

    def test_parallel(self, directory):
        a = Parse(directory)
        b = Parse(directory)
        a.parse_mscx()
        b.parse_mscx(parallel=False)
        assert a.info(return_str=True) == b.info(return_str=True)

    def test_add_corpus(self, directory):
        a = Parse(directory)
        b = Parse()
        c = Parse()
        b.add_corpus(directory)
        for sd in first_level_subdirs(directory):
            corpus = os.path.join(directory, sd)
            c.add_corpus(corpus)
        assert a.count_extensions(per_key=True) == c.count_extensions(per_key=True)
        assert len(b.count_extensions(per_key=True)) == 1