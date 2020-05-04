#!/usr/bin/env python

"""Tests for `ms3` package."""

import pytest
import os

import ms3


# @pytest.fixture
# def response():
#     """Sample pytest fixture.
#
#     See more at: http://doc.pytest.org/en/latest/fixture.html
#     """
#     import requests
#     return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


# def test_content(response):
#     """Sample pytest test function with the pytest fixture as an argument."""
#     from bs4 import BeautifulSoup
#     assert 'GitHub' in BeautifulSoup(response.content).title.string

def create_score_object():
    s = ms3.Score()
    assert isinstance(s, ms3.Score)


@pytest.mark.parametrize('mscx, version', [('D973deutscher01.mscx', '3.3.4')])
def test_parse_versions(mscx, version):
    mscx = os.path.realpath(os.path.join('mscx', mscx))
    s = ms3.Score()
    s.parse_mscx(mscx, parser='bs4')
    assert s.xml.version == version
    s.xml.measures.to_csv('measures.tsv', sep='\t', index=False)
    s.xml.events.to_csv('events.tsv', sep='\t', index=False)
    s.xml.notes.to_csv('notes.tsv', sep='\t', index=False)
