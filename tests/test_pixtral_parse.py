"""Unit tests for Pixtral response parsing (no network, no PIL/httpx needed)."""

from docbench.extractors.vlm_pixtral import PixtralExtractor

parse = PixtralExtractor._parse_json


def test_parses_bare_json():
    assert parse('{"total.total_price": "12.50"}') == {"total.total_price": "12.50"}


def test_parses_fenced_json_block():
    text = 'Here you go:\n```json\n{"total.total_price": "9.99"}\n```'
    assert parse(text) == {"total.total_price": "9.99"}


def test_parses_json_embedded_in_prose():
    text = 'The result is {"total.total_price": "1.00"} as requested.'
    assert parse(text) == {"total.total_price": "1.00"}


def test_malformed_returns_empty_dict():
    assert parse("sorry, I cannot read this image") == {}


def test_non_object_json_returns_empty_dict():
    assert parse("[1, 2, 3]") == {}
