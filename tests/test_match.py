from docbench.scoring.match import CANONICAL_FIELDS, is_correct, normalize_price


def test_normalize_strips_currency_and_separators():
    assert normalize_price("$1,234.00") == normalize_price("1234")
    assert normalize_price(" 12.50 ") == normalize_price("12.5")
    assert normalize_price("0.00") == "0"


def test_normalize_handles_none_and_empty():
    assert normalize_price(None) == ""
    assert normalize_price("") == ""


def test_normalize_non_numeric_falls_back_to_text():
    assert normalize_price("Cash") == "cash"
    assert normalize_price("N/A") == normalize_price("n/a")


def test_is_correct_binary_outcome():
    assert is_correct("$1,234.00", "1234") == 1
    assert is_correct("1234", "1235") == 0
    assert is_correct(None, "10.00") == 0  # missing prediction is wrong, not skipped


def test_canonical_fields_are_dotted_group_key():
    for f in CANONICAL_FIELDS:
        assert f.count(".") == 1, f
