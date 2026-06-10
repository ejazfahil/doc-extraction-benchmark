"""Pure-logic tests for the OCR field parser (no engine, no PIL)."""

from docbench.extractors.ocr import parse_fields_from_text

RECEIPT = """\
MARIO'S TRATTORIA
Margherita        8.50
Espresso          2.00
Subtotal         10.50
Tax (10%)         1.05
Service           1.00
TOTAL            12.55
Cash             20.00
Change            7.45
"""


def test_parses_summary_fields():
    fields = parse_fields_from_text(RECEIPT)
    assert fields["sub_total.subtotal_price"] == "10.50"
    assert fields["sub_total.tax_price"] == "1.05"
    assert fields["sub_total.service_price"] == "1.00"
    assert fields["total.total_price"] == "12.55"
    assert fields["total.cashprice"] == "20.00"
    assert fields["total.changeprice"] == "7.45"


def test_total_does_not_steal_subtotal_line():
    # The generic "total" keyword must not capture the subtotal amount.
    assert parse_fields_from_text(RECEIPT)["total.total_price"] == "12.55"


def test_missing_fields_are_absent_not_guessed():
    fields = parse_fields_from_text("TOTAL  9.99")
    assert fields == {"total.total_price": "9.99"}
    assert "sub_total.tax_price" not in fields


def test_rightmost_amount_chosen():
    assert parse_fields_from_text("TOTAL 3 items 15.00")["total.total_price"] == "15.00"
