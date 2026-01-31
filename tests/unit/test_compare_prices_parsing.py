import re
import unittest


TITLE_RE = re.compile(r"(?im)^\s*title:\s*(.+?)\s*$")
PRICE_RE = re.compile(r"(?i)price\s+is\s*\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)")


class TestComparePricesParsing(unittest.TestCase):
    def test_extract_title(self):
        blob = "Title: iPhone X\nCategory: Electronics\n"
        m = TITLE_RE.search(blob)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1).strip(), "iPhone X")

    def test_extract_price_with_commas(self):
        blob = "Some text\nPrice is $1,234.56\nMore text"
        m = PRICE_RE.search(blob)
        self.assertIsNotNone(m)
        price = float(m.group(1).replace(",", ""))
        self.assertAlmostEqual(price, 1234.56)

    def test_extract_logged_result_from_stderr(self):
        stderr = "INFO:root:[pricer-service] Result: 189.0\n"
        m = re.search(r"Result:\s*([-+]?\d+(?:\.\d+)?)", stderr)
        self.assertIsNotNone(m)
        self.assertEqual(float(m.group(1)), 189.0)

