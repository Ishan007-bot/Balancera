"""Money helper tests.

SPEC section 14 priority 2: rounding half-up, paise/rupee round-trips.
These are cheap and they underpin every amount in the system, so a break
here would corrupt every downstream comparison silently.
"""

import pytest

from recon.money import (
    FEE_BP, GST_BP, fee_for, format_paise, gst_for, net_for, pct_bp,
    rupees_to_paise,
)


class TestRoundingHalfUp:
    """Half-up, not banker's rounding -- what Indian gateways/GST actually use."""

    def test_exact_half_rounds_up(self):
        # 2% of 1250 == 25.0 exactly; no rounding needed.
        assert pct_bp(1250, 200) == 25
        # 2% of 1275 == 25.5 -> half-up gives 26 (banker's would give 26 too)
        assert pct_bp(1275, 200) == 26
        # 2% of 1225 == 24.5 -> half-up gives 25; banker's would give 24.
        # This is the case that distinguishes the two rules.
        assert pct_bp(1225, 200) == 25

    def test_below_half_rounds_down(self):
        assert pct_bp(1224, 200) == 24  # 24.48

    def test_known_fee_values(self):
        assert fee_for(12345) == 247  # 246.90
        assert fee_for(99999) == 2000  # 1999.98
        assert fee_for(1000000) == 20000  # exactly 20000

    def test_gst_on_fee(self):
        assert gst_for(20000) == 3600  # 18% of 200.00
        assert gst_for(247) == 44  # 44.46

    def test_zero(self):
        assert fee_for(0) == 0
        assert gst_for(0) == 0


class TestNetIdentity:
    """gross - fee - gst == net must hold for every value, with no exceptions."""

    def test_spec_worked_example(self):
        # SPEC section 2: Rs 10,000 gross -> fee Rs 200, GST Rs 36,
        # bank credit Rs 9,764.
        fee, gst, net = net_for(1_000_000)
        assert (fee, gst, net) == (20_000, 3_600, 976_400)
        assert format_paise(net) == "9,764.00"

    @pytest.mark.parametrize("gross", [1, 99, 12345, 50_000, 99_999,
                                       100_001, 555_555, 900_000, 10_000_000])
    def test_identity_holds(self, gross):
        fee, gst, net = net_for(gross)
        assert gross - fee - gst == net

    def test_identity_holds_exhaustively_over_a_range(self):
        # The generator draws gross uniformly from 50_000..900_000; spot-check
        # a dense slice of that range rather than trusting parametrised samples.
        for gross in range(50_000, 51_000):
            fee, gst, net = net_for(gross)
            assert gross - fee - gst == net


class TestFormatting:
    """format_paise is the single display boundary."""

    def test_indian_digit_grouping(self):
        assert format_paise(1_234_567) == "12,345.67"
        assert format_paise(100_000_000) == "10,00,000.00"  # 1 crore paise
        assert format_paise(10_000_000) == "1,00,000.00"

    def test_small_amounts(self):
        assert format_paise(0) == "0.00"
        assert format_paise(5) == "0.05"
        assert format_paise(100) == "1.00"
        assert format_paise(99) == "0.99"

    def test_negative(self):
        assert format_paise(-1_234_567) == "-12,345.67"
        assert format_paise(-5) == "-0.05"


class TestRoundTrip:
    @pytest.mark.parametrize("text", ["1234.56", "0.05", "1,00,000.00",
                                      "9764.00", "0.00", "99,99,999.99"])
    def test_paise_survive_a_round_trip(self, text):
        """paise -> display -> paise must be lossless. The display form may
        add Indian grouping, so compare in paise, not as strings."""
        paise = rupees_to_paise(text)
        assert rupees_to_paise(format_paise(paise)) == paise

    def test_known_display_forms(self):
        assert format_paise(rupees_to_paise("1234.56")) == "1,234.56"
        assert format_paise(rupees_to_paise("0.05")) == "0.05"
        assert format_paise(rupees_to_paise("1,00,000.00")) == "1,00,000.00"

    def test_int_rupees(self):
        assert rupees_to_paise(100) == 10_000

    def test_rejects_sub_paise_precision(self):
        with pytest.raises(ValueError):
            rupees_to_paise("1.234")

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            rupees_to_paise("abc")


class TestTypeDiscipline:
    """Amounts are ints. Floats must never enter the money path."""

    def test_rejects_float(self):
        with pytest.raises(TypeError):
            pct_bp(100.5, 200)
        with pytest.raises(TypeError):
            format_paise(100.5)

    def test_rejects_bool(self):
        # bool is an int subclass in Python; it must still be refused.
        with pytest.raises(TypeError):
            pct_bp(True, 200)

    def test_rejects_negative_input(self):
        with pytest.raises(ValueError):
            pct_bp(-100, 200)


def test_commercial_terms_are_what_the_spec_says():
    assert FEE_BP == 200   # 2%
    assert GST_BP == 1800  # 18%
