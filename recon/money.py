"""Integer-paise money helpers.

Every monetary value in this system is an ``int`` number of paise. Never a
float, never a Decimal, never a rupee string. Floats lose exactness under
addition, which is fatal when we compare a sum of payments against a bank
credit and expect equality to within a couple of paise.

Rupees exist only at the display boundary, via ``format_paise``.
"""

# Gateway commercial terms. Basis points avoid any fractional arithmetic:
# 200 bp = 2.00% fee on gross, 1800 bp = 18.00% GST on the fee.
FEE_BP = 200
GST_BP = 1800


def pct_bp(amount_paise: int, basis_points: int) -> int:
    """Return ``basis_points`` of ``amount_paise``, rounded half-up.

    Half-up (not banker's rounding) because that is what Indian payment
    gateways and GST rules actually use, and the generator must reproduce
    real settlement arithmetic rather than Python's default.
    """
    if not isinstance(amount_paise, int) or isinstance(amount_paise, bool):
        raise TypeError(f"amount_paise must be int, got {type(amount_paise).__name__}")
    if amount_paise < 0:
        raise ValueError(f"amount_paise must be non-negative, got {amount_paise}")
    # +5000 then floor-divide by 10000 == round-half-up on the bp scale.
    return (amount_paise * basis_points + 5000) // 10000


def fee_for(gross_paise: int) -> int:
    """Gateway fee on a gross amount: 2%, rounded half-up."""
    return pct_bp(gross_paise, FEE_BP)


def gst_for(fee_paise: int) -> int:
    """GST charged on the gateway fee: 18%, rounded half-up."""
    return pct_bp(fee_paise, GST_BP)


def net_for(gross_paise: int) -> tuple[int, int, int]:
    """Split a gross amount into ``(fee, gst, net)``.

    ``net = gross - fee - gst`` holds exactly by construction. This identity
    is asserted by the validator for every generated payment.
    """
    fee = fee_for(gross_paise)
    gst = gst_for(fee)
    return fee, gst, gross_paise - fee - gst


def rupees_to_paise(rupees: str | int) -> int:
    """Parse a rupee figure into integer paise.

    Accepts ``"1234.56"``, ``"1,234.56"`` or a plain int of whole rupees.
    Used only at input boundaries; internal code passes paise directly.
    """
    if isinstance(rupees, int) and not isinstance(rupees, bool):
        return rupees * 100
    text = str(rupees).strip().replace(",", "").replace("\u20b9", "")
    if not text:
        raise ValueError("empty rupee string")
    negative = text.startswith("-")
    if negative:
        text = text[1:]
    if "." in text:
        whole, _, frac = text.partition(".")
        if len(frac) > 2:
            raise ValueError(f"more than 2 decimal places: {rupees!r}")
        frac = frac.ljust(2, "0")
    else:
        whole, frac = text, "00"
    if not whole.isdigit() or not frac.isdigit():
        raise ValueError(f"not a valid rupee amount: {rupees!r}")
    total = int(whole) * 100 + int(frac)
    return -total if negative else total


def format_paise(amount_paise: int) -> str:
    """Format paise as a rupee string with Indian digit grouping.

    The single display boundary. ``1234567`` -> ``"12,345.67"``.
    """
    if not isinstance(amount_paise, int) or isinstance(amount_paise, bool):
        raise TypeError(f"amount_paise must be int, got {type(amount_paise).__name__}")
    sign = "-" if amount_paise < 0 else ""
    whole, frac = divmod(abs(amount_paise), 100)
    return f"{sign}{_group_indian(whole)}.{frac:02d}"


def _group_indian(n: int) -> str:
    """Group digits the Indian way: last 3, then 2s. 1234567 -> 12,34,567."""
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail
