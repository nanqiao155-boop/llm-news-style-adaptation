from __future__ import annotations

from numbers import Real


def release_adjusted_score(
    raw_total: Real,
    major_release_risks: int,
    publishable: bool,
    unsupported_claim_count: int,
) -> float:
    """Apply the frozen Release-Adjusted Score v1 rule."""
    if not isinstance(raw_total, Real):
        raise TypeError("raw_total must be numeric")
    if isinstance(major_release_risks, bool) or not isinstance(major_release_risks, int):
        raise TypeError("major_release_risks must be an integer")
    if not isinstance(publishable, bool):
        raise TypeError("publishable must be a boolean")
    if isinstance(unsupported_claim_count, bool) or not isinstance(unsupported_claim_count, int):
        raise TypeError("unsupported_claim_count must be an integer")
    if major_release_risks < 0 or unsupported_claim_count < 0:
        raise ValueError("risk counts cannot be negative")

    if major_release_risks >= 2:
        cap = 59
    elif major_release_risks == 1:
        cap = 69
    elif publishable is False:
        cap = 79
    else:
        cap = 100

    return float(max(0, min(float(raw_total), cap) - 3 * unsupported_claim_count))
