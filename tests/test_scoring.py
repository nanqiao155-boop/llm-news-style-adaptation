import pytest

from demo.scoring import release_adjusted_score


@pytest.mark.parametrize(
    ("raw", "major", "publishable", "unsupported", "expected"),
    [
        (96, 2, True, 0, 59),
        (96, 3, False, 2, 53),
        (96, 1, True, 0, 69),
        (96, 0, False, 0, 79),
        (96, 0, True, 0, 96),
        (120, 0, True, 1, 97),
        (2, 0, True, 1, 0),
        (-5, 0, True, 0, 0),
    ],
)
def test_release_adjusted_score(raw, major, publishable, unsupported, expected):
    assert release_adjusted_score(raw, major, publishable, unsupported) == expected


def test_major_risk_precedence_over_publishable():
    assert release_adjusted_score(90, 1, False, 1) == 66


@pytest.mark.parametrize("field", ["major", "unsupported"])
def test_negative_counts_rejected(field):
    args = {"raw_total": 90, "major_release_risks": 0, "publishable": True, "unsupported_claim_count": 0}
    args["major_release_risks" if field == "major" else "unsupported_claim_count"] = -1
    with pytest.raises(ValueError):
        release_adjusted_score(**args)
