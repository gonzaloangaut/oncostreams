"""Enforce that historical experimental backends cannot silently run."""
import pytest

from tests._helpers import make_batch_culture, make_elongation_culture


@pytest.mark.parametrize("retired", ["direct_numba", "batch_numba", "batch_cached_numba"])
def test_culture_rejects_retired_interactions(retired):
    with pytest.raises(ValueError, match="interaction_backend"):
        make_batch_culture(retired)


def test_culture_rejects_retired_elongation():
    with pytest.raises(ValueError, match="elongation_backend"):
        make_elongation_culture("fast_numba")


