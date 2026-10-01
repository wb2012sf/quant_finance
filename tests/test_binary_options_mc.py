import math

import pytest

from BinaryOptions_MC import FREQ_DT, binaryClosedForm, binaryMC, cnd

# S = E = 100, r = 5%, d = 0, sigma = 20%, T = 1
ARGS = (100, 100, 0.05, 0.0, 0.2, 1.0, 0.0)
DISCOUNT = math.exp(-0.05)


def test_cnd_known_values():
    assert cnd(0) == pytest.approx(0.5)
    assert cnd(1.959964) == pytest.approx(0.975, abs=1e-6)
    assert cnd(-1.0) == pytest.approx(1 - cnd(1.0))


def test_closed_form_reference_value():
    # e^{-rT} N(d2) with d2 = (r - sigma^2/2) T / (sigma sqrt(T)) = 0.15
    assert binaryClosedForm(*ARGS, "gt") == pytest.approx(DISCOUNT * cnd(0.15), rel=1e-12)
    assert binaryClosedForm(*ARGS, "gt") == pytest.approx(0.5323248, abs=1e-7)


def test_closed_form_gt_plus_lt_is_discount_factor():
    assert binaryClosedForm(*ARGS, "gt") + binaryClosedForm(*ARGS, "lt") == pytest.approx(DISCOUNT)


def test_closed_form_dividend_lowers_call_digital():
    withDiv = binaryClosedForm(100, 100, 0.05, 0.03, 0.2, 1.0, 0.0, "gt")
    assert withDiv < binaryClosedForm(*ARGS, "gt")


@pytest.mark.parametrize("bad", [
    dict(optType="ge"),
    dict(s=0),
    dict(sigma=0),
    dict(t=1.0),
])
def test_closed_form_rejects_invalid_input(bad):
    kwargs = dict(s=100, e=100, r=0.05, d=0.0, sigma=0.2, tFin=1.0, t=0.0, optType="gt") | bad
    with pytest.raises(AssertionError):
        binaryClosedForm(**kwargs)


def test_freq_steps():
    assert FREQ_DT["a"] == 1.0
    assert FREQ_DT["s"] == 0.5
    assert FREQ_DT["q"] == 0.25
    assert FREQ_DT["d"] == pytest.approx(1 / 252)


def test_mc_is_reproducible_with_seed():
    assert binaryMC(*ARGS, "gt", 1000, "m", seed=7) == binaryMC(*ARGS, "gt", 1000, "m", seed=7)


def test_mc_gt_plus_lt_is_discount_factor_for_same_paths():
    total = binaryMC(*ARGS, "gt", 1000, "q", seed=3) + binaryMC(*ARGS, "lt", 1000, "q", seed=3)
    assert total == pytest.approx(DISCOUNT)


@pytest.mark.parametrize("freq", ["a", "q"])
def test_mc_converges_to_closed_form(freq):
    numPaths = 200_000
    exact = binaryClosedForm(*ARGS, "gt")
    stdErr = DISCOUNT * math.sqrt(0.56 * 0.44 / numPaths)
    assert binaryMC(*ARGS, "gt", numPaths, freq, seed=11) == pytest.approx(exact, abs=4 * stdErr)


def test_mc_rejects_unknown_freq():
    with pytest.raises(ValueError):
        binaryMC(*ARGS, "gt", 10, "x")


@pytest.mark.parametrize("bad", [dict(optType="ge"), dict(paths=0), dict(sigma=-0.2)])
def test_mc_rejects_invalid_input(bad):
    kwargs = dict(s=100, e=100, r=0.05, d=0.0, sigma=0.2, tFin=1.0, t=0.0,
                  optType="gt", paths=10, freq="a") | bad
    with pytest.raises(AssertionError):
        binaryMC(**kwargs)
