import math

import numpy as np
import pytest

from BinaryOptions_MC import binaryClosedForm, cnd
from ExplicitFD_CS import digitalCS, explicitFD, solveExplicitFD
from ExplicitFD_bin import explicitFdBin

S, E, R, D, SIGMA, T_FIN, T = 100.0, 100.0, 0.05, 0.0, 0.2, 1.0, 0.0
DISCOUNT = math.exp(-R * (T_FIN - T))


def bsCall(s, e, r, sigma, tau):
    d1 = (math.log(s / e) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
    return s * cnd(d1) - e * math.exp(-r * tau) * cnd(d1 - sigma * math.sqrt(tau))


def test_constant_payoff_is_discounted():
    value = solveExplicitFD(S, R, D, SIGMA, T_FIN, T, 20, 200.0, lambda stock: np.ones_like(stock))
    assert value == pytest.approx(DISCOUNT, abs=1e-4)


def test_value_is_interpolated_between_nodes():
    # a linear payoff stays linear (zero gamma), so interpolation off the grid is exact
    linear = lambda stock: stock.copy()
    onGrid = solveExplicitFD(100.0, 0.0, 0.0, SIGMA, T_FIN, T, 20, 200.0, linear)
    offGrid = solveExplicitFD(103.0, 0.0, 0.0, SIGMA, T_FIN, T, 20, 200.0, linear)
    assert onGrid == pytest.approx(100.0)
    assert offGrid == pytest.approx(103.0)


def test_call_matches_black_scholes():
    assert explicitFD(S, E, R, D, SIGMA, T_FIN, T, "c", 200) == pytest.approx(bsCall(S, E, R, SIGMA, T_FIN), abs=0.05)


def test_put_call_parity():
    call = explicitFD(S, E, R, D, SIGMA, T_FIN, T, "c", 100)
    put = explicitFD(S, E, R, D, SIGMA, T_FIN, T, "p", 100)
    assert call - put == pytest.approx(S - E * DISCOUNT, abs=1e-3)


def test_call_spread_digital_matches_closed_form():
    exact = binaryClosedForm(S, E, R, D, SIGMA, T_FIN, T, "gt")
    assert digitalCS(S, E, R, D, SIGMA, T_FIN, T, "c", 400, 1.0, 1.0) == pytest.approx(exact, abs=5e-3)


def test_binary_call_matches_closed_form():
    exact = binaryClosedForm(S, E, R, D, SIGMA, T_FIN, T, "gt")
    assert explicitFdBin(S, E, R, D, SIGMA, T_FIN, T, "c", 400) == pytest.approx(exact, abs=5e-3)


def test_binary_call_plus_put_is_discount_factor():
    call = explicitFdBin(S, E, R, D, SIGMA, T_FIN, T, "c", 100)
    put = explicitFdBin(S, E, R, D, SIGMA, T_FIN, T, "p", 100)
    assert call + put == pytest.approx(DISCOUNT, abs=1e-4)


@pytest.mark.parametrize("bad", [
    dict(oType="x"),
    dict(nas=2),
    dict(sigma=0.0),
    dict(t=1.0),
    dict(s=250.0),  # beyond the grid top of 2 * strike
])
def test_vanilla_rejects_invalid_input(bad):
    kwargs = dict(s=S, e=E, r=R, d=D, sigma=SIGMA, tFin=T_FIN, t=T, oType="c", nas=20) | bad
    with pytest.raises(AssertionError):
        explicitFD(**kwargs)


@pytest.mark.parametrize("bad", [dict(oType="x"), dict(e=0.0)])
def test_binary_rejects_invalid_input(bad):
    kwargs = dict(s=S, e=E, r=R, d=D, sigma=SIGMA, tFin=T_FIN, t=T, oType="c", nas=20) | bad
    with pytest.raises(AssertionError):
        explicitFdBin(**kwargs)


@pytest.mark.parametrize("width", [0.0, 200.0])
def test_call_spread_rejects_invalid_width(width):
    with pytest.raises(AssertionError):
        digitalCS(S, E, R, D, SIGMA, T_FIN, T, "c", 20, 1.0, width)
