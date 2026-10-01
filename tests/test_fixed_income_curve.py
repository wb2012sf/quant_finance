import numpy as np
import pytest

from FixedIncomeCurve import fixedIncomeCurve


def test_zero_rates_give_unit_discount_factors():
    curve = fixedIncomeCurve(5, np.ones(5), np.zeros(5))
    np.testing.assert_allclose(curve.calcDiscFact(), np.ones(5))


def test_first_discount_factor():
    curve = fixedIncomeCurve(3, np.ones(3), np.array([0.02, 0.03, 0.04]))
    assert curve.calcDiscFact()[0] == pytest.approx(1 / 1.02)


def test_bootstrapped_factors_reprice_par_swaps():
    # a par swap of maturity i: r_i * sum_{k<=i} delta_k DF_k + DF_i = 1
    rates = np.array([0.001, 0.002, 0.003, 0.0035, 0.0037, 0.0038])
    delta = np.ones(5)
    df = fixedIncomeCurve(5, delta, rates).calcDiscFact()
    for i in range(5):
        assert rates[i] * np.sum(delta[:i + 1] * df[:i + 1]) + df[i] == pytest.approx(1.0)


def test_discount_factors_decrease_for_positive_rates():
    df = fixedIncomeCurve(5, np.ones(5), np.full(5, 0.03)).calcDiscFact()
    assert np.all(np.diff(df) < 0)


@pytest.mark.parametrize("periods, deltaTime, rates", [
    (0, np.ones(5), np.ones(5)),
    (5, np.ones(3), np.ones(5)),
    (5, np.ones(5), np.ones(3)),
    (5, np.zeros(5), np.ones(5)),
])
def test_rejects_invalid_input(periods, deltaTime, rates):
    with pytest.raises(AssertionError):
        fixedIncomeCurve(periods, deltaTime, rates)
