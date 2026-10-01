import math

import numpy as np
import pytest
from scipy import stats

from CdsBasket import cdsBasket

NUM_ASSETS = 5
CORR = 0.5 * np.ones((NUM_ASSETS, NUM_ASSETS)) + 0.5 * np.eye(NUM_ASSETS)


def writeCdsFile(path, numRows=260, numAssets=NUM_ASSETS, seed=0):
    '''Write a synthetic HistCdsValues.csv: header row, date column, correlated lognormal CDS levels.'''
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((numRows, numAssets)) @ np.linalg.cholesky(CORR[:numAssets, :numAssets]).T * 0.03
    levels = np.array([173, 61, 113, 390, 40][:numAssets]) * np.exp(np.cumsum(shocks, axis=0))
    with open(path, "w") as f:
        f.write("Date," + ",".join("A{}".format(i) for i in range(numAssets)) + "\n")
        for i, row in enumerate(levels):
            f.write("{},".format(i) + ",".join("{:.4f}".format(x) for x in row) + "\n")
    return path


@pytest.fixture
def cdsFile(tmp_path):
    return str(writeCdsFile(tmp_path / "HistCdsValues.csv"))


@pytest.fixture
def basket(cdsFile):
    return cdsBasket(cdsFile)


@pytest.fixture(scope="module")
def pricedBasket(tmp_path_factory):
    '''A basket after a full run (module scoped: the run takes a few seconds).'''
    cb = cdsBasket(str(writeCdsFile(tmp_path_factory.mktemp("data") / "HistCdsValues.csv")))
    cb.runFunc()
    return cb


def test_init_reads_levels(basket):
    assert basket.cdsLevels.shape == (261, NUM_ASSETS + 1)
    assert basket.numAssets == NUM_ASSETS
    assert basket.simDays == 259


def test_init_rejects_wrong_number_of_assets(tmp_path):
    with pytest.raises(AssertionError):
        cdsBasket(str(writeCdsFile(tmp_path / "four.csv", numAssets=4)))


def test_calc_ret(basket):
    basket.calcRet()
    levels = basket.cdsLevels[1:, 1:]
    assert basket.cdsReturns.shape == (259, NUM_ASSETS)
    assert basket.cdsReturns[0, 0] == pytest.approx(levels[1, 0] / levels[0, 0] - 1)
    assert basket.cdsReturns[-1, 4] == pytest.approx(levels[-1, 4] / levels[-2, 4] - 1)


def test_calc_ret_rejects_non_positive_levels(basket):
    basket.cdsLevels[5, 2] = 0.
    with pytest.raises(AssertionError):
        basket.calcRet()


def test_calc_corr_without_and_with_shift(basket):
    basket.calcRet()
    basket.calcCorr()
    np.testing.assert_allclose(np.diag(basket.corrMatrix), 1.)
    np.testing.assert_allclose(basket.corrMatrix, basket.corrMatrix.T)
    # the synthetic data has a true correlation of 0.5
    offDiag = basket.corrMatrix[~np.eye(NUM_ASSETS, dtype=bool)]
    assert np.all(np.abs(offDiag - 0.5) < 0.15)

    basket.shiftMatrix.fill(0.05)
    np.fill_diagonal(basket.shiftMatrix, 0.)
    basket.calcCorr(1)
    np.testing.assert_allclose(basket.corrMatrix, basket.corrMatrixOrig + basket.shiftMatrix)


def test_calc_eigenvalue(basket):
    basket.calcRet()
    basket.calcCorr()
    basket.calcEigenvalue()
    assert np.sum(basket.eigenvalue) == pytest.approx(NUM_ASSETS)  # trace of a correlation matrix
    assert np.all(basket.eigenvalue > 0)


def test_create_iid_unif(basket):
    basket.createIidUnif()
    assert basket.iidUnif.shape == (basket.numSim, NUM_ASSETS)
    assert basket.iidUnifMixing.shape == (basket.numSim,)
    allPoints = np.column_stack([basket.iidUnif, basket.iidUnifMixing])
    assert np.all((allPoints > 0) & (allPoints < 1))
    # low discrepancy: every dimension has a sample mean very close to 1/2
    np.testing.assert_allclose(allPoints.mean(axis=0), 0.5, atol=2e-3)
    # the mixing dimension is uncorrelated with the asset dimensions
    assert np.all(np.abs(np.corrcoef(allPoints.T)[-1, :-1]) < 0.05)

    first = allPoints.copy()
    basket.createIidUnif()
    np.testing.assert_array_equal(np.column_stack([basket.iidUnif, basket.iidUnifMixing]), first)


def test_create_iid_unif_depends_on_seed(basket):
    basket.createIidUnif()
    first = basket.iidUnif.copy()
    basket.sobolSeed += 1
    basket.createIidUnif()
    assert not np.allclose(basket.iidUnif, first)


def test_calc_tau_inverts_exponential_cdf(basket):
    times = np.linspace(0.1, 10, basket.numSim * NUM_ASSETS).reshape(basket.numSim, NUM_ASSETS)
    unif = 1 - np.exp(-basket.hazardRates[:, 0] * times)
    tau, sortedTau = basket.calcTau(unif)
    np.testing.assert_allclose(tau, times)
    np.testing.assert_allclose(sortedTau, np.sort(times, axis=1))


def test_calc_tau_rejects_values_outside_unit_interval(basket):
    with pytest.raises(AssertionError):
        basket.calcTau(np.full((basket.numSim, NUM_ASSETS), 1.5))


def test_get_disc_fact_int(pricedBasket):
    df = pricedBasket.discFact
    assert pricedBasket.getDiscFactInt(0.) == 1.
    for year in range(1, 6):
        assert pricedBasket.getDiscFactInt(float(year)) == pytest.approx(df[year - 1])
    # between two curve points the factor lies between their values
    assert df[2] < pricedBasket.getDiscFactInt(2.5) < df[1]


def test_get_disc_fact_int_rejects_tau_beyond_curve(pricedBasket):
    with pytest.raises(AssertionError):
        pricedBasket.getDiscFactInt(5.5)


def test_arg_max_fct_matches_scipy_reference(pricedBasket):
    cb = pricedBasket
    unif = stats.rankdata(cb.cdsReturns, axis=0) / (cb.simDays + 1.)
    for nu in (3, 7, 11):
        z = stats.t.ppf(unif, nu)
        reference = np.sum(stats.multivariate_t.logpdf(z, loc=np.zeros(NUM_ASSETS), shape=cb.corrMatrix, df=nu)) \
            - np.sum(stats.t.logpdf(z, nu))
        assert cb.argValArray[nu] == pytest.approx(reference, rel=1e-10)


@pytest.mark.parametrize("trueNu", [4, 6])
def test_arg_max_fct_recovers_degrees_of_freedom(basket, trueNu):
    rng = np.random.default_rng(trueNu)
    numObs = 3000
    chi2 = rng.chisquare(trueNu, numObs) / trueNu
    tSample = rng.standard_normal((numObs, NUM_ASSETS)) @ np.linalg.cholesky(CORR).T / np.sqrt(chi2)[:, None]
    # non-t margins on purpose: the copula fit must only see the dependence structure
    basket.cdsReturns = np.exp(0.02 * stats.norm.ppf(stats.t.cdf(tSample, trueNu))) - 1
    basket.simDays = numObs
    basket.corrMatrix = CORR
    basket.argMaxFct()
    assert abs(basket.nuVal - trueNu) <= 1


def test_calc_spreads_by_hand(basket):
    basket.numSim = 2
    basket.discFact = np.full(basket.periods, 0.99)
    sortedTau = np.array([[0.5, 6., 6., 6., 6.],   # only the first name defaults, half way into year 1
                          [6., 6., 6., 6., 6.]])  # nothing defaults before maturity
    spreads, denom = basket.calcSpreads(sortedTau)
    # numerator (1-R) DF / n over denominator DF * 0.5: the discount factor cancels
    assert spreads[0, 0] == pytest.approx((1 - basket.recov) / NUM_ASSETS / 0.5)
    assert denom[0, 0] == pytest.approx(0.5 * basket.getDiscFactInt(0.5))
    assert np.all(spreads[0, 1:] == 0) and np.all(spreads[1] == 0)


def test_calc_spreads_applies_tau_floor(basket):
    basket.numSim = 1
    basket.discFact = np.full(basket.periods, 0.99)
    spreads, _ = basket.calcSpreads(np.array([[0.01, 6., 6., 6., 6.]]))
    # a default before tauFloor is treated as a default at tauFloor
    assert spreads[0, 0] == pytest.approx((1 - basket.recov) / NUM_ASSETS / basket.tauFloor)


def test_calc_spreads_rejects_unsorted_times(basket):
    basket.numSim = 1
    basket.discFact = np.full(basket.periods, 0.99)
    with pytest.raises(AssertionError):
        basket.calcSpreads(np.array([[2., 1., 6., 6., 6.]]))


def test_run_func_results(pricedBasket):
    cb = pricedBasket
    for results in (cb.gaussResults, cb.tResults):
        assert results.shape == (NUM_ASSETS,)
        assert np.all(np.isfinite(results)) and np.all(results >= 0)
        # a k-th to default contract is cheaper than the (k-1)-th
        assert np.all(np.diff(results) < 0)
    assert 3 <= cb.nuVal <= 11
    assert math.isclose(np.sum(cb.cdsSpreads[:, 4]) * 10000, 777.41)


def test_mixing_variable_has_mean_close_to_nu_over_nu_minus_2(pricedBasket):
    nu = pricedBasket.nuVal
    assert np.mean(pricedBasket.mixing) == pytest.approx(nu / (nu - 2), rel=0.05)


def test_print_result(pricedBasket, capsys):
    pricedBasket.printResult()
    out = capsys.readouterr().out
    assert out.count("Gaussian copula is:") == NUM_ASSETS
    assert out.count("t copula is:") == NUM_ASSETS
    assert "777.41 bp" in out


def test_create_latex_tables(pricedBasket, capsys):
    pricedBasket.createLatexTables()
    out = capsys.readouterr().out
    assert out.count("\\begin{tabular}{|c|c|c|c|c|c|}") == 2
    assert "390.41" in out and "Thyssen Krupp & 1.00" in out


def test_print_latex_table_rejects_short_rows(basket):
    with pytest.raises(AssertionError):
        basket.printLatexTable("x", ["row"], [["1", "2"]])


def test_insert_graphs_latex(basket, capsys):
    basket.insertGraphsLatex()
    lines = capsys.readouterr().out.splitlines()
    pairs = NUM_ASSETS * (NUM_ASSETS - 1) // 2
    assert len(lines) == 2 * pairs + 2 * NUM_ASSETS
    assert len(set(lines)) == len(lines)


def test_plots_are_saved(pricedBasket, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pricedBasket.runTauHist()
    pricedBasket.runTauScatter()
    saved = {p.name for p in tmp_path.glob("*.png")}
    assert {"histGaussTauAll.png", "histTTauAll.png", "histGaussTau1.png", "histTTau5.png",
            "scatGaussTau1Tau2.png", "scatTTau4Tau5.png"} <= saved


def test_run_sensitivities(cdsFile, tmp_path, monkeypatch, capsys):
    cb = cdsBasket(cdsFile)
    cb.numSim = 256  # keep the 21 repricings quick (a power of 2 for the Sobol points)
    monkeypatch.chdir(tmp_path)
    cb.runSensitivities()
    capsys.readouterr()
    assert {"gaussCorrSensi.png", "tCorrSensi.png"} <= {p.name for p in tmp_path.glob("*.png")}
    assert np.all(np.isfinite(cb.spdGaussResultStore)) and np.all(np.isfinite(cb.spdTResultStore))
    # higher correlation makes a first-to-default cheaper
    assert cb.spdGaussResultStore[-1, 0] < cb.spdGaussResultStore[0, 0]
