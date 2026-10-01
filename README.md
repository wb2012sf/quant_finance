# quant_finance

[![CI](https://github.com/wb2012sf/quant_finance/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/wb2012sf/quant_finance/actions/workflows/ci.yml)

Python implementations of numerical pricing methods in quantitative finance: binary options (closed form,
Monte Carlo and explicit finite differences) and a k-th to default CDS basket priced with Gaussian and
Student t copulas.

## Setup

Requires Python 3.10 or newer. The code is fully type-annotated and checks its inputs with `assert` statements.
Running Python with `-O` disables those checks.

Dependencies are declared in `pyproject.toml`. `requirements.txt` is a lock file generated from it: exact versions
with SHA-256 hashes, resolved for **Python 3.12+**. For a reproducible, hash-verified install:

```bash
pip install --require-hashes -r requirements.txt
```

On Python 3.10 or 3.11, install from `pyproject.toml` instead. This resolves versions compatible with your interpreter:

```bash
pip install -e .            # runtime dependencies
pip install -e ".[dev]"     # plus mypy, pyflakes, pytest, scipy-stubs
mypy                        # type-check (configured in pyproject.toml)
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The unit tests in `tests/` cover every module (about 100 tests, a few seconds). They check prices against
closed-form Black-Scholes values and parity relations, test the discount curve against par swap repricing, check
the Sobol draws for balance and reproducibility, and test the t-copula likelihood against `scipy.stats`. They
also check that invalid input trips the asserts. The CDS basket tests generate a synthetic `HistCdsValues.csv`
in a temporary directory, so they do not need the real data file.

To update the lock file after changing dependencies (requires `pip install pip-tools`):

```bash
pip-compile --generate-hashes --strip-extras --allow-unsafe --output-file requirements.txt pyproject.toml
```

## Scripts

| File | What it does | Run |
| --- | --- | --- |
| `BinaryOptions_MC.py` | Cash-or-nothing binary option: Black-Scholes closed form vs. Monte Carlo for different path counts and time-step frequencies (`a`nnual, `s`emi-annual, `q`uarterly, `m`onthly, `w`eekly, `d`aily). | `python BinaryOptions_MC.py` |
| `ExplicitFD_CS.py` | Explicit finite difference solver for the Black-Scholes PDE (`solveExplicitFD`), vanilla calls/puts, and a digital option approximated by a call spread. Prints a convergence table for increasing asset steps (NAS). | `python ExplicitFD_CS.py` |
| `ExplicitFD_bin.py` | Binary call/put priced directly with the same explicit FD solver, with a convergence table. | `python ExplicitFD_bin.py` |
| `CdsBasket.py` | Fair spreads of a k-th to default basket on five CDS names using Sobol quasi-random numbers with Gaussian and t copulas. | `python CdsBasket.py` |
| `FixedIncomeCurve.py` | Bootstraps discount factors from OIS rates. Used by `CdsBasket.py`. | – |

The default parameters (S = E = 100, r = 5%, σ = 20%, T = 1 year) are set in each script's
`if __name__ == "__main__":` block. Edit them there, or import the functions:

```python
from BinaryOptions_MC import binaryClosedForm, binaryMC
from ExplicitFD_CS import explicitFD, digitalCS
from ExplicitFD_bin import explicitFdBin

binaryClosedForm(100, 100, 0.05, 0.0, 0.2, 1.0, 0.0, "gt")              # 0.5323
binaryMC(100, 100, 0.05, 0.0, 0.2, 1.0, 0.0, "gt", 100000, "w", seed=1)
explicitFdBin(100, 100, 0.05, 0.0, 0.2, 1.0, 0.0, "c", 400)
```

The argument order is `(s, e, r, d, sigma, tFin, t, type, ...)`: spot, strike, rate, dividend yield,
volatility, expiry and valuation time.

## CDS basket

`CdsBasket.py` reads historical CDS levels from **`HistCdsValues.csv`** in the working directory.
That file is **not included** in the repository. The format it expects:

- comma separated, with one header row (skipped)
- first column: date (ignored)
- the following five columns: CDS levels for Thyssen Krupp, Daimler, Renault, Fiat and Siemens

Pipeline (`cdsBasket.runFunc`):

1. Bootstrap OIS discount factors (`FixedIncomeCurve.py`).
2. Compute returns and their correlation matrix from the historical CDS levels.
3. Draw scrambled Sobol uniforms (`scipy.stats.qmc.Sobol`, seeded with `cb.sobolSeed` for reproducible results)
   and turn them into correlated Gaussian-copula and t-copula uniforms (Cholesky, plus a chi² mixing variable
   for the t copula, drawn from its own Sobol dimension).
4. Map the uniforms to exponential default times using the hazard rates.
5. Compute the fair spread of the 1st to 5th to default contracts and average over the simulations.

The t-copula degrees of freedom are estimated by maximum likelihood over ν ∈ [3, 11]
(`cb.nuVal`, log-likelihoods per ν in `cb.argValArray`). Earlier versions overrode the estimate
with a fixed ν = 40; the estimate is now used directly.

### Note: correction of the degrees-of-freedom likelihood (`argMaxFct`)

**Previous formula.** With raw weekly returns r_j, the correlation matrix P and n = 5 assets, ν was chosen to maximise

    L(ν) = Σ_j [ ln((ν+n)/2) − (n/2)·ln ν − ln Γ(ν/2) − ((ν+n)/2)·ln(1 + r_j' P⁻¹ r_j / (ν−2)) ]

**Problems with it:**

1. **Missing gamma function.** The multivariate t density has ln Γ((ν+n)/2) where the formula had
   ln((ν+n)/2). The term depends on ν, so it biased which ν maximised the likelihood.
2. **Inconsistent scaling.** Dividing the quadratic form by (ν−2) treats P as the *covariance* of the
   t distribution (Cov = ν/(ν−2)·shape). The matching normalising term would then be −(n/2)·ln(ν−2), not
   −(n/2)·ln ν. The two parametrisations were mixed.
3. **Unstandardised data.** P is a correlation matrix, which implies unit-variance margins, but it was
   applied to raw returns with a standard deviation of a few percent. That makes r_j' P⁻¹ r_j close to 0, so the
   tail term had almost no influence. The maximiser was driven by the ν-only constant terms and tended to land
   on a boundary of the search range.
4. **Wrong density.** It was (an attempt at) the multivariate t *density* of the returns, not the
   t *copula* density. Fitting a copula's ν means modelling only the dependence structure: the margins have
   to be removed, both by transforming the data to uniforms and by dividing out the marginal densities.
   Otherwise the margins' own tail heaviness leaks into ν.

**Corrected formula (canonical maximum likelihood).** Each asset's returns are mapped to pseudo-observations
through their empirical CDF, u_ij = rank(r_ij) / (T+1), then to t quantiles z_ij = t_ν⁻¹(u_ij). The t copula
log-likelihood is the multivariate t log-density minus the univariate t log-densities:

    L(ν) = Σ_j [ ln Γ((ν+n)/2) + (n−1)·ln Γ(ν/2) − n·ln Γ((ν+1)/2) − ½·ln|P|
                 − ((ν+n)/2)·ln(1 + z_j' P⁻¹ z_j / ν) + ((ν+1)/2)·Σ_i ln(1 + z_ij² / ν) ]

The π terms cancel. The implementation agrees with `scipy.stats.multivariate_t.logpdf − scipy.stats.t.logpdf`
to about 1e-13. On simulated t-copula data it recovers the true ν (4 → 4, 6 → 6, 9 → 8).

A maximiser at the upper bound (ν = 11) means the data are close to Gaussian dependence; consider
widening the search range in that case.

Hazard rates, CDS spreads, LIBOR/OIS rates, recovery (40%) and the number of simulations (1024, a power
of 2 so the Sobol points stay balanced) are hard-coded in `cdsBasket.__init__`.

Optional outputs (commented out in `main()`):

| Method | Output |
| --- | --- |
| `runSensitivities()` | Spreads under parallel correlation shifts of −10.5% to +9.5%. Saves `gaussCorrSensi.png` and `tCorrSensi.png`. |
| `runTauHist()` | Histograms of default times (`histGaussTau*.png`, `histTTau*.png`). |
| `runTauScatter()` | Pairwise scatter plots of default times (`scatGaussTau*.png`, `scatTTau*.png`). |
| `createLatexTables()` | LaTeX tables of the input spreads and the correlation matrix. |
| `insertGraphsLatex()` | LaTeX `\includegraphics` lines for the generated plots. |

## Licence

Licensed under the [Apache License, Version 2.0](LICENSE).
