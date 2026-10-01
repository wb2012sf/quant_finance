import math as m

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# time step in years per monitoring frequency
FREQ_DT: dict[str, float] = {"a": 1.0, "s": 0.5, "q": 0.25, "m": 1.0 / 12.0, "w": 1.0 / 52.0, "d": 1.0 / 252.0}


def cnd(z: float) -> float:
    '''
    Cumulative standard normal distribution.
    '''
    return (1.0 + m.erf(float(z) / m.sqrt(2.0))) / 2.0


def binaryClosedForm(s: float, e: float, r: float, d: float, sigma: float, tFin: float, t: float,
                     optType: str) -> float:
    '''
    Black-Scholes value of a cash-or-nothing binary paying 1 if S_T >= e (optType "gt") or S_T < e (optType "lt").
    '''
    assert optType in ("gt", "lt"), "optType must be 'gt' or 'lt'"
    assert s > 0 and e > 0, "spot and strike must be positive"
    assert sigma > 0, "volatility must be positive"
    assert tFin > t, "expiry must be after the valuation time"
    tau: float = tFin - t
    d2: float = (m.log(s / e) + (r - d - 0.5 * sigma**2) * tau) / (sigma * m.sqrt(tau))
    prob: float = cnd(d2) if optType == "gt" else 1.0 - cnd(d2)
    return m.exp(-r * tau) * prob


def binaryMC(s: float, e: float, r: float, d: float, sigma: float, tFin: float, t: float,
             optType: str, paths: int, freq: str, seed: int | None = None) -> float:
    '''
    Monte Carlo value of the binary option, simulating GBM paths with the time step given by freq
    (one of the keys of FREQ_DT).
    '''
    if freq not in FREQ_DT:
        raise ValueError("freq must be one of {}".format(sorted(FREQ_DT)))
    assert optType in ("gt", "lt"), "optType must be 'gt' or 'lt'"
    assert s > 0 and e > 0, "spot and strike must be positive"
    assert sigma > 0, "volatility must be positive"
    assert tFin > t, "expiry must be after the valuation time"
    assert paths >= 1, "need at least one path"
    rng: np.random.Generator = np.random.default_rng(seed)

    nts: int = int((tFin - t) / FREQ_DT[freq]) + 1  # overall steps, not per year
    dt: float = (tFin - t) / nts  # adjust for discrepancies

    logS: FloatArray = np.full(int(paths), m.log(s))
    for _ in range(nts):
        logS += (r - d - 0.5 * sigma**2) * dt + sigma * m.sqrt(dt) * rng.standard_normal(logS.size)

    sT: FloatArray = np.exp(logS)
    hits: npt.NDArray[np.bool_] = sT >= e if optType == "gt" else sT < e
    return m.exp(-r * (tFin - t)) * float(np.mean(hits))


if __name__ == "__main__":
    myVal: float = binaryClosedForm(100, 100, 0.05, 0.0, 0.2, 1.0, 0.0, "gt")
    print("the closed form value is " + str(myVal))

    for numPaths in (1000, 10000, 100000, 1000000):
        for freq in ("a", "q", "m", "w", "d"):
            myVal = binaryMC(100, 100, 0.05, 0.0, 0.2, 1.0, 0.0, "gt", numPaths, freq)
            print("the MC value with {} paths and freq {}, is {}".format(numPaths, freq, myVal))
