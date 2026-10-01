from collections.abc import Callable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Payoff = Callable[[FloatArray], FloatArray]


def solveExplicitFD(s: float, r: float, d: float, sigma: float, tFin: float, t: float,
                    nas: int, sMax: float, payoff: Payoff) -> float:
    '''
    Price a European option with the explicit finite difference scheme for the Black-Scholes PDE.
    :param s: spot price
    :param r: risk free rate
    :param d: dividend yield
    :param sigma: volatility
    :param tFin: expiry
    :param t: valuation time
    :param nas: number of asset steps (the grid has nas+1 nodes from 0 to sMax)
    :param sMax: upper bound of the asset grid
    :param payoff: function mapping the array of grid prices to the payoff at expiry
    :return: the option value at s (linearly interpolated between grid nodes)
    '''
    assert 0 <= s <= sMax, "spot must lie on the grid [0, sMax]"
    assert sigma > 0, "volatility must be positive"
    assert tFin > t, "expiry must be after the valuation time"
    assert nas >= 3, "need at least 3 asset steps"

    dS: float = sMax / nas
    # time step from the stability condition, then adjusted so the steps divide tFin - t exactly
    dt: float = 0.9 / (sigma**2 * nas**2)
    nts: int = int((tFin - t) / dt) + 1
    dt = (tFin - t) / nts

    stock: FloatArray = np.arange(nas + 1, dtype=np.float64) * dS
    v: FloatArray = payoff(stock).astype(float)
    assert v.shape == stock.shape, "payoff must return one value per grid node"
    sInner: FloatArray = stock[1:-1]

    for _ in range(nts):
        delta: FloatArray = (v[2:] - v[:-2]) / (2 * dS)
        gamma: FloatArray = (v[2:] - 2 * v[1:-1] + v[:-2]) / dS**2
        # derived from Black Scholes theta
        theta: FloatArray = r * v[1:-1] - 0.5 * sigma**2 * sInner**2 * gamma - (r - d) * sInner * delta

        vNew: FloatArray = np.empty_like(v)
        vNew[1:-1] = v[1:-1] - theta * dt
        # at the zero boundary
        vNew[0] = v[0] * (1 - r * dt)
        # at the assumed value max boundary with 0 gamma
        vNew[-1] = 2 * vNew[-2] - vNew[-3]
        v = vNew

    return float(np.interp(s, stock, v))


def explicitFD(s: float, e: float, r: float, d: float, sigma: float, tFin: float, t: float,
               oType: str, nas: int) -> float:
    '''
    Vanilla European call (oType "c") or put (oType "p") on a grid up to 2 * strike.
    '''
    assert oType in ("c", "p"), "oType must be 'c' or 'p'"
    assert e > 0, "strike must be positive"
    q: float = -1.0 if oType == "p" else 1.0
    payoff: Payoff = lambda stock: np.maximum(q * (stock - e), 0.0)
    return solveExplicitFD(s, r, d, sigma, tFin, t, nas, 2.0 * e, payoff)


def digitalCS(s: float, e: float, r: float, d: float, sigma: float, tFin: float, t: float,
              oType: str, nas: int, payoff: float, width: float) -> float:
    '''
    Approximate a digital option with a call spread of the given width around the strike, scaled to the payoff.
    '''
    assert 0 < width < 2 * e, "width must be positive and keep both strikes positive"
    leverage: float = float(payoff) / float(width)
    lowerCall: float = explicitFD(s, e - 0.5 * width, r, d, sigma, tFin, t, oType, nas)
    upperCall: float = explicitFD(s, e + 0.5 * width, r, d, sigma, tFin, t, oType, nas)
    return leverage * (lowerCall - upperCall)


def runParams(s: float, e: float, r: float, d: float, sigma: float, tFin: float, t: float,
              oType: str, payoff: float, width: float) -> None:
    for myNas in (100, 200, 400):
        dt: float = 0.9 / (sigma**2 * myNas**2)
        myNTS: int = int((tFin - t) / dt) + 1
        myVal: float = digitalCS(s, e, r, d, sigma, tFin, t, oType, myNas, payoff, width)
        print("The FD value with NAS={} and NTS={}, is={}".format(myNas, myNTS, myVal))


if __name__ == "__main__":
    runParams(100, 100, 0.05, 0.0, 0.2, 1.0, 0.0, "c", 1.0, 1.0)
