import numpy as np

from ExplicitFD_CS import Payoff, solveExplicitFD


def explicitFdBin(s: float, e: float, r: float, d: float, sigma: float, tFin: float, t: float,
                  oType: str, nas: int) -> float:
    '''
    Binary (cash-or-nothing, payoff 1) call (oType "c") or put (oType "p") on a grid up to 1.5 * strike.
    '''
    assert oType in ("c", "p"), "oType must be 'c' or 'p'"
    assert e > 0, "strike must be positive"
    payoff: Payoff
    if oType == "c":
        payoff = lambda stock: np.where(stock >= e, 1.0, 0.0)
    else:
        payoff = lambda stock: np.where(stock >= e, 0.0, 1.0)
    return solveExplicitFD(s, r, d, sigma, tFin, t, nas, 1.5 * e, payoff)


def runParams(s: float, e: float, r: float, d: float, sigma: float, tFin: float, t: float, oType: str) -> None:
    for myNas in (100, 200, 400, 800):
        dt: float = 0.9 / (sigma**2 * myNas**2)
        myNTS: int = int((tFin - t) / dt) + 1
        myVal: float = explicitFdBin(s, e, r, d, sigma, tFin, t, oType, myNas)
        print("The FD value with NAS={} and NTS={}, is={}".format(myNas, myNTS, myVal))


if __name__ == "__main__":
    runParams(100, 100, 0.05, 0.0, 0.2, 1.0, 0.0, "c")
