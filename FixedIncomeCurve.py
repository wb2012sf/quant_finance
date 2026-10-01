import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class fixedIncomeCurve:

    def __init__(self, periods: int, deltaTime: FloatArray, oisFwdrates: FloatArray) -> None:
        assert periods > 0, "periods must be positive"
        assert len(deltaTime) >= periods and np.all(deltaTime > 0), "need a positive deltaTime per period"
        assert len(oisFwdrates) >= periods, "need an OIS rate per period"
        self.periods: int = periods
        self.deltaTime: FloatArray = deltaTime
        self.oisFwdrates: FloatArray = oisFwdrates
        self.discFact: FloatArray = np.zeros(self.periods)

    def calcDiscFact(self) -> FloatArray:
        '''
        Bootstrap discount factors from the OIS (par) rates:
        DF_i = (1 - r_i * sum_{k<i} delta_k * DF_k) / (1 + r_i * delta_i)
        :return: a np array with the bootstrapped discount factors
        '''
        for i in range(self.periods):
            annuity: float = float(np.sum(self.deltaTime[:i] * self.discFact[:i]))
            self.discFact[i] = (1 - self.oisFwdrates[i] * annuity) / (1 + self.oisFwdrates[i] * self.deltaTime[i])
        assert np.all((self.discFact > 0) & (self.discFact <= 1)), "bootstrapped discount factors outside (0, 1]"
        return self.discFact
