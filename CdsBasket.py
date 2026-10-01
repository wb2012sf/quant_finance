from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import special, stats
from scipy.stats import qmc

import FixedIncomeCurve as fi

FloatArray = npt.NDArray[np.float64]


class cdsBasket:
    """
    Calculates the fair spreads of a k-th to default CDS basket using a Gaussian
    and a Student t copula.

    Intermediate and final results are stored on the object (self.*) so they can be
    inspected after a run. Most methods therefore return nothing; the "Sets" section
    of each docstring names the attribute that is populated.
    """

    def __init__(self, cdsFile: str = 'HistCdsValues.csv') -> None:
        # Historical CDS levels: one header row, first column is the date.
        self.cdsLevels: FloatArray = np.genfromtxt(cdsFile, delimiter=",")
        assert self.cdsLevels.ndim == 2, "expected a 2D table of CDS levels in {}".format(cdsFile)
        assert self.cdsLevels.shape[0] >= 3, "need a header row and at least two data rows"
        self.cdsReturns: FloatArray = np.zeros((self.cdsLevels.shape[0] - 2, self.cdsLevels.shape[1] - 1))

        self.periods: int = 5  # number of years
        self.deltaTime: FloatArray = np.ones(self.periods)
        self.tauFloor: float = 0.25  # earliest default time used for discounting (3 months)
        self.simDays: int = self.cdsReturns.shape[0]

        self.numAssets: int = self.cdsReturns.shape[1]
        self.numSim: int = 1024  # a power of 2 keeps the Sobol points balanced
        self.sobolSeed: int = 42  # seed of the Sobol scrambling, for reproducible results

        self.underlyingNames: tuple[str, ...] = ("Thyssen Krupp", "Daimler", "Renault", "Fiat", "Siemens")

        # rows: assets, columns: years 1..5
        self.cdsSpreads: FloatArray = np.array([
            [173.00, 173.00, 173.00, 173.00, 173.00],
            [61.35, 61.35, 61.35, 61.35, 61.35],
            [112.75, 112.75, 112.75, 112.75, 112.75],
            [390.41, 390.41, 390.41, 390.41, 390.41],
            [39.90, 39.90, 39.90, 39.90, 39.90]]) / 10000.

        # rows: assets, columns: years 1..5 (only the first column is used)
        self.hazardRates: FloatArray = np.array([
            [0.0280, 0.0280, 0.0280, 0.0280, 0.0280],
            [0.0101, 0.0101, 0.0101, 0.0101, 0.0101],
            [0.0184, 0.0184, 0.0184, 0.0184, 0.0184],
            [0.0611, 0.0611, 0.0611, 0.0611, 0.0611],
            [0.0066, 0.0066, 0.0066, 0.0066, 0.0066]])

        assert self.numAssets == len(self.underlyingNames), \
            "{} has {} asset columns, expected {}".format(cdsFile, self.numAssets, len(self.underlyingNames))
        assert self.cdsSpreads.shape == (self.numAssets, self.periods)
        assert self.hazardRates.shape[0] == self.numAssets and np.all(self.hazardRates > 0)

        self.recov: float = 0.4
        assert 0 <= self.recov < 1

        # Log-likelihood per candidate degree of freedom (index = nu); unused slots stay -inf.
        self.argValArray: FloatArray = np.full(20, -np.inf)
        # Degrees of freedom of the t copula (maximum likelihood estimate)
        self.nuVal: int = 0

        self.corrMatrixOrig: FloatArray = np.zeros((self.numAssets, self.numAssets))
        self.corrMatrix: FloatArray = np.zeros((self.numAssets, self.numAssets))
        self.rawCorrMatrix: FloatArray = np.zeros((self.numAssets, self.numAssets))

        self.shiftMatrix: FloatArray = np.zeros((self.numAssets, self.numAssets))
        self.shiftSteps: int = 21
        self.spdGaussResultStore: FloatArray = np.zeros((self.shiftSteps, self.numAssets))
        self.spdTResultStore: FloatArray = np.zeros((self.shiftSteps, self.numAssets))

        self.eigenvalue: FloatArray = np.zeros(self.numAssets)
        self.chol: FloatArray = np.zeros((self.numAssets, self.numAssets))
        self.rawChol: FloatArray = np.zeros((self.numAssets, self.numAssets))

        self.iidUnif: FloatArray = np.zeros((self.numSim, self.numAssets))
        self.iidUnifMixing: FloatArray = np.zeros(self.numSim)
        self.iidNorm: FloatArray = np.zeros((self.numSim, self.numAssets))

        self.correlatedNormGauss: FloatArray = np.zeros((self.numSim, self.numAssets))
        self.correlatedUnifGauss: FloatArray = np.zeros((self.numSim, self.numAssets))
        self.rawCorrelatedNorm: FloatArray = np.zeros((self.numSim, self.numAssets))
        self.rawCorrT: FloatArray = np.zeros((self.numSim, self.numAssets))

        self.tauGaussArray: FloatArray = np.zeros((self.numSim, self.numAssets))
        self.tauTArray: FloatArray = np.zeros((self.numSim, self.numAssets))
        self.sortedTauGauss: FloatArray = np.zeros((self.numSim, self.numAssets))
        self.sortedTauT: FloatArray = np.zeros((self.numSim, self.numAssets))

        self.liborFwdRates12m: FloatArray = np.array([0.004, 0.005, 0.006, 0.0065, 0.0067, 0.0068])
        self.oisSpd: float = 0.0030
        self.oisFwdrates: FloatArray = self.liborFwdRates12m - self.oisSpd
        # none of the LIBOR rates are < oisSpd, so negatives can only come from 0 values which are not of interest
        self.oisFwdrates[self.oisFwdrates < 0] = 0

        self.discFact: FloatArray = np.zeros(self.periods)

        self.gaussDenom: FloatArray = np.zeros((self.numSim, self.numAssets))
        self.tDenom: FloatArray = np.zeros((self.numSim, self.numAssets))
        self.gaussSpreads: FloatArray = np.zeros((self.numSim, self.numAssets))
        self.tSpreads: FloatArray = np.zeros((self.numSim, self.numAssets))
        self.gaussResults: FloatArray = np.zeros(self.numAssets)
        self.tResults: FloatArray = np.zeros(self.numAssets)

        self.mixing: FloatArray = np.zeros(self.numSim)

    def calcRet(self) -> None:
        '''
        Sets cdsReturns: the simple period returns of the CDS levels (header row and date column skipped).
        '''
        levels: FloatArray = self.cdsLevels[1:, 1:]
        assert not np.isnan(levels).any(), "CDS levels contain missing or non-numeric values"
        assert np.all(levels > 0), "CDS levels must be positive"
        self.cdsReturns = levels[1:] / levels[:-1] - 1.

    def calcCorr(self, flag: int = 0) -> None:
        '''
        Sets corrMatrix: the correlation matrix of the returns.
        :param flag: if non-zero, shiftMatrix is added to the correlation matrix
        '''
        self.corrMatrixOrig = np.corrcoef(np.transpose(self.cdsReturns))
        if flag == 0:
            self.corrMatrix = self.corrMatrixOrig
        else:
            self.corrMatrix = self.corrMatrixOrig + self.shiftMatrix
        assert self.corrMatrix.shape == (self.numAssets, self.numAssets)
        assert np.allclose(self.corrMatrix, self.corrMatrix.T), "correlation matrix must be symmetric"
        assert np.allclose(np.diag(self.corrMatrix), 1.), "correlation matrix must have a unit diagonal"

    def calcEigenvalue(self) -> None:
        '''
        Sets eigenvalue: the eigenvalues of the correlation matrix (diagnostic only).
        '''
        self.eigenvalue = np.linalg.eigvalsh(self.corrMatrix)

    def createIidUnif(self) -> None:
        '''
        Sets iidUnif: numSim x numAssets iid uniform Sobol quasi-random numbers
        and iidUnifMixing: numSim iid uniform Sobol quasi-random numbers for the chi2 mixing variable.
        Both come from one scrambled Sobol sequence of dimension numAssets + 1, so the mixing variable
        uses its own dimension and is independent of the asset draws.
        '''
        sobol: qmc.Sobol = qmc.Sobol(d=self.numAssets + 1, scramble=True, seed=self.sobolSeed)
        points: FloatArray = sobol.random(self.numSim)
        self.iidUnif = points[:, :self.numAssets]
        self.iidUnifMixing = points[:, self.numAssets]
        assert self.iidUnif.shape == (self.numSim, self.numAssets)
        assert self.iidUnifMixing.shape == (self.numSim,)

    def calcIidNorm(self) -> None:
        '''
        Sets iidNorm: numSim x numAssets iid standard normal Sobol quasi-random numbers.
        '''
        self.iidNorm = stats.norm.ppf(self.iidUnif)

    def calcCorrelatedNormGauss(self) -> None:
        '''
        Sets correlatedNormGauss: iidNorm correlated via the Cholesky factor of corrMatrix.
        '''
        self.chol = np.linalg.cholesky(self.corrMatrix)
        self.correlatedNormGauss = np.transpose(np.dot(self.chol, np.transpose(self.iidNorm)))

    def calcCorrelatedUnifGauss(self) -> None:
        '''
        Sets correlatedUnifGauss: correlated uniforms obtained by applying the normal CDF.
        '''
        self.correlatedUnifGauss = stats.norm.cdf(self.correlatedNormGauss)

    def calcTau(self, correlatedUnif: FloatArray) -> tuple[FloatArray, FloatArray]:
        '''
        Map correlated uniforms to exponentially distributed default times.
        :param correlatedUnif: numSim x numAssets array of correlated uniforms
        :return: (tau, sortedTau), sortedTau sorted per simulation
        '''
        assert correlatedUnif.shape == (self.numSim, self.numAssets)
        assert np.all((correlatedUnif >= 0) & (correlatedUnif <= 1)), "uniforms must lie in [0, 1]"
        tau: FloatArray = np.asarray(stats.expon.ppf(correlatedUnif, loc=0, scale=1. / self.hazardRates[:, 0]),
                                     dtype=np.float64)
        return tau, np.sort(tau, axis=1)

    def calcTauGauss(self) -> None:
        '''
        Sets tauGaussArray and sortedTauGauss: default times under the Gaussian copula.
        '''
        self.tauGaussArray, self.sortedTauGauss = self.calcTau(self.correlatedUnifGauss)

    def argMaxFct(self) -> None:
        '''
        Find the degrees of freedom for the t copula by maximising the t copula log-likelihood over nu in [3, 11]
        (canonical maximum likelihood, see README). Sets argValArray (log-likelihood per nu) and nuVal
        (the maximiser).
        '''
        n: int = self.numAssets
        assert self.simDays > n, "need more return observations than assets to fit the t copula"
        bMatrix: FloatArray = np.linalg.inv(self.corrMatrix)
        sign, logDet = np.linalg.slogdet(self.corrMatrix)
        assert sign > 0, "correlation matrix must be positive definite"
        # pseudo-observations: uniforms from the empirical CDF of each asset's returns
        pseudoUnif: FloatArray = stats.rankdata(self.cdsReturns, axis=0) / (self.simDays + 1.)

        for nu in range(3, 12):  # candidate degrees of freedom 3..11
            z: FloatArray = np.asarray(stats.t.ppf(pseudoUnif, nu), dtype=np.float64)
            # quadratic form z_j' B z_j for every observation j
            quadrForm: FloatArray = np.einsum('ij,jk,ik->i', z, bMatrix, z)
            argJ: FloatArray = special.gammaln((nu + n) / 2.) + (n - 1) * special.gammaln(nu / 2.) \
                - n * special.gammaln((nu + 1) / 2.) - 0.5 * logDet \
                - (nu + n) / 2. * np.log1p(quadrForm / nu) \
                + (nu + 1) / 2. * np.sum(np.log1p(z**2 / nu), axis=1)
            self.argValArray[nu] = np.sum(argJ)

        assert np.all(np.isfinite(self.argValArray[3:12])), "non-finite t copula log-likelihood"
        self.nuVal = int(np.argmax(self.argValArray))

    def calcRawCorrel(self) -> None:
        '''
        Sets rawCorrMatrix (a copy of corrMatrix) and rawChol, its Cholesky decomposition.
        '''
        self.rawCorrMatrix = self.corrMatrix
        self.rawChol = np.linalg.cholesky(self.rawCorrMatrix)

    def calcMixing(self) -> None:
        '''
        Sets mixing: nu / chi2(nu) mixing variable for each simulation.
        '''
        assert self.nuVal > 0, "run argMaxFct before calcMixing"
        self.mixing = self.nuVal / stats.chi2.ppf(self.iidUnifMixing, self.nuVal)

    def calcRawCorrelatedNorm(self) -> None:
        '''
        Sets rawCorrelatedNorm: correlated normals scaled by the square root of the mixing variable.
        '''
        self.rawCorrelatedNorm = np.transpose(np.sqrt(self.mixing) * np.dot(self.rawChol, np.transpose(self.iidNorm)))

    def calcRawCorrelatedT(self) -> None:
        '''
        Sets rawCorrT: correlated uniforms obtained by applying the t CDF.
        '''
        assert self.nuVal > 0, "run argMaxFct before calcRawCorrelatedT"
        self.rawCorrT = stats.t.cdf(self.rawCorrelatedNorm, self.nuVal)

    def calcTauT(self) -> None:
        '''
        Sets tauTArray and sortedTauT: default times under the t copula.
        '''
        self.tauTArray, self.sortedTauT = self.calcTau(self.rawCorrT)

    def getDiscFactInt(self, tau: float) -> float:
        '''
        Discount factor at tau, with the zero rate linearly interpolated between floor(tau) and ceil(tau).
        :param tau: the time of default
        :return: the discount factor exp(-r(tau) * tau)
        '''
        assert 0 <= tau <= self.periods, "tau={} outside the curve [0, {}]".format(tau, self.periods)
        if np.ceil(tau) == 0:
            return 1.
        tauLow: float = float(np.floor(tau))
        tauHigh: float = float(np.ceil(tau))
        rEarlier: float
        if tauLow == 0:
            # we assume that the hypothetical instantaneous spot rate is equal to the 1y OIS rate
            rEarlier = float(self.oisFwdrates[0])
        else:
            rEarlier = float(-np.log(self.discFact[int(tauLow) - 1]) / tauLow)
        rLater: float = float(-np.log(self.discFact[int(tauHigh) - 1]) / tauHigh)
        if tauHigh == tauLow:
            return float(np.exp(-rLater * tau))
        rInterp: float = rEarlier + (rLater - rEarlier) * (tau - tauLow) / (tauHigh - tauLow)
        return float(np.exp(-rInterp * tau))

    def calcSpreads(self, sortedTau: FloatArray) -> tuple[FloatArray, FloatArray]:
        '''
        Calculate the fair k-th to default spreads for every simulation.
        :param sortedTau: numSim x numAssets array of default times, sorted per simulation
        :return: (spreads, denominators), both numSim x numAssets
        '''
        assert sortedTau.shape == (self.numSim, self.numAssets)
        assert np.all(np.diff(sortedTau, axis=1) >= 0), "default times must be sorted per simulation"
        assert np.all(self.discFact > 0), "run calcDiscFact before calcSpreads"
        spreads: FloatArray = np.zeros((self.numSim, self.numAssets))
        denom: FloatArray = np.zeros((self.numSim, self.numAssets))
        for i in range(self.numSim):
            for j in range(self.numAssets):
                # defaults after maturity pay nothing (and, being sorted, neither do all later ones)
                if sortedTau[i, j] >= self.periods:
                    continue
                theTau: float = max(self.tauFloor, float(sortedTau[i, j]))
                tauInt: int = int(np.floor(theTau))
                tauFrac: float = theTau - tauInt

                tauDF: float = self.getDiscFactInt(theTau)
                numerator: float = (1 - self.recov) * tauDF / self.numAssets
                prevJDenomSum: float = denom[i, j - 1] if j > 0 else 0.
                # discretise the premium payments
                currentJDenomSum: float = float(np.sum(self.discFact[:tauInt]) * self.deltaTime[0])
                denom[i, j] = prevJDenomSum + (currentJDenomSum + tauDF * tauFrac) * (self.numAssets - j) / self.numAssets
                spreads[i, j] = numerator / denom[i, j]
        return spreads, denom

    def calcGaussSpreads(self) -> None:
        '''
        Sets gaussSpreads (per simulation) and gaussResults (mean): fair spreads under the Gaussian copula.
        '''
        self.gaussSpreads, self.gaussDenom = self.calcSpreads(self.sortedTauGauss)
        self.gaussResults = np.mean(self.gaussSpreads, axis=0)

    def calcTSpreads(self) -> None:
        '''
        Sets tSpreads (per simulation) and tResults (mean): fair spreads under the t copula.
        '''
        self.tSpreads, self.tDenom = self.calcSpreads(self.sortedTauT)
        self.tResults = np.mean(self.tSpreads, axis=0)

    def printResult(self) -> None:
        '''
        Print the fair spreads (lines end in LaTeX line breaks for pasting into the report).
        '''
        for i in range(self.numAssets):
            print("The fair spread for the asset {} with a Gaussian copula is: {:.2f}bp\\\\"
                  .format(i + 1, self.gaussResults[i] * 10000))
        print("The sum of spreads with Gaussian copula is {:.2f}bp\\\\".format(np.sum(self.gaussResults) * 10000))

        for i in range(self.numAssets):
            print("The fair spread for the asset {} with a t copula is: {:.2f}bp \\\\"
                  .format(i + 1, self.tResults[i] * 10000))
        print("The sum of spreads with t copula is {:.2f}bp \\\\".format(np.sum(self.tResults) * 10000))

        print("----")
        print("The sum of spreads in the original CDSs is {:.2f} bp".format(np.sum(self.cdsSpreads[:, 4]) * 10000))
        print("----")

    def printLatexTable(self, header: str, rowLabels: Iterable[object], rows: Sequence[Sequence[str]]) -> None:
        '''
        Print a LaTeX tabular with one column per underlying.
        :param header: text of the top-left header cell
        :param rowLabels: label of each row
        :param rows: the formatted cell values of each row
        '''
        assert all(len(row) == self.numAssets for row in rows), "each row needs one cell per underlying"
        print("\\begin{tabular}{|c" + "|c" * self.numAssets + "|}")
        print("\\hline")
        print("\\hline")
        print("{} {} \\\\ [0.5ex]".format(header, "".join(" & " + name for name in self.underlyingNames)))
        print("\\hline\\hline")
        for label, row in zip(rowLabels, rows, strict=True):
            print(str(label) + "".join(" & " + cell for cell in row) + "\\\\")
            print("\\hline")
        print("\\hline")
        print("\\end{tabular}")

    def createLatexTables(self) -> None:
        '''
        Print LaTeX tables of the CDS spreads and the correlation matrix.
        '''
        numYears: int = self.cdsSpreads.shape[1]
        self.printLatexTable("Years", range(1, numYears + 1),
                             [["{:.2f}".format(self.cdsSpreads[j, i] * 10000) for j in range(self.numAssets)]
                              for i in range(numYears)])
        print("-----------")
        self.printLatexTable(" correlations ", self.underlyingNames,
                             [["{:.2f}".format(c) for c in row] for row in self.corrMatrix])

    def runSensitivities(self) -> None:
        '''
        Calculate the sensitivity of the Gaussian and t spreads to parallel shifts in correlation and plot them.
        '''
        sns.set_style("darkgrid")

        for h in range(self.shiftSteps):
            self.shiftMatrix.fill((-self.shiftSteps / 2. + h) / 100.)
            np.fill_diagonal(self.shiftMatrix, 0.)

            self.runFunc(1)
            self.spdGaussResultStore[h, :] = self.gaussResults * 10000
            self.spdTResultStore[h, :] = self.tResults * 10000
        print(self.spdGaussResultStore)

        xAxisLabels: FloatArray = np.arange(-self.shiftSteps / 2., self.shiftSteps / 2., 1)
        labels: list[str] = ["Spd Asset " + str(i + 1) for i in range(self.numAssets)]

        for results, copula, fileName in ((self.spdGaussResultStore, "Gaussian", "gaussCorrSensi"),
                                          (self.spdTResultStore, "t", "tCorrSensi")):
            plt.plot(xAxisLabels, results, label=labels)
            plt.suptitle("Spreads with {} copula".format(copula))
            plt.xlabel('Parallel Shift in Correlation[%]')
            plt.ylabel('Spread[bp]')
            plt.xlim(-self.shiftSteps / 2., self.shiftSteps / 2.)
            plt.legend()
            plt.savefig(fileName, dpi=100)
            plt.show()
            plt.clf()

    def runTauHist(self) -> None:
        '''
        Save histograms of the default times, for all assets together and for each asset separately.
        '''
        numBins: int = 25
        sns.set_style("darkgrid")

        for tauArray, copula, prefix in ((self.tauGaussArray, "Gaussian", "histGaussTau"),
                                         (self.tauTArray, "t", "histTTau")):
            plt.hist(tauArray, label=["tau" + str(i + 1) for i in range(self.numAssets)], bins=numBins)
            plt.xlabel('years')
            plt.ylabel('number')
            plt.suptitle("Histogram {} tau".format(copula))
            plt.legend()
            plt.savefig(prefix + "All", dpi=100)
            plt.clf()

            for i in range(self.numAssets):
                plt.hist(tauArray[:, i], label="tau {} {}".format(copula, i + 1), bins=numBins * 4)
                plt.xlabel('years')
                plt.ylabel('number')
                plt.suptitle("Histogram {} tau{}".format(copula, i + 1))
                plt.legend()
                plt.savefig(prefix + str(i + 1), dpi=100)
                plt.clf()

    def runTauScatter(self) -> None:
        '''
        Save pairwise scatter plots of the default times.
        '''
        sns.set_style("darkgrid")
        for tauArray, copula, prefix in ((self.tauGaussArray, "Gaussian", "scatGaussTau"),
                                         (self.tauTArray, "t", "scatTTau")):
            for i in range(self.numAssets):
                for j in range(i + 1, self.numAssets):
                    plt.plot(tauArray[:, i], tauArray[:, j], ".")
                    plt.xlabel('tau' + str(i + 1) + " [y]")
                    plt.ylabel('tau' + str(j + 1) + " [y]")
                    plt.suptitle("Scatterplot {} tau {} vs tau {}".format(copula, i + 1, j + 1))
                    plt.savefig(prefix + str(i + 1) + "Tau" + str(j + 1), dpi=100)
                    plt.clf()

    def insertGraphsLatex(self) -> None:
        '''
        Print LaTeX code to include the graphs produced by runTauScatter and runTauHist.
        '''
        template: str = "\\includegraphics[scale=0.8]{../CdsBasket/images/%s}"
        for prefix in ("scatGaussTau", "scatTTau"):
            for i in range(self.numAssets):
                for j in range(i + 1, self.numAssets):
                    print(template % (prefix + str(i + 1) + "Tau" + str(j + 1)))
        for prefix in ("histGaussTau", "histTTau"):
            for i in range(self.numAssets):
                print(template % (prefix + str(i + 1)))

    def runFunc(self, flag: int = 0) -> None:
        '''
        Run the full pricing in order and print the results.
        :param flag: passed to calcCorr; non-zero applies shiftMatrix to the correlations
        '''
        myFi: fi.fixedIncomeCurve = fi.fixedIncomeCurve(self.periods, self.deltaTime, self.oisFwdrates)
        self.discFact = myFi.calcDiscFact()

        self.calcRet()
        self.calcCorr(flag)
        self.calcEigenvalue()

        self.createIidUnif()
        self.calcIidNorm()
        self.calcCorrelatedNormGauss()
        self.calcCorrelatedUnifGauss()

        self.argMaxFct()
        self.calcRawCorrel()
        self.calcMixing()
        self.calcRawCorrelatedNorm()
        self.calcRawCorrelatedT()
        self.calcTauGauss()
        self.calcTauT()
        self.calcGaussSpreads()
        self.calcTSpreads()

        self.printResult()


def main() -> None:
    cb: cdsBasket = cdsBasket()
    cb.runFunc()
    # Optional outputs for the report:
    # cb.runSensitivities()
    # cb.createLatexTables()
    # cb.runTauScatter()
    # cb.runTauHist()
    # cb.insertGraphsLatex()


if __name__ == "__main__":
    main()
