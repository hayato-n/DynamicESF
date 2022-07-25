import numpy as np
from scipy import spatial
from scipy.sparse.csgraph import minimum_spanning_tree


class MC:
    def __init__(self, connectivity, site, r=None, approx=False):
        self.connectivity = connectivity(site, r, approx)
        self.N = self.connectivity.N
        if not approx:
            self._ones = np.ones((self.N, 1))
            self.M = np.eye(self.N) - self._ones @ self._ones.T / self.N

    def EigenDecomp(self):
        self.connectivity.update_C()
        self._MCM = self.M @ self.connectivity.C @ self.M

        self.lam, self.e = np.linalg.eigh(self._MCM)
        self.lam, self.e = self.lam[::-1], self.e[:, ::-1]

    def MoransI(self, z):
        self.connectivity.update_C()
        self._MCM = self.M @ self.connectivity.C @ self.M

        self.z = np.array(z)
        self.z = self.z.reshape((self.N, 1))
        denominator = float(self.N * self.z.T @ self._MCM @ self.z)
        numerator = float(
            self._ones.T @ self.connectivity.C @ self._ones * self.z.T @ self.M @ self.z
        )
        return denominator / numerator


class MClarge(MC):
    def __init__(self, connectivity, site, knots, r=None):
        super().__init__(connectivity, site, r, approx=True)
        self.MC_knot = MC(connectivity, knots, r, approx=False)
        if r is None:
            self.connectivity.update_params([self.MC_knot.connectivity.r])

        # eigen knot
        self.MC_knot.connectivity.update_C()
        self.L = self.MC_knot.N
        self._CplusL = self.MC_knot.connectivity.C + np.eye(self.L)
        self._MCM_L = self.MC_knot.M @ self._CplusL @ self.MC_knot.M
        self._lam_LI, self._e_L = np.linalg.eigh(self._MCM_L)
        self._lam_LI, self._e_L = self._lam_LI[::-1], self._e_L[:, ::-1]
        # get positive eigenvalues
        mask = self._lam_LI > 1e-8
        self._lam_LI, self._e_L = self._lam_LI[mask], self._e_L[:, mask]

        # we can get approximated eigenvalues without coordinates
        self.lam_full = self.N * self._lam_LI / self.L
        self.lam_full -= 1

        mask = self.lam_full > 1e-8
        self.lam = self.lam_full[mask]

    def EigenDecomp(self):

        # self.connectivity.update_CnL(self.MC_knot.connectivity.x)
        # correct approximation following spmoran
        # self.lam_full = (self.N + self.L) * self._lam_LI / self.L
        self.lam_full = self.N * self._lam_LI / self.L
        self.lam_full -= 1

        # self.e = (
        #     (
        #         self.connectivity.CnL
        #         - np.kron(
        #             self._ones, (self.MC_knot._ones.T @ self._CplusL) / self.MC_knot.N
        #         )
        #     )
        #     @ self.MC_knot.M
        #     @ self.MC_knot.e
        # )

        # self.e_full = (
        #     self.connectivity.CnL
        #     - np.mean(self.connectivity.CnL, axis=1, keepdims=True)
        #     - np.mean(self._CplusL, axis=0, keepdims=True)
        #     + np.mean(self._CplusL, keepdims=True)
        # ) @ self._e_L
        # self.e_full = self.e_full @ np.diag(1 / self._lam_LI)

        self.e_full = self.getEV_full()

        # get positives
        mask = self.lam_full > 1e-8
        self.lam = self.lam_full[mask]
        self.e = self.e_full[:, mask]

    def getEV_full(self, x=None):
        if x is None:
            self.connectivity.update_CnL(self.MC_knot.connectivity.x)
            CnL = self.connectivity.CnL
        else:
            CnL = self.connectivity.calc_CnL(x, self.MC_knot.connectivity.x)

        e_full = (
            CnL
            - np.mean(CnL, axis=1, keepdims=True)
            - np.mean(self._CplusL, axis=0, keepdims=True)
            + np.mean(self._CplusL, keepdims=True)
        ) @ self._e_L
        e_full = e_full @ np.diag(1 / self._lam_LI)
        return e_full


class _connectivity_base:
    def __init__(self, x):
        self.x = np.array(x)
        self.N = self.x.shape[0]

    def update_C(self):
        raise NotImplementedError()

    def update_CnL(self, site):
        self.CnL = self.calc_CnL(self.x, site)

    def calc_CnL(self, x1, x2):
        raise NotImplementedError

    def update_params(self, params):
        raise NotImplementedError()


class ExpConnectivity(_connectivity_base):
    def __init__(self, x, r=None, approx=False):
        super().__init__(x)
        self.approx = approx
        if self.approx:
            self.r = r
        else:
            self.d = spatial.distance_matrix(self.x, self.x)

            if r is None:
                r = minimum_spanning_tree(self.d).toarray().max()
            self.r = float(r)
            if r <= 0:
                raise ValueError("r must be larger than zero")

    def k(self, d):
        return np.exp(-d / self.r)

    def update_C(self):
        self.C = self.k(self.d)
        self.C[np.eye(len(self.d), dtype=bool)] = 0.0

    def calc_CnL(self, x1, x2):
        d = spatial.distance_matrix(x1, x2)
        return self.k(d)

    def update_params(self, params):
        self.r = params[0]


class GaussianConnectivity(ExpConnectivity):
    def k(self, d):
        return np.exp(-np.square(d) / self.r)

