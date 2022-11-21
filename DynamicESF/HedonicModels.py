import numpy as np
from scipy import sparse


from . import DynamicLinearModel, MoranI


class _baseOLS(object):
    def __init__(
        self, Ys, Xs, sites=None, MClarge=None, autoL=True, ev_ratio=0.25, L=25
    ):
        self.Ys = Ys
        self.Xs = Xs

        self.sites = sites
        self.MClarge = MClarge
        self.autoL = bool(autoL)
        self.L = int(L)
        if sites is not None:
            assert isinstance(MClarge, MoranI.MClarge)
            if self.autoL:
                # Helbich and Griffith (2016, CEUS) (ev_ratio=0.25)
                lam_max = self.MClarge.lam_full.max()
                self.L = np.count_nonzero(self.MClarge.lam_full / lam_max > ev_ratio)

        self.T = len(self.Ys)
        assert self.T == len(self.Xs)

        self.Ns = [len(Y) for Y in self.Ys]
        self.D = self.Xs[0].shape[-1]

        if self.sites is not None:
            assert self.T == len(self.sites)
            self.EV = [None] * self.T
            for t, site in enumerate(self.sites):
                self.EV[t] = self.MClarge.getEV_full(site)[:, : self.L]

    def _lstsq(self, y, X):
        # solve Ax = b
        A = X.T @ X
        b = X.T @ y
        return np.linalg.lstsq(A, b)[0]

    def _stack_EV(self, Xs):
        Xs_ev = []
        for X, site, EV in zip(Xs, self.sites, self.EV):
            Xs_ev.append(np.concatenate([X, EV], axis=1))
        return Xs_ev

    def predict_inner(self, t, X, sites=None):
        pred = X @ self.beta[t]
        if sites is not None:
            ev = self.MClarge.getEV_full(sites)[:, : self.L]
            pred += ev @ self.gamma[t]
        return pred


class SeparateHedonic(_baseOLS):
    def __init__(
        self, Ys, Xs, sites=None, MClarge=None, autoL=True, ev_ratio=0.25, L=25
    ):
        super().__init__(Ys, Xs, sites, MClarge, autoL, ev_ratio, L)
        self.setup()

    def fit(self):
        if self.sites is None:
            Xs = self.Xs
        else:
            Xs = self._stack_EV(self.Xs)

        self.beta = np.array([self._lstsq(y, X) for y, X in zip(self.Ys, Xs)])

        if self.sites is not None:
            self.gamma = self.beta[:, -self.L :]
            self.beta = self.beta[:, : -self.L]

        if self.lam_HP_filter is not None:
            self.beta_HP = np.empty_like(self.beta)
            for i in range(self.D):
                self.beta_HP[:, i] = self.HPfilter(self.beta[:, i], self.lam_HP_filter)

            if self.sites is not None:
                self.gamma_HP = np.empty_like(self.gamma)
                for i in range(self.L):
                    self.gamma_HP[:, i] = self.HPfilter(
                        self.gamma[:, i], self.lam_HP_filter
                    )

    def setup(self, lam_HP_filter="monthly"):
        if type(lam_HP_filter) in [int, float]:
            if lam_HP_filter >= 0:
                self.lam_HP_filter = lam_HP_filter
            else:
                raise ValueError("lam should be non-negative")
        elif lam_HP_filter == "quarterly":
            self.lam_HP_filter = 1600
        elif lam_HP_filter == "annual":
            self.lam_HP_filter = 1600 / 4 ** 4
        elif lam_HP_filter == "monthly":
            self.lam_HP_filter = 1600 / 3 ** 4
        else:
            raise ValueError("unknown parameter")

    def HPfilter(self, x, lam):
        # ref. statsmodels implementation
        offsets = np.array([0.0, 1.0, 2.0])
        data = np.repeat([[1.0], [-2.0], [1.0]], self.T, axis=1)
        K = sparse.dia_matrix((data, offsets), shape=(self.T - 2, self.T))
        A = sparse.eye(self.T, self.T) + lam * K.T @ K
        return sparse.linalg.spsolve(A, x)


class RollingHedonic(_baseOLS):
    def __init__(
        self, Ys, Xs, sites=None, MClarge=None, autoL=True, ev_ratio=0.25, L=25
    ):
        super().__init__(Ys, Xs, sites, MClarge, autoL, ev_ratio, L)
        self.setup()

    def setup(self, window_size=[6, 6]):
        assert type(window_size[0]) is int and window_size[0] >= 0
        assert type(window_size[1]) is int and window_size[1] >= 0
        self.window_size = window_size

    def fit(self):
        if self.sites is not None:
            Xs = self._stack_EV(self.Xs)
        else:
            Xs = self.Xs

        self.beta = np.empty((self.T, Xs[0].shape[-1]))
        for t in range(self.T):
            s = max(0, t - int(self.window_size[0]))
            e = min(self.T, t + int(self.window_size[1]) + 1)
            y = np.concatenate(self.Ys[s:e], axis=0)
            X = np.concatenate(Xs[s:e], axis=0)
            self.beta[t] = self._lstsq(y, X)

        if self.sites is not None:
            self.gamma = self.beta[:, -self.L :]
            self.beta = self.beta[:, : -self.L]


class DynamicHedonic(_baseOLS):
    def __init__(
        self,
        Ys,
        Xs,
        sites=None,
        MClarge=None,
        autoL=True,
        ev_ratio=0.25,
        L=25,
        weights=None,
    ):
        super().__init__(Ys, Xs, sites, MClarge, autoL, ev_ratio, L)
        self.setup(weights=weights)

    def setup(
        self,
        randomwalk=True,
        trend=False,
        seasonal=None,
        deterministic=False,
        ev_systems={"randomwalk": True},
        weights=None,
    ):
        endogs = self.Ys
        if self.sites is None:
            exogs = self.Xs
        else:
            exogs = [None] * self.T
            for t in range(self.T):
                exogs[t] = np.concatenate([self.Xs[t], self.EV[t]], axis=1)

        self.randomwalk = np.array(randomwalk, dtype=bool)
        if type(randomwalk) == bool:
            self.randomwalk = np.tile(self.randomwalk, self.D)

        self.trend = np.array(trend, dtype=bool)
        if type(trend) == bool:
            self.trend = np.tile(self.trend, self.D)

        if seasonal is None:
            seasonal = 0
        self.seasonal = np.array(seasonal, dtype=int)
        if type(seasonal) is int:
            self.seasonal = np.tile(self.seasonal, self.D)

        self.deterministic = np.array(deterministic, dtype=bool)
        if type(deterministic) == bool:
            self.deterministic = np.tile(self.deterministic, self.D)

        self.ev_systems = ev_systems
        if self.sites is not None:

            def joint(sys, b):
                if type(b) is bool:
                    return np.concatenate([sys, np.ones(self.L, dtype=bool) * b])
                elif type(b) is int:
                    return np.concatenate([sys, np.ones(self.L) * b])

            if "randomwalk" in ev_systems.keys():
                self.randomwalk = joint(self.randomwalk, ev_systems["randomwalk"])
            else:
                self.randomwalk = joint(self.randomwalk, False)

            if "trend" in ev_systems.keys():
                self.trend = joint(self.trend, ev_systems["trend"])
            else:
                self.trend = joint(self.trend, False)

            if "seasonal" in ev_systems.keys():
                self.seasonal = joint(self.seasonal, ev_systems["seasonal"])
            else:
                self.seasonal = joint(self.seasonal, 0)

            if "deterministic" in ev_systems.keys():
                self.deterministic = joint(
                    self.deterministic, ev_systems=["deterministic"]
                )
            else:
                self.deterministic = joint(self.deterministic, False)

        self.DLM = DynamicLinearModel.DynamicLinearRegression(
            endogs=endogs,
            exogs=exogs,
            randomwalk=self.randomwalk,
            trend=self.trend,
            seasonal=self.seasonal,
            deterministic=self.deterministic,
            weights=weights,
        )
        if self.sites is not None:
            p0 = np.diag(self.DLM.P0)
            p0.flags.writeable = True
            for key, val in self.ev_systems.items():
                if key == "randomwalk" and val:
                    p0[self.DLM._rwalk_idx[-self.L :]] = self.MClarge.lam_full[: self.L]
                if key == "trend" and val:
                    p0[self.DLM._trend_idx[-self.L :]] = self.MClarge.lam_full[: self.L]
                if key == "seasonal" and val:
                    p0[self.DLM._season_idx[-self.L :]] = self.MClarge.lam_full[
                        : self.L
                    ]
            self.DLM.P0 = np.diag(p0)

    def fit(self, maxiter, loss_record=False, Q_record=True, miniter=10, tol=1e-3):
        self.DLM.fit(maxiter, loss_record, Q_record, miniter, tol)

        if self.sites is None:
            self.beta = self.DLM.beta
            self.beta_std = self.DLM.beta_std
            self.beta_randomwalk = self.DLM.beta_randomwalk
            self.beta_randomwalk_std = self.DLM.beta_randomwalk_std
            self.beta_trend = self.DLM.beta_trend
            self.beta_trend_std = self.DLM.beta_trend_std
            self.beta_season = self.DLM.beta_season
            self.beta_season_std = self.DLM.beta_season_std
        else:
            self.beta = self.DLM.beta[:, : -self.L]
            self.beta_std = self.DLM.beta_std[:, : -self.L]
            self.beta_randomwalk = self.DLM.beta_randomwalk[:, : -self.L]
            self.beta_randomwalk_std = self.DLM.beta_randomwalk_std[:, : -self.L]
            self.beta_trend = self.DLM.beta_trend[:, : -self.L]
            self.beta_trend_std = self.DLM.beta_trend_std[:, : -self.L]
            self.beta_season = self.DLM.beta_season[:, : -self.L]
            self.beta_season_std = self.DLM.beta_season_std[:, : -self.L]

            self.gamma = self.DLM.beta[:, -self.L :]
            self.gamma_std = self.DLM.beta_std[:, -self.L :]
            self.gamma_randomwalk = self.DLM.beta_randomwalk[:, -self.L :]
            self.gamma_randomwalk_std = self.DLM.beta_randomwalk_std[:, -self.L :]
            self.gamma_trend = self.DLM.beta_trend[:, -self.L :]
            self.gamma_trend_std = self.DLM.beta_trend_std[:, -self.L :]
            self.gamma_season = self.DLM.beta_season[:, -self.L :]
            self.gamma_season_std = self.DLM.beta_season_std[:, -self.L :]

    def getSpatial(self, t, sites, return_var=False, return_cov=False):
        ev = self.MClarge.getEV_full(sites)[:, : self.L]
        mu, V = self.DLM.mu_[t], self.DLM.V_[t]
        mu = mu.flatten()

        gammaMat = np.zeros((self.L, self.DLM.Dx))
        for idx in [self.DLM._rwalk_idx, self.DLM._trend_idx, self.DLM._season_idx]:
            for j, i in enumerate(idx[-self.L :]):
                if i is not None:
                    gammaMat[j, i] = 1

        A = ev @ gammaMat
        # x ~ N(mu, S) -> Ax ~ N(Amu, AS(A^T))
        if return_cov:
            return A @ mu, A @ V @ A.T
        elif return_var:
            return A @ mu, np.sum((A @ V) * A, axis=1)
        else:
            return A @ mu

    def get_future_coefs(self, n_step, return_gamma=False):
        coefs = self.DLM.get_future_beta(n_step)
        if self.MClarge is not None:
            beta = coefs[: -self.L]
        else:
            beta = coefs
        if return_gamma:
            return beta, coefs[-self.L :]
        else:
            return beta

    def forecast(self, n_step, X, sites=None):
        if sites is None:
            beta = self.get_future_coefs(n_step, False)
            return (X @ beta).flatten()

        else:
            ev = self.MClarge.getEV_full(sites)[:, : self.L]
            beta, gamma = self.get_future_coefs(n_step, True)
            return (X @ beta + ev @ gamma).flatten()

    def get_future_spatial(self, n_step, sites):
        ev = self.MClarge.getEV_full(sites)[:, : self.L]
        _, gamma = self.get_future_coefs(n_step, True)
        return (ev @ gamma).flatten()


class DynamicHedonicSVC(_baseOLS):
    def __init__(self, Ys, Xs, sites, MClarge, autoL=True, ev_ratio=0.25, L=25):
        super().__init__(Ys, Xs, sites, MClarge, autoL, ev_ratio, L)
        self.setup()

    def setup(self, svc=True):
        self.svc = np.array(svc, dtype=bool)
        if self.svc.shape == ():
            self.svc = np.ones(self.D, dtype=bool) * self.svc
        elif self.svc.shape != (self.D,):
            raise ValueError("shape error: shape of svc should be (D,) or ()")

        endogs = self.Ys
        exogs = [None] * self.T
        for t in range(self.T):
            exog = self._combine_X2EV(self.Xs[t], self.EV[t])
            exogs[t] = np.copy(exog)

        self._coef_idx = []
        total = 0
        for d in range(self.D):
            if self.svc[d]:
                self._coef_idx.append([total + i for i in range(self.L + 1)])
                total += self.L + 1
            else:
                self._coef_idx.append([total])
                total += 1

        self.DLM = DynamicLinearModel.DynamicLinearRegression(
            endogs=endogs,
            exogs=exogs,
            randomwalk=True,
            trend=False,
            seasonal=0,
            deterministic=False,
        )
        p0 = np.diag(self.DLM.P0)
        p0.flags.writeable = True
        for d in range(self.D):
            if self.svc[d]:
                idx = self._coef_idx[d]
                p0[idx[1:]] = self.MClarge.lam_full[: self.L]
        self.DLM.P0 = np.diag(p0)

    def _combine_X2EV(self, X, EV):
        exog = []
        for d in range(self.D):
            x = X[:, d, None]
            if self.svc[d]:
                exog.append(np.concatenate([x, x * EV], axis=1))
            else:
                exog.append(x)
        exog = np.concatenate(exog, axis=1)
        return exog

    def fit(self, maxiter, loss_record=False, Q_record=True, miniter=10, tol=1e-3):
        self.DLM.fit(maxiter, loss_record, Q_record, miniter, tol)

        self.beta = np.empty((self.T, self.D))
        self.beta_std = np.empty_like(self.beta)
        for d in range(self.D):
            self.beta[:, d] = self.DLM.beta[:, self._coef_idx[d][0]]
            self.beta_std[:, d] = self.DLM.beta_std[:, self._coef_idx[d][0]]

        self.gamma = np.empty(self.D)
        self.gamma_ev = np.zeros((self.D, self.L))
        self.p0 = np.empty_like(self.gamma)
        self.p0_ev = np.zeros_like(self.gamma_ev)
        for d in range(self.D):
            self.gamma[d] = self.DLM.gamma[self._coef_idx[d][0]]
            if self.svc[d]:
                self.gamma_ev[d] = self.DLM.gamma[self._coef_idx[d][1:]]

            self.p0[d] = np.diag(self.DLM.P0)[self._coef_idx[d][0]]
            if self.svc[d]:
                self.p0_ev[d] = np.diag(self.DLM.P0)[self._coef_idx[d][1:]]

    def get_future_coefs(self, n_step):
        coefs = self.DLM.get_future_beta(n_step)
        beta = coefs[:, [self._coef_idx[d][0] for d in range(self.D)]]
        return beta

    def forecast(self, n_step, X, sites):
        ev = self.MClarge.getEV_full(sites)[:, : self.L]
        coefs = self.DLM.get_future_beta(n_step)
        exog = self._combine_X2EV(X, ev)
        return (exog @ coefs).flatten()

    def predict_inner(self, t, X, sites):
        ev = self.MClarge.getEV_full(sites)[:, : self.L]
        exogs = self._combine_X2EV(X, ev)
        pred = exogs @ self.DLM.beta[t]
        return pred

    def getSpatial(self, t, sites):
        N = len(sites)
        X = np.ones((N, self.DLM.Dx))
        ev = self.MClarge.getEV_full(sites)[:, : self.L]
        exog = self._combine_X2EV(X, ev)
        coefs = self.DLM.beta[t]
        beta = np.empty((N, self.D))
        for d in range(self.D):
            beta[:, d] = exog[:, self._coef_idx[d]] @ coefs[self._coef_idx[d]]
        return beta

    def get_future_spatial(self, n_step, sites):
        N = len(sites)
        X = np.ones((N, self.DLM.Dx))
        ev = self.MClarge.getEV_full(sites)[:, : self.L]
        exog = self._combine_X2EV(X, ev)
        coefs = self.DLM.get_future_beta(n_step)
        beta = np.empty((N, self.D))
        for d in range(self.D):
            beta[:, d] = exog[:, self._coef_idx[d]] @ coefs[self._coef_idx[d]]
        return beta

    def get_spatial_t(self, t, sites, include_non_spatial=True):
        N = len(sites)
        ev = self.MClarge.getEV_full(sites)[:, : self.L]
        if include_non_spatial:
            ev = np.concatenate([np.ones((N, 1)), ev], axis=1)

        mu, V = self.DLM.mu_[t], self.DLM.V_[t]
        mu = mu.flatten()

        D_exogs = np.max(self._coef_idx) + 1
        C = np.zeros((D_exogs, self.DLM.Dx))
        # coefs = C @ mu
        for idx in [self.DLM._rwalk_idx, self.DLM._trend_idx, self.DLM._season_idx]:
            for j, i in enumerate(idx):
                if i is not None:
                    C[j, i] = 1

        Bs = []
        if include_non_spatial:
            L = self.L + 1
        else:
            L = self.L
        # gamma = B @ mu = B_ @ C @ mu
        for d in range(self.D):
            if self.svc[d]:
                B = np.zeros(((L, D_exogs)))
                if include_non_spatial:
                    for j, i in enumerate(self._coef_idx[d]):
                        B[j, i] = 1
                else:
                    for j, i in enumerate(self._coef_idx[d][1:]):
                        B[j, i] = 1

                Bs.append(B @ C)
            else:
                Bs.append(None)

        # x ~ N(mu, S) -> Ax ~ N(Amu, AS(A^T))
        means = np.zeros((N, self.D))
        stds = np.zeros((N, self.D))
        pseudu_t = np.zeros((N, self.D))
        for d in range(self.D):
            if self.svc[d]:
                A = ev @ Bs[d]
                means[:, d] = A @ mu
                stds[:, d] = np.sqrt(np.sum((A @ V) * A, axis=1))
                pseudu_t[:, d] = means[:, d] / stds[:, d]

        return pseudu_t, means, stds
