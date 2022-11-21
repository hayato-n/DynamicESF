import numpy as np
from scipy import linalg
from tqdm import tqdm


class DiagonalDLS:
    def __init__(self, Y, A, C, gamma, sigma2, learn_gamma, mu0, P0, weights=None):
        """Dynamic Linear Model with Diagonal Covariances

        Parameters
        ----------
        Y : array-like, (T, Dyt, 1)
            observed sequence
        A : array-like, (T, Dx, Dx)
            coefficient matrices of latent process (system model)
        C : array-like, (T, Dyt, Dx)
            coefficient matrices of observation model
        gamma : array-like, (Dx,)
            diagonal vector of system covariance
        sigma2 : float
            variance of observation model
        learn_gamma : array-like, boolean, (Dx,)
            if True, correspoinding element of gamma is estimated with EM algorithm
        mu0 : array-like, (Dx, 1)
            mean of initial prior
        P0 : array-like, (Dx, Dx)
            covariance matrix of initial prior
        weights: None or array-like (same shape with Y)
            weights on observations. The observation covariance matrix is set to be sigma2 * diag(1/weights).
            All elements must be positive
            If None, all weights are set to be one
        """

        self.Y = np.array(Y, dtype=object)
        self.T = len(Y)
        self.Dy = np.array([len(y) for y in self.Y], dtype=int)

        self.A = np.array(A)
        self.Dx = self.A.shape[1]

        self.C = np.array(C)

        self.gamma = np.array(gamma, dtype=float)
        self.sigma2 = float(sigma2)
        self.learn_gamma = np.array(learn_gamma, dtype=bool)

        self.mu0 = np.array(mu0, dtype=float).reshape((self.Dx, 1))
        self.P0 = np.array(P0, dtype=float)

        self.mode_filter = "direct"

        if weights is None:
            self.weights = np.array(
                [np.ones_like(self.Y[t], dtype=float) for t in range(self.T)],
                dtype=object,
            )
        else:
            try:
                w = [weights[t].reshape(self.Y[t].shape) for t in range(self.T)]
            except:
                raise ValueError("weights must be the same shape to Y")

            if not np.all([np.all(weights[t] > 0) for t in range(self.T)]):
                raise ValueError("All weights must be positive")

            self.weights = np.array(w, dtype=object)

    def EMalgorithm(
        self, maxiter, loss_record=False, Q_record=True, miniter=10, tol=1e-5
    ):
        if loss_record:
            self.loss = []
        if Q_record:
            self.Q = []

        if not (loss_record or Q_record):
            print("loss or Q should be recorded.")

        params = np.concatenate(
            [np.diag(self.P0), self.gamma[self.learn_gamma], np.array([self.sigma2])]
        )
        self.soc1 = []
        self.soc2 = []
        with tqdm(total=maxiter) as pbar:
            for i in range(maxiter):
                self.Estep(eval=loss_record)
                self.Mstep()
                if loss_record:
                    self.loss.append(-np.sum(self.loglik))
                    if not Q_record:
                        pbar.set_description(desc="loss={:.3f}".format(self.loss[i]))
                        if i > 1:
                            self.soc2.append(
                                np.abs(-self.loss[-2] + self.loss[-1])
                                / (1 + np.abs(self.loss[-1]))
                            )
                if Q_record:
                    self.Q.append(self._calc_Q())
                    pbar.set_description(desc="Q={:.3f}".format(self.Q[i]))
                    if i > 1:
                        self.soc2.append(
                            np.abs(self.Q[-2] - self.Q[-1]) / (1 + np.abs(self.Q[-1]))
                        )

                new_params = np.concatenate(
                    [
                        np.diag(self.P0),
                        self.gamma[self.learn_gamma],
                        np.array([self.sigma2]),
                    ]
                )
                self.soc1.append(
                    np.linalg.norm(new_params - params)
                    / (1 + np.linalg.norm(new_params))
                )

                if i > miniter and self.soc1[-1] < np.sqrt(tol) and self.soc2[-1] < tol:
                    break
                else:
                    params = np.copy(new_params)

                pbar.update(1)

    def KalmanFilter(self, eval=False):
        self.forward(eval=eval)
        self.backward()

    def Estep(self, eval=False):
        self.KalmanFilter(eval=eval)
        # mean fields
        self._x = np.copy(self.mu_)
        self._xx_1T = self.V_[1:] @ self.J.transpose((0, 2, 1)) + self.mu_[
            1:
        ] @ self.mu_[:-1].transpose((0, 2, 1))
        self._xxT = self.V_ + self.mu_ @ self.mu_.transpose((0, 2, 1))

    def Mstep(self):
        self.update_mu0()
        self.update_P0()
        self.update_gamma()
        self.update_sigma2()

    def forward(self, eval=False):
        self.mu = np.empty((self.T, self.Dx, 1), dtype=float)
        self.V = np.empty((self.T, self.Dx, self.Dx), dtype=float)
        self.P = np.empty((self.T, self.Dx, self.Dx), dtype=float)

        Gamma = np.diag(self.gamma)
        Sigma = [np.diag(self.sigma2 / w.flatten().astype(float)) for w in self.weights]
        if self.mode_filter == "direct":
            filtering = self.filtering_direct
            eval_lik = self.eval_lik_direct
        else:
            filtering = self.filtering
            eval_lik = self.eval_lik

        t = 0
        self.P[t - 1] = self.P0
        self.mu[t], self.V[t] = filtering(
            y=self.Y[t],
            pred_mean=self.mu0,
            P=self.P[t - 1],
            C=self.C[t],
            Sigma=Sigma[t],
        )

        for t in range(1, self.T):
            pred_mean, self.P[t - 1] = self.predict(
                mu=self.mu[t - 1], V=self.V[t - 1], A=self.A[t], Gamma=Gamma
            )
            self.mu[t], self.V[t] = filtering(
                y=self.Y[t],
                pred_mean=pred_mean,
                P=self.P[t - 1],
                C=self.C[t],
                Sigma=Sigma[t],
            )

        if eval:
            self.loglik = np.empty((self.T), dtype=float)
            t = 0
            self.loglik[t] = eval_lik(
                y=self.Y[t],
                pred_mean=self.mu0,
                P=self.P[t - 1],
                C=self.C[t],
                Sigma=Sigma[t],
            )
            for t in range(1, self.T):
                self.loglik[t] = eval_lik(
                    self.Y[t],
                    pred_mean=self.A[t] @ self.mu[t - 1],
                    P=self.P[t - 1],
                    C=self.C[t],
                    Sigma=Sigma[t],
                )

    def backward(self):
        self.mu_ = np.empty((self.T, self.Dx, 1), dtype=float)
        self.V_ = np.empty((self.T, self.Dx, self.Dx), dtype=float)
        self.mu_[-1] = self.mu[-1]
        self.V_[-1] = self.V[-1]
        self.J = np.empty((self.T - 1, self.Dx, self.Dx), dtype=float)

        for t in reversed(range(self.T - 1)):
            self.mu_[t], self.V_[t], self.J[t] = self.smoothing(
                mu=self.mu[t],
                V=self.V[t],
                mu_=self.mu_[t + 1],
                V_=self.V_[t + 1],
                A=self.A[t + 1],
                P=self.P[t],
            )

    def predict(self, mu, V, A, Gamma):
        mean = A @ mu
        P = Gamma + A @ V @ A.T
        return mean, P

    def filtering(self, y, pred_mean, P, C, Sigma):
        K = P @ C.T @ self._inv(Sigma + C @ P @ C.T)  # Kalman Gain
        V_filter = (np.eye(self.Dx) - K @ C) @ P
        mu_filter = pred_mean + K @ (y - C @ pred_mean)
        return mu_filter, V_filter

    def eval_lik(self, y, pred_mean, P, C, Sigma):
        return self._logpdf_mvgauss(obs=y, mean=C @ pred_mean, cov=Sigma + C @ P @ C.T)

    def filtering_direct(self, y, pred_mean, P, C, Sigma):
        Pinv = self._inv(P)
        #         Sigmainv = np.diag(1/np.diag(Sigma)) # Sigma is assumed to be diagonal
        #         V_filter = self._inv(Pinv + C.T @ Sigmainv @ C)
        #         mu_filter = V_filter @ (C.T @ Sigmainv @ y + Pinv @ pred_mean)
        sigmainv = 1 / np.diag(Sigma)  # Sigma is assumed to be diagonal
        CtSigmainv = (sigmainv.reshape(-1, 1) * C).T
        V_filter = self._inv(Pinv + CtSigmainv @ C)
        mu_filter = V_filter @ (CtSigmainv @ y + Pinv @ pred_mean)
        return mu_filter, V_filter

    def eval_lik_direct(self, y, pred_mean, P, C, Sigma):
        Pinv = self._inv(P)

        sigmainv = 1 / np.diag(Sigma)  # Sigma is assumed to be diagonal
        CtSigmainv = (sigmainv.reshape(-1, 1) * C).T
        prec = (
            np.diag(sigmainv)
            - CtSigmainv.T @ self._inv(Pinv + CtSigmainv @ C) @ CtSigmainv
        )  # Woodbury identity

        # Sigmainv = np.diag(1 / np.diag(Sigma))  # Sigma is assumed to be diagonal
        # prec = (
        #     Sigmainv
        #     - Sigmainv @ C @ self._inv(Pinv + C.T @ Sigmainv @ C) @ C.T @ Sigmainv
        # )  # Woodbury identity

        # sigmainv = (
        #     1 / Sigma
        # )  # Sigma is assumed to be diagonal and all elements are same
        # prec = (
        #     sigmainv * np.eye(len(y))
        #     - sigmainv * C @ self._inv(Pinv + sigmainv * C.T @ C) @ C.T * sigmainv
        # )  # Woodbury identity

        return self._logpdf_mvgauss(
            obs=y, mean=C @ pred_mean, cov=None, prec=prec.astype(float)
        )

    def smoothing(self, mu, V, mu_, V_, A, P):
        J = V @ A.T @ self._inv(P)
        V_smoo = V + J @ (V_ - P) @ J.T
        mu_smoo = mu + J @ (mu_ - A @ mu)
        return mu_smoo, V_smoo, J

    def update_mu0(self):
        # self.mu0 = self._x[0]
        self.mu0 = np.zeros_like(self._x[0])

    def update_P0(self):
        # self.P0 = self._xxT[0] - self._x[0] @ self._x[0].T
        self.P0 = np.diag(np.diag(self._xxT[0]))

    def update_gamma(self):
        M = np.sum(
            self._xxT[1:]
            - self.A[1:] @ self._xx_1T.transpose((0, 2, 1))
            - self._xx_1T @ self.A[1:].transpose((0, 2, 1))
            + self.A[1:] @ self._xxT[:-1] @ self.A[1:].transpose((0, 2, 1)),
            axis=0,
        )
        for d in range(self.Dx):
            if self.learn_gamma[d]:
                self.gamma[d] = M[d, d] / (self.T - 1)

    def update_sigma2(self):
        sigma2T = 0
        for t in range(self.T):
            y = self.Y[t]
            c = self.C[t]
            x = self._x[t]
            xxT = self._xxT[t]
            # sigma2T += np.trace(y @ y.T - c @ x @ y.T - y @ x.T @ c.T + c @ xxT @ c.T)
            y_hat = c @ x
            w = self.weights[t].astype(float)
            Wy = w * y
            WC = w * c
            sigma2T += np.trace(Wy.T @ (y - 2 * y_hat)) + np.trace(xxT @ c.T @ WC)
        self.sigma2 = sigma2T / np.sum(self.Dy)

    def _calc_Q(self):
        # logp(x1)
        P0inv = self._inv(self.P0)
        Q = -0.5 * (
            self.Dx * np.log(2 * np.pi)
            + np.linalg.slogdet(self.P0)[1]
            + np.trace(self._xxT[0] @ P0inv)
            - self.mu0.T @ P0inv @ self._x[0]
            - self._x[0].T @ P0inv @ self.mu0
            + self.mu0.T @ P0inv @ self.mu0
        )

        # logp(xt|xt-1)
        Gammainv = self._inv(np.diag(self.gamma))[None]
        Q -= (
            0.5
            * (self.T - 1)
            * (self.Dx * np.log(2 * np.pi) + self._logdetdiag(self.gamma))
        )

        Q -= 0.5 * np.sum(
            np.trace(self._xxT[1:] @ Gammainv, axis1=1, axis2=2)
            - np.trace(
                self._xx_1T.transpose(0, 2, 1) @ Gammainv @ self.A[1:],
                axis1=1,
                axis2=2,
            )
            - np.trace(
                self.A[1:].transpose(0, 2, 1) @ Gammainv @ self._xx_1T,
                axis1=1,
                axis2=2,
            )
            + np.trace(
                self._xxT[:-1].transpose(0, 2, 1)
                @ self.A[1:].transpose(0, 2, 1)
                @ Gammainv
                @ self.A[1:],
                axis1=1,
                axis2=2,
            )
        )

        # logp(yt|xt)
        for t in range(self.T):
            sigmainv = self.weights[t].flatten().astype(float) / self.sigma2
            y = self.Y[t]
            C = self.C[t]
            ytSigmainv = (y.flatten() * sigmainv).reshape(y.T.shape)
            SigmainvC = sigmainv.reshape(-1, 1) * C
            Q -= 0.5 * (
                self.Dy[t] * np.log(2 * np.pi)
                + self._logdetdiag(
                    self.sigma2 * (1 / self.weights[t].flatten().astype(float))
                )
                + (
                    ytSigmainv @ y
                    - ytSigmainv @ C @ self._x[t]
                    - self._x[t].T @ C.T @ ytSigmainv.T
                    + np.trace(self._xxT[t] @ C.T @ SigmainvC)
                ).astype(float)
                # + y.T @ Sigmainv @ y
                # - y.T @ Sigmainv @ C @ self._x[t]
                # - self._x[t].T @ C.T @ Sigmainv @ y
                # + np.trace(self._xxT[t] @ C.T @ Sigmainv @ C)
            )

        return float(Q)

    def Mahalanobis_sq(self):
        """Calc squared Mahalanobis' distance to evaluate the structural change"""
        Gammainv = self._inv(np.diag(self.gamma))
        v = np.trace(
            Gammainv @ (self._xxT[1:] - 2 * self._xx_1T + self._xxT[:-1]),
            axis1=1,
            axis2=2,
        )
        return v

    def _inv(self, M):
        return np.linalg.pinv(M.astype(float))

    def _logdetdiag(self, diag):
        mask = diag != 0
        return np.sum(np.log(diag[mask]))

    def _logpdf_mvgauss(self, obs, mean, cov, prec=None):
        D = len(obs)
        obs = obs.reshape((D, 1))
        mean = mean.reshape((D, 1))
        if prec is None:
            prec = self._inv(cov)
        return (
            -0.5
            * (
                D * np.log(2 * np.pi)
                - np.linalg.slogdet(prec)[1]
                + (obs - mean).T @ prec @ (obs - mean)
            ).flatten()
        )


class DynamicLinearRegression(DiagonalDLS):
    def __init__(
        self,
        endogs,
        exogs,
        randomwalk=True,
        trend=False,
        seasonal=None,
        deterministic=False,
        weights=None,
    ):
        """Dynamic Linear Regression Model

        Parameters
        ----------
        endogs : array-like, (T, Dyt)
            time series endogenous variables
        exogs : array-like, (T, Dyt, Dx)
            time series exogenous variables (constant term should be included)
        randomwalk : bool or array-like (Dx,), optional, by default, True
            indicate dth coefficient has random walk system or not
            (if it has single element, all coefficients are indicated to be same)
        trend : bool or array-like (Dx,), optional, by default, True
            indicate dth coefficient has 2nd order trend system or not
            (if it has single element, all coefficients are indicated to be same)
        seasonal : {int, None} or array-like (Dx,), optional, by default, True
            periods of seasonal components
            (if it has single element, all coefficients are indicated to be same)
        deterministic : bool or array-like (Dx,), optional, by default, False
            indicate dth coefficient is deterministic system or not
            (if it has single element, all coefficients are indicated to be same)
        weights: None or array-like (same shape with Y)
            weights on observations. The observation covariance matrix is set to be sigma2 * diag(1/weights).
            All elements must be positive
            If None, all weights are set to be one
        """

        self.endogs = np.array(endogs, dtype=object)
        self.exogs = np.array(exogs, dtype=object)
        T = len(self.endogs)
        Dx = self.exogs[0].shape[-1]

        self.randomwalk = np.array(randomwalk, dtype=bool)
        if type(randomwalk) == bool:
            self.randomwalk = np.tile(self.randomwalk, Dx)

        self.trend = np.array(trend, dtype=bool)
        if type(trend) == bool:
            self.trend = np.tile(self.trend, Dx)

        if seasonal is None:
            seasonal = 0
        self.seasonal = np.array(seasonal, dtype=int)
        if type(seasonal) is int:
            self.seasonal = np.tile(self.seasonal, Dx)

        self.deterministic = np.array(deterministic, dtype=bool)
        if type(deterministic) == bool:
            self.deterministic = np.tile(self.deterministic, Dx)

        D_latent = 0
        A = []
        C = [[] for t in range(T)]
        gamma = []
        learn_gamma = []
        self._rwalk_idx = [None] * Dx
        self._trend_idx = [None] * Dx
        self._season_idx = [None] * Dx
        for d in range(Dx):

            if self.randomwalk[d]:
                self._rwalk_idx[d] = D_latent
                D_latent += 1
                A.append(np.eye(1))
                for t in range(T):
                    C[t].append(self.exogs[t][:, d, None])

                if self.deterministic[d]:
                    gamma.append(0)
                    learn_gamma.append(False)
                else:
                    gamma.append(1)
                    learn_gamma.append(True)

            if self.trend[d]:
                self._trend_idx[d] = D_latent
                D_latent += 2
                A.append(np.array([[2, -1], [1, 0]]))
                for t in range(T):
                    C[t].append(
                        np.stack(
                            [self.exogs[t][:, d], np.zeros_like(self.exogs[t][:, d])],
                            axis=1,
                        )
                    )
                if self.deterministic[d]:
                    gamma.extend([0, 0])
                    learn_gamma.extend([False, False])
                else:
                    gamma.extend([1, 0])
                    learn_gamma.extend([True, False])

            if self.seasonal[d] >= 2:
                self._season_idx[d] = D_latent
                p = self.seasonal[d]
                D_latent += p - 1
                sea_mat = np.eye(p - 1, k=-1, dtype=int)
                sea_mat[0, :] = -1
                A.append(sea_mat)
                for t in range(T):
                    C[t].append(np.zeros((len(self.endogs[t]), p - 1)))
                    C[t][-1][:, 0] = self.exogs[t][:, d]

                if self.deterministic[d]:
                    gamma.extend([0] * (p - 1))
                    learn_gamma.extend([False] * (p - 1))
                else:
                    gamma.append(1)
                    gamma.extend([0] * (p - 2))
                    learn_gamma.append(True)
                    learn_gamma.extend([False] * (p - 2))

        A = np.tile(linalg.block_diag(*A)[None], reps=(T, 1, 1))
        for t in range(T):
            C[t] = np.concatenate(C[t], axis=-1)
        C = np.array(C, dtype=object)
        gamma = np.array(gamma).flatten()
        learn_gamma = np.array(learn_gamma).flatten()

        super().__init__(
            Y=[y.reshape((-1, 1)) for y in self.endogs],
            A=A,
            C=C,
            gamma=gamma,
            sigma2=10,
            learn_gamma=learn_gamma,
            mu0=np.zeros(D_latent),
            P0=np.eye(D_latent) * 10,
            weights=weights,
        )

    def fit(self, maxiter, loss_record=False, Q_record=True, miniter=10, tol=1e-3):
        self.EMalgorithm(maxiter, loss_record, Q_record, miniter, tol)

        # coefficients
        self.beta_randomwalk, self.beta_randomwalk_std = self._get_coefficients(
            self.mu_, self.V_, self._rwalk_idx
        )
        self.beta_trend, self.beta_trend_std = self._get_coefficients(
            self.mu_, self.V_, self._trend_idx
        )
        self.beta_season, self.beta_season_std = self._get_coefficients(
            self.mu_, self.V_, self._season_idx
        )

        A = np.zeros((len(self._rwalk_idx), self.Dx))
        for idx in [self._rwalk_idx, self._trend_idx, self._season_idx]:
            for d, i in enumerate(idx):
                if i is None:
                    pass
                else:
                    A[d, i] = 1

        # x ~ N(mu, S) -> Ax ~ N(Amu, AS(A^T))
        self.beta = (A[None] @ self.mu_).reshape((self.T, -1))
        self.beta_std = np.sqrt([np.diag(A @ V @ A.T) for V in self.V_])

    def _get_coefficients(self, mu, V, idx):
        beta = np.zeros((len(mu), self.exogs[0].shape[-1]))
        beta_std = np.zeros_like(beta)
        for d, i in enumerate(idx):
            if i is None:
                pass
            else:
                beta[:, d] = mu[:, i, 0]
                beta_std[:, d] = np.sqrt(V[:, i, i])

        return beta, beta_std

    def forecast_oneahead(self, exogs):
        """one step ahead predction of price

        Parameters
        ----------
        exogs : array-like
            exogenous variables (N, D)

        Returns
        -------
        predictive price (N, )
        """
        return self.forecast(exogs, 1)

    def forecast(self, exogs, n_step):
        """n step ahead predction of price

        Parameters
        ----------
        exogs : array-like
            exogenous variables (N, D)
        n_step : int
            number of steps

        Returns
        -------
        predictive price (N, )
        """
        beta = self.get_future_beta(n_step)
        return (exogs @ beta).flatten()

    def get_future_beta(self, n_step):
        """n step ahead predction of coefficients

        Parameters
        ----------
        n_step : int
            number of steps

        Returns
        -------
        predictive coefficients (D, )
        """
        assert type(n_step) == int and n_step >= 1

        mu = self.mu_[-1]
        for _ in range(n_step):
            mu = self.A[-1] @ mu

        beta = np.zeros((1, self.exogs[-1].shape[-1]))
        beta += self._get_coefficients(mu[None], self.V_[0, None], self._rwalk_idx)[0]
        beta += self._get_coefficients(mu[None], self.V_[0, None], self._trend_idx)[0]
        beta += self._get_coefficients(mu[None], self.V_[0, None], self._season_idx)[0]
        beta = beta.flatten()
        return beta
