import sys
import tqdm
import numpy as np
import scipy.stats
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold


def normal(X, mean, sigma):
    cov = sigma ** 2 * np.eye(len(mean))
    return scipy.stats.multivariate_normal.pdf(X, mean=mean, cov=cov)


def mutual_information(X, Y, sigma=1, b=200, maxiter=1000):
    n_x, d_x = X.shape
    n_y, d_y = Y.shape
    b = min(b, n_x)
    
    XY = np.hstack([X, Y])
    UV = KMeans(b).fit(XY).cluster_centers_
    U, V = np.split(UV, [d_x], axis=1)
    phi_x = np.array([normal(X, u, sigma) for u in U])
    phi_y = np.array([normal(Y, v, sigma) for v in V])
    phi = phi_x * phi_y
    h_x = np.sum(phi_x, axis=1)
    h_y = np.sum(phi_y, axis=1)
    h_xy = np.sum(phi, axis=1)
    h = (h_x * h_y - h_xy) / (n_x ** 2 - n_x)

    def fun(alpha):
        return -np.sum(np.log(alpha.dot(phi)))
        
    def jac(alpha):
        return -phi.dot(1 / alpha.dot(phi))
    
    bounds = [(0, None)] * b
    constraints = [{'type': 'eq', 'fun': lambda alpha: alpha.dot(h) - 1}]

    alpha0 = np.random.uniform(0, 1, b)
    result = minimize(fun=fun, jac=jac, x0=alpha0, bounds=bounds, constraints=constraints, 
                      options={'maxiter': maxiter})
    
    if not result.success:
        print('Optimization failed: %s' % result.message, file=sys.stderr)
        
    return np.mean(np.log(result.x.dot(phi)))


class MutualInformationEstimator(object):
    def __init__(self, sigma=1, b=200, maxiter=1000):
        self._sigma = sigma
        self._b = b
        self._maxiter = maxiter
        
    def fit(self, X, Y):
        n_x, d_x = X.shape
        n_y, d_y = Y.shape
        b = min(self._b, n_x)

        XY = np.hstack([X, Y])
        UV = KMeans(b).fit(XY).cluster_centers_
        self._UV = UV
        U, V = np.split(UV, [d_x], axis=1)
        phi_x = np.array([normal(X, u, self._sigma) for u in U])
        phi_y = np.array([normal(Y, v, self._sigma) for v in V])
        phi = phi_x * phi_y
        h_x = np.sum(phi_x, axis=1)
        h_y = np.sum(phi_y, axis=1)
        h_xy = np.sum(phi, axis=1)
        h = (h_x * h_y - h_xy) / (n_x ** 2 - n_x)

        def fun(alpha):
            return -np.sum(np.log(alpha.dot(phi)))

        def jac(alpha):
            return -phi.dot(1 / alpha.dot(phi))

        bounds = [(0, None)] * b
        constraints = [{'type': 'eq', 'fun': lambda alpha: alpha.dot(h) - 1}]

        alpha0 = np.random.uniform(0, 1, b)
        result = minimize(fun=fun, jac=jac, x0=alpha0, bounds=bounds, constraints=constraints, 
                          options={'maxiter': self._maxiter})

        if not result.success:
            print('Optimization failed: %s' % result.message, file=sys.stderr)

        self._alpha = result.x
        self.mi = np.mean(np.log(self._alpha.dot(phi)))
        return self

    def score(self, X, Y):
        n_x, d_x = X.shape
        n_y, d_y = Y.shape
        U, V = np.split(self._UV, [d_x], axis=1)
        phi_x = np.array([normal(X, u, self._sigma) for u in U])
        phi_y = np.array([normal(Y, v, self._sigma) for v in V])
        phi = phi_x * phi_y
        return np.mean(np.log(self._alpha.dot(phi)))


class MutualInformationEstimatorCV(object):
    def __init__(self, sigmas=None, b=200, maxiter=1000, n_splits=5):
        self._b = b
        self._maxiter = maxiter
        self._n_splits = n_splits
        self._sigmas = sigmas if sigmas is not None else [0.3, 0.5, 0.8, 1, 2]

    def fit(self, X, Y):
        score_min = np.inf
        for sigma in self._sigmas:
            scores = []
            kf = KFold(n_splits=self._n_splits)
            for train_idx, test_idx in kf.split(X):
                estimator = MutualInformationEstimator(sigma=sigma, b=self._b, maxiter=self._maxiter)
                estimator.fit(X[train_idx], Y[train_idx])
                scores += [estimator.score(X[test_idx], Y[test_idx])]

            score_new = np.mean(scores)
            if score_new < score_min:
                score_min = score_new
                self.sigma_opt = sigma
                print('sigma:', sigma)

        estimator = MutualInformationEstimator(sigma=self.sigma_opt, b=self._b, maxiter=self._maxiter)
        estimator.fit(X, Y)
        self.mi = estimator.mi
        return self