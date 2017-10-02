import numpy as np
import scipy.stats
import sklearn.cluster


def norm(X, mean, sigma):
    cov = np.eye(len(mean)) * sigma
    return scipy.stats.multivariate_normal.pdf(X, mean=mean, cov=cov)


class DensityRatioCV(object):
    def __init__(self, b=100, method='ulsif', sigmas=None, lambdas=None):
        self._b = b
        self._method = method
        self._sigmas = sigmas or [1e-2, 5e-2, 0.1, 0.3, 0.5, 1, 2, 5]
        self._lambdas = lambdas or [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    
    
    def fit(self, X_de, X_nu):
        n_tr = len(X_de)
        n_te = len(X_nu)
        n = min(n_tr, n_te)
        b = min(self._b, n)
        means = sklearn.cluster.KMeans(b).fit(X_de).cluster_centers_
        
        score = np.inf
        for sigma in self._sigmas:
            phi_tr = np.array([norm(X_de, mean, sigma) for mean in means])
            phi_te = np.array([norm(X_nu, mean, sigma) for mean in means])
            H = phi_tr.dot(phi_tr.transpose()) / n_tr
            h = np.mean(phi_te, axis=1)
            
            for lamb in self._lambdas:
                B = H + lamb * (n_tr - 1) / n_tr * np.eye(b)
                B_inv_phi_tr = np.linalg.solve(B, phi_tr[:, :n])
                denom = n_tr - np.sum(phi_tr[:, :n] * B_inv_phi_tr, axis=0)
                B0 = np.repeat(np.linalg.solve(B, h), n).reshape(b, n)
                B0 += B_inv_phi_tr * (h.dot(B_inv_phi_tr) / denom).reshape(1, -1)
                B1 = np.linalg.solve(B, phi_te[:, :n])
                B1 += B_inv_phi_tr * (np.sum(phi_te[:, :n] * B_inv_phi_tr, axis=0) / denom).reshape(1, -1)
                B2 = (n_tr - 1) / n_tr / (n_te - 1) * (n_te * B0 - B1)
                B2[B2 < 0] = 0
                w_tr = np.sum(phi_tr[:, :n] * B2, axis=0)
                score_new = w_tr.dot(w_tr) / n / 2 - np.sum(phi_te[:, :n] * B2) / n
                if score > score_new:
                    score = score_new
                    self.sigma_opt = sigma
                    self.lambda_opt = lamb
                    print('sigma: %f, lambda: %f, score: %f' % (sigma, lamb, score))
        
        return self