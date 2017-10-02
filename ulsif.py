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


class MutualInformation(object):
    def __init__(self, sigma=1, lamb=1e-6, b=100):
        self._sigma = sigma
        self._lambda = lamb
        self._b = b
        
    
    def fit(self, X, Y):
        n_x, d_x = X.shape
        n_y, d_y = Y.shape
        if n_x != n_y:
            raise Exception()

        XY = np.hstack([X, Y])
        n, d = XY.shape
        b = min(self._b, n)
        
        UV = sklearn.cluster.KMeans(b).fit(XY).cluster_centers_
        U, V = np.split(UV, [d_x], axis=1)
        
        phi_x = np.array([norm(X, u, self._sigma) for u in U])
        phi_y = np.array([norm(Y, v, self._sigma) for v in V])
        h = np.mean(phi_x * phi_y, axis=1)
        H_x = phi_x.dot(phi_x.transpose()) / n
        H_y = phi_y.dot(phi_y.transpose()) / n
        H = H_x * H_y

        alpha = np.linalg.solve(H + self._lambda * np.eye(b), h)
        alpha[alpha < 0] = 0
        
        self.mi = 0.5 * (h.dot(alpha) - 1)
        
        return self


class MutualInformationCV(object):
    def __init__(self, b=100, sigmas=None, lambdas=None):
        self._sigmas = sigmas or [1e-2, 5e-2, 0.1, 0.3, 0.5, 1, 2, 5]
        self._lambdas = lambdas or [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        self._b = b
        
    
    def fit(self, X, Y):
        n_x, d_x = X.shape
        n_y, d_y = Y.shape
        if n_x != n_y:
            raise Exception()
        
        XY = np.hstack([X, Y])
        n, d = XY.shape
        b = min(self._b, n)
        
        UV = sklearn.cluster.KMeans(b).fit(XY).cluster_centers_
        U, V = np.split(UV, [d_x], axis=1)
        
        score = np.inf
        for sigma in self._sigmas:
            phi_x = np.array([norm(X, u, sigma) for u in U])
            phi_y = np.array([norm(Y, v, sigma) for v in V])
            phi = phi_x * phi_y
            h = np.mean(phi, axis=1)
            H_x = phi_x.dot(phi_x.transpose()) / n
            H_y = phi_y.dot(phi_y.transpose()) / n
            H = H_x * H_y
            
            for lamb in self._lambdas:
                A = H + lamb * (n ** 2 - 1) / n ** 2 * np.eye(b)
                a = np.linalg.solve(A, h)
                A_inv_phi = np.linalg.solve(A, phi)
                score_new = 0
                for i in range(5):
                    for j in range(n):
                        phi_ij = phi_x[:, i] * phi_y[:, j]
                        a_ij = np.linalg.solve(A, phi_ij)
                        denom = n ** 2 - a_ij.dot(phi_ij)
                        alpha = (n + 1) / n * (a + phi_ij.dot(a) / denom * a_ij)
                        alpha -= (n + 1) / n ** 2 * (A_inv_phi[:, i] + phi_ij.dot(A_inv_phi[:, i]) / denom * a_ij)
                        alpha[alpha < 0] = 0
                        score_new += phi_ij.dot(alpha) ** 2 / 2 - phi[:, i].dot(alpha)
                
                score_new /= n ** 2
                
                if score > score_new:
                    score = score_new
                    self.sigma_opt = sigma
                    self.lambda_opt = lamb
                    theta = np.linalg.solve(H + lamb * np.eye(b), h)
                    theta[theta < 0] = 0
                    self.mi = 0.5 * (h.dot(theta) - 1)
                    print('sigma: %f, lambda: %f, score: %f' % (sigma, lamb, score))

        return self
