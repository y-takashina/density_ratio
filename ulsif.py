import numpy as np
import scipy.stats
import sklearn.cluster
import cvxopt


def norm(X, mean, sigma):
    cov = sigma ** 2 * np.eye(len(mean))
    return scipy.stats.multivariate_normal.pdf(X, mean=mean, cov=cov)


def mutual_information(X, Y, Z=None, b=100, sigma=1, lamb=1e-3):
    n_x, d_x = X.shape
    n_y, d_y = Y.shape
    b = min(b, n_x)
    
    if Z is None:
        XY = np.hstack([X, Y])
        UV = sklearn.cluster.KMeans(b).fit(XY).cluster_centers_
        U, V = np.split(UV, [d_x], axis=1)
        phi_x = np.array([norm(X, u, sigma) for u in U])
        phi_y = np.array([norm(Y, v, sigma) for v in V])
        phi = phi_x * phi_y
        H_x = phi_x.dot(phi_x.transpose()) / n_x
        H_y = phi_y.dot(phi_y.transpose()) / n_x
        H = H_x * H_y
        h = np.mean(phi, axis=1)
        
    else:
        XYZ = np.hstack([X, Y, Z])
        UVW = sklearn.cluster.KMeans(b).fit(XYZ).cluster_centers_
        U, V, W = np.split(UVW, [d_x, d_x + d_y], axis=1)
        phi_x = np.array([norm(X, u, sigma) for u in U])
        phi_y = np.array([norm(Y, v, sigma) for v in V])
        phi_z = np.array([norm(Z, w, sigma) for w in W])
        phi = phi_x * phi_y * phi_z
        H_x = phi_x.dot(phi_x.transpose()) / n_x
        H_y = phi_y.dot(phi_y.transpose()) / n_x
        H_z = phi_z.dot(phi_z.transpose()) / n_x
        H = H_x * H_y * H_z
        h = np.mean(phi, axis=1)
        
    alpha = np.linalg.solve(H + lamb * np.eye(b), h)
    alpha[alpha < 0] = 0
    
    return np.mean(np.log(alpha.dot(phi)))
    #return 0.5 * (h.dot(alpha) - 1)


class MutualInformation(object):
    def __init__(self, sigma=1, lamb=1e-6, b=100):
        self._sigma = sigma
        self._lambda = lamb
        self._b = b
        
        
    def fit(self, X, Y, Z=None):
        n_x, d_x = X.shape
        n_y, d_y = Y.shape
        b = min(self._b, n_x)
        
        if Z is None:
            XY = np.hstack([X, Y])
            UV = sklearn.cluster.KMeans(b).fit(XY).cluster_centers_
            U, V = np.split(UV, [d_x], axis=1)
            
            phi_x = np.array([norm(X, u, self._sigma) for u in U])
            phi_y = np.array([norm(Y, v, self._sigma) for v in V])
            phi = phi_x * phi_y
            H_x = phi_x.dot(phi_x.transpose()) / n_x
            H_y = phi_y.dot(phi_y.transpose()) / n_x
            H = H_x * H_y
            h = np.mean(phi, axis=1)
            #h_x = np.sum(phi_x, axis=1)
            #h_y = np.sum(phi_y, axis=1)
            #h_xy = np.sum(phi, axis=1)
            #h = (h_x * h_y - h_xy) / (n_x ** 2 - n_x)
            
        else:
            XYZ = np.hstack([X, Y, Z])
            UVW = sklearn.cluster.KMeans(b).fit(XYZ).cluster_centers_
            U, V, W = np.split(UVW, [d_x, d_x + d_y], axis=1)
            
            phi_x = np.array([norm(X, u, self._sigma) for u in U])
            phi_y = np.array([norm(Y, v, self._sigma) for v in V])
            phi_z = np.array([norm(Z, w, self._sigma) for w in W])
            phi = phi_x * phi_y * phi_z
            H_x = phi_x.dot(phi_x.transpose()) / n_x
            H_y = phi_y.dot(phi_y.transpose()) / n_x
            H_z = phi_z.dot(phi_z.transpose()) / n_x
            H = H_x * H_y * H_z
            h = np.mean(phi, axis=1)
            #h_x = np.sum(phi_x, axis=1)
            #h_y = np.sum(phi_y, axis=1)
            #h_z = np.sum(phi_z, axis=1)
            #h_xyz = np.sum(H, axis=0)
            #h = (h_x * h_y * h_z - h_xyz) / (n_x ** 3 - n_x)


        alpha = np.linalg.solve(H + self._lambda * np.eye(b), h)
        alpha[alpha < 0] = 0
        
        #self.mi = 0.5 * (h.dot(alpha) - 1)
        self.mi = np.mean(np.log(alpha.dot(phi)))
        
        return self


class MutualInformationCV(object):
    def __init__(self, b=100, sigmas=None, lambdas=None):
        self._sigmas = sigmas or [0.05, 0.1, 0.3, 0.5, 1, 2, 5]
        self._lambdas = lambdas or [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
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
                    alpha = np.linalg.solve(H + lamb * np.eye(b), h)
                    alpha[alpha < 0] = 0
                    self.mi = np.mean(np.log(alpha.dot(phi)))
                    #self.mi = 0.5 * (h.dot(alpha) - 1)
                    print('sigma: %f, lambda: %f, score: %f' % (sigma, lamb, score))

        return self
