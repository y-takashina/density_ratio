import numpy as np
import scipy.stats
import sklearn.cluster


def norm(mean, sigma):
    cov = np.eye(len(mean)) * sigma
    return lambda x: scipy.stats.multivariate_normal.pdf(x, mean=mean, cov=cov)


class MutualInformation(object):
    def __init__(self, sigma=1, lamb=1e-6, b=200):
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
        
        #XY_ = np.array([np.hstack([x, y]) for x in X for y in Y]).reshape(n**2, d)
        XY_ = np.hstack([np.repeat(X, len(Y), axis=0), np.tile(Y, [len(X), 1])])
        
        UV = sklearn.cluster.KMeans(self._b).fit(XY).cluster_centers_
        bases = [norm(uv, self._sigma) for uv in UV]
        
        phi = np.array([b(XY) for b in bases])   # design matrix for observed data
        phi_ = np.array([b(XY_) for b in bases]) # design matrix for non-observed data
        H = phi_.dot(phi_.transpose()) / len(XY_)
        h = np.mean(phi, axis=1)
        
        alpha = np.linalg.solve(H + self._lambda * np.eye(self._b), h)
        alpha[alpha < 0] = 0
        
        self.alpha = alpha
        self.mi = np.mean(np.log(alpha.dot(phi)))
        
        return self

