from unittest import TestCase
import numpy as np
import scipy.stats
import scipy.optimize
import sklearn.cluster
import mlmi


class TestMlmi(TestCase):
    def test_create_gaussian_design(self):
        n_b = 200
        sigma = 1
        cov = np.array([[1, 0.5], [0.5, 1]])
        X = scipy.stats.multivariate_normal(mean=[0, 0], cov=cov).rvs(3000)
        means = sklearn.cluster.KMeans(n_b).fit(X).cluster_centers_
        U, V = np.split(means, [1], axis=1)
        
        A, b = mlmi.create_gaussian_design(X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1), U, V, sigma)
        A_old, b_old = mlmi.create_gaussian_design_old(X, means, sigma)
        
        self.assertTrue(np.all(A - A_old < 1e-6))
        self.assertTrue(np.all(b - b_old < 1e-6))