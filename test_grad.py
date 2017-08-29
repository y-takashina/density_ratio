from unittest import TestCase
import numpy as np
import scipy.stats
import scipy.optimize
import sklearn.cluster
import approximate_conditional_mutual_information


class TestGrad(TestCase):
    def test_grad(self):
        n_b = 200
        cov = [[1, 0.5, 0],
               [0.5, 1, 0],
               [0, 0, 1]]
        X = scipy.stats.multivariate_normal(mean=[0, 0, 0], cov=cov).rvs(3000)
        means = sklearn.cluster.KMeans(n_b).fit(X).cluster_centers_
        A, b = approximate_conditional_mutual_information.create_gaussian_design(X, means, sigma=0.3, i=0, j=1)
        fun = approximate_conditional_mutual_information.create_objective_function(A)
        jac = approximate_conditional_mutual_information.create_derivative_function(A)
        for _ in range(10):
            x = np.random.uniform(0, 1, n_b)
            self.assertAlmostEqual(scipy.optimize.check_grad(fun, jac, x), 0, delta=1e-3)


