from unittest import TestCase
import numpy as np
import scipy.stats
import scipy.optimize
import sklearn.cluster
import mlmi


class TestMlmi(TestCase):
    def test_mlmi(self):
        for c in np.arange(0, 0.5, 0.1):
            cov = np.array([[1, c], [c, 1]])
            X = scipy.stats.multivariate_normal(mean=[0, 0], cov=cov).rvs(1000)
            ami = mlmi.mutual_information(X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1), n_b=200, maxiter=1000)
            tmi = 0.5 * (np.sum(np.log(np.diag(cov))) - np.log(np.linalg.det(cov)))
            print('ami: %.4f, tmi: %.4f, error: %.4f' % (ami, tmi, abs(ami - tmi)))
            self.assertAlmostEqual(ami, tmi, delta=0.05)

            
    def test_mlmi3(self):
        self.fail("not implemented")
