import numpy as np
import scipy.stats
import scipy.optimize
import sklearn.cluster


def create_gaussian_design(X, means, sigma):
    n, d = X.shape
    cov = np.eye(3) * sigma
    A = np.transpose([scipy.stats.multivariate_normal.pdf(x=X, mean=mean, cov=cov) for mean in means])
    b_x = np.sum([scipy.stats.norm.pdf(x=X[:, 0], loc=mean[0], scale=sigma) for mean in means], axis=1)
    b_y = np.sum([scipy.stats.norm.pdf(x=X[:, 1], loc=mean[1], scale=sigma) for mean in means], axis=1)
    b_z = np.sum([scipy.stats.norm.pdf(x=X[:, 2], loc=mean[2], scale=sigma) for mean in means], axis=1)
    b_xyz = np.sum(A, axis=0)
    b = (b_x * b_y * b_z - b_xyz) / (n**3 - n)
    return A, b


def create_objective_function(A):
    return lambda alpha: -np.sum(np.log(A.dot(alpha)))


def create_derivative_function(A):
    return lambda alpha: -A.transpose().dot(1 / A.dot(alpha))


def mutual_information(x, n_b=200):
    means = sklearn.cluster.KMeans(n_b).fit(x).cluster_centers_
    A, b = create_gaussian_design(x, means, 1)

    fun = create_objective_function(A)
    jac = create_derivative_function(A)
    bounds = [(0, None) for _ in range(n_b)]
    constraints = [{'type': 'eq', 'fun': lambda alpha: alpha.dot(b) - 1}]

    alpha0 = np.random.uniform(0, 1, n_b)
    result = scipy.optimize.minimize(fun=fun, jac=jac, x0=alpha0, bounds=bounds, constraints=constraints)
    alpha = result.x
    return np.mean(np.log(A.dot(alpha)))


