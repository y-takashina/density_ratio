import scipy.stats
import scipy.optimize
import numpy as np


def generate_gaussian_data(mean, cov, n):
    return scipy.stats.multivariate_normal(mean=mean, cov=cov).rvs(n)


def select_centroids(x, n_b=200):
    indices = np.random.choice(range(len(x)), n_b)
    return x[indices]


def create_gaussian_design(x, means, sigma):
    n = len(x)
    cov = np.eye(2) * sigma
    A = np.transpose([scipy.stats.multivariate_normal.pdf(x=x, mean=mean, cov=cov) for mean in means])
    b_x = np.sum([scipy.stats.norm.pdf(x=x[:, 0], loc=mean[0], scale=cov[0, 0]) for mean in means], axis=1)
    b_y = np.sum([scipy.stats.norm.pdf(x=x[:, 1], loc=mean[1], scale=cov[1, 1]) for mean in means], axis=1)
    b_xy = np.sum(A, axis=0)
    b = (b_x * b_y - b_xy) / n / (n - 1)
    return A, b


def create_objective_function(A):
    return lambda alpha: -np.sum(np.log(A.dot(alpha)))


def create_derivative_function(A):
    return lambda alpha: -A.transpose().dot(1 / A.dot(alpha))


def true_ratio(x, mean, cov):
    return scipy.stats.multivariate_normal(x=x, mean=mean, cov=cov) / \
           scipy.stats.multivariate_normal(x=x, mean=mean, cov=np.diag(np.diag(cov)))


def approximated_ratio(x, alpha, means, cov):
    return sum(map(lambda t: t[0] * t[1](x),
                   zip(alpha, [scipy.stats.multivariate_normal(mean=mean, cov=cov).pdf for mean in means])))


def theoretical_mutual_information(cov):
    return 0.5 * (np.sum(np.log(np.diag(cov))) - np.log(np.linalg.det(cov)))


def approximated_mutual_information(x, n_b=200):
    means = select_centroids(x)
    A, b = create_gaussian_design(x, means, 1)

    fun = create_objective_function(A)
    jac = create_derivative_function(A)
    bounds = [(0, None) for _ in range(n_b)]
    constraints = [{'type': 'eq', 'fun': lambda alpha: alpha.dot(b) - 1}]

    alpha0 = np.random.uniform(0, 1, n_b)
    result = scipy.optimize.minimize(fun=fun, jac=jac, x0=alpha0, bounds=bounds, constraints=constraints,
                                     method='SLSQP')
    alpha = result.x
    return np.mean(np.log(A.dot(alpha)))
