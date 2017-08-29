import scipy.stats
import scipy.optimize
import numpy as np


def select_centroids(data, n_b):
    indices = np.random.choice(range(len(data)), n_b)
    return data[indices]


def create_gaussian_design(data, means, sigma, i, j):
    # `means` is the means of kernels
    n = len(data)
    d = len(data[0])
    mask = np.logical_and(np.arange(d) != i, np.arange(d) != j)
    A = np.transpose([scipy.stats.multivariate_normal.pdf(x=data, mean=mean, cov=np.eye(d) * sigma) for mean in means])
    b_x = np.sum([scipy.stats.norm.pdf(x=data[:, i], loc=mean[i], scale=sigma) for mean in means], axis=1)
    b_y = np.sum([scipy.stats.norm.pdf(x=data[:, j], loc=mean[j], scale=sigma) for mean in means], axis=1)
    b_z = np.sum([scipy.stats.norm.pdf(x=data[:, mask], loc=mean[mask], scale=sigma) for mean in means], axis=1)
    # b_xyz = np.sum(A, axis=0)
    # b = (b_x * b_y * b_z - b_xyz) / n / (n - 1) / (n - 2)
    b = b_x * b_y * b_z / n**3
    return A, b


def create_objective_function(A):
    return lambda alpha: -np.sum(np.log(A.dot(alpha)))


def create_derivative_function(A):
    return lambda alpha: -A.transpose().dot(1 / A.dot(alpha))


def true_ratio(x, mean, cov):
    return scipy.stats.multivariate_normal(x=x, mean=mean, cov=cov) / \
           scipy.stats.multivariate_normal(x=x, mean=mean, cov=np.diag(np.diag(cov)))


def approximated_ratio(x, alpha, means, sigma):
    cov = np.eye(len(alpha)) * sigma
    return sum(map(lambda t: t[0] * t[1](x),
                   zip(alpha, [scipy.stats.multivariate_normal(mean=mean, cov=cov).pdf for mean in means])))


def theoretical_mutual_information(cov): # wrong
    return 0.5 * (np.sum(np.log(np.diag(cov))) - np.log(np.linalg.det(cov)))


def approximated_conditional_mutual_information(x, i, j, n_b=200):
    means = select_centroids(x, n_b)
    A, b = create_gaussian_design(x, means, sigma=2, i=i, j=j)

    fun = create_objective_function(A)
    jac = create_derivative_function(A)
    bounds = [(0, None) for _ in range(n_b)]
    constraints = [{'type': 'eq', 'fun': lambda alpha: alpha.dot(b) - 1}]

    alpha0 = np.random.uniform(0, 1, n_b)
    result = scipy.optimize.minimize(fun=fun, jac=jac, x0=alpha0, bounds=bounds, constraints=constraints,
                                     method='SLSQP') #SLSQP, COBYLA
    print(result)
    alpha = result.x
    return np.mean(np.log(A.dot(alpha)))
