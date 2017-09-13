import numpy as np
import scipy.stats
import scipy.optimize
import sklearn.cluster


def create_objective_function(A):
    return lambda alpha: -np.sum(np.log(A.dot(alpha)))


def create_derivative_function(A):
    return lambda alpha: -A.transpose().dot(1 / A.dot(alpha))


def create_gaussian_design(X, Y, U, V, sigma):
    n_x, d_x = X.shape
    n_y, d_y = Y.shape
    n_b_u, d_u = U.shape
    n_b_v, d_v = V.shape
    if n_x != n_y or n_b_u != n_b_v or d_x != d_u or d_y != d_v:
        raise Exception('Invalid argument.')
    
    d_xy = d_x + d_y
    XY = np.hstack([X, Y])
    UV = np.hstack([U, V])
    A = np.transpose([scipy.stats.multivariate_normal.pdf(x=XY, mean=uv, cov=sigma ** 2 * np.eye(d_xy)) for uv in UV])
    b_x = np.sum([scipy.stats.multivariate_normal.pdf(x=X, mean=u, cov=sigma ** 2 * np.eye(d_x)) for u in U], axis=1)
    b_y = np.sum([scipy.stats.multivariate_normal.pdf(x=Y, mean=v, cov=sigma ** 2 * np.eye(d_y)) for v in V], axis=1)
    b_xy = np.sum(A, axis=0)
    b = (b_x * b_y - b_xy) / (n_x ** 2 - n_x)
    return A, b


def mutual_information(X, Y, sigma=1, n_b=200, maxiter=200):
    n_x, d_x = X.shape
    n_y, d_y = Y.shape
    if n_x != n_y:
        raise Exception('Invalid argument: the lengths of X and Y must be the same.')
    
    XY = np.hstack([X, Y])
    UV = sklearn.cluster.KMeans(n_b).fit(XY).cluster_centers_
    U, V = np.split(UV, [d_x], axis=1)
    A, b = create_gaussian_design(X, Y, U, V, sigma)
    
    fun = create_objective_function(A)
    jac = create_derivative_function(A)
    bounds = [(0, None)] * n_b
    constraints = [{'type': 'eq', 'fun': lambda alpha: alpha.dot(b) - 1}]

    alpha0 = np.random.uniform(0, 1, n_b)
    result = scipy.optimize.minimize(fun=fun, jac=jac, x0=alpha0, bounds=bounds, constraints=constraints, options={'maxiter': maxiter})
    if not result.success:
        raise Exception('Optimization failed.')
    alpha = result.x
    return np.mean(np.log(A.dot(alpha)))

