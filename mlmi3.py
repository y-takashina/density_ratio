import numpy as np
import scipy.stats
import scipy.optimize
import sklearn.cluster


def create_objective_function(A):
    return lambda alpha: -np.sum(np.log(A.dot(alpha)))


def create_derivative_function(A):
    return lambda alpha: -A.transpose().dot(1 / A.dot(alpha))


def create_gaussian_design(X, Y, Z, U, V, W, sigma):
    n_x, d_x = X.shape
    n_y, d_y = Y.shape
    n_z, d_z = Z.shape
    n_b_u, d_u = U.shape
    n_b_v, d_v = V.shape
    n_b_w, d_w = W.shape
    if not (n_x == n_y and n_x == n_z and n_b_u == n_b_v and n_b_u == n_b_w and d_z == d_w):
        raise Exception('Invalid argument.')
    
    means = np.hstack([U, V, W])
    cov = sigma ** 2 * np.eye(d_x + d_y + d_z)
    A = np.transpose([scipy.stats.multivariate_normal.pdf(x=np.hstack([X, Y, Z]), mean=mean, cov=cov) for mean in means])
    b_x = np.sum([scipy.stats.multivariate_normal.pdf(x=X, mean=u, cov=cov[:d_x, :d_x]) for u in U], axis=1)
    b_y = np.sum([scipy.stats.multivariate_normal.pdf(x=Y, mean=v, cov=cov[d_x:-d_z, d_x:-d_z]) for v in V], axis=1)
    b_z = np.sum([scipy.stats.multivariate_normal.pdf(x=Z, mean=w, cov=cov[-d_z:, -d_z:]) for w in W], axis=1)
    b_xyz = np.sum(A, axis=0)
    b = (b_x * b_y * b_z - b_xyz) / (n_x ** 3 - n_x)
    return A, b


def mutual_information(X, Y, Z, sigma=1, n_b=200, maxiter=200):
    n_x, d_x = X.shape
    n_y, d_y = Y.shape
    n_z, d_z = Z.shape
    if not (n_x == n_y and n_x == n_z):
        raise Exception('Invalid argument: the lengths of X and Y must be the same.')
    
    means = sklearn.cluster.KMeans(n_b).fit(np.hstack([X, Y, Z])).cluster_centers_
    U, V, W = np.split(means, [d_x, d_x + d_y], axis=1)
    A, b = create_gaussian_design(X, Y, Z, U, V, W, sigma)

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


def create_gaussian_design_old(X, means, sigma):
    n, d = X.shape
    cov = np.eye(3) * sigma
    A = np.transpose([scipy.stats.multivariate_normal.pdf(x=X, mean=mean, cov=cov) for mean in means])
    b_x = np.sum([scipy.stats.norm.pdf(x=X[:, 0], loc=mean[0], scale=sigma) for mean in means], axis=1)
    b_y = np.sum([scipy.stats.norm.pdf(x=X[:, 1], loc=mean[1], scale=sigma) for mean in means], axis=1)
    b_z = np.sum([scipy.stats.norm.pdf(x=X[:, 2], loc=mean[2], scale=sigma) for mean in means], axis=1)
    b_xyz = np.sum(A, axis=0)
    b = (b_x * b_y * b_z - b_xyz) / (n**3 - n)
    return A, b


def mutual_information_old(x, n_b=200):
    means = sklearn.cluster.KMeans(n_b).fit(x).cluster_centers_
    A, b = create_gaussian_design(x, means, 1)

    fun = create_objective_function(A)
    jac = create_derivative_function(A)
    bounds = [(0, None)] * n_b
    constraints = [{'type': 'eq', 'fun': lambda alpha: alpha.dot(b) - 1}]

    alpha0 = np.random.uniform(0, 1, n_b)
    result = scipy.optimize.minimize(fun=fun, jac=jac, x0=alpha0, bounds=bounds, constraints=constraints)
    alpha = result.x
    return np.mean(np.log(A.dot(alpha)))


