import scipy.stats
import scipy.optimize
import sklearn.cluster
import numpy as np


def create_gaussian_design(X, means, sigma):
    n, d = X.shape
    # 各データを各基底に与えたときの値(計画行列)。
    A = np.transpose([scipy.stats.multivariate_normal.pdf(x=X, mean=mean, cov=np.eye(d) * sigma) for mean in means])
    mask = (np.arange(d) != i) & (np.arange(d) != j)
    b_x = np.sum([scipy.stats.norm.pdf(x=X[:, i], loc=mean[i], scale=sigma) for mean in means], axis=1)
    b_y = np.sum([scipy.stats.norm.pdf(x=X[:, j], loc=mean[j], scale=sigma) for mean in means], axis=1)
    b_z = np.sum([scipy.stats.norm.pdf(x=X[:, mask], loc=mean[mask], scale=sigma) for mean in means], axis=1)
    b_xyz = np.sum(A, axis=0)
    b = (b_x * b_y * np.prod(b_z, axis=1) - b_xyz) / (n ** 3 - n)
    # b = np.prod(b_z, axis=1) / n # b_z だけを使って正規化？
    return A, b


def create_objective_function(A):
    return lambda alpha: -np.sum(np.log(A.dot(alpha)))


def create_derivative_function(A):
    return lambda alpha: -A.transpose().dot(1 / A.dot(alpha))


def conditional_mutual_information(x, i, j, n_b=200, maxiter=1000):
    means = sklearn.cluster.KMeans(n_b).fit(x).cluster_centers_
    A, b = create_gaussian_design(x, means, sigma=1, i=i, j=j)

    fun = create_objective_function(A)
    jac = create_derivative_function(A)
    bounds = [(0, None) for _ in range(n_b)]
    constraints = [{'type': 'eq', 'fun': lambda alpha: alpha.dot(b) - 1}]
    
    alpha0 = np.random.uniform(0, 1, n_b)
    result = scipy.optimize.minimize(fun=fun, jac=jac, x0=alpha0, bounds=bounds, constraints=constraints, options={'maxiter': maxiter})
    print('succeed: ', result.success)
    alpha = result.x
    return np.mean(np.log(A.dot(alpha)))
