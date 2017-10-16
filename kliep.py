import tqdm
import numpy as np
import scipy.stats
import scipy.optimize
import sklearn.cluster


def norm(X, mean, sigma):
    cov = sigma ** 2 * np.eye(len(mean))
    return scipy.stats.multivariate_normal.pdf(X, mean=mean, cov=cov)


def mutual_information(X, Y, Z=None, b=100, sigma=1, maxiter=1000):
    n_x, d_x = X.shape
    n_y, d_y = Y.shape
    b = min(b, n_x)
    
    if Z is None:
        XY = np.hstack([X, Y])
        UV = sklearn.cluster.KMeans(b).fit(XY).cluster_centers_
        U, V = np.split(UV, [d_x], axis=1)
        phi_x = np.array([norm(X, u, sigma) for u in U])
        phi_y = np.array([norm(Y, v, sigma) for v in V])
        phi = phi_x * phi_y
        h_x = np.sum(phi_x, axis=1)
        h_y = np.sum(phi_y, axis=1)
        h_xy = np.sum(phi, axis=1)
        h = (h_x * h_y - h_xy) / (n_x ** 2 - n_x)
        
    else:
        XYZ = np.hstack([X, Y, Z])
        UVW = sklearn.cluster.KMeans(b).fit(XYZ).cluster_centers_
        U, V, W = np.split(UVW, [d_x, d_x + d_y], axis=1)
        phi_x = np.array([norm(X, u, sigma) for u in U])
        phi_y = np.array([norm(Y, v, sigma) for v in V])
        phi_z = np.array([norm(Z, w, sigma) for w in W])
        phi = phi_x * phi_y * phi_z
        h_x = np.sum(phi_x, axis=1)
        h_y = np.sum(phi_y, axis=1)
        h_z = np.sum(phi_z, axis=1)
        h_xyz = np.sum(phi, axis=1)
        h = (h_x * h_y * h_z - h_xyz) / (n_x ** 3 - n_x)

    def fun(alpha):
        return -np.sum(np.log(alpha.dot(phi)))
        
    def jac(alpha):
        return -phi.dot(1 / alpha.dot(phi))
    
    bounds = [(0, None)] * b
    constraints = [{'type': 'eq', 'fun': lambda alpha: alpha.dot(h) - 1}]

    alpha0 = np.random.uniform(0, 1, b)
    result = scipy.optimize.minimize(fun=fun, 
                                     jac=jac, 
                                     x0=alpha0, 
                                     bounds=bounds, 
                                     constraints=constraints, 
                                     options={'maxiter': maxiter})
    
    if not result.success:
        raise Exception('Optimization failed.')
        
    return np.mean(np.log(result.x.dot(phi)))


def calc_mi_matrix(X):
    d = len(X[0])
    mi = np.zeros([d, d])
    for i, j in tqdm.tqdm([(i, j) for i in range(d) for j in range(d) if i > j]):
        x_i = X[:, i].reshape(-1, 1)
        x_j = X[:, j].reshape(-1, 1)
        mi[i, j] = mi[j, i] = mutual_information(X=x_i, Y=x_j, maxiter=1000)
    
    return mi


def calc_cmi_matrix(X):
    d = len(X[0])
    cmi = np.zeros([d, d])
    for i, j in tqdm.tqdm([(i, j) for i in range(d) for j in range(d) if i > j]):
        x_i = X[:, i].reshape(-1, 1)
        x_j = X[:, j].reshape(-1, 1)
        x_rest = X[:, (np.arange(d) != i) & (np.arange(d) != j)]
        mi_xz = mutual_information(X=x_i, Y=x_rest, maxiter=1000)
        mi_yz = mutual_information(X=x_j, Y=x_rest, maxiter=1000)
        mi_xyz = mutual_information(X=x_i, Y=x_j, Z=x_rest, maxiter=1000)
        cmi[i, j] = cmi[j, i] = mi_xyz - (mi_xz + mi_yz)

    cmi[cmi < 0] = 0
    return cmi