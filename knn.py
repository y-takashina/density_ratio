import tqdm
import numpy as np
import scipy.stats
import sklearn
import sklearn.datasets


def cmi_gaussian(cov):
    H = np.linalg.slogdet(cov)[1] / 2
    H1 = [np.linalg.slogdet(np.delete(np.delete(cov, [i], axis=0), [i], axis=1))[1] / 2 for i in range(3)]
    H2 = [[np.linalg.slogdet(np.delete(np.delete(cov, [i, j], axis=0), [i, j], axis=1))[1] / 2 for i in range(3)] for j in range(3)]
    cmi = np.array([[H1[i] + H1[j] - (H + H2[i][j]) for i in range(3)] for j in range(3)])
    cmi[np.eye(3, dtype=bool)] = None
    return cmi


def volume_unit_ball(d):
    return np.pi ** (d/2) / scipy.special.gamma(1 + d/2) / (2 ** d)


def entropy_knn(X, k=3):
    n, d = X.shape
    distances = np.linalg.norm(X.reshape([n, -1, d]) - X, axis=2)
    epsilons = 2 * np.sort(distances, axis=1)[:, k]
    entropy = -scipy.special.digamma(k) + scipy.special.digamma(n)
    entropy += np.log(volume_unit_ball(d))
    entropy += np.mean(np.log(epsilons)) * d
    return entropy


def mi_3h(X, Y, k=3):
    XY = np.hstack([X, Y])
    return entropy_knn(X) + entropy_knn(Y) - entropy_knn(XY)


def cmi_4h(X, Y, Z, k=3):
    XYZ = np.hstack([X, Y, Z])
    XZ = np.hstack([X, Z])
    YZ = np.hstack([Y, Z])
    cmi = entropy_knn(XZ) + entropy_knn(YZ)
    cmi -= entropy_knn(XYZ) + entropy_knn(Z)
    return cmi


def mi_knn(X, Y, k=3):
    if X.size == 0 or Y.size == 0:
        return 0
    n_x, d_x = X.shape
    n_y, d_y = Y.shape
    n = n_x
    distances_x = np.linalg.norm(X.reshape([n, -1, d_x]) - X, axis=2)
    distances_y = np.linalg.norm(Y.reshape([n, -1, d_y]) - Y, axis=2)
    distances = np.max([distances_x, distances_y], axis=0)
    epsilons = np.sort(distances, axis=1)[:, k] # 自分を除いて k 番目だけど 0 始まりだからこれでいい
    ks = np.repeat(k, n)
    indices_discrete = np.isclose(epsilons, 0)
    ks[indices_discrete] = np.sum(np.isclose(distances, 0), axis=1)[indices_discrete] - 1 # 自分を除く
    n_x = np.sum(distances_x <= epsilons, axis=0) # 自分を含むので元々本来の値より 1 大きい
    n_y = np.sum(distances_y <= epsilons, axis=0) # 同上
    mi = np.mean(scipy.special.digamma(ks))
    mi += np.log(n) - np.mean(np.log(n_x) + np.log(n_y))
    return mi


def cmi_knn(X, Y, Z, k=3):
    if Z.size == 0:
        return mi_knn(X, Y, k)
    XZ = np.hstack([X, Z])
    return mi_knn(XZ, Y, k) - mi_knn(Z, Y, k)


def learn_mrf(X, verbose=True, alpha=0.0, k=10):
    n, d = X.shape
    edges = []
    while True:
        max_cmi = -np.inf
        cmis = np.zeros([d, d])
        for i in range(d):
            neighbour = [edge[0] for edge in edges if edge[1] == i]
            neighbour += [edge[1] for edge in edges if edge[0] == i]
            non_neighbour = set(range(d)) - set(neighbour) - set([i])
            x = X[:, [i]]
            z = X[:, neighbour]
            for j in non_neighbour:
                if i == j:
                    continue
                y = X[:, [j]]
                cmis[i, j] = cmi_knn(x, y, z, k=k)

        cmis = cmis + cmis.transpose()
        idx = cmis.argmax()
        max_pair = int(idx / d), idx % d
        max_cmi = cmis[max_pair]
        if max_cmi <= alpha or len(edges) == d * (d - 1) / 2:
            return edges
        
        edges += [max_pair]