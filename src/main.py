import numpy as np
import numpy.random as rnd
import scipy.sparse as sp
from src.mbpg import MBPG
from src.alternating import AM, MU
import src.mbpg


def obj(x):
    return loss(x)


def loss(x):
    w, h = x[:m, :], x[m:, :].T
    wh = np.dot(w, h)
    return np.sum(-X*np.log(wh + eps) + wh) + loss_const


def burg_entropy(x):
    return -np.sum(np.log(x + eps))


def kernel(x):
    return burg_entropy(x) + np.sum(x ** 2)/2


def grad_kernel(x):
    w, h = x[:m, :], x[m:, :].T
    grad = np.empty(x.shape)
    grad[:m, :] = -1 / (w + eps) + w
    grad[m:, :] = (-1 / (h + eps) + h).T
    return grad


if __name__ == '__main__':
    m, n = 200, 200
    r = 10
    sparsity = 1.0
    eps = 0
    MAX_ITER = 3000
    scaled = False
    src.mbpg.WRITE, src.alternating.WRITE = [True] * 2

    rnd.seed(42)
    W = sp.random(m, r, sparsity, data_rvs=rnd.rand).toarray()
    a = 2 * np.ones(n)
    H = rnd.dirichlet(a, r)
    X = W.dot(H)
    OPT = np.concatenate((W, H.T))
    x0 = rnd.rand(m + n, r)
    W0, H0 = x0[:m, :], x0[m:, :].T
    if scaled:
        alpha = np.sum(X) / np.sum(W0.dot(H0))
    else:
        alpha = 1
    x0 = np.sqrt(alpha) * np.concatenate((W0, H0.T))

    w_x = np.sum(X, axis=1)
    h_x = np.sum(X, axis=0)
    lsmad = max(np.amax(w_x), np.amax(h_x))
    loss_const = np.sum(X * (np.log(X + eps) - 1))

    dir_path = './results/'
    nprob = 'NMFdirichlet_x0{}'.format('scaled' if scaled else 'unscaled')
    csv_paths = []

    nalg = 'MBPG'
    path = dir_path + '{}-{}-{}-{}-{}-{}'.format(nprob, nalg, n, m, r, sparsity)
    csv_path = path + '.csv'
    csv_paths += [csv_path]

    mbpg = MBPG(x0, obj, X, OPT, m, kernel, grad_kernel, lsmad, csv_path, MAX_ITER=MAX_ITER)
    mbpg.run()

    nalg = 'MBPGe'
    path = dir_path + '{}-{}-{}-{}-{}-{}'.format(nprob, nalg, n, m, r, sparsity)
    csv_path = path + '.csv'
    csv_paths += [csv_path]

    mbpge = MBPG(x0, obj, X, OPT, m, kernel, grad_kernel, lsmad, csv_path, MAX_ITER=MAX_ITER)
    mbpge.run(extrapolation=True)

    nalg = 'MU'
    path = dir_path + '{}-{}-{}-{}-{}-{}'.format(nprob, nalg, n, m, r, sparsity)
    csv_path = path + '.csv'
    csv_paths += [csv_path]

    mu = MU(x0, obj, OPT, m, X, csv_path, MAX_ITER=MAX_ITER)
    mu.run()

    nalg = 'AGD'
    path = dir_path + '{}-{}-{}-{}-{}-{}'.format(nprob, nalg, n, m, r, sparsity)
    csv_path = path + '.csv'
    csv_paths += [csv_path]

    am = AM(x0, obj, OPT, m, X, csv_path, MAX_ITER=MAX_ITER)
    am.run()
