import numpy as np
import numpy.linalg as LA
import time
from abc import abstractmethod


WRITE = False


class Iteration:
    def __init__(self, x0, obj, opt, m, MAX_ITER=1000):
        self.TOL = 1e-6
        self.MAX_ITER = MAX_ITER
        self.iter = 0
        self.time = 0.0
        self.xk = np.copy(x0)
        self.xk_old = np.copy(x0)
        self.obj = obj
        self.opt = opt
        self.m = m

    def stop_criteria(self):
        return LA.norm(self.xk - self.xk_old) / max(1.0, LA.norm(self.xk)) < self.TOL

    @abstractmethod
    def update(self, x, grad):
        pass


class FOM(Iteration):
    def __init__(self, x0, obj, X, opt, m, csv_path='', MAX_ITER=1000):
        super().__init__(x0, obj, opt, m, MAX_ITER)
        self.X = X
        self.yk = np.copy(x0)
        self.beta, self.__theta, self.__theta_old = 0.0, 1.0, 1.0
        self.restart_iter = 200
        self.__restart = self.restart_iter

    def adaptive_scheme(self):
        return np.dot(self.yk - self.xk, self.xk - self.xk_old) > 0

    def beta_update(self):
        self.beta = (self.__theta_old - 1) / self.__theta
        self.yk = self.xk + self.beta * (self.xk - self.xk_old)
        self.__theta_update()
        self.__theta_restart()

    def __theta_update(self):
        self.__theta_old, self.__theta = self.__theta, (1 + (1 + 4 * self.__theta ** 2) ** 0.5) * 0.5

    def __theta_restart(self):
        if self.adaptive_scheme():
            self.beta, self.__theta, self.__theta_old = 0.0, 1.0, 1.0
        if self.iter == self.restart_iter:
            self.beta, self.__theta, self.__theta_old = 0.0, 1.0, 1.0
            self.restart_iter += self.__restart

    @abstractmethod
    def grad(self, x, y):
        pass

    def run(self, extrapolation=False):
        m = self.m
        start = time.time()
        for _ in range(self.MAX_ITER):
            self.iter += 1
            if extrapolation:
                self.beta_update()
            else:
                self.beta = 0.0

            self.yk = self.xk + self.beta * (self.xk - self.xk_old)
            grad = self.grad(self.xk, self.yk)
            self.xk_old, self.xk = self.xk, self.update(self.yk, grad)

            if WRITE:
                w, h = self.xk[:m, :], self.xk[m:, :].T
                self.file.write(
                    '{},{},{},{},{},{}\n'.format(self.iter, time.time()-start, self.obj(self.xk),
                                                 LA.norm(self.xk*grad), LA.norm(w*grad[:m, :]), LA.norm(h*grad[m:, :].T),
                                                 ))
            if self.stop_criteria():
                break
        end = time.time()
        self.time = end - start
        print('Time: {} sec.'.format(self.time))
        if WRITE:
            self.file.close()
        return self.xk


class MBPG(FOM):
    def __init__(self, x0, obj, grad, opt, m, kernel, grad_kernel, lsmad, csv_path='', MAX_ITER=1000):
        super().__init__(x0, obj, grad, opt, m, csv_path, MAX_ITER)
        self.kernel = kernel
        self.grad_kernel = grad_kernel
        self.lsmad = lsmad
        self.rho = 0.99
        self.bregman_dist = lambda x, y: self.kernel(x) - self.kernel(y) - np.einsum('ij,ij->', self.grad_kernel(y), x - y)
        if WRITE:
            self.file = open(csv_path, 'w')
            w, h = self.xk[:m, :], self.xk[m:, :].T
            grad = self.grad(self.xk, self.xk)
            self.file.write('iter,time,obj,kkt,kkt_w,kkt_h\n')
            self.file.write(
                '{},0,{},{},{},{}\n'.format(self.iter, self.obj(self.xk),
                                            LA.norm(self.xk * grad), LA.norm(w * grad[:m, :]),
                                            LA.norm(h * grad[m:, :].T)))

    def grad(self, x, y):
        m = self.m
        w, h = x[:m, :], x[m:, :].T
        wy, hy = y[:m, :], y[m:, :].T
        alpha = self.X / w.dot(h)

        w_h = np.sum(hy, axis=1)
        h_w = np.sum(wy, axis=0)

        alpha_w = alpha.dot(hy.T)
        alpha_h = alpha.T.dot(wy)

        grad = np.empty(x.shape)
        grad[:m, :] = -alpha_w
        grad[:m, :] = np.apply_along_axis(lambda y: y + w_h, 1, grad[:m, :])
        grad[m:, :] = -alpha_h
        grad[m:, :] = np.apply_along_axis(lambda y: y + h_w, 1, grad[m:, :])

        self.lsmad = max(np.max(wy * alpha_w), np.max(hy.T * alpha_h))
        return grad

    def adaptive_scheme(self):
        if (self.yk < 0).any():
            return True
        return ((1+self.rho)*self.kernel(self.xk) - self.kernel(self.yk)
                - np.einsum('ij,ij->', self.grad_kernel(self.yk), self.xk - self.yk)
                > self.rho*(self.kernel(self.xk_old)
                            - np.einsum('ij,ij->', self.grad_kernel(self.xk), self.xk_old - self.xk)))

    def update(self, x, grad):
        p = grad / self.lsmad - self.grad_kernel(x)
        v = (-p + np.sqrt(p**2 + 4))/2
        return v
