import numpy as np
import numpy.linalg as LA
import time
from abc import abstractmethod
from src.mmbpg import Iteration


WRITE = False


class Alternating(Iteration):
    def __init__(self, x0, obj, opt, m, X, csv_path='', MAX_ITER=1000):
        super().__init__(x0, obj, opt, m, MAX_ITER)
        self.X = X
        w, h = self.xk[:m, :], self.xk[m:, :].T
        self.alpha = self.X / (w.dot(h) + 1e-9)
        self.gradv = self.grad()
        if WRITE:
            self.file = open(csv_path, 'w')
            self.file.write('iter,time,obj,kkt,kkt_w,kkt_h\n')
            self.file.write(
                '{},0.0,{},{},{}\n'.format(self.iter, self.obj(self.xk),
                                           LA.norm(x0*self.gradv), LA.norm(w*self.gradv[:m, :]), LA.norm(h*self.gradv[m:, :].T),
                                           ))

    @abstractmethod
    def alternating_update(self):
        pass

    def grad(self):
        m = self.m
        w, h = self.xk[:m, :], self.xk[m:, :].T

        w_h = np.sum(h, axis=1)
        h_w = np.sum(w, axis=0)

        alpha_w = self.alpha.dot(h.T)
        alpha_h = self.alpha.T.dot(w)

        grad = np.empty(self.xk.shape)
        grad[:m, :] = -alpha_w
        grad[:m, :] = np.apply_along_axis(lambda y: y + w_h, 1, grad[:m, :])
        grad[m:, :] = -alpha_h
        grad[m:, :] = np.apply_along_axis(lambda y: y + h_w, 1, grad[m:, :])

        return grad

    def run(self):
        m = self.m

        start = time.time()
        for _ in range(self.MAX_ITER):
            self.iter += 1

            self.xk_old, self.xk = self.xk, np.vstack(self.alternating_update())

            if WRITE:
                w, h = self.xk[:m, :], self.xk[m:, :].T
                self.file.write(
                    '{},{},{},{},{},{}\n'.format(self.iter, time.time() - start, self.obj(self.xk),
                                                 LA.norm(self.xk*self.gradv), LA.norm(w*self.gradv[:m, :]), LA.norm(h*self.gradv[m:, :].T),
                                                 )
                )

            if self.stop_criteria():
                break

        end = time.time()
        self.time = end - start
        print('Time: {} sec.'.format(self.time))
        if WRITE:
            self.file.close()
        return self.xk


class MU(Alternating):
    def __init__(self, x0, obj, opt, m, X, csv_path='', MAX_ITER=1000):
        super().__init__(x0, obj, opt, m, X, csv_path, MAX_ITER)

    def alternating_update(self):
        w, h = self.xk[:self.m, :], self.xk[self.m:, :].T

        w_h = np.sum(h, axis=1)
        wk = np.apply_along_axis(lambda y: y / w_h, 1, w) * self.alpha.dot(h.T)
        wk = np.maximum(wk, 1e-9)

        alpha = self.X / (wk.dot(h) + 1e-9)
        h_w = np.sum(wk, axis=0)
        hk = np.apply_along_axis(lambda y: y / h_w, 0, h) * wk.T.dot(alpha)
        self.alpha = self.X / (wk.dot(hk) + 1e-9)
        hk = np.maximum(hk, 1e-9)

        if WRITE:
            self.gradv = self.grad()
        return wk, hk.T


class AM(Alternating):
    def __init__(self, x0, obj, opt, m, X, csv_path='', MAX_ITER=1000):
        super().__init__(x0, obj, opt, m, X, csv_path, MAX_ITER)
        self.eta = 1.1
        self.lsmooth = 0
        self.eps = 1e-9

    def alternating_update(self):
        m = self.m
        w, h = self.xk[:m, :], self.xk[m:, :].T

        alpha_w = self.alpha.dot(h.T)

        loss_xk = self.obj(self.xk)
        if self.iter == 1:
            l = np.max(alpha_w) * 10
        else:
            l = self.lsmooth
        wk = w - self.gradv[:m, :] / l
        wk = np.maximum(wk, self.eps)
        xk = np.vstack((wk, h.T))
        while self.obj(xk) > loss_xk + np.trace(self.gradv[:m, :].T.dot(wk - w)) + l * LA.norm(wk - w) ** 2 / 2:
            l *= self.eta
            wk = w - self.gradv[:m, :] / l
            wk = np.maximum(wk, self.eps)
            xk = np.vstack((wk, h.T))
        self.lsmooth = l

        loss_xk = self.obj(xk)

        h_w = np.sum(wk, axis=0)
        alpha = self.X / wk.dot(h)
        alpha_h = alpha.T.dot(wk)
        self.gradv[m:, :] = -alpha_h
        self.gradv[m:, :] = np.apply_along_axis(lambda y: y + h_w, 1, self.gradv[m:, :])
        hk = np.copy(h) - self.gradv[m:, :].T / l
        hk = np.maximum(hk, self.eps)
        xk = np.vstack((wk, hk.T))
        while self.obj(xk) > loss_xk + np.trace(self.gradv[m:, :].dot(hk - h)) + l * LA.norm(hk - h) ** 2 / 2:
            l *= self.eta
            hk = h - self.gradv[m:, :].T / l
            hk = np.maximum(hk, self.eps)
            xk = np.vstack((wk, hk.T))
        self.lsmooth = l
        self.alpha = self.X / wk.dot(hk)
        self.gradv = self.grad()
        return wk, hk.T
