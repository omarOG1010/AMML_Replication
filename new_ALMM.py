import numpy as np
import collections as cl
import multiprocessing as mp
import time
import pickle
from itertools import repeat

class PF():
    def __init__(self, ls=10, reg=1, alp=1, iters=10):
        print('Model parameter')
        self.cc = mp.cpu_count()
        self.cc = 5
        self.ls = ls
        self.reg = reg
        self.alp = alp
        self.iters = iters
        self.mn = 'PF'
        self.output = '%s iters=%d factor=%d alpha=%d reg=%f' % (
            self.mn, self.iters, self.ls, self.alp, self.reg)

        print(self.output)

    def _init_f(self, uidx, tidx, nidx):
        self.uLF = {i: np.random.normal(size=(self.ls)) for i in uidx}
        self.tLF = {i: np.random.normal(size=(self.ls)) for i in tidx}
        self.nLF = {i: np.random.normal(size=(self.ls)) for i in nidx}

    def _init_w(self, fs):
        self.wt = np.random.normal(size=(fs, self.ls))
        self.wn = np.random.normal(size=(fs, self.ls))

def evl_RMSE(i, R, m):
    u, t, e, p = R[i]
    prd = np.dot(m.uLF[u], m.tLF[t]) + np.dot(m.uLF[u], m.nLF[e]) + np.dot(m.tLF[t], m.nLF[e])
    RMSE = (p - prd) ** 2
    return RMSE

def update_U(u, R, uidx, m):
    C = 1 + m.alp * np.log(1 + R[uidx[u], 3])
    X = np.array([m.tLF[i] for i in R[uidx[u], 1]])
    Y = np.array([m.nLF[i] for i in R[uidx[u], 2]])
    I = m.reg * np.eye(m.ls)

    XY = X + Y
    XYXY = XY.T @ XY
    C_XY = C - np.sum(X * Y, axis=1)
    C_XYXY = XY.T @ C_XY
    return [u, np.linalg.inv(XYXY + I) @ C_XYXY]

def update_T(t, R, tidx, m):
    C = 1 + m.alp * np.log(1 + R[tidx[t], 3])
    X = np.array([m.uLF[i] for i in R[tidx[t], 0]])
    Y = np.array([m.nLF[i] for i in R[tidx[t], 2]])
    I = m.reg * np.eye(m.ls)

    XY = X + Y
    XYXY = XY.T @ XY
    C_XY = C - np.sum(X * Y, axis=1)
    C_XYXY = XY.T @ C_XY
    return [t, np.linalg.inv(XYXY + I) @ C_XYXY]

def update_N(n, R, nidx, m):
    C = 1 + m.alp * np.log(1 + R[nidx[n], 3])
    X = np.array([m.uLF[i] for i in R[nidx[n], 0]])
    Y = np.array([m.tLF[i] for i in R[nidx[n], 1]])
    I = m.reg * np.eye(m.ls)

    XY = X + Y
    XYXY = XY.T @ XY
    C_XY = C - np.sum(X * Y, axis=1)
    C_XYXY = XY.T @ C_XY
    return [n, np.linalg.inv(XYXY + I) @ C_XYXY]

def fea_w(t, sid, reg):
    A = np.array([F[i] for i in sid])
    I = reg * np.eye(fs)
    if t == 'tLF':
        X = np.array([m.tLF[i] for i in sid])
        m.wt = X.T @ A @ np.linalg.inv(A.T @ A + I)
    elif t == 'nLF':
        X = np.array([m.nLF[i] for i in sid])
        m.wn = X.T @ A @ np.linalg.inv(A.T @ A + I)

def put_LF(utn):
    for idx, f in LF:
        if utn == 'u':
            m.uLF[idx] = f
        elif utn == 't':
            m.tLF[idx] = f
        elif utn == 'n':
            m.nLF[idx] = f

def load_data():
    R = np.genfromtxt('./Rating.dat', dtype=int, delimiter=',', skip_header=1)
    fea = np.genfromtxt('./feature.dat', dtype=float, delimiter=',', skip_header=1)

    F = {int(row[0]) + 1: row[1:] for row in fea}
    uidx, tidx, nidx = cl.defaultdict(list), cl.defaultdict(list), cl.defaultdict(list)

    for idx, val in enumerate(R):
        u, t, n, p = val
        uidx[u].append(idx)
        tidx[t].append(idx)
        nidx[n].append(idx)

    fs = len(fea[0][1:])
    return R, F, uidx, tidx, nidx, fs

if __name__ == '__main__':
    mp.freeze_support()
    R, F, uidx, tidx, nidx, fs = load_data()
    m = PF()
    m._init_f(uidx.keys(), tidx.keys(), nidx.keys())
    m._init_w(fs)
    w_reg = 0.1
    print('Training model')

    for i in range(m.iters):
        mt = time.time()

        with mp.Pool(processes=m.cc) as pool:
            LF = pool.starmap(update_U, zip(uidx.keys(), repeat(R), repeat(uidx), repeat(m)))
        put_LF('u')

        with mp.Pool(processes=m.cc) as pool:
            LF = pool.starmap(update_T, zip(tidx.keys(), repeat(R), repeat(tidx), repeat(m)))
        put_LF('t')

        fea_w('tLF', tidx.keys(), w_reg)
        for j in m.tLF:
            m.tLF[j] = m.wt @ F[j]

        with mp.Pool(processes=m.cc) as pool:
            LF = pool.starmap(update_N, zip(nidx.keys(), repeat(R), repeat(nidx), repeat(m)))
        put_LF('n')

        fea_w('nLF', nidx.keys(), w_reg)
        for j in m.nLF:
            m.nLF[j] = m.wn @ F[j]

        with mp.Pool(processes=mp.cpu_count()) as pool:
            RE = np.array(pool.starmap(evl_RMSE, zip(range(len(R)), repeat(R), repeat(m))))
        print('Train %2d loss=%f time=%f' % (i + 1, np.sqrt(RE.mean()), time.time() - mt))
