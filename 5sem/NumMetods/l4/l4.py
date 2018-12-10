#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import argparse

sys.path.append('../')

import numpy as np

from gauss.gauss import solve_gauss

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default='l4.txt', type=str)
    parser.add_argument('-x', '--x', default=1.5, type=float)
    return parser

def readata(path: str):
    res = []
    with open(path, 'r') as f:
        for line in f:
            res.append(line)
    res = [arr.split() for arr in res]
    for i in range(len(res)):
        res[i] = [float(t) for t in res[i]]
    return res

def findh(x: list, dlt = 4) -> list:
    return [round(x[i + 1] - x[i], dlt) for i in range(len(x) - 1)]

def matrix(h: list, y: list, n = 3):
    res = []
    for i in range(n):
        res.append([0.0] * (n+1))

    # diag:
    for i in range(len(res)):
        res[i][i] = 2 * (h[i] + h[i+1])

    # romb:
    items = [[(0, 1), (1, 0)], [(1, 2), (2, 1)]]
    for n, lst in enumerate(items):
        for i, j in lst:
            res[i][j] = h[i+1]

    # res:
    for i in range(len(res)):
        res[i][-1] = 3 * ((y[i+2] - y[i+1]) / h[i+1]) - 3 * ((y[i+1] - y[i]) / h[i])

    return np.array(res)

def makec(m, dlt = 6) -> list:
    c = solve_gauss(m)
    res = [0.0]
    for el in c:
        res.append(round(float(el), dlt))
    return res

def abd(c: list, h: list, y: list, dlt = 6):
    a = [el for el in y[:-1]]

    b = []
    for i in range(len(h)-1):
        # try:
        #     b.append(round( ((y[i+1] - y[i]) / h[i]) - (h[i] * (c[i+1] + 2 * c[i])) / 3 , dlt))
        # except:
        #     b.append(round((y[i+1] - y[i]) / h[i] - h[i] * (0 + 2 * c[i]) / 3, dlt))
        b.append(round( ((y[i+1] - y[i]) / h[i]) - (h[i] * (c[i+1] + 2 * c[i])) / 3 , dlt))
    b.append(round((y[-1] - y[-2]) / h[-1] - (2 * c[-1] * h[-1]) / 3, dlt))

    d = []
    for i in range(len(h)-1):
        # try:
        #     d.append(round((c[i+1] - c[i]) / (3 * h[i]), dlt))
        # except:
        #     d.append(round((0 - c[i]) / (3 * h[i]), dlt))
        d.append(round((c[i+1] - c[i]) / (3 * h[i]), dlt))
    d.append(round( ((-1) * c[-1]) / (3 * h[-1]) , dlt))

    return a, b, d

class S(object):
    def __init__(self, a: list, b: list, c: list, d: list, x: list):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.x = x


    def fit(self, X: float, dlt = 6) -> float:
        if not len(self.a) == len(self.b) == len(self.c) == len(self.d):
            print('except')
            return 0.0

        res = 0.0

        i = self.findi(X, self.x)
        if i == -1:
            print('approximation does not work within these limits')
            return 0.0

        res += self.a[i] + self.b[i] * (X - self.x[i]) + self.c[i] * ((X - self.x[i]) ** 2) + self.d[i] * ((X - self.x[i]) ** 3)

        return round(res, dlt)


    def findi(self, X, x: list) -> int:
        for i, el in enumerate(x):
            if X >= el and i < len(x) - 1 and X <= x[i+1]:
                return i
        return -1
    
    def LandR(self, X: float):
        if X not in self.x:
            print('X not in x')
            return
        
        i = self.findi(X, self.x)
        j = i + 1
        
        ds1 = self.b[i] + 2 * self.c[i] * (X - self.x[i]) + 3 * self.d[i] * ((X - self.x[i]) ** 2)
        ds2 = self.b[j] + 2 * self.c[j] * (X - self.x[j]) + 3 * self.d[j] * ((X - self.x[j]) ** 2)
        
        print('dS left = {0}\ndS right = {1}'.format(ds1, ds2))
        if round(ds1, 3) == round(ds2, 3):
            print('Correct!')
        else:
            print('nope')


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    #print(namespace.data)
    x, y = readata(namespace.data)
    print('x = {0}\ny = {1}\n'.format(x, y))

    h = findh(x)
    print('h = {0}\n'.format(h))

    m = matrix(h, y)
    print('matrix = \n{0}\n'.format(m))

    c = makec(m)
    print('c = {0}\n'.format(c))

    a, b, d = abd(c, h, y)
    print('a = {0}\nb = {1}\nd = {2}\n'.format(a, b, d))

    s = S(a, b, c, d, x)
    res = s.fit(namespace.x)
    print(res)
