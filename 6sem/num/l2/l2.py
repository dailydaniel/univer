#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import argparse

from functools import reduce
import numpy as np

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default=0, type=int)
    return parser

def checkForD(m):
    pass

def abc(m, n = 3):
    k = (0, 2)

    for i in range(len(m[0][:-1])):
        t = [m[i][j] for j in range(k[0], k[1])]
        d = m[i][-1]
        if k[1] - k[0] < n:
            if k[0] == 0:
                for j in range(n - k[1]):
                    t.insert(0, 0.0)

                k_new = (0, k[1] + 1)

            elif k[1] == len(m[0][:-1]):
                for i in range(n - (k[1] - k[0])):
                    t.append(0.0)

                k_new = (k[0] + 1, k[1])

        else:
            if k[1] == len(m[0][:-1]):
                k_new = (k[0] + 1, k[1])
            else:
                k_new = (k[0] + 1, k[1] + 1)

        k = k_new
        t.append(d)
        yield t


def PQ(m, diag = None, pq = None):
    diag= abc(m) if not diag else diag

    flag = True
    try:
        a, b, c, d = next(diag)
    except:
        flag = False

    if flag == False:
        return pq

    if not pq:
        pq = ([-c / b], [d / b])
    else:
        curP = pq[0][-1]
        pq[0].append(-c / (b + a * pq[0][-1]))
        pq[1].append((d - a * pq[1][-1]) / (b + a * curP))

    return PQ(m, diag, pq)

def generatePQ(m):
    pq = PQ(m)
    for i in range(len(pq[0])-1, -1, -1):
        yield(pq[0][i], pq[1][i])


def solvePR(m, x = None, pq = None):
    pq = generatePQ(m) if not pq else pq

    flag = True
    try:
        P, Q = next(pq)
    except:
        flag = False

    if flag == False:
        x.reverse()
        x = [round(el, 4) for el in x]
        return x

    if not x:
        x = [Q]
    else:
        x.append(P * x[-1] + Q)

    return solvePR(m, x, pq)

if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    m = np.loadtxt('matrix.txt')

    pq = solvePR(m)
    print(pq)

    # for i in range(4, -1, -1):
    #     print(i)
