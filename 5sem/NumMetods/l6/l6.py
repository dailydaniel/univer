#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import argparse

sys.path.append('../')

from functools import reduce
from itertools import count, takewhile

import numpy as np
import pprint as pp

from gauss.gauss import solve_gauss

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--a', default=-1.0, type=float)
    parser.add_argument('-b', '--b', default=1.0, type=float)
    parser.add_argument('-h', '--h', default=0.5, type=float)
    return parser

def readata(path: str):
    '''считывание данных в список из двух списков (x, y)'''
    res = []
    with open(path, 'r') as f:
        for line in f:
            res.append(line)
    res = [arr.split() for arr in res]
    for i in range(len(res)):
        res[i] = [float(t) for t in res[i]]
    return res

def frange(*args):
    '''альтернатива range() для float'''
    if not args:
        return [0.0]
    elif len(args) == 1:
        stop = args[0]
        return [round(t, 1) for t in takewhile(lambda x: x < stop, count(0.0, 1.0))]
    elif len(args) == 2:
        start, stop = args
        return [round(t, 1) for t in takewhile(lambda x: x < stop, count(start, 1.0))]
    elif len(args) == 3:
        start, stop, step = args
        return [round(t, len(str(step)[2:])) for t in takewhile(lambda x: x < stop, count(start, step))]
    return [0.0]

def preduce(arr: list):
    '''эффективная по памяти предобработка списка для добавления в reduce\nиспользуется при необходимости просуммировать ряд с действиями над каждым элементом'''
    yield 0.0
    for el in arr:
        yield el

def createMatrix(x: list, y: list, n: int):
    '''создание матрицы для поиска многочлена степени n'''
    res = []

    for k in range(n):
        cur = [0.0] * (n + 1)
        for i in range(n):
            cur[i] = reduce(lambda a, b: a + b ** (k + i), preduce(x))
        cur[-1] = reduce(lambda a, b: a + b[0] * b[1] ** k, preduce(list(zip(y, x))))
        res.append(cur)
    return np.array(res)

def makea(m, dlt = 6) -> list:
    '''нахождение неизвестных a для поиска многочлена путем решения СЛАУ'''
    a = solve_gauss(m)
    res = []
    for el in a:
        res.append(round(float(el), dlt))
    return res

def makeArr(a: list, X: float) -> float:
    '''возвращение значения многочлена в точке X'''
    return reduce(lambda a, b: a + b[0] * X ** b[1], preduce(list(zip(a, range(len(a))))))

def qd(x: list, y: list, A: list, fn=makeArr, delta=5) -> float:
    '''квадратичное отклонение'''
    return round(reduce(lambda a, b: a + (fn(A, b[0]) - b[1]) ** 2, preduce(list(zip(x, y)))), delta)

def lagr(x: list, y: list, X: float, delta=5) -> float:
    '''многочлен лагранжа'''
    res = 0
    for i in range(len(x)):
        cur = 1
        for k in range(len(x)):
            if i != k:
                cur *= (X-x[k]) / (x[i] - x[k])
        res += cur * y[i]
    return round(res, delta)

if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    pp.pprint(readata(namespace.data))
