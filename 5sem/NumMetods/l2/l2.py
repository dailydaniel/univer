#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import argparse

import numpy as np
import pprint

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default='l2.txt', type=str)
    parser.add_argument('-x', '--x', default=0.2, type=float)
    return parser

def returnInd(arr: list, x = None, y = None) -> int:
    if x and y:
        for i, el in enumerate(arr):
            if x == el[0] and y == el[1]:
                return i
    elif x and not y:
        for i, el in enumerate(arr):
            if x == el[0]:
                return i
    elif y and not x:
        for i, el in enumerate(arr):
            if y == el[1]:
                return i
    else:
         return -1

def calc(arr: list, x: float) -> list:
    res = []
    ex = False
    for el in arr:
        if el[0] == x:
            ex = True
    if not ex:
        print('error: x not in arr')
        return res

    y = 0.0
    for el in arr:
        if x == el[0]:
            y = el[1]

    t = (y - arr[returnInd(arr, x)-1][1]) / (x - arr[returnInd(arr, x)-1][0])
    res.append((round(t, 3), 'first left'))

    t = (arr[returnInd(arr, x)+1][1] - y) / (arr[returnInd(arr, x)+1][0] - x)
    res.append((round(t, 3), 'first right'))

    t0 = arr[returnInd(arr, x)+1][1] - arr[returnInd(arr, x)-1][1]
    t1 = arr[returnInd(arr, x)+1][0] - arr[returnInd(arr, x)-1][0]
    t = t0 / t1
    res.append((round(t, 3), 'first centr'))

    t0 = arr[returnInd(arr, x)+1][1] - 2*y + arr[returnInd(arr, x)-1][1]
    t1 = pow(arr[returnInd(arr, x)+1][0] - x, 2)
    t = t0 / t1
    res.append((round(t, 3), 'second'))
    return res

if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    pp = pprint.PrettyPrinter()

    arr = list(np.loadtxt(namespace.file))
    #print(namespace.x == arr[2][0])
    res = calc(arr, namespace.x)
    pp.pprint(res)
