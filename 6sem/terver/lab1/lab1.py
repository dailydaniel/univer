#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import argparse
from collections import Counter

from functools import reduce
from scipy import stats as st
import numpy as np
from glob import glob
import pickle
import re

import matplotlib.pyplot as plt

variant = 9

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--write', default=0, type=int)
    parser.add_argument('-r', '--read', default=1, type=int)
    return parser

def save(r, type_ryad):
    reg = 'data/' + type_ryad + '*'
    files = glob(reg) # type: ryadN.pickle
    if files:
        n = int(re.sub(r'[^\d]', '', files[-1]))
        n += 1
    else:
        n = 0
    filename = reg[:-1] + str(n) + '.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(r, f)

def load(type_ryad, n = -1):
    reg = 'data/' + type_ryad + '*'
    files = glob(reg)
    if not files:
        return
    if abs(n) > len(files):
        return
    with open(files[n], 'rb') as f:
        return(pickle.load(f))

def binomial(n, p, size, write, read):
    r = load('binomial') if read else st.binom.rvs(n, p, size=size)
    if write:
        save(r, 'binomial')

    st_r = sorted(r)
    p_ch = Counter(r)
    return r, st_r, p_ch


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    n = 5 + variant % 17
    p = 0.1 + 0.01 * variant
    t, _, _ = binomial(n = n, p = p, size = 150, write = namespace.write,
                                           read = namespace.read)
    print(t)
