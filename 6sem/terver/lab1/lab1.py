#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import argparse
from collections import Counter
from itertools import islice

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

def draw_poligon(count, name):
    X = list(count.keys())
    Y = [count[x] for x in X]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(X, Y, label='Полигон относительных частот')
    ax.set_title('Полигон относительных частот')
    ax.legend(loc='upper left')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_xlim(xmin=0, xmax=max(X))
    ax.set_ylim(ymin=0, ymax=max(Y) + 5)
    ax.grid()
    fig.tight_layout()
    fig.savefig('data/poligon_' + name + '.png')

def draw_cdf(Xlist, Ylist, name):
    fig, ax = plt.subplots(figsize=(5, 5))

    for X, Y in zip(Xlist, Ylist):
        ax.plot(X, Y, label='')
    ax.set_title('Эмпирическая функция распределения')
    ax.legend(loc='upper left')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_xlim(xmin=0, xmax=max(X))
    ax.set_ylim(ymin=0, ymax=1)
    fig.tight_layout()
    ax.grid()
    fig.savefig('data/cdf_' + name + '.png')

def binomial(np, size, write, read):
    n, p = np
    r = load('binomial') if read else st.binom.rvs(n, p, size=size)
    if write:
        save(r, 'binomial')

    st_r = sorted(r)
    p_ch = Counter(r)
    nv = {key: (val, val / sum(p_ch.values())) for key, val in p_ch.items()}

    draw_poligon(p_ch, 'binomial')

    delta = 0.01
    items = list(zip(list(range(max(r) + 2)), islice(list(range(max(r) + 2)), 1, None))) # [(0, 1), (1, 2), ...]
    Xlist = [[x * delta for x in range(item[0] * int(1 / delta), item[1] * int(1 / delta))]
              for item in items]
    Ylist = [[st.binom.cdf(x, n, p) for x in line] for line in Xlist]

    draw_cdf(Xlist, Ylist, 'binomial')

    cdf_func = [el[0] for el in Ylist]
    cdf_func.append(1.0)
    cdf_func.insert(0, 0.0)

    mean, variance, skew, kurtosis = st.binom.stats(n, p, moments='mvsk') # среднее, дисперсия, ассиметрия, эксцесс
    standart, med = st.binom.std(n, p), st.binom.median(n, p) # стантартное отклонение, медиана
    mode = st.mode(r) # мода

    return (r, st_r, cdf_func, nv, float(mean),
                                  float(variance),
                                  float(skew),
                                  float(kurtosis),
                                  standart,
                                  med,
                                  float(mode.mode))

def geometr(p, size, write, read):
    r = load('geometr') if read else st.geom.rvs(p, size=size)
    if write:
        save(r, 'geometr')

    st_r = sorted(r)
    p_ch = Counter(r)
    nv = {key: (val, val / sum(p_ch.values())) for key, val in p_ch.items()}

    draw_poligon(p_ch, 'geometr')

    delta = 0.01
    items = list(zip(list(range(max(r) + 2)), islice(list(range(max(r) + 2)), 1, None))) # [(0, 1), (1, 2), ...]
    Xlist = [[x * delta for x in range(item[0] * int(1 / delta), item[1] * int(1 / delta))]
              for item in items]
    Ylist = [[st.geom.cdf(x, p) for x in line] for line in Xlist]

    draw_cdf(Xlist, Ylist, 'geometr')

    cdf_func = [el[0] for el in Ylist]
    cdf_func.append(1.0)
    cdf_func.insert(0, 0.0)

    mean, variance, skew, kurtosis = st.geom.stats(p, moments='mvsk') # среднее, дисперсия, ассиметрия, эксцесс
    standart, med = st.geom.std(p), st.geom.median(p) # стантартное отклонение, медиана
    mode = st.mode(r) # мода

    return (r, st_r, cdf_func, nv, float(mean),
                                  float(variance),
                                  float(skew),
                                  float(kurtosis),
                                  standart,
                                  med,
                                  float(mode.mode))

def puasson(mu, size, write, read):
    r = load('puasson') if read else st.poisson.rvs(mu, size=size)
    if write:
        save(r, 'puasson')

    st_r = sorted(r)
    p_ch = Counter(r)
    nv = {key: (val, val / sum(p_ch.values())) for key, val in p_ch.items()}

    draw_poligon(p_ch, 'puasson')

    delta = 0.01
    items = list(zip(list(range(max(r) + 2)), islice(list(range(max(r) + 2)), 1, None))) # [(0, 1), (1, 2), ...]
    Xlist = [[x * delta for x in range(item[0] * int(1 / delta), item[1] * int(1 / delta))]
              for item in items]
    Ylist = [[st.poisson.cdf(x, mu) for x in line] for line in Xlist]

    draw_cdf(Xlist, Ylist, 'puasson')

    cdf_func = [el[0] for el in Ylist]
    cdf_func.append(1.0)
    cdf_func.insert(0, 0.0)

    mean, variance, skew, kurtosis = st.poisson.stats(mu, moments='mvsk') # среднее, дисперсия, ассиметрия, эксцесс
    standart, med = st.poisson.std(mu), st.poisson.median(mu) # стантартное отклонение, медиана
    mode = st.mode(r) # мода

    return (r, st_r, cdf_func, nv, float(mean),
                                  float(variance),
                                  float(skew),
                                  float(kurtosis),
                                  standart,
                                  med,
                                  float(mode.mode))

def save_result(result):
    reg = 'data/result*'
    files = glob(reg)
    if files:
        n = int(re.sub(r'[^\d]', '', files[-1]))
        n += 1
    else:
        n = 0
    filename = reg[:-1] + str(n) + '.txt'
    with open(filename, 'w') as f:
        f.write(result)


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    result = ['выборка: {0}',
              'упорядоченная: {1}',
              'эмпирическая функция распределения: {2}',
              'статистический ряд: {3}',
              'среднее: {4}',
              'дисперсия: {5}',
              'ассиметрия: {6}',
              'эксцесс: {7}',
              'среднее квадратичное отклонение: {8}',
              'медиана: {9}',
              'мода: {10}']

    line = '\n\n'.join(result)

    task = {'binomial': (binomial, [5 + variant % 17, 0.1 + 0.01 * variant]),
            'geometr': (geometr, 0.1 + 0.01 * variant),
            'puasson': (puasson, 0.7 + 0.07 * variant)}

    out = ''
    for r_type, fdata in task.items():
        func, params = fdata
        res = func(params, size=150,
                           write = namespace.write,
                           read = namespace.read)

        out += r_type + '\n\n' + line.format(*res) + '\n\n<' + '='*50 + '>\n\n'

    save_result(out)
