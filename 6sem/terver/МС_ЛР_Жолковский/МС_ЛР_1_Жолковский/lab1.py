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
        n = int(re.sub(r'[^\d]', '', files[0]))
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

def draw_poligon(count, pr, name, sizex=5, sizey=5):
    X = list(count.keys())
    Y = [count[x][1] for x in X]
    fig, ax = plt.subplots(figsize=(sizex, sizey))
    ax.plot(X, Y, label='относительные частоты')
    ax.plot(X, pr, label='вероятности')
    ax.set_title('Полигон относительных частот')
    ax.legend(loc='upper left')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_xlim(xmin=0, xmax=max(X))
    ax.set_ylim(ymin=0, ymax=max(Y) + 0.1)
    if sizex != 5:
        # Don't allow the axis to be on top of your data
        ax.set_axisbelow(True)

        # Turn on the minor TICKS, which are required for the minor GRID
        ax.minorticks_on()

        # Customize the major grid
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
        # Customize the minor grid
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    else:
        ax.grid()
    fig.tight_layout()
    fig.savefig('data/poligon_' + name + '.png')

def draw_cdf(Xlist, Ylist, name, sizex=5, sizey=5):
    fig, ax = plt.subplots(figsize=(sizex, sizey))

    for X, Y in zip(Xlist, Ylist):
        ax.plot(X, Y, label='')
    ax.set_title('Эмпирическая функция распределения')
    ax.legend(loc='upper left')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_xlim(xmin=0, xmax=max(X))
    ax.set_ylim(ymin=0, ymax=1)
    fig.tight_layout()
    if sizex != 5:
        # Don't allow the axis to be on top of your data
        ax.set_axisbelow(True)

        # Turn on the minor TICKS, which are required for the minor GRID
        ax.minorticks_on()

        # Customize the major grid
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
        # Customize the minor grid
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    else:
        ax.grid()
    fig.savefig('data/cdf_' + name + '.png')

def expect(nv, key):
    if key in nv.keys():
        return nv[key][1]
    else:
        return 0.0
    # elif key not in nv.keys() and key != 0:
    #     return expect(nv, key-1)
    # else:
    #     return 0.0

def generate_ef(nv, r):
    delta = 0.01
    items = list(zip(list(range(max(r) + 2)), islice(list(range(max(r) + 2)), 1, None))) # [(0, 1), (1, 2), ...]
    keys = list(range(list(nv.keys())[-1] + 1)) # list(nv.keys())
    Y = [sum([expect(nv, key) for key in keys[:i + 1]]) for i, el in enumerate(keys)]
    Xlist = [[x * delta for x in range(item[0] * int(1 / delta), item[1] * int(1 / delta))]
              for item in items]
    Ylist = [[y] * len(Xlist[i]) for i, y in enumerate(Y)]
    # Ylist = [[st.binom.cdf(x, n, p) for x in line] for line in Xlist]
    return Xlist, Ylist, Y

def exp_stats(nv, r, st_r):
    mean = sum([key * val[1] for key, val in nv.items()]) # mean
    var = sum([(key - mean) ** 2 * val[1] for key, val in nv.items()])
    standart = var ** 0.5
    mu = lambda k: sum([(key - mean) ** k * val[1] for key, val in nv.items()])
    skew = mu(3) / (standart ** 3)
    kurtosis = mu(4) / (standart ** 4) - 3
    mode = st.mode(r)
    if len(st_r) % 2 != 0:
        med = st_r[int(len(st_r) / 2)]
    else:
        med = 0.5 * (st_r[int(len(st_r) / 2)] + st_r[int(len(st_r) / 2) - 1])
    return mean, var, standart, skew, kurtosis, mode, med

def binomial(np, size, write, read):
    n, p = np
    r = load('binomial') if read else st.binom.rvs(n, p, size=size)
    if write:
        save(r, 'binomial')

    st_r = sorted(r)
    p_ch = Counter(r)
    nv = {key: (val, val / sum(p_ch.values())) for key, val in p_ch.items()}

    ex_mean, ex_var, ex_standart, ex_skew, ex_kurtosis, ex_mode, ex_med = exp_stats(nv, r, st_r)

    pr = [st.binom.pmf(x, n, p) for x in list(nv.keys())]
    draw_poligon(nv, pr, 'binomial')

    # delta = 0.01
    # items = list(zip(list(range(max(r) + 2)), islice(list(range(max(r) + 2)), 1, None))) # [(0, 1), (1, 2), ...]
    # Xlist = [[x * delta for x in range(item[0] * int(1 / delta), item[1] * int(1 / delta))]
    #           for item in items]
    # Ylist = [[st.binom.cdf(x, n, p) for x in line] for line in Xlist]

    Xlist, Ylist, Y = generate_ef(nv, r)

    draw_cdf(Xlist, Ylist, 'binomial')

    cdf_func = [el[0] for el in Ylist]
    # cdf_func.append(1.0)
    # cdf_func.insert(0, 0.0)

    mean, variance, skew, kurtosis = st.binom.stats(n, p, moments='mvsk') # среднее, дисперсия, ассиметрия, эксцесс
    standart, med = st.binom.std(n, p), st.binom.median(n, p) # стантартное отклонение, медиана
    mode = st.mode(r) # мода

    pmf = [abs(val[1] - st.binom.pmf(key, n, p)) for key, val in nv.items()]

    return (r, st_r, cdf_func, nv, float(mean),
                                  float(variance),
                                  float(skew),
                                  float(kurtosis),
                                  standart,
                                  med,
                                  float(mode.mode), ex_mean,
                                                    ex_var,
                                                    ex_standart,
                                                    ex_skew,
                                                    ex_kurtosis,
                                                    float(ex_mode.mode),
                                                    ex_med, pmf, Y)

def geometr(p, size, write, read):
    r = load('geometr') if read else st.geom.rvs(p, loc=-1, size=size)
    if write:
        save(r, 'geometr')

    st_r = sorted(r)
    p_ch = Counter(r)
    nv = {key: (val, val / sum(p_ch.values())) for key, val in p_ch.items()}

    ex_mean, ex_var, ex_standart, ex_skew, ex_kurtosis, ex_mode, ex_med = exp_stats(nv, r, st_r)

    pr = [st.geom.pmf(x, p, loc=-1) for x in list(nv.keys())]
    draw_poligon(nv, pr, 'geometr', 30, 30)

    # delta = 0.01
    # items = list(zip(list(range(max(r) + 2)), islice(list(range(max(r) + 2)), 1, None))) # [(0, 1), (1, 2), ...]
    # Xlist = [[x * delta for x in range(item[0] * int(1 / delta), item[1] * int(1 / delta))]
    #           for item in items]
    # Ylist = [[st.geom.cdf(x, p) for x in line] for line in Xlist]

    Xlist, Ylist, Y = generate_ef(nv, r)

    draw_cdf(Xlist, Ylist, 'geometr', 30, 30)

    cdf_func = [el[0] for el in Ylist]
    cdf_func.append(1.0)
    cdf_func.insert(0, 0.0)

    mean, variance, skew, kurtosis = st.geom.stats(p, loc=-1, moments='mvsk') # среднее, дисперсия, ассиметрия, эксцесс
    standart, med = st.geom.std(p, loc=-1), st.geom.median(p, loc=-1) # стантартное отклонение, медиана
    mode = st.mode(r) # мода

    pmf = [abs(val[1] - st.geom.pmf(key, p, loc=-1)) for key, val in nv.items()]

    return (r, st_r, cdf_func, nv, float(mean),
                                  float(variance),
                                  float(skew),
                                  float(kurtosis),
                                  standart,
                                  med,
                                  float(mode.mode), ex_mean,
                                                    ex_var,
                                                    ex_standart,
                                                    ex_skew,
                                                    ex_kurtosis,
                                                    float(ex_mode.mode),
                                                    ex_med, pmf, Y)

def puasson(mu, size, write, read):
    r = load('puasson') if read else st.poisson.rvs(mu, size=size)
    if write:
        save(r, 'puasson')

    st_r = sorted(r)
    p_ch = Counter(r)
    nv = {key: (val, val / sum(p_ch.values())) for key, val in p_ch.items()}

    ex_mean, ex_var, ex_standart, ex_skew, ex_kurtosis, ex_mode, ex_med = exp_stats(nv, r, st_r)

    pr = [st.poisson.pmf(x, mu) for x in list(nv.keys())]
    draw_poligon(nv, pr, 'puasson')

    # delta = 0.01
    # items = list(zip(list(range(max(r) + 2)), islice(list(range(max(r) + 2)), 1, None))) # [(0, 1), (1, 2), ...]
    # Xlist = [[x * delta for x in range(item[0] * int(1 / delta), item[1] * int(1 / delta))]
    #           for item in items]
    # Ylist = [[st.poisson.cdf(x, mu) for x in line] for line in Xlist]

    Xlist, Ylist, Y = generate_ef(nv, r)

    draw_cdf(Xlist, Ylist, 'puasson')

    cdf_func = [el[0] for el in Ylist]
    cdf_func.append(1.0)
    cdf_func.insert(0, 0.0)

    mean, variance, skew, kurtosis = st.poisson.stats(mu, moments='mvsk') # среднее, дисперсия, ассиметрия, эксцесс
    standart, med = st.poisson.std(mu), st.poisson.median(mu) # стантартное отклонение, медиана
    mode = st.mode(r) # мода

    pmf = [abs(val[1] - st.poisson.pmf(key, mu)) for key, val in nv.items()]

    return (r, st_r, cdf_func, nv, float(mean),
                                  float(variance),
                                  float(skew),
                                  float(kurtosis),
                                  standart,
                                  med,
                                  float(mode.mode), ex_mean,
                                                    ex_var,
                                                    ex_standart,
                                                    ex_skew,
                                                    ex_kurtosis,
                                                    float(ex_mode.mode),
                                                    ex_med, pmf, Y)

def save_result(result):
    reg = 'data/result*'
    files = glob(reg)
    if files:
        n = int(re.sub(r'[^\d]', '', files[0]))
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
              'теор. среднее: {4} <=> среднее: {11}',
              'теор. дисперсия: {5} <=> дисперсия: {12}',
              'теор. ассиметрия: {6} <=> ассиметрия: {14}',
              'теор. эксцесс: {7} <=> эксцесс: {15}',
              'теор. среднее квадратичное отклонение: {8} <=> среднее квадратичное отклонение: {13}',
              'теор. медиана: {9} <=> медиана: {17}',
              'теор. мода: {10} <=> мода: {16}',
              'pmf: {18}',
              'ex_pmf {19}']

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
