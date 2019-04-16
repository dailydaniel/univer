#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import argparse

from scipy import stats as st
import numpy as np
import pandas as pd

from copy import deepcopy
from collections import namedtuple
from collections import Counter
from functools import reduce
from copy import deepcopy
from random import choice
from itertools import islice
from math import log2
import string

import matplotlib.pyplot as plt


def findInterv(val: float, arr: list) -> int:
    test = lambda x, item: True if item[0] <= x <= item[1] else False
    for i, item in enumerate(arr):
        if test(val, item):
            return i
    return None

def expStats(sec_counter, first_counter, h):
    tmp = list(sec_counter.items())
    tmp.insert(0, 0.0)
    mean = reduce(lambda a, b: a + b[0] * b[1][1], tmp)
    s2 = reduce(lambda a, b: a + (b[0] - mean) ** 2 * b[1][1], tmp) - (h ** 2) / 12
    ds = s2 ** 0.5

    ak, most_common = first_counter.most_common(1)[0]
    wn = sorted(first_counter.items(), key=lambda x: x[0])
    k = {val: i for i, (key, val) in enumerate(sorted(first_counter.items(), key=lambda x: x[0]))}[most_common]
    wk = lambda k: sorted(sec_counter.items(), key=lambda x: x[0])[k][1][1]
    mode = ak[0] + h * ((wk(k) - wk(k-1)) / (2 * wk(k) - wk(k-1) - wk(k+1)))

    w = [val[1] for key, val in sorted(sec_counter.items(), key=lambda x: x)]
    sums = [reduce(lambda a, b: a + b, w[:i+1]) for i in range(len(w))]
    dsums = {el: i for i, el in enumerate(sums)}
    try:
        prk = [(dsums[l], dsums[r]) for l, r in list(zip(sums, islice(sums, 1, None))) if l <= 0.5 < r][0]
        a = [key for key, val in sorted(first_counter.items(), key=lambda x: (x[0]))]
        med = a[prk[0]][1] if sums[prk[0]] == 0.5 else a[prk[0]][0] + (h / w[prk[0]]) * (0.5 - sums[prk[0]])
    except:
        prk = (0, 1)
        a = [key for key, val in sorted(first_counter.items(), key=lambda x: (x[0]))]
        med = a[prk[0]][1] if sums[prk[0]] == 0.5 else a[prk[0]][0] + (h / w[prk[0]]) * (0.5 - sums[prk[0]])
        med = abs(med*10)

    mk = lambda k: sum(map(lambda item: item[0] ** k * item[1][1], sec_counter.items()))
    mck = lambda k: sum(map(lambda item: (item[0] - mean) ** k * item[1][1], sec_counter.items()))
    skew, kurtosis = mck(3) / (ds ** 3), (mck(4) / (ds ** 4)) - 3

    return mean, s2, ds, mode, med, skew, kurtosis

class LabFitter3000(object):
    '''IN: distribution name, params, scipy class
    OUT: path to figs, experimental stats, theoretical stats'''
    def __init__(self, obj, N, name, *params):
        self.distribution = obj
        self.N = N
        self.distribution_name = name
        self.params = params
        self.rvs = None
        self.h = None
        self.first_counter = None
        self.sec_counter = None
        self.pathes = []
        self.experimental_stats = None
        self.theoretical_stats = None
        self.Stats = namedtuple('Stats', 'mean variance std mode med skew kurtosis')

    def create_rvs(self):
        self.rvs = list(self.distribution.rvs(size=self.N))

    def create_first_counter(self):
        m = int(1 + log2(self.N))
        sr = sorted(self.rvs)
        f, l = sr[0], sr[-1]
        d = abs(l - f)
        self.h = d / m
        interv = [f + self.h * i for i in range(m)]
        interv.append(l)
        interv = list(zip(interv, islice(interv, 1, None)))

        self.first_counter = Counter()
        for val in sr:
            self.first_counter[interv[findInterv(val, interv)]] += 1

    def create_second_counter(self):
        self.sec_counter = {(key[1] + key[0]) / 2: (val, val / self.N)
                            for key, val in self.first_counter.items()}

    def hist(self, show: bool = False):#, sec_counter: dict, h: float, show: bool = False, path: str = 'Data/hist.png'):
        tmp = {key: val[1] / self.h for key, val in self.sec_counter.items()}
        plt.bar(list(tmp.keys()),
                list(tmp.values()),
                color='b',
                edgecolor='black',
                width=self.h)
        if show:
            plt.show()
        else:
            path = 'Data/' + self.distribution_name + '_hist.png'
            plt.savefig(path)
            plt.clf()

    def cdf(self, show: bool = False, sizex: int = 30, sizey: int = 30):#, sr: list, N: int, sizex: int = 30, sizey: int = 30, show: bool = False, path: str = 'Data/cdf.png'):

        sr = sorted(self.rvs)
        items = list(zip(sr, islice(sr, 1, None)))
        delta = 0.00001
        Xlist = [[x * delta for x in range(int(item[0] / delta), int(item[1] / delta))]
                 for item in items]
        Ylist = [[i / self.N] * len(Xlist[i]) for i in range(self.N-1)]

        ##############################################################

        fig, ax = plt.subplots(figsize=(sizex, sizey))

        for X, Y in zip(Xlist, Ylist):
            ax.plot(X, Y, label='', color='black')
        ax.set_title('Эмпирическая функция распределения')
        ax.legend(loc='upper left')
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        ax.set_xlim(xmin=sr[0], xmax=sr[-1])
        ax.set_ylim(ymin=0, ymax=1)
        fig.tight_layout()
        if sizex != 5:
            ax.set_axisbelow(True)
            ax.minorticks_on()
            ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
            ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        else:
            ax.grid()
        if show:
            plt.show()
        else:
            path = 'Data/' + self.distribution_name + '_cdf.png'
            plt.savefig(path)
            plt.clf()

    def create_theoretical_stats(self, mode_generator):
        mean, variance, skew, kurtosis = self.distribution.stats(moments='mvsk')
        med = self.distribution.median()
        std = self.distribution.std()
        mode = mode_generator(*self.params)

        self.theoretical_stats = self.Stats(mean=float(mean),
                                            variance=float(variance),
                                            std=std,
                                            mode=mode,
                                            med=med,
                                            skew=float(skew),
                                            kurtosis=float(kurtosis))

    def create_experimental_stats(self, exp_stats_generator):
        mean, variance, std, mode, med, skew, kurtosis = exp_stats_generator(self.sec_counter,
                                                                             self.first_counter,
                                                                             self.h)
        self.experimental_stats = self.Stats(mean=mean,
                                            variance=variance,
                                            std=std,
                                            mode=mode,
                                            med=med,
                                            skew=skew,
                                            kurtosis=kurtosis)

    def save_rvs(self, shape: int = 20, dec = 4):
        rvs = np.array(self.rvs)
        rvs = rvs.reshape(shape, -1)
        rvs = np.round(rvs, dec)
        df_rvs = pd.DataFrame(rvs,
                              columns=list(string.ascii_lowercase)[:rvs.shape[1]])
        df_rvs.to_csv('Data/' + self.distribution_name + '_rvs.csv',
                      sep=';',
                      encoding='utf-8',
                      index=False)

        sorted_rvs = np.array(sorted(self.rvs))
        sorted_rvs = sorted_rvs.reshape(shape, -1)
        sorted_rvs = np.round(sorted_rvs, dec)
        df_sorted_rvs = pd.DataFrame(sorted_rvs,
                                     columns=list(string.ascii_lowercase)[:sorted_rvs.shape[1]])
        df_sorted_rvs.to_csv('Data/' + self.distribution_name + '_sorted_rvs.csv',
                             sep=';',
                             encoding='utf-8',
                             index=False)

    def save_dict(self):
        res = {str((round(key[0], 4), round(key[1], 4))): [val]
               for key, val in self.first_counter.items()}
        df = pd.DataFrame(res)
        df.to_csv('Data/' + self.distribution_name + '_first_counter.csv',
                  sep=';',
                  encoding='utf-8')
        res = {round(key, 4): val
               for key, val in self.sec_counter.items()}
        df = pd.DataFrame(res)
        df.to_csv('Data/' + self.distribution_name + '_second_counter.csv',
                  sep=';',
                  encoding='utf-8')

    def save_stats(self, dec = 4):
        experimental = self.experimental_stats._asdict()
        df = pd.DataFrame(list(np.round(list(experimental.values()), dec)),
                          index=experimental.keys(),
                          columns=[['values']])
        df.to_csv('Data/' + self.distribution_name + '_experimental_stats.csv',
                  sep=';',
                  encoding='utf-8')

        theoretical = self.theoretical_stats._asdict()
        df = pd.DataFrame(list(np.round(list(theoretical.values()), dec)),
                          index=theoretical.keys(),
                          columns=[['values']])
        df.to_csv('Data/' + self.distribution_name + '_theoretical_stats.csv',
                  sep=';',
                  encoding='utf-8')

        all_stats = list(zip(self.experimental_stats._asdict().keys(), list(zip(self.experimental_stats, self.theoretical_stats))))
        result = {key: (round(e, dec), round(t, dec), round(abs(e-t), dec), round(abs(e-t)/t, dec))
                  for key, (e, t) in all_stats}
        df = pd.DataFrame(list(result.values()),
                          index=list(result.keys()))
        df.to_csv('Data/' + self.distribution_name + '_all_stats.csv',
                  sep=';',
                  encoding='utf-8')

    def create_and_save_p(self, dec = 4):
        cur_r = self.distribution
        cur_x = sorted(list(self.first_counter.keys()))
        cur_w = [self.sec_counter[key][1]
                 for key in sorted(list(self.sec_counter.keys()))]


        p = [cur_r.cdf(b) - cur_r.cdf(a) for a, b in cur_x]
        s = sum(p)
        wp = [abs(a - b) for a, b in zip(cur_w, p)]
        m = max(wp)
        new_w = deepcopy(cur_w)
        new_w.append(1.0)
        p.append(s)
        wp.append(m)
        new_x = deepcopy(cur_x)
        new_x.append('-')

        df = pd.DataFrame(np.round(np.array([new_w, p, wp]).T, 4),
                          index=new_x)
        df.to_csv(self.distribution_name + '_wp' + '.csv',
                  sep=';',
                  encoding='utf-8')


def fitter(obj, mode_generator):
    obj.create_rvs()
    obj.create_first_counter()
    obj.create_second_counter()
    obj.hist()
    obj.cdf()
    obj.create_theoretical_stats(mode_generator)
    obj.create_experimental_stats(expStats)
    obj.save_rvs()
    obj.save_dict()
    obj.save_stats()
    obj.create_and_save_p()
    return obj


if __name__ == '__main__':
    variant = 9
    N = 200

    mode = {'norm': lambda a, sgm: a,
            'expon': lambda lmbd: 0,
            'uniform': lambda a, b: sum([a, b])/2}

    Obj = namedtuple('Obj', 'name params distribution')

    mu = (-1) ** variant * 0.1 * variant
    sgm = (0.01 * variant + 1) ** 2

    lmbd = 2 + (-1) ** variant * 0.01 * variant

    a = (-1) ** variant * 0.05 * variant
    b = a + 0.05 * variant + 1

    print('mu = {0}\nsgm = {1}\nlmbd = {2}\na = {3}\nb = {4}'.format(mu,
                                                                     sgm,
                                                                     lmbd,
                                                                     a,
                                                                     b))

    Data = [Obj(name='norm',
                params=[mu, sgm],
                distribution=st.norm(mu, sgm)),
            Obj(name='expon',
                params=[1/lmbd],
                distribution=st.expon(1/lmbd)),
            Obj(name='uniform',
                params=[a, b],
                distribution=st.uniform(a, b))]

    result = []
    for obj in Data:
        result.append(LabFitter3000(obj.distribution, N, obj.name, *obj.params))

    result = [fitter(lf, mode[Data[i].name]) for i, lf in enumerate(result)]
