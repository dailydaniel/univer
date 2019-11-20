#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import argparse

from IPython.display import Latex

import numpy as np
from numpy.linalg import matrix_power, solve
from random import choices, uniform
from math import log
import pandas as pd

from itertools import product
from inspect import isgeneratorfunction
from copy import copy

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default='Data/input.csv', type=str)
    return parser

# ### Методы

# Декоратор для округления возвращаемых значений других функций
def round_output(fn):
    rn = 5
    if isgeneratorfunction(fn): 
        def rounder(*args, **kwargs):
            for output in fn(*args, **kwargs):
                if type(output) != tuple:
                    yield np.round(output, rn)
                else:
                    res = []
                    for element in output:
                        res.append(np.round(element, rn))
                    yield tuple(res)
        return rounder
    else:
        def rounder(*args, **kwargs):
            output = fn(*args, **kwargs)
            if type(output) != tuple:
                return np.round(output, rn) 
            else:
                res = []
                for element in output:
                    res.append(np.round(element, rn))
                return tuple(res)
        return rounder

np.set_printoptions(precision=5, suppress=True) # formatter={'float': '{: 0.5f}'.format})
pd.options.display.float_format = '{:,.5f}'.format


def M2W(P):
    graph = {(str(i+1), str(j+1)): round(P[i][j], 5)
             for i, j in product(range(P.shape[0]), repeat=2) if P[i][j] > 0.}
    return graph

def plot_graph(P, path='graph'):
    graph = M2W(P)
    G=nx.MultiDiGraph()

    for edge in graph:
        G.add_edge(edge[0], edge[1])

    G.graph['edge'] = {'arrowsize': '1', 'splines': 'curved'}
    G.graph['graph'] = {'scale': '3'}

    A = to_agraph(G)
    A.layout('neato')#neato, dot, twopi, circo, fdp, nop, wc, acyclic, gvpr, gvcolor, ccomps, sccmap, tred, sfdp, unflatten

    for pair in graph:
        edge = A.get_edge(pair[0], pair[1])
        edge.attr['label'] = str(graph[pair]) + "  "

    A.draw(f'{path}.png')


def save_table_csv(table, path, columns, index, convert=True):
    df = pd.DataFrame(np.array(table).T if convert else table,
                      index=index,
                      columns=columns)
    df.to_csv(path+'.csv', sep=';', encoding='utf-8')


def matrix2latex(matr, toint=False):
    start = r'$$\begin{pmatrix} '
    end = r' \end{pmatrix}$$'
    if not toint:
        body = r' \\ '.join([r' & '.join([str(x) for x in matr[i]]) for i in range(matr.shape[0])]) 
    else:
        body = r' \\ '.join([r' & '.join([str(int(x)) for x in matr[i]]) for i in range(matr.shape[0])]) 
    return start + body + end


def ode_system2latex(P, toint=False):
    matr = P.T
    
    start = r'\[\left\{\begin{array}{ r @{{}={}} r  >{{}}c<{{}} r  >{{}}c<{{}}  r } '
    end = r' \end{array}\right.\]'
    
    frac = lambda i: r'\frac{dp_' + r'{0}'.format(i+1) + r'}{q}' + r'&=&'
    check_first = lambda arr, j: j == [i for i, x in enumerate(arr) if x != 0][0]
    def element(k, j, is_first):
        if not is_first:
            st = r'&+&' if k > 0 else r'&-&' 
        else:
            st = r''
            
        if not toint:
            if abs(k) != 1:
                bd = r'{0}'.format(abs(k)) if not is_first else r'{0}'.format(k) 
            else:
                bd = r''
                if is_first:
                    bd = r'' if k > 0 else r'-'
        else:
            if abs(k) != 1:
                bd = r'{0}'.format(abs(int(k))) if not is_first else r'{0}'.format(int(k)) 
            else:
                bd = r''
                if is_first:
                    bd = r'' if k > 0 else r'-'
    
            nd = r'p_{0}'.format(j+1)
        return st + bd + nd
    
    body = [frac(i) + r' '.join([element(matr[i][j], j, check_first(matr[i], j)) for j in range(matr.shape[1]) if matr[i][j] != 0]) + r'\\' 
            for i in range(matr.shape[0])]
    return start + r' '.join(body) + end


def stationary_system2latex(P, toint=False):
    matr = P.T
    
    start = r'\[\left\{\begin{array}{ r @{{}={}} r  >{{}}c<{{}} r  >{{}}c<{{}}  r } '
    end = r' \end{array}\right.\]'
    
    check_first = lambda arr, j: j == [i for i, x in enumerate(arr) if x != 0][0]
    
    def element(k, j, is_first):
        if not is_first:
            st = r'&+&' if k > 0 else r'&-&' 
        else:
            st = r''
            
        if not toint:
            if abs(k) != 1:
                bd = r'{0}'.format(abs(k)) if not is_first else r'{0}'.format(k) 
            else:
                bd = r''
                if is_first:
                    bd = r'' if k > 0 else r'-'
        else:
            if abs(k) != 1:
                bd = r'{0}'.format(abs(int(k))) if not is_first else r'{0}'.format(int(k)) 
            else:
                bd = r''
                if is_first:
                    bd = r'' if k > 0 else r'-'
    
            nd = r'p_{0}'.format(j+1)
        return st + bd + nd
    
    body = [r' '.join([element(matr[i][j], j, check_first(matr[i], j)) for j in range(matr.shape[1]) if matr[i][j] != 0]) + r' &=& 0' + r'\\' 
            for i in range(matr.shape[0])]
    body += [r' '.join([element(1.0, j, check_first(np.ones(matr.shape[1]), j)) for j in range(matr.shape[1])]) + r' &=& 1' + r'\\']
    return start + r' '.join(body) + end    


@round_output
def get_stationary_d(P):
    A = np.concatenate((P.T[:-1], np.ones((1, P.shape[0]))), axis=0)
    B = np.zeros(P.shape[0])
    B[-1] = 1.
    x = solve(A, B)
    return x


def generate_next_state(current_state_id, P):
    return choices(population=range(5), weights=P[current_state_id])[0]

def generate_time_and_next_state(current_state_id, P):
    weights = [(i, (-1. / x) * log(uniform(1e-10, 1.))) 
               for i, x in enumerate(P[current_state_id]) if x > 0.]
    return sorted(weights, key=lambda x: x[1])[0]
        
@round_output
def generate_n_states(P, stationary, dm=1e-4):
    matr = [[x if x > 0 else 0. for x in P[i]] for i in range(P.shape[0])]
    matr = [[x / sum(matr[i]) for x in matr[i]] for i in range(P.shape[0])]
    matr = np.array(matr)
    
    current_state_id = 0
    prev_state_id = current_state_id
    d = 1.
    t = 0
    K = 0
    v = [np.zeros(5)]
    r = [np.zeros(5)]
    tkidq = [[0, 0, 0, 0, 0]] 
    while d > dm:
        prev_state_id = current_state_id
        current_state_id, cur_q = generate_time_and_next_state(current_state_id, P)
        t += cur_q
        if current_state_id == prev_state_id:
            print('cur = prev')
            continue
        K += 1
        cur_r = copy(r[-1])
        cur_r[current_state_id] = cur_r[current_state_id] + 1
        r.append(cur_r)
        cur_v = copy(v[-1])
        cur_v = cur_r / K #(tkid[-1][1]+1)
        v.append(cur_v)
        d = max(v[-1] - v[-2])
        tkidq.append([t, K, current_state_id+1, d, cur_q])
    return (np.array(tkidq), np.array(r), np.array(v))


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    
    
    # ### 0) Подготовка


    P = np.genfromtxt(namespace.path, comments="%")


    display(Latex(matrix2latex(P, True)))


    # ### 1) Построить граф

    plot_graph(P, 'Data/test')


    # ### 2) СДУ Колмогорова


    display(Latex(ode_system2latex(P, True)))


    # ### 3) СУ для стационарных вероятностей


    display(Latex(stationary_system2latex(P, True)))


    # ### 4) Стационарное распределение


    stationary = get_stationary_d(P)


    display(Latex(matrix2latex(stationary.reshape(1, -1))))


    # ### 5) Генерация


    tkidq, R, v = generate_n_states(P, stationary)


    print(f'K = {int(tkidq[-1][1])}')


    # ### 6) Таблица 1


    table = np.concatenate(
        (np.array([tkidq[1:101, 1], tkidq[1:101, 0], tkidq[1:101, 2], tkidq[1:101, 4], tkidq[1:101, 3]]).T,
         np.array([tkidq[-6:, 1], tkidq[-6:, 0], tkidq[-6:, 2], tkidq[-6:, 4], tkidq[-6:, 3]]).T)
    )


    table1 = pd.DataFrame(table, columns=['l', 't', 'C', 'tau', 'delta'])

    table1.l = table1.l.astype(int)
    table1.C = table1.C.astype(int)
    table1 = table1.set_index('l')


    table1.to_csv('Data/table1.csv', encoding='utf-8', sep=';')


    # ### 7) Таблица 2


    table = []

    for i in range(5):
        table.append([i+1, R[100][i], v[100][i], 
                      sum([t for t, _, state, _, _ in tkidq[1:101] if state-1 == i]),
                      sum([t for t, _, state, _, _ in tkidq[1:101] if state-1 == i]) / sum(tkidq[1:101, 0])
                     ])
    table = np.round(np.array(table), 5)


    table2 = pd.DataFrame(table, columns=['i', 'R', 'v', 'T', 'delta'])

    table2.i = table2.i.astype(int)
    table2.R = table2.R.astype(int)
    table2 = table2.set_index('i')


    table2.to_csv('Data/table2.csv', encoding='utf-8', sep=';')


    # ### 8) Таблица 3


    table = []

    for i in range(5):
        table.append([i+1, R[-1][i], v[-1][i], 
                      sum([t for t, _, state, _, _ in tkidq[1:] if state-1 == i]),
                      sum([t for t, _, state, _, _ in tkidq[1:] if state-1 == i]) / sum(tkidq[1:, 0])
                     ])
    table = np.round(np.array(table), 5)


    table3 = pd.DataFrame(table, columns=['i', 'R', 'v', 'T', 'delta'])

    table3.i = table3.i.astype(int)
    table3.R = table3.R.astype(int)
    table3 = table3.set_index('i')


    table3.to_csv('Data/table3.csv', encoding='utf-8', sep=';')


    # ### Анализ


    table = []

    for i in range(5):
        table.append([stationary[i], v[100][i], v[-1][i], 
                      sum([t for t, _, state, _, _ in tkidq[1:101] if state-1 == i]) / sum(tkidq[1:101, 0]),
                      sum([t for t, _, state, _, _ in tkidq[1:] if state-1 == i]) / sum(tkidq[1:, 0])
                     ])
    table = np.round(np.array(table).T, 5)


    table4 = pd.DataFrame(table, columns=['1', '2', '3', '4', '5'])

    table4.to_csv('Data/table4.csv', encoding='utf-8', sep=';')