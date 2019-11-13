
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import argparse

import numpy as np
from numpy.linalg import matrix_power, solve
import pandas as pd
from random import choices

from itertools import product
from copy import copy
from inspect import isgeneratorfunction

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default='Data/input.csv', type=str)
    return parser

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
            return np.round(output, rn)
        return rounder


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


@round_output
def n_delta_powers_from_2(P, dm=1e-5):
    Pn = copy(P)
    d = 2 * dm
    while d >= dm:
        d = np.max(np.absolute(Pn @ P - Pn))
        Pn = Pn @ P
        yield Pn, d


def save_table_csv(table, path, columns, index, convert=True):
    df = pd.DataFrame(np.array(table).T if convert else table,
                      index=index,
                      columns=columns)
    df.to_csv(path+'.csv', sep=';', encoding='utf-8')


@round_output
def get_stationary_d(P):
    A = np.concatenate(((P.T - np.identity(3))[:-1], np.ones((1, 3))), axis=0)
    B = np.array([0, 0, 1])
    x = solve(A, B)
    return x


@round_output
def p_distributions(start, P, stationary, dm=1e-5):
    Pn = copy(P)
    x = copy(start)
    d = np.max(np.absolute(x - stationary))
    while d >= dm:
        d = np.max(np.absolute(x @ P - stationary)) # Pn
        x = start @ Pn
        Pn = Pn @ P
        yield x, d

@round_output
def p_distribution(start, P, stationary, n):
    Pn = copy(P)
    for i in range(n):
        x = start @ Pn
        Pn = Pn @ P
    return x


def generate_next_state(current_state_id, P):
    return choices(population=[0, 1, 2], weights=P[current_state_id])[0]

@round_output
def generate_n_states(start_state_id, P, stationary, dm=1e-3):
    current_state_id = start_state_id
    count_start_state = 0
    N = dm * 2
    n = 0
    while N >= dm:
        current_state_id = generate_next_state(current_state_id, P)
        n += 1
        count_start_state += 1 if current_state_id == start_state_id else 0
        v = count_start_state / n
        N = abs(v - stationary[start_state_id])
        yield count_start_state, v, N

if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    # ### Подготовка

    np.set_printoptions(suppress=True)

    df = pd.read_csv(namespace.path, encoding='utf-8', sep=';', dtype=np.float)

    P = df.values

    # #### 0) Построение графа

    plot_graph(P, 'Data/test')

    # #### 1) Матрицы переходных вероятностей

    ndp = [[np.round(P, 5).tolist(), None]]

    for Pn, d in n_delta_powers_from_2(P):
        ndp.append([Pn.tolist(), d])

    save_table_csv(ndp, 'Data/ndp',
                   columns=['Pn', 'd'],
                   index=range(1, len(ndp)+1),
                   convert=False)

    print(f'n_min = {len(ndp)}')

    # #### 2) Стационарное распределение вероятностей

    x = get_stationary_d(P)
    print(x)
    print(np.round(x @ P, 5) == x)

    # #### 3) Распределения вероятностей состояний через n шагов

    starts = np.identity(3)

    distr = [[[starts[i].tolist(), np.max(np.absolute(starts[i] - x))]] for i in range(starts.shape[0])]

    for i in range(starts.shape[0]):
        for p, d in p_distributions(starts[i], P, x):
            distr[i].append([p.tolist(), d])

    for i, dist in enumerate(distr):
        save_table_csv(dist, f'Data/distr{i}',
                       columns=['(p1(n), p2(n), p3(n))', 'd(n)'],
                       index=range(len(dist)),
                       convert=False)

    # #### 4) и 5) Генерация последовательности номеров состояний через n шагов

    generates = [[], [], []]
    for i, generation in enumerate(generates):
        for R, v, d in generate_n_states(i, P, x):
            generation.append((R, v, d))

    for i, generation in enumerate(generates):
        if len(generation) > 16:
            save_table_csv(generation[0:10]+generation[-6:], f'Data/generation{i}',
                           columns=['R', 'v', 'd'],
                           index=list(range(1, 11)) + list(range(len(generation)-5, len(generation)+1)),
                           convert=False)
        else:
            save_table_csv(generation, f'Data/generation{i}',
                           columns=['R', 'v', 'd'],
                           index=range(len(generation)),
                           convert=False)

    for i, generation in enumerate(generates):
        print(f'Nmin for state {i+1} = {len(generation)}')

    # #### Анализ результатов и выводы

    print(f'Стационарное распределение: {tuple(x)}\n')
    k = len(ndp)
    print(f'Строки Матрицы P^{k}:')
    print(np.round(matrix_power(P, k), 5))

    for dist in distr:
        print(f'(p1({len(dist)}), p2({len(dist)}), p3({len(dist)})) = {tuple(dist[-1][0])}')

    pn = np.concatenate((np.array(distr[0][-1][0]).reshape(1, -1),
                         np.array(distr[1][-1][0]).reshape(1, -1),
                         np.array(distr[2][-1][0]).reshape(1, -1)))
    d = np.max(np.abs(np.round(matrix_power(P, k), 5) - pn))
    print(f'\nВывод: с погрешностью {d} значения векторов совпадают')
