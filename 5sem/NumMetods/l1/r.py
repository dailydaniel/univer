#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import re
import numpy as np

class ryad():
    def __init__(self):
        self.type_ = None
        self.res = None

    def calc(self, x, type_):
        self.type_ = type_
        res = 1
        e = 0.000001
        if self.type_ == 'cos':
            x = self.torad(x)
            current = 1
            prev = 0
            n = 1
            while abs(abs(current) - abs(prev)) > e:
                prev = current
                current *= ((-1) * x*x) / ((2*n-1) * 2 * n)
                res += current
                n += 1
            self.res = res
            return
        elif self.type_ == 'exp':
            current = 1
            prev = 0
            n = 1
            while abs(abs(current) - abs(prev)) > e:
                prev = current
                current *= x / n
                if n == x:
                    prev = current - 2*e
                res += current
                n += 1
            self.res = res
            return

        elif self.type_ == 'log':
            t = 0

            if abs(x) > 1:
                while x >= 1:
                    x /= 10
                    t += 1

            t *= np.log(10)
            x -= 1
            res -= 1

            current = x
            prev = 0
            n = 1
            while abs(abs(current) - abs(prev)) > e:
                prev = current
                res += current
                current *= (-1) * x * n / (n + 1)
                n += 1
            self.res = (res + t) / np.log(10)
            return

    def torad(self, x):
        return (x * math.pi) / 180

    def test(self, x):
        if self.type_ == 'cos':
            test = math.cos(self.torad(x))
            print('cos(x) = {0}, result = {1}'.format(test, self.res))
            if round(test, 3) == round(self.res, 3):
                print('correct')
            else:
                print('nope')
        elif self.type_ == 'exp':
            test = math.exp(x)
            print('exp(x) = {0}, result = {1}'.format(test, self.res))
            if round(test, 3) == round(self.res, 3):
                print('correct')
            else:
                print('nope')
        elif self.type_ == 'log':
            test = np.log(x) / np.log(10)
            print('ln(x) = {0}, result = {1}'.format(test, self.res))
            if round(test, 3) == round(self.res, 3):
                print('correct')
            else:
                print('nope')

if __name__ == "__main__":
    r = ryad()
    str_ = str(input())
    x = float(re.findall(r'\(([0-9\.]+)\)', str_)[0])
    type_ = re.findall(r'^(\w+)\(', str_)[0]
    r.calc(x, type_)
    r.test(x)
