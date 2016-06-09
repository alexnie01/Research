# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 12:33:44 2016

@author: anie

Run simulation using a fourth-order symplectic integrator and then export to
csv
"""

import numpy as np
import pandas as pd
from sympy import *
import matplotlib.pyplot as plt
import pdb

# save file for data
filename = raw_input("Filename? ")

# integrator coefficients (fourth order)
prefactor = 1./(2 - np.power(2, 1./3))
c = .5 * prefactor * np.array([1, 1-np.power(2, 1./3), 1-np.power(2, 1./3), 1])
d = prefactor * np.array([1, -np.power(2, 1./3), 1, 0])

# box configuration
walls = True
moving_walls = True

# constants (spring constant, mass, box size, wall frequency, wall amplitude)
k = 1
m = 1
a = .5
omega = 1
delta = .1

# potential function
X, P = symbols('X P')
# inverted harmonic
U = - .5 * k * X ** 2
# cubic
#U = a * X**3 - X ** 2

# Hamiltonian (inverted harmonic oscillator)
H = .5*P**2/m + U
dx = diff(H, P)
dp = -diff(H, X)

def step(dt, x0, p0, time):
    ''' advance simulation by timesteps dt'''
    x = x0
    p = p0
    t = time
    assert(len(c) == len(d))
    for i in range(0, len(c)):
        x += dt * c[i] * dx.subs(P, p)
        p += dt * d[i] * dp.subs(X, x)
        t += dt * 2
    return (x, p, t)
    
def point_run(x0, p0, t, dt):
    ''' simulate trial for a single point '''
    time = 0
    x = x0
    p = p0
    data = np.array([[x0, p0]])
    while time <= t:
        x, p, time = step(dt, x, p, time)
        data = np.append(data, [[x, p]], axis=0)
    return data
    
def hline_run(n, xmin, xmax, y, t, dt):
    ''' simulate trials for a row of points '''
    pass
def vline_run(n, x, ymin, ymax, t, dt):
    ''' simulate trials for a column of points '''
    pass
def square_run(n, m, xmin, xmax, ymin, ymax, t, dt):
    ''' simulate trials for a square of n x m points '''
    pass
if __name__ == "__main__":
    data = point_run(0,.1, 10, .001)
    plt.plot(data[:, 0], data[:, 1])