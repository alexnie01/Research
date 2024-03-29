# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 12:33:44 2016

@author: anie

Run inverted harmonic potential in a boxsimulation using a fourth-order 
symplectic integrator and then export to .npy
"""

import numpy as np
from sympy import *
import matplotlib.pyplot as plt
import pdb

# save file for data
filename = "quad_int_long"

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
delta = 1./16

# potential function
X, P, T = symbols('X P T')
# inverted harmonic
U = - .5 * k * X ** 2
# cubic
#U = a * (X**3/3. - X ** 2/2.)

# positive wall position as a function of time
wx = delta * sin(omega * T)
wv = diff(wx, T)

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
        Dx = dt * c[i] * dx.subs(P, p)
        try:
            if np.abs(x+Dx) >= np.abs(a + wx.subs(T,t)):
                p = -p + 2 * wv.subs(T,t)
            else:
                x += Dx
            t += dt
            p += dt * d[i] * dp.subs(X, x) 
            t += dt
        except:
            pdb.set_trace()
    return (x, p, t)
    
def point_run(x0, p0, t, dt):
    ''' simulate trial for a single point '''
    time = 0
    x = x0
    p = p0
    data = np.array([[x0, p0, time]])
    while time <= t:
        x, p, time = step(dt, x, p, time)
        data = np.append(data, [[x, p, time]], axis=0)
    return data
    
def hline_run(n, xmin, xmax, p, t, dt):
    ''' simulate trials for a row of points. 
        access time slices with data[:, t] '''
    points = np.arange(xmin, xmax, float(xmax-xmin)/n)
    points = map(lambda a: (a,p), points)
    data = []
    pt = 1
    print "starting runs"
    for point in points:
        point_data = point_run(point[0], point[1], t, dt)
        data.append(point_data)
        print "finished calculations for {}".format(pt)
        pt += 1
    data = np.array(data)

    return data
def vline_run(n, x, pmin, pmax, t, dt):
    ''' simulate trials for a column of points 
        access time slices with data[:, t]'''
    points = np.arange(pmin, pmax, float(pmax-pmin)/n)
    points = map(lambda a: (x,a), points)
    data = []
    pt = 1
    print "starting runs"
    for point in points:
        point_data = point_run(point[0], point[1], t, dt)
        data.append(point_data)
        print "finished point {}".format(pt)
        pt += 1
    data = np.array(data)
    return data
def square_run(n, m, xmin, xmax, pmin, pmax, t, dt):
    ''' simulate trials for a square of n x m points '''
    data = hline_run(n, xmin, xmax, pmin, t, dt)
    for p in np.arange(pmin+(pmax-pmin)/m, pmax, (pmax-pmin)/m):
        line_data = hline_run(n, xmin, xmax, p, t, dt)
        data = np.append(data, line_data, axis=0)
    return data
if __name__ == "__main__":
    # run for a single point
#    data = point_run(0,.1, 10, .001)
#    plt.plot(data[:, 0], data[:, 1])
    # run for a line of points
    data = vline_run(25, .1, -.1, .1, 100, .01)
    # plot by initial point
#    for i in range(0, 5):
#        plt.plot(data[i, :, 0], data[i, :, 1])
    # plot by time slice
#    for i in range(len(data[0, :, 2])):
#        plt.plot(data[:, i, 0], data[:, i, 1])
    data = square_run(14, 14, 0, .2, 0, .2, 1000, .1)
    np.save(filename, data)
    data = None
    