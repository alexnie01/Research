# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 12:33:44 2016

@author: anie

Methods for performing fourth-order symplectic integration given an initial
array of points. Main method runs simulation using a fourth-order symplectic 
integrator and then exports to <filename>.npy
"""

import numpy as np
import pandas as pd
from sympy import *
import matplotlib.pyplot as plt
import pdb
from Configuration import filename, walls, dx, dp, X, P, T

if walls:
    from Configuration import a, wall_x, wall_v

""" Fourth Order Symplectic Integrator Coefficients """
prefactor = 1./(2 - np.power(2, 1./3))
c = .5 * prefactor * np.array([1, 1-np.power(2, 1./3), 1-np.power(2, 1./3), 1])
d = prefactor * np.array([1, -np.power(2, 1./3), 1, 0])

def step(dt, x0, p0, time):
    ''' advance simulation by timesteps dt'''
    x = x0
    p = p0
    t = time
    for i in range(0, len(c)):
        x += dt * c[i] * dx.subs(P, p)
        t += dt
        p += dt * d[i] * dp.subs(X, x)
        t += dt
    return (x, p, t)

def wall_step(dt, x0, p0, time):
    ''' symplectic step for system with a wall
        Note: this calculates the wall position at every timestep for every 
        point and is REALLY slow. Check the integrator which proceeds by 
        timestep rather than for each point.
    '''
    x = x0
    p = p0
    t = time
    for i in range(0, len(c)):
        x_step = dt * c[i] * dx.subs(P, p)
        x += x_step
        # collision detection
        if np.abs(x) >= np.abs(wall_x.subs(T, t)):
            x -= x_step
            p = np.sign(x) * 2 * wall_v.subs(T, t) - p
        t += dt
        p += dt * d[i] * dp.subs(X, x)
        t += dt
    return (x, p, t)
    
def point_run(x0, p0, t, dt):
    ''' simulate trial for a single point '''
    time = 0
    x = x0
    p = p0
    data = np.array([[x0, p0, time]])
    while time <= t:
        if walls:
            x, p, time = wall_step(dt, x, p, time)
        else:
            x, p, time = step(dt, x, p, time)
        data = np.append(data, [[x, p, time]], axis=0)
    return data
    
def hline_run(N, xmin, xmax, p, t, dt):
    ''' simulate trials for a row of points. 
        access time slices with data[:, t] '''
    points = np.arange(xmin, xmax, float(xmax-xmin)/N)
    points = map(lambda dum: (dum, p), points)
    data = []
    for point in points:
        point_data = point_run(point[0], point[1], t, dt)
        data.append(point_data)
    data = np.array(data)

    return data
def vline_run(N, x, pmin, pmax, t, dt):
    ''' simulate trials for a column of points 
        access time slices with data[:, t]'''
    points = np.arange(pmin, pmax, float(pmax-pmin)/N)
    points = map(lambda dum: (x, dum), points)
    data = []
    for point in points:
        point_data = point_run(point[0], point[1], t, dt)
        data.append(point_data)
    data = np.array(data)
    return data
def square_run(N, M, xmin, xmax, pmin, pmax, t, dt):
    ''' simulate trials for a square of n x m points '''
    data = hline_run(N, xmin, xmax, pmin, t, dt)
    pt = 1
    for p in np.arange(pmin+(pmax-pmin)/M, pmax, (pmax-pmin)/M):
        line_data = hline_run(N, xmin, xmax, p, t, dt)
        data = np.append(data, line_data, axis=0)
        print "finished row {}".format(pt)
        pt += 1
    return data
if __name__ == "__main__":
    print "running trials"
    
    # run for a single point
#    data = point_run(0,.1, 10, .001)
#    plt.plot(data[:, 0], data[:, 1])
    
    # run for a line of points
#    data = hline_run(50, -.3, .3, 0, 5000, 1)
#     plot by initial point
#    for i in range(0, len(data)):
#        plt.plot(data[i, :, 0], data[i, :, 1])
    # plot by time slice
#    for i in range(len(data[0, :, 2])):
#        plt.plot(data[:, i, 0], data[:, i, 1])


    # run for square of points
#    data = square_run(20, 20, .8, 1.3, -.3, .3, 4000, .4)
    data = square_run(10, 10, .05, .1, .1, .2, 200, .01)
    np.save(filename, data)
    data = None
    