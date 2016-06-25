# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 19:30:24 2016

@author: anie

Exact solver which outputs position and momentum in the inverted harmonic
oscillator potential in a box problem.
"""
go = raw_input('Solver.py overwrites data in <filename>.py, which may contain data from an Integrator.py simulation.\n Do you want to continue? [y/n]')
assert(go == 'y')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brenth, newton
from sympy import *
import pdb
from Configuration import walls, wall_x, wall_v, k, m

filename = 'quad_solver_1000_fast'

N = 50
M = 20

xmin = .4
xmax = .405
pmin = .005
pmax = .01

dt = .01

pt_arr = np.array([])

def build_pos(x0, p0, T):
    return (lambda t: x0 * np.cosh(np.sqrt(k/m) * (t-T)) +\
                        p0 * np.sinh(np.sqrt(k/m) * (t-T))/(np.sqrt(k * m)))
def build_mom(x0, p0, T):
    return (lambda t: p0 * np.cosh(np.sqrt(k/m) * (t-T)) +\
                    np.sqrt(k * m) * x0 * np.sinh(np.sqrt(k/m) * (t-T)))    
def step(pt_id, time, repeat = False):
    particle = pt_arr[pt_id]
    timeslice = np.array([particle[0](time), particle[1](time), time])
    if np.abs(timeslice[0]) >= np.abs(wall_x(time)):
        # new function for root-finding:
        f = lambda t: particle[0](t) - np.sign(particle[0](t)) * wall_x(t)
        # numerically calculate collision time
        t_col = brenth(f, time - dt, time)
        if repeat:
            try:
                t_col1 = brenth(f, time - dt, t_col)
            except:
                pass
        # calculate collision position and momentum
        x_col = particle[0](t_col)
        p_col = 2 * m * np.sign(x_col) * wall_v(t_col) - particle[1](t_col)
        # build new particle
        new_particle = (build_pos(x_col, p_col, t_col), 
                        build_mom(x_col, p_col, t_col), pt_id)
        if repeat:
            pdb.set_trace()
            new
        pt_arr[pt_id] = new_particle
        # may cause an infinite loop?
        return step(pt_id, time, True)
    return timeslice

if __name__ == "__main__":
#    pt0 = (build_pos(xmin, pmin, 0), build_mom(xmin, pmin, 0), 0)
#    pt1 = (build_pos(xmax, pmax, 0), build_mom(xmax, pmax, 0), 1)
    pt_id = 1
    pt_arr = np.array([(build_pos(xmax, pmax, 0), 
                        build_mom(xmax, pmax, 0), 0)])
    for x in np.arange(xmin, xmax, (xmax-xmin)/N):
        for p in np.arange(pmin, pmax, (pmax-pmin)/M):
            particle = (build_pos(x, p, 0), build_mom(x, p, 0), pt_id)
            pt_id += 1
            pt_arr = np.vstack((pt_arr, particle))
    data = []
    for pt_id in np.arange(0, N * M + 1):
        particle = pt_arr[pt_id]
        pt_data = np.array((particle[0](0), particle[1](0), 0))
        for t in np.arange(dt, 150, dt):
            timeslice = step(pt_id, t)
            pt_data = np.vstack((pt_data, timeslice))
        data.append(pt_data)
    data = np.array(data)
    np.save(filename, data)
    data = None
        
