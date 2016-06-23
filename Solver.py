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
from sympy import *
import pdb
from Configuration import filename, walls, dx, dp, X, P, T, k, m

N = 2
M = 2

xmin = .4
xmax = .45
pmin = .1
pmax = .15

def build_pos(x0, p0):
    return (lambda t: x0 * np.cosh(np.sqrt(k/m) * t) +\
                        p0 * np.sinh(np.sqrt(k/m) * t)/(np.sqrt(k * m)))
def build_mom(x0, p0):
    return (lambda t: p0 * np.cosh(np.sqrt(k/m) * t) +\
                    np.sqrt(k * m) * x0 * np.sinh(np.sqrt(k/m) * t))    
def step(particle, time):
    timeslice = np.array([particle[0](time), particle[1](time), time])
    pdb.set_trace()
    return timeslice

if __name__ == "__main__":
    pt_id = 1
    particle_array = np.array([(build_pos(xmax, pmax), build_mom(xmax, pmax), 0)])
    for x in np.arange(xmin, xmax, (xmax-xmin)/N):
        for p in np.arange(pmin, pmax, (pmax-pmin)/M):
            particle = (build_pos(x, p), build_mom(x, p), pt_id)
            pt_id += 1
            particle_array = np.vstack((particle_array, particle))
        
