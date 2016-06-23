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
from Configuration import filename, walls, dx, dp, X, P, T

N = 4
M = 4

xmin = .4
xmax = .45
pmin = .1
pmax = .15

def build_pos(x0, p0):
    position = x0 * cosh(np.sqrt(k/m)*T) +\
    p0 * sinh(np.sqrt(k/m) * T)/(np.sqrt(k * m))
    return (lambda t: position.subs(T, t))
def build_mom(x0, p0):
    momentum = p0 * cosh(np.sqrt(k/m)*T) +\
    np.sqrt(k * m) * x0 * sinh(np.sqrt(k/m) * T)
    return (lambda t: position.subs(T, t))
    
def step(time):
    pass

if __name__ == "__main__":
    print "hey!"