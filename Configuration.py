# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 18:38:21 2016

@author: anie

Physical constants and other physical configurations of system. To change 
symplectic integrator configuration or methods, see Integrator.py
"""
from sympy import *
import numpy as np
# save file for data
filename = "cubic_1600_1000_1"

""" Box Configuration for Quadratic Integration """
walls = False
moving_walls = False

# constants (spring constant, mass, box size, wall frequency, wall amplitude)
k = 1
m = 1
a = 1
omega = 1
delta = .1

# potential function
X, P, T = symbols('X P T')

# inverted harmonic
#U = - .5 * k * X ** 2
# cubic
U = - a * (X**3/3. - X ** 2/2.)

# Hamiltonian (inverted harmonic oscillator)
H = .5*P**2/m + U
dx = diff(H, P)
dp = -diff(H, X)

# Wall Position and Velocity
def wall_x(t):
    return a + delta * np.sin(omega * t)
def wall_v(t):
    return - delta * omega * np.sin(omega * t)

