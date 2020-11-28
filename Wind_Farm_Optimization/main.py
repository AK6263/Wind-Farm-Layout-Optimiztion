# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 21:31:42 2020

@author: abhay
To run
py main.py --t 50 --p 50 --i 50
"""

from PSO import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--turbines", "--t", help = "Enter the number of turbines",type = int)
parser.add_argument("--particles", "--p", help = "Enter the number of particles",type = int)
parser.add_argument("--iterations", "--i", help = "Enter the number of iterations",type = int)
args = parser.parse_args()
#num_particles, max_iterations = list(map(int, input().split(',')))
n_turbines, num_particles,  iterations = args.turbines, args.particles, args.iterations
swarm = PSO(n_turbines = 50,n_particles=num_particles,iterations= iterations)
swarm.run()