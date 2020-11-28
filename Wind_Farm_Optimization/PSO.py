# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 17:04:58 2020

@author: abhay
"""

import numpy as np
import pandas as pd
import numpy.random as rand
import shutil
import os
# COST Function we will use the given cst calculator
from Farm_Evaluator_Vec_copy import AEP, binWindResourceData, loadPowerCurve
import matplotlib.pyplot as plt

# MAIN CLASS PARTICLES
# Each particle is a wind farm layout of lenght 4000x4000 dimensions
# Each windmill in the particle is bounded between (50,3950)
# The distance between each windmill should be at a distance 400 m

class Particle:
    def __init__(self,n_turbines = 50):
        # ABout thte particle
        self.length = 4000
        self.boundary = 50
        self.separation = 400
        self.n_turbines = n_turbines
        
        # fitness of the particle
        self.fitness = 0

        # personal best
        self.p_best_x = np.zeros(50)
        self.p_best_y = np.zeros(50)
        
        self.best_aep = 0
        self.best_fitness = None
        
        #Initially we will assign random coordinates to the turbines
        #self.position_x = np.ones(n_turbines)*self.boundary + rand.uniform(size = n_turbines) * (self.length - 2* self.boundary)
        #self.position_y = np.ones(n_turbines)*self.boundary + rand.uniform(size = n_turbines) * (self.length - 2* self.boundary)
        self.position_x = np.random.uniform(50,3950,50)
        self.position_y = np.random.uniform(50,3950,50)
        self.position_x,self.position_y = self.calc_postion(np.zeros(self.n_turbines), np.zeros(self.n_turbines),
                                                            np.zeros(self.n_turbines), np.zeros(self.n_turbines))
        
        
    def calc_postion(self, v1,v2, g_best_x, g_best_y):
        new_pos_x = self.position_x + np.multiply(v1[0], (g_best_x - self.position_x)) + np.multiply(v2[0], (self.p_best_x - self.position_x))
        new_pos_y = self.position_y + np.multiply(v1[1], (g_best_y - self.position_y)) + np.multiply(v2[1], (self.p_best_y - self.position_y))
        for i in range(len(new_pos_x)):
            for j in range(len(new_pos_x)):
                if (i == j):
                    continue

                x_1, y_1 = new_pos_x[i],new_pos_y[i]
                x_2, y_2 = new_pos_x[j],new_pos_y[j]
                dist = np.sqrt((x_1 - x_2)** 2 + (y_1 - y_2)** 2)
                if (dist < self.separation):
                    radius = (self.separation - dist) / 2
                    angle = np.arctan((y_2 - y_1) / (x_2 - x_1 + 1e-3))
                    # turn = np.random.uniform(low = -1.57, high = 1.57)
                    turn = 0
                    c=(x_2 * y_1 - x_1 * y_2) / (x_2 - x_1 + 1e-3)
                    
                    s_x_1 = x_1
                    s_y_1 = y_1 + c
                    s_x_2 = x_2
                    s_y_2 = y_2 + c

                    r_x_1 = s_x_1 * np.cos(angle) + s_y_1 * np.sin(angle) + radius*np.cos(turn)
                    r_y_1 = s_y_1 * np.cos(angle) - s_x_1 * np.sin(angle) + radius*np.sin(turn)
                    r_x_2 = s_x_2 * np.cos(angle) + s_y_2 * np.sin(angle) - radius*np.cos(turn)
                    r_y_2 = s_y_2 * np.cos(angle) - s_x_2 * np.sin(angle) - radius * np.sin(turn)
                    
                    [new_pos_x[i],new_pos_y[i]] = [r_x_1 * np.cos(angle) - r_y_1 * np.sin(angle), r_x_1 * np.sin(angle) + r_y_1 * np.cos(angle) - c]
                    [new_pos_x[j],new_pos_y[j]] = [r_x_2 * np.cos(angle) - r_y_2 * np.sin(angle), r_x_2 * np.sin(angle) + r_y_2 * np.cos(angle) - c]

            new_pos_x[i] = min(new_pos_x[i], self.length - self.boundary)
            new_pos_y[i] = min(new_pos_y[i], self.length - self.boundary)
            new_pos_x[i] = max(self.boundary, new_pos_x[i])
            new_pos_y[i] = max(self.boundary, new_pos_y[i])
                
        return new_pos_x,new_pos_y
    
    def calc_violation(self,position_x,position_y):
        cost = 0
        for i in range(self.n_turbines) :
            x1, y1 = position_x[i],position_y[i]
            # Check boundary constraints
            if x1 < self.boundary :
                cost = cost + (self.boundary - x1)**2
            if y1 < self.boundary :
                cost = cost + (self.boundary - y1)**2
            if x1 > self.length - self.boundary :
                cost = cost + (self.length - self.boundary - x1)**2
            if y1 > self.length - self.boundary :
                cost = cost + (self.length - self.boundary - y1)**2
            
            for j in range(self.n_turbines) :
                if i == j : 
                    continue
                x2 , y2 = position_x[j],position_y[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < self.separation:
                    cost =cost + self.separation - dist
        return cost
    

# THe SWARM CLASS

class PSO:
    def __init__(self, n_turbines = 50,n_particles = 100, iterations = 50):
        # Will will load the power curve of the turbine and the wind data
        self.power_curve = loadPowerCurve(r'Shell_Hackathon Dataset/power_curve.csv')
        self.wind_data = binWindResourceData(r'Shell_Hackathon Dataset/Wind Data/wind_data_2017.csv')
        
        self.total_iterations = iterations
        self.n_particles = n_particles
        self.n_turbines = n_turbines
        
        self.g_best_x = []
        self.g_best_y = []
        
        self.g_best_aep = 0
        self.best_aep = []# for logging the aeps to graph
        self.g_best_fitness = None
        self.counter = 0
        #Loading The particles to the swarm
        self.particles = [Particle() for i in range(n_particles)]
        
    def calc_AEP(self, position_x, position_y):
        return AEP(position_x, position_y, self.power_curve, self.wind_data)
    
    def calc_fitness(self, aep, violation_cost):
        if violation_cost != 0:
            return -violation_cost
        else:
            return aep
    
    def log_best_plots(self,g_best_x,g_best_y,count):
        if count < 1:
            shutil.rmtree("globalbest")
            os.mkdir("globalbest")
        data = pd.DataFrame()
        data['x'] = g_best_x
        data['y'] = g_best_y
        data.to_csv("globalbest/iteration_"+str(count) + ".csv",index = False)
    
    def iterate(self):
        for i in range(self.n_particles):
            v1 = np.random.uniform(low = 0, high = 1, size = [2,self.n_turbines])
            v2 = np.random.uniform(low = 0, high = 1, size = [2,self.n_turbines])
            new_pos_x,new_pos_y = self.particles[i].calc_postion(v1,v2,self.g_best_x, self.g_best_y)
            
            violation_cost = self.particles[i].calc_violation(
                self.particles[i].position_x, self.particles[i].position_y)
            
            new_aep = self.calc_AEP(new_pos_x, new_pos_y)
            new_fitness = self.calc_fitness(new_aep, violation_cost)
            # Update particle fitness
            if new_fitness > self.particles[i].best_fitness:
                self.particles[i].best_fitness = new_fitness
                self.particles[i].best_aep = new_aep
                self.particles[i].p_best_x = new_pos_x
                self.particles[i].p_best_y = new_pos_y
            # Update global fitness
            if (new_fitness > self.g_best_fitness) :
                self.g_best_fitness = new_fitness
                self.g_best_aep = new_aep
                self.g_best_x = new_pos_x
                self.g_best_y = new_pos_y
                
            
            self.particles[i].fitness = new_fitness
            
            #self.particles[i].pos_history.append(self.particles[i].pos) # check these and local bests also
            self.particles[i].position_x = new_pos_x
            self.particles[i].position_y = new_pos_y
        self.best_aep.append(new_aep)
            
    
    def run(self):
        #initializing the variables(iteration = 0)
        for particle in self.particles:
            violation_cost = particle.calc_violation(
                particle.position_x, particle.position_y)
            aep = self.calc_AEP(particle.position_x, particle.position_y)
            particle.fitness = self.calc_fitness(aep, violation_cost)
            # Updating the initial global best
            if self.g_best_fitness is None or particle.fitness > self.g_best_fitness:
                self.g_best_fitness = particle.fitness
                self.g_best_x = particle.position_x
                self.g_best_y = particle.position_y
                self.g_best_aep = aep
            particle.best_fitness = particle.fitness
            particle.best_aep = aep
            particle.p_best_x = particle.position_x
            particle.p_best_y = particle.position_y
            
        self.best_aep.append(aep)
        self.log_data()
        
        self.log_best_plots(self.g_best_x, self.g_best_y, self.counter)
        self.counter =self.counter + 1
        
        while self.counter < self.total_iterations:
            self.iterate()
            self.log_data()
            self.log_best_plots(self.g_best_x, self.g_best_y, self.counter)
            self.counter = self.counter + 1
        self.log_data()
        self.plotter(self.g_best_x, self.g_best_y, self.best_aep, self.total_iterations)
        
    
    def log_data(self):
        print("Iteration: ", self.counter, "Best Fitness: ", self.g_best_fitness, "Best AEP: ", self.g_best_aep)
    
    def plotter(self,x,y,aep,iterations):
        fig,ax = plt.subplots(1, 2, figsize=(10, 5))
        
        a = range(iterations)
        print(a,aep)
        ax[0].set_xlabel("X (m)")
        ax[0].set_ylabel("Y (m)")
        ax[0].axis(xmin=0,xmax=4000)
        ax[0].axis(ymin=0,ymax=4000)
        ax[0].axhline(y=3950, color='r', linestyle='-')
        ax[0].axhline(y=50, color='r', linestyle='-')
        ax[0].axvline(x=3950, color='r', linestyle='-')
        ax[0].axvline(x=50, color='r', linestyle='-')
        #ax[0].set_title('Iterations : ' + str(i))
        ax[0].scatter(x,y)
        
        ax[1].set_ylabel('No of iterations')
        ax[1].set_xlabel('Annual Energy Production')
        ax[1].plot(a,aep)
        
        fig.tight_layout()
        plt.show()
        


