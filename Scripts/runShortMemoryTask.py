import mod
from mod import *
import numpy as np
import pandas as pd
import time, random
import sys, os
from simFunc import simFunc  # Simulation functions
from modregClass import modRegression
from matplotlib import pyplot as plt
from scipy.stats import truncnorm
#from logger_setup import logger

class topologyRunner:
    def __init__(self, seed, dg, moleculeInflux, arguments):
        np.random.seed(seed)
        self.DG = dg
        self.moleculeInflux=moleculeInflux
        self.Food1 = self.DG[0][0]
        self.arguments=arguments

    @staticmethod
    def truncated_normal_distribution(moleculeInflux, size, repetition=5):
        size = int(size / repetition)
        mu, sigma, lower, upper = 0.5, 0.25, 0, 1
        a, b = (lower - mu) / sigma, (upper - mu) / sigma
        truncated_normal = truncnorm(a, b, loc=mu, scale=sigma).rvs(size)*moleculeInflux# 60
        return np.repeat(np.array(truncated_normal), repetition)

    def main(self):
        input_graphs, input_rules = self.DG[0], self.DG[1]
        sim = simFunc(self.arguments)
        sim.Food1=self.Food1
        modReg = modRegression()
        sim.inputGraphs, sim.inputRules = input_graphs, input_rules
        sim.reactionRatesDict = {input_rules[i]: float(rate)*sim.reactionRateScale for i, rate in enumerate(sim.reactionRatesList)}
        sim.numberOfMolecules = len(input_graphs)
        sim.numberOfReactions = len(input_rules)
        sim.initializedMolecules = [self.Food1]
        sim.initializeIRate()
        sim.initializeMoleculeLabels()
        initialStates = {self.Food1: 0}
        sim.totalRunTime = 50 #20 #100 #50 #200 #100 #40 #150 #40

        inflow = topologyRunner.truncated_normal_distribution(self.moleculeInflux, sim.totalRunTime, repetition=25)
        sim.generateRates(inflow, rStart=10, rEnd=100)
        traces=sim.simulation(input_rules, input_graphs, initialStates)
        matrix = np.array(sim.moleculeFlow)
        modReg.scalingRate = sim.scalingRate
        modReg.moleculeFlow = matrix
        modReg.inputFlowTimes = sim.inputFlowTimes
        modReg.numberOfMolecules = sim.numberOfMolecules
        stochData = sim.retrieveConcentrationInfoOverTime(traces, sim.numberOfMolecules, sim.totalRunTime)
        error = modReg.runShortMemoryTask(stochData)
        return error
