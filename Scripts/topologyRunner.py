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
    def __init__(self, seed, dg, tau, moleculeInflux, arguments):
        np.random.seed(seed)
        self.seed=seed
        self.DG = dg
        self.moleculeInflux=moleculeInflux
        self.tau=tau
        self.Food1 = self.DG[0][0]
        self.arguments=arguments

    @staticmethod
    def truncated_normal_distribution(moleculeInflux, molAmount, size, repetition=5):
        size = int(size / repetition)
        mu, sigma, lower, upper = 0.5, 0.25, 0, 1
        a, b = (lower - mu) / sigma, (upper - mu) / sigma
        #truncated_normal = truncnorm(a, b, loc=mu, scale=sigma).rvs(size, random_state=seed) * moleculeInflux
        truncated_normal = truncnorm(a, b, loc=mu, scale=sigma).rvs(size) *moleculeInflux #10 #20 #60
        return np.repeat(np.array(truncated_normal), repetition)

    def main(self):
        input_graphs, input_rules = self.DG[0], self.DG[1]
        sim = simFunc(self.arguments)
        sim.Food1=self.Food1
        modReg = modRegression()
        sim.inputGraphs, sim.inputRules = input_graphs, input_rules
        #logger.info(f"reaction rates: {sim.reactionRatesList}")
        #print("before: sim.reactionRatesList", sim.reactionRatesList)
        sim.reactionRatesDict = {input_rules[i]: float(rate)*sim.reactionRateScale for i, rate in enumerate(sim.reactionRatesList)}
        '''
        {input_rules[i]: float(rate)*sim.reactionRateScale*2.0 if i==0 else float(rate)*sim.reactionRateScale for i, rate in enumerate(sim.reactionRatesList)}
        '''
        #print("after", sim.reactionRatesDict)
        sim.numberOfMolecules = len(input_graphs)
        sim.numberOfReactions = len(input_rules)
        sim.initializedMolecules = [self.Food1]
        sim.initializeIRate()
        sim.initializeMoleculeLabels()
        initialStates = {self.Food1: 0}
        sim.totalRunTime = 50 #100 #50 #30 #150 #40

        inflow = topologyRunner.truncated_normal_distribution(self.moleculeInflux, len(input_graphs), sim.totalRunTime, repetition=2)
        #logger.info(f"inflow: {inflow}")
        sim.generateRates(inflow, rStart=10, rEnd=100)
        #simTimeBegin = time.time()
        traces=sim.simulation(input_rules, input_graphs, initialStates)
        #simTimeEnd = time.time()
        #logger.info(f"simulation overall, {(simTimeEnd-simTimeBegin)}")
        matrix = np.array(sim.moleculeFlow)
        modReg.scalingRate = sim.scalingRate
        modReg.moleculeFlow = matrix
        modReg.inputFlowTimes = sim.inputFlowTimes
        modReg.numberOfMolecules = sim.numberOfMolecules
        #stochRetrievalBegin=time.time()
        #self.modReg.stochData
        stochData = sim.retrieveConcentrationInfoOverTime(traces, sim.numberOfMolecules, sim.totalRunTime)
        #stochRetrievalEnd = time.time()
        #logger.info(f"stoch data retrieval: {(stochRetrievalEnd-stochRetrievalBegin)}")
        #regressionBegin = time.time()
        error = modReg.runLongMemoryTask(stochData, tau=self.tau)
        #regressionEnd = time.time()
        #logger.info(f"regression: {(regressionEnd-regressionBegin)}")
        return error
