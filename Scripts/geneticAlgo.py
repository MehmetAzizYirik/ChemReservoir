import numpy as np
from deap import base, creator, tools, algorithms
import random, time
import re
from scipy.stats import sem, t
import pickle
from topologyRunner import topologyRunner
#from runShortMemoryTask import topologyRunner
#from logger_setup import logger

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
class geneticAlgo:
    def __init__(self, dg, seed, num_nodes, edges, logger, tau, moleculeInflux):
        self.logger=logger
        self.tau=tau
        self.moleculeInflux=moleculeInflux
        self.DG = dg
        self.SEED = seed
        self.moleculeAmount = num_nodes
        self.edges = edges
        self.INT_BOUNDS = {
            'seed': (1, 1),
            'scalingRate': (1,10), #(1,1), #(6, 9),
            'moleculeAmount': (self.moleculeAmount, self.moleculeAmount),
            'reactionScale': (1,10), #(1,1),#(3,6),
        }
        self.FLOAT_BOUNDS =  {'outFlow': (0.05, 1.0)} #{'outFlow': (1.0, 5.0)} #(0.05, 0.1)} #(1.0, 10.0)
        self.EDGE_WEIGHT_BOUNDS = (10, 100) #(40, 80) #(5, 100)
        self.NUM_EDGE_WEIGHTS = lambda: int(self.edges)
        #random.seed(self.SEED)
        #np.random.seed(self.SEED)
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.initialize_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.custom_mutate, indpb=0.5)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.objective)
    
    def run_simulation(self, seed, outFlow, scalingRate, moleculeAmount, reactionScale, edge_weights):
        seed = int(seed)
        outFlow = float(outFlow)
        scalingRate = int(scalingRate)
        moleculeAmount = int(moleculeAmount)
        reactionScale = int(reactionScale)
        edge_weights = [float(weight) for weight in edge_weights]
        edge_weights_str = ','.join(map(str, edge_weights))
        arguments = [seed, outFlow, scalingRate, moleculeAmount, reactionScale, edge_weights_str]
        topologyEvaluator=topologyRunner(self.SEED, self.DG, self.tau, self.moleculeInflux, arguments)
        return topologyEvaluator.main()
    
    def compute_confidence_interval(self, scores, confidence=0.95):
        n = len(scores)
        mean_score = np.mean(scores)
        std_error = sem(scores)
        h = std_error * t.ppf((1 + confidence) / 2, n - 1)
        return mean_score, h

    def objective(self, params, num_runs=2, confidence=0.95):
        seed, outFlow, scalingRate, moleculeAmount, reactionScale, *edge_weights = params
        edge_weights = [float(round(float(weight / 100), 3)) for weight in edge_weights]
        scores = [self.run_simulation(seed, outFlow, scalingRate, moleculeAmount, reactionScale, edge_weights) for _ in range(num_runs)]
        mean_score, margin_of_error = self.compute_confidence_interval(scores, confidence)
        return mean_score + margin_of_error,
    
    def initialize_individual(self):
        seed = random.randint(*self.INT_BOUNDS['seed'])
        outFlow = round(random.uniform(*self.FLOAT_BOUNDS['outFlow']), 3)
        scalingRate = random.randint(*self.INT_BOUNDS['scalingRate'])
        moleculeAmount = self.moleculeAmount
        reactionScale = random.randint(*self.INT_BOUNDS['reactionScale'])
        num_edges = self.NUM_EDGE_WEIGHTS()
        edge_weights = [random.randint(*self.EDGE_WEIGHT_BOUNDS) for _ in range(num_edges)]
        return [seed, outFlow, scalingRate, moleculeAmount, reactionScale] + edge_weights
    
    def mutate_integer(self, individual, indpb):
        integer_indices = [0, 2, 3, 4]
        for i in integer_indices:
            if random.random() < indpb:
                step = 1
                key = list(self.INT_BOUNDS.keys())[integer_indices.index(i)] #[integer_indices.index(i)] because the above ones are the index in the individual not in the int_bound dict.
                low, high = self.INT_BOUNDS[key]
                individual[i] = max(low, min(high, individual[i] + random.choice([-step, step])))
        return individual
    
    def mutate_float(self, individual, indpb):
        if random.random() < indpb:
            step = 0.2 #1.0 #2.0 #0.1
            low, high = self.FLOAT_BOUNDS['outFlow']
            individual[1] = max(low, min(high, individual[1] + random.choice([-step, step])))
        individual[1] = round(individual[1], 3)
        return individual
    
    def mutate_edge_weights(self, individual, indpb):
        low, high = self.EDGE_WEIGHT_BOUNDS
        num_edges = self.NUM_EDGE_WEIGHTS()
        for i in range(5, 5 + num_edges):
            if random.random() < indpb:
                step = 10
                individual[i] = max(low, min(high, individual[i] + random.choice([-step, step])))
        return individual
    
    def custom_mutate(self, individual, indpb=0.5):
        individual = self.mutate_integer(individual, indpb)
        individual = self.mutate_float(individual, indpb)
        individual = self.mutate_edge_weights(individual, indpb)
        return individual,
    
    def main(self, population_size=4, elite_size=2, max_generations=10, max_time=500): #200 for 40 top run
        #3, 1 population elite
        population = self.toolbox.population(n=population_size)
        #logger.info(f"population: {len(population)}")
        start_time = time.time()
        prevMinValue=0.0
        count=0
        worseCount=0
        neighborhood_threshold = 0.1  # Define the neighborhood range
        currentMinValue=0.0
        for gen in range(max_generations):
            elapsed_time = time.time() - start_time
            if elapsed_time >= max_time:
                self.logger.info("Elapsed time reached, stopping.")
                break
            elites = tools.selBest(population, elite_size)
            population = algorithms.eaSimple(population, self.toolbox, cxpb=1.0, mutpb=0.5, ngen=1, verbose=False)[0]
            population.extend(elites)
            #logger.info(f"population: {len(population)}")
            population = tools.selBest(population, len(population) - elite_size)
            fits = [ind.fitness.values[0] for ind in population]
            self.logger.info(f"Generation {gen}: Min {min(fits)}, Max {max(fits)}, Avg {sum(fits) / len(fits)}")
            currentMinValue=min(fits)
            '''
            if float(currentMinValue) == float(20):
                worseCount += 1
            else:
                worseCount=0
            '''
            if gen==0:
                prevMinValue=currentMinValue
            else:
                self.logger.info(f"currentMinValue: {currentMinValue}")
                '''
                if abs(currentMinValue - prevMinValue) <= neighborhood_threshold:
                #if currentMinValue==prevMinValue:
                    count+=1
                else:
                    count=0
            if count == 2: #so 3 guys having the same value
                self.logger.info(f"same 3 guys worked babe")
                break
            if worseCount==3:
                self.logger.info(f"worse value worked babe")
                break
            '''
            prevMinValue=currentMinValue
        best_individual = tools.selBest(population, 1)[0]
        weights = best_individual[5:]
        edge_weights = [float(round(float(weight / 100), 3)) for weight in weights]
        best_individual[5:]=edge_weights
        self.logger.info(f"Best parameters: {best_individual}")
        '''
        bestAverage=[self.run_simulation(best_individual[0], best_individual[1], best_individual[2], best_individual[3], best_individual[4], edge_weights) for _ in range(2)]
        best_mean_score, best_margin_of_error = self.compute_confidence_interval(bestAverage, confidence=0.95)
        logger.info(f"Highest average R2 score: {best_mean_score}")
        logger.info(f"Confidence interval margin: {best_margin_of_error}")
        logger.info(f"Confidence interval: {(best_mean_score - best_margin_of_error, best_mean_score + best_margin_of_error)}")
        return best_mean_score
        '''
        return currentMinValue
    
