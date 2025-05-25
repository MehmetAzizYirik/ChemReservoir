import numpy as np
from deap import base, creator, tools, algorithms
import random, time
from scipy.stats import sem, t
from runLongMemoryTask import longMemoryTask
from runShortMemoryTask import shortMemoryTask

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
class geneticAlgo:
    def __init__(self, dg, seed, num_nodes, edges, logger, moleculeInflux, memoryTask, tau=0, runTime=0, repetition=0):
        """
        Setting instance variables of the genetic algorithm for the network parameter optimization.

        Args:
            dg (tuple): (molecules, rules) used for the construction of derivation graph in MOD.
            seed (int): seed value for the random and np.random libraries
            num_nodes (int): number of nodes.
            edges (int): number of edges.
            logger (logging.Logger): logger file for intermediate steps.
            moleculeInflux (int): the fixed molecule amount to be used in inflow signal generation.
            memoryTask (string): choosing the memory tasks for regression step.
            tau (int): past input time, tau, should be defined if long memory task is chosen.
            runTime (int): simulation run time
            repetition (int): repetition value in input signal for the input hold time.
        """

        self.logger=logger
        self.tau=tau
        self.memoryTask=memoryTask
        self.runTime=runTime
        self.repetition=repetition
        self.moleculeInflux=moleculeInflux
        self.DG = dg
        self.SEED = seed
        self.moleculeAmount = num_nodes
        self.edges = edges
        self.INT_BOUNDS = {
            'seed': (1, 1),
            'scalingRate': (1,10),
            'moleculeAmount': (self.moleculeAmount, self.moleculeAmount),
            'reactionScale': (1,10),
        }
        self.FLOAT_BOUNDS =  {'outFlow': (0.05, 1.0)}
        self.EDGE_WEIGHT_BOUNDS = (10, 100)
        self.NUM_EDGE_WEIGHTS = lambda: int(self.edges)
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.initialize_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.custom_mutate, indpb=0.5)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.objective)
    
    def run_simulation(self, seed, outFlow, scalingRate, moleculeAmount, reactionScale, edge_weights):
        """
        Calling stochastic simulation for the topology and input parameters.

        Args:
            seed (int): seed value for the random and np.random libraries
            outFlow (float): the amount to exist system over the time.
            scalingRate (int): scaling rate for the input signal.
            moleculeAmount (int): molecule amount to be used in input signal generation.
            reactionScale (int): scaling rate for the reaction rates.
            edge_weights (list[float]): the edge weights for each reaction.
        Returns:
            float: the lowest normalized root mean square error from network optimization process.
        """

        seed = int(seed)
        outFlow = float(outFlow)
        scalingRate = int(scalingRate)
        moleculeAmount = int(moleculeAmount)
        reactionScale = int(reactionScale)
        edge_weights = [float(weight) for weight in edge_weights]
        edge_weights_str = ','.join(map(str, edge_weights))
        arguments = [seed, outFlow, scalingRate, moleculeAmount, reactionScale, edge_weights_str]
        if self.memoryTask=="short":
            topologyEvaluator = shortMemoryTask(self.SEED, self.DG, self.moleculeInflux, arguments, runTime=self.runTime, repetition=self.repetition)
        else:
            topologyEvaluator = longMemoryTask(self.SEED, self.DG, self.tau, self.moleculeInflux, arguments, runTime=self.runTime, repetition=self.repetition)

        return topologyEvaluator.main()
    
    def compute_confidence_interval(self, scores, confidence=0.95):
        """
        Computing the confidence interval for the scores obtained from multi-calls of simulation function.

        Args:
            scores (list): list of minimum root mean square errors obtained from the simulation calls.
            confidence (float): confidence value for the standard error calculation.
        Returns:
            tuple: mean and margin values for the input scores.
        """

        n = len(scores)
        mean_score = np.mean(scores)
        std_error = sem(scores)
        h = std_error * t.ppf((1 + confidence) / 2, n - 1)
        return mean_score, h

    def objective(self, params, num_runs=2, confidence=0.95):
        """
        Objective function for the genetic algorithm. Running the stochastic simulation and regression to obtain
        lowest root mean square error.

        Args:
            params (list): input parameters for the simulation.
            num_runs (int): number of runs for the simulation.
            confidence (float): confidence value for the standard error calculation.
        Returns:
            float: error value obtained by confidence interval calculation.
        """
        seed, outFlow, scalingRate, moleculeAmount, reactionScale, *edge_weights = params
        edge_weights = [float(round(float(weight / 100), 3)) for weight in edge_weights]
        scores = [self.run_simulation(seed, outFlow, scalingRate, moleculeAmount, reactionScale, edge_weights) for _ in range(num_runs)]
        mean_score, margin_of_error = self.compute_confidence_interval(scores, confidence)
        return mean_score + margin_of_error,
    
    def initialize_individual(self):
        """
        Initializing the parameters to be used in the objective function.

        Returns:
            list: input parameters for the objective function.
        """
        seed = random.randint(*self.INT_BOUNDS['seed'])
        outFlow = round(random.uniform(*self.FLOAT_BOUNDS['outFlow']), 3)
        scalingRate = random.randint(*self.INT_BOUNDS['scalingRate'])
        moleculeAmount = self.moleculeAmount
        reactionScale = random.randint(*self.INT_BOUNDS['reactionScale'])
        num_edges = self.NUM_EDGE_WEIGHTS()
        edge_weights = [random.randint(*self.EDGE_WEIGHT_BOUNDS) for _ in range(num_edges)]
        return [seed, outFlow, scalingRate, moleculeAmount, reactionScale] + edge_weights
    
    def mutate_integer(self, individual, indpb):
        """
        Mutation function for integer values.

        Args:
            individual (list): individual from the genetic algorithm population.
            indpb (float): individual probability

        Returns:
            list: mutated individual
        """
        integer_indices = [0, 2, 3, 4]
        for i in integer_indices:
            if random.random() < indpb:
                step = 1
                key = list(self.INT_BOUNDS.keys())[integer_indices.index(i)]
                low, high = self.INT_BOUNDS[key]
                individual[i] = max(low, min(high, individual[i] + random.choice([-step, step])))
        return individual
    
    def mutate_float(self, individual, indpb):
        """
        Mutation function for float values.

        Args:
            individual (list): individual from the genetic algorithm population.
            indpb (float): individual probability

        Returns:
            list: mutated individual
        """
        if random.random() < indpb:
            step = 0.2
            low, high = self.FLOAT_BOUNDS['outFlow']
            individual[1] = max(low, min(high, individual[1] + random.choice([-step, step])))
        individual[1] = round(individual[1], 3)
        return individual
    
    def mutate_edge_weights(self, individual, indpb):
        """
        Mutation function for edge_weighs (reaction rates).

        Args:
            individual (list): individual from the genetic algorithm population.
            indpb (float): individual probability

        Returns:
            list: mutated individual
        """
        low, high = self.EDGE_WEIGHT_BOUNDS
        num_edges = self.NUM_EDGE_WEIGHTS()
        for i in range(5, 5 + num_edges):
            if random.random() < indpb:
                step = 10
                individual[i] = max(low, min(high, individual[i] + random.choice([-step, step])))
        return individual
    
    def custom_mutate(self, individual, indpb=0.5):
        """
        Main custom mutation function, calling sub mutation functions for the integer, float and edge-weights values.

        Args:
            individual (list): individual from the genetic algorithm population.
            indpb (float): individual probability

        Returns:
            list: mutated individual
        """
        individual = self.mutate_integer(individual, indpb)
        individual = self.mutate_float(individual, indpb)
        individual = self.mutate_edge_weights(individual, indpb)
        return individual,
    
    def main(self, population_size=4, elite_size=2, max_generations=10, max_time=500):
        """
        Main function of the network optimization genetic algorithm.

        Args:
            population_size (int): genetic algorithm population size
            elite_size (int): genetic algorithm elite size
            max_generations (int): number of generations
            max_time (int): maximum run time of the genetic algorithm.

        Returns:
            float: the lowest normalized root mean square error from network optimization process.
        """
        population = self.toolbox.population(n=population_size)
        start_time = time.time()
        prevMinValue=0.0
        currentMinValue=0.0
        for gen in range(max_generations):
            elapsed_time = time.time() - start_time
            if elapsed_time >= max_time:
                self.logger.info("Elapsed time reached, stopping.")
                break
            elites = tools.selBest(population, elite_size)
            population = algorithms.eaSimple(population, self.toolbox, cxpb=1.0, mutpb=0.5, ngen=1, verbose=False)[0]
            population.extend(elites)
            population = tools.selBest(population, len(population) - elite_size)
            fits = [ind.fitness.values[0] for ind in population]
            self.logger.info(f"Generation {gen}: Min {min(fits)}, Max {max(fits)}, Avg {sum(fits) / len(fits)}")
            currentMinValue=min(fits)
            if gen==0:
                prevMinValue=currentMinValue
            else:
                self.logger.info(f"currentMinValue: {currentMinValue}")
            prevMinValue=currentMinValue
        best_individual = tools.selBest(population, 1)[0]
        weights = best_individual[5:]
        edge_weights = [float(round(float(weight / 100), 3)) for weight in weights]
        best_individual[5:]=edge_weights
        self.logger.info(f"Best parameters: {best_individual}")
        return currentMinValue
    
