import mod
from mod import *
from deap import base, creator, tools, algorithms
import random, time
from scripts.networkParameterOptimization import geneticAlgo

def generateRingWithFixedDistancedChords(nodes, chordLength, chordStep):
    """
    Generating cycle topologies with chords.

    Args:
        nodes (int): number of nodes
        chordLength (int): chord length
        chordStep (int): chord step

    Returns:
        list: Edge list of a reservoir topology
    """
    edges = [(i, (i + 1)) for i in range(nodes)]
    edges.append((nodes, 0))
    edges.extend(
        [(i, ((i + chordLength) % (nodes + 1))) if (i + chordLength) > nodes else (i, (i + chordLength)) for i in
         range(0, nodes + 1, chordStep)])
    return edges

def generateLabels(nodes):
    """
    Generating node labels.

    Args:
        nodes (int): number of nodes

    Returns:
        list: Node labels.
    """
    labels = []
    for i in range(1, nodes + 1):
        labels.append(str(i))
    return labels

def generateMolecules(nodes):
    """
    Generating node labels.

    Args:
        nodes (int): number of nodes

    Returns:
        list: Node labels.
    """
    graphs = []
    labels = generateLabels(nodes)
    for i in range(nodes):
        graphs.append(Graph.fromDFS("[" + labels[i] + "]", name=labels[i]))
    return graphs

def defineModRules(nodes, edges):
    """
    Defining list of MOD rules for the input edge list.
    These rules are used for the construction of MOD chemical reaction network.

    Args:
        nodes (int): number of nodes
        edges (list): edge list

    Returns:
        list: MOD rule list.
    """
    rules = []
    labels = generateLabels(nodes)
    rules.append(Rule.fromDFS("[" + str(0) + "]>>[" + labels[0] + "]"))
    for i in edges:
        rules.append(Rule.fromDFS("[" + str(i[0]) + "]>>[" + str(i[1]) + "]"))
    return rules

def generateInputGraphs(nodes):
    """
    Generating input graphs (molecules) for MOD StochSim.
    Each node represents a pseudo-molecule, graph, in the network.

    Args:
        nodes (int): number of nodes

    Returns:
        list: MOD input graph list.
    """
    molecules = [Graph.fromDFS("[" + str(0) + "]", name=str(0))]
    molecules.extend(generateMolecules(nodes))
    return molecules

def buildNetwork(nodes, length, step):
    """
    Building the MOD network for MOD-StochSim.
    Networks are represented by tuples (molecules, rules),
    list of input graphs and reaction rules.

    Args:
        nodes (int): number of nodes
        length (int): chord length
        step (int): chord step

    Returns:
        tuple: MOD input graphs and reaction rules.
    """
    edges = generateRingWithFixedDistancedChords(nodes, length, step)
    rules = defineModRules(nodes, edges)
    molecules = generateInputGraphs(nodes)
    return molecules, rules


class chemReservoir:
    def __init__(self, inputSeed, inputLogger, memoryTask=None, tau=None, runTime=None, repetition=None, networkOptMaxTime=None):
        """
        Setting instance variables of the genetic algorithm for topology selection optimization.

        Args:
            inputSeed (int): seed value for the random and np.random libraries
            inputLogger (logging.Logger): logger file for intermediate steps.
            memoryTask (string): choosing the memory tasks for regression step.
            tau (int): past input time, tau, should be defined if long memory task is chosen.
        """

        self.logger = inputLogger #logger file for storing genetic algorithm steps.
        self.seed = inputSeed
        self.memoryTaskValues=(memoryTask, tau)
        self.runTime=runTime
        self.repetition=repetition
        self.networkOptimizationMaxTime = networkOptMaxTime
        random.seed(inputSeed) #seed for random library for reproducibility.
        self.INT_BOUNDS=None
        '''
        The boundaries for genetic algorithm.
        '''
        '''
        self.INT_BOUNDS = {
            'nodeValues': (50, 300),
            'moleculeInflow': (50, 200),
            'chordDistance': (5, 25),
            'chordStep': (5, 25)
        }
        '''
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.initialize_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.custom_mutate, indpb=0.5)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.objective)

    def callGeneticAlgorithm(self, dg, nodes, numEdges, moleculeAmount, memoryTask, runTime=None, repetition=None, networkOptTime=None, tau=0):
        """
        Calling the genetic algorithm for the optimal topology selection.

        Args:
            dg (tuple): (molecules, rules) used for the construction of derivation graph in MOD.
            nodes (int): number of nodes.
            numEdges (int): number of edges.
            tau (int): time lag value.
            moleculeAmount (int): the fixed molecule amount to be used in inflow signal generation.
            memoryTask (string): choosing the memory tasks for regression step.
            runTime (int): simulation run time
            repetition (int): repetition value in input signal for the input hold time.
            networkOptTime (int): max run time for network optimization process.
            tau (int): past input time, tau, should be defined if long memory task is chosen.

        Returns:
            float: the lowest normalized root mean square error from topology selection.
        """
        genetic = geneticAlgo(dg, self.seed, nodes + 1, numEdges, self.logger,
                              moleculeAmount, memoryTask, tau=tau, runTime=runTime, repetition=repetition) #+1 due to the food molecule, indexed 0.

        error = genetic.main(max_time=networkOptTime)
        return error

    def objective(self, params):
        """
        Objective function of the genetic algorithm for topology selection.

        Args:
            params (list):

        Returns:
            float: the lowest normalized root mean square error from topology selection.
        """
        self.logger.info(f"topology parameters objective: {params}")
        nodes, moleculeInflow, distance, step = params
        network = buildNetwork(nodes, distance, step)
        score = self.callGeneticAlgorithm(network, nodes, len(network[1]), moleculeInflow, self.memoryTaskValues[0], runTime=self.runTime, repetition=self.repetition, networkOptTime=self.networkOptimizationMaxTime, tau=self.memoryTaskValues[1])
        return score,  # , due to the deap library.

    def initialize_individual(self):
        """
        Initializing the random initial individuals for genetic algorithm.

        Returns:
            list: List of random initial individuals.
        """
        random_values = []
        for i, (key, (low, high)) in enumerate(self.INT_BOUNDS.items()):
            step = 10 if i < 2 else 2
            value = random.randrange(low, high + 1, step)
            value = min(random_values[0] // 2, value) if i > 1 else value
            random_values.append(value)
        self.logger.info(f"initial values in topology- genetic algo: {random_values}")
        return random_values

    def mutation(self, individual, indpb):
        """
        Mutation function for topology selection, genetic algorithm.

        Args:
            individual (list): Individual for the genetic algorithm.
            indpb (float): Individual probability for the mutation function, random library criteria.

        Returns:
            list: Mutated individual.
        """
        original = individual[:]
        size = len(individual)
        while True:
            for i in range(size):
                if random.random() < indpb:
                    step = 2 if i > 1 else 10
                    key = list(self.INT_BOUNDS.keys())[i]
                    low, high = self.INT_BOUNDS[key]
                    randomChoice = random.choice([-step, step])
                    fixedValue = max(low, min(high, individual[i] + randomChoice))
                    individual[i] = fixedValue if i < 2 else min(individual[0] // 2, fixedValue)
            if individual != original:
                break
        return individual

    def custom_mutate(self, individual, indpb=0.5):
        """
        Mutation function for topology selection, genetic algorithm.

        Args:
            individual (list): Individual for the genetic algorithm.
            indpb (float): Individual probability for the mutation function, random library criteria.

        Returns:
            list: Mutated individual.
        """
        individual = self.mutation(individual, indpb)
        return individual,

    def run(self, population_size=4, elite_size=2, max_generations=10, max_time=10000):
        """
        Main function for the genetic algorithm of the topology selection process.

        Args:
            population_size (int): total number of individual for each generation
            elite_size (int): number of the best individuals to be stored for the next iteration
            max_generations(int): maximum number of iterations in genetic algorithm
            max_time(int): maximum run time for the genetic algorithm as termination criteria.

        Returns:
            tuple: The best individual and the lowest root mean square errors from the genetic algorithm.
        """
        populations = self.toolbox.population(n=population_size)
        start_time = time.time()
        currentMinValue = None
        for gen in range(max_generations):
            elapsed_time = time.time() - start_time
            if elapsed_time >= max_time:
                self.logger.info("Topology-Elapsed time reached, stopping.")
                break
            elites = tools.selBest(populations, elite_size)
            populations = algorithms.eaSimple(populations, self.toolbox, cxpb=1.0, mutpb=0.5, ngen=1, verbose=False)[0]
            populations.extend(elites)
            populations = tools.selBest(populations, len(populations) - elite_size)
            fits = [ind.fitness.values[0] for ind in populations]
            self.logger.info(
                f"Topology Generation {gen}: Min {min(fits)}, Max {max(fits)}, Avg {sum(fits) / len(fits)}")
            currentMinValue = min(fits)
            self.logger.info(f"Topology currentMinValue: {currentMinValue}")
            local_individual = tools.selBest(populations, 1)[0]
            self.logger.info(f"Topology local best parameters: {local_individual}")
        best_individual = tools.selBest(populations, 1)[0]
        self.logger.info(f"Topology Best parameters: {best_individual}")
        return best_individual, currentMinValue
