import numpy as np
import mod
from mod import *
from deap import base, creator, tools, algorithms
import random, time, cProfile
from loggerClean import setup_logger
from geneticAlgo import geneticAlgo


def generateRingWithChords(nodes, chords, chordRange):  # done nodes chords
    # print("generate ring nodes: ", nodes) #self.numNodes)
    edges = [(i, (i + 1)) for i in
             range(nodes)]  # done nodes range(self.numNodes)] #+1 because I also include the max value.
    edges.append((nodes, 0))  # done self.numNodes,0))
    addedChords = 0
    while addedChords < chords:  # done self.numChords:
        u, v = random.sample(range(nodes), 2)  # done self.numNodes), 2)
        if abs(u - v) in chordRange and (u, v) not in edges and (v, u) not in edges:
            edges.append((u, v))
            addedChords += 1
    return edges


def generateRingWithFixedDistancedChords(nodes, chordDistance, chordStep):
    edges = [(i, (i + 1)) for i in range(nodes)]
    edges.append((nodes, 0))
    edges.extend(
        [(i, ((i + chordDistance) % (nodes + 1))) if (i + chordDistance) > nodes else (i, (i + chordDistance)) for i in
         range(0, nodes + 1, chordStep)])
    return edges


def generateLabels(nodes):  # done nodes
    labels = []
    for i in range(1, nodes + 1):  # self.numNodes+1):
        labels.append(str(i))  # "mol"+str(i)
    return labels


def generateMolecules(nodes):  # done nodes
    graphs = []
    labels = generateLabels(nodes)  # done nodes
    # "molecules labels: ", labels)
    for i in range(nodes):  # done self.numNodes):
        graphs.append(Graph.fromDFS("[" + labels[i] + "]", name=labels[i]))
    return graphs


def defineModRules(nodes, edges):  # done nodes
    rules = []
    # print("labels nodes: ", nodes) #self.numNodes)
    labels = generateLabels(nodes)  # done nodes
    # print("labels: ", labels)
    rules.append(Rule.fromDFS("[" + str(0) + "]>>[" + labels[0] + "]"))  # "[Food"+str(1)+"]>>["+labels[0]+"]"
    for i in edges:
        # print("edge ", i)
        # rules.append(Rule.fromDFS("["+labels[i[0]]+"]>>["+labels[i[1]]+"]"))
        rules.append(Rule.fromDFS("[" + str(i[0]) + "]>>[" + str(i[1]) + "]"))
    return rules


def generateInputGraphs(nodes):  # done nodes
    molecules = [Graph.fromDFS("[" + str(0) + "]", name=str(0))]
    molecules.extend(generateMolecules(nodes))  # done nodes
    return molecules


def buildNetwork(nodes, distance, step):
    edges = generateRingWithFixedDistancedChords(nodes, distance, step)
    rules = defineModRules(nodes, edges)
    molecules = generateInputGraphs(nodes)
    return molecules, rules


class topologyGeneticAlgo:
    def __init__(self, inputSeed, inputLogger):
        self.logger = inputLogger
        self.seed = inputSeed
        random.seed(inputSeed)
        self.logger.info(f"seed: {self.seed}")
        self.INT_BOUNDS = {
            'nodeValues': (50, 300),  # (10, 100),
            'moleculeInflow': (50, 200),
            'chordDistance': (5, 25),
            'chordStep': (5, 25)
        }
        self.logger.info(f"{self.INT_BOUNDS}")
        for i, (key, (low, high)) in enumerate(self.INT_BOUNDS.items()):
            self.logger.info(f"int bounds: {i} {key} {low} {high}")
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.initialize_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.custom_mutate, indpb=0.5)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.objective)

    def callGeneticAlgorithm(self, dg, nodes, numEdges, tau, moleculeAmount):  # done nodes
        genetic = geneticAlgo(dg, self.seed, nodes + 1, numEdges, self.logger, tau,
                              moleculeAmount)  # due to feed molecule plus one.
        error = genetic.main()
        return error

    def objective(self, params):
        self.logger.info(f"topology psrsms objective: {params}")
        # print(f"topology psrsms objective: {params}")
        nodes, moleculeInflow, distance, step = params
        network = buildNetwork(nodes, distance, step)
        score = self.callGeneticAlgorithm(network, nodes, len(network[1]),24, moleculeInflow)
        return score,  # , due to the deap library.

    def initialize_individual(self):
        random_values = []
        for i, (key, (low, high)) in enumerate(self.INT_BOUNDS.items()):
            step = 10 if i < 2 else 2
            value = random.randrange(low, high + 1, step)  # This generates an int value
            value = min(random_values[0] // 2, value) if i > 1 else value
            random_values.append(value)
        self.logger.info(f"initial values in gen algo topology: {random_values}")
        # print(f"initial values in gen algo topology: {random_values}")
        return random_values

    def mutation(self, individual, indpb):
        original = individual[:]
        self.logger.info(f"topology mutation giren individual: {individual}")
        # print(f"topology mutation giren individual: {individual}")
        size = len(individual)
        while True:
            for i in range(size):
                if random.random() < indpb:
                    step = 2 if i > 1 else 10
                    key = list(self.INT_BOUNDS.keys())[i]
                    low, high = self.INT_BOUNDS[key]
                    randomChoice = random.choice([-step, step])
                    self.logger.info(
                        f"topology mutation index ind i random choice and step low high: {i} {individual[i]} {randomChoice} {step} {low} {high}")
                    # print(f"topology mutation index ind i random choice and step low high: {i} {individual[i]} {randomChoice} {step} {low} {high}")
                    # fixedValue=max(low, min(high, individual[i] + random.choice([-step, step])))
                    fixedValue = max(low, min(high, individual[i] + randomChoice))
                    individual[i] = fixedValue if i < 2 else min(individual[0] // 2, fixedValue)  # //4
                    self.logger.info(
                        f"Topology mutation individual index low high step value new value: {individual} {i} {low} {high} {step} {fixedValue} {individual[i]}")
                    # print(f"Topology mutation individual index low high step value new value: {individual} {i} {low} {high} {step} {fixedValue} {individual[i]}")
            if individual != original:
                break
        return individual

    def custom_mutate(self, individual, indpb=0.5):
        individual = self.mutation(individual, indpb)
        return individual,

    def main(self, population_size=4, elite_size=2, max_generations=10, max_time=10000):
        self.logger.info(
            f"topology genetic algo features population size {population_size} {elite_size} {max_generations} {max_time}")
        populations = self.toolbox.population(n=population_size)
        start_time = time.time()
        currentMinValue = None
        for gen in range(max_generations):
            elapsed_time = time.time() - start_time
            if elapsed_time >= max_time:
                self.logger.info("Topology Elapsed time reached, stopping.")
                break
            elites = tools.selBest(populations, elite_size)
            populations = algorithms.eaSimple(populations, self.toolbox, cxpb=1.0, mutpb=0.5, ngen=1, verbose=False)[0]
            populations.extend(elites)
            populations = tools.selBest(populations, len(populations) - elite_size)
            fits = [ind.fitness.values[0] for ind in populations]
            self.logger.info(
                f"Topology Generation {gen}: Min {min(fits)}, Max {max(fits)}, Avg {sum(fits) / len(fits)}")
            # print(f"Topology Generation {gen}: Min {min(fits)}, Max {max(fits)}, Avg {sum(fits) / len(fits)}")
            currentMinValue = min(fits)
            self.logger.info(f"Topology currentMinValue: {currentMinValue}")
            # print(f"Topology currentMinValue: {currentMinValue}")
            local_individual = tools.selBest(populations, 1)[0]
            self.logger.info(f"Topology local best parameters: {local_individual}")
            # print(f"Topology local best parameters: {local_individual}")
        best_individual = tools.selBest(populations, 1)[0]
        self.logger.info(f"Topology Best parameters: {best_individual}")
        # print(f"Topology Best parameters: {best_individual}")
        return currentMinValue


profiler = cProfile.Profile()
profiler.enable()
logger = setup_logger("tau24longtest50secondsdenemerep2")
logger.info(f"long memory topology genetic test regular chords tau 24")
iterative = topologyGeneticAlgo(1, logger)
iterative.main()
profiler.disable()
profiler.print_stats(sort='time')

