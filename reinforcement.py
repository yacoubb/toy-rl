import gym
import numpy as np
from toy_nn.toy_nn.nn import NeuralNetwork
from multiprocessing import Pool, cpu_count


def neuro_evolution_rl(
    environment_name,
    network_shape,
    network_activations,
    population_size,
    generation_count,
    fitness_func,
    pareto_param=1,
    mutation_rate=0.1,
    stochastic_repeats=20,
    render_best_rollout=False,
):
    # all of the above are hyperparameters that will need tuning
    # fitness func: a function that takes in {network, env_name, seed, stochastic_repeats} and produces a fitness for the network
    # pareto_param: the parameter used to control the distribution from which successful networks are selected and bred
    # mutation rate: the distribution variance by which network weights and biases are randomly adjusted during mutation
    # stochastic repeats: the number of repeat simulations to run each network on, to ensure networks don't just get lucky

    print(f"starting rl evolution with {cpu_count()} cpu cores")

    population = []
    # initialise population_size different neural networks
    # remember the weights are initalised randomly
    # networks are initialised using np.random, seed np.random for reproduceability
    for i in range(population_size):
        population.append(NeuralNetwork(network_shape, network_activations))

    for e in range(generation_count):
        # initialise every population member's fitness to 0
        fitnesses = [0 for i in range(population_size)]
        with Pool(cpu_count()) as p:
            fitnesses = p.starmap(
                fitness_func, [(x, environment_name, e * population_size + population.index(x), stochastic_repeats) for x in population]
            )

        # order the networks in a list based off of their fitnesses
        # reverse since we want highest fitness first
        ordered_networks = sorted(list(zip(fitnesses, population)), key=lambda x: x[0], reverse=True)

        # split the ordered list up into fitnesses and population
        ord_fitness, ord_pop = list(zip(*ordered_networks))
        print(f"max fitness: {ord_fitness[0]} avg fitness: {sum(ord_fitness) / population_size}")

        if render_best_rollout:
            fitness_func(ord_pop[0], environment_name, np.random.randint(0, 19134234), 1, True)

        # now select networks to survive from the ordered list using the pareto distribution
        # https://en.wikipedia.org/wiki/Pareto_distribution
        # we first create a list of indices from the pareto distribution
        next_generation = []
        indices = list(np.random.pareto(pareto_param, population_size))
        indices = list(map(lambda x: int(x) % population_size, indices))
        # select the networks chosen by the pareto distribution
        for i in range(population_size):
            # create a copy of the network
            mutation = ord_pop[indices[i]].copy()
            # and mutate
            for l in range(len(mutation.weights)):
                delta_w = np.random.normal(0, mutation_rate, mutation.weights[l].shape)
                delta_b = np.random.normal(0, mutation_rate, mutation.biases[l].shape)
                mutation.weights[l] += delta_w
                mutation.biases[l] += delta_b
            next_generation.append(mutation)
        population = next_generation
