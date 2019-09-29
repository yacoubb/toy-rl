import gym
import numpy as np
from toy_nn.toy_nn.nn import NeuralNetwork


def calc_fitness(network, env_name):
    # initalise the environment and fitness
    env = gym.make(env_name)
    fitness = 0
    observation = env.reset()
    for t in range(1000):
        # toy-nn doesnt handle having squeezed shapes like (3,) but it handles (3,1) fine
        observation = np.expand_dims(observation, -1)
        prediction = network.predict(observation)
        action = np.argmax(prediction)
        observation, reward, done, info = env.step(action)
        fitness += reward
        if done:
            break
    env.close()
    return fitness


def cartpole_rl(
    population_size,
    generation_count,
    pareto_param=1,
    mutation_rate=0.09,
    stochastic_repeats=20,
):
    # population size: the number of neural networks competing in each generation
    # generation_count: the number of generations to simulate and mutate for
    # pareto_param: the parameter used to control the distribution from which successful networks are selected and bred
    # mutation rate: the distribution variance by which network weights and biases are randomly adjusted during mutation
    # stochastic repeats: the number of repeat simulations to run each network on, to ensure networks don't just get lucky

    network_shape = [4, 2]
    activations = ["Sigmoid"]
    # all of the above are hyperparameters that will need tuning

    population = []
    # initialise population_size different neural networks
    # remember the weights are initalised randomly
    for i in range(population_size):
        population.append(NeuralNetwork(network_shape, activations))

    for e in range(generation_count):
        print(f"starting generation {e}")
        # initialise every population member's fitness to 0
        fitnesses = [0 for i in range(population_size)]

        # for every member of the population, see how it performs at the cartpole task
        for i in range(population_size):
            for j in range(stochastic_repeats):
                fitnesses[i] += (
                    calc_fitness(population[i], "CartPole-v1") / stochastic_repeats
                )
            fitnesses[i] = int(fitnesses[i])

        # order the networks in a list based off of their fitnesses
        # reverse since we want highest fitness first
        ordered_networks = sorted(
            list(zip(fitnesses, population)), key=lambda x: x[0], reverse=True
        )

        # split the ordered list up into fitnesses and population
        ord_fitness, ord_pop = list(zip(*ordered_networks))
        print(
            f"max fitness: {ord_fitness[0]} avg fitness: {sum(ord_fitness) / population_size}"
        )
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


if __name__ == "__main__":
    np.random.seed(4)
    cartpole_rl(100, 50)

