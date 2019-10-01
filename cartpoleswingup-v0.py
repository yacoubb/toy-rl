import gym
import gym_cartpole_swingup
import numpy as np
import reinforcement


def calc_fitness(network, env_name, seed, stochastic_repeats, render=False):
    # initalise the environment and fitness
    env = gym.make(env_name)
    fitness = 0
    for i in range(stochastic_repeats):
        env.seed(seed + i * 298310)
        observation = env.reset()
        for t in range(1000):
            if render:
                env.render()
            # toy-nn doesnt handle having squeezed shapes like (3,) but it handles (3,1) fine
            observation = np.expand_dims(observation, -1)
            prediction = network.predict(observation)
            # action = np.argmax(prediction)
            action = prediction[0]
            observation, reward, done, info = env.step(action)
            fitness += reward / stochastic_repeats
            if done:
                break
    env.close()
    return int(fitness)


if __name__ == "__main__":
    args = {
        "environment_name": "CartPoleSwingUp-v0",
        "network_shape": [5, 3, 1],
        "network_activations": ["Relu", "Linear"],
        "population_size": 100,
        "generation_count": 50,
        "fitness_func": calc_fitness,
        "pareto_param": 0.4,
        "mutation_rate": 0.05,
        "stochastic_repeats": 20,
        "render_best_rollout": True,
    }
    print(args)
    np.random.seed(4)
    final_pop, final_fitnesses = reinforcement.neuro_evolution_rl(**args)
