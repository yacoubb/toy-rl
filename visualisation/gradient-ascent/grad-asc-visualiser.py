import numpy as np
import cv2


def gradient_ascend(fitness_heatmap_img_path, population_size, generation_count, pareto_param=1, mutation_rate=0.1):
    population_history = []
    fitness_history = []
    heatmap = cv2.imread(fitness_heatmap_img_path)
    rgb_heatmap = heatmap.copy()
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HLS)
    scale = 0.2
    population = []
    for i in range(population_size):
        population.append(np.random.normal(0, 1, 2))

    for e in range(generation_count):
        fitnesses = [0 for i in range(population_size)]
        for i in range(len(population)):
            x = int(population[i][0] * scale * heatmap.shape[1] + heatmap.shape[1] / 2)
            y = int(population[i][1] * scale * heatmap.shape[0] + heatmap.shape[0] / 2)
            if x < 0 or x >= heatmap.shape[1] or y < 0 or y >= heatmap.shape[1]:
                continue
            fitnesses[i] = -heatmap[x][y][0]

        ordered_networks = sorted(list(zip(fitnesses, population)), key=lambda x: x[0], reverse=True)

        ord_fitness, ord_pop = list(zip(*ordered_networks))
        print(f"{e} [max fitness: {ord_fitness[0]} avg fitness: {sum(ord_fitness) / population_size}]")

        pop_representation = rgb_heatmap.copy()
        for member in population:
            x = int(member[0] * scale * pop_representation.shape[1] + pop_representation.shape[1] / 2)
            y = int(member[1] * scale * pop_representation.shape[0] + pop_representation.shape[0] / 2)
            if x < 0 or x >= pop_representation.shape[1] or y < 0 or y >= pop_representation.shape[1]:
                continue
            cv2.circle(pop_representation, (y, x), 5, (51, 51, 51), thickness=-1)
        cv2.imshow("heatmap", pop_representation)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

        next_generation = []
        # indices = list(np.random.pareto(pareto_param, population_size))
        # indices = list(map(lambda x: int(x) % population_size, indices))
        indices = list(range(int(population_size / 2))) * 2
        # print(indices)
        for i in range(population_size):
            mutation = np.copy(ord_pop[indices[i]])
            delta = np.random.normal(0, mutation_rate, 2)
            mutation += delta
            next_generation.append(mutation)
        population = next_generation
        population_history.append(ord_pop)
        fitness_history.append(ord_fitness)
    cv2.destroyAllWindows()
    return population_history, fitness_history


if __name__ == "__main__":
    np.random.seed(4)
    gradient_ascend("./fitness-heatmap-2.png", 100, 1000, mutation_rate=0.05)
