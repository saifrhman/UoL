import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#hyperparams
NUM_CITIES = 100
NUM_ANTS = 100  #population

ALPHA = 1  #pheromone importance
BETA = 2  #heuristic importance (distance in tsp)
EVAPORATION_RATE = 0.25  #evaporation rate of pheromone
Q = 100  #pheromone deposit factor

NUM_ITERATIONS = 500  #number of iterations

#setup: 500 random cities
np.random.seed(42)
cities = np.random.rand(NUM_CITIES, 2) * 500

#calculate distance between two nodes
def distance(a, b):
    return np.linalg.norm(a - b)

#distance matrix
dist_matrix = np.array([[distance(a, b) for j, b in enumerate(cities)] for i, a in enumerate(cities)])

#pheronome matrix
#all pheronome levels are initialised to 1
pheromone_matrix = np.ones((NUM_CITIES, NUM_CITIES))

#select the next node based on pheronome and distance
#this is where the params ALPHA and BETA are used
#equation in week 10's lecture
def select_next_city(ant_path, pheromone, dist):
    current_city = ant_path[-1]
    unvisited = list(set(range(NUM_CITIES)) - set(ant_path))
    
    if not unvisited:
        return ant_path[0]  #if all cities visited, return to start (that's what you do in tsp)

    #each city has a probability of being visited by an ant given the ALPHA and BETA values
    probabilities = []
    for city in unvisited:
        prob = (pheromone[current_city][city] ** ALPHA) * ((1 / dist[current_city][city]) ** BETA)
        probabilities.append(prob)
    
    probabilities = np.array(probabilities) / sum(probabilities)
    return np.random.choice(unvisited, p=probabilities)

#solution - each ant presents its solution
def construct_solution():
    all_paths = []
    path_lengths = []
    for _ in range(NUM_ANTS):
        path = [np.random.randint(0, NUM_CITIES)]  #start at a random node
        while len(path) < NUM_CITIES:
            path.append(select_next_city(path, pheromone_matrix, dist_matrix))
        all_paths.append(path)
        path_lengths.append(sum(dist_matrix[path[i]][path[i+1]] for i in range(NUM_CITIES-1)) + dist_matrix[path[-1]][path[0]])
    return all_paths, path_lengths

#pheromone update
def update_pheromone(paths, lengths):
    global pheromone_matrix
    pheromone_matrix *= (1 - EVAPORATION_RATE)  #evaporation
    for path, length in zip(paths, lengths):
        for i in range(NUM_CITIES - 1):
            pheromone_matrix[path[i]][path[i+1]] += Q / length
        pheromone_matrix[path[-1]][path[0]] += Q / length

#visualisation path between first and last cities is highlighted in red
def plot_tour(best_path, iteration):
    plt.clf()
    for i in range(NUM_CITIES - 1):
        plt.plot([cities[best_path[i]][0], cities[best_path[i+1]][0]],
                 [cities[best_path[i]][1], cities[best_path[i+1]][1]], 'bo-')
    plt.plot([cities[best_path[-1]][0], cities[best_path[0]][0]],
             [cities[best_path[-1]][1], cities[best_path[0]][1]], 'ro-')
    plt.scatter(cities[:, 0], cities[:, 1], c='red', marker='o')
    plt.title(f"Best Tour Found - Iteration {iteration}")
    plt.pause(0.1)

#aco
plt.ion()
best_path = None
best_length = float('inf')

for iteration in range(NUM_ITERATIONS):
    paths, lengths = construct_solution()
    update_pheromone(paths, lengths)
    
    #track best sol
    min_length = min(lengths)
    if min_length < best_length:
        best_length = min_length
        best_path = paths[np.argmin(lengths)]
        plot_tour(best_path, iteration)
    
    print(f"Iteration {iteration}: Best Length = {best_length:.2f}")

plt.ioff()
plt.show()
