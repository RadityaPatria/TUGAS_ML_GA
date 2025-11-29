import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# ============================
# LOAD DATASET (x,y tanpa header)
# ============================

def load_dataset(filename):
    data = np.loadtxt(filename, delimiter=",")  # langsung baca 2 kolom
    return data  # shape: (N,2)

FILENAME = "data/small.csv"  # ubah sesuai kebutuhan
cities = load_dataset(FILENAME)
N = len(cities)

# ============================
# GA SETUP
# ============================

def euclidean(a, b):
    return np.linalg.norm(a - b)  # jarak euclid

def total_distance(order):
    dist = 0
    for i in range(N):
        a = cities[order[i]]
        b = cities[order[(i + 1) % N]]  # kembali ke awal
        dist += euclidean(a, b)
    return dist

# hindari error double-create
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("indices", random.sample, range(N), N)  # generate route
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(ind):
    return (total_distance(ind),)

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.02)

# ============================
# RUN GA
# ============================

def solve_tsp_ga():
    pop = toolbox.population(n=300)
    NGEN = 500

    result, log = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.8,
        mutpb=0.2,
        ngen=NGEN,
        verbose=True
    )

    best = tools.selBest(pop, 1)[0]
    return best, evaluate(best)[0]

# ============================
# VISUALIZATION
# ============================

def plot_route(best):
    ordered = cities[best]      # urutkan koordinatnya
    xs = ordered[:, 0]
    ys = ordered[:, 1]

    plt.figure(figsize=(10, 7))
    plt.plot(xs, ys, 'o-', linewidth=2)
    plt.plot([xs[-1], xs[0]], [ys[-1], ys[0]], 'o-')
    plt.title(f"TSP Route ({FILENAME})")
    plt.grid(True)
    plt.show()

# ============================
# MAIN
# ============================

if __name__ == "__main__":
    print(f"Running GA TSP using {FILENAME}...\n")

    best_route, best_distance = solve_tsp_ga()

    print("\n======================")
    print(" BEST RESULT FOUND")
    print("======================")
    print("Route:", best_route)
    print("Distance:", best_distance)

    plot_route(best_route)
