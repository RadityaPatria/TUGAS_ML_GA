import pandas as pd
import math
import random

# --- Load Dataset ---
def load_dataset(path):
    df = pd.read_csv(path, header=None)
    df.columns = ["X", "Y"]
    df["City"] = [f"C{i}" for i in range(1, len(df) + 1)]
    return df


# --- Jarak antar kota ---
def distance(city1, city2):
    return math.dist((city1["X"], city1["Y"]), (city2["X"], city2["Y"]))


# --- Total jarak satu rute ---
def route_distance(route):
    dist = 0
    for i in range(len(route) - 1):
        dist += distance(route[i], route[i + 1])
    dist += distance(route[-1], route[0])    # kembali ke kota awal
    return dist


# --- Fitness (semakin kecil jarak, semakin besar fitness) ---
def fitness(route):
    d = route_distance(route)
    return 1 / d


# --- Bangkitkan rute random ---
def create_route(city_list):
    route = city_list.copy()
    random.shuffle(route)
    return route


# --- Seleksi roulette ---
def roulette_selection(pop):
    max_val = sum(fitness(r) for r in pop)
    pick = random.uniform(0, max_val)
    current = 0

    for route in pop:
        current += fitness(route)
        if current >= pick:
            return route


# --- Crossover: Ordered Crossover (OX) ---
def ordered_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    segment = parent1[start:end]
    child = segment.copy()

    for city in parent2:
        if city not in child:
            child.append(city)

    return child


# --- Mutasi: swap dua kota ---
def mutate(route, rate=0.02):
    for i in range(len(route)):
        if random.random() < rate:
            j = random.randint(0, len(route) - 1)
            route[i], route[j] = route[j], route[i]
    return route


# --- Genetic Algorithm Main ---
def genetic_algorithm(df, population_size=50, generations=300):
    cities = df.to_dict("records")
    population = [create_route(cities) for _ in range(population_size)]

    for gen in range(generations):
        new_population = []

        for _ in range(population_size):
            p1 = roulette_selection(population)
            p2 = roulette_selection(population)
            child = ordered_crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    best = min(population, key=lambda r: route_distance(r))
    return best, route_distance(best)
