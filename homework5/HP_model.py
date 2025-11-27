import random
import math
from utils import *
from parameter import *
from local_opt import local_opt

def tournament_selection(population, k=3, winner_count=2):
    selected = []

    for _ in range(winner_count):
        # k個ランダムにサンプリング
        contenders = random.sample(population, k)
        # score の高い順に並べて1位を取る
        winner = max(contenders, key=lambda x: x[1])
        selected.append(winner)

    return selected

def generate_initial_population(sequence, population_size, evolution_type):
    initial_population = []
    while len(initial_population) < population_size:
        candidate = [random.choice([-1, 0, 1]) for _ in range(len(sequence)-1)]
        if evolution_type == DARWIN:
            if detect_lethal(candidate):
                continue
            candidate_score = score(candidate, sequence)
        elif evolution_type == BALDWIN:
            candidate_repr, candidate_score = local_opt(candidate, sequence)
            if detect_lethal(candidate_repr):
                continue
        elif evolution_type == LAMARCK:
            candidate, candidate_score = local_opt(candidate, sequence)
            if detect_lethal(candidate):
                continue
        else:
            raise Exception("wrong evolution type")
        initial_population.append((candidate, candidate_score))
    
    initial_population.sort(key=lambda x: x[1], reverse=True)

    return initial_population

def crossover(candidate1, candidate2):
    assert len(candidate1) == len(candidate2)
    a, b, c, d = get_abcd_discrete(len(candidate1))

    child1 = candidate1[:a] + candidate2[c:d] + candidate1[b:]
    child2 = candidate2[:c] + candidate1[a:b] + candidate2[d:]
    return child1, child2
