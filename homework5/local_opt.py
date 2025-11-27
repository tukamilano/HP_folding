from utils import *
import random
import math

def find_best_neighbour(candidate, sequence):
    now = score(candidate, sequence)
    best = now
    best_neighbour = None

    for i in range(len(candidate)):
        for j in [-1, 0, 1]:
            if candidate[i] == j:
                continue
            neighbour = candidate.copy()
            neighbour[i] = j
            if detect_lethal(neighbour):
                continue
            a = score(neighbour, sequence)
            if best < a:
                best = a
                best_neighbour = neighbour
    return best_neighbour


def local_opt(candidate, sequence): #山登り法
    while True:
        next_candidate = find_best_neighbour(candidate, sequence)
        if next_candidate is None:
            break
        candidate = next_candidate
    if detect_lethal(candidate):
        candidate_score = -math.inf
    else:
        candidate_score = score(candidate, sequence)

    return candidate, candidate_score

def random_neighbour(candidate):
    neighbour = candidate.copy()

    i = random.randrange(len(candidate))

    choices = [-1, 0, 1]
    choices.remove(candidate[i])
    j = random.choice(choices)

    neighbour[i] = j
    return neighbour

def simulated_annealing(candidate, sequence, step=100000, T0=1.0, Tmin=0.1, alpha=0.99999):
    v = score(candidate, sequence)
    best_candid, best_v = candidate, v
    T = T0
    for step_num in range(step):
        new_neighbour = random_neighbour(candidate)
        if detect_lethal(new_neighbour):
            continue
        new_v = score(new_neighbour, sequence)
        delta = v - new_v

        if delta < 0 or random.random() < math.exp(-delta / T):
            candidate, v = new_neighbour, new_v
            if best_v < v:
                best_candid, best_v = candidate, v
                print(f"Step {step_num}: Improved! best_v={v:.3f}")
            
        T = max(T * alpha, Tmin)

        if step_num % 1000 == 0:
            print(f"Step {step_num}: T={T:.3f}, v={v:.3f}")
    
    return best_candid, best_v