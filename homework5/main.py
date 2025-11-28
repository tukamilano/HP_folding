from HP_model import *
from parameter import *
import random
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

random.seed(42)

sequence1 = 'HPHPPHHPHPPHPHHPPHPH'
sequence2 = 'PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH'
sequence3 = 'HPHHPPHHHHPHHHPPHHPPHPHHPHPHHPPHHPPPHPPPPPPPPHHH'
sequence4 = 'PHPHHPPHPHHHPPHHPHHPPPHHHHHPPHPHHPHPHPPPPHPPHPHP'
sequence5 = 'PHHPPPPPPHHPPPHHHPHPPHPHHPPHPPHPPHHPPHHHHHHHPPHH'

generation_num = 100
population_size = 100
elite_ratio = 0.1
tournament_k = 10
sequence = sequence2


def run_ga(evolution_type):
    population = generate_initial_population(
        sequence,
        population_size=population_size,
        evolution_type=evolution_type,
    )
    best_scores = []
    local_cache = {}

    def cached_local_opt(candidate):
        key = tuple(candidate)
        if key in local_cache:
            cached_repr, cached_score = local_cache[key]
            return list(cached_repr), cached_score
        optimized_candidate, optimized_score = local_opt(candidate, sequence)
        local_cache[key] = (tuple(optimized_candidate), optimized_score)
        return optimized_candidate, optimized_score

    for _ in tqdm(range(generation_num), desc="Generations", leave=False):
        population = sorted(population, key=lambda x: x[1], reverse=True)
        best_scores.append(population[0][1])
        next_population = []
        seen_candidates = set()

        assert len(population) == population_size

        elite_population = population[:int(len(population) * elite_ratio)]
        next_population += elite_population
        seen_candidates.update(tuple(candidate) for candidate, _ in elite_population)

        while len(next_population) < population_size:
            assert all(individual_score >= 0 for _, individual_score in population)
            selected = tournament_selection(population, k=tournament_k)
            child1, child2 = crossover(selected[0][0], selected[1][0])

            if evolution_type == DARWIN:
                child1_key = tuple(child1)
                if child1_key not in seen_candidates and not detect_lethal(child1):
                    child1_score = score(child1, sequence)
                    next_population.append((child1, child1_score))
                    seen_candidates.add(child1_key)
                child2_key = tuple(child2)
                if child2_key not in seen_candidates and not detect_lethal(child2):
                    child2_score = score(child2, sequence)
                    next_population.append((child2, child2_score))
                    seen_candidates.add(child2_key)
            elif evolution_type == BALDWIN:
                child1_repr, child1_score = cached_local_opt(child1)
                child1_key = tuple(child1)
                if child1_key not in seen_candidates and not detect_lethal(child1_repr):
                    next_population.append((child1, child1_score))
                    seen_candidates.add(child1_key)
                child2_repr, child2_score = cached_local_opt(child2)
                child2_key = tuple(child2)
                if child2_key not in seen_candidates and not detect_lethal(child2_repr):
                    next_population.append((child2, child2_score))
                    seen_candidates.add(child2_key)
            elif evolution_type == LAMARCK:
                child1_opt, child1_score = cached_local_opt(child1)
                child1_key = tuple(child1_opt)
                if child1_key not in seen_candidates and not detect_lethal(child1_opt):
                    next_population.append((child1_opt, child1_score))
                    seen_candidates.add(child1_key)
                child2_opt, child2_score = cached_local_opt(child2)
                child2_key = tuple(child2_opt)
                if child2_key not in seen_candidates and not detect_lethal(child2_opt):
                    next_population.append((child2_opt, child2_score))
                    seen_candidates.add(child2_key)

        next_population = sorted(next_population, key=lambda x: x[1], reverse=True)
        population = next_population[:population_size]

    population = sorted(population, key=lambda x: x[1], reverse=True)
    return population[0], best_scores


def main():
    evolution_configs = [
        (DARWIN, "Darwin"),
        (BALDWIN, "Baldwin"),
        (LAMARCK, "Lamarck"),
    ]

    best_tracks = {}
    best_individuals = {}

    for evo_type, label in evolution_configs:
        tqdm.write(f"Running {label} evolution...")
        start = time.time()
        best_individual, best_scores = run_ga(evo_type)
        duration = time.time() - start
        best_tracks[label] = best_scores
        best_individuals[label] = best_individual
        print(f"{label}: best score={best_individual[1]} candidate={best_individual[0]} runtime={duration:.2f}s")

    plt.figure(figsize=(10, 6))
    generations = list(range(generation_num))

    for label, scores in best_tracks.items():
        plt.plot(generations, scores, label=label)

    plt.title("Best score per generation")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('HP_score.png')
    plt.show()


if __name__ == "__main__":
    main()
