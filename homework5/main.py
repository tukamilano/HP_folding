from HP_model import *
from parameter import *

sequence1 = 'HPHPPHHPHPPHPHHPPHPH'
#sequence2 = 'PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH'
#sequence3 = 'HPHHPPHHHHPHHHPPHHPPHPHHPHPHHPPHHPPPHPPPPPPPPHHH'
#sequence4 = 'PHPHHPPHPHHHPPHHPHHPPPHHHHHPPHPHHPHPHPPPPHPPHPHP'
#sequence5 = 'PHHPPPPPPHHPPPHHHPHPPHPHHPPHPPHPPHHPPHHHHHHHPPHH'

generation_num = 3
population_size = 10
elite_ratio = 0.1
evolution_type = DARWIN
initial_population = generate_initial_population(sequence1, population_size=population_size, evolution_type=evolution_type)
tournament_k = 10
sequence = sequence1

population = initial_population
for generation in range(generation_num):
    next_population = []

    assert len(population) == population_size

    # エリート保存
    elite_population = population[:int(len(population)*elite_ratio)]
    next_population += elite_population

    while len(next_population) < population_size:
        #トーナメント選択
        assert all(score >= 0 for _, score in population)
        selected = tournament_selection(population, k=tournament_k)
        # crossover
        child1, child2 = crossover(selected[0], selected[1])
        # local search
        if evolution_type == DARWIN:
            if not detect_lethal(child1):
                child1_score = score(child1, sequence)
                next_population.append((child1, child1_score))
            if not detect_lethal(child2):
                child2_score = score(child2, sequence)
                next_population.append((child2, child2_score))
        elif evolution_type == BALDWIN:
            child1_repr, child1_score = local_opt(child1, sequence)
            if not detect_lethal(child1_repr):
                next_population.append((child1, child1_score))
            child2_repr, child2_score = local_opt(child2, sequence)
            if not detect_lethal(child2_repr):
                next_population.append((child2, child2_score))
        elif evolution_type == LAMARCK:
            child1, child1_score = local_opt(child1, sequence)
            if not detect_lethal(child1):
                next_population.append((child1, child1_score))        
            child2, child2_score = local_opt(child2, sequence)
            if not detect_lethal(child2):
                next_population.append((child2, child2_score))        
    population = next_population
    
print(population[0])