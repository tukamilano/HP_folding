from HP_model import generate_initial_population, score
from local_opt import local_opt, simulated_annealing

sequence1 = 'HPHPPHHPHPPHPHHPPHPH'
#sequence2 = 'PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH'
#sequence3 = 'HPHHPPHHHHPHHHPPHHPPHPHHPHPHHPPHHPPPHPPPPPPPPHHH'
#sequence4 = 'PHPHHPPHPHHHPPHHPHHPPPHHHHHPPHPHHPHPHPPPPHPPHPHP'
#sequence5 = 'PHHPPPPPPHHPPPHHHPHPPHPHHPPHPPHPPHHPPHHHHHHHPPHH'

'''
initial_population = generate_initial_population(sequence1, population_size=10)
print(initial_population)
'''
candidate_pos = [1, 0, 1, 0, 0, 1, 1, 0, -1, 0, 0, 0, 1, 0, -1, 1, 0, 0, 0]
#a, b = local_opt(candidate_pos, sequence1)
a, b = simulated_annealing(candidate_pos, sequence1)
print(a)
print(b)