from __future__ import annotations

from typing import TypeVar, List
from random import choices, random, shuffle
from heapq import nlargest
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
import numpy as np
import logging


class Chromosome(ABC):
    @abstractmethod
    def get_fitness(self) -> float:
        ...

    @classmethod
    @abstractmethod
    def make_random_instance(cls) -> Chromosome:
        ...
    @abstractmethod
    def mutate(self) -> None:
        ...
    @abstractmethod
    def exec_crossover(self, other: Chromosome) -> List[Chromosome]:
        ...
    def __lt__(self, other: Chromosome) -> bool:
        return self.get_fitness() < other.get_fitness()


C = TypeVar('C', bound=Chromosome)

class GeneticAlgorithm:
    SelectionType = int
    SELECTION_TYPE_ROULETTE_WHEEL: SelectionType = 1
    SELECTION_TYPE_TOURNAMENT: SelectionType = 2

    def __init__(
            self, initial_population: List[C], 
            threshold: float,
            max_generations: int, mutation_probability: float,
            crossover_probability: float,
            selection_type: SelectionType) -> None:
        self._population: List[Chromosome] = initial_population
        self._threshold: float = threshold
        self._max_generations: int = max_generations
        self._mutation_probability: float = mutation_probability
        self._crossover_probability: float = crossover_probability
        self._selection_type: int = selection_type
        self.c_cross = 0.3
        self.c_mut = 2.0

    def _exec_roulette_wheel_selection(self) -> List[Chromosome]:
        weights: List[float] = [
            chromosome.get_fitness() for chromosome in self._population
        ]
        selected_chromosomes: List[Chromosome] = choices(
            self._population, weights=weights, k=2
        )
        return selected_chromosomes
    
    def _exec_tournament_selection(self) ->List[Chromosome]:
        participants_num: int = len(self._population) // 2
        participants: List[Chromosome] = choices(self._population, k=participants_num)
        selected_chromosomes: List[Chromosome] = nlargest(2, iterable=participants)
        return selected_chromosomes
    
    def _to_next_generation(self) -> None:
        new_population: List[Chromosome] = []

        new_population.append(deepcopy(self._get_best_chromosome_from_population()))
        while len(new_population) < len(self._population):
            parents: List[Chromosome] = self._get_parents_by_selection_type()
            next_generation_chromosomes: List[Chromosome] = \
                self._get_next_generation_chromosomes(parents=parents)
            new_population.extend(next_generation_chromosomes)
        
        if len(new_population) > len(self._population):
            del new_population[0]
        
        self._population = new_population

    def _get_next_generation_chromosomes(
            self, parents: List[Chromosome]) -> List[Chromosome]:
        
        random_val: float = random()
        next_generation_chromosomes: List[Chromosome] = parents
        if random_val < self._crossover_probability:
            next_generation_chromosomes = parents[0].exec_crossover(other=parents[1], c=self.c_cross)

        random_val = random()
        if random_val < self._mutation_probability:
            for chromosome in next_generation_chromosomes:
                chromosome.mutate(c=self.c_mut)
        return next_generation_chromosomes

    def _get_parents_by_selection_type(self) -> List[Chromosome]:
        if self._selection_type == self.SELECTION_TYPE_ROULETTE_WHEEL:
            parents: List[Chromosome] = self._exec_roulette_wheel_selection()
        elif self._selection_type == self.SELECTION_TYPE_TOURNAMENT:
            parents = self._exec_tournament_selection()
        else:
            raise ValueError(f'Invalid selection type: {self._selection_type}')
        return parents
    
    def run_algorithm(self) -> Chromosome:
        best_chromosome: Chromosome = \
            deepcopy(self._get_best_chromosome_from_population())
        for generation_idx in range(1, self._max_generations+1):
            if (generation_idx % 5 == 0):
                self.c_cross *= 0.99
                self.c_mut *= 0.97
            logging.info(
                f'generation index : {generation_idx} best : {best_chromosome}'
                  )
            if best_chromosome.get_fitness() >= self._threshold:
                return best_chromosome
            
            self._to_next_generation()

            current_generation_best_chromosome: Chromosome = \
                self._get_best_chromosome_from_population()
            current_gen_best_fitness: float = \
                current_generation_best_chromosome.get_fitness()
            # logging.info(
            #     datetime.now(),
            #     f'generation index : {generation_idx}',
            #     f'best : {current_generation_best_chromosome}'
            # )
            if best_chromosome.get_fitness() < current_gen_best_fitness:
                best_chromosome = deepcopy(current_generation_best_chromosome)
        return best_chromosome
    
    def _get_best_chromosome_from_population(self) -> Chromosome:
        best_chromosome: Chromosome = self._population[0]
        for chromosome in self._population:
            if chromosome.get_fitness() > best_chromosome.get_fitness():
                best_chromosome = chromosome
        return best_chromosome


def rotate90deg(original, point, times):
    array = deepcopy(original)
    frag = deepcopy(array)[point+1:]
    frag -= array[point]
    for _ in range(times):
        frag = frag @ np.array([[0,1],[-1,0]])
    frag += array[point]
    array[point+1:] = frag
    return array

class HP(Chromosome):
    
    def __init__(self, array, parent, hp) -> None:
        self.array: List[List[int]] = array
        self.parent: List[List[int]] = parent
        self.hp: str = hp

    def get_fitness(self, ignore=False) -> int:
        array = self.array
        hp = self.hp
        fitness = 1
        for i in range(len(array)):
            for j in range(i+1, len(array)):
                dist = abs(array[i][0] - array[j][0]) + abs(array[i][1] - array[j][1])
                if (dist == 0):
                    if not ignore: logging.info(f"trouble: {array.tolist()}")
                    fitness = -1000
                    return fitness
                elif (j == i+1 and dist != 1):
                    fitness = -1000
                    if not ignore: logging.info(f"error: {array.tolist()}")
                    return fitness
                if (j >= i+2 and hp[i] =='H' and hp[j] == 'H' and dist == 1):
                    fitness += 1
        return fitness

    
    @classmethod
    def make_random_instance(cls, hp) -> HP:
        array = [[i,0] for i in range(len(hp))]
        array = np.array(array)
        return HP(array=array,parent=array, hp=hp)
    
    def mutate(self, c: float) -> None:
        self_fitness = self.get_fitness()
        array = deepcopy(self.array)
        rand_idxes = [i for i in range(len(self.array))]
        shuffle(rand_idxes)
        for i in range(len(self.array)):
            rand_idx = rand_idxes[i]
            rand_rot = [1,2,3]
            shuffle(rand_rot)
            for _ in range(3):
                tmp = rotate90deg(array, rand_idx, rand_rot.pop())
                new_fitness = HP(array=tmp, parent=tmp, hp=self.hp).get_fitness(ignore=True)
                if (new_fitness < 0): continue
                if (new_fitness > self_fitness or random() < np.exp((new_fitness-self_fitness)/c)):
                    self.array = tmp
                    return

    def exec_crossover(self, other: HP, c: float) -> List[HP]:
        rand_pivots = [i+2 for i in range(len(self.array)-4)]
        shuffle(rand_pivots)
        ave_fitness = (self.get_fitness() + other.get_fitness()) / 2
        best1 = -1
        best2 = -1
        child1: HP = deepcopy(self)
        child2: HP = deepcopy(other)

        for x in range(len(rand_pivots)):
            pivot = rand_pivots[x]
            arr1 = deepcopy(self.array)
            child1_joint = arr1[pivot]
            other_head = deepcopy(other.array[:pivot])
            grand_head = deepcopy(other.parent[:pivot])
            other_head += child1_joint - other_head[-1] - np.array([1,0])
            grand_head += child1_joint - grand_head[-1] - np.array([1,0])
            if (random() < 0.1):
                arr1[:pivot] = grand_head
            else:
                arr1[:pivot] = other_head

            rand_rot1 = [i for i in range(4)]
            shuffle(rand_rot1)
            for _ in range(len(rand_rot1)):
                tmp = rotate90deg(arr1, pivot-1, rand_rot1.pop())
                child1_fitness = HP(array=tmp, parent=tmp, hp=self.hp).get_fitness(ignore=True)
                if (child1_fitness < 0): continue
                if (child1_fitness >= max(ave_fitness, best1) or random() < np.exp((child1_fitness-ave_fitness)/c)):
                    child1.array = tmp
                    child1.parent = self.array
                    best1 = max(child1_fitness, best1)
                    break
            
            arr2 = deepcopy(other.array)
            child2_joint = arr2[pivot]
            self_tail = deepcopy(self.array[pivot:])
            grand_tail = deepcopy(self.parent[pivot:])
            self_tail += child2_joint - self_tail[0] + np.array([1,0])
            grand_tail += child2_joint - grand_tail[0] + np.array([1,0])
            if (random() < 0.1):
                arr2[pivot:] = grand_tail
            else:
                arr2[pivot:] = self_tail
            rand_rot2 = [i for i in range(4)]
            shuffle(rand_rot2)
            for _ in range(len(rand_rot2)):
                tmp = rotate90deg(arr2, pivot-1, rand_rot2.pop())
                child2_fitness = HP(array=tmp, parent=tmp, hp=self.hp).get_fitness(ignore=True)
                if (child2_fitness < 0): continue
                if (child2_fitness >= ave_fitness or random() < np.exp((child2_fitness-ave_fitness)/c)):
                    child2.array = tmp
                    child2.parent = other.array
                    best2 = max(child2_fitness, best2)
                    break

        child1.array -= np.array(child1.array[0])
        if (best1==-1):
            child1.array = self.array
        if (best2==-1):
            child2.array = other.array
        result_chromosomes: List[HP] = [child1, child2]
        # logging.info(f"child1: {result_chromosomes[0].array.tolist()}")
        # logging.info(f"child2: {child2.array.tolist()}")
        return result_chromosomes
    
    def __str__(self) -> str:
        # return f'error: {self.cnt_error} trouble: {self.cnt_trouble} '
        return f'contacts =  {self.get_fitness()-1} array = {self.array.tolist()}'


# hp = "HPHPPHHPHPPHPHHPPHPH"; id="20a" #optimal: 9
hp = "PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH"; id="48a" #optimal: 23(not 22)
population = 300
generation = 300
mutation_probability = 1.0
crossover_probability = 1.0

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(level=logging.WARNING, 
                    format='{asctime} [{levelname:.4}] {name}: {message}', 
                    style='{', 
                    filename=f"/home/u01230/midterm/results/2d/{id}/{timestamp}.log",
                    filemode='w')
logging.getLogger().setLevel(logging.INFO)

logging.info(f'hp: {hp}')
logging.info(f'population: {population}')
logging.info(f'generation: {generation}')
logging.info(f'mutation_probability: {mutation_probability}')
logging.info(f'crossover_probability: {crossover_probability}')
logging.info(f'atavism: 0.1')

simple_equation_initial_population: List[HP] = \
    [HP.make_random_instance(hp) for _ in range(population)]
ga: GeneticAlgorithm = GeneticAlgorithm(
    initial_population=simple_equation_initial_population,
    threshold=100,
    max_generations=generation,
    mutation_probability=mutation_probability,
    crossover_probability=crossover_probability,
    selection_type=GeneticAlgorithm.SELECTION_TYPE_ROULETTE_WHEEL)
_ = ga.run_algorithm()


