# GA

from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.ga_creators.float_vector_creator import GAFloatVectorCreator
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation import FloatVectorUniformNPointMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.statistics.minimal_print_statistics import MinimalPrintStatistics
from eckity.subpopulation import Subpopulation
from eckity.termination_checkers.best_fitness_stagnation_termination_checker import BestFitnessStagnationTerminationChecker
import eckity.termination_checkers
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
import numpy as np
import constants
import setup


class BugHuntingEvaluator(SimpleIndividualEvaluator):
    def __init__(self, items=None, max_weight=30):
        super().__init__()

        if items is None:
            # Generate ramdom items for the problem (keys=weights, values=values)
            # items = np.array(random_x(first_dim=m,second_dim=n, bounds=bounds))  **************
            items = np.array(np.random.rand(constants.n_features)) * constants.multip

            # items = np.array(random_x())
        self.items = items
        self.max_weight = max_weight


    def evaluate_individual(self, individual):
          return setup.run_test(individual.vector)


def run_ga(pop_size=50, max_gen=100, bounds=constants.bounds):

      algo = SimpleEvolution(
              Subpopulation(creators=GAFloatVectorCreator(length=len(bounds),bounds=bounds),
                            population_size=pop_size,
                            # user-defined fitness evaluation method
                            evaluator=BugHuntingEvaluator(),
                            # maximization problem (fitness is sum of values), so higher fitness is better
                            higher_is_better=True,
                            # genetic operators sequence to be applied in each generation
                            operators_sequence=[
                                VectorKPointsCrossover(probability=0.5, k=2),
                                FloatVectorUniformNPointMutation(n=10, probability=0.15)
                            ],
                            selection_methods=[
                                # (selection method, selection probability) tuple
                                (TournamentSelection(tournament_size=4, higher_is_better=True), 1)
                            ]),
              breeder=SimpleBreeder(),
              max_workers=1,
              max_generation=max_gen,
              # statistics=BestAverageWorstStatistics()
              # statistics=MinimalPrintStatistics()
              termination_checker = BestFitnessStagnationTerminationChecker(stagnation_generations_to_terminate=100),
              statistics=None
          )

      # evolve the generated initial population
      algo.n_elite = 1
      algo.evolve()
      best=algo.execute()

      return best, setup.prob(best)


