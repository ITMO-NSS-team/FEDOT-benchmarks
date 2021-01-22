import csv
import datetime
import gc
import os

import numpy as np

from typing import List
from experiments.credit_scoring_experiment import run_credit_scoring_problem
from experiments.viz import show_history_optimization_comparison
from experiments.multi_objective_schemes_experiment import proj_root

from fedot.core.composer.optimisers.crossover import CrossoverTypesEnum
from fedot.core.composer.optimisers.gp_optimiser import GPChainOptimiserParameters
from fedot.core.composer.optimisers.inheritance import GeneticSchemeTypesEnum
from fedot.core.composer.optimisers.mutation import MutationTypesEnum
from fedot.core.composer.optimisers.regularization import RegularizationTypesEnum
from fedot.core.composer.optimisers.selection import SelectionTypesEnum


def write_header_to_csv(f, row: List[str] = None):
    f = f'../../tmp/{f}'
    if not os.path.isdir('../../tmp'):
        os.mkdir('../../tmp')
    with open(f, 'w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        if not row:
            row = ['t_opt', 'regular', 'AUC', 'n_models', 'n_layers']
        writer.writerow(row)


def add_result_to_csv(f, t_opt, regular, auc, n_models, n_layers, exp_number=None, iteration=None, complexity=None):
    f = f'../../tmp/{f}'
    with open(f, 'a', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        row = [t_opt, regular, auc, n_models, n_layers]
        for col in [complexity, iteration + 1, exp_number]:
            if col is not None:
                row.insert(0, col)
        writer.writerow(row)


def _reduced_history_best(history: List[List[float]]):
    reduced = []
    for i, fitness_values in enumerate(history):
        best = max(fitness_values)
        print(f'Max in generation #{i}: {best}')
        reduced.append(best)

    return reduced


def results_preprocess_and_quality_visualisation(history_gp, labels, iterations):
    reduced_fitness_gp = [[] for _ in range(len(history_gp))]
    for launch_num in range(len(history_gp)):
        for history in history_gp[launch_num]:
            fitness = _reduced_history_best(history)
            reduced_fitness_gp[launch_num].append(fitness)
    np.save('reduced_fitness_gp', reduced_fitness_gp)
    show_history_optimization_comparison(optimisers_fitness_history=reduced_fitness_gp,
                                         iterations=[_ for _ in range(iterations)],
                                         labels=labels)


if __name__ == '__main__':
    max_amount_of_time = 800
    step = 800
    file_path_train = 'test_cases/scoring/data/scoring_train.csv'
    full_path_train = os.path.join(str(proj_root()), file_path_train)
    file_path_test = 'test_cases/scoring/data/scoring_test.csv'
    full_path_test = os.path.join(str(proj_root()), file_path_test)
    file_path_result = 'regular_exp.csv'
    history_file = 'history.csv'
    write_header_to_csv(file_path_result)
    time_amount = step
    genetic_schemes_set = [GeneticSchemeTypesEnum.parameter_free, GeneticSchemeTypesEnum.parameter_free,
                           GeneticSchemeTypesEnum.steady_state, GeneticSchemeTypesEnum.steady_state]
    depth_config = [False, True, False, True]
    max_depths = [3, 4, 3, 4]
    history_gp = [[] for _ in range(len(genetic_schemes_set))]
    pop_size = 4
    iterations = 3
    runs = 4
    while time_amount <= max_amount_of_time:
        for type_num, scheme_type in enumerate(genetic_schemes_set):
            for run in range(runs):
                gc.collect()
                selection_types = [SelectionTypesEnum.tournament]
                crossover_types = [CrossoverTypesEnum.one_point, CrossoverTypesEnum.subtree]
                mutation_types = [MutationTypesEnum.simple, MutationTypesEnum.growth, MutationTypesEnum.reduce]
                regular_type = RegularizationTypesEnum.decremental
                genetic_scheme_type = scheme_type
                with_auto_depth_configuration = depth_config[type_num]
                max_depth_in_exp = max_depths[type_num]

                optimiser_parameters = GPChainOptimiserParameters(selection_types=selection_types,
                                                                  crossover_types=crossover_types,
                                                                  mutation_types=mutation_types,
                                                                  regularization_type=regular_type,
                                                                  genetic_scheme_type=genetic_scheme_type,
                                                                  with_auto_depth_configuration=
                                                                  with_auto_depth_configuration)
                roc_auc, chain, composer = run_credit_scoring_problem(full_path_train, full_path_test,
                                                                      max_lead_time=datetime.timedelta(
                                                                          minutes=time_amount),
                                                                      gp_optimiser_params=optimiser_parameters,
                                                                      pop_size=pop_size, generations=iterations,
                                                                      max_depth=max_depth_in_exp)

                is_regular = regular_type == RegularizationTypesEnum.decremental
                add_result_to_csv(file_path_result, time_amount, is_regular, round(roc_auc, 4), len(chain.nodes),
                                  chain.depth)

                historical_fitness = [[-chain.fitness for chain in pop] for pop in composer.history.individuals]
                history_gp[type_num].append(historical_fitness)
        time_amount += step

    labels = ['parameter-free', 'parameter-free with depth config', 'steady-state', 'steady-state with depth config']
    results_preprocess_and_quality_visualisation(history_gp=history_gp, labels=labels, iterations=iterations)
