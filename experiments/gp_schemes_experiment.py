import csv
import datetime
import gc
import os

import numpy as np

from typing import List
from experiments.credit_scoring_experiment import run_credit_scoring_problem
from experiments.viz import show_history_optimization_comparison
from FEDOT.core.composer.optimisers.crossover import CrossoverTypesEnum
from FEDOT.core.composer.optimisers.gp_optimiser import GPChainOptimiserParameters
from FEDOT.core.composer.optimisers.inheritance import GeneticSchemeTypesEnum
from FEDOT.core.composer.optimisers.mutation import MutationTypesEnum
from FEDOT.core.composer.optimisers.regularization import RegularizationTypesEnum
from FEDOT.core.composer.optimisers.selection import SelectionTypesEnum
from FEDOT.core.utils import project_root


def write_header_to_csv(f):
    f = f'../../../tmp/{f}'
    if not os.path.isdir('../../../tmp'):
        os.mkdir('../../../tmp')
    with open(f, 'w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        writer.writerow(['t_opt', 'regular', 'AUC', 'n_models', 'n_layers'])


def add_result_to_csv(f, t_opt, regular, auc, n_models, n_layers):
    f = f'../../../tmp/{f}'
    with open(f, 'a', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        writer.writerow([t_opt, regular, auc, n_models, n_layers])


def _reduced_history_best(history, generations: int, pop_size: int, pop_sizes: List[List[int]] = None):
    reduced = []
    for gen in range(generations):
        if pop_sizes is not None:
            pop_size = pop_sizes[gen]
            if gen == 0:
                num_from = 0
            else:
                num_from = sum(pop_sizes[:gen])

            fitness_values = [abs(individ) for individ in history[num_from: num_from + pop_size]]
        else:
            fitness_values = [abs(individ) for individ in history[gen * pop_size: (gen + 1) * pop_size]]
        best = max(fitness_values)
        print(f'Max in generation #{gen}: {best}')
        reduced.append(best)

    return reduced


if __name__ == '__main__':
    max_amount_of_time = 800
    step = 800
    file_path_train = 'cases/data/scoring/scoring_train.csv'
    full_path_train = os.path.join(str(project_root()), file_path_train)
    file_path_test = 'cases/data/scoring/scoring_test.csv'
    full_path_test = os.path.join(str(project_root()), file_path_test)
    file_path_result = 'regular_exp.csv'
    history_file = 'history.csv'
    write_header_to_csv(file_path_result)
    time_amount = step
    genetic_schemes_set = [GeneticSchemeTypesEnum.parameter_free, GeneticSchemeTypesEnum.parameter_free,
                           GeneticSchemeTypesEnum.steady_state, GeneticSchemeTypesEnum.steady_state]
    depth_config = [False, True, False, True]
    max_depths = [4, 4, 4, 4]
    history_gp = [[] for _ in range(len(genetic_schemes_set))]
    param_free_pop_sizes = []
    pop_size = 20
    iterations = 20
    runs = 4
    parameter_free_launches = []
    while time_amount <= max_amount_of_time:
        for type_num, scheme_type in enumerate(genetic_schemes_set):
            if scheme_type == GeneticSchemeTypesEnum.parameter_free:
                parameter_free_launches.append(type_num)
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
                historical_fitness = [chain.fitness for chain in composer.history]
                history_gp[type_num].append(historical_fitness)
                if genetic_scheme_type == GeneticSchemeTypesEnum.parameter_free:
                    param_free_pop_sizes.append(composer.pop_sizes)
        time_amount += step
    reduced_fitness_gp = [[] for _ in range(len(history_gp))]
    for launch_num in range(len(history_gp)):
        for history in history_gp[launch_num]:
            if launch_num in parameter_free_launches:
                fitness = _reduced_history_best(history, iterations, pop_size, param_free_pop_sizes[0])
                param_free_pop_sizes = param_free_pop_sizes[1:]
            else:
                fitness = _reduced_history_best(history, iterations, pop_size)
            reduced_fitness_gp[launch_num].append(fitness)
    np.save('reduced_fitness_gp', reduced_fitness_gp)
    print(reduced_fitness_gp)
    m = [_ * pop_size for _ in range(iterations)]
    show_history_optimization_comparison(optimisers_fitness_history=reduced_fitness_gp,
                                         iterations=[_ for _ in range(iterations)],
                                         labels=['parameter-free', 'parameter-free with depth config', 'steady-state',
                                                 'steady-state with depth config'])
