import datetime
import gc
import os
from pathlib import Path
import seaborn as sns
from pygmo import hypervolume
from experiments.credit_scoring_experiment import run_credit_scoring_problem
from experiments.viz import show_history_optimization_comparison

from fedot.core.composer.optimisers.crossover import CrossoverTypesEnum
from fedot.core.composer.optimisers.gp_optimiser import GPChainOptimiserParameters
from fedot.core.composer.optimisers.inheritance import GeneticSchemeTypesEnum
from fedot.core.composer.optimisers.mutation import MutationTypesEnum
from fedot.core.composer.optimisers.regularization import RegularizationTypesEnum
from fedot.core.composer.optimisers.selection import SelectionTypesEnum
from fedot.core.composer.visualisation import ComposerVisualiser
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, ComplexityMetricsEnum, \
    MetricsRepository
from experiments.gp_schemes_experiment import write_header_to_csv, add_result_to_csv, \
    results_preprocess_and_quality_visualisation
from experiments.viz import viz_pareto_fronts_comparison

all_results_chains_file = 'all_result_chains.csv'


def proj_root():
    return Path(__file__).parent.parent


def run_multi_obj_exp(selection_types, history_file='history.csv', labels=None, genetic_schemes_set=None,
                      depth_config=None, iterations=10,
                      runs=1, pop_sizes=(10, 10, 20, 20), crossover_types=None, metrics=None, mutation_types=None,
                      regular_type=RegularizationTypesEnum.decremental, train_path=None, test_path=None,
                      visualize_pareto=False, visualize_hv=False):
    max_amount_of_time = 800
    step = 800
    file_path_train = 'test_cases/scoring/data/scoring_train.csv'
    full_path_train = os.path.join(str(proj_root()), file_path_train)
    full_path_train = train_path
    file_path_test = 'test_cases/scoring/data/scoring_test.csv'
    full_path_test = os.path.join(str(proj_root()), file_path_test)
    full_path_test = test_path
    file_path_result = 'multiobj_exp_all.csv'
    file_path_best = 'multiobj_exp_best.csv'
    row = ['exp_number', 'iteration', 'complexity', 't_opt', 'regular', 'AUC', 'n_models', 'n_layers']
    write_header_to_csv(file_path_result, row=row)
    time_amount = step
    if not metrics:
        metrics = [ClassificationMetricsEnum.ROCAUC, ComplexityMetricsEnum.computation_time]
    max_depths = [3, 3, 3, 3]
    start_depth = [2, 2, 2, 2]  # starting depth for 1st population initialization
    history_quality_gp = [[] for _ in range(len(genetic_schemes_set))]
    inds_history_gp = [[] for _ in range(len(genetic_schemes_set))]
    fitness_history_gp = [[] for _ in range(len(genetic_schemes_set))]
    pareto_fronts_metrics = []
    n = 0
    while time_amount <= max_amount_of_time:
        for type_num, scheme_type in enumerate(genetic_schemes_set):
            for run in range(runs):
                n += 1
                gc.collect()
                if not crossover_types:
                    crossover_types = [CrossoverTypesEnum.one_point, CrossoverTypesEnum.subtree]
                if not mutation_types:
                    mutation_types = [MutationTypesEnum.simple, MutationTypesEnum.growth, MutationTypesEnum.reduce]
                genetic_scheme_type = scheme_type
                with_auto_depth_configuration = depth_config[type_num]
                max_depth_in_exp = max_depths[type_num]
                start_depth_in_exp = start_depth[type_num]

                if any([type(m) is list for m in metrics]):
                    metric = metrics[type_num]
                else:
                    metric = metrics
                selection_type = selection_types[type_num]
                optimiser_parameters = GPChainOptimiserParameters(selection_types=selection_type,
                                                                  crossover_types=crossover_types,
                                                                  mutation_types=mutation_types,
                                                                  regularization_type=regular_type,
                                                                  genetic_scheme_type=genetic_scheme_type,
                                                                  with_auto_depth_configuration=
                                                                  with_auto_depth_configuration)

                calculated_metrics, chains, composer = run_credit_scoring_problem(full_path_train, full_path_test,
                                                                                  max_lead_time=datetime.timedelta(
                                                                                      minutes=time_amount),
                                                                                  gp_optimiser_params=optimiser_parameters,
                                                                                  pop_size=pop_sizes[type_num],
                                                                                  generations=iterations,
                                                                                  max_depth=max_depth_in_exp,
                                                                                  start_depth=start_depth_in_exp,
                                                                                  metrics=metric)

                is_regular = regular_type == RegularizationTypesEnum.decremental

                try:
                    historical_fit = [[[obj for obj in chain.fitness.values] for chain in pop] for pop
                                      in
                                      composer.history.individuals]
                except Exception as ex:
                    print(ex)
                    historical_fit = [[[chain.fitness] for chain in pop] for pop
                                      in
                                      composer.history.individuals]
                fitness_history_gp[type_num].append(historical_fit)
                inds_history_gp[type_num].append(composer.history.individuals)

                if visualize_pareto:
                    archive_len = len(composer.history.archive_history)
                    pareto_front = composer.history.archive_history[archive_len - 1]
                    roc_auc_list = [-ind.fitness.values[0] for ind in pareto_front]
                    complexity_list = [ind.fitness.values[1] for ind in pareto_front]

                    pareto_fronts_metrics.append([roc_auc_list, complexity_list])

                if type(metric) is list:
                    roc_auc_metrics = calculated_metrics[0]
                    complexity_metrics = calculated_metrics[1]
                else:
                    roc_auc_metrics = calculated_metrics

                if type(metric) is list:
                    historical_quality = [
                        [-chain.fitness.values[0] for chain in pop] + [-chain.fitness.values[0] for chain in
                                                                       composer.history.archive_history[i]] for i, pop
                        in enumerate(composer.history.individuals)]
                    history_quality_gp[type_num].append(historical_quality)

                else:
                    historical_quality = [[-chain.fitness for chain in pop] for pop in composer.history.individuals]
                    history_quality_gp[type_num].append(historical_quality)

                for i, roc_auc in enumerate(roc_auc_metrics):
                    if type(metric) is list:
                        compl = complexity_metrics[i]
                    else:
                        compl = MetricsRepository().metric_by_id(ComplexityMetricsEnum.computation_time)(chains[i])

                    add_result_to_csv(file_path_best, time_amount, is_regular, round(roc_auc, 4),
                                      len(chains[i].nodes),
                                      chains[i].depth, exp_type=labels[type_num], iteration=run,
                                      complexity=compl, exp_number=type_num)

        time_amount += step

    if runs > 1:
        results_preprocess_and_quality_visualisation(history_gp=history_quality_gp, labels=labels,
                                                     iterations=iterations)
    if visualize_pareto:
        if runs == 1:
            pareto_metrics = pareto_fronts_metrics
        else:
            pareto_metrics = [pareto_fronts_metrics[i] for i in range(0, len(pareto_fronts_metrics), runs)]
        viz_pareto_fronts_comparison(pareto_metrics, labels=labels)
    if visualize_hv:
        exps_ref_points = []
        for exp_num, exp_history in enumerate(inds_history_gp):
            max_qual, max_compl = [], []
            for run_num, run_history in enumerate(exp_history):
                all_objectives = ComposerVisualiser.objectives_transform(run_history, objectives_numbers=(0, 1),
                                                                         transform_from_minimization=False)
                max_qual.append(max(all_objectives[0]) + 0.01)
                max_compl.append(max(all_objectives[1]) + 0.01)
            exps_ref_points.append((max(max_qual), max(max_compl)))

        hv_set = []
        for exp_num, exp_history in enumerate(inds_history_gp):
            hv_set.append([])
            for run_num, run_history in enumerate(exp_history):
                ref = exps_ref_points[exp_num]
                hv_set[exp_num].append([hypervolume(pop).compute(ref) for pop in fitness_history_gp[exp_num][run_num]])

        color_pallete = sns.color_palette('Dark2')
        show_history_optimization_comparison(optimisers_fitness_history=hv_set,
                                             iterations=[_ for _ in range(iterations)],
                                             labels=labels, color_pallete=color_pallete, ylabel='Hypervolume')


def exp_self_config_vs_fix_params(train_path: str,
                                  test_path: str):
    history_file = 'history_selfconf_vs_fixparams.csv'
    labels = ['parameter-free', 'parameter-free with depth config', 'steady-state',
              'steady-state with depth config']
    genetic_schemes_set = [GeneticSchemeTypesEnum.parameter_free, GeneticSchemeTypesEnum.parameter_free,
                           GeneticSchemeTypesEnum.steady_state, GeneticSchemeTypesEnum.steady_state]
    metrics = [[ClassificationMetricsEnum.ROCAUC, ComplexityMetricsEnum.computation_time] for _ in range(len(labels))]
    multi_obj_sel = [SelectionTypesEnum.nsga2, SelectionTypesEnum.spea2]
    selection_types = [multi_obj_sel for _ in range(len(labels))]
    depth_config_option = [False, True, False, True]  # depth configuration option (Active/No active)
    run_multi_obj_exp(history_file=history_file, labels=labels, genetic_schemes_set=genetic_schemes_set, runs=3,
                      metrics=metrics, selection_types=selection_types, depth_config=depth_config_option,
                      train_path=train_path, test_path=test_path,
                      visualize_pareto=True, visualize_hv=True)


def exp_single_vs_multi_objective(train_path: str,
                                  test_path: str):
    history_file = 'history_single_vs_multiobj.csv'
    labels = ['steady_state single-obj', 'steady_state single-obj penalty', 'steady-state multi-obj']
    runs = 4
    genetic_schemes_set = [GeneticSchemeTypesEnum.steady_state for _ in range(len(labels))]
    metrics = [ClassificationMetricsEnum.ROCAUC, ClassificationMetricsEnum.ROCAUC_penalty,
               [ClassificationMetricsEnum.ROCAUC, ComplexityMetricsEnum.computation_time]]
    single_obj_sel = [SelectionTypesEnum.tournament]
    multi_obj_sel = [SelectionTypesEnum.nsga2, SelectionTypesEnum.spea2]
    selection_types = [single_obj_sel, single_obj_sel, multi_obj_sel]
    depth_config_option = [False, False, False, False]  # depth configuration option (Active/No active)
    run_multi_obj_exp(history_file=history_file, labels=labels, genetic_schemes_set=genetic_schemes_set, runs=runs,
                      metrics=metrics, selection_types=selection_types, depth_config=depth_config_option,
                      train_path=train_path, test_path=test_path, )


def exp_multi_obj_selections(train_path: str,
                             test_path: str):
    history_file = 'history_selfconf_vs_fixparams.csv'
    labels = ['nsga selection', 'spea2 selection']
    genetic_schemes_set = [GeneticSchemeTypesEnum.parameter_free, GeneticSchemeTypesEnum.parameter_free]
    metrics = [[ClassificationMetricsEnum.ROCAUC, ComplexityMetricsEnum.computation_time] for _ in range(len(labels))]
    selection_types = [[SelectionTypesEnum.nsga2], [SelectionTypesEnum.spea2]]
    depth_config_option = [False, False]  # depth configuration option (Active/No active)
    run_multi_obj_exp(history_file=history_file, labels=labels, genetic_schemes_set=genetic_schemes_set, runs=4,
                      metrics=metrics, selection_types=selection_types, depth_config=depth_config_option,
                      train_path=train_path, test_path=test_path,
                      visualize_pareto=True)


def exp_complexity_metrics():
    history_file = 'history_selfconf_vs_fixparams.csv'
    labels = ['computation time', 'structural complexity']
    genetic_schemes_set = [GeneticSchemeTypesEnum.steady_state, GeneticSchemeTypesEnum.steady_state]
    metrics = [[ClassificationMetricsEnum.ROCAUC, ComplexityMetricsEnum.computation_time],
               [ClassificationMetricsEnum.ROCAUC, ComplexityMetricsEnum.structural]]
    multi_obj_sel = [SelectionTypesEnum.nsga2, SelectionTypesEnum.spea2]
    selection_types = [multi_obj_sel, multi_obj_sel]
    depth_config_option = [False, False]  # depth configuration option (Active/No active)
    run_multi_obj_exp(history_file=history_file, labels=labels, genetic_schemes_set=genetic_schemes_set, runs=4,
                      metrics=metrics, selection_types=selection_types, depth_config=depth_config_option,
                      visualize_hv=True)


if __name__ == '__main__':
    # exp_single_vs_multi_objective()
    exp_self_config_vs_fix_params()
    # exp_multi_obj_selections()
    # exp_complexity_metrics()
