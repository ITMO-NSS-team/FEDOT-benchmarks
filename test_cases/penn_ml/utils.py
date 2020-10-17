import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from typing import Tuple
from experiments.viz import viz_hv_comparison, viz_pareto_fronts_comparison
from experiments.gp_operators_experiment import results_preprocess_and_quality_visualisation
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.composer.optimisers.multi_objective_fitness import MultiObjFitness
from fedot.core.composer.visualisation import ComposerVisualiser
from experiments.multi_objective_schemes_experiment import extract_quality_list


def viz_pareto_fronts_by_iteration(fronts, labels, objectives_order=(1, 0),
                                   objectives_names=('ROC-AUC penalty metric', 'Computation time'),
                                   name_of_dataset=None, save=True):
    fig, ax = plt.subplots()
    current_palette = sns.color_palette('Dark2')
    for i, pareto_front in enumerate(fronts):
        color = np.array(current_palette[i])
        c = color.reshape(1, -1)
        ax.scatter(sorted(pareto_front[objectives_order[0]]), sorted(pareto_front[objectives_order[1]]), c=c)
        ax.plot(sorted(pareto_front[objectives_order[0]]), sorted(pareto_front[objectives_order[1]]), color=color,
                label=labels[i])
    plt.xlabel(objectives_names[objectives_order[0]], fontsize=15)
    plt.ylabel(objectives_names[objectives_order[1]], fontsize=15)
    plt.yticks(fontsize=12)
    ax.set_title('Pareto frontiers', fontsize=15)
    ax.legend(loc='lower right', shadow=False, fontsize=15)
    fig.set_figwidth(8)
    fig.set_figheight(8)
    if save:
        if not os.path.isdir('../../tmp'):
            os.mkdir('../../tmp')

        file_name = name_of_dataset + '_pareto_fronts_comp.png'
        path = f'../../tmp/{file_name}'
        plt.savefig(path, bbox_inches='tight')

    plt.show()


class PMLB_report():

    def __init__(self,
                 labels: list,
                 runs: int,
                 datasets: list, task: Task):
        self.labels = labels
        self.runs = runs
        self.datasets = datasets
        self.task = task

    def choose_clf_datasets(self):
        summary_stats = pd.read_csv(r'./datasets/all_summary_stats.tsv', sep='\t')
        clf_datasets = summary_stats[summary_stats['task'] == 'classification']
        clf_datasets = clf_datasets[(clf_datasets['n_instances'] > 1000) & (clf_datasets['n_classes'] == 2)]
        return clf_datasets

    def create_dataframe(self, path: str):
        names = ['exp_number', 'iteration', 'complexity', 't_opt', 'regular', 'AUC', 'n_models', 'n_layers']
        df = pd.read_csv(path, names=names, sep=',')
        return df

    def get_experiment_report(self):
        composers = []
        for name_of_dataset in self.datasets:
            for label in self.labels:
                composer_list = []
                for i in range(self.runs):
                    tmp_folder = str(i + 1) + '_experiment'
                    path = f'D:\результаты экспериментов\{name_of_dataset}\{tmp_folder}\{name_of_dataset}_{label}_run_number_{i + 1}_composer_history.npy'
                    massive = np.load(path, allow_pickle=True)
                    composer = massive[0]
                    composer_list.append(composer)

                self.viz_pareto(composer_list, name_of_dataset, self.runs, label)
                composers.append(composer_list)
        return composers

    def viz_pareto(self,
                   composer_list: list,
                   name_of_dataset: str,
                   runs: int,
                   label: str, relative_complexity: bool = False):

        runs = [str(i) + '_iteration' for i in range(1, runs + 1)]
        pareto_fronts_metrics = self.get_pareto_data(selected_composers=composer_list,
                                                     relative_complexity=relative_complexity)
        viz_pareto_fronts_by_iteration(pareto_fronts_metrics, labels=runs,
                                       name_of_dataset=name_of_dataset + "_" + label)

    def get_pareto_data(self, selected_composers, relative_complexity):
        pareto_fronts_metrics = []
        compl_metrics = []
        quality_metrics = []
        for comp_num, composer in enumerate(selected_composers):
            archive_len = len(composer.history.archive_history)
            pareto_front = composer.history.archive_history[archive_len - 1]
            if type(composer.history.individuals[0][0].fitness) is MultiObjFitness:
                quality_list = extract_quality_list(task=self.task, pop=pareto_front)
                complexity_list = [ind.fitness.values[1] for ind in pareto_front]
            else:
                quality_list = extract_quality_list(task=self.task, pop=pareto_front)
                complexity_list = [ind.computation_time for ind in pareto_front]

            if not relative_complexity:
                pareto_fronts_metrics.append([quality_list, complexity_list])
            else:
                quality_metrics.append(quality_list)
                compl_metrics.append(complexity_list)

        if relative_complexity:
            all_compl_metrics = []
            for compl_metr in compl_metrics:
                all_compl_metrics = all_compl_metrics + compl_metr
            max_compl, min_compl = max(all_compl_metrics), min(all_compl_metrics)
            compl_metrics = [[round((value / max_compl) * 100, 1) for value in exp] for exp in compl_metrics]
            pareto_fronts_metrics = [[quality_metrics[exp_num], compl_metrics[exp_num]] for exp_num in
                                     range(len(compl_metrics))]
        return pareto_fronts_metrics

    def viz_pareto_comparison(self, objectives_names: Tuple, labels: Tuple, pareto_run_numbers=None,
                              name_of_dataset='None',
                              relative_complexity=False, print_pareto_chains=False):
        if pareto_run_numbers is None:
            pareto_run_numbers = tuple([1 for _ in range(len(self.labels))])
        all_history_report = self.get_experiment_report()
        selected_composers = [all_history_report[exp_num][front_num] for exp_num, front_num in
                              enumerate(pareto_run_numbers)]

        pareto_fronts_metrics = self.get_pareto_data(selected_composers=selected_composers,
                                                     relative_complexity=relative_complexity)

        if print_pareto_chains:
            if not relative_complexity:
                max_compl = None
            self.pareto_chains_viz(selected_composers=selected_composers, relative_complexity=relative_complexity,
                                   max_compl=max_compl)

        viz_pareto_fronts_comparison(pareto_fronts_metrics, labels=labels, name_of_dataset=name_of_dataset,
                                     objectives_names=objectives_names)

    def pareto_chains_viz(self, selected_composers, relative_complexity, max_compl=None):
        for comp_num, composer in enumerate(selected_composers):
            archive_len = len(composer.history.archive_history)
            pareto_front = composer.history.archive_history[archive_len - 1]
            pareto_chains_dir = f'../../tmp/pareto_{self.labels[comp_num]}'
            if not os.path.isdir(pareto_chains_dir):
                os.mkdir(pareto_chains_dir)
            for ind in pareto_front:
                if relative_complexity:
                    ind_complexity_val = round((ind.fitness.values[1] / max_compl) * 100, 1)
                else:
                    ind_complexity_val = ind.fitness.values[1]
                ComposerVisualiser.visualise(ind,
                                             save_path=f'{pareto_chains_dir}/({-ind.fitness.values[0]}, \
                                             {(ind_complexity_val)}).png')

    def viz_hv(self, iterations: int, labels: Tuple, color_pallete=sns.color_palette('Dark2'), name_of_dataset='None'):
        all_history_report = self.get_experiment_report()

        viz_hv_comparison(labels=labels, all_history_report=all_history_report, name_of_dataset=name_of_dataset,
                          color_pallete=color_pallete, iterations=iterations)

    def viz_best_quality_comparison(self, iterations, xy_labels: Tuple, name_of_dataset='None'):
        all_history_report = self.get_experiment_report()
        history_quality_gp = [[] for _ in range(len(self.labels))]
        for exp_num, exp in enumerate(all_history_report):
            for comp_num, composer in enumerate(exp):
                if all_history_report[exp_num][comp_num].history.archive_history:
                    historical_quality = [extract_quality_list(pop=pop, task=self.task) + extract_quality_list(
                        pop=composer.history.archive_history[i], task=self.task) for i, pop in
                                          enumerate(composer.history.individuals)]
                    history_quality_gp[exp_num].append(historical_quality)

                else:
                    historical_quality = [
                        [chain.fitness if self.task.task_type == TaskTypesEnum.regression else -chain.fitness for chain
                         in pop] for pop in composer.history.individuals]
                    history_quality_gp[exp_num].append(historical_quality)
        results_preprocess_and_quality_visualisation(history_gp=history_quality_gp, labels=self.labels,
                                                     iterations=iterations, name_of_dataset=name_of_dataset,
                                                     task=self.task)


if __name__ == '__main__':
    labels_dict = {'1_experiment':
                       ['steady_state single-obj GP', 'steady_state single-obj penalty GP',
                        'steady-state multi-obj GP'],
                   '2_experiment':
                       ['parameter-free GP with fixed max_depth', 'parameter-free GP',
                        'steady-state GP with fixed max_depth',
                        'steady-state GP'],
                   '3_experiment':
                       ['parameter_free GP with nsga selection', 'parameter_free GP with spea2 selection']}
    task = Task(TaskTypesEnum.classification)
    runs = 4
    datasets = ['churn', 'dis']
    number_of_experiment = '3_experiment'
    labels = labels_dict['2_experiment']
    report_model = PMLB_report(labels=labels,
                               runs=runs,
                               datasets=datasets, task=task)
    # datasets = report_model.choose_clf_datasets()
    report_model.get_experiment_report()
