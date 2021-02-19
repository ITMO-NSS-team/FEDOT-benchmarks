import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from experiments.viz import viz_hv_comparison, viz_pareto_fronts_comparison
from experiments.gp_operators_experiment import results_preprocess_and_quality_visualisation
from fedot.core.composer.optimisers.multi_objective_fitness import MultiObjFitness


def viz_pareto_fronts_by_iteration(fronts, labels, objectives_order=(1, 0),
                                   objectives_names=('ROC-AUC penalty metric', 'Computation time'),
                                   name_of_dataset=None, save=True):
    fig, ax = plt.subplots()
    current_palette = sns.color_palette('Dark2')
    for i, pareto_front in enumerate(fronts):
        color = np.array(current_palette[i])
        c = color.reshape(1, -1)
        ax.scatter(sorted(pareto_front[0][objectives_order[0]]), sorted(pareto_front[0][objectives_order[1]]), c=c)
        ax.plot(sorted(pareto_front[0][objectives_order[0]]), sorted(pareto_front[0][objectives_order[1]]), color=color,
                label=labels[i])
    plt.xlabel(objectives_names[objectives_order[0]], fontsize=15)
    plt.ylabel(objectives_names[objectives_order[1]], fontsize=15)
    plt.yticks(fontsize=12)
    ax.set_title('Pareto front', fontsize=15)
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
                 datasets: list, ):
        self.labels = labels
        self.runs = runs
        self.datasets = datasets

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
                   label: str,
                   single_flag: bool = True):

        if label in ['steady_state single-obj', 'steady_state single-obj penalty']:
            single_flag = False

        runs = [str(i) + '_iteration' for i in range(1, runs + 1)]
        pareto_history = []
        for composer in composer_list:
            pareto_fronts_metrics = []
            if single_flag:
                archive_len = len(composer.history.archive_history)
                pareto_front = composer.history.archive_history[archive_len - 1]
                roc_auc_list = [-ind.fitness.values[0] for ind in pareto_front]
                complexity_list = [ind.fitness.values[1] for ind in pareto_front]
                pareto_fronts_metrics.append([roc_auc_list, complexity_list])
                pareto_history.append(pareto_fronts_metrics)
            else:
                archive_len = len(composer.history.individuals)
                pareto_front = composer.history.individuals[archive_len - 1]
                roc_auc_list = [-ind.fitness for ind in pareto_front]
                complexity_list = [ind.computation_time for ind in pareto_front]
                pareto_fronts_metrics.append([roc_auc_list, complexity_list])
                pareto_history.append(pareto_fronts_metrics)

        viz_pareto_fronts_by_iteration(pareto_history, labels=runs, name_of_dataset=name_of_dataset + "_" + label)

    def viz_pareto_comparison(self, pareto_run_numbers=(1, 1, 1, 1), name_of_dataset='None'):
        all_history_report = self.get_experiment_report()
        selected_composers = [all_history_report[exp_num][front_num] for exp_num, front_num in
                              enumerate(pareto_run_numbers)]
        pareto_fronts_metrics = []
        for composer in selected_composers:
            archive_len = len(composer.history.archive_history)
            pareto_front = composer.history.archive_history[archive_len - 1]
            roc_auc_list = [-ind.fitness.values[0] for ind in pareto_front]
            complexity_list = [ind.fitness.values[1] for ind in pareto_front]
            pareto_fronts_metrics.append([roc_auc_list, complexity_list])

        objectives_names = ('ROC-AUC metric', 'Computation time')
        viz_pareto_fronts_comparison(pareto_fronts_metrics, labels=self.labels, name_of_dataset=name_of_dataset,
                                     objectives_names=objectives_names)

    def viz_hv(self, iterations: int, color_pallete=sns.color_palette('Dark2'), name_of_dataset='None'):
        all_history_report = self.get_experiment_report()

        viz_hv_comparison(labels=self.labels, all_history_report=all_history_report, name_of_dataset=name_of_dataset,
                          color_pallete=color_pallete, iterations=iterations)

    def viz_best_quality_comparison(self, iterations, name_of_dataset='None'):
        all_history_report = self.get_experiment_report()
        history_quality_gp = [[] for _ in range(len(self.labels))]
        for exp_num, exp in enumerate(all_history_report):
            for comp_num, composer in enumerate(exp):
                if type(all_history_report[exp_num][comp_num].history.archive_history[
                            0][0].fitness) is MultiObjFitness:
                    historical_quality = [
                        [-chain.fitness.values[0] for chain in pop] + [-chain.fitness.values[0] for chain in
                                                                       composer.history.archive_history[i]] for i, pop
                        in enumerate(composer.history.individuals)]
                    history_quality_gp[exp_num].append(historical_quality)

                else:
                    historical_quality = [[-chain.fitness for chain in pop] for pop in composer.history.individuals]
                    history_quality_gp[exp_num].append(historical_quality)
        results_preprocess_and_quality_visualisation(history_gp=history_quality_gp, labels=self.labels,
                                                     iterations=iterations, name_of_dataset=name_of_dataset)

if __name__ == '__main__':
    labels_dict = {'1_experiment':
                       ['steady_state single-obj', 'steady_state single-obj penalty', 'steady-state multi-obj'],
                   '2_experiment':
                       ['parameter-free', 'parameter-free with depth config', 'steady-state',
                        'steady-state with depth config'],
                   '3_experiment':
                       ['nsga selection', 'spea2 selection']}
    runs = 4
    datasets = ['churn', 'dis']
    number_of_experiment = '3_experiment'
    labels = labels_dict['2_experiment']
    report_model = PMLB_report(labels=labels,
                               runs=runs,
                               datasets=datasets)
    # datasets = report_model.choose_clf_datasets()
    report_model.get_experiment_report()
