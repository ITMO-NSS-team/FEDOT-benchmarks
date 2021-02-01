import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


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


def plox_boxplot(self, df: pd.DataFrame):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="exp_number", y="AUC", data=df, color="seagreen")
    plt.title('ROC-AUC metric for each genetic scheme')
    plt.ylabel('ROC-AUC value')
    plt.xlabel('genetic scheme')
    plt.show()
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="exp_number", y="complexity", data=df, color="seagreen")
    plt.title('Complexity metric for each genetic scheme')
    plt.ylabel('Complexity value')
    plt.xlabel('genetic scheme')
    plt.show()
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="exp_number", y="n_models", data=df, color="seagreen")
    plt.title('Number of models c for each genetic scheme')
    plt.ylabel('Number of models')
    plt.xlabel('genetic scheme')
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

    def get_experiment_report(self, number_of_experiment: str):
        for name_of_dataset in self.datasets:
            for label in self.labels:
                composer_list = []
                for i in range(self.runs):
                    path = f'D:\результаты экспериментов\{name_of_dataset}\{number_of_experiment}\{label}\{name_of_dataset}_{label}_run_number_{i}_composer_history.npy'
                    massive = np.load(path, allow_pickle=True)
                    composer = massive[0]
                    composer_list.append(composer)

                self.viz_pareto(composer_list, name_of_dataset, self.runs, label)
        return composer_list

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
    number_of_experiment = '2_experiment'
    labels = labels_dict['2_experiment']
    report_model = PMLB_report(labels=labels,
                               runs=runs,
                               datasets=datasets)
    datasets = report_model.choose_clf_datasets()
    report_model.get_experiment_report(number_of_experiment)
