import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
import seaborn as sns

from itertools import cycle
from typing import List, Any, Tuple
from pygmo import hypervolume


def fitness_by_generations_boxplots(history_runs, iterations, name_of_dataset=None, save=True):
    iters = [it for it in range(iterations)]
    fitness_by_iter = []
    for it in iters:
        fitness_values = []
        for history in history_runs:
            value = history[it]
            fitness_values.append(value)
        fitness_by_iter.append(fitness_values)
    sns.boxplot(iters, fitness_by_iter, color="seagreen")
    plt.title('Fitness history by generations')
    plt.ylabel('Fitness')
    plt.xlabel('Generation, #')
    if save:
        if not os.path.isdir('../../tmp'):
            os.mkdir('../../tmp')

        file_name = name_of_dataset + '_fitness_by_generations_boxplots.png'
        path = f'../../tmp/{file_name}'
        plt.savefig(path, bbox_inches='tight')

    plt.show()


def show_fitness_history_all(history_runs, iterations, name_of_dataset=None, with_bands=False, save=True):
    color_to_take = cycle('bgrcmykw')
    iters = [it for it in range(iterations)]

    if not with_bands:
        for history in history_runs:
            sns.tsplot(history, iters, legend=True, color=next(color_to_take))
        plt.legend(labels=[idx for idx in range(len(history_runs))], loc='lower right')
    else:
        sns.tsplot(history_runs, iters, legend=True, color=next(color_to_take))
    plt.ylabel('Fitness')
    plt.xlabel('Iteration, #')
    if save:
        if not os.path.isdir('../../tmp'):
            os.mkdir('../../tmp')

        file_name = name_of_dataset + '_fitness_history_all.png'
        path = f'../../tmp/{file_name}'
        plt.savefig(path, bbox_inches='tight')

    plt.show()


def show_history_optimization_comparison(optimisers_fitness_history: List[List[int]], iterations: List[int],
                                         labels: List[str], color_pallete: str = 'bgrcmykw', xlabel='Generation, #',
                                         ylabel='Best fitness', name_of_dataset=None, save=True):
    color_to_take = cycle(color_pallete)
    plt.yticks(fontsize=12)
    for fitness_history in optimisers_fitness_history:
        sns.tsplot(fitness_history, iterations, legend=True, color=next(color_to_take))

    plt.xticks(range(0, max(iterations)), fontsize=13)
    plt.legend(labels=labels, loc='lower right', fontsize=12)
    plt.ylabel(ylabel, fontsize=13)
    plt.xlabel(xlabel, fontsize=13)
    plt.tight_layout()
    if save:
        if not os.path.isdir('../../tmp'):
            os.mkdir('../../tmp')

        if ylabel == 'Hypervolume':
            file_name = name_of_dataset + '_Hypervolume_comparison.png'
        else:
            file_name = name_of_dataset + '_history_optimization_comparison.png'
        path = f'../../tmp/{file_name}'
        plt.savefig(path, bbox_inches='tight')

    plt.show()

def objectives_transform(individuals: List[List[Any]], objectives_numbers: Tuple[int] = None,
                         transform_from_minimization=True):
    objectives_numbers = [i for i in range(
        len(individuals[0][0].fitness.values))] if not objectives_numbers else objectives_numbers
    all_inds = list(itertools.chain(*individuals))
    all_objectives = [[ind.fitness.values[i] for ind in all_inds] for i in objectives_numbers]
    if transform_from_minimization:
        all_objectives = list(
            map(lambda obj_values: obj_values if obj_values[0] > 0 else list(1 + np.array(obj_values)),
                all_objectives))
    return all_objectives

def viz_pareto_fronts_comparison(fronts, labels, objectives_order=(1, 0),
                                 objectives_names=('ROC-AUC penalty metric', 'Computation time'),
                                 name_of_dataset=None, save=True):
    fig, ax = plt.subplots()
    current_palette = sns.color_palette('Dark2')
    for i, pareto_front in enumerate(fronts):
        color = np.array(current_palette[i])
        c = color.reshape(1, -1)
        ax.scatter(pareto_front[objectives_order[0]], pareto_front[objectives_order[1]], c=c)
        ax.plot(pareto_front[objectives_order[0]], pareto_front[objectives_order[1]], color=color, label=labels[i])
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


def viz_hv_comparison(labels, iterations, all_history_report, name_of_dataset='None',
                      color_pallete=sns.color_palette('Dark2')):
    all_history_report_transformed = [
        [[front.items for front in comp_run.history.archive_history] for comp_run in hist] for hist in
        all_history_report]
    fitness_history_gp = [[[[[1 + it.fitness.values[0], it.fitness.values[1]] for it in front.items] for front in
                            comp_run.history.archive_history] for comp_run in hist] for hist in all_history_report]

    inds_history_gp = all_history_report_transformed

    ref = [[], []]
    for exp_history in inds_history_gp:
        max_qual, max_compl = [], []
        for run_history in exp_history:
            all_objectives = objectives_transform(run_history, objectives_numbers=(0, 1),
                                                                     transform_from_minimization=True)
            max_qual.append(max(all_objectives[0]) + 0.0001)
            max_compl.append(max(all_objectives[1]) + 0.0001)
        ref[0].append(max(max_qual))
        ref[1].append(max(max_compl))
    ref_point = (max(ref[0]), max(ref[1]))

    hv_set = []
    for exp_num, exp_history in enumerate(fitness_history_gp):
        hv_set.append([])
        for run_num, run_history in enumerate(exp_history):
            hv_set[exp_num].append(
                [hypervolume(pop).compute(ref_point) for pop in fitness_history_gp[exp_num][run_num]])

    show_history_optimization_comparison(optimisers_fitness_history=hv_set,
                                         iterations=[_ for _ in range(iterations)],
                                         labels=labels, color_pallete=color_pallete, ylabel='Hypervolume',
                                         name_of_dataset=name_of_dataset)

    try:
        path_to_save_hv = name_of_dataset + '_hv_set_gp'
        np.save(path_to_save_hv, hv_set)
    except Exception as ex:
        print(ex)