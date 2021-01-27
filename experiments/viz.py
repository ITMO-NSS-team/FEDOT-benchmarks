import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from itertools import cycle
from typing import List


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
    plt.show()
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
    plt.show()
    if save:
        if not os.path.isdir('../../tmp'):
            os.mkdir('../../tmp')

        file_name = name_of_dataset + '_pareto_fronts_comp.png'
        path = f'../../tmp/{file_name}'
        plt.savefig(path, bbox_inches='tight')

    plt.show()


def show_history_optimization_comparison(optimisers_fitness_history: List[List[int]], iterations: List[int],
                                         labels: List[str], color_pallete: str = 'bgrcmykw', xlabel='Generation, #',
                                         ylabel='Best fitness',name_of_dataset=None, save=True):
    color_to_take = cycle(color_pallete)
    plt.yticks(fontsize=12)
    for fitness_history in optimisers_fitness_history:
        sns.tsplot(fitness_history, iterations, legend=True, color=next(color_to_take))

    plt.xticks(range(0, max(iterations)), fontsize=13)
    plt.legend(labels=labels, loc='lower right', fontsize=12)
    plt.ylabel(ylabel, fontsize=13)
    plt.xlabel(xlabel, fontsize=13)
    plt.tight_layout()
    plt.show()
    if save:
        if not os.path.isdir('../../tmp'):
            os.mkdir('../../tmp')

        file_name = name_of_dataset+'_pareto_fronts_comp.png'
        path = f'../../tmp/{file_name}'
        plt.savefig(path, bbox_inches='tight')

    plt.show()


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
    plt.show()
    if save:
        if not os.path.isdir('../../tmp'):
            os.mkdir('../../tmp')

        file_name = name_of_dataset + '_pareto_fronts_comp.png'
        path = f'../../tmp/{file_name}'
        plt.savefig(path, bbox_inches='tight')

    plt.show()
