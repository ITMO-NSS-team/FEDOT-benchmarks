from fedot.core.repository.tasks import TaskTypesEnum, Task

from test_cases.penn_ml.utils import PMLB_report

# number of iterations
iterations = 30

# exp labels chosing
# labels = ['steady-state with depth config', 'steady-state', 'parameter-free with depth config', 'parameter-free']
# labels = ['steady_state single-obj', 'steady_state single-obj penalty', 'steady-state multi-obj']
labels = ['nsga selection', 'spea2 selection']

name_of_dataset = 'dis'

# task type choosing
# task = Task(TaskTypesEnum.regression)
task = Task(TaskTypesEnum.classification)

report = PMLB_report(labels=labels, runs=4, datasets=[name_of_dataset], task=task)

# Hypervolume variability visualize
chart_labels = ('GPComp@Free with NSGA2 selection', 'GPComp@Free with SPEA2 selection')
report.viz_hv(iterations=iterations, name_of_dataset=name_of_dataset, labels=chart_labels)

# pareto frontiers comparison visualization
pareto_run_numbers = [0, 2]  # chosing the best frontier for each dataset
objectives_names = ('ROC-AUC metric', 'Relative computation time, %')
report.viz_pareto_comparison(pareto_run_numbers=pareto_run_numbers, name_of_dataset=name_of_dataset,
                             objectives_names=objectives_names,
                             labels=chart_labels)

# quality variability visualize
xy_labels = ('Generation, #', 'Best ROC-AUC')
report.viz_best_quality_comparison(iterations=iterations, name_of_dataset=name_of_dataset,
                                   xy_labels=xy_labels)

# history report
# all_history_report = report.get_experiment_report()
