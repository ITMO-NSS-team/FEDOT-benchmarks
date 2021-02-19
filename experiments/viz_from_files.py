from test_cases.penn_ml.utils import PMLB_report
iterations=30
#labels = ['steady-state with depth config', 'steady-state', 'parameter-free with depth config', 'parameter-free']
labels = ['parameter-free', 'parameter-free with depth config', 'steady-state',
              'steady-state with depth config']
name_of_dataset = 'dis'
report = PMLB_report(labels=labels, runs=3, datasets=['dis'])
#all_history_report = report.viz_hv(iterations=iterations, name_of_dataset=name_of_dataset)
pareto_run_numbers =(2,0,2,2) #chosing the best frontier for each dataset
report.viz_pareto_comparison(pareto_run_numbers, name_of_dataset)
#report.viz_best_quality_comparison(iterations=iterations, name_of_dataset=name_of_dataset)
all_history_report = report.get_experiment_report()
print("g")