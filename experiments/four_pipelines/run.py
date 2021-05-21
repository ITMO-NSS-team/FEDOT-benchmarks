import pandas as pd
import numpy as np
from benchmark_utils import get_penn_case_data_paths

from experiments.four_pipelines.utils import create_folder, _problem_and_metric_for_dataset, Fedot_model, \
    tpot_model, baseline_model, calculate_metrics, create_report_dataframe, linear_pipeline_model

import warnings
warnings.filterwarnings("ignore")

dataset_dict = {'regression': ['227_cpu_small',
                               '228_elusage',
                               '1096_FacultySalaries',
                               '605_fri_c2_250_25',
                               '1027_ESL'],

                'classification': ['magic',
                                   'labor',
                                   'flare',
                                   'ionosphere',
                                   'spect']}


class Experiment_runner():
    def __init__(self,
                 problem_type: str,
                 number_of_experiments: int,
                 time_for_experiment: int,
                 report_flag: bool = False,
                 frameworks: list = None,
                 datasets: list = None):

        self.problem_type = problem_type

        if datasets is None:
            self.datasets = dataset_dict[self.problem_type]
        else:
            self.datasets = datasets

        if frameworks is None:
            self.frameworks = ['FEDOT', 'TPOT', 'Baseline', 'Mlbox']
        else:
            self.frameworks = frameworks

        self.task, self.metrics = _problem_and_metric_for_dataset(self.datasets[0])

        self.report_flag = report_flag
        self.number_of_experiments = number_of_experiments
        self.time_for_experiment = time_for_experiment

    def load_data(self, dataset: str):
        if not self.task or not self.metrics:
            print(f'Incorrect dataset: {dataset}')
        else:
            print('Running framework{} for dataset -{}'.format(frameworks, dataset))
            train_file, test_file = get_penn_case_data_paths(dataset)

            return train_file, test_file

    def run_frameworks(self):
        experiments_results = []
        for framework in self.frameworks:
            for dataset in self.datasets:
                train_file, test_file = self.load_data(dataset)
                for run in range(self.number_of_experiments):
                    print('Experiment number - {}'.format(str(run)))
                    if framework == 'FEDOT':
                        predicted_labels, metric_list, baseline_fedot_model = Fedot_model(train_path=train_file,
                                                                                    test_path=test_file,
                                                                                    task=self.problem_type,
                                                                                    metric_names=self.metrics,
                                                                                    time_for_exp=self.time_for_experiment)
                        path =create_folder(run, framework, dataset, self.time_for_experiment, metric_list, baseline_fedot_model)
                        results = create_report_dataframe(path, dataset, framework)
                    elif framework == 'TPOT':
                        true_target, predicted, predicted_labels = tpot_model(train_path=train_file,
                                                                              test_path=test_file,
                                                                              name_of_dataset=dataset,
                                                                              task=self.task,
                                                                              time_for_exp=self.time_for_experiment,
                                                                              experiment_number=run)
                        metric_list = calculate_metrics(self.metrics, true_target, predicted, predicted_labels)
                        path = create_folder(run, framework, dataset, self.time_for_experiment, metric_list)
                        results = create_report_dataframe(path, dataset, framework)
                    elif framework == 'Baseline':
                        true_target, predicted, predicted_labels = baseline_model(train_path=train_file,
                                                                                  test_path=test_file,
                                                                                  task=self.task)
                        metric_list = calculate_metrics(self.metrics, true_target, predicted, predicted_labels)
                        path = create_folder(run, framework, dataset, self.time_for_experiment, metric_list)
                        results = create_report_dataframe(path, dataset, framework)
                    else:
                        true_target, predicted = linear_pipeline_model(train_path=train_file,
                                                                       test_path=test_file,
                                                                       name_of_dataset=dataset,
                                                                       task=self.task)
                        metric_list = calculate_metrics(self.metrics, true_target, predicted.values, convert_flag=True)
                        path = create_folder(run, framework, dataset, self.time_for_experiment, metric_list)
                        results = create_report_dataframe(path, dataset, framework)
                    experiments_results.append(results)

        if not self.report_flag:
            return
        else:
            df_all = pd.concat(experiments_results)
            self.save_experiment_results(df_all)
            return

    def save_experiment_results(self, df_all_results: pd.DataFrame):
        df = df_all_results.pivot_table(
            values='metrics',
            index=['name_of_metric', 'type_of_pipeline'],
            columns='name_of_experiment',
            aggfunc=np.mean)
        df.to_csv(f'./{self.problem_type}_pivot_table.csv')
        return


if __name__ == '__main__':
    frameworks = ['FEDOT', 'TPOT', 'Baseline', 'Mlbox']
    runner = Experiment_runner(problem_type='classification',
                               time_for_experiment=10,
                               number_of_experiments=3,
                               frameworks=frameworks,
                               report_flag=True)
    runner.run_frameworks()
