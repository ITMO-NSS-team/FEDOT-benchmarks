from pmlb import classification_dataset_names, fetch_data, regression_dataset_names
from pmlb.update_dataset_files import compute_imbalance
from benchmark_utils import get_penn_case_data_paths
from fedot.core.repository.tasks import TaskTypesEnum, Task
from experiments.four_pipelines.utils import create_folder, _problem_and_metric_for_dataset, Fedot_model,\
    tpot_model, baseline_model, calculate_metrics

dataset_dict = {'regression': ['227_cpu_small',
                               '228_elusage',
                               '605_fri_c2_250_25',
                               '606_fri_c2_1000_10',
                               '1096_FacultySalaries'],
                'classification': [#'magic',
                                   #'labor',
                                   #'vowel',
                                   #'flare',
                                   'ionosphere']}

if __name__ == '__main__':
    num_of_experiments = 3
    time_for_exp = 10
    task_type = 'classification'
    #task_type = 'regression'
    name_of_method = 'FEDOT'
    datasets = dataset_dict[task_type]
    for name_of_dataset in datasets:
        print('Cейчас идет эксперимент над датасетом-' + str(name_of_dataset))
        if name_of_dataset in classification_dataset_names:
            task = Task(TaskTypesEnum.classification)
        elif name_of_dataset in regression_dataset_names:
            task = Task(TaskTypesEnum.regression)
        else:
            raise ValueError('Selected dataset is not classification or regression problem')
        try:
            pmlb_data = fetch_data(name_of_dataset)
            imbalance_report = compute_imbalance(pmlb_data['target'].values.tolist())
            num_classes = imbalance_report[0]
            problem_class, metric_names = _problem_and_metric_for_dataset(name_of_dataset, num_classes)
        except ValueError as ex:
            print(ex)
            continue
        if not problem_class or not metric_names:
            print(f'Incorrect dataset: {name_of_dataset}')
            continue
        train_file, test_file = get_penn_case_data_paths(name_of_dataset)

        for run in range(num_of_experiments):
            print('Идет эксперимент номер - ' + str(run))
            if name_of_method == 'FEDOT':
                predicted_labels, metric_list, baseline_model = Fedot_model(train_path=train_file,
                                                                            test_path=test_file,
                                                                            task=task_type,
                                                                            metric_names=metric_names,
                                                                            time_for_exp=time_for_exp)
                create_folder(run, name_of_method, name_of_dataset, time_for_exp, metric_list, baseline_model)
            elif name_of_method == 'TPOT':
                true_target, predicted, predicted_labels = tpot_model(train_path=train_file,
                                                                      test_path=test_file,
                                                                      name_of_dataset=name_of_dataset,
                                                                      task=task,
                                                                      time_for_exp=time_for_exp)
                metric_list = calculate_metrics(metric_names, true_target, predicted, predicted_labels)
                create_folder(run, name_of_method, name_of_dataset, time_for_exp, metric_list)
            elif name_of_method == 'Baseline':
                true_target, predicted, predicted_labels = tpot_model(train_path=train_file,
                                                                      test_path=test_file,
                                                                      name_of_dataset=name_of_dataset,
                                                                      task=task,
                                                                      time_for_exp=time_for_exp)
                metric_list = calculate_metrics(metric_names, true_target, predicted, predicted_labels)
                create_folder(run, name_of_method, name_of_dataset, predicted_labels, metric_list)