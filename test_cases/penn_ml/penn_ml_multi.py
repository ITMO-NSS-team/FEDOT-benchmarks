from pmlb import classification_dataset_names, fetch_data, regression_dataset_names
from pmlb.update_dataset_files import compute_imbalance
from benchmark_utils import get_penn_case_data_paths
from fedot.core.repository.tasks import TaskTypesEnum
import experiments.multi_objective_schemes_experiment as multi_model


def _problem_and_metric_for_dataset(name_of_dataset: str, num_classes: int):
    if num_classes == 2 and name_of_dataset in classification_dataset_names:
        return TaskTypesEnum.classification, ['roc_auc', 'f1']
    elif num_classes > 2 and name_of_dataset in classification_dataset_names:
        return TaskTypesEnum.classification, ['balanced_accuracy']
    elif name_of_dataset in regression_dataset_names:
        return TaskTypesEnum.regression, ['mse', 'r2']
    else:
        return None, None


if __name__ == '__main__':
    dataset = ['dis', 'churn', 'Hill_Valley_without_noise']
    number_of_experiment = 1
    for name_of_dataset in dataset:
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

        if number_of_experiment == 1:
            model = multi_model.exp_single_vs_multi_objective(train_path=train_file,
                                                              test_path=test_file,
                                                              name_of_dataset=name_of_dataset)
        elif number_of_experiment == 2:
            model = multi_model.exp_self_config_vs_fix_params(train_path=train_file,
                                                              test_path=test_file,
                                                              name_of_dataset=name_of_dataset)
        elif number_of_experiment == 3:
            model = multi_model.exp_multi_obj_selections(train_path=train_file,
                                                         test_path=test_file,
                                                         name_of_dataset=name_of_dataset)
        else:
            model = multi_model.exp_complexity_metrics(train_path=train_file,
                                                       test_path=test_file,
                                                       name_of_dataset=name_of_dataset)
