import os
import numpy as np
import pandas as pd
import glob
from fedot.api.main import Fedot
from pmlb import classification_dataset_names, regression_dataset_names
from benchmark_utils import save_metrics_result_file
from fedot.core.repository.tasks import TaskTypesEnum

from model.mlbox.b_mlbox import run_mlbox
from model.tpot.b_tpot import run_tpot
from baseline.b_xgboost import run_xgboost
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score, balanced_accuracy_score, mean_absolute_error


def _problem_and_metric_for_dataset(name_of_dataset: str):
    if name_of_dataset in classification_dataset_names:
        return TaskTypesEnum.classification, ['roc_auc', 'f1']
    elif name_of_dataset in regression_dataset_names:
        return TaskTypesEnum.regression, ['rmse', 'mae']
    else:
        return None, None


def calculate_metrics(metric_list: list, target: list, predicted_probs: list, predicted_labels: list = None,
                      convert_flag: bool = False):
    metric_dict = {'roc_auc': roc_auc_score,
                   'f1': f1_score,
                   'rmse': mean_squared_error,
                   'mae': mean_absolute_error,
                   'balanced_accuracy': balanced_accuracy_score
                   }
    label_only_metrics = ['f1', 'accuracy', 'precision', 'balanced_accuracy']
    result_metric = []
    for metric_name in metric_list:
        predicted = predicted_probs
        if metric_name in label_only_metrics:
            predicted_labels = predicted_probs
            predicted = predicted_labels
            convert_flag = True

        if metric_name == 'rmse':
            result_metric.append(round(metric_dict[metric_name](target, predicted, squared=False), 3))
        else:
            if convert_flag:
                _ = [round(x) for x in predicted]
                predicted = _
            result_metric.append(round(metric_dict[metric_name](target, predicted), 3))

    result_dict = dict(zip(metric_list, result_metric))

    return result_dict


def create_folder(run, name_of_method, name_of_dataset, time_for_exp, calculated_metrics, model=None):
    try:
        tmp_folder = str(run + 1) + '_experiment'
        experiment_path = f'./{name_of_method}/{name_of_dataset}/{tmp_folder}'
        if not os.path.isdir(experiment_path):
            os.makedirs(experiment_path)
        name_of_experiment = name_of_dataset + '_run_number_' + str(run + 1)
        save_model_history(experiment_path, name_of_experiment, time_for_exp, calculated_metrics, model)
    except Exception as ex:
        print(ex)

    return experiment_path


def save_model_history(experiment_path: str,
                       name_of_experiment: str,
                       time: int,
                       metric_list: dict,
                       model=None,
                       ):
    metric_save_path = os.path.join(str(experiment_path), name_of_experiment + '_best_metric')
    model_save_path = os.path.join(str(experiment_path), name_of_experiment + '_best_model')
    result_dict = {'name_of_experiment': name_of_experiment,
                   'time': time,
                   'metrics': metric_list}
    save_metrics_result_file(result_dict, metric_save_path)
    if model is not None:
        np.save(model_save_path, model, allow_pickle=True)
    return


def create_report_dataframe(experiment_path,name_of_dataset,name_of_pipeline):
    tmp_lst = []
    path = f'{experiment_path}/*.json'.format(experiment_path)
    for f in glob.glob(path, recursive=True):
        df = pd.read_json(f, orient='records')
        df['name_of_experiment'] = name_of_dataset
        df = df.reset_index()
        df.rename(columns={'index': 'name_of_metric'}, inplace=True)
        tmp_lst.append(df)
    df_all = pd.concat(tmp_lst)
    df_all = df_all.groupby(by=['name_of_metric']).mean()
    df_all = df_all.reset_index()
    df_all.rename(columns={'index': 'name_of_metric'}, inplace=True)
    df_all['name_of_experiment'] = name_of_dataset
    df_all['type_of_pipeline'] = name_of_pipeline
    return df_all


def Fedot_model(train_path,
                test_path,
                task,
                metric_names,
                time_for_exp):
    params = {'max_depth': 6,
              'max_arity': 3}

    baseline_model = Fedot(problem=task, learning_time=time_for_exp, preset='without_knn', composer_params=params)
    baseline_model.fit(features=train_path, target='target')
    predicted_labels = baseline_model.predict(features=test_path, save_predictions=True)
    predicted = baseline_model.get_metrics(metric_names=metric_names)

    return predicted_labels, predicted, baseline_model


def tpot_model(train_path,
               test_path,
               name_of_dataset,
               task,
               time_for_exp,
               experiment_number):
    true_target, predicted, predicted_labels = run_tpot(train_path,
                                                        test_path,
                                                        name_of_dataset,
                                                        task,
                                                        time_for_exp,
                                                        experiment_number)

    return true_target, predicted, predicted_labels


def linear_pipeline_model(train_path,
                          test_path,
                          name_of_dataset,
                          task):
    true_target, predicted = run_mlbox(train_path,
                                       test_path,
                                       name_of_dataset,
                                       task)

    return true_target, predicted


def baseline_model(train_path,
                   test_path,
                   task):
    target, predicted, predicted_labels = run_xgboost(train_path,
                                                      test_path,
                                                      task)

    return target, predicted, predicted_labels
