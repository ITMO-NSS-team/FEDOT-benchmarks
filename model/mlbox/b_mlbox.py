import os

import pandas as pd
from mlbox.optimisation import Optimiser
from mlbox.prediction import Predictor
from mlbox.preprocessing import Reader, Drift_thresholder
from benchmark_utils import get_models_hyperparameters
from fedot.core.repository.tasks import Task, TaskTypesEnum
import warnings
warnings.filterwarnings("ignore")


def separate_target_column(file_path: str, target_name: str, name_of_dataset: str):
    df = pd.read_csv(file_path)
    target = df[target_name].values

    df = df.drop([target_name], axis=1)

    path_to_file, _ = os.path.split(file_path)
    new_filename = name_of_dataset + '.csv'
    new_file_path = os.path.join(path_to_file, new_filename)

    df.to_csv(new_file_path, index=False)

    return new_file_path, target


def run_mlbox(train_file_path: str,
              test_file_path: str,
              case_label: str,
              task: str,
              target_name: str = 'target'):
    config_data = get_models_hyperparameters()['MLBox']
    new_test_file_path, true_target = separate_target_column(test_file_path, target_name, case_label)
    paths = [train_file_path, new_test_file_path]
    cur_work_dir = os.path.abspath(os.curdir)
    save_path_model = os.path.join(cur_work_dir, f'save\{case_label}_model')
    # save_path_preds = os.path.join(cur_work_dir, f'save\{case_label}_preds_and_features')
    load_path = os.path.join(cur_work_dir, f'save\{case_label}_model\{target_name}_predictions.csv')

    reader = Reader(sep=",", to_path=save_path_model)
    data = reader.train_test_split(paths, target_name)
    data = Drift_thresholder(to_path=save_path_model).fit_transform(data)

    score = 'roc_auc' if task.name == 'classification' else 'neg_mean_squared_error'
    target_col = '1.0' if task.name == 'classification' else 'target_predicted'

    opt = Optimiser(scoring=score, n_folds=5, to_path=save_path_model)
    params = opt.optimise(config_data['space'], data, max_evals=config_data['max_evals'])
    opt.evaluate(params, data)

    Predictor(to_path=save_path_model, verbose=False).fit_predict(params, data)

    predicted_df = pd.read_csv(load_path)
    predicted = predicted_df[target_col]

    os.remove(new_test_file_path)

    return true_target, predicted
