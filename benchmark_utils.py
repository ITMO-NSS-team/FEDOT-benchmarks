import gc
import json
import os
from typing import Tuple

import pandas as pd
from pmlb import fetch_data
from pathlib import Path

from fedot.core.utils import ensure_directory_exists, get_split_data_paths, \
    save_file_to_csv, split_data


def project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent


def ensure_directory_exists(dir_names: list):
    main_dir = os.path.join(str(project_root()), dir_names[0], dir_names[1])
    dataset_dir = os.path.join(str(project_root()), dir_names[0], dir_names[1], dir_names[2])
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)


def get_split_data_paths(directory_names: list):
    train_file_path = os.path.join(directory_names[0], directory_names[1], directory_names[2], 'train.csv')
    full_train_file_path = os.path.join(str(project_root()), train_file_path)
    test_file_path = os.path.join(directory_names[0], directory_names[1], directory_names[2], 'test.csv')
    full_test_file_path = os.path.join(str(project_root()), test_file_path)
    return full_train_file_path, full_test_file_path


def get_scoring_case_data_paths() -> Tuple[str, str]:
    train_file_path = os.path.join('test_cases', 'scoring', 'data', 'scoring_train.csv')
    test_file_path = os.path.join('test_cases', 'scoring', 'data', 'scoring_test.csv')
    full_train_file_path = os.path.join(str(project_root()), train_file_path)
    full_test_file_path = os.path.join(str(project_root()), test_file_path)

    return full_train_file_path, full_test_file_path


def get_cancer_case_data_paths() -> Tuple[str, str]:
    train_file_path = os.path.join('test_cases', 'cancer', 'data', 'cancer_train.csv')
    test_file_path = os.path.join('test_cases', 'cancer', 'data', 'cancer_test.csv')
    full_train_file_path = os.path.join(str(project_root()), train_file_path)
    full_test_file_path = os.path.join(str(project_root()), test_file_path)

    return full_train_file_path, full_test_file_path


def get_penn_case_data_paths(name_of_dataset: str) -> Tuple[str, str]:
    df = fetch_data(name_of_dataset)
    directory_names = ['test_cases', 'penn_ml', 'data', name_of_dataset]
    penn_train, penn_test = split_data(df, 0.5)
    ensure_directory_exists(directory_names)
    full_train_file_path, full_test_file_path = get_split_data_paths(directory_names)
    save_file_to_csv(penn_train, full_train_file_path)
    save_file_to_csv(penn_test, full_test_file_path)
    return full_train_file_path, full_test_file_path


def convert_json_stats_to_csv(dataset: list, include_hyper: bool = True):
    list_of_df = []
    new_col = []
    dataset_name_column_place = 1
    for name_of_dataset in dataset:
        filename = f'penn_ml_metrics_for_{name_of_dataset}.json'
        with open(filename, 'r') as f:
            data = json.load(f)
            df = pd.json_normalize(data)
            df.insert(dataset_name_column_place, 'name_of_dataset', name_of_dataset, True)
            list_of_df.append(df)

    df_final = pd.concat(list_of_df)

    for column_name in df_final.columns:
        if 'hyper' not in column_name:
            new_col.append(column_name)

    if include_hyper:
        df_final = df_final[new_col]

    pd.DataFrame.to_csv(df_final, './final_combined.csv', sep=',', index=False)
    return df_final


def save_metrics_result_file(data: dict, file_name: str):
    with open(f'{file_name}.json', 'w') as file:
        json.dump(data, file, indent=4)


def get_models_hyperparameters(timedelta: int = 10) -> dict:
    # MAX_RUNTIME_MINS should be equivalent to MAX_RUNTIME_SECS

    tpot_config = {'MAX_RUNTIME_MINS': timedelta,
                   'GENERATIONS': 200,
                   'POPULATION_SIZE': 20
                   }

    fedot_config = {'MAX_RUNTIME_MINS': timedelta,
                    'GENERATIONS': 200,
                    'POPULATION_SIZE': 20
                    }

    h2o_config = {'MAX_MODELS': 20,
                  'MAX_RUNTIME_SECS': timedelta * 60}

    autokeras_config = {'MAX_TRIAL': 10,
                        'EPOCH': 100}

    space_for_mlbox = {

        'ne__numerical_strategy': {"space": [0, 'mean']},

        'ce__strategy': {"space": ["label_encoding", "random_projection", "entity_embedding"]},

        'fs__strategy': {"space": ["variance", "rf_feature_importance"]},
        'fs__threshold': {"search": "choice", "space": [0.1, 0.2, 0.3, 0.4, 0.5]},

        'est__strategy': {"space": ["LightGBM"]},
        'est__max_depth': {"search": "choice", "space": [5, 6]},
        'est__subsample': {"search": "uniform", "space": [0.6, 0.9]},
        'est__learning_rate': {"search": "choice", "space": [0.07]}

    }

    mlbox_config = {'space': space_for_mlbox, 'max_evals': 20}

    config_dictionary = {'TPOT': tpot_config, 'FEDOT': fedot_config, 'H2O': h2o_config,
                         'autokeras': autokeras_config, 'MLBox': mlbox_config}
    gc.collect()

    return config_dictionary


def get_target_name(file_path: str) -> str:
    print('Make sure that your dataset target column is the last one')
    dataframe = pd.read_csv(file_path)
    column_names = dataframe.columns()
    target_name = column_names[-1]

    return target_name


def get_h2o_connect_config():
    IP = '127.0.0.1'
    PORT = 8888
    return IP, PORT
