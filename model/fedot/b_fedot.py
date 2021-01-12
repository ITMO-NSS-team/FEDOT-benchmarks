import datetime
import os
import random
from pickle import dump, load

import numpy as np
from fedot.core.composer.gp_composer.gp_composer import GPComposerBuilder, GPComposerRequirements
from fedot.core.composer.visualisation import ComposerVisualiser
from fedot.core.data.data import InputData
from fedot.core.repository.model_types_repository import ModelTypesRepository
from fedot.core.repository.quality_metrics_repository import \
    (ClassificationMetricsEnum,
     MetricsRepository,
     RegressionMetricsEnum)
from fedot.core.repository.tasks import Task, TaskTypesEnum

from benchmark_utils import get_models_hyperparameters

random.seed(1)
np.random.seed(1)


def save_fedot_model(chain, file_name: str):
    path_to_file = str(os.path.dirname(__file__))
    with open(f'{path_to_file}/{file_name}.pkl', 'wb') as pickle_file:
        dump(chain, pickle_file)
    ComposerVisualiser.visualise(chain, f'{path_to_file}/{file_name}.png')


def load_fedot_model(file_name):
    path_to_file = str(os.path.dirname(__file__))
    try:
        if os.path.exists(f'{path_to_file}/{file_name}.pkl'):
            with open(f'{path_to_file}/{file_name}.pkl', 'rb') as pickle_file:
                return load(pickle_file)
    except Exception as ex:
        print(f'Model load error {ex}')
    return None


def run_fedot(params: 'ExecutionParams'):
    train_file_path = params.train_file
    test_file_path = params.test_file
    case_label = params.case_label
    task_type = params.task

    if task_type == TaskTypesEnum.classification:
        metric = ClassificationMetricsEnum.ROCAUC
    elif task_type == TaskTypesEnum.regression:
        metric = RegressionMetricsEnum.RMSE
    else:
        raise NotImplementedError()

    metric_func = MetricsRepository().metric_by_id(metric)

    task = Task(task_type)
    dataset_to_compose = InputData.from_csv(train_file_path, task=task)
    dataset_to_validate = InputData.from_csv(test_file_path, task=task)

    models_hyperparameters = get_models_hyperparameters()['FEDOT']
    cur_lead_time = models_hyperparameters['MAX_RUNTIME_MINS']

    saved_model_name = f'fedot_{case_label}_{task_type.name}_{cur_lead_time}_{metric.name}'
    loaded_model = load_fedot_model(saved_model_name)

    if not loaded_model:
        generations = models_hyperparameters['GENERATIONS']
        population_size = models_hyperparameters['POPULATION_SIZE']

        # the search of the models provided by the framework that can be used as nodes in a chain'
        models_repo = ModelTypesRepository()
        available_model_types, _ = models_repo.suitable_model(task.task_type)

        heavy_models = ['svc', 'multinb', 'tfidf', 'qda']
        available_model_types = [model for model in available_model_types if model not in heavy_models]

        # the choice and initialisation of the GP search
        composer_requirements = GPComposerRequirements(
            primary=available_model_types,
            secondary=available_model_types, max_arity=3,
            max_depth=2, pop_size=population_size, num_of_generations=generations,
            crossover_prob=0.8, mutation_prob=0.8, max_lead_time=datetime.timedelta(minutes=cur_lead_time),
            add_single_model_chains=True)

        # Create GP-based composer
        builder = GPComposerBuilder(task).with_requirements(composer_requirements).with_metrics(metric_func)
        gp_composer = builder.build()

        chain_gp_composed = gp_composer.compose_chain(data=dataset_to_compose)

        chain_gp_composed.fit_from_scratch(input_data=dataset_to_compose)
        save_fedot_model(chain_gp_composed, saved_model_name)
    else:
        chain_gp_composed = loaded_model

    evo_predicted = chain_gp_composed.predict(dataset_to_validate)
    evo_predicted_labels = chain_gp_composed.predict(dataset_to_validate, output_mode='labels')

    return dataset_to_validate.target, evo_predicted.predict, evo_predicted_labels.predict
