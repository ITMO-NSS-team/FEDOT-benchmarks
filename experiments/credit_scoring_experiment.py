import datetime
import random
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.chains.chain import Chain
from fedot.core.composer.gp_composer.gp_composer import GPComposerBuilder, GPComposerRequirements
from fedot.core.composer.optimisers.crossover import CrossoverTypesEnum
from fedot.core.composer.optimisers.gp_optimiser import GPChainOptimiserParameters
from fedot.core.composer.optimisers.mutation import MutationTypesEnum
from fedot.core.composer.optimisers.regularization import RegularizationTypesEnum
from fedot.core.composer.optimisers.selection import SelectionTypesEnum
from fedot.core.data.data import InputData
from fedot.core.repository.model_types_repository import ModelTypesRepository
from fedot.core.repository.quality_metrics_repository import \
    (ClassificationMetricsEnum, MetricsRepository)
from fedot.core.repository.tasks import TaskTypesEnum, Task

random.seed(1)
np.random.seed(1)


def calculate_validation_metric(chain: Chain, dataset_to_validate: InputData) -> float:
    # the execution of the obtained composite models
    predicted = chain.predict(dataset_to_validate)
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict)
    return roc_auc_value


def run_credit_scoring_problem(train_file_path, test_file_path,
                               max_lead_time: datetime.timedelta = datetime.timedelta(minutes=5),
                               gp_optimiser_params: Optional[GPChainOptimiserParameters] = None, pop_size=None,
                               generations=None, max_depth=3, metrics=ClassificationMetricsEnum.ROCAUC_penalty,
                               start_depth: int = 3):
    task = Task(TaskTypesEnum.classification)
    dataset_to_compose = InputData.from_csv(train_file_path, task=task)
    dataset_to_validate = InputData.from_csv(test_file_path, task=task)

    available_model_types, _ = ModelTypesRepository().suitable_model(task_type=task.task_type)
    available_model_types.remove('pca_data_model')
    # the choice of the metric for the chain quality assessment during composition
    metric_function = metrics

    if gp_optimiser_params:
        optimiser_parameters = gp_optimiser_params
    else:
        selection_types = [SelectionTypesEnum.tournament]
        crossover_types = [CrossoverTypesEnum.subtree]
        mutation_types = [MutationTypesEnum.simple, MutationTypesEnum.growth, MutationTypesEnum.reduce]
        regularization_type = RegularizationTypesEnum.decremental
        optimiser_parameters = GPChainOptimiserParameters(selection_types=selection_types,
                                                          crossover_types=crossover_types,
                                                          mutation_types=mutation_types,
                                                          regularization_type=regularization_type)

    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=4,
        max_depth=max_depth, pop_size=pop_size, num_of_generations=generations,
        crossover_prob=0.8, mutation_prob=0.8, max_lead_time=max_lead_time, add_single_model_chains=False,
        start_depth=start_depth, model_fit_time_constraint=200)

    # Create GP-based composer
    builder = GPComposerBuilder(task=task).with_requirements(composer_requirements).with_metrics(
        metric_function).with_optimiser_parameters(optimiser_parameters)

    composer = builder.build()

    chains_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                 is_visualise=False)

    chains_roc_auc = []
    chains_complexity = []
    if type(metrics) is list:
        if len(metrics) > 1:
            for chain_evo_composed in chains_evo_composed:
                chain_evo_composed.fit(input_data=dataset_to_compose, verbose=True)
                roc_on_valid_evo_composed = calculate_validation_metric(chain_evo_composed,
                                                                        dataset_to_validate)
                complexity_metric = MetricsRepository().metric_by_id(metrics[1])(chain_evo_composed)

                chains_roc_auc.append(roc_on_valid_evo_composed)
                chains_complexity.append(complexity_metric)

        calculated_metrics = [chains_roc_auc, chains_complexity]
    else:
        chains_evo_composed.fit(input_data=dataset_to_compose, verbose=True)

        calculated_metrics = [calculate_validation_metric(chains_evo_composed, dataset_to_validate)]
        chains_evo_composed = [chains_evo_composed]
    return calculated_metrics, chains_evo_composed, composer
