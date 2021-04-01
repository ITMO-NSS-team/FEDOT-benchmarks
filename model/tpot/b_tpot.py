import os

import joblib

from benchmark_utils import get_models_hyperparameters

from fedot.core.data.data import InputData
from fedot.core.models.evaluation.automl_eval import fit_tpot, predict_tpot_class, predict_tpot_reg
from fedot.core.repository.tasks import Task, TaskTypesEnum


def run_tpot(train_file_path,
             test_file_path,
             case_label,
             task,
             time_for_exp):
    models_hyperparameters = get_models_hyperparameters()['TPOT']
    generations = models_hyperparameters['GENERATIONS']
    population_size = models_hyperparameters['POPULATION_SIZE']
    time_to_eval = time_for_exp

    result_model_filename = f'{case_label}_g{generations}' \
                            f'_p{population_size}_{task.task_type.name}.pkl'
    current_file_path = str(os.path.dirname(__file__))
    result_file_path = os.path.join(current_file_path, result_model_filename)

    train_data = InputData.from_csv(train_file_path, task=Task(task))

    if result_model_filename not in os.listdir(current_file_path):
        # TODO change hyperparameters to actual from variable
        model = fit_tpot(train_data, time_to_eval)

        model.export(output_file_name=f'{result_model_filename[:-4]}_pipeline.py')

        # sklearn pipeline object
        fitted_model_config = model.fitted_pipeline_
        joblib.dump(fitted_model_config, result_file_path, compress=1)

    imported_model = joblib.load(result_file_path)

    predict_data = InputData.from_csv(test_file_path, task=Task(task))
    true_target = predict_data.target
    if task.task_type == TaskTypesEnum.regression:
        predicted = predict_tpot_reg(imported_model, predict_data)
        predicted_labels = predicted
    elif task.task_type == TaskTypesEnum.classification:
        predicted, predicted_labels = predict_tpot_class(imported_model, predict_data)
    else:
        print('Incorrect type of ml task')
        raise NotImplementedError()

    print(f'BEST_model: {imported_model}')

    return true_target, predicted, predicted_labels
