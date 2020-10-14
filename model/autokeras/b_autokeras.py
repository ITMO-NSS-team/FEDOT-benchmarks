import autokeras as ak

from benchmark_utils import get_models_hyperparameters
from FEDOT.core.models.data import InputData
from FEDOT.core.repository.tasks import TaskTypesEnum


def run_autokeras(params: 'ExecutionParams'):
    train_file_path = params.train_file
    test_file_path = params.test_file
    task = params.task

    config_data = get_models_hyperparameters()['autokeras']
    max_trial = config_data['MAX_TRIAL']
    epoch = config_data['EPOCH']

    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    # TODO Save model to file

    if task == TaskTypesEnum.classification:
        estimator = ak.StructuredDataClassifier
    else:
        estimator = ak.StructuredDataRegressor

    model = estimator(max_trials=max_trial)

    model.fit(train_data.features, train_data.target, epochs=epoch)

    predicted = model.predict(test_data.features)

    return test_data.target, predicted
