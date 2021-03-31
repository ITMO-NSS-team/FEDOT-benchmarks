import xgboost as xgb

from fedot.core.data.data import InputData
from fedot.core.repository.tasks import TaskTypesEnum


def run_xgboost(train_file_path,
             test_file_path,
             task):
    train_file_path = train_file_path
    test_file_path = test_file_path
    task = task

    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    if task.task_type == TaskTypesEnum.classification:
        model = xgb.XGBClassifier(max_depth=2, learning_rate=1.0, objective='binary:logistic')
        model.fit(train_data.features, train_data.target)
        predicted = model.predict_proba(test_data.features)[:, 1]
        predicted_labels = model.predict(test_data.features)

    elif task.task_type == TaskTypesEnum.regression:
        xgbr = xgb.XGBRegressor(max_depth=3, learning_rate=0.3, n_estimators=300,
                                objective='reg:squarederror')
        xgbr.fit(train_data.features, train_data.target)
        predicted = xgbr.predict(test_data.features)
        predicted_labels = None

    else:
        raise NotImplementedError()
    return test_data.target, predicted, predicted_labels
