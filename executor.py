from dataclasses import dataclass
from typing import List

from sklearn.metrics import f1_score, mean_squared_error, r2_score, roc_auc_score, balanced_accuracy_score

from model.H2O.b_h2o import run_h2o
from model.autokeras.b_autokeras import run_autokeras
from baseline.b_xgboost import run_xgboost
from benchmark_model_types import BenchmarkModelTypesEnum
from model.fedot.b_fedot import run_fedot
from model.tpot.b_tpot import run_tpot
from fedot.core.repository.tasks import TaskTypesEnum


def calculate_metrics(metric_list: list, target: list, predicted_probs: list, predicted_labels: list):
    metric_dict = {'roc_auc': roc_auc_score,
                   'f1': f1_score,
                   'mse': mean_squared_error,
                   'r2': r2_score,
                   'balanced_accuracy': balanced_accuracy_score
                   }
    label_only_metrics = ['f1', 'accuracy', 'precision', 'balanced_accuracy']
    result_metric = []
    for metric_name in metric_list:
        predicted = predicted_probs
        if metric_name in label_only_metrics:
            predicted = predicted_labels

        result_metric.append(round(metric_dict[metric_name](target, predicted), 3))

    result_dict = dict(zip(metric_list, result_metric))

    return result_dict


@dataclass
class ExecutionParams:
    train_file: str
    test_file: str
    case_label: str
    target_name: str
    task: TaskTypesEnum


@dataclass
class CaseExecutor:
    models: List[BenchmarkModelTypesEnum]
    metric_list: List[str]
    params: ExecutionParams

    _strategy_by_type = {
        BenchmarkModelTypesEnum.tpot: run_tpot,
        BenchmarkModelTypesEnum.h2o: run_h2o,
        BenchmarkModelTypesEnum.autokeras: run_autokeras,
        BenchmarkModelTypesEnum.fedot: run_fedot,
        BenchmarkModelTypesEnum.baseline: run_xgboost
    }

    def execute(self):
        print('START EXECUTION')

        result = {'task': self.params.task.value}

        strategies = {model_type: self._strategy_by_type[model_type] for
                      model_type in self.models}

        for model_type, strategy_func in strategies.items():
            print(f'---------\nRUN {model_type.name}\n---------')
            target, predicted, predicted_labels = strategy_func(self.params)
            result[f'{model_type.name}_metric'] = calculate_metrics(self.metric_list,
                                                                    target=target,
                                                                    predicted_probs=predicted,
                                                                    predicted_labels=predicted_labels)

        return result
