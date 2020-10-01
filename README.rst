AutoML-benchmark
================

This tool will help you to execute different AutoMl frameworks with
problem data you want. The repository already has some cases
(i.e. credit_scoring), the ability to work with PMLB datasets and open
to new experiments.

Installation
------------
AutoML benchmark includes
`FEDOT framework <https://github.com/nccr-itmo/FEDOT>`__ as a submodule.

To work with the FEDOT submodule without extra efforts and mistakes
follow the steps:

1. To clone module with the content of submodule

   ::

   $ git clone –-recursive https://github.com/ITMO-NSS-team/AutoML-benchmark


2. From the project root directory create a soft link to the core of
   the FEDOT Framework typing following command in terminal:

   Linux/OSX

   ::

   $ ln -s FEDOT/core core



   Windows (run terminal as administrator)

   ::

   $ mklink /j "core" "FEDOT/core"


**Please, do not add your link directory to the commits**. If you don’t want
to use the link anymore type the following command from the project root
directory:

Linux/OSX

::

$ unlink core

Windows (run terminal as administrator)

::

$ rmdir core


How to
------

Execute existing cases
~~~~~~~~~~~~~~~~~~~~~~

All the existing cases are located in test_cases directory. To execute
an experiment open the directory with the case and run the script
case_name.py inside.

The main part presents the CaseExecutor with the params, models and
metrics to run.

.. code:: python
   result_metrics = CaseExecutor(params=ExecutionParams(train_file=train_file,
                                                        test_file=test_file,
                                                        task=TaskTypesEnum.classification,
                                                        target_name='default',
                                                        case_label='scoring'),
                                 models=[BenchmarkModelTypesEnum.baseline,
                                         BenchmarkModelTypesEnum.tpot,
                                         BenchmarkModelTypesEnum.fedot],
                                 metric_list=['roc_auc', 'f1']).execute()


To understand which hyperparameters were used for AutoML models have a
look at the realisation of the get_models_hyperparameters function to
see or tailor the requirement parameters.

.. code:: python

   result_metrics['hyperparameters'] = get_models_hyperparameters()

The following function saves the result of the execution to json file
next to the case script.

.. code:: python

   save_metrics_result_file(result_metrics, file_name='scoring_metrics')

Add custom experiment
~~~~~~~~~~~~~~~~~~~~~

To build an experiment create a directory with the name of your case in
test_cases directory. Create a directory named ``data`` inside to put your data
files here and a script named as your case and fill it in as follows:

Note! Do not forget to replace all the ``your_case`` phrases in names to the name of
your case

.. code:: python

   from benchmark_model_types import BenchmarkModelTypesEnum
   from executor import CaseExecutor, ExecutionParams
   from core.repository.tasks import TaskTypesEnum
   from benchmark_utils import (get_models_hyperparameters,
                                save_metrics_result_file,
                                get_your_case_data_paths,
                                )

   if __name__ == '__main__':
       train_file, test_file = get_your_case_data_paths()

       result_metrics = CaseExecutor(params=ExecutionParams(train_file=train_file,
                                                            test_file=test_file,
                                                            task=TaskTypesEnum.classification,
                                                            target_name='default',
                                                            case_label='your_case'),
                                     models=[BenchmarkModelTypesEnum.baseline,
                                             BenchmarkModelTypesEnum.tpot,
                                             BenchmarkModelTypesEnum.fedot],
                                     metric_list=['roc_auc', 'f1']).execute()

        result_metrics['hyperparameters'] = get_models_hyperparameters()

        save_metrics_result_file(result_metrics, file_name='your_case_metrics')

To import your data properly make a corresponding function for your case
in benchmark_utils script:

.. code:: python

   def get_your_case_data_paths() -> Tuple[str, str]:
       train_file_path = os.path.join('test_cases', 'your_directory', 'data', 'your_case_name_train.csv')
       test_file_path = os.path.join('test_cases', 'your_directory', 'data', 'your_case_name_test.csv')
       full_train_file_path = os.path.join(str(project_root()), train_file_path)
       full_test_file_path = os.path.join(str(project_root()), test_file_path)

       return full_train_file_path, full_test_file_path


Pay attention to the task and model types and target_name(the target
column name). All the supported task types and model types are available in the
TaskTypesEnum and BenchmarkModelTypesEnum objects respectively.
