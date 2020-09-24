from enum import Enum


class BenchmarkModelTypesEnum(Enum):
    h2o = 'h2o',
    tpot = 'tpot',
    autokeras = 'AutoKeras',
    fedot = 'fedot',
    baseline = 'baseline'
