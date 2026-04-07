from enum import Enum


class Mode(str, Enum):
    OPTIMIZATION = "optimization"
    GENERALIZATION = "generalization"


class Variant(str, Enum):
    RAW = "raw"
    FULL = "full"


class HPSetting(str, Enum):
    DEFAULT = "default"
    OPTIMIZED = "optimized"
