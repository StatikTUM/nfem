from enum import Enum


class ModelStatus(Enum):
    """Enum for the model status """
    initial = 0
    duplicate = 1
    prediction = 2
    iteration = 3
    equilibrium = 4
    eigenvector = 5
