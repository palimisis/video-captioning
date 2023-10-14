from abc import abstractmethod, ABC

import numpy as np


class DatasetInterface(ABC):

    @abstractmethod
    def load(self) -> (np.array, np.array):
        pass
