import numpy as np
import kaggle
from dataset_interface import DatasetInterface


class MSRVTT(DatasetInterface):
    def __init__(self) -> None:
        pass

    def load(self) -> (np.array, np.array):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('vishnutheepb/msrvtt', path='.', unzip=True, quiet=False)
        return np.array([1, 2]), np.array([3, 4])
