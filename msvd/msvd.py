import numpy as np
from dataset_interface import DatasetInterface
import requests
import wget

class MSVD(DatasetInterface):
    def __init__(self) -> None:
        pass

    def load(self) -> (np.array, np.array):
        url = "https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar"
        target_path = "./YouTubeClips.tar"
        filename = wget.download(url, out=target_path)

        return np.array([1, 2]), np.array([3, 4])
