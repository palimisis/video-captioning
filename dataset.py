import numpy as np

from dataset_types import DatasetTypes
from msrvtt.msrvtt import MSRVTT
from msvd.msvd import MSVD
from youcook.youcook import YouCook


class Dataset:
    def __init__(self, dataset: DatasetTypes) -> None:
        self.dataset = dataset

        self.X = None
        self.y = None

        pass

    def load(self) -> (np.array, np.array):
        if self.dataset == DatasetTypes.MSRVTT:
            msrvtt = MSRVTT()
            self.X, self.y = msrvtt.load()
        elif self.dataset == DatasetTypes.MSVD:
            msvd = MSVD()
            self.X, self.y = msvd.load()
        elif self.dataset == DatasetTypes.YOU_COOK:
            youcook = YouCook()
            self.X, self.y = youcook.load()
        return self.X, self.y

    def __len__(self):
        return len(self.X)
