import numpy as np
from dataset_interface import DatasetInterface
import json


class YouCook(DatasetInterface):
    def __init__(self, path="datasets/youcookii_annotations_trainval.json") -> None:
        self.path = path
        self.X, self.y = [], []

    def load(self) -> (np.array, np.array):
        dataset = json.load(open(self.path))

        if not self.is_downloaded():
            self.download()

        for item in dataset["database"].items():
            video_id, video_metadata = item
            if video_metadata["subset"] == "training":
                self.X.append(video_metadata["video_url"])
                self.y.append(video_metadata["annotations"])
        return self.X, self.y

    def is_downloaded(self) -> bool:
        pass

    def download(self):
        pass
