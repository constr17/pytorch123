# How to build custom Datasets for Images in Pytorch (https://youtu.be/ZoZHd0Zm3RY?si=ee6jGDoVWIfvqtcm)
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io


class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(Path(root_dir)/csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)
