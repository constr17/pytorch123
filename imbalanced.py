# How to deal with Imbalanced Datasets in PyTorch - Weighted Random Sampler Tutorial (https://youtu.be/4JFVhJyTZ44?si=9B9u_x8DfQimgxiS)

# Methods for dealing with imbalanced datasets:
# 1. Oversampling: Duplicate the minority class samples.
# 2. Class weighting: Assign higher weights to the minority class.

import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

# Shortest way to use the imbalanced dataset
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([11, 1]))

def get_loader(root_dir, batch_size):
    my_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(root_dir, transform=my_transform)
    class_weights = []
    for root, subdir, files in os.walk(root_dir):
        if len(files) > 0:
            class_weights.append(1/len(files))
    sample_weight = [0] * len(dataset)

    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weight[idx] = class_weight

    sampler = WeightedRandomSampler(sample_weight, len(sample_weight), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return loader

def main():
    loader = get_loader('dataset/imbalanced', batch_size=8)

    num0 = 0
    num1 = 0
    for epoch in range(10):
        for data, label in loader:
            num0 += torch.sum(label == 0)
            num1 += torch.sum(label == 1)
    print(num0, num1)


if __name__ == '__main__':
    main()