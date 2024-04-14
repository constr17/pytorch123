# Albumentations Tutorial for Data Augmentation (Pytorch focused) (https://youtu.be/rAdLwKJBvPM?si=of3Yru5kkz_3WbxF)
# https://albumentations.ai/
from PIL import Image
import torch.nn as nn
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import cv2

class ImageFolder(nn.Module):
    def __init__(self, root_dir, transform=None):
        super(ImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = os.listdir(root_dir)
        self.class_names.remove('classname.txt')

        for index, name in enumerate(self.class_names):
            if os.path.isdir(os.path.join(root_dir, name)):
                files = os.listdir(os.path.join(root_dir, name))
                self.data += list(zip(files, [index] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_name, class_id = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[class_id])
        image = np.array(Image.open(os.path.join(root_and_dir, file_name)))

        if self.transform is not None:
            augmentation = self.transform(image=image)
            image = augmentation["image"]

        return image, class_id


transform = A.Compose(
    [
        A.Resize(height=300, width=300),
        A.RandomCrop(height=280, width=280),
        A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.ColorJitter(p=0.5),
            ], p=1
        ),
        # ToTensor(), -> Normalize(mean, std)
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ]
)

dataset = ImageFolder(root_dir="./dataset/cats_dogs/train", transform=transform)

for x, y in dataset:
    print(x.shape, y)
