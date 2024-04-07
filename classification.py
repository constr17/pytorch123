# Albumentations Tutorial for Data Augmentation (Pytorch focused) (https://youtu.be/rAdLwKJBvPM?si=of3Yru5kkz_3WbxF)
# https://albumentations.ai/
import cv2
import albumentations as A
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def plot_examples(image_list):
    plt.figure()
    columns = 4
    for i, image in enumerate(image_list):
        plt.subplot(int(len(image_list) / columns + 1), columns, i + 1)
        plt.imshow(image)
    plt.show()

image = Image.open('dataset/aug/cow.png')

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
        )
    ]
)
images_list = [image]
image = np.array(image)
for i in range(15):
    augmentation = transform(image=image)
    augmented_image = augmentation['image']
    images_list.append(augmented_image)

plot_examples(images_list)
