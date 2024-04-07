# Albumentations Tutorial for Data Augmentation (Pytorch focused) (https://youtu.be/rAdLwKJBvPM?si=of3Yru5kkz_3WbxF)
# https://albumentations.ai/
import cv2
import albumentations as A
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as pcs


def plot_examples(image_list, box=[]):
    plt.figure()
    columns = 4
    for i, image in enumerate(image_list):
        ax = plt.subplot(int(len(image_list) / columns + 1), columns, i + 1)
        plt.imshow(image)
        if len(box) > 0:
            b = box[i]
            rect = pcs.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color='r', linewidth=4)
            ax.add_patch(rect)
    plt.show()


image = cv2.imread('dataset/aug/cow.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.imread('dataset/aug/cow_mask.png', cv2.IMREAD_GRAYSCALE)
print('Check sizes', cv2.imread('dataset/aug/cow.png').shape, cv2.imread('dataset/aug/cow_mask.png').shape)

bboxes =[[60, 4, 360, 420]]  # Cow bounding box
# pascal_voc [x_min, y_min, x_max, y_max]
# https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#different-annotations-formats

transform = A.Compose(
    [
        A.Resize(height=600, width=600),
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
    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=300, min_visibility=0.3,  label_fields=[])
)
images_list = [image]
saved_bboxes = [bboxes[0]]

for i in range(8):
    augmentation = transform(image=image, bboxes=bboxes)
    augmented_image = augmentation['image']
    if len(augmentation['bboxes']) == 0:  # No objects in image
        continue
    images_list.append(augmented_image)
    saved_bboxes.append(augmentation['bboxes'][0])

plot_examples(images_list, saved_bboxes)
print(saved_bboxes)
