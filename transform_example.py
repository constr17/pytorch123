# Pytorch Data Augmentation using Torchvision (https://youtu.be/Zvd276j9sZ8?si=E5IbO0xATzIpnnDB)
import torch
import torchvision.transforms as transforms  # https://pytorch.org/docs/stable/torchvision/transforms.html
from torchvision.utils import save_image
from custom_dataset import CatsAndDogsDataset

transform=transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy arrays to PIL Images
    transforms.Resize((280, 280)),
    transforms.RandomCrop((256, 256)),
    # transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(0.5),  # Randomly flip images
    transforms.RandomVerticalFlip(0.05),  # Randomly flip images
    transforms.RandomGrayscale(p=0.2),  # Randomly convert images to grayscale
    transforms.ToTensor(),  # Convert PIL image to Tensor
    transforms.Normalize(mean=(0, 0, 0), std=(0.5, 0.5, 0.5)),  # Normalize images to range [-1, 1]
])

test_dataset = CatsAndDogsDataset(csv_file='aug.csv', root_dir='./dataset/cats_dogs/', train=False, transform=transform)

img_num = 0
for _ in range(10): # 10 images
    for img, label in test_dataset:
        save_image(img, 'img' + str(img_num) + '.jpg')
        img_num += 1

for img, label in test_dataset:
    print(img.shape)