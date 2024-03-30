# How to build custom Datasets for Images in Pytorch (https://youtu.be/ZoZHd0Zm3RY?si=ee6jGDoVWIfvqtcm)
import torch # Imports
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import time
from custom_dataset import CatsAndDogsDataset

from torchvision.models import GoogLeNet_Weights

start_time = time.time()
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('started:', device)

# Hyperparameters
in_channel = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 10
num_epochs = 5

# Load dataset
train_dataset = CatsAndDogsDataset(csv_file='train.csv', root_dir='./dataset/cats_dogs/', transform=transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy arrays to PIL Images
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert PIL image to Tensor
]))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Create data loader
test_dataset = CatsAndDogsDataset(csv_file='val.csv', root_dir='./dataset/cats_dogs/', transform=transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy arrays to PIL Images
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert PIL image to Tensor
]))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # Create data loader

# If dataset contain full list of data we can divide it to train and test set
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [55000, 5000])

# Load pretrained model. If you don't need pretrained model use `weights=None`
model = torchvision.models.googlenet(weights=GoogLeNet_Weights.DEFAULT)
model.to(device)  # Move model to GPU

criterion = nn.CrossEntropyLoss() # Loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Optimizer

print('Start learning...') # Print message
for epoch in range(num_epochs): # Training loop
    start = time.time()
    losses = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) # Get data to CUDO if possible

        # Forward pass
        scores = model(data) # Get scores
        loss = criterion(scores, target) # Calculate loss
        losses.append(loss.item()) # Append loss to list

        # Backward pass
        optimizer.zero_grad() # Clear gradients for this training step
        loss.backward() # Backpropagation
        optimizer.step() # Update weights
    print('Epoch: {}/{}... Loss: {:.4f}, time spent: {:.2f} sec.'.format(epoch+1, num_epochs, loss.item(), time.time() - start))


def check_accuracy(loader, model, train): # Check accuracy of model
    if train: # If training set
        print('Training set')
    else: # If test set
        print('Test set')
    num_correct = 0
    num_samples = 0
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            scores = model(x) # Get scores
            _, predictions = torch.max(scores.data, 1) # Get predictions classes
            num_correct += (predictions == y).sum() # Count number of num_correct predictions
            num_samples += predictions.size(0) # Count num_samples number of predictions
        print(f'Got {num_correct}/{num_samples} with accuracy: {float(num_correct)/float(num_samples)*100:.2f}%')
    model.train()


check_accuracy(train_loader, model, True)
check_accuracy(test_loader, model, False)

print(f'Time spent: {time.time() - start_time:.2f} sec.')