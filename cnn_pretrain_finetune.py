# Pytorch Transfer Learning and Fine Tuning Tutorial (https://youtu.be/qaDe0qQZ5AQ?si=JKz4LstFv4ivmMv3)
import torch # Imports
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import time

from torchvision.models import VGG16_Weights

start_time = time.time()
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('started:', device)

# Hyperparameters
in_channel = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 5


class Identity(nn.Module): # Identity class
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x # Return input


# Load pretrained model. If you don't need pretrained model use `weights=None`
model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
# TURN OFF LEARNING RATE FOR ALL PARAMETERS
for param in model.parameters(): # Set all parameters to learnable
    param.requires_grad = False # Set all parameters to not require gradients

# Modify model to use identity function for all fully connected layers
model.avgpool = Identity() # Replace average pooling layer
model.classifier = nn.Sequential(nn.Linear(512, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, num_classes))
# for i in range(len(model.classifier)): # Replace all fully connected layers
#     model.classifier[i] = Identity() # Set all fully connected layers to identity function
model.to(device) # Move model to GPU

# Load dataset
train_dataset = datasets.CIFAR10(root='dataset/', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.CIFAR10(root='dataset/', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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


def check_accuracy(loader, model): # Check accuracy of model
    if loader.dataset.train: # If training set
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


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

print(f'Time spent: {time.time() - start_time:.2f} sec.')