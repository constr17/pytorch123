# Pytorch CNN example (Convolutional Neural Network) (https://youtu.be/wnK3uWv_WkU?si=mEQcGoZrJJm_FhmH)
import torch # Imports
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time


class CNN(nn.Module): # CNN class
    def __init__(self, in_channels = 1, num_classes = 10):
        super(CNN, self).__init__()
        # n_out = (n_in + 2 * padding - kernel_size)//stride + 1 = (28 + 2*1 - 3)//1 + 1 = 28
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set device
print(device)

in_channel = 1 # Hyperparameters
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

train_dataset = datasets.MNIST(root='dataset/', train=True, download=True, transform=transforms.ToTensor()) # Load dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = CNN(in_channels=in_channel, num_classes=num_classes).to(device) # Initialize network

criterion = nn.CrossEntropyLoss() # Loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Optimizer

for epoch in range(num_epochs): # Training loop
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) # Get data to CUDO if possible
        scores = model(data) # Get scores
        loss = criterion(scores, target) # Calculate loss
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
