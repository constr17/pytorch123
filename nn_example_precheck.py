# PYTORCH COMMON MISTAKES - How To Save Time 🕒 (https://youtu.be/O2wJ3tkc-TU?si=bMr1tc4cbiHCgg1e)
import torch # Imports
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time


class NN(nn.Module): # CNN class
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.dropout = nn.Dropout(p=0.5)  # При переключении оценки удаляется droput
        # self.softmax = nn.Softmax(dim=1) # Распространенная ошибка при использовании CrossEntropyLoss

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.softmax(x) # Распространенная ошибка при использовании CrossEntropyLoss
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set device
print(device)

input_size = 28*28  # Hyperparameters
num_classes = 10
learning_rate = 0.001
batch_size = 64  # 3. Replace to "1", 5. Replace to "64"
num_epochs = 3  # 4. Replace to "1000"

train_dataset = datasets.MNIST(root='dataset/', train=True, download=True, transform=transforms.ToTensor()) # Load dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = NN(input_size=input_size, num_classes=num_classes).to(device) # Initialize network

criterion = nn.CrossEntropyLoss() # Loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Optimizer

# 1. Add one row, 6. Remove row
# data, target = next(iter(train_loader))

for epoch in range(num_epochs): # Training loop
    start = time.time()
    # 2. Comment for, 7. Uncomment for
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Get data to CUDO if possible
        data = data.reshape(data.shape[0], -1)  # Reshape data to fit the model
        # forward
        scores = model(data)  # Get scores
        loss = criterion(scores, target)  # Calculate loss
        # backward
        optimizer.zero_grad()  # Clear gradients for this training step
        loss.backward()  # Backpropagation
        # gradient descent
        optimizer.step()  # Update weights
    print('Epoch: {}/{}... Loss: {:.4f}, time spent: {:.2f} sec.'.format(epoch+1, num_epochs, loss.item(), time.time() - start))


def check_accuracy(loader, model): # Check accuracy of model
    if loader.dataset.train: # If training set
        print('Training set')
    else: # If test set
        print('Test set')
    num_correct = 0
    num_samples = 0
    # model.eval() # Set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = x.reshape(x.shape[0], -1)  # Reshape data to fit the model
            scores = model(x) # Get scores
            _, predictions = scores.max(1) # Get predictions classes
            num_correct += (predictions == y).sum() # Count number of num_correct predictions
            num_samples += predictions.size(0) # Count num_samples number of predictions
        print(f'Got {num_correct}/{num_samples} with accuracy: {float(num_correct)/float(num_samples)*100:.2f}%')
    # model.train()


# check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

model.eval()
check_accuracy(test_loader, model)
model.train()

# Для CNN не надо использовать BatchNorm2D в конце, его можно заменить на nn.Conv2d(bias=False)