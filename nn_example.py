import torch # Imports
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class NN(nn.Module): # Neural network class
    def __init__(self, input_size, num_classes): # 28*28=784
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set device
print(device)

input_size = 28*28 # Hyperparameter
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

train_dataset = datasets.MNIST(root='dataset/', train=True, download=True, transform=transforms.ToTensor()) # Load dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = NN(input_size=input_size, num_classes=num_classes).to(device) # Initialize network

criterion = nn.CrossEntropyLoss() # Loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Optimizer

for epoch in range(num_epochs): # Training loop
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) # Get data to CUDO if possible
        data = data.reshape(data.shape[0], -1) # Reshape data to fit the model
        scores = model(data) # Get scores
        loss = criterion(scores, target) # Calculate loss
        optimizer.zero_grad() # Clear gradients for this training step
        loss.backward() # Backpropagation
        optimizer.step() # Update weights
    print('Epoch: {}/{}... Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


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
            x = x.reshape(x.shape[0], -1)  # Reshape data to fit the model
            scores = model(x) # Get scores
            _, predictions = torch.max(scores.data, 1) # Get predictions classes
            num_correct += (predictions == y).sum() # Count number of num_correct predictions
            num_samples += predictions.size(0) # Count num_samples number of predictions
        print(f'Got {num_correct}/{num_samples} with accuracy: {float(num_correct)/float(num_samples)*100:.2f}%')
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
