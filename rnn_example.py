# Pytorch RNN example (Recurrent Neural Network) (https://youtu.be/Gl2WXLIMvKA?si=6G2EgCYQsKQYaMOu)
import torch # Imports
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set device
print(device)

input_size = 28 # Hyperparameter
sequence_length = 28
num_layers = 2
hidden_size = 128
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2


class RNN(nn.Module): # RNN class
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True) # N*time_seq*features
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True) # N*time_seq*features
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # Initialize hidden state

        out, _ = self.gru(x, h0) # Forward pass
        out = out.reshape(out.size(0), -1) # Reshape output
        out = self.fc(out) # Linear layer
        return out

train_dataset = datasets.MNIST(root='dataset/', train=True, download=True, transform=transforms.ToTensor()) # Load dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device) # Initialize network

criterion = nn.CrossEntropyLoss() # Loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Optimizer

for epoch in range(num_epochs): # Training loop
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).squeeze(1), target.to(device) # Get data to CUDO if possible
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
            x, y = x.to(device).squeeze(1), y.to(device)
            scores = model(x) # Get scores
            _, predictions = torch.max(scores.data, 1) # Get predictions classes
            num_correct += (predictions == y).sum() # Count number of num_correct predictions
            num_samples += predictions.size(0) # Count num_samples number of predictions
        print(f'Got {num_correct}/{num_samples} with accuracy: {float(num_correct)/float(num_samples)*100:.2f}%')
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
