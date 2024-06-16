import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time

start_time = time.time()

# Загрузка данных
data = np.load("sensor_data.npy")

# Параметры
window_size = 3 * 60
prediction_horizon = 3 * 60
input_size = data.shape[1] - 1  # Количество датчиков
hidden_size = 128  # Размер скрытого слоя LSTM
num_layers = 2  # Количество слоёв LSTM
output_size = 1  # Бинарная классификация (проблема или нет)
batch_size = 64
learning_rate = 0.001
num_epochs = 20

# Устройство (CPU или GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')

# Создание окон и целевой переменной
X = []
y = []
for i in range(len(data) - window_size - prediction_horizon):
    X.append(data[i:i + window_size, :-1])  # Исключаем столбец с метками
    y.append(data[i + window_size + prediction_horizon - 1, -1])
X = np.array(X)
y = np.array(y)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразование данных в тензоры PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Создание DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Модель LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Используем последний выход LSTM
        out = self.sigmoid(out)
        return out

# Инициализация модели, оптимизатора и функции потерь
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss

# Обучение модели
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Обнуление градиентов
        optimizer.zero_grad()

        # Прямой проход
        outputs = model(inputs)

        # Вычисление потерь
        loss = criterion(outputs, labels.unsqueeze(1))

        # Обратное распространение ошибки и обновление весов
        loss.backward()
        optimizer.step()
    print('Epoch: {}/{}... Loss: {:.4f}, time spent: {:.2f} sec.'.format(epoch + 1, num_epochs, loss.item(),
                                                                             time.time() - start_time))

# Оценка модели
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = (outputs > 0.5).float()  # Пороговое значение для классификации
    accuracy = (predicted == y_test_tensor.unsqueeze(1)).float().mean()
    print(f'Accuracy: {accuracy.item():.4f} Duration: {time.time() - start_time:.2f}s')
    torch.cuda.empty_cache()

# Accuracy: 0.8361 Duration: 32.71s
# Accuracy: 0.8361 Duration: 54.70s num_layers=3
# Accuracy: 0.8402 Duration: 70.43s bidirectional=True
# Accuracy: 0.8361 Duration: 109.55s num_layers=3, bidirectional=True
# Accuracy: 0.8439 Duration: 138.22s bidirectional=True, num_epochs = 20
# Accuracy: 0.8366 Duration: 138.32s bidirectional=True, num_epochs = 20, learning_rate = 0.0001
