import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

start_time = time.time()

# Предположим, что у вас есть данные в переменной 'data'
# data.shape = (количество_измерений, количество_датчиков)
data = np.load("sensor_data.npy")

# Параметры
window_size = 180  # 3 минут в секундах
prediction_horizon = 180  # 3 минут в секундах

# Создание окон и целевой переменной
X = []
y = []
for i in range(len(data) - window_size - prediction_horizon):
    X.append(data[i:i+window_size].flatten())  # Flatten the window
    y.append(1 if any(data[i+window_size:i+window_size+prediction_horizon, 50] == 1) else 0)  # Проверка на наличие проблемы в следующем окне

# Преобразование в массивы NumPy
X = np.array(X)
y = np.array(y)

# Разделение на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Создание и обучение модели
# Используйте DecisionTreeClassifier или RandomForestClassifier
model = RandomForestClassifier(verbose=2)
# model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Прогнозирование
y_pred = model.predict(X_test)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy, "Duration:", time.time() - start_time)
# RandomForestClassifier Accuracy: 0.8433609958506224
# DecisionTreeClassifier Accuracy: 0.7525933609958506 Duration: 948.7821686267853
