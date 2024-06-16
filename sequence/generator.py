"""Пример рядов данных."""""

import numpy as np
import matplotlib.pyplot as plt

# Параметры данных
num_samples = 10000  # Количество временных шагов
num_sensors = 50
frequency = 1  # Hz
window_size = 3 * 60  # 3 минут в секундах
prediction_horizon = 3 * 60  # 3 минут в секундах

np.random.seed(125)

# Генерация случайных данных датчиков
data = np.random.rand(num_samples, num_sensors)

# Определение типов проблем и реагирующих датчиков
problem_sensors = {
    "problem_1": [1, 5, 10],
    "problem_2": [15, 20, 25],
    "problem_3": [30, 35, 40]
}

# Массив для хранения end_index для каждой проблемы
problem_end_indices = []

# Добавление искусственных аномалий
for problem_type, sensors in problem_sensors.items():
    for _ in range(3):
        start_index = np.random.randint(window_size, num_samples - window_size - prediction_horizon)
        end_index = start_index + window_size
        problem_end_indices.append(end_index)

        data[start_index:end_index, sensors] = data[start_index:end_index, sensors] * 2 - 1

print(problem_end_indices)

# Добавление столбца для индикации проблемы (0 - нет проблем, 1 - проблема)
data = np.hstack((data, np.zeros((num_samples, 1))))

# Установка меток проблем с помощью problem_end_indices
for end_index in problem_end_indices:
    data[end_index - prediction_horizon:end_index, -1] = 1
    print(len(data[end_index - prediction_horizon:end_index, -1]))

print(data[:, -1].sum())

# Сохранение данных в файл
np.save("sensor_data.npy", data)
print(sum(data.T[-1].tolist()))

with open("sensor_desc.txt", "w") as f:
    f.write("sensor_data.npy\nproblem_sensors: ")
    f.write(str(problem_sensors) + "\nproblem_end_indices: ")
    f.write(str(problem_end_indices) + "\n")

plt.plot(data[:, 0], alpha=0.5, label="standard")
plt.plot(data[:, 1], alpha=0.5, label="deviated")
plt.plot(data[:, 15], alpha=0.5, label="deviated")
plt.plot(data[:, 30], alpha=0.5, label="deviated")
plt.plot(-data[:, -1], label="indicator", color="black")
plt.grid()
plt.legend()
plt.show()
