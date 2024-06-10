import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cupy as cp  # Импортируем cuXfilter: pip install cupy-cuda12x
import time

start_time = time.time()

# Загрузка данных
data = np.load("sensor_data.npy")

# Параметры
window_size = 3 * 60  # 3 минут в секундах
prediction_horizon = 3 * 60  # 3 минут в секундах

# Создание окон и целевой переменной
X = []
y = []
for i in range(len(data) - window_size - prediction_horizon):
    X.append(data[i:i+window_size].flatten())
    y.append(data[i + window_size + prediction_horizon, -1])  # Берём значение метки через prediction_horizon

X = np.array(X)
y = np.array(y)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Переносим данные на GPU
X_train_gpu = cp.array(X_train)
y_train_gpu = cp.array(y_train)
X_test_gpu = cp.array(X_test)

# Создание и обучение модели XGBoost
model = XGBClassifier(
    tree_method='hist',      # Используем 'hist' вместо 'gpu_hist'
    device='cuda',           # Указываем использовать CUDA
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train_gpu, y_train_gpu)

# Прогнозирование на тестовом наборе
y_pred = model.predict(X_test_gpu)

# Преобразование y_pred в NumPy массив
y_pred_np = cp.asnumpy(y_pred)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred_np)
print("Accuracy:", accuracy, "Duration:", time.time() - start_time)
# Accuracy: 0.8345435684647303 Duration: 21.848129749298096

# **Основные:**
#
# * `n_estimators`:  Количество деревьев в ансамбле.
# * `learning_rate`:  Скорость обучения (обычно маленькое значение, например, 0.01, 0.1).
# * `max_depth`:  Максимальная глубина дерева.
# * `subsample`:  Доля выборок для обучения каждого дерева (меньше 1 - добавление случайности).
# * `colsample_bytree`: Доля признаков для обучения каждого дерева.
# * `gamma`:  Минимальное уменьшение потерь для дальнейшего разделения (регуляризация).
#
# **Дополнительные:**
#
# * `reg_alpha`:  L1 регуляризация.
# * `reg_lambda`:  L2 регуляризация.
# * `scale_pos_weight`:  Балансировка классов (для несбалансированных данных).
#
# **Пример тюнинга с помощью RandomizedSearchCV:**
#
# ```python
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint
#
# param_dist = {
#     'n_estimators': randint(50, 200),
#     'max_depth': randint(3, 10),
#     'learning_rate': [0.01, 0.05, 0.1],
#     'subsample': [0.7, 0.8, 0.9],
#     'colsample_bytree': [0.7, 0.8, 0.9],
#     'gamma': [0, 0.1, 0.5],
# }
#
# model = XGBClassifier(random_state=42)
# random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=5)
# random_search.fit(X_train, y_train)
#
# print("Best parameters:", random_search.best_params_)
# ```
# https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier