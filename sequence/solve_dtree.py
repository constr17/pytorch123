import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import time

start_time = time.time()

# Предположим, что у вас есть данные в переменной 'data'
# data.shape = (количество_измерений, количество_датчиков)
data = np.load("sensor_data.npy")
#
# # Выбор значимых столбцов для обучения
# columns_to_select = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50]
# data = data[:, columns_to_select]

param_grid = {
    'n_estimators': [50, 100],  # , 200
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2'],
}

# Параметры
window_size = 180  # 3 минут в секундах
prediction_horizon = 180  # 3 минут в секундах

# Создание окон и целевой переменной
X = []
y = []
for i in range(len(data) - window_size - prediction_horizon):
    X.append(data[i:i+window_size].flatten())  # Flatten the window
    y.append(1 if any(data[i+window_size:i+window_size+prediction_horizon, len(data[0]) - 1] == 1) else 0)  # Проверка на наличие проблемы в следующем окне

# Преобразование в массивы NumPy
X = np.array(X)
y = np.array(y)

# Разделение на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Создание и обучение модели
# Используйте DecisionTreeClassifier или RandomForestClassifier
model = RandomForestClassifier(verbose=2)
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)

# Прогнозирование
# y_pred = model.predict(X_test)

# Оценка точности
# accuracy = accuracy_score(y_test, y_pred)
print("Duration:", time.time() - start_time)
# print("Accuracy:", accuracy, "Duration:", time.time() - start_time)

# RandomForestClassifier Accuracy: 0.8433609958506224
# DecisionTreeClassifier Accuracy: 0.7525933609958506 Duration: 948.7821686267853
# RandomForestClassifier Accuracy: 0.8428423236514523 Duration: 1688.030225276947 n_estimators=200
# RandomForestClassifier Accuracy: 0.8485477178423236 Duration: 171.87895274162292 n_estimators=100, max_depth=10
# RandomForestClassifier Accuracy: 0.8433609958506224 Duration: 237.83010005950928 n_estimators=100, max_depth=15
# RandomForestClassifier Accuracy: 0.8350622406639004 Duration: 228.96248817443848 Значимые столбцы

# **Основные:**
#
# * `n_estimators`: Количество деревьев в лесу. Увеличивайте, пока производительность не перестанет улучшаться (больше - лучше, но медленнее).
# * `max_depth`: Максимальная глубина каждого дерева. Контролирует сложность модели (слишком большая глубина ведет к переобучению).
# * `min_samples_split`: Минимальное количество выборок для разделения узла дерева.
# * `min_samples_leaf`: Минимальное количество выборок в листе дерева.
# * `max_features`: Количество признаков, рассматриваемых при поиске лучшего разделения (`"sqrt"`, `"log2"`,  число, процент от общего количества).
#
# **Дополнительные:**
#
# * `criterion`: Функция измерения качества разделения (`"gini"`, `"entropy"`).
# * `bootstrap`: Использовать ли бутстрап выборок для обучения каждого дерева (`True`, `False`).
# * `class_weight`: Веса классов для несбалансированных данных (`"balanced"`, словарь весов, `None`).
#
# **Пример тюнинга с помощью GridSearchCV:**
#
# ```python
# from sklearn.model_selection import GridSearchCV
#
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 5, 10],
#     'min_samples_split': [2, 5],
#     'max_features': ['sqrt', 'log2'],
# }
#
# model = RandomForestClassifier(random_state=42)
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
# grid_search.fit(X_train, y_train)
#
# print("Best parameters:", grid_search.best_params_)
# ```
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html