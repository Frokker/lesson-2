import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import logging
import unittest
from itertools import combinations
from regression_basics.utils import make_regression_data, make_classification_data, RegressionDataset, \
    ClassificationDataset

# Добавляем логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LinearRegressionExp(nn.Module):
    """Линейная регрессия для экспериментов"""

    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


class LogisticRegressionExp(nn.Module):
    """Логистическая регрессия для экспериментов"""

    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


class HyperparameterExperiment:
    """Класс для проведения экспериментов с гиперпараметрами"""

    def __init__(self, model_class, dataset, task_type='regression'):
        """
        Args:
            model_class: Класс модели (LinearRegressionExp или LogisticRegressionExp)
            dataset: Датасет для экспериментов
            task_type: Тип задачи ('regression' или 'classification')
        """
        self.model_class = model_class
        self.dataset = dataset
        self.task_type = task_type
        self.results = []

    def run_experiment(self, learning_rates, batch_sizes, optimizers, epochs=50):
        """
        Проводит эксперименты с различными гиперпараметрами

        Args:
            learning_rates: Список скоростей обучения
            batch_sizes: Список размеров батчей
            optimizers: Список оптимизаторов
            epochs: Количество эпох
        """
        logger.info("Начинаем эксперименты с гиперпараметрами")

        for lr in learning_rates:
            for batch_size in batch_sizes:
                for opt_name in optimizers:
                    logger.info(f"Эксперимент: lr={lr}, batch_size={batch_size}, optimizer={opt_name}")

                    # Создаем DataLoader
                    dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

                    # Создаем модель
                    model = self.model_class(in_features=self.dataset.X.shape[1])

                    # Выбираем оптимизатор
                    if opt_name == 'SGD':
                        optimizer = optim.SGD(model.parameters(), lr=lr)
                    elif opt_name == 'Adam':
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                    elif opt_name == 'RMSprop':
                        optimizer = optim.RMSprop(model.parameters(), lr=lr)

                    # Выбираем функцию потерь
                    if self.task_type == 'regression':
                        criterion = nn.MSELoss()
                    else:
                        criterion = nn.BCEWithLogitsLoss()

                    # Обучаем модель
                    losses = self._train_model(model, dataloader, criterion, optimizer, epochs)

                    # Сохраняем результаты
                    self.results.append({
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'optimizer': opt_name,
                        'final_loss': losses[-1],
                        'losses': losses
                    })

        logger.info("Эксперименты завершены")

    def _train_model(self, model, dataloader, criterion, optimizer, epochs):
        """Обучает модель и возвращает историю потерь"""
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)

        return losses

    def visualize_results(self):
        """Визуализирует результаты экспериментов"""
        df = pd.DataFrame(self.results)

        # График 1: Влияние learning rate
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        lr_results = df.groupby('learning_rate')['final_loss'].mean()
        plt.bar(range(len(lr_results)), lr_results.values)
        plt.xticks(range(len(lr_results)), [f'{lr:.4f}' for lr in lr_results.index])
        plt.title('Влияние Learning Rate')
        plt.xlabel('Learning Rate')
        plt.ylabel('Final Loss')

        # График 2: Влияние batch size
        plt.subplot(1, 3, 2)
        batch_results = df.groupby('batch_size')['final_loss'].mean()
        plt.bar(range(len(batch_results)), batch_results.values)
        plt.xticks(range(len(batch_results)), batch_results.index)
        plt.title('Влияние Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Final Loss')

        # График 3: Влияние оптимизатора
        plt.subplot(1, 3, 3)
        opt_results = df.groupby('optimizer')['final_loss'].mean()
        plt.bar(range(len(opt_results)), opt_results.values)
        plt.xticks(range(len(opt_results)), opt_results.index)
        plt.title('Влияние Оптимизатора')
        plt.xlabel('Optimizer')
        plt.ylabel('Final Loss')

        plt.tight_layout()
        plt.savefig('plots/hyperparameter_results.png')
        plt.show()


class FeatureEngineering:
    """Класс для создания новых признаков"""

    def __init__(self, X, y):
        """
        Args:
            X: Исходные признаки
            y: Целевая переменная
        """
        self.X_original = X
        self.y = y

    def create_polynomial_features(self, degree=2):
        """
        Создает полиномиальные признаки

        Args:
            degree: Степень полинома

        Returns:
            X_poly: Данные с полиномиальными признаками
        """
        logger.info(f"Создание полиномиальных признаков степени {degree}")
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(self.X_original.numpy())
        return torch.tensor(X_poly, dtype=torch.float32)

    def create_interaction_features(self):
        """
        Создает признаки взаимодействия между исходными признаками

        Returns:
            X_interaction: Данные с признаками взаимодействия
        """
        logger.info("Создание признаков взаимодействия")
        X_np = self.X_original.numpy()
        n_features = X_np.shape[1]

        # Создаем все возможные парные взаимодействия
        interactions = []
        for i, j in combinations(range(n_features), 2):
            interactions.append((X_np[:, i] * X_np[:, j]).reshape(-1, 1))

        if interactions:
            X_interactions = np.hstack(interactions)
            X_combined = np.hstack([X_np, X_interactions])
        else:
            X_combined = X_np

        return torch.tensor(X_combined, dtype=torch.float32)

    def create_statistical_features(self, window_size=5):
        """
        Создает статистические признаки (среднее, дисперсия)

        Args:
            window_size: Размер окна для вычисления статистик

        Returns:
            X_stats: Данные со статистическими признаками
        """
        logger.info("Создание статистических признаков")
        X_np = self.X_original.numpy()

        # Вычисляем скользящее среднее и дисперсию
        means = []
        stds = []

        for i in range(len(X_np)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            window_data = X_np[start_idx:end_idx]

            means.append(np.mean(window_data, axis=0))
            stds.append(np.std(window_data, axis=0))

        means = np.array(means)
        stds = np.array(stds)

        X_combined = np.hstack([X_np, means, stds])
        return torch.tensor(X_combined, dtype=torch.float32)

    def compare_features(self, model_class, task_type='regression', epochs=50):
        """
        Сравнивает качество модели с различными наборами признаков

        Args:
            model_class: Класс модели
            task_type: Тип задачи
            epochs: Количество эпох обучения

        Returns:
            results: Словарь с результатами сравнения
        """
        logger.info("Сравнение различных наборов признаков")

        feature_sets = {
            'original': self.X_original,
            'polynomial': self.create_polynomial_features(degree=2),
            'interaction': self.create_interaction_features(),
            'statistical': self.create_statistical_features()
        }

        results = {}

        for name, X_features in feature_sets.items():
            logger.info(f"Обучение модели с признаками: {name}")

            # Создаем датасет
            if task_type == 'regression':
                dataset = RegressionDataset(X_features, self.y)
                criterion = nn.MSELoss()
            else:
                dataset = ClassificationDataset(X_features, self.y)
                criterion = nn.BCEWithLogitsLoss()

            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            # Создаем и обучаем модель
            model = model_class(in_features=X_features.shape[1])
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            losses = []
            for epoch in range(epochs):
                epoch_loss = 0
                for X_batch, y_batch in dataloader:
                    optimizer.zero_grad()
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                losses.append(epoch_loss / len(dataloader))

            results[name] = {
                'final_loss': losses[-1],
                'losses': losses,
                'n_features': X_features.shape[1]
            }

        return results

    def visualize_feature_comparison(self, results):
        """Визуализирует сравнение различных наборов признаков"""
        plt.figure(figsize=(12, 4))

        # График 1: Финальные потери
        plt.subplot(1, 2, 1)
        names = list(results.keys())
        final_losses = [results[name]['final_loss'] for name in names]
        plt.bar(names, final_losses)
        plt.title('Сравнение финальных потерь')
        plt.ylabel('Final Loss')
        plt.xticks(rotation=45)

        # График 2: Кривые обучения
        plt.subplot(1, 2, 2)
        for name in names:
            plt.plot(results[name]['losses'], label=f"{name} ({results[name]['n_features']} features)")
        plt.title('Кривые обучения')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('plots/feature_comparison.png')
        plt.show()


# 3.1 Исследование гиперпараметров
def run_hyperparameter_experiments():
    """Проводит эксперименты с гиперпараметрами"""
    logger.info("Запуск экспериментов с гиперпараметрами")

    # Создаем данные для регрессии
    X, y = make_regression_data(n=500, source='random')
    dataset = RegressionDataset(X, y)

    # Настройки экспериментов
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32, 64]
    optimizers = ['SGD', 'Adam', 'RMSprop']

    # Проводим эксперименты
    experiment = HyperparameterExperiment(LinearRegressionExp, dataset, 'regression')
    experiment.run_experiment(learning_rates, batch_sizes, optimizers, epochs=30)
    experiment.visualize_results()

    return experiment.results


# 3.2 Feature Engineering
def run_feature_engineering_experiments():
    """Проводит эксперименты с feature engineering"""
    logger.info("Запуск экспериментов с feature engineering")

    # Создаем данные
    X, y = make_regression_data(n=300, source='random')

    # Создаем объект для feature engineering
    fe = FeatureEngineering(X, y)

    # Сравниваем различные наборы признаков
    results = fe.compare_features(LinearRegressionExp, task_type='regression', epochs=30)
    fe.visualize_feature_comparison(results)

    return results


# Unit тесты
class TestExperiments(unittest.TestCase):
    """Unit тесты для критических функций"""

    def setUp(self):
        """Подготовка данных для тестов"""
        self.X, self.y = make_regression_data(n=100, source='random')
        self.fe = FeatureEngineering(self.X, self.y)

    def test_polynomial_features(self):
        """Тест создания полиномиальных признаков"""
        X_poly = self.fe.create_polynomial_features(degree=2)
        self.assertGreater(X_poly.shape[1], self.X.shape[1])
        self.assertEqual(X_poly.shape[0], self.X.shape[0])

    def test_interaction_features(self):
        """Тест создания признаков взаимодействия"""
        X_interact = self.fe.create_interaction_features()
        self.assertGreaterEqual(X_interact.shape[1], self.X.shape[1])
        self.assertEqual(X_interact.shape[0], self.X.shape[0])

    def test_statistical_features(self):
        """Тест создания статистических признаков"""
        X_stats = self.fe.create_statistical_features()
        self.assertGreater(X_stats.shape[1], self.X.shape[1])
        self.assertEqual(X_stats.shape[0], self.X.shape[0])


if __name__ == "__main__":
    # Создаем директории для сохранения результатов
    import os

    os.makedirs('plots', exist_ok=True)

    # Запускаем эксперименты
    print("=== Эксперименты с гиперпараметрами ===")
    hyperparams_results = run_hyperparameter_experiments()

    print("\n=== Эксперименты с Feature Engineering ===")
    feature_results = run_feature_engineering_experiments()

    # Запускаем тесты
    print("\n=== Запуск unit тестов ===")
    unittest.main(argv=[''], exit=False, verbosity=2)
