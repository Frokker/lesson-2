import os
import logging
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing, load_breast_cancer
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

from regression_basics.utils import make_regression_data, make_classification_data

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories():
    """Создает необходимые директории проекта"""
    directories = ['data', 'models', 'plots']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        logger.info(f"Создана директория: {dir_name}/")

# 2.1 Кастомный Dataset класс для CSV файлов
class CSVDataset(Dataset):
    def __init__(self, csv_path=None, X=None, y=None, target_column=None,
                 categorical_columns=None, normalize=True, task_type='regression'):
        """
        csv_path: путь к CSV файлу
        target_column: название целевой колонки
        categorical_columns: список категориальных колонок
        normalize: применять ли нормализацию к числовым признакам
        task_type: 'regression' или 'classification'
        """

        if not os.path.exists(csv_path) and not csv_path.startswith('data/'):
            csv_path = os.path.join('data', csv_path)

        self.task_type = task_type
        self.normalize = normalize
        self.categorical_columns = categorical_columns or []

        if csv_path is not None:
            self.data = pd.read_csv(csv_path)
            self.X = self.data.drop(columns=[target_column])
            self.y = self.data[target_column]
        else:
            self.X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
            self.y = pd.Series(y) if not isinstance(y, pd.Series) else y

        self._preprocess_data()

    def _preprocess_data(self):
        # Обработка категориальных признаков
        if self.categorical_columns:
            for col in self.categorical_columns:
                if col in self.X.columns:
                    le = LabelEncoder()
                    self.X[col] = le.fit_transform(self.X[col].astype(str))

        # Нормализация числовых признаков
        if self.normalize:
            numeric_columns = self.X.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                scaler = StandardScaler()
                self.X[numeric_columns] = scaler.fit_transform(self.X[numeric_columns])

        # Преобразование в тензоры
        self.X_tensor = torch.tensor(self.X.values, dtype=torch.float32)

        if self.task_type == 'regression':
            self.y_tensor = torch.tensor(self.y.values, dtype=torch.float32).unsqueeze(1)
        else:
            # Для классификации кодируем метки
            if self.y.dtype == 'object':
                le = LabelEncoder()
                y_encoded = le.fit_transform(self.y)
                self.y_tensor = torch.tensor(y_encoded, dtype=torch.float32).unsqueeze(1)
            else:
                self.y_tensor = torch.tensor(self.y.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.y_tensor[idx]

    def get_feature_dim(self):
        return self.X_tensor.shape[1]


def save_synthetic_datasets():
    """Сохраняет синтетические датасеты в директорию data/"""
    logger.info("Создание синтетических датасетов...")

    # Регрессия
    X, y = make_regression_data(n=1000, source='random')
    df_reg = pd.DataFrame(X.numpy(), columns=['feature_1'])
    df_reg['target'] = y.numpy().flatten()
    df_reg.to_csv('data/regression_synthetic.csv', index=False)
    logger.info("Сохранен: data/regression_synthetic.csv")

    # Классификация
    X, y = make_classification_data(n=1000, source='random')
    df_cls = pd.DataFrame(X.numpy(), columns=['feature_1', 'feature_2'])
    df_cls['target'] = y.numpy().flatten().astype(int)
    df_cls.to_csv('data/classification_synthetic.csv', index=False)
    logger.info("Сохранен: data/classification_synthetic.csv")

    # Реальные датасеты
    X, y = make_regression_data(n=442, source='diabetes')
    df_diabetes = pd.DataFrame(X.numpy())
    df_diabetes['target'] = y.numpy().flatten()
    df_diabetes.to_csv('data/diabetes.csv', index=False)
    logger.info("Сохранен: data/diabetes.csv")

    X, y = make_classification_data(n=569, source='breast_cancer')
    df_cancer = pd.DataFrame(X.numpy())
    df_cancer['target'] = y.numpy().flatten().astype(int)
    df_cancer.to_csv('data/breast_cancer.csv', index=False)
    logger.info("Сохранен: data/breast_cancer.csv")


# Модели из предыдущего задания
class LinearRegressionReg(nn.Module):
    def __init__(self, in_features, l1=0.0, l2=0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.l1 = l1
        self.l2 = l2

    def forward(self, x):
        return self.linear(x)

    def regularization_loss(self):
        l1_loss = self.l1 * torch.norm(self.linear.weight, 1)
        l2_loss = self.l2 * torch.norm(self.linear.weight, 2)
        return l1_loss + l2_loss


class LogisticRegressionReg(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


def train_regression_model(model, train_loader, val_loader, epochs=100, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch) + model.regularization_loss()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    y_pred = model(X_val)
                    val_loss += criterion(y_pred, y_val).item()
            print(
                f'Epoch {epoch + 1}: Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}')


def train_classification_model(model, train_loader, val_loader, epochs=100, lr=0.01):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    y_pred = model(X_val)
                    val_loss += criterion(y_pred, y_val).item()
                    predicted = (torch.sigmoid(y_pred) > 0.5).float()
                    total += y_val.size(0)
                    correct += (predicted == y_val).sum().item()
            accuracy = correct / total
            print(
                f'Epoch {epoch + 1}: Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Accuracy: {accuracy:.4f}')


# 2.2 Эксперименты с различными датасетами
def experiment_regression():
    print("=== Эксперимент с регрессией (California Housing) ===")
    # Загружаем датасет California Housing
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    # Создаем кастомный датасет
    dataset = CSVDataset(X=X, y=y, task_type='regression')

    # Разделяем на train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Обучаем модель
    model = LinearRegressionReg(in_features=dataset.get_feature_dim(), l1=1e-4, l2=1e-3)
    train_regression_model(model, train_loader, val_loader, epochs=100, lr=0.001)


def experiment_classification():
    print("\n=== Эксперимент с классификацией (Breast Cancer) ===")
    # Загружаем датасет Breast Cancer
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    # Создаем кастомный датасет
    dataset = CSVDataset(X=X, y=y, task_type='classification')

    # Разделяем на train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Обучаем модель
    model = LogisticRegressionReg(in_features=dataset.get_feature_dim())
    train_classification_model(model, train_loader, val_loader, epochs=100, lr=0.001)



if __name__ == "__main__":
    setup_directories()
    save_synthetic_datasets()

    # Запускаем эксперименты
    experiment_regression()
    experiment_classification()

"""
Выходные данные: 

=== Эксперимент с регрессией (California Housing) ===
Epoch 20: Train Loss: 0.5246, Val Loss: 0.5412
Epoch 40: Train Loss: 0.5233, Val Loss: 0.5428
Epoch 60: Train Loss: 0.5233, Val Loss: 0.5434
Epoch 80: Train Loss: 0.5228, Val Loss: 0.5438
Epoch 100: Train Loss: 0.5234, Val Loss: 0.5448

=== Эксперимент с классификацией (Breast Cancer) ===
Epoch 20: Train Loss: 0.1805, Val Loss: 0.1631, Accuracy: 0.9561
Epoch 40: Train Loss: 0.1224, Val Loss: 0.1134, Accuracy: 0.9737
Epoch 60: Train Loss: 0.0995, Val Loss: 0.0959, Accuracy: 0.9825
Epoch 80: Train Loss: 0.0886, Val Loss: 0.0881, Accuracy: 0.9825
Epoch 100: Train Loss: 0.0792, Val Loss: 0.0841, Accuracy: 0.9825

"""