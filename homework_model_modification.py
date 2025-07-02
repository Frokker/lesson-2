import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    ConfusionMatrixDisplay

from regression_basics.utils import make_regression_data, make_classification_data, mse, RegressionDataset, \
    ClassificationDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_model(model, model_name, metrics=None):
    """Сохраняет модель в директорию models/"""
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{model_name}.pth'

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'input_features': model.linear.in_features,
        'metrics': metrics or {}
    }, model_path)

    logger.info(f"Модель сохранена: {model_path}")
    return model_path


def load_model(model_class, model_name, in_features):
    """Загружает модель из директории models/"""
    model_path = f'models/{model_name}.pth'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    checkpoint = torch.load(model_path)
    model = model_class(in_features=in_features)
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"Модель загружена: {model_path}")
    return model, checkpoint.get('metrics', {})


# 1.1 Модифицированная линейная регрессия с L1/L2 и early stopping
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


def train_linreg_with_early_stopping(
        model, dataloader, val_dataloader, epochs=100, lr=0.1, patience=10
):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    logger.info("Начинаем обучение линейной регрессии с early stopping")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch) + model.regularization_loss()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Валидация
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                y_pred = model(X_val)
                val_loss = criterion(y_pred, y_val)
                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)
        avg_train_loss = train_loss / len(dataloader)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping на эпохе {epoch}")
                model.load_state_dict(best_state)
                break

        if epoch % 10 == 0:
            logger.info(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Сохраняем лучшую модель
    final_metrics = {'best_val_loss': best_loss, 'epochs_trained': epoch}
    save_model(model, 'linear_regression_best', final_metrics)

    return best_loss


# 1.2 Логистическая регрессия с поддержкой мультикласса и метриками
class LogisticRegressionMulti(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)


def train_logreg_multiclass(
        model, dataloader, val_dataloader, epochs=100, lr=0.1
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    logger.info("Начинаем обучение логистической регрессии")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch.squeeze().long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if epoch % 20 == 0:
            avg_train_loss = train_loss / len(dataloader)
            logger.info(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}')

        # Оценка и сохранение
        precision, recall, f1, roc_auc, _, _ = evaluate_multiclass(model, val_dataloader, num_classes=3)
        metrics = {'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc}
        save_model(model, 'logistic_regression_multiclass', metrics)

        logger.info(f"Финальные метрики: Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        return metrics


def evaluate_multiclass(model, dataloader, num_classes):
    model.eval()
    y_true = []
    y_pred = []
    y_score = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(y_batch.squeeze().cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs.cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    # Для ROC-AUC: one-vs-rest
    try:
        roc_auc = roc_auc_score(y_true, y_score, multi_class='ovr')
    except:
        roc_auc = float('nan')
    return precision, recall, f1, roc_auc, y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


# Пример использования для регрессии:
if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    # Линейная регрессия с L1/L2 и early stopping
    X, y = make_regression_data(n=200, noise=0.1, source='random')
    dataset = RegressionDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    model = LinearRegressionReg(in_features=X.shape[1], l1=1e-3, l2=1e-2)
    train_linreg_with_early_stopping(model, train_loader, val_loader, epochs=100, lr=0.1, patience=10)

    # Логистическая регрессия для мультикласса
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = torch.tensor(iris['data'], dtype=torch.float32)
    y = torch.tensor(iris['target'], dtype=torch.int64).unsqueeze(1)
    dataset = ClassificationDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    model = LogisticRegressionMulti(in_features=X.shape[1], num_classes=3)
    train_logreg_multiclass(model, train_loader, val_loader, epochs=100, lr=0.1)
    precision, recall, f1, roc_auc, y_true, y_pred = evaluate_multiclass(model, val_loader, num_classes=3)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    plot_confusion_matrix(y_true, y_pred, class_names=iris['target_names'])
