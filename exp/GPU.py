import os
import numpy as np
from simplicial_complex.data_loading import *
from mp.models import *
import torch
from tqdm import tqdm
from sklearn import metrics as met
from collections import Counter
import random
from root_dir import ROOT_DIR
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import random_split, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
import time
import multiprocessing

# 固定随机种子
seed = 43
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


# 优化后的焦点损失函数
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=1.5, smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        targets = targets.long()
        num_classes = inputs.size(1)
        true_dist = torch.zeros_like(inputs)
        true_dist.scatter_(1, targets.unsqueeze(1), 1)
        true_dist = true_dist * (1 - self.smoothing) + self.smoothing / num_classes
        log_prob = F.log_softmax(inputs, dim=1)
        ce_loss = - (true_dist * log_prob).sum(dim=1)
        prob = torch.exp(log_prob)
        p_t = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        modulating_factor = (1 - p_t) ** self.gamma
        focal_loss = modulating_factor * ce_loss
        return self.alpha * focal_loss.mean()


# 计算指标的函数
def calculate_auc_pr(y_true, y_prob):
    precision, recall, _ = met.precision_recall_curve(y_true, y_prob)
    return met.auc(recall, precision)


def weighted_accuracy(y_true, y_pred):
    class_counts = Counter(y_true)
    total = len(y_true)
    acc0 = (y_pred[y_true == 0] == 0).sum() / class_counts[0] if class_counts[0] > 0 else 0
    acc1 = (y_pred[y_true == 1] == 1).sum() / class_counts[1] if class_counts[1] > 0 else 0
    weighted_acc = (acc0 * class_counts[0] + acc1 * class_counts[1]) / total
    return weighted_acc, acc0, acc1


# 梯度噪声增强
def add_gradient_noise(model, scale=0.005):
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * scale
            param.grad += noise


if __name__ == '__main__':
    # 输入参数
    dataset = 'email-Enron'
    max_dim = 2
    lr = 0.0001
    hidden = 128
    num_layers = 5
    num_input_features = 1
    num_hops = 1
    iteration = 200
    batch_size = 32
    patience = 15

    # 创建结果目录
    result_folder = os.path.join(ROOT_DIR, 'exp', 'results', 'enhanced_imbalanced_exp')
    os.makedirs(result_folder, exist_ok=True)
    filename = os.path.join(result_folder, 'results.txt')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据集
    print("Loading datasets...")
    start_time = time.time()
    test_dataset = load_dataset(name=dataset, split='test', num_hops=num_hops, max_dim=2, init_method='sum')
    train_dataset = load_dataset(name=dataset, split='train', num_hops=num_hops, max_dim=2, init_method='sum')
    print(f"Datasets loaded in {time.time() - start_time:.2f} seconds")

    # 创建验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # 提前计算采样器权重
    train_labels = [data.y.item() for data in train_subset]
    class_counts = Counter(train_labels)
    weights = torch.tensor([1.0 / class_counts[label] for label in train_labels], dtype=torch.float)
    train_sampler = WeightedRandomSampler(weights, len(train_subset) * 2, replacement=True)

    val_labels = [data.y.item() for data in val_subset]
    class_counts_val = Counter(val_labels)
    weights_val = torch.tensor([1.0 / class_counts_val[label] for label in val_labels], dtype=torch.float)
    val_sampler = WeightedRandomSampler(weights_val, len(val_subset) * 2, replacement=True)

    # 创建数据加载器（优化CPU多核处理）
    num_workers = min(4, multiprocessing.cpu_count())  # 安全地获取CPU核心数
    print(f"Using {num_workers} workers for data loading")

    # 数据加载器配置（Windows兼容）
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, sampler=val_sampler,
                            num_workers=num_workers, persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, persistent_workers=num_workers > 0)

    # 初始化模型
    model = SparseSIN(num_input_features=num_input_features, num_layers=num_layers,
                      num_classes=2, hidden=hidden, max_dim=max_dim)
    model.to(device)

    # 使用AdamW优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # OneCycle学习率调度
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.001,
        total_steps=len(train_loader) * iteration,
        anneal_strategy='cos'
    )

    # 使用焦点损失函数
    criterion = FocalLoss(alpha=0.8, gamma=1.5, smoothing=0.05).to(device)

    # 早停机制
    best_auc_pr = 0.0
    patience_counter = 0
    best_model_weights = None

    # 指标记录
    metrics = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_weighted_acc': [],
        'val_auc_pr': [],
        'val_acc_0': [],
        'val_acc_1': []
    }

    # 训练循环
    print("Starting training...")
    for epoch in range(1, iteration + 1):
        epoch_start = time.time()
        metrics['epoch'].append(epoch)

        # ===== 训练阶段 =====
        model.train()
        epoch_train_loss = 0.0
        train_steps = 0

        # 使用tqdm显示进度条
        train_iterator = tqdm(train_loader, desc=f'Epoch {epoch} Training', leave=False)
        for step, data in enumerate(train_iterator):
            optimizer.zero_grad()
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()

            # 每5步添加一次梯度噪声
            if step % 5 == 0:
                add_gradient_noise(model, scale=0.005)

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            epoch_train_loss += loss.item() * len(data.y)
            train_steps += 1

        avg_train_loss = epoch_train_loss / len(train_subset)
        metrics['train_loss'].append(avg_train_loss)

        # ===== 验证阶段 =====
        model.eval()
        epoch_val_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            val_iterator = tqdm(val_loader, desc=f'Epoch {epoch} Validation', leave=False)
            for data in val_iterator:
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.y)
                epoch_val_loss += loss.item() * len(data.y)

                # 获取预测结果
                probs = F.softmax(out, dim=1)
                preds = out.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())

        avg_val_loss = epoch_val_loss / len(val_subset)
        metrics['val_loss'].append(avg_val_loss)

        # 计算核心指标
        weighted_acc, acc0, acc1 = weighted_accuracy(np.array(all_labels), np.array(all_preds))
        auc_pr = calculate_auc_pr(all_labels, all_probs)

        metrics['val_weighted_acc'].append(weighted_acc)
        metrics['val_auc_pr'].append(auc_pr)
        metrics['val_acc_0'].append(acc0)
        metrics['val_acc_1'].append(acc1)

        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch}/{iteration} - Time: {epoch_time:.2f}s')
        print(f'Train loss = {avg_train_loss:.4f}')
        print(f'Val: Loss={avg_val_loss:.4f}, Weighted Acc={weighted_acc:.4f}, '
              f'AUC-PR={auc_pr:.4f}, Acc0={acc0:.4f}, Acc1={acc1:.4f}')

        # 早停检查
        if auc_pr > best_auc_pr:
            best_auc_pr = auc_pr
            patience_counter = 0
            best_model_weights = model.state_dict()
            torch.save(model.state_dict(), os.path.join(result_folder, f'best_model_epoch{epoch}.pth'))
            print(f"New best AUC-PR: {auc_pr:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # 加载最佳模型
    if best_model_weights:
        model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), os.path.join(result_folder, 'final_model.pth'))
    model.eval()

    # ===== 最终测试评估 =====
    test_preds = []
    test_probs = []
    test_labels = []

    with torch.no_grad():
        test_iterator = tqdm(test_loader, desc='Testing', leave=False)
        for data in test_iterator:
            data = data.to(device)
            out = model(data)
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_probs.extend(probs[:, 1].cpu().numpy())
            test_labels.extend(data.y.cpu().numpy())

    # 计算测试集核心指标
    test_weighted_acc, test_acc0, test_acc1 = weighted_accuracy(np.array(test_labels), np.array(test_preds))
    test_auc_pr = calculate_auc_pr(test_labels, test_probs)
    test_f1 = met.f1_score(test_labels, test_preds)
    test_recall = met.recall_score(test_labels, test_preds)
    test_precision = met.precision_score(test_labels, test_preds)

    print("\n===== Final Test Results =====")
    print(f"Weighted Accuracy: {test_weighted_acc:.4f}")
    print(f"AUC-PR: {test_auc_pr:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"Class 0 Accuracy: {test_acc0:.4f}")
    print(f"Class 1 Accuracy: {test_acc1:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"Precision: {test_precision:.4f}")

    # 保存结果
    with open(filename, 'w') as f:
        f.write("===== Training Metrics =====\n")
        for key in metrics:
            f.write(f"{key}: {metrics[key]}\n")

        f.write("\n===== Final Test Results =====\n")
        f.write(f"Weighted Accuracy: {test_weighted_acc:.4f}\n")
        f.write(f"AUC-PR: {test_auc_pr:.4f}\n")
        f.write(f"F1 Score: {test_f1:.4f}\n")
        f.write(f"Recall: {test_recall:.4f}\n")
        f.write(f"Precision: {test_precision:.4f}\n")
        f.write(f"Class 0 Accuracy: {test_acc0:.4f}\n")
        f.write(f"Class 1 Accuracy: {test_acc1:.4f}\n")

    # 可视化
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(metrics['epoch'], metrics['train_loss'], 'b-', label='Train Loss')
    plt.plot(metrics['epoch'], metrics['val_loss'], 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(2, 2, 2)
    plt.plot(metrics['epoch'], metrics['val_auc_pr'], 'm-', label='AUC-PR')
    plt.xlabel('Epoch')
    plt.ylabel('AUC-PR')
    plt.legend()
    plt.title('Validation AUC-PR')

    plt.subplot(2, 2, 3)
    plt.plot(metrics['epoch'], metrics['val_acc_0'], 'b-', label='Class 0 Acc')
    plt.plot(metrics['epoch'], metrics['val_acc_1'], 'r-', label='Class 1 Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Class Accuracy')

    plt.subplot(2, 2, 4)
    plt.plot(metrics['epoch'], metrics['val_weighted_acc'], 'g-', label='Weighted Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Weighted Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, 'training_metrics.png'))
    plt.close()

    # 绘制PR曲线
    precision, recall, _ = met.precision_recall_curve(test_labels, test_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR Curve (AUC = {test_auc_pr:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_folder, 'test_pr_curve.png'))
    plt.close()

    # 绘制混淆矩阵
    cm = met.confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Class 0', 'Class 1'])
    plt.yticks(tick_marks, ['Class 0', 'Class 1'])

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, 'confusion_matrix.png'))
    plt.close()

    print(f"All results saved to: {result_folder}")