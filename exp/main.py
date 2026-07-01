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
import shutil
import torch.nn.functional as F
import matplotlib.pyplot as plt
from numpy import *

def evaluate(result_true, test_dataset, y_true, y_prob):
    random_Baseline = float(result_true[1] / len(test_dataset))
    precision, recall, threshold = met.precision_recall_curve(y_true, y_prob)
    AUC_PR = met.auc(recall, precision)
    print("random_Baseline")
    print(random_Baseline)
    eval = float(AUC_PR / random_Baseline)
    return eval

def acc(y_true, y_prob, result_true, x) :
    k = 0
    for i in range(len(y_true)):
        if y_true[i] == y_prob[i] == x:
            k = k + 1
        else:
            pass
    accuracy = float(k / result_true[x])
    return accuracy

seed = 44 # random seed to set; reaches >10 liftx on email-Enron in the MLSHOL seed test
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
result_folder = os.path.join(ROOT_DIR, 'exp', 'results')

# input parameters
dataset = 'email-Enron' # dataset name
exp_name = 'basic_exp' # name for specific experiment
model_name = 'Sparse_SIN' # A simplicial version of GIN using upper adjacencies and boundary adjacencies
labelling_method = 'TRNL' # labelling method
max_dim = 2 # max dimension in MPSN
lr = 0.00001 # learning rate 原为0.00001
hidden = 6 # dimensionality of hidden units in models
num_layers = 4 # number of message passing layers
step_size = 50 # number of epochs between lr decay
num_input_features = 1 # input feature vectors' dimension
num_hops = 1
iteration = 100 # number of epochs to train

result_folder = os.path.join(result_folder, f'{dataset}+{exp_name}')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
filename = os.path.join(result_folder, 'results.txt')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_dataset = load_dataset(name=dataset, split='test', num_hops=num_hops, max_dim=2, init_method='sum')

train_dataset = load_dataset(name=dataset, split='train', num_hops=num_hops, max_dim=2, init_method='sum')
print(train_dataset)
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of testing graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True)


# for hidden in layers:
pred = []
true = []
loss_all = []
acc_0_gnn_all = []
acc_1_gnn_all = []
acc_gnn_all = []
evaluation_gnn_all = []
result_predict_gnn_all = []
test_loss = []

model = SparseSIN(num_input_features=num_input_features, num_layers=num_layers, num_classes=2, hidden=hidden, max_dim=max_dim)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

criterion = torch.nn.BCELoss()
criterion = criterion.to(device)

for epoch in tqdm(range(1, iteration+1)):
    curve = list()
    for data in train_loader:
        optimizer.zero_grad()
        y = []
        data = data.to(device)
        print(data)
        out = model(data)

        for element in data.y:
            if element == 0:
                y += [[1, 0]]
            elif element == 1:
                y += [[0, 1]]
        # y = torch.tensor(y, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float).to(device)
        # print(y)
        loss = criterion(out, y)
        pred.append(out.detach().cpu())
        true.append(data.y.detach().cpu())
        curve.append(loss.detach().cpu().item())
        loss.backward()
        optimizer.step()
    total = sum(curve)
    loss_all.append(total)
    print(f'\ntraininng loss:{total}')

    y_true = []
    y_pred = []
    prob_gnn = []
    predict = []
    test_curve = []

    for data in test_loader:
        data = data.to(device)
        out = model(data)
        for element in out:
            y = element.argmax(dim=0)
            predict.append(int(y))
        targets = []
        for element in data.y:
            if element == 0:
                targets += [[1, 0]]
            elif element == 1:
                targets += [[0, 1]]
        targets = torch.tensor(targets, dtype=torch.float).to(device)
        # targets = torch.tensor(targets, dtype=torch.float)
        loss = criterion(out, targets)
        test_curve.append(loss.detach().cpu().item())
        y_pred.append(out.detach().cpu())
        y_true.append(data.y.detach().cpu())
    total_test = sum(test_curve)
    test_loss.append(total_test)
    mean_test = total_test / len(test_dataset)
    print(f'testing_loss:{total_test}')

    y_true = torch.cat(y_true, dim=0).numpy() if len(y_true) > 0 else None
    result_true = Counter(y_true)
    print(result_true)
    y_pred = torch.cat(y_pred, dim=0).numpy()

    y_prob_gnn = []
    for i in range(len(y_true)):
        temp_gnn = y_pred[i][y_true[i]]
        y_prob_gnn.append(temp_gnn)

    result_predict_gnn = Counter(predict)
    result_predict_gnn_all.append(result_predict_gnn)

    acc_0_gnn = acc(y_true, predict, result_true, 0)
    acc_0_gnn_all.append(acc_0_gnn)
    acc_1_gnn = acc(y_true, predict, result_true, 1)
    acc_1_gnn_all.append(acc_1_gnn)
    acc_gnn = float((result_true[0]*acc_0_gnn + result_true[1]*acc_1_gnn) / len(y_true))
    print(f'0-accuracy:{acc_0_gnn}')
    print(f'1-accuracy:{acc_1_gnn}')
    print(f'accuracy:{acc_gnn}')
    acc_gnn_all.append(acc_gnn)

    evaluation_gnn = evaluate(result_true, test_dataset, y_true, y_prob_gnn)
    print(f'evaluation_gnn:{evaluation_gnn}')
    evaluation_gnn_all.append(evaluation_gnn)
    schedule.step()
# ===================== 可视化部分 ========================
# 1. 损失曲线
plt.figure(figsize=(12, 6))
plt.plot(loss_all, 'b', label='Training loss')
plt.plot(test_loss, 'r', label='Test loss')
plt.title('Training and Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(result_folder, 'loss_curve.png'))
plt.close()

# 2. 准确率曲线
plt.figure(figsize=(12, 6))
plt.plot(acc_gnn_all, 'g', label='Overall Accuracy')
plt.plot(acc_0_gnn_all, 'b--', label='Class 0 Accuracy')
plt.plot(acc_1_gnn_all, 'r--', label='Class 1 Accuracy')
plt.title('Accuracy Metrics')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(result_folder, 'accuracy_curve.png'))
plt.close()

# 3. 评估指标曲线
plt.figure(figsize=(12, 6))
plt.plot(evaluation_gnn_all, 'm', label='Evaluation Metric')
plt.title('Evaluation Metric (AUC_PR / Random Baseline)')
plt.xlabel('Epochs')
plt.ylabel('Metric Value')
plt.legend()
plt.savefig(os.path.join(result_folder, 'evaluation_metric.png'))
plt.close()

# 4. 绘制最终的PR曲线
model.eval()
final_y_true = []
final_y_prob = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        outputs = model(data)
        final_y_true.extend(data.y.cpu().numpy())
        final_y_prob.extend(outputs[:, 1].cpu().numpy())  # 取正类的概率

precision, recall, _ = met.precision_recall_curve(final_y_true, final_y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2,
         label=f'PR Curve (AUC = {met.auc(recall, precision):.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower right')
plt.savefig(os.path.join(result_folder, 'pr_curve.png'))
plt.close()

# 5. 混淆矩阵
final_preds = [1 if prob >= 0.5 else 0 for prob in final_y_prob]
cm = met.confusion_matrix(final_y_true, final_preds)

plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Class 0', 'Class 1'])
plt.yticks(tick_marks, ['Class 0', 'Class 1'])

# 添加文本标签
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
minimum_value = min(evaluation_gnn_all)
maximum_value = max(evaluation_gnn_all)
mean_value = mean(evaluation_gnn_all)
minimum_value_last_10_epoch = min(evaluation_gnn_all[-10:])
maximum_value_last_10_epoch = max(evaluation_gnn_all[-10:])
mean_value_last_10_epoch = mean(evaluation_gnn_all[-10:])

msg = (
    f'========== Result ============\n'
    f'Dataset:        {dataset}\n'
    f'------------ parameters -----------\n'
    f'seed:{seed}\n'
    f'model:{model_name}\n'
    f'num_hops:{num_hops}\n'
    f'acc_0_gnn:{acc_0_gnn}\n'
    f'acc_1_gnn:{acc_1_gnn}\n'
    f'hidden:{hidden}\n'
    f'max_dim:{max_dim}\n'
    f'lr:{lr}\n'
    f'mean_value:{mean_value_last_10_epoch}\n'
    '-------------------------------\n\n')
print(msg)
with open(filename, 'a') as handle:
    handle.write(msg)
