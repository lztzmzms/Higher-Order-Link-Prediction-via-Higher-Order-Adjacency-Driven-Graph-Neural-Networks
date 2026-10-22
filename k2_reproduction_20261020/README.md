# K=2 Higher-Order Link Prediction — Reproduction Package

本包提供 **Dual-Mapping / MLSHoL 的完整 K=2 预测流程**。只需指定数据集与模型，程序自动选择时间协议、固定候选集、训练参数、随机种子、RW、历史分支和分数融合，并输出逐种子成绩及均值/样本标准差。

## 下载运行资源

完整代码和运行资源可从 [Release](https://github.com/lztzmzms/Higher-Order-Link-Prediction-via-Higher-Order-Adjacency-Driven-Graph-Neural-Networks/releases/tag/k2-reproduction-20261020) 下载。

- **直接运行：** 下载并解压 [`k2_reproduction_20261020.zip`](https://github.com/lztzmzms/Higher-Order-Link-Prediction-via-Higher-Order-Adjacency-Driven-Graph-Neural-Networks/releases/download/k2-reproduction-20261020/k2_reproduction_20261020.zip)，进入解压目录后安装依赖。
- **使用GitHub源码：** 下载 [`k2_reproduction_20261020_assets.zip`](https://github.com/lztzmzms/Higher-Order-Link-Prediction-via-Higher-Order-Adjacency-Driven-Graph-Neural-Networks/releases/download/k2-reproduction-20261020/k2_reproduction_20261020_assets.zip)，解压到仓库根目录，与同名 `k2_reproduction_20261020` 文件夹合并。资源文件最终应位于本目录的 `backend/protocol`、`backend/material` 和 `backend/artifacts` 下。
- 单独下载源码ZIP时，将源码包与资源包解压到同一个父目录即可合并。

## 快速运行

建议解压到较短的路径，例如 `D:/k2`。需要 Python 3.10–3.12；本地执行检查使用 PyTorch 2.4.1+cu118 和可用的 NVIDIA GPU。先安装与机器相匹配的 PyTorch，再安装其余依赖：

```bash
python -m pip install -r requirements.txt
python run.py --dataset tags-ask-ubuntu --model dual
python run.py --dataset NDC-substances --model mlshol
```

以上命令默认执行三种子的正式训练、完整流程推理和成绩汇总。自动选择GPU，未检测到GPU时使用CPU；CPU运行可能较慢。中断后可用相同命令继续，已完成输出会复核后保留。不要同时在同一解压目录启动多个训练。

模型名称支持 `dual` / `Dual-Mapping`、`mlshol` / `MLSHoL`；NDC亦支持 `--dataset ndc`。

## 数据集与固定设置

| 数据集 | Dual GNN层数 | 模型种子 | 评价候选/正例 | 推理候选上下文 |
|---|---:|---|---:|---:|
| email-Enron | 4 | 41,42,43 | 2574/119 | 2574 |
| email-Eu | 4 | 41,42,43 | 3222/31 | 3222 |
| contact-primary-school | 4 | 41,42,43 | 4184/22 | 4184 |
| contact-high-school | 4 | 41,42,43 | 25056/201 | 25056 |
| NDC-substances | **2** | 71,72,73 | 1750/23 | **6000** |
| tags-ask-ubuntu | 4 | 41,42,43 | 4414/36 | 4414 |
| tags-math-sx | 4 | 41,42,43 | 2000/26 | 2000 |
| threads-math-sx | 4 | 206,207,208 | 28872/27 | **30000** |

NDC为经作者确认的两层例外；其余七数据集的Dual采用四层。MLSHoL保留各自来源配置，没有统一修改层数。所有实际运行设置均可用 `--check` 查看，完整配置在 `backend/configs` 和 `backend/PLAN.json`。

NDC及threads须先在原完整推理上下文中预测，再评价固定子集；不能把子集直接替换推理图。多个队列存在使用标签或模型表现选样的历史，结果限于这些固定队列。除高中保留来源的原始文件顺序协议外，其他七个队列使用冻结的时间戳顺序。高中不能描述为已重新按真实时间排序的复现。

## 完整预测流程

- email-Eu、tags-ask-ubuntu：保留80%历史秩分数＋20%模型秩分数的固定融合。
- email-Enron、contact-primary-school、tags-math-sx：保留固定历史分支和训练的结构残差，参数由配置自动选取。
- NDC、threads：保留原生模型、历史频次特征/权重与锁定RW数组；无外部历史评分融合。
- contact-high-school：保留冻结来源的模型、评分先验及融合设置。

`RW concat`表示模型输入中的特征拼接，与输出层历史分数融合不同。默认运行产生完整流程成绩，不是纯神经分数消融。

## 检查与复核

```bash
# 只检查该组合的数据、配置哈希和候选集，不训练
python run.py --dataset NDC-substances --model dual --check

# 单轮执行检查；不作为正式性能
python run.py --dataset NDC-substances --model dual --smoke

# 重新计算本目录已完成训练输出的指标
python run.py --dataset NDC-substances --model dual --verify

# 复算随包历史预测；不重新训练
python run.py --dataset NDC-substances --model dual --replay

# 可选：仅运行冻结种子中的一个，或指定设备
python run.py --dataset tags-ask-ubuntu --model mlshol --seeds 41 --device cuda:0
```

`--replay`支持 tags-ask-ubuntu 与 NDC-substances 的两个模型，共12份随包历史预测。此选项仅重新计算已有预测的指标；正式训练使用不带 `--replay` 的默认命令。

指标为 `AP = sklearn.metrics.average_precision_score`，`Lift = AP / 评价候选集正例率`。均值逐种子计算，标准差为样本标准差 `ddof=1`。单种子运行没有跨种子标准差。未完成任务保留实际完成数，不填入论文目标值。

## 输出位置

```text
results/<dataset>/<model>/summary.json   # 完成状态、逐种子结果、AP/Lift均值与SD
results/<dataset>/<model>/per_seed.csv
results/logs/                          # 每种子训练日志
backend/results/jobs/<job>/<seed>/      # 配置、原始预测、检查点及资源记录
```

## 目录与GitHub分发

```text
run.py                         # 两参数入口
backend/code/                  # 原模型与依赖实现
backend/configs/               # 各数据集完整流程参数
backend/material/              # 冻结历史分支输入
backend/protocol/              # 数据、候选、标签和时间协议
backend/artifacts/rw/           # 必需的锁定RW输入
backend/artifacts/locked_geometry/
reference_predictions/         # 可独立复算的历史输出
supplementary_sources/         # 启发式及K2–K4结构实验源码附录
```

完整ZIP包含运行资源，可直接解压。另提供 `*_source.zip` 与 `*_assets.zip`：源码包用于GitHub仓库；资源包用于单独分发，解压到相同父目录后与源码包合并。只有源码包时不能直接训练。`SHA256SUMS.json`记录完整分发文件的校验值，ZIP另有SHA256文件。

源码附录包含原始启发式及K2–K4结构实验实现，不运行K3模型实验。附录不是本入口的16组模型任务，其数据和实验输出依赖见附录README，不应将附录源码存在解释为整套结构实验均已重新执行。

本包没有替作者指定新许可证；公开前应核对原始代码和数据的许可及引用要求。
