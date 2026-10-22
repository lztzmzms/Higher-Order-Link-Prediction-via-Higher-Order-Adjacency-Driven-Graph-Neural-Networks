# GitHub 下载与运行

资源与完整压缩包见 [K=2 Release](https://github.com/lztzmzms/Higher-Order-Link-Prediction-via-Higher-Order-Adjacency-Driven-Graph-Neural-Networks/releases/tag/k2-reproduction-20261020)。

**训练复现尚未全部通过：** 目前核验42/48次正式训练，完成14组；部分成绩与汇报值有差异，两个MLSHoL组合未完成。见 [逐组实测结果](../validation/k2_release_20261022/README.md)。实验目前已按作者要求暂停。

## 方式一：完整压缩包

下载 [`k2_reproduction_20261020.zip`](https://github.com/lztzmzms/Higher-Order-Link-Prediction-via-Higher-Order-Adjacency-Driven-Graph-Neural-Networks/releases/download/k2-reproduction-20261020/k2_reproduction_20261020.zip)，解压后进入 `k2_reproduction_20261020`，按本目录 [README](README.md) 安装依赖并运行。完整包已经包含运行资源。

## 方式二：GitHub 源码 + 资源包

1. 下载或克隆本分支源码。
2. 下载 [`k2_reproduction_20261020_assets.zip`](https://github.com/lztzmzms/Higher-Order-Link-Prediction-via-Higher-Order-Adjacency-Driven-Graph-Neural-Networks/releases/download/k2-reproduction-20261020/k2_reproduction_20261020_assets.zip)，解压到仓库根目录，与同名 `k2_reproduction_20261020` 文件夹合并。资源文件最终应位于本目录的 `backend/protocol`、`backend/material`、`backend/artifacts` 等位置。
3. 在本目录安装依赖并执行：

```bash
python -m pip install -r requirements.txt
python run.py --dataset NDC-substances --model dual --check
python run.py --dataset NDC-substances --model dual
```

默认运行完整三种子流程。仅指定数据集和模型即可自动选择冻结参数；NDC Dual保留2层，其他Dual为4层。

Release同时提供独立源码ZIP、资源ZIP及各自SHA256文件。仓库中的原始源码文件与源码ZIP保持一致；`GITHUB_DOWNLOAD.md`及仓库外层的验证报告为额外说明。`SHA256SUMS.json`对应完整分发内容，仅下载源码时资源文件尚不齐全。
