# Higher-Order Link Prediction

本仓库提供 8 个数据集的 K=2 Dual-Mapping / MLSHoL 实验入口。

## 下载与运行

- [完整运行包](https://github.com/lztzmzms/Higher-Order-Link-Prediction-via-Higher-Order-Adjacency-Driven-Graph-Neural-Networks/releases/download/k2-reproduction-20261020/k2_reproduction_20261020.zip)：包含代码及必要资源，解压后进入 `k2_reproduction_20261020`。
- 使用本仓库源码时，下载[配套资源包](https://github.com/lztzmzms/Higher-Order-Link-Prediction-via-Higher-Order-Adjacency-Driven-Graph-Neural-Networks/releases/download/k2-reproduction-20261020/k2_reproduction_20261020_assets.zip)，解压到仓库根目录，与同名 `k2_reproduction_20261020` 文件夹合并。
- [运行说明](k2_reproduction_20261020/README.md) · [数据集与模型配置](k2_reproduction_20261020/backend/PLAN.json)

先安装适合设备的 PyTorch（参考版本 2.4.1），然后执行：

```bash
cd k2_reproduction_20261020
python -m pip install -r requirements.txt
python run.py --dataset tags-ask-ubuntu --model dual
python run.py --dataset NDC-substances --model mlshol
```

只需指定数据集和模型，入口会选择相应配置并默认运行三个种子。模型参数支持 `dual` / `Dual-Mapping` 和 `mlshol` / `MLSHoL`。

数据集：`email-Enron`、`email-Eu`、`contact-primary-school`、`contact-high-school`、`NDC-substances`、`tags-ask-ubuntu`、`tags-math-sx`、`threads-math-sx`。

NDC 的 Dual-Mapping 使用 2 层，其他数据集的 Dual-Mapping 使用 4 层。结果写入 `results/<dataset>/<model>/`，日志写入 `results/logs/`。不要在同一目录并发启动多个训练。

## 检查与校验

```bash
python run.py --dataset NDC-substances --model dual --check
python run.py --dataset NDC-substances --model dual --verify
```

Release 同时提供完整包、源码包、资源包及各自的 SHA256 文件。源码包与资源包包含共同根目录，解压到同一父目录即可合并；`SHA256SUMS.json` 记录包内文件校验值。
