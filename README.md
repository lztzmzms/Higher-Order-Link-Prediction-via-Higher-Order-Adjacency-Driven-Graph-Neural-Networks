# Higher-Order Link Prediction

本仓库提供 Dual-Mapping / MLSHoL 的 K=2 完整预测流程，以及启发式指标、K2–K4结构实验源码。

## 下载与运行

- [完整代码和运行资源](https://github.com/lztzmzms/Higher-Order-Link-Prediction-via-Higher-Order-Adjacency-Driven-Graph-Neural-Networks/releases/tag/k2-reproduction-20261020)
- [详细README](k2_reproduction_20261020/README.md)
- [数据集与模型配置](k2_reproduction_20261020/backend/PLAN.json)

使用GitHub源码时，先下载 Release 中的 `k2_reproduction_20261020_assets.zip`，解压到仓库根目录，与同名文件夹合并；完整ZIP已包含所需资源。

```bash
cd k2_reproduction_20261020
# 先安装与设备匹配的PyTorch，再安装其余依赖
python -m pip install -r requirements.txt
python run.py --dataset tags-ask-ubuntu --model dual
python run.py --dataset NDC-substances --model mlshol
```

只需指定数据集和模型，入口会自动选择冻结配置、候选集、三种子、历史分支和融合，并输出成绩汇总。

## 固定设置

支持 email-Enron、email-Eu、contact-primary-school、contact-high-school、NDC-substances、tags-ask-ubuntu、tags-math-sx、threads-math-sx 八个数据集。

- Dual-Mapping：GNN层数为4层。
- MLSHoL：沿用各数据集的冻结配置。
- `--check`检查输入和配置，`--verify`复核已有训练输出，`--replay`复算随包历史预测。

## 代码目录

```text
k2_reproduction_20261020/
  run.py                    # 数据集 + 模型入口
  requirements.txt
  backend/code/             # 模型实现
  backend/configs/          # 冻结实验配置
  reference_predictions/    # 历史预测
  supplementary_sources/    # 启发式及K2–K4结构实验源码
```

K3模型实验不在本次统一入口范围内；结构实验附录的运行方式与数据依赖见其目录内的README。
