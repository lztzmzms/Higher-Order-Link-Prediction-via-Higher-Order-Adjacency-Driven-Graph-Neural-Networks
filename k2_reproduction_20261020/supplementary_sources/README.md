# 启发式与 K2–K4 task-core

在主包目录安装依赖：

```bash
python -m pip install -r supplementary_sources/requirements.txt
```

## AP 版启发式指标

```bash
python supplementary_sources/heuristics/run.py --dataset email-Enron --check
python supplementary_sources/heuristics/run.py --dataset email-Enron --method Simplicial_CN
```

提供 Simplicial_CN、RA、Local Path、katz、AA、PA、Global AA 和 Global RA，另支持 Degree。AP 使用 `sklearn.metrics.average_precision_score`，Lift 为 AP 除以评价候选的正例率。

默认按输入事件行序做 80%/20% 切分并评价全部开放三角形。可用 `--raw-dir` 指定完整原始数据的三个文本文件目录；本包也包含 task-core 的 canonical 数据，位于 `task_core_server_bundle_v2/dataset/raw_data/<dataset>/`。

方法和输入选项见 [启发式说明](heuristics/README.md)。

## K2–K4 task-core

```bash
python supplementary_sources/task_core_server_bundle_v2/run.py --check
python supplementary_sources/task_core_server_bundle_v2/run.py --order 2 --dataset email-Eu
python supplementary_sources/task_core_server_bundle_v2/run.py --order 3 --dataset tags-ask-ubuntu
python supplementary_sources/task_core_server_bundle_v2/run.py --order 4 --dataset email-Eu
```

支持各阶的 task-core 比较；K4 另提供 `--experiment retention` 和 `--experiment perturbation`。不指定数据集时运行该阶配置面板。必要原始事件、特征、划分与模型输入随资源包提供，运行结果写入组件内的 `output/`。

具体数据集、设置和运行参数见 [task-core 说明](task_core_server_bundle_v2/README.md)。

## K4 扩展数据集

`k4_server_benchmark_expansion_bundle_v1/` 提供 NDC-classes、congress-bills 和 DAWN 的校验、候选枚举、支持集检查、特征与 MLP、汇总阶段。

```bash
cd supplementary_sources/k4_server_benchmark_expansion_bundle_v1
python scripts/k4_server_pipeline.py s0 --dry-run
python scripts/k4_server_pipeline.py selftest
```

完整阶段命令及资源要求见 [K4 扩展说明](k4_server_benchmark_expansion_bundle_v1/README_SERVER.md)。
