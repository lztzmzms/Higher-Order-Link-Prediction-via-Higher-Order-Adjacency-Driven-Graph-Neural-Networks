# 模型、启发式与 task-core 实验

使用 Python 3.10–3.12，安装适合本机的 PyTorch（参考环境版本为 2.4.1），然后执行：

```bash
python -m pip install -r requirements.txt
python run.py --dataset tags-ask-ubuntu --model dual
python run.py --dataset NDC-substances --model mlshol
```

模型参数支持 `dual` / `Dual-Mapping` 和 `mlshol` / `MLSHoL`。

数据集：`email-Enron`、`email-Eu`、`contact-primary-school`、`contact-high-school`、`NDC-substances`、`tags-ask-ubuntu`、`tags-math-sx`、`threads-math-sx`。

默认运行该组合的三个种子。其他用法：

```bash
python run.py --dataset NDC-substances --model dual --check
python run.py --dataset NDC-substances --model dual --verify
python run.py --dataset tags-ask-ubuntu --model mlshol --seeds 41
python run.py --help
```

结果位于 `results/<dataset>/<model>/`，日志位于 `results/logs/`。不要在同一目录并发启动多个训练。

补充实验位于 [`supplementary_sources/`](supplementary_sources/README.md)：

- [AP 版启发式指标](supplementary_sources/heuristics/README.md)。
- [K2–K4 task-core](supplementary_sources/task_core_server_bundle_v2/README.md)。
- [K4 扩展数据集实验](supplementary_sources/k4_server_benchmark_expansion_bundle_v1/README_SERVER.md)。

```bash
python -m pip install -r supplementary_sources/requirements.txt
python supplementary_sources/heuristics/run.py --dataset email-Enron --method Simplicial_CN
python supplementary_sources/task_core_server_bundle_v2/run.py --order 2 --dataset email-Eu --check
```
