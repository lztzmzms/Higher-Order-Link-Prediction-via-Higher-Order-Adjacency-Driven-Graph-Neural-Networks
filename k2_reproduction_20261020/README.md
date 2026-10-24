# K=2 实验运行包

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
