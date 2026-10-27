# Higher-Order Link Prediction

八个数据集的 K=2 Dual-Mapping / MLSHoL 实验在原 [k2_reproduction_20261020](k2_reproduction_20261020/README.md) 目录中运行；两个模型均开启 RW，数据和必要配置已包含在源码中。

```bash
cd k2_reproduction_20261020
python -m pip install -r requirements.txt
python reproduce.py --dataset tags-math-sx
```

八个数据集使用同一接口：`email-Enron`、`email-Eu`、`contact-high-school`、`contact-primary-school`、`tags-ask-ubuntu`、`NDC-substances`、`tags-math-sx`、`threads-math-sx`。每次默认运行两个模型及各自三个种子；也可加 `--model dual` 或 `--model mlshol`。

- [完整运行命令、实验协议及输出说明](k2_reproduction_20261020/README.md)
- [八个数据集的配置](k2_reproduction_20261020/configs/)
- [逐种子 AP / Lift](k2_reproduction_20261020/results/seed_results.csv)
- [三种子汇总](k2_reproduction_20261020/results/summary.json)
- [原 AP 启发式及 K2–K4 task-core 补充实验](k2_reproduction_20261020/supplementary_sources/README.md)
