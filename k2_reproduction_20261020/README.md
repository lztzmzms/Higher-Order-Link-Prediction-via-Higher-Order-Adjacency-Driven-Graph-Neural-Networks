# K=2 higher-order link prediction

八个数据集的 Dual-Mapping / MLSHoL 可运行复现。两个模型的 RW 均开启；各数据集保留自己的数据划分、输入、训练配置和评估候选。

Python 3.12，依赖版本见 requirements.txt。在本目录执行：

```bash
python -m pip install -r requirements.txt
python reproduce.py --dataset email-Enron
python reproduce.py --dataset email-Eu
python reproduce.py --dataset contact-high-school
python reproduce.py --dataset contact-primary-school
python reproduce.py --dataset tags-ask-ubuntu
python reproduce.py --dataset NDC-substances
python reproduce.py --dataset tags-math-sx
python reproduce.py --dataset threads-math-sx
```
每个命令默认运行两个模型及各自原始三个种子。可加 `--model dual` 或 `--model mlshol` 单独运行，`--device cpu` 指定 CPU，`--check` 仅核对配置及真实输入。原 `python run.py --dataset ... --model ...` 入口也保留，调用同一实现。

数据、固定历史输入和原始邻居抽样结果已附在 `data/`；压缩输入首次运行时自动解压到 `.cache/`，无需下载旧实验包。训练与评估不读取预存的最终预测或成绩，结果来自真实计算。CPU 运行可能较慢。

结果只有 `results/seed_results.csv`（dataset, model, seed, AP, Lift）和 `results/summary.json`（dataset, model, seeds, AP_mean, AP_std, Lift_mean）。AP 使用 sklearn average precision；Lift=AP/同一候选集的正例比例。AP_std 为各配置原始三种子的样本标准差（ddof=1），未完成的种子组不生成汇总。原生模型内部为训练选模临时生成的文件会自动清理，不作为结果交付。

随附结果覆盖八个数据集、两个模型各三个种子，共 48 行真实运行结果和 16 组汇总。threads-math 的 MLSHoL 种子 207、208 由核对过的原同配置真实预测重算，种子 206 与本版重跑结果完全一致。

执行 `python evaluate.py --compare` 可从已有结果重新生成该说明；不会训练模型或增加正式结果字段。

其他实验仍位于 [supplementary_sources/](supplementary_sources/README.md)，与本次 K=2 替换分开保留。
