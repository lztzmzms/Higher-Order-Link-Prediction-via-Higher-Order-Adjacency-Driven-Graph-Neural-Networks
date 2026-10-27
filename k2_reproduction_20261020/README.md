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

随附结果仅包含实际完成的运行。threads-math 的 MLSHoL 保留已完成的种子 206，不提供未完成种子的成绩或三种子汇总。

|数据集|实际协议与配置|种子|
|---|---|---|
|email-Enron、contact-primary-school|最终选定 F；完整时间轴 80%→100% 测试|9101、9102、9103|
|email-Eu、tags-ask-ubuntu|固定预算 F 迁移；修复的训练 H60，原测试 H80 和原候选|9101、9102、9103|
|contact-high-school|原选定配方，修复 H60 方向选择；严格时间戳切点|9101、9102、9103|
|tags-math-sx|同一固定 F 联合方案；200 步历史预训练，Dual 联合 25 步、MLSHoL 联合 50 步，原固定 2,000 个测试候选|41、42、43|
|NDC-substances|原生模型；原共享训练/验证划分，完整 6,000 候选上下文中固定评估 1,750 个候选|71、72、73|
|threads-math-sx|原生模型；原共享训练/验证划分，完整 30,000 候选上下文中固定评估 28,872 个候选|206、207、208|

采用 F 的六个数据集使用历史 H、真实高阶结构编码器和一个神经预测头。历史 H 的四通道为原值、并列平均分位数、分位数 logit/6、常数 1；最终神经训练标签来自 60%→80%，测试标签来自原始末尾 80%→100%。Enron 的 35%→55%、小学的 40%→60% 是此前选择训练步数的历史验证窗口，不是最终测试划分。

F 的 Dual-Mapping 使用原两步候选对偶图 RW 扩散特征；F 的 MLSHoL 在 SparseSIN 后拼接同源 RW 投影。原指定 F 实验的 MLSHoL 关闭 RW，本版按要求开启，因此其旧成绩不作为本版成绩。tags-math 采用这一方案完成本地测试；原历史 MLP 的配置、输入及重建代码一并保留。

NDC、threads-math 保留原生架构、损失、优化器、RW 和模型选择步骤：NDC 的 Dual 为 2 层，threads-math 的 Dual 为 4 层，两者 MLSHoL 为原 4 层、hidden=6 的配置。它们使用原时间边界与固定测试掩码，不套用 F 训练流程，不改变候选来匹配成绩。三新增数据集的原始参数保存在各自 `configs/` 文件中；tags-math 的旧参数位于 `original_configurations`。

高中、Eu、Ubuntu 继承了最终标签参与节点或配方选择的偏差；原最终窗口此前也已查看。这些固定窗口结果属于探索性复现，高中尤其不能解释为独立、无偏的未来测试。新增数据集也沿用其固定原协议，不据本次测试分数重新筛选数据或调参。

其他实验仍位于 [supplementary_sources/](supplementary_sources/README.md)，与本次 K=2 替换分开保留。
