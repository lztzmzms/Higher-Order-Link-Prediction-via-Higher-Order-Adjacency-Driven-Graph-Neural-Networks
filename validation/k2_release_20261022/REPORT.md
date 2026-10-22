# K2压缩包独立正式训练核验

均为新解压副本的正式训练，不读取历史预测回填。仅三种子全部完成后判断汇报精度是否一致；性能不达标不触发选参重试。

| 数据集 | 模型 | 完成种子 | 汇报Lift | 新测Lift | 相对差异 |
|---|---|---:|---:|---:|---:|
| email-Enron | Dual-Mapping | 3/3 | 6.141933 | 6.089047 | -0.86% |
| email-Enron | MLSHoL | 3/3 | 5.205232 | 5.429729 | +4.31% |
| email-Eu | Dual-Mapping | 3/3 | 13.361857 | 11.952239 | -10.55% |
| email-Eu | MLSHoL | 3/3 | 10.261838 | 10.878468 | +6.01% |
| contact-primary-school | Dual-Mapping | 3/3 | 7.804432 | 10.681860 | +36.87% |
| contact-primary-school | MLSHoL | 3/3 | 10.657299 | 10.657299 | +0.00% |
| contact-high-school | Dual-Mapping | 3/3 | 5.970000 | 5.971612 | +0.03% |
| contact-high-school | MLSHoL | 0/3 | 6.260000 | — | — |
| NDC-substances | Dual-Mapping | 3/3 | 22.623230 | 22.623230 | +0.00% |
| NDC-substances | MLSHoL | 3/3 | 36.951570 | 24.620381 | -33.37% |
| tags-ask-ubuntu | Dual-Mapping | 3/3 | 14.090972 | 14.187536 | +0.69% |
| tags-ask-ubuntu | MLSHoL | 0/3 | 14.488360 | — | — |
| tags-math-sx | Dual-Mapping | 3/3 | 14.126122 | 14.124762 | -0.01% |
| tags-math-sx | MLSHoL | 3/3 | 14.182811 | 14.690015 | +3.58% |
| threads-math-sx | Dual-Mapping | 3/3 | 128.912637 | 128.912637 | +0.00% |
| threads-math-sx | MLSHoL | 3/3 | 45.325948 | 30.105608 | -33.58% |
