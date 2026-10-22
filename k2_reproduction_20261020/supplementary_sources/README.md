# 启发式与K2–K4结构实验源码附录

本目录保留初始打包范围内的原始启发式、任务核心/结构缩减、精确保留与外围扰动代码。K3模型训练不在本次入口范围内。

- `heuristics/`：原 HOLP_Heuristics Python源码及数据读取工具，`root_dir.py`相对定位。原入口 `Heuristics_using_HodgeLaplacian/saving.py`先构造预处理对象，`main.py`读取它们并评价；这两个原文件仍以内部 `dataset_name`变量选择数据集。原始数据需要按其相对目录提供。原指标使用梯形PR-AUC，应与模型AP区分。
- `heuristic_reference_tables/`：当前论文两张启发式表的原始数值，不表示源码已重新生成全部表值。
- `task_core_server_bundle_v2/`：K2/K3严格T2与任务核心实验的来源源码、配置和数据要求。
- `k4_server_benchmark_expansion_bundle_v1/`：K4扩展数据集来源源码、配置和清单。
- `formal_entrypoints/`：工作区后续正式实验入口；依赖来源项目的结果目录与冻结数据，不是独立一键任务。

该附录为源码档案，未附齐所有结构实验中间结果、原始数据及模型预测，不能独立复现K2–K4全部结构实验表格。模型训练所需资源位于主包backend中。本次已验证的一键接口仅为主包的K2八数据集×两模型。
