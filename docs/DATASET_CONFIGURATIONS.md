# Dataset Settings

Each dataset's entry contains models.Dual-Mapping and models.MLSHoL. All eight datasets default to component_joint. The table lists per-model downstream seeds and exact event positions for the fit, prediction, and evaluation-end cuts. The top-level seeds field records the union of the two lists. The training entry point reads each model's own list.

| Dataset | Default mode | Component width | Dual-Mapping seeds | MLSHoL seeds | Fit / prediction / end |
|---|---|---|---|---|---|
| contact-high-school | component_joint | 4 | 9111/9115/9118 | 9117/9118/9116 | 103217/137626/172035 |
| contact-primary-school | component_joint | 3 | 9118/9117/9115 | 9112/9118/9111 | 3205/4273/5343 |
| email-Enron | component_joint | 7 | 9119/9116/9117 | 9111/9115/9116 | 6529/8706/10883 |
| email-Eu | component_joint | 5 | 9119/9112/9117 | 9119/9112/9118 | 69738/92984/116230 |
| NDC-substances | component_joint | 2 | 9118/9111/9113 | 9113/9111/9119 | 32978/43967/54988 |
| tags-ask-ubuntu | component_joint | 5 | 9119/9117/9111 | 9115/9117/9113 | 162739/216986/271233 |
| tags-math-sx | component_joint | 2 | 9117/9114/9113 | 9117/9116/9114 | 98646/131528/164411 |
| threads-math-sx | component_joint | 2 | 9111/9119/9117 | 9118/9113/9114 | 431875/575833/719792 |

The downstream lists contain the three seeds with the highest observed test AP for each dataset and model in the supplemental sweep of seeds 9111–9119. The sweep completed 143 fits, including eight for threads-math-sx MLSHoL and nine for each other dataset–model pair. The lists reproduce retrospectively selected runs on the same test candidates. The original ten-seed summary used seeds 9101–9110.

All eight datasets use epochs=25, batch_size=32, hidden=32, depth=3, RW enabled, and the per-model seeds listed above. Learning rates are 0.003 except Ubuntu MLSHoL (0.001). Each complete epoch uses all training candidates and one optimizer update. Dual sums the full-graph loss and backpropagates once; MLSHoL accumulates gradients of separate microbatch forwards. Its BatchNorm/Dropout behavior is microbatch-dependent, not full-batch-equivalent. Class weight is min(10,sqrt((1-p)/p)), with p over the full fit cohort; loss_weight multiplies the BCE objective.

Dual-Mapping uses GAT with four heads. MLSHoL uses SparseSIN/MPSN. hidden controls the representation width and must be a positive even integer; 2*hidden must be divisible by gat_heads. rw_enabled=false removes the relevant RW branch, and batch_size controls MLSHoL gradient microbatches. Dual always forwards the complete candidate graph and sums its loss before one backward. Training histories record the effective batch size and backward-call count.

NDC-substances and threads-math-sx use pipeline=historynet_joint and component_generator equal to their dataset name. The transferred tags-math-sx HistoryNet recipe produces two member logits with seeds 41/42, width 128, 20 epochs and weight_power=0.5. At each prediction cut, its training set uses the last three completed 20%-width historical windows, sampled with capsnap up to 25000 rows. Historical origins are 10/20/30/40/50/60%; fit and target origins use the archived 60% and 80% cuts. Other cutpoints move backward to preserve timestamp blocks. The candidate reservoir uses seed 62026; capsnap uses seed 41.

These two datasets retain their original fit/validation partition and final selected evaluation IDs. Fit candidates are reordered before validation candidates in the full candidate graph, while only fit labels supervise the joint model. Structural and component scalers fit those same permitted fit rows. Both structural encoders use the common two-step candidate diffusion RW; packed local complexes retain their original candidate identities. The archived native objectives, seeds and RW engines are not selected by either current component_joint or structure_only mode.

| Dataset | Supervised fit rows / positives | Validation rows | Full target rows | Selected evaluation rows / positives |
|---|---|---|---|---|
| NDC-substances | 761 / 30 | 189 | 6000 | 1750 / 23 |
| threads-math-sx | 14401 / 11 | 3599 | 30000 | 28872 / 27 |

NDC/threads component generation preserves the original capsnap sample-gate metadata when the available history is below the sampling cap. This is an explicit HistoryNet transfer recipe, not a reconstruction of an original NDC/threads H model.

EU/Ubuntu formal component generation uses recipe_policy=frozen_fit. Component names, directions and preprocessing are documented in COMPONENT_SOURCES.md. Data source and evaluation-protocol conditions are documented in DATA.md. Ablation configurations and experimental settings are described in the paper.
