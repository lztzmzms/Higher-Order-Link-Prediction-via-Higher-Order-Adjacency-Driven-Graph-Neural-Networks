# Historical Prediction Components

`historical_components.generators.create_generator(name, config).generate(output_dir=...)` returns `fit` and `target` ComponentBatch objects with `q`, `values`, `names`, `mask_indices`, and metadata. Each row is aligned to a sorted int64 three-node candidate ID; generated rows are matched against the configured candidate cohort by ID, not row count.

## Dataset routes

| Dataset | Components | Predictor and feature route |
|---|---|---|
| email-Enron | 7: three HG probabilities (seeds 41/42/43), group_sync64 rule, three PAIR_LR decision values (41/42/43) | Event, temporal-decay, clock and community features; historical ECDF and HG fitting; the intermediate rank/rule backbone is used inside the local PAIR_LR predictor. See legacy.py, event_features.py, broad.py, enron_local.py. |
| email-Eu | 5: three supervised outputs (41/42/43), rule score, rule activity indicator | frozen_fit uses the fit-origin HG_FULL recipe at both origins, including its temporal feature view, training ECDF and duration matching; the rule is inactive. |
| contact-high-school | 4: edge_ratio_logprod, clock_overlap_min, temporal overlap at 0.01 (logsum), temporal overlap at 0.005 (mean) | highschool_rules.py contains the exact rule formulas. Predictive directions are selected from completed historical windows before component encoding. |
| contact-primary-school | 3: PAIR_LR decision values (41/42/43) | Temporal features, completed historical windows, duration matching, recency/deduplication weights, feature selection, scaling, and paired logistic fitting; no separate rule fusion. |
| tags-ask-ubuntu | 5: three PAIR_MLP decisions (41/42/43), synchronization rule, rule activity indicator | frozen_fit uses the fit-origin PAIR_MLP recipe and active rule at both origins, with its supervised feature selection, scaling and pairwise objective. |
| tags-math-sx | 2: HistoryNet member logits (41/42) | tag_features.py and tag_predictor.py; last three completed historical windows, quota sampling up to 25000 rows, width 128, 20 epochs, and historical weighted BCE. |
| NDC-substances | 2: HistoryNet member logits (41/42) | Explicit transfer of the tags-math-sx HistoryNet recipe using NDC temporal prefixes and archived candidate/split identities; tag_features.py, tag_predictor.py and pipelines/historynet_joint.py. |
| threads-math-sx | 2: HistoryNet member logits (41/42) | Explicit transfer of the same recipe using threads temporal prefixes and archived candidate/split identities; no pre-existing threads H predictor is reconstructed. |

Ordered component names are emitted with each batch. The final aggregate historical H score is not a component_joint input. Upstream supervised predictors are fitted as part of component generation; their objectives are separate from the joint model's true-link weighted BCE.

## Time windows and preprocessing

For the original six routes, historical candidate cohorts and cutpoints are stored in upstream/cohorts/ as q/cut records. For NDC/threads, fixed fit/target candidates, split identities and 60%/80% cutpoints come from data/<dataset>/inputs.zip. Historical candidate reservoirs are rebuilt at 10/20/30/40/50/60% origins with cap 25000 and seed 62026; non-native boundaries move backward to avoid splitting timestamp blocks. Labels span 20% windows. Each prediction origin selects the last three completed windows through capsnap with cap 25000 and seed 41, then fits the width-128, 20-epoch HistoryNet members with seeds 41/42 and weight_power=0.5. Both member logits are kept separately. Features are rebuilt from observation prefixes. Only labels whose completed window ends by the current prediction cut can be used to fit upstream predictors. Fit candidates use the configured training prefix; target candidates use the prediction prefix. Final evaluation-window labels are not used by component generation.

For high-school, directions at the 60% and 80% origins use completed 40–60% and 60–80% label windows, respectively. For email-Eu and Ubuntu, recipe_policy=frozen_fit keeps the supervised family, target, transformations, and rule state identical at fit and target; supervised fitting may use more completed past data at the later origin. Original per-stage routes remain available only to the independent H audit tool.

Each score column is transformed within its complete fixed prediction-origin cohort using (average_rank - 0.5) / N. Activity columns remain separate; inactive rules are zero before standardization. StandardScaler is fitted only on allowed training candidates, with float64 transforms and clipping to [-8,8]. Ubuntu retains float32 raw component storage; Enron and EU retain their original member-output precision. Encoding must use the complete cohort, never an individual minibatch.

## Extension interface

A custom module:Class must inherit ComponentGenerator and implement generate(output_dir=None, context=None). The supplied context contains prefix events, candidate IDs, a prediction cut, and completed historical windows. Return a ComponentBatch with ordered names and explicit activity-mask indices; mask names must contain mask. The implementation must define its own supervised/rule recipe. Built-in recipes are bound to their configured cohorts and are not automatically applied to unknown data.

## Optional aggregate-score audit

```sh
python -m historical_components.generators.audit_h --dataset email-Enron --output outputs/enron_h_audit
```

This independent tool reconstructs historical ensemble/rule/reordering outputs for the original six routes and is not called by joint training. NDC-substances and threads-math-sx use the separate transfer generators exposed by create_generator and scripts/generate_components.py; the aggregate-H audit does not cover these transfer recipes. Their standalone outputs contain prepared/ and components/ subdirectories. Dataset acquisition and protocol conditions are documented separately in DATA.md.
