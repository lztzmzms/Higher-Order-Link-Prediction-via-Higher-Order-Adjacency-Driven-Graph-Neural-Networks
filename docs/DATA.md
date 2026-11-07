# Data Preparation and Protocol

## Public event data

Obtain raw events from [ScHoLP Data](https://www.cs.cornell.edu/~arb/data/) or [ScHoLP-Data](https://github.com/arbenson/ScHoLP-Data). The times, nverts and simplices files contain event timestamps, per-event sizes, and flattened node IDs. scripts/convert_scholp.py converts these arrays to NPZ without changing event order.

The event format is times[N], counts[N], vertices[sum(counts)]. JSONL records with timestamp and nodes are also supported for new tasks. Events must be chronological, preserve empty event positions where present, and contain unique nodes. New tasks use left-closed, right-open time intervals and prefix-only open-triangle candidates.

## Bundled experiment inputs

The original six dataset routes include projected event streams, fixed train/target candidates, training labels, candidate-derived graph features and RW, and packed local complexes. candidates.npz stores train_q, target_q and train_y. NDC-substances and threads-math-sx retain inputs.zip with their archived protocol, shared split, sampling indices and RW resources. Their current default method is the component_joint HistoryNet transfer: it rebuilds the common structural features and two-step diffusion RW from the archived observed graphs, uses the packed local complexes, and adds two historical member logits. Archived native RW resources are not the RW inputs of this current route. These inputs are prepared experimental cohorts, not full raw source collections.

For the original six routes, training and target cuts are recorded in configs/<dataset>.json and historical predictor windows in historical_components/upstream/cohorts/INDEX.json. NDC/threads obtain their cuts from protocol/READY.json inside inputs.zip and generate their historical windows as specified in DATASET_CONFIGURATIONS.md and COMPONENT_SOURCES.md. Preserve candidate order and align all components by sorted three-node ID.



## Access and licensing

Follow the source providers' data licenses and access conditions. The package includes integer-ID event projections and necessary experiment inputs, not email text or user credentials; no additional rights to third-party data are granted. If redistribution of an input is restricted, obtain it from the provider and reproduce the required cohort using its original selection record. Supplementary K2–K4 experiments require separate raw events, candidate/feature resources, splits or checkpoints as described in supplementary_sources/.
