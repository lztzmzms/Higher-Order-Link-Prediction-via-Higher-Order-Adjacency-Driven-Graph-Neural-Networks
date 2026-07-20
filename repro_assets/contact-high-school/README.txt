This directory contains the frozen random-walk embedding used by the recorded
contact-high-school Dual-Mapping result:

  rw_seed=0
  rw_impl=fast_positive
  rw_walk_length=10
  rw_walks_per_node=40
  rw_window_size=2
  rw_embedding_dim=64
  rw_epochs=10

The cache is an input artifact, not an experiment output.  It is required
because regenerating SkipGram embeddings under a different PyTorch/CPU/GPU
environment changes the test embedding and does not reproduce the recorded
17.271601 mean lift.

checkpoint_seed_54.bin is the recorded best independent model-seed checkpoint.
Loading it re-runs inference for seed 54 only.  This verifies that recorded
single-seed result exactly; it is evaluation-only and is not a replacement for
retraining or for computing a new ten-seed mean.

The .bin suffixes are intentional.  The contents remain NPZ and PyTorch
checkpoint formats, but the neutral suffix prevents capsule/Git packaging
rules from silently excluding *.npz and *.pt files.

For Code Ocean, both .bin files are placed directly in /code so that an
uploader that does not preserve nested folders still includes them.
