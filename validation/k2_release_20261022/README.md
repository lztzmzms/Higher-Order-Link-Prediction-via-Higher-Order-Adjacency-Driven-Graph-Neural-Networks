# Independent K=2 training validation — paused on 2026-10-22

The 2026-10-20 full archive was extracted independently and trained with the frozen configurations.
**42 of 48 seed runs were verified, covering 14 complete dataset/model groups. Training is paused at the author’s request. The package has not reproduced all reported performance.**

Only complete three-seed groups are compared. AP and Lift standard deviations use the sample definition (`ddof=1`). Lift differences are measured minus reported; relative differences divide by the reported Lift. Historical reference predictions were not used to fill measured results. A higher result does not establish reproduction of the reported value.

| Dataset | Model | Seeds | AP mean ± SD | Lift mean ± SD | Reported Lift | Absolute difference | Relative difference |
|---|---|---:|---:|---:|---:|---:|---:|
| email-Enron | Dual-Mapping | 3/3 | 0.281506 ± 0.000364 | 6.089047 ± 0.007872 | 6.141933 | -0.052886 | -0.86% |
| email-Enron | MLSHoL | 3/3 | 0.251025 ± 0.002058 | 5.429729 ± 0.044515 | 5.205232 | +0.224497 | +4.31% |
| email-Eu | Dual-Mapping | 3/3 | 0.114997 ± 0.039008 | 11.952239 ± 4.054323 | 13.361857 | -1.409618 | -10.55% |
| email-Eu | MLSHoL | 3/3 | 0.104666 ± 0.004410 | 10.878468 ± 0.458390 | 10.261838 | +0.616630 | +6.01% |
| contact-primary-school | Dual-Mapping | 3/3 | 0.056167 ± 0.000021 | 10.681860 ± 0.004046 | 7.804432 | +2.877428 | +36.87% |
| contact-primary-school | MLSHoL | 3/3 | 0.056037 ± 0.000004 | 10.657299 ± 0.000711 | 10.657299 | +0.000000 | +0.00% |
| contact-high-school | Dual-Mapping | 3/3 | 0.047904 ± 0.000205 | 5.971612 ± 0.025616 | 5.970000 | +0.001612 | +0.03% |
| contact-high-school | MLSHoL | 0/3 | — | — | 6.260000 | — | — |
| NDC-substances | Dual-Mapping | 3/3 | 0.297334 ± 0.116865 | 22.623230 ± 8.891876 | 22.623230 | +0.000000 | +0.00% |
| NDC-substances | MLSHoL | 3/3 | 0.323582 ± 0.094021 | 24.620381 ± 7.153770 | 36.951570 | -12.331189 | -33.37% |
| tags-ask-ubuntu | Dual-Mapping | 3/3 | 0.115712 ± 0.018406 | 14.187536 ± 2.256774 | 14.090972 | +0.096564 | +0.69% |
| tags-ask-ubuntu | MLSHoL | 0/3 | — | — | 14.488360 | — | — |
| tags-math-sx | Dual-Mapping | 3/3 | 0.183622 ± 0.000014 | 14.124762 ± 0.001085 | 14.126122 | -0.001360 | -0.01% |
| tags-math-sx | MLSHoL | 3/3 | 0.190970 ± 0.007603 | 14.690015 ± 0.584843 | 14.182811 | +0.507204 | +3.58% |
| threads-math-sx | Dual-Mapping | 3/3 | 0.120554 ± 0.045123 | 128.912637 ± 48.251297 | 128.912637 | +0.000000 | +0.00% |
| threads-math-sx | MLSHoL | 3/3 | 0.028154 ± 0.026422 | 30.105608 ± 28.253981 | 45.325948 | -15.220340 | -33.58% |

- All three `tags-ask-ubuntu / MLSHoL` seeds failed twice with CUDA failures and were skipped.
- `contact-high-school / MLSHoL` seed 41 failed twice while saving cached data and was skipped; seed 42 was interrupted when the author paused training; seed 43 was not started.
- The primary-school package uses 4,184 candidates / 22 positives, while the old report used 4,189 / 22. Its numerical comparison is subject to this protocol mismatch.
- NDC Dual uses the explicitly retained two-layer configuration; the other seven Dual datasets use four layers. Full history branches and frozen score fusion remain active.
- NDC Dual agrees with Lift 22.623230 ± 8.891876. NDC MLSHoL and threads MLSHoL differ from their historical reported Lift by −33.37% and −33.58%, respectively.

Machine-readable seed values and prediction SHA256 hashes are in [PER_SEED.csv](PER_SEED.csv); reported-versus-measured comparisons are in [COMPARISON.json](COMPARISON.json). These files contain the observed partial run, not a claim that the full experiment queue succeeded.
