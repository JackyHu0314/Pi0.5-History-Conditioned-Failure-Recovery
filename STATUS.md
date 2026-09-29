# Current status

Last updated: 2026-09-29.

- The history-conditioned π0.5 interface, controlled-slip intervention, paired audit, recovery-window builder and eight-GPU runners are implemented.
- The original controlled-slip diagnosis is complete.
- Recovery-distillation hyperparameter selection is complete on initial states 40–49.
- Three training seeds are complete.
- Frozen evaluation on initial states 0–9 is complete for the three selected training seeds.
- Post-training probes and the audited finalizer completed successfully.
- The selected recovery-distilled runs reached 76.7% ± 11.5% control and 83.3% ± 5.8% slip success across three seeds.
- The result remains a one-task development study: it did not clearly beat the no-history or last-valid-frame baselines and did not establish immediate history-to-action routing.

The next update will package selected checkpoints and their Model Card for Hugging Face after license and artifact checks.
