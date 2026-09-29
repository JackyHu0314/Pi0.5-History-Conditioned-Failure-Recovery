# Results

## `controlled_slip_baseline/`

Completed mechanism diagnosis on one LIBERO-10 task. It compares the original
history policy, no-history controls and a last-valid-frame observation control.

## `recovery_distill_dev/`

Development split used to choose the recovery-distillation learning rate.
Initial states 40–49 were used here; these numbers must not be treated as the
frozen final result.

## `recovery_distill_final/`

Completed frozen evaluation on initial states 0–9, including three selected
training seeds, paired outcome records, post-training probes, training curves
and artifact hashes. Read [`结论.md`](recovery_distill_final/结论.md) before
using the JSON: the result is a one-task development study and does not show a
clear advantage over the no-history or last-valid-frame controls.
