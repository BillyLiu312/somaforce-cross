# SomaForce-Cross Implementation Plan

Updated: 2026-07-25
Status: compatibility index; the former teacher/student route is retired for V1

The authoritative V1 implementation sequence is:

```text
docs/v1_force_residual_plan.md
docs/v1_force_residual_implementation_plan.md
```

Do not implement the earlier privileged residual-teacher and student-distillation
pipeline from repository history. V1 directly trains a deployable-input residual
actor with asymmetric PPO:

```text
frozen task scaffold -> a_nom [B,23]
noisy two-wrist F/T history + proprioception + action history
  -> p_dir, p_mag -> P_cross -> z_cross
  -> contact-gated bounded residual
  -> a_total [B,23]

clean simulated wrench -> auxiliary p_dir/p_mag targets
privileged simulator state -> critic only
```

The current implementation order is:

1. Audit scaffold, action, history, frame, and contact contracts.
2. Implement the pure PyTorch wrist-token, semantic, cross, and gate core.
3. Implement normalized 23-D residual composition and safety limits.
4. Implement and validate the virtual two-wrist 6D F/T sensing path.
5. Build the vectorized residual environment with asymmetric observations.
6. Train the existing four tasks with authority, mismatch, and sensor curricula.
7. Add the remaining three training scaffolds one at a time after parity gates.
8. Freeze residual weights and run held-out zero-shot evaluation.

Scaffold export, provenance, licensing, and deployment requirements remain in
`docs/pretrained_hdmi_scaffold_rules.md`. Canonical reference construction for
future tasks remains in `docs/hdmi_omomo_scaffold_pipeline.md`.

Historical rollout and validation reports remain factual evidence. Their use of
the term HDMI teacher describes the current non-deployable `phase=train`
artifact and does not define a V1 residual teacher actor.
