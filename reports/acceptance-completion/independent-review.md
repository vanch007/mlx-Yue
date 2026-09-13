# Independent review and closure

Reviewer: `/root/review_completion_fixes`, model `gpt-5.6-sol`. Review was read-only and did not run GPU workloads. The earlier `/root/review_transcription` review belongs to the initial acceptance audit, before these fixes.

The completion reviewer found no actionable correctness defects in the cancellation/resource-check propagation, saved configuration restoration, and reserved batch-ID fixes. Cancellation remains cooperative at model download, hashing, loading and individual GPU operation boundaries; checks run before/after those operations rather than preempting a native kernel.

The reviewer also checked the added reference tooling. The FP64 VAE calibration uses the original decoder, matching checkpoint/input hashes and crop geometry. Releasing unused acoustic modules in the original AR capture does not change the pure AR computation. Complete-song reference reports retain listening as pending.

Two evidence issues were found and addressed:

1. Saving complete Torch and MLX transcription trajectories alone did not compare them. `tools/compare_transcription.py` now checks input/task/checkpoint identity, verifies capture-file hashes, validates integer shapes and compares every token. The actual 183-token run passes; see `transcription-generated-comparison.json`.
2. A global float32 label incorrectly described integer generated tokens. `tools/transcription_oracle.py` now records each array's dtype. Comparator tests cover exact agreement, first mismatch and changed-input rejection.

The new comparator and dtype correction were verified by the primary agent's tests. They were communicated to the reviewer; no second independent approval is claimed.

## Additional isolated-install defect

The actual Torch-free wheel check subsequently discovered that native
`save_pretrained` called the original `copy_model_files`, which imported
`modeling_vae` and therefore Torch. The new `lyra.checkpoints.copy_vae_checkpoint`
and its integration were independently re-reviewed. No actionable defect was
found: the helper preserves config bytes and checkpoint identity, validates
allowed weight names before writing, excludes arbitrary source code/private
files, copies the installed compatibility code without importing it, and retains
the nonempty-destination guard. Regression tests cover those behaviors.

The reviewer also rechecked `CapturedPipeline.synthesize`: the temporary SDPA
wrapper changes synchronization and unused allocator-cache reclamation only. It
retains the original operands, call arguments, operation order and RNG. Restoration
is protected by `finally`; the reference runner is single-threaded. This review
does not itself establish that a complete reference render fits the memory budget.

## Staged original reference and full-prefix AR capture

The reviewer confirmed `staged_reference.py` preserves original BF16 modules,
the complete CPU FP32 noise draw, context/chunks, prefix K/V visibility and 32
midpoint equations while changing only which checkpoint modules are resident.
`layerwise_ar.py` was also reviewed: absolute RoPE, causal query tiling, residual
and MLP ordering, K/V extraction shapes and the final five logit positions match
the normal cached MLX comparator. It is explicitly a full-prefix reference, not
a claim that original incremental cache execution passes the memory gate.

The staged-song review found that the earlier saved semantic stage had no
contemporaneous model/VAE identity receipt. Subsequent captures now record
identities, and `resume_reference.py` rejects missing or mismatched identities.
The already-running capture used the earlier script, retained verbatim as
`resume_reference-staged.py`; its known command uses the pinned model but this
missing stage receipt is not retroactively invented. The resulting song is a
listening candidate with that provenance limitation, not strict lineage approval.
