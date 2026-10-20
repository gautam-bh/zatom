# LoRA Band Gap Fine-Tune — Pre-Deploy Test Plan

> Goal: never again spend a Perlmutter allocation to discover a bug that could have been found on a
> login node in minutes. Every failure in the run history below would have been caught by one of
> the tests in this plan.

**Where tests run.** No host Python on Perlmutter login nodes; everything runs inside the same image
the jobs use, so library versions match exactly:

```bash
ZIMG=registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1
shifter --image=$ZIMG --env=PROJECT_ROOT=$PWD python -m pytest tests/predeploy -x -q
```

Three tiers, each a gate for the next:

| Tier | Where | Wall time | Gate for |
|---|---|---|---|
| T0 static/config | login node, CPU | < 1 min | anything |
| T1 model/data on CPU | login node, CPU (1–2 ranks, gloo) | ~5–10 min | debug-queue smoke |
| T2 GPU smoke | `salloc` interactive / debug QOS, 4×A100 | ~15 min | 18 h regular job |

---

## 1. Run history: what failed, and why

| Job | Date | Symptom | Root cause | Fixed? | Test that catches it |
|---|---|---|---|---|---|
| 55419435 | 07-05 | `SyntaxError` in `scripts/generate_id.py` → empty `WANDB_RUN_ID` → `SettingsValidationError: Invalid value for property run_id: .` | Launcher ran the helper with the *host* `python` (not the container's py3.10). Empty result never checked. | Yes (uses `/dev/urandom`) | T0-3 |
| 55777952 | 07-12 | `wandb UsageError: api_key not configured (no-tty)` | No W&B credentials in the batch environment | Yes (preflight) | T0-4 |
| 56395971 | 08-07 | Rank 0: W&B 401. Ranks 1–3: hung 30 min in `ALLREDUCE`, then NCCL watchdog abort | Invalid key. `wandb.Table` serialization hits the API even when offline. Rank 0 died outside a collective and the others sat in NCCL's default 30-min timeout | Auth: yes. 30-min hang: **no** | T0-4, T1-8 |
| 56547853 | 08-09 | CUDA OOM, `training_step` → attention, first batch | Joint-pretraining default batch 128 × `multiplicity` 8; dense fp32 attention with explicit mask (no flash); cost ∝ B·L² | Partly | T1-6, T2-1 |
| 56548543 | 08-09 | Same OOM at bs=32 | `B_eff` = 32×8 = 256 graphs, L=446 | Partly | T1-6, T2-1 |
| 56750303 | 08-12 | Same OOM at bs=32 | Same | → bs=8, accum=16 | T1-6, T2-1 |
| 56923536 | 08-13 | `RuntimeError: Expected to have finished reduction in the prior iteration` — 80 of 212 trainable tensors (indices 132–211) got no gradient. Peak memory ~47.8 GiB/GPU | `task_name=finetune_fm` replaces the loss with aux-only (band gap *prediction*), so the cross-attention decoder blocks' LoRA is disconnected (§2.1 I-1, reproduced on CPU) | **No (current blocker)** | T1-2, T1-8 |

Each failure was found only *after* queue wait + container start + data load + model build, which is
10–30 min per attempt. 7 attempts over ~6 weeks, and none got past the first optimizer step.

---

## 2. Issues found

### 2.1 Blocking (the pipeline cannot train until these are fixed)

**I-1 The run trains band gap *prediction*, not conditional *generation*. The DDP crash in job 56923536 is a symptom of this.**

*Confirmed on CPU* (`docs/plans/diag/grad_flow.py`, output `docs/plans/diag/grad_flow.log`):
one matbench batch, forward + backward. Exactly **80 / 212** trainable tensors get `grad is None`, at indices
**132–211**, the same indices DDP reported. All 80 are LoRA adapters in the 5 cross-attention decoder blocks
(`{atom_types,pos,frac_coords,lengths_scaled,angles_radians}_cross_attention.{self,cross}_attention.{q,k,v,o}_proj.lora_{A,B}`).

*Mechanism:* `zatom/models/zatom.py:571-573`

```python
# Recompute loss when finetuning
if self.hparams.task_name == "finetune_fm":
    loss_dict["loss"] = sum(v for k, v in loss_dict.items() if k.startswith("aux_"))
```

Upstream, `finetune_fm` means property-prediction fine-tuning, so it **discards the flow-matching losses**
and keeps only the auxiliary heads. The LoRA band-gap launcher reuses `task_name=finetune_fm`, so the
optimized loss is only `aux_global_property_loss`. In the diagnostic, `loss = 0.4409 = aux_global_property_loss`,
while `atom_types_loss=1.26`, `frac_coords_loss=4.66`, `lengths_scaled_loss=14.5` and
`angles_radians_loss=5.03` are computed and then thrown away. The decoder blocks only feed the generative output
heads, so they're disconnected from the loss. The trunk LoRA still gets gradient through the aux head, via
`repr_layer=15`.

*Why this matters beyond the crash:* setting `find_unused_parameters=true` would make the job *run*, but it would
spend 11 h teaching the model to **predict** the band gap it is simultaneously **given as conditioning**
(see I-6). That's a trivial objective, and the generative heads would never learn to use the conditioning.
So the "obvious" DDP fix produces a scientifically useless checkpoint that looks healthy (falling loss).

*Fix direction (for review, not implemented):* decouple "which losses to optimize" from `task_name`. For
example, add a config flag like `model.finetune_loss: aux_only | generative | generative+aux`, defaulting to
`aux_only` for existing property fine-tunes and set to `generative` (optionally `+aux` with a small weight) in
`lora_bandgap.yaml`. Keep `task_name=finetune_fm`, because `on_load_checkpoint` relies on it to reset optimizer/loop
state on the first run. Leave `find_unused_parameters=false`, so a future regression fails loudly instead of
training silently.

*Note:* 64 trunk `lora_A` tensors get an all-zero grad at step 0. That's expected, because `lora_B` is zero-initialized. It
isn't an issue, but T1-2 must check for `grad is None`, not `grad == 0`.

**I-2 The crash checkpoint becomes the resume point.** `last_model_checkpoint.save_on_exception: True`
wrote `0-1.ckpt` when job 56923536 crashed. `train_fm.py` resumes from the newest file matching
`^(?:epoch=\d+-step=\d+|\d+-\d+)\.ckpt$`, so resubmitting into the same run dir resumes from the crash
state (which can be mid-accumulation).

### 2.2 High (wasted runs, silent bad results)

**I-3 OOM headroom is not guaranteed.** Attention memory ∝ `batch_size × multiplicity × L²`, where L is the
largest structure in the batch. The explicit `attn_mask` rules out the flash SDPA backend. bs=8 peaked at
~47.8 GiB/80, but that was measured over only 64 batches; the tail of the matbench size distribution
was never sampled.

**I-4 A dead rank hangs the job for 30 min.** In 56395971, rank 0 died and ranks 1–3 sat in `ALLREDUCE` until
the default 1800 s NCCL timeout. No DDP `timeout` is set, and neither is `TORCH_NCCL_ASYNC_ERROR_HANDLING`.

**I-5 Silent weight-loading failures.** `train_fm.py` sets `model.strict_loading = False`. If the
`{name}.weight → {name}.linear.weight` remap in `on_load_checkpoint` ever misses a key, the base weights stay
random and nothing errors.

**I-6 The checkpoint-selection metric doesn't measure the goal (closely tied to I-1).** `model_checkpoint.monitor =
val_matbench/aux_global_property_loss` is the auxiliary band-gap *prediction* loss, but the band gap is also
fed in as *conditioning*, so the loss can fall for trivial reasons (label leak). It says nothing
about whether generated crystals hit the target gap. `trainer.test()` then loads this "best" checkpoint.

**I-7 Validation can deadlock across ranks.** Each rank logs its own `gen_metrics_dict` keys with `sync_dist=True`.
If the key sets differ between ranks (e.g. a metric is skipped when one rank gets 0 valid samples), the
collectives don't match and NCCL hangs (see I-4).

### 2.3 Medium / hygiene

- **I-8** The forward pass reads band-gap de-normalization stats from `self.trainer.datamodule.matbench_train_dataset`,
  so `forward` can't run without a trainer and datamodule (this blocks unit tests and standalone inference).
- **I-9** `on_load_checkpoint` re-initializes aux heads for qm9/omol25/mptrj but not matbench `global_property`.
- **I-10** Stale comments and messages in the scripts: the smoke PASS line says "16 batches" (it's 64);
  the smoke comment mentions "multiplicity cut 8 → 2" (config keeps 8); the diag script's header analysis is for bs=32
  and it never trap-kills its memory sampler.
- **I-11** pscratch (checkpoints) has an 8-week purge, so best checkpoints must be copied to CFS after the run.
- **I-12** Code is bind-run from `$HOME` against a fixed image tag (`zatom:0.0.1`), so code and dependency drift is invisible.
  Every test must run in that image.

### 2.4 Already fixed (keep as regression tests)
Empty run_id / `generate_id.py` SyntaxError; missing W&B key; W&B 401 even in offline mode; bs=128/32 OOM;
inherited pretraining `best_model_path` (f7d388d); Hydra `name_date_date` run-dir mismatch → silent restart from scratch.

---

## 3. Test suite

All under `tests/predeploy/`, pytest, run inside the image. Shared fixture `conftest.py`:
compose the Hydra config exactly as `train_fm.py` does (`experiment=lora_bandgap`, pop `datasets.*._target_`,
`PROJECT_ROOT` set), plus a tiny-model override (`hidden_size=64, num_layers=2`) for speed. The full
160M build is used only where a test is specifically about the real config. Note: this fixture needs I-8 fixed
(or a minimal trainer stub with `datamodule`, `overfit_batches`, `global_step`) to call `forward` without a Trainer.

### T0: static / config (login node, < 1 min, no model build)

| ID | Test | Catches |
|---|---|---|
| T0-1 | Compose `experiment=lora_bandgap` with the **exact override list** parsed out of `finetune_ddp_tft_lora_bandgap.sh` and `smoke_lora_bandgap.sh` (grep the `srun` block). Assert it resolves with no missing interpolations. | Typos / renamed keys in launcher overrides |
| T0-2 | Assert the resolved values: `batch_size.train=8`, `accumulate_grad_batches=16`, `multiplicity=8`, `devices == ntasks-per-node`, `num_nodes == --nodes`, `check_val_every_n_epoch=1`, `max_time < --time`. | Config/launcher drift (the bs=128 inheritance bug) |
| T0-3 | `bash -n` on every launcher. Run the run-id/date generation lines in isolation and assert they're non-empty and match `^[a-z0-9]{8}$`. | 55419435 |
| T0-4 | Run the W&B preflight block (credential check + `wandb.Api().viewer`) inside the container. | 55777952, 56395971 |
| T0-5 | Assert Hydra run dir `${name}_${date}` == launcher `RUN_DIR_NAME`, and that `CKPT_PATH` sits under pscratch. | silent restart-from-scratch bug |
| T0-6 | Assert `configs/strategy` sets `find_unused_parameters` consistently with what T1-2 finds, and that a DDP `timeout` ≤ 10 min is set. | I-1, I-4 |
| T0-7 | Assert the pretrained ckpt exists, is the `_l_` (160M) file, and that `hidden_size` in its `hyper_parameters` == the config's. | wrong-checkpoint mistakes |

### T1: model + data on CPU (login node, ~5–10 min)

| ID | Test | Catches |
|---|---|---|
| T1-1 | **Weight-load completeness.** Build the 160M model, run `on_load_checkpoint` on the real pretrained ckpt, then `load_state_dict(strict=False)`. Assert `missing_keys` ⊆ {LoRA A/B, `band_gap_embedder.*`} and `unexpected_keys` == ∅. Assert every `*.linear.weight` equals the ckpt's `*.weight`. | I-5 |
| T1-2 | **Gradient reachability + objective check (the DDP test).** Promote `docs/plans/diag/grad_flow.py` into a test. One matbench batch, `multiplicity=2`, forward + backward. Assert (a) **every** `requires_grad` param has `.grad is not None` (not `!= 0`, because `lora_A` is legitimately zero at init), and (b) `loss` ≥ the sum of the generative terms, i.e. the flow-matching losses are actually in the objective. Run with CFG dropout forced on (`dropout_prob=1`) and off. **Currently fails: 80 no-grad, `loss == aux_global_property_loss`.** | 56923536 (I-1) |
| T1-3 | **Freeze correctness.** After `LoRABandGapFinetuning.freeze_before_training`, the trainable set == exactly LoRA A/B + embedder; count == 212; numel ≈ 1.9M. | callback regressions |
| T1-4 | **Conditioning actually wired.** Same noised input, two band gaps (0 eV vs 4 eV) with a non-zero LoRA B / embedder weights: outputs differ. With `band_gap=NaN`: output == unconditional. Band gap fed to the embedder is in **eV** (assert the range is ~[0, 10], not z-scored). | f31258c regression, silent no-op conditioning |
| T1-5 | **CFG batch layout.** `sample()` with `cfg_scale≠1`: first half has the band gap, second half NaN; result is deterministic under a fixed seed. | 6a9f57c regression |
| T1-6 | **Memory model.** Scan matbench train `num_atoms`; report the p99/max L. Compute the predicted attention bytes for `bs × mult × heads × L² × 4 × layers × 2` at the worst-case L and assert < 60 GiB. Optionally enforce with a dataloader max-L or token-budget sampler. | 56547853/56548543/56750303, I-3 |
| T1-7 | **Resume selection.** Temp dir with `0-1.ckpt` (exception save), `epoch=0-step=5000.ckpt`, `lora-bandgap-epoch@...ckpt`: assert the resume logic picks the intended file and never the exception checkpoint. Also: `init_run` is False on resume, so LoRA weights are restored, not reset. | I-2 |
| T1-8 | **2-rank gloo DDP on CPU.** `Trainer(accelerator="cpu", devices=2, strategy="ddp")`, tiny model, `limit_train_batches=4`, `accumulate_grad_batches=2`, `limit_val_batches=1`, W&B disabled. This is the cheapest reproduction of 56923536, since DDP's unused-param check fires on CPU too. Variant: inject an exception on rank 0 only and assert the job dies in < 2 min (needs a DDP timeout). | I-1, I-4 |
| T1-9 | **Val loop per-rank key parity.** Run `CrystalGenerationEvaluator.get_metrics` on (a) all-invalid and (b) all-valid predictions; assert the *same key set*. | I-7 |
| T1-10 | **Monitor metric exists.** After a 1-batch val run, assert `val_matbench/aux_global_property_loss` (or its replacement, see I-6) is in `trainer.callback_metrics`. | best-ckpt never saved |
| T1-11 | Existing `tests/test_cfg_band_gap.py` passes. | unit regressions |

### T2: GPU smoke (interactive `salloc -q interactive -C gpu&hbm80g -N1 -G4 -t 30`)

Use interactive QOS rather than `sbatch -q debug`: no requeue churn, and you can rerun immediately.

| ID | Test | Catches |
|---|---|---|
| T2-1 | **Worst-case memory, single GPU.** One `training_step` on a *synthetic* batch padded to the max L from T1-6 at bs=8, mult=8; record `torch.cuda.max_memory_allocated()`; require ≤ 70 GiB. | I-3 (deterministic instead of luck-of-the-batch) |
| T2-2 | Existing `smoke_lora_bandgap.sh`, unchanged, as the last gate (4 GPUs, W&B online, 4 optimizer steps, val, ckpt, test). | end-to-end plumbing |
| T2-3 | Resume: rerun T2-2 with the same `RUN_DATE`, assert it logs "resuming from …step=…" and `global_step` continues. | preemption path |

### Run order / gate

```
make predeploy        # T0 + T1, ~10 min on a login node; must be green to submit anything
salloc ... && make smoke   # T2, ~15 min
sbatch finetune_ddp_tft_lora_bandgap.sh ...
```

Wire `predeploy` into the launcher as a refuse-to-start check: the `sbatch` script exits early if
`.predeploy_ok` is older than the newest commit on the branch.

---

## 4. Proposed order of work (awaiting review — nothing below is implemented)

1. **Write T1-2 first** and confirm it fails on today's code (80 no-grad; loss = aux only).
2. **Fix I-1** with a separate loss-selection flag (keep `task_name`). T1-2 goes green.
   Decide what checkpoint selection should use instead of the aux loss (I-6): e.g. val generative
   loss, or validity × |gap_target − predicted gap| from the eval pipeline.
3. T1-1 (weight load), T1-3, T1-4, T1-5: cheap, and they guard the conditioning path that I-1 now exercises.
4. T1-8 (2-rank gloo DDP on CPU) plus a DDP `timeout` (I-4). This is the CPU stand-in for the debug-queue smoke.
5. T1-6 / T2-1 memory bound (I-3); T1-7 resume (I-2); T1-9/T1-10 val-loop parity.
6. T0 suite and the launcher refuse-to-start gate.
7. Only then: `smoke_lora_bandgap.sh` (T2-2), then the 18 h job.

Re-measure memory at step 2: with the generative loss back in the graph, the backward pass now
runs through the 5 decoder blocks, so the 47.8 GiB peak from job 56923536 is an **underestimate**
for the corrected objective.

## Appendix: diagnostic artifacts
- `docs/plans/diag/grad_flow.py` reproduces I-1 on a login node in ~10 min (CPU, inside the job image).
- `docs/plans/diag/grad_flow.log` is its output (80 no-grad tensors, indices 132–211; loss terms).
