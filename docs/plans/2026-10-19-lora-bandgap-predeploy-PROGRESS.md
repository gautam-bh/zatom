# LoRA band gap — pre-deploy test plan: WORKING NOTES (resume point)

Task (user, 2026-10-19): read all Perlmutter traces/logs for the conditional (band gap LoRA)
training pipeline, list every issue found (from logs and in general), and build a plan for
local tests to run before deploying to Perlmutter GPU nodes. Deliverable = plan doc
(`2026-10-19-lora-bandgap-predeploy-tests.md`, same dir). No code changes requested yet.

## Status
- [x] Read all 7 SLURM .err logs in scripts/perlmutter/regular/logs/
- [x] Read launcher/smoke/diag scripts, experiment + callback configs, LoRA/embedder/callback code
- [x] I-1 ROOT CAUSE CONFIRMED: zatom.py:571-573 finetune_fm -> loss = aux only; decoder-block LoRA disconnected. Reproduced on CPU (docs/plans/diag/grad_flow.log).

- [x] Review zatom.py training_step/band_gap path, datamodule, val loop for further issues
- [x] Write the plan doc (issues §2 + test suite §3 drafted; only I-1 mechanism pending)
- grad_flow.py needed: trainer stub w/ datamodule, overfit_batches=0, global_step=0; and set
  batch.sample_is_periodic / node_is_periodic before model.forward (training_step normally does it).
  Fresh output -> scratchpad/grad_flow.run2.log
- 2026-10-19 (session 2, login03): scratchpad /tmp was node-local and lost. Diagnostic moved to
  docs/plans/diag/grad_flow.py, output docs/plans/diag/grad_flow.log.
- USER SCOPE (session 2): "just build the plan" — NO code/config changes; user reads everything before
  confirming any implementation.

## Run history (chronological)
| Job | Date | Failure | Root cause | Status |
|---|---|---|---|---|
| 55419435 | 07-05 | `SyntaxError` in scripts/generate_id.py (py2-style interpreter?) -> empty WANDB run_id -> `SettingsValidationError: Invalid value for property run_id: .` | launcher called `python scripts/generate_id.py` outside container (host python lacks type hints support / not py3) | fixed: launcher now uses `tr -dc ... /dev/urandom` |
| 55777952 | 07-12 | `wandb UsageError: api_key not configured (no-tty)` | no W&B creds in batch env | fixed: preflight cred check |
| 56395971 | 08-07 | W&B 401 on rank0 (offline mode still hits graphql via wandb.Table) -> other ranks hang in ALLREDUCE 30 min NCCL timeout -> abort | invalid key; rank0 death not propagated -> 30-min hang | preflight auth added; NCCL timeout hang remains a general risk |
| 56547853 | 08-09 | CUDA OOM in training_step attention (bs=128 default from joint) | B_eff = batch x multiplicity(8); dense fp32 attn w/ explicit mask (no flash) | bs reduced |
| 56548543 | 08-09 | OOM, bs=32 accum=4 | same; L=446 | |
| 56750303 | 08-12 | OOM, bs=32 | same | -> bs=8 accum=16 |
| 56923536 | 08-13 | DDP `Expected to have finished reduction in the prior iteration` — 80 trainable params (indices 132–211 of 212) got no grad; peak mem ~47.8 GiB/GPU at bs=8 | TBD (being diagnosed) | OPEN — latest blocker |

## Observations so far (feed into issues list)
- `configs/strategy/optimized_ddp.yaml`: find_unused_parameters=false, static_graph=false.
- `num_aux_layers: 0` but auxiliary_tasks exist; `separate_energy_force_transformers: true`.
- LoRA wraps every Linear named q/k/v/o_proj anywhere in model (inject_lora walks whole tree),
  so modules unused in the matbench forward path get trainable LoRA params -> DDP unused.
- `model.strict_loading = False` in train_fm.py -> a LoRA key remap bug would silently leave
  base weights random. Need a test asserting no missing base keys after load.
- `on_load_checkpoint` aux-head reinit only checks qm9/omol25/mptrj, not matbench global_property.
- Smoke script `/tmp` MEMLOG copied after run; PASS message hardcodes "16 batches" (stale).
- diag script header still references bs=32 analysis; MEMLOG not trap-killed.
- Smoke OOM jobs: smoke script referenced "multiplicity cut 8 -> 2" in comment but config keeps 8 (stale comment).
- CPU smoke run 07-19: trainer.test() FileNotFoundError on pretraining author's best_model_path
  (fixed in f7d388d by popping callbacks state).
- No python on login host; container image registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1 works via
  `shifter --image=... python` on login node (torch 2.8.0+cu128, lightning 2.6.0) -> local CPU tests feasible.
- Existing tests: tests/test_cfg_band_gap.py (embedder + LoRA unit tests only).
- Model summary (job 56923536): trainable 1.9M / 164M total. 212 trainable tensors = 4 (embedder) +
  16 trunk layers x 4 proj x 2 (=128) + 5 cross-attn decoder blocks x 2 attn x 4 proj x 2 (=80).
  Unused indices 132..211 = exactly 80 -> strongly suggests the 5 cross-attention decoder blocks'
  LoRA params (registered after trunk). Code path looks like they ARE called in forward
  (transformer_module.py:514-548), so verifying with a real CPU fwd/bwd:
  scratchpad grad_flow.py (instantiates datamodule+model in container, one batch, lists grad=None).
  Container needs `--env=PROJECT_ROOT=$PWD`; model must be built like train_fm.py (pop datasets._target_).
- Checkpoint-selection metric `val_matbench/aux_global_property_loss` = aux band-gap PREDICTION loss,
  while band gap is also fed in as CONDITIONING -> label leak; metric says nothing about conditional
  generation quality. Flag as design issue.
- Val loop: per-rank `gen_metrics_dict` keys logged with sync_dist -> if key sets differ across ranks
  (e.g. 0 valid samples on one rank) => NCCL deadlock (cf. 30-min ALLREDUCE hang in 56395971).
- band gap de-normalization reads `self.trainer.datamodule.matbench_train_dataset` inside forward ->
  forward can't run without a trainer+datamodule (affects tests and inference).
- [x] PLAN COMPLETE (session 2). Awaiting user review; do not implement until confirmed.

## Decisions (user, 2026-10-20)
- **Band gap roles.** The generator *conditions* on band gap (`BandGapEmbedder`, raw eV) and must be
  trained on the generative flow-matching loss. *Predicting* band gap is the evaluator's job only
  (`scripts/eval_bandgap/predict_bandgap.py`: external MEGNet via matgl, ALIGNN optional). Zatom's
  in-model aux `global_property` head isn't the training objective. At most it's an optional
  small-weight side loss. So the I-1 fix must put the generative loss back into the objective; don't
  use `find_unused_parameters=true`. I-6 follows from this: select checkpoints on a generation
  metric, not `val_matbench/aux_global_property_loss`.
- **Workflow.** Develop on the local machine, push to `origin/bandgap-conditioning`, and pull on Perlmutter
  only to submit jobs. Keep shared context in this repo (`docs/plans/`), because Claude memory on
  Perlmutter doesn't sync to the local machine. T0/T1 tests should run in a plain local env
  (no Shifter); T2 stays on Perlmutter GPU.
- Implementation is still not approved: the user reviews the plan locally first.

## Implementation ledger (session 3, 2026-10-20, local Mac)
User approved implementing the whole plan (§4 order), running what can run locally, and building
(not running) the GPU / heavy tests for Perlmutter. Branch: `bandgap-conditioning`.
- Local env: conda env `zatom` (py3.10, `pip install -e .`), CPU only. Run tests with
  `~/miniconda3/envs/zatom/bin/python -m pytest tests/predeploy`.
- Ruling: worked directly on `bandgap-conditioning` (no worktree) — it is already a feature
  branch and the user develops here — cost if wrong: none (nothing touches `main`).
- Ruling: the plan has no per-task briefs, so §4's numbered order is the task list and this
  section is the ledger — cost if wrong: none.
- New finding I-13: matbench "val" set = the full train split. `joint_datamodule.py` builds
  val from `split="train"` and keeps `max(train_subset_size, 0.1*N)` = all 106k rows at
  proportion 1.0. Val loss is training loss; best-ckpt selection can't detect overfitting.
- New finding (refines I-3): `forward` pads every batch to the dataset max
  (`max_num_nodes = len(num_nodes_bincount)-1`, 444 for matbench) via `to_dense_batch`, so in the
  current code attention memory per step depends on B_eff only. Caveat: job 56547853's OOM asked
  for 1.30 GiB, which fits L=206 at bs=128, not L=444 -- that run may have had a different
  `num_nodes.pt`. T1-6 checks the padding; T2-1 measures the real peak.
- Ruling (I-1): new hparam `model.finetune_loss` = `aux_only` (default, unchanged upstream
  behavior) | `generative` | `generative+aux` (+ `finetune_aux_loss_weight`, default 0.1);
  `lora_bandgap.yaml` sets `generative`. `task_name` stays `finetune_fm` (on_load_checkpoint
  needs it) and `find_unused_parameters` stays false — cost if wrong: none for other finetunes.
- Ruling (I-6): `model_checkpoint.monitor = val_matbench/loss` (conditional flow-matching loss
  on val batches, mode min). The eval-pipeline gap MAE can't run inside the val loop (external
  MEGNet). Caveat I-13: val batches are training data — cost if wrong: best ckpt ≈ latest ckpt.
- Ruling (I-2): `save_on_exception: False` for `last_model_checkpoint` in lora_bandgap. Verified in
  Lightning 2.6 source that `on_exception` uses `format_checkpoint_name`, i.e. the crash ckpt is
  indistinguishable from a periodic `{epoch}-{step}.ckpt`. SLURM requeue covers preemption.
- Ruling (I-4): `strategy.timeout = 10 min` (timedelta) in lora_bandgap. Risk: any collective that
  legitimately waits >10 min (e.g. rank 0 processing matbench from scratch while others wait)
  would now abort; matbench is already processed on Perlmutter — cost if wrong: one failed job.
- Ruling (I-5): `on_load_checkpoint` raises if an init run's checkpoint lacks any non-LoRA,
  non-embedder weight of the LoRA model (only when lora_rank > 0). Unexpected keys aren't
  checked there (a remap miss already shows up as missing) — T1-1 checks both on the real ckpt.
- Ruling (I-8): `Zatom.setup()` caches the band gap (shift, scale) from the datamodule;
  `forward` uses it (falls back to `trainer.datamodule`) and no longer touches `self.trainer`
  when none is attached. Other datasets' aux paths still read the datamodule (untouched).
- Ruling (I-9): not changed. Re-initializing the matbench `global_property` head would hand the
  LoRA run a random frozen head, and could change upstream matbench property finetunes that use
  the pretrained head — cost if wrong: none for the LoRA run (`generative` ignores the aux head).
- Ruling (I-10/11/12): stale smoke/diag text fixed; prod launcher now copies top-k + last.ckpt
  to `${CFS_CKPT_DIR:-/global/cfs/cdirs/m5008/$USER/zatom_checkpoints/...}` (non-fatal; path is a
  guess — override if m5008 CFS isn't writable), logs code commit + image, and exits with the
  training status (it used to always exit 0).
- Ruling (gate): `scripts/perlmutter/predeploy.sh` (not a Makefile; the repo has none) runs
  tests/predeploy and writes `.predeploy_ok` = HEAD on a clean tree; both launchers refuse to start
  unless it matches HEAD (`SKIP_PREDEPLOY=1` overrides). The stamp is gitignored, so it must be
  produced on Perlmutter (login node, via shifter) after `git pull`.
- Ruling (tests): conftest stubs `fairchem` and `jvp_flash_attention` when absent (macOS: no
  GitHub access for fairchem here; no triton). Neither is used by the LoRA matbench path.
- Env note: `mendeleev` is imported by `zatom/utils/data_utils.py` but missing from
  pyproject.toml; installed by hand locally. Not changed in pyproject (container already has it).
- New finding I-14 (fixed): `train_fm.py` tested `trainer.checkpoint_callback.best_model_path`,
  but `checkpoint_callback` is the *first* ModelCheckpoint = `last_model_checkpoint` (monitors
  `global_step`) in every callbacks config in the repo, so `trainer.test()` loaded the latest
  periodic ckpt, not the monitored best. Found by T1-10. Fix: `best_checkpoint_path()` picks the
  callback whose monitor == `callbacks.model_checkpoint.monitor`, else the old behavior.
  Ruling: applies to all runs, not just LoRA (it is what the `train` docstring promises) —
  cost if wrong: other experiments' test metrics now come from their best ckpt, not their latest.
- Evidence I-1 (local, tiny model, real matbench batch): with `finetune_loss=aux_only` (= old
  code) loss == aux loss (0.9614) and 80/100 trainable tensors get no grad, all in the 5
  `*_cross_attention` decoder blocks — same pattern as job 56923536. With `generative`: T1-2 green.
- Local data note: `JointDataModule.setup` downloads/processes MP20, QM9, QMOF even at
  proportion 0.0 (only OMol25/GEOM/MPtrj are skipped). First local run: ~18 min, 2.5 GB in data/.
  Not changed (out of plan scope).

### Status / handoff (end of session 3)
- Local (macOS, CPU): `pytest tests/predeploy tests/test_cfg_band_gap.py` -> 43 passed, 3 skipped
  (T0-4 no W&B creds; T0-7 and T1-1-real need PRETRAINED_CKPT_PATH), 3m46s. T1-6: matbench
  num_atoms p99=144, pad=444, B_eff=64 -> attention lower bound 19.6 GiB. T1-8 DDP (gloo, 2 ranks)
  trains clean; with rank 0 killed and a 30 s timeout the job exits in ~100 s.
- Final review: self-review only (no reviewer subagent was spawned; user didn't ask for one).
- To do on Perlmutter (needs GPU or the real checkpoint), in order:
  1. `git pull`; `PRETRAINED_CKPT_PATH=<base.ckpt> bash scripts/perlmutter/predeploy.sh`
     (runs T0-4 W&B, T0-7 and T1-1 on the real ckpt in the job image; writes `.predeploy_ok`).
  2. T2-1: `salloc -q interactive -C "gpu&hbm80g" -N 1 -G 1 -t 30 -A m5008`, then
     `shifter --image=registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1 --env=PROJECT_ROOT=$PWD python tests/predeploy/gpu/t2_1_peak_memory.py`
     (peak must be <= 70 GiB; generative loss now backprops through the decoder blocks).
  3. T2-2: `sbatch scripts/perlmutter/regular/smoke_lora_bandgap.sh <base.ckpt>`.
  4. T2-3: `bash tests/predeploy/gpu/t2_3_resume_smoke.sh <base.ckpt> <smoke job id>`.
  5. Only then the 18 h job. Decide I-13 (val = train split) before relying on best-ckpt selection.
