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
