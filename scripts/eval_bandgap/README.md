# Band gap conditioning evaluation pipeline

Evaluates a LoRA band gap-conditioned Zatom checkpoint end to end: sample
conditionally, keep only S.U.N.-worthy structures, relax with an MLFF, score
stability against the MP hull, and measure how well the achieved band gaps match
the conditioning targets (ML surrogate for all structures, VASP for a subset).

Stages (each reads the previous stage's manifest CSV):

| # | Script | What it does | Where to run |
|---|--------|--------------|--------------|
| 1 | `generate_conditional.py` | CFG sampling at each target gap → CIFs + `manifest.csv` | zatom container, GPU node |
| 2 | `sun_filter.py` | SMACT/structural validity, uniqueness, novelty vs matbench_mp_gap train → `manifest_survivors.csv` | zatom container, CPU |
| 3 | `mlff_relax.py` | CHGNet relaxation, energies + relaxed CIFs → `manifest_relaxed.csv` | zatom container, GPU preferred |
| 4 | `compute_ehull.py` | MP2020-corrected e_above_hull vs MP hull (S.U.N. complete) → `manifest_ehull.csv` | zatom container, CPU |
| 5 | `predict_bandgap.py` | MEGNet (matgl) / ALIGNN PBE gap prediction, accuracy vs targets → `bandgap_accuracy.json` | separate env with `pip install matgl` |
| 6 | `dft_validate.py` | `prepare`: VASP static inputs + SLURM array for a per-target subset; `collect`: parse gaps, score vs targets & surrogate | login node / VASP license required |

Example sweep:

```bash
export IMG=registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1
OUT=$SCRATCH/zatom_eval/bandgap_gen

# 1. sample (GPU node)
shifter --image=$IMG python scripts/eval_bandgap/generate_conditional.py \
    --ckpt logs/finetune_fm/runs/<run>/checkpoints/last.ckpt \
    --out-dir $OUT --targets 0.5 1.0 2.0 3.0 4.0 --num-per-target 200

# 2. filter
shifter --image=$IMG python scripts/eval_bandgap/sun_filter.py \
    --manifest $OUT/manifest.csv \
    --reference data/matbench/processed/matbench_mp_gap_train_0.pt

# 3. relax
shifter --image=$IMG python scripts/eval_bandgap/mlff_relax.py \
    --manifest $OUT/manifest_survivors.csv

# 4. stability (download the MP hull pickle once, see compute_ehull.py docstring)
shifter --image=$IMG python scripts/eval_bandgap/compute_ehull.py \
    --manifest $OUT/manifest_relaxed.csv \
    --ppd forks/flowmm/mp_02072023/2023-02-07-ppd-mp.pkl.gz \
    --sun-manifest $OUT/manifest_filtered.csv

# 5. band gap accuracy (env with matgl)
python scripts/eval_bandgap/predict_bandgap.py --manifest $OUT/manifest_ehull.csv

# 6. limited DFT validation
python scripts/eval_bandgap/dft_validate.py prepare --manifest $OUT/manifest_ehull.csv \
    --out $OUT/dft --per-target 10
sbatch $OUT/dft/run_vasp_array.sh
python scripts/eval_bandgap/dft_validate.py collect --dft-root $OUT/dft \
    --ml-manifest $OUT/manifest_bandgap.csv
```
