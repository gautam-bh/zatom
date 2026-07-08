"""Stage 5 (optional): limited DFT validation of predicted band gaps with VASP.

Because full DFT on every structure is expensive, this validates a small
stratified subset (--per-target structures per conditioning target, preferring
metastable ones when e_above_hull is available):

  prepare: writes VASP inputs per structure + a Perlmutter SLURM array script.
           Default is a static (MPStaticSet) calculation on the CHGNet-relaxed
           geometry — the cheapest way to a PBE gap; --vasp-set relax uses
           MPRelaxSet if you want DFT relaxation too.
  collect: parses each vasprun.xml, extracts the PBE band gap, and scores it
           against the conditioning targets (and the ML-predicted gaps, when a
           manifest_bandgap.csv is given).

NOTE: VASP on NERSC requires membership in the VASP license group
(https://docs.nersc.gov/applications/vasp/). Writing real POTCARs additionally
needs pymatgen's PMG_VASP_PSP_DIR configured; otherwise inputs fall back to
POTCAR.spec files (fill in POTCARs on a licensed machine).

    # in the zatom container (needs pymatgen only)
    python scripts/eval_bandgap/dft_validate.py prepare \
        --manifest $SCRATCH/zatom_eval/bandgap_gen/manifest_ehull.csv \
        --out $SCRATCH/zatom_eval/bandgap_gen/dft --per-target 10
    sbatch $SCRATCH/zatom_eval/bandgap_gen/dft/run_vasp_array.sh
    python scripts/eval_bandgap/dft_validate.py collect \
        --dft-root $SCRATCH/zatom_eval/bandgap_gen/dft
"""

import argparse
import json
from pathlib import Path

import pandas as pd

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --account={account}
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=4
#SBATCH --job-name=vasp_bandgap_val
#SBATCH --output={root}/slurm-%A_%a.out
#SBATCH --array=0-{max_index}

module load vasp/{vasp_version}

DIRS=({dirs})
cd "${{DIRS[$SLURM_ARRAY_TASK_ID]}}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun -c $SLURM_CPUS_PER_TASK --cpu-bind=cores vasp_std
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    prep = sub.add_parser("prepare", help="Subsample structures and write VASP inputs")
    prep.add_argument(
        "--manifest", required=True,
        help="manifest_ehull.csv (preferred) or manifest_relaxed.csv",
    )
    prep.add_argument("--out", required=True, help="Root dir for the VASP calculations")
    prep.add_argument("--per-target", type=int, default=10)
    prep.add_argument("--vasp-set", choices=["static", "relax"], default="static")
    prep.add_argument("--vasp-version", default="6.4.3-cpu")
    prep.add_argument("--account", default="m5008", help="SLURM account for the array job")
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument(
        "--potcar-spec", action="store_true",
        help="Write POTCAR.spec instead of real POTCARs (no pseudopotential setup needed)",
    )

    coll = sub.add_parser("collect", help="Parse vasprun.xml files and score accuracy")
    coll.add_argument("--dft-root", required=True, help="--out dir used at prepare time")
    coll.add_argument(
        "--ml-manifest", default=None,
        help="manifest_bandgap.csv from predict_bandgap.py, to also score the ML surrogate "
        "against DFT on this subset",
    )
    coll.add_argument("--tolerance", type=float, default=0.5)
    return p.parse_args()


def subsample(manifest: pd.DataFrame, per_target: int, seed: int) -> pd.DataFrame:
    ok = manifest[manifest["relaxed_cif"].notna()]
    if "metastable" in ok.columns:
        # Prefer metastable structures; top up with the lowest e_above_hull ones
        ok = ok.sort_values(
            ["metastable", "e_above_hull_per_atom"], ascending=[False, True]
        )
        picks = ok.groupby("target_bandgap", group_keys=False).head(per_target)
    else:
        picks = ok.groupby("target_bandgap", group_keys=False).apply(
            lambda g: g.sample(min(per_target, len(g)), random_state=seed)
        )
    return picks.reset_index(drop=True)


def prepare(args):
    from pymatgen.core import Structure
    from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet

    manifest = pd.read_csv(args.manifest)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    picks = subsample(manifest, args.per_target, args.seed)
    input_set_cls = MPStaticSet if args.vasp_set == "static" else MPRelaxSet

    dirs, rows = [], []
    for _, row in picks.iterrows():
        calc_dir = root / row["id"]
        structure = Structure.from_file(row["relaxed_cif"])
        vis = input_set_cls(structure)
        try:
            vis.write_input(
                output_dir=str(calc_dir),
                make_dir_if_not_present=True,
                potcar_spec=args.potcar_spec,
            )
        except Exception as e:  # noqa: BLE001 - typically missing POTCAR library
            print(f"Real POTCAR failed for {row['id']} ({e}); writing POTCAR.spec instead")
            vis.write_input(
                output_dir=str(calc_dir), make_dir_if_not_present=True, potcar_spec=True
            )
        dirs.append(str(calc_dir.resolve()))
        rows.append({"id": row["id"], "target_bandgap": row["target_bandgap"], "dir": dirs[-1]})

    subset = pd.DataFrame(rows)
    subset.to_csv(root / "dft_subset.csv", index=False)
    (root / "run_vasp_array.sh").write_text(
        SLURM_TEMPLATE.format(
            account=args.account,
            root=str(root.resolve()),
            max_index=len(dirs) - 1,
            vasp_version=args.vasp_version,
            dirs=" ".join(f'"{d}"' for d in dirs),
        )
    )
    print(
        f"Wrote {len(dirs)} {args.vasp_set} calculations under {root}\n"
        f"Submit with: sbatch {root / 'run_vasp_array.sh'}"
    )


def collect(args):
    from pymatgen.io.vasp.outputs import Vasprun

    root = Path(args.dft_root)
    subset = pd.read_csv(root / "dft_subset.csv")

    records = []
    for _, row in subset.iterrows():
        rec = dict(row)
        rec.update({"dft_bandgap": None, "dft_direct": None, "dft_converged": None})
        vasprun_path = Path(row["dir"]) / "vasprun.xml"
        if vasprun_path.exists():
            try:
                vrun = Vasprun(str(vasprun_path), parse_potcar_file=False)
                gap, _cbm, _vbm, is_direct = vrun.eigenvalue_band_properties
                rec["dft_bandgap"] = float(gap)
                rec["dft_direct"] = bool(is_direct)
                rec["dft_converged"] = bool(vrun.converged)
            except Exception as e:  # noqa: BLE001
                print(f"Failed to parse {vasprun_path}: {e}")
        else:
            print(f"No vasprun.xml for {row['id']} (job not run or failed)")
        records.append(rec)

    result = pd.DataFrame(records)
    if args.ml_manifest:
        ml = pd.read_csv(args.ml_manifest)[["id", "pred_bandgap"]]
        result = result.merge(ml, on="id", how="left")
    result.to_csv(root / "dft_validation.csv", index=False)

    done = result[result["dft_bandgap"].notna()]
    summary = {"n_subset": int(len(result)), "n_with_dft": int(len(done))}
    if len(done):
        err = done["dft_bandgap"] - done["target_bandgap"]
        summary["target_vs_dft"] = {
            "mae": float(err.abs().mean()),
            "mean_error": float(err.mean()),
            f"within_{args.tolerance}eV": float((err.abs() <= args.tolerance).mean()),
        }
        if "pred_bandgap" in done.columns and done["pred_bandgap"].notna().any():
            both = done[done["pred_bandgap"].notna()]
            ml_err = both["pred_bandgap"] - both["dft_bandgap"]
            summary["ml_vs_dft"] = {
                "n": int(len(both)),
                "mae": float(ml_err.abs().mean()),
                "mean_error": float(ml_err.mean()),
            }
    (root / "dft_validation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    if args.mode == "prepare":
        prepare(args)
    else:
        collect(args)


if __name__ == "__main__":
    main()
