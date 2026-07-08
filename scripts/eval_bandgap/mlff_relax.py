"""Stage 2: MLFF relaxation of SUN-filtered structures with CHGNet.

Relaxes every survivor from sun_filter.py with CHGNet (the MLFF available in the
zatom container; MatterSim/matgl are not installed there), records unrelaxed and
relaxed energies, and writes relaxed CIFs. The relaxed energies feed
compute_ehull.py for the stability ("S") part of S.U.N.; the relaxed structures
feed predict_bandgap.py / dft_validate.py.

Conventions (fmax, step cap, initial-vs-relaxed StructureMatcher RMS) follow
forks/flowmm/src/flowmm/chgnet_.py and matbench-discovery's CHGNet setup.

Run inside the zatom container, ideally on a GPU node:

    shifter --image=registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1 \
        python scripts/eval_bandgap/mlff_relax.py \
        --manifest $SCRATCH/zatom_eval/bandgap_gen/manifest_survivors.csv
"""

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest", required=True,
        help="manifest_survivors.csv from sun_filter.py (any CSV with id,cif columns works)",
    )
    p.add_argument("--out", default=None, help="Output dir (default: manifest's dir)")
    p.add_argument("--fmax", type=float, default=0.1, help="Force convergence criterion, eV/A")
    p.add_argument("--max-steps", type=int, default=500, help="Max optimizer steps per structure")
    p.add_argument(
        "--relax-cell", action="store_true", default=True,
        help="Relax lattice as well as ions (default: on)",
    )
    p.add_argument("--no-relax-cell", dest="relax_cell", action="store_false")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    return p.parse_args()


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    out_dir = Path(args.out) if args.out else Path(args.manifest).parent
    relaxed_dir = out_dir / "relaxed_cifs"
    relaxed_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from chgnet.model import CHGNet, StructOptimizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    chgnet = CHGNet.load()
    optimizer = StructOptimizer(model=chgnet, use_device=device)
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    # ASE optimizers are chatty; keep the log readable
    warnings.filterwarnings("ignore", module="ase")

    records = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="CHGNet relax"):
        rec = {
            "id": row["id"],
            "target_bandgap": row.get("target_bandgap"),
            "e_gen": None,
            "e_relax": None,
            "e_relax_per_atom": None,
            "num_sites": None,
            "n_steps": None,
            "converged": False,
            "rms_dist": None,
            "matched": False,
            "relaxed_cif": None,
            "exception": False,
        }
        try:
            structure = Structure.from_file(row["cif"])
            rec["num_sites"] = len(structure)

            result = optimizer.relax(
                structure,
                fmax=args.fmax,
                steps=args.max_steps,
                relax_cell=args.relax_cell,
                verbose=False,
            )
            traj = result["trajectory"]
            relaxed: Structure = result["final_structure"]

            rec["e_gen"] = float(traj.energies[0])
            rec["e_relax"] = float(traj.energies[-1])
            rec["e_relax_per_atom"] = float(traj.energies[-1]) / len(relaxed)
            rec["n_steps"] = len(traj.energies) - 1
            rec["converged"] = len(traj.energies) - 1 < args.max_steps

            rms = matcher.get_rms_dist(structure, relaxed)
            rec["rms_dist"] = float(rms[0]) if rms is not None else None
            rec["matched"] = rms is not None

            relaxed_cif = relaxed_dir / f"{row['id']}_relaxed.cif"
            relaxed.to(filename=str(relaxed_cif))
            rec["relaxed_cif"] = str(relaxed_cif)
        except Exception as e:  # noqa: BLE001 - keep going on pathological cells
            print(f"Relaxation failed for {row['id']}: {e}")
            rec["exception"] = True
        records.append(rec)

    result_df = pd.DataFrame(records)
    result_df.to_csv(out_dir / "manifest_relaxed.csv", index=False)

    ok = result_df[~result_df["exception"]]
    summary = {
        "n_input": int(len(result_df)),
        "n_relaxed": int(len(ok)),
        "n_converged": int(ok["converged"].sum()) if len(ok) else 0,
        "match_rate": float(ok["matched"].mean()) if len(ok) else 0.0,
        "mean_rms_dist": float(ok["rms_dist"].dropna().mean()) if len(ok) else None,
        "mean_steps": float(ok["n_steps"].mean()) if len(ok) else None,
        "fmax": args.fmax,
        "max_steps": args.max_steps,
        "relax_cell": args.relax_cell,
        "mlff": f"CHGNet {getattr(__import__('chgnet'), '__version__', 'unknown')}",
    }
    (out_dir / "mlff_relax_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
