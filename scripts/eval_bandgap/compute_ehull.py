"""Stage 3: stability (energy above hull) of CHGNet-relaxed structures.

Completes S.U.N.: builds a ComputedStructureEntry from each relaxed structure +
CHGNet energy, applies MP2020 energy corrections (matching how the MP reference
hull energies were corrected, per matbench-discovery's CHGNet evaluation), and
computes e_above_hull against the Materials Project convex hull.

The hull reference is FlowMM's PatchedPhaseDiagram pickle. Download it once
(https://figshare.com/articles/dataset/23713842, file 2023-02-07-ppd-mp.pkl.gz,
from matbench-discovery) into forks/flowmm/mp_02072023/ — that directory
currently only holds a .gitkeep.

Run inside the zatom container (CPU is fine):

    shifter --image=registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1 \
        python scripts/eval_bandgap/compute_ehull.py \
        --manifest $SCRATCH/zatom_eval/bandgap_gen/manifest_relaxed.csv \
        --ppd forks/flowmm/mp_02072023/2023-02-07-ppd-mp.pkl.gz
"""

import argparse
import gzip
import json
import pickle  # nosec - trusted matbench-discovery/flowmm artifact
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from tqdm import tqdm

# MP flags stability as e_above_hull <= 0; generative-model papers (FlowMM, CDVAE
# followups) count metastability within 0.1 eV/atom of the hull.
METASTABLE_THRESHOLD = 0.1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, help="manifest_relaxed.csv from mlff_relax.py")
    p.add_argument(
        "--ppd", required=True,
        help="PatchedPhaseDiagram pickle (.pkl or .pkl.gz) of the MP hull",
    )
    p.add_argument("--out", default=None, help="Output dir (default: manifest's dir)")
    p.add_argument(
        "--no-mp2020", action="store_true",
        help="Skip MaterialsProject2020Compatibility corrections (raw CHGNet energies)",
    )
    p.add_argument(
        "--sun-manifest", default=None,
        help="manifest_filtered.csv from sun_filter.py, used to fold unique/novel flags "
        "into the final S.U.N. summary (optional)",
    )
    return p.parse_args()


def load_ppd(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        return pickle.load(f)  # nosec


def mp_run_parameters(structure: Structure) -> dict:
    """run_type/hubbards mimicking MP's GGA(+U) scheme so that
    MaterialsProject2020Compatibility applies the same corrections as it did to
    the hull entries. MP applies +U only to compounds containing O or F."""
    from pymatgen.io.vasp.sets import MPRelaxSet

    ldauu = MPRelaxSet.CONFIG["INCAR"]["LDAUU"]
    elements = {el.symbol for el in structure.composition.elements}
    anion = "O" if "O" in elements else ("F" if "F" in elements else None)
    hubbards = {}
    if anion:
        hubbards = {el: ldauu[anion].get(el, 0.0) for el in elements}
    run_type = "GGA+U" if any(u > 0 for u in hubbards.values()) else "GGA"
    return {"run_type": run_type, "hubbards": hubbards, "is_hubbard": run_type == "GGA+U"}


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    out_dir = Path(args.out) if args.out else Path(args.manifest).parent

    print(f"Loading phase diagram from {args.ppd} ...")
    ppd = load_ppd(args.ppd)

    compat = None
    if not args.no_mp2020:
        from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

        compat = MaterialsProject2020Compatibility(check_potcar=False)

    e_hull_col, corrected_col = [], []
    relaxed_ok = manifest["relaxed_cif"].notna() & ~manifest["exception"].fillna(True)
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="e_above_hull"):
        e_hull, corrected = None, False
        if relaxed_ok.loc[row.name] and pd.notna(row.get("e_relax")):
            try:
                structure = Structure.from_file(row["relaxed_cif"])
                params = mp_run_parameters(structure)
                entry = ComputedStructureEntry(
                    structure, energy=float(row["e_relax"]), parameters=params
                )
                if compat is not None:
                    processed = compat.process_entries([entry], verbose=False, clean=True)
                    if processed:
                        entry = processed[0]
                        corrected = True
                e_hull = float(ppd.get_e_above_hull(entry, on_error="raise"))
            except Exception as e:  # noqa: BLE001 - chemsys may be missing from the hull
                print(f"e_above_hull failed for {row['id']}: {e}")
        e_hull_col.append(e_hull)
        corrected_col.append(corrected)

    manifest["e_above_hull_per_atom"] = e_hull_col
    manifest["mp2020_corrected"] = corrected_col
    manifest["stable"] = manifest["e_above_hull_per_atom"] <= 0.0
    manifest["metastable"] = manifest["e_above_hull_per_atom"] <= METASTABLE_THRESHOLD
    manifest.to_csv(out_dir / "manifest_ehull.csv", index=False)

    scored = manifest[manifest["e_above_hull_per_atom"].notna()]
    summary = {
        "n_input": int(len(manifest)),
        "n_scored": int(len(scored)),
        "stable_rate": float(scored["stable"].mean()) if len(scored) else 0.0,
        "metastable_rate": float(scored["metastable"].mean()) if len(scored) else 0.0,
        "mean_e_above_hull": float(scored["e_above_hull_per_atom"].mean()) if len(scored) else None,
        "metastable_threshold": METASTABLE_THRESHOLD,
        "mp2020_corrections": not args.no_mp2020,
        "per_target": {
            str(t): {
                "n_scored": int((scored["target_bandgap"] == t).sum()),
                "metastable_rate": float(
                    scored.loc[scored["target_bandgap"] == t, "metastable"].mean()
                )
                if (scored["target_bandgap"] == t).any()
                else None,
            }
            for t in sorted(manifest["target_bandgap"].dropna().unique())
        },
    }

    # Inputs to this stage already passed validity+uniqueness+novelty filtering, so
    # S.U.N. rate over the *generated* set needs the original counts from sun_filter.
    if args.sun_manifest:
        sun = pd.read_csv(args.sun_manifest)
        n_generated = int(len(sun))
        n_sun = int(manifest["metastable"].fillna(False).sum())
        summary["sun_rate_metastable"] = n_sun / n_generated if n_generated else 0.0
        summary["n_generated_total"] = n_generated

    (out_dir / "ehull_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
