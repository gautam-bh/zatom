"""Stage 1: validity / uniqueness / novelty filtering of generated structures.

Cheap CPU-only filters run BEFORE MLFF relaxation so GPU time is only spent on
structures worth relaxing. Stability (the "S" in S.U.N.) is determined later from
relaxed energies by compute_ehull.py.

Checks (matching the conventions used by CDVAE/FlowMM and zatom.eval.crystal.Crystal):
  - comp_valid:   SMACT charge-neutrality + Pauling electronegativity test
  - struct_valid: no interatomic distance < 0.5 A, volume > 0.1 A^3
  - unique:       first representative of each StructureMatcher group
  - novel:        no StructureMatcher match in the matbench_mp_gap training split
                  (composition-prefiltered so the 106k-structure reference is tractable)

Usage (any env with torch + pymatgen + smact, e.g. the zatom container or mateval env):

    python scripts/eval_bandgap/sun_filter.py \
        --manifest $SCRATCH/zatom_eval/bandgap_gen/manifest.csv \
        --reference data/matbench/processed/matbench_mp_gap_train_0.pt
"""

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Lattice, Structure
from tqdm import tqdm

import smact
from smact.screening import pauling_test


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, help="manifest.csv from generate_conditional.py")
    p.add_argument(
        "--reference", nargs="+", default=[],
        help="Processed matbench .pt file(s) (PyG collated) used as the novelty reference. "
        "If omitted, novelty is skipped (novel=True for all).",
    )
    p.add_argument("--out", default=None, help="Output dir (default: manifest's dir)")
    p.add_argument("--stol", type=float, default=0.5)
    p.add_argument("--angle-tol", type=float, default=10.0)
    p.add_argument("--ltol", type=float, default=0.3)
    return p.parse_args()


def smact_validity(comp: Composition, use_pauling_test: bool = True) -> bool:
    """CDVAE-style composition validity: exists a charge-neutral oxidation state
    assignment that passes the Pauling electronegativity test."""
    count = [int(c) for c in comp.reduced_composition.values()]
    elem_symbols = tuple(str(el) for el in comp.reduced_composition.elements)

    if len(elem_symbols) == 1:
        return True

    space = smact.element_dictionary(elem_symbols)
    smact_elems = [space[el] for el in elem_symbols]
    electronegs = [el.pauling_eneg for el in smact_elems]
    ox_combos = [el.oxidation_states for el in smact_elems]

    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        cn_e, cn_r = smact.neutral_ratios(ox_states, stoichs=stoichs, threshold=int(max(count)))
        if not cn_e:
            continue
        if not use_pauling_test:
            return True
        try:
            if pauling_test(ox_states, electronegs):
                return True
        except TypeError:
            # Missing electronegativity data -> give benefit of the doubt
            return True
    return False


def structural_validity(structure: Structure, cutoff: float = 0.5) -> bool:
    """No two atoms closer than `cutoff` Angstrom (incl. periodic images), sane volume."""
    if structure.volume < 0.1:
        return False
    if len(structure) == 1:
        return True
    dm = structure.distance_matrix
    np.fill_diagonal(dm, np.inf)
    return bool(dm.min() >= cutoff)


def load_reference_by_formula(paths, formulas_needed):
    """Rebuild pymatgen Structures from zatom's processed matbench .pt files, keeping only
    compositions that occur among the generated structures (novelty prefilter)."""
    import torch

    ref = defaultdict(list)
    for path in paths:
        data, slices = torch.load(path, map_location="cpu", weights_only=False)  # nosec
        n = len(slices["lengths"]) - 1
        at_sl = slices["atom_types"]
        fc_sl = slices["frac_coords"]
        for i in tqdm(range(n), desc=f"Reference {Path(path).name}"):
            atom_types = data.atom_types[at_sl[i]: at_sl[i + 1]].numpy()
            comp = Composition({int(z): int(c) for z, c in zip(*np.unique(atom_types, return_counts=True))})
            formula = comp.reduced_formula
            if formula not in formulas_needed:
                continue
            frac = data.frac_coords[fc_sl[i]: fc_sl[i + 1]].numpy()
            lengths = data.lengths[i].flatten().numpy()
            angles = data.angles[i].flatten().numpy()
            try:
                ref[formula].append(
                    Structure(
                        Lattice.from_parameters(*lengths, *angles),
                        atom_types.tolist(),
                        frac,
                        coords_are_cartesian=False,
                    )
                )
            except Exception:  # noqa: BLE001
                continue
    return ref


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    out_dir = Path(args.out) if args.out else Path(args.manifest).parent
    matcher = StructureMatcher(stol=args.stol, angle_tol=args.angle_tol, ltol=args.ltol)

    # --- Load structures + validity checks -------------------------------------------
    structures: dict[str, Structure] = {}
    comp_valid, struct_valid = [], []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Validity"):
        cv = sv = False
        if row.get("decode_ok", True):
            try:
                s = Structure.from_file(row["cif"])
                structures[row["id"]] = s
                cv = smact_validity(s.composition)
                sv = structural_validity(s)
            except Exception:  # noqa: BLE001
                pass
        comp_valid.append(cv)
        struct_valid.append(sv)
    manifest["comp_valid"] = comp_valid
    manifest["struct_valid"] = struct_valid
    manifest["valid"] = manifest["comp_valid"] & manifest["struct_valid"]

    # --- Uniqueness (composition-bucketed StructureMatcher grouping) ------------------
    valid_ids = manifest.loc[manifest["valid"], "id"].tolist()
    by_formula = defaultdict(list)
    for sid in valid_ids:
        by_formula[structures[sid].composition.reduced_formula].append(sid)

    unique_flag = {sid: False for sid in valid_ids}
    for formula, sids in tqdm(by_formula.items(), desc="Uniqueness"):
        groups = matcher.group_structures([structures[sid] for sid in sids])
        # Mark one representative per group unique; map back via object identity
        for group in groups:
            rep = group[0]
            for sid in sids:
                if structures[sid] is rep:
                    unique_flag[sid] = True
                    break
    manifest["unique"] = manifest["id"].map(unique_flag).fillna(False).astype(bool)

    # --- Novelty vs training data ------------------------------------------------------
    unique_ids = manifest.loc[manifest["unique"], "id"].tolist()
    if args.reference:
        formulas_needed = {structures[sid].composition.reduced_formula for sid in unique_ids}
        ref = load_reference_by_formula(args.reference, formulas_needed)
        novel_flag = {}
        for sid in tqdm(unique_ids, desc="Novelty"):
            s = structures[sid]
            candidates = ref.get(s.composition.reduced_formula, [])
            novel_flag[sid] = not any(matcher.fit(s, r, skip_structure_reduction=True) for r in candidates)
    else:
        print("WARNING: no --reference given, skipping novelty (novel=True for all unique)")
        novel_flag = {sid: True for sid in unique_ids}
    manifest["novel"] = manifest["id"].map(novel_flag).fillna(False).astype(bool)

    # --- Outputs ------------------------------------------------------------------------
    manifest.to_csv(out_dir / "manifest_filtered.csv", index=False)
    survivors = manifest[manifest["valid"] & manifest["unique"] & manifest["novel"]]
    survivors.to_csv(out_dir / "manifest_survivors.csv", index=False)

    n = len(manifest)
    summary = {
        "n_generated": n,
        "comp_valid_rate": float(manifest["comp_valid"].mean()),
        "struct_valid_rate": float(manifest["struct_valid"].mean()),
        "valid_rate": float(manifest["valid"].mean()),
        "unique_rate_among_valid": float(manifest.loc[manifest["valid"], "unique"].mean()) if manifest["valid"].any() else 0.0,
        "novel_rate_among_unique": float(manifest.loc[manifest["unique"], "novel"].mean()) if manifest["unique"].any() else 0.0,
        "n_survivors_for_relaxation": int(len(survivors)),
        "per_target": {
            str(t): {
                "n": int((manifest["target_bandgap"] == t).sum()),
                "valid_rate": float(manifest.loc[manifest["target_bandgap"] == t, "valid"].mean()),
            }
            for t in sorted(manifest["target_bandgap"].unique())
        },
    }
    (out_dir / "sun_filter_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
