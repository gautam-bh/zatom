"""Check every Niggli-reduced matbench structure is the same crystal as its raw original.

Rebuilds each structure from what the `*_niggli.pt` processed file actually stores (cell,
atom types, frac coords) and matches it against the raw matbench structure with pymatgen's
`StructureMatcher`, strictly: no primitive-cell reduction, no supercells, no volume scaling.
Also checks atom count, composition and cell volume agree. Mismatches go to a CSV.

Usage (repo root, job image):
    shifter --image=registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1 --env=PROJECT_ROOT=$PWD \
        python docs/plans/diag/niggli_match.py [n_procs]
"""

import csv
import multiprocessing as mp
import sys

sys.path.insert(0, ".")
from pymatgen.analysis.structure_matcher import StructureMatcher  # noqa: E402
from pymatgen.core import Lattice, Structure  # noqa: E402

from zatom.data.components.matbench_dataset import Matbench  # noqa: E402

N_PROCS = int(sys.argv[1]) if len(sys.argv) > 1 else 32
MATCHER = StructureMatcher(primitive_cell=False, attempt_supercell=False, scale=False)
DS = None  # set per split before forking, so workers inherit it instead of pickling it
NAMES = None  # raw sample names in processing order (`DS.data` is the tensor store once loaded)


def check(i):
    raw, _ = DS.load_sample(NAMES[i])
    g = DS.get(i)
    reduced = Structure(Lattice(g.cell.reshape(3, 3).numpy()), g.atom_types.tolist(), g.frac_coords.numpy())
    problems = []
    if len(raw) != len(reduced):
        problems.append(f"num_atoms {len(raw)} != {len(reduced)}")
    if raw.composition != reduced.composition:
        problems.append("composition differs")
    dvol = abs(raw.volume - reduced.volume) / raw.volume
    if dvol > 1e-3:
        problems.append(f"volume differs by {dvol:.2%}")
    if not problems and not MATCHER.fit(raw, reduced):
        problems.append("StructureMatcher: no match")
    return i, NAMES[i], problems


def main():
    global DS, NAMES
    rows, totals = [], {}
    for split in ("train", "test"):
        DS = Matbench(root="data/matbench", task_name="matbench_mp_gap", split=split, niggli=True)
        NAMES = DS.all_folds[DS.fold_idx][split]
        assert len(NAMES) == len(DS)
        with mp.get_context("fork").Pool(N_PROCS) as pool:
            n_bad = 0
            for n, (i, name, problems) in enumerate(pool.imap_unordered(check, range(len(DS)), chunksize=64)):
                if problems:
                    n_bad += 1
                    rows.append({"split": split, "index": i, "name": name, "problems": "; ".join(problems)})
                if (n + 1) % 10000 == 0:
                    print(f"{split}: {n + 1}/{len(DS)} checked, {n_bad} mismatches", flush=True)
        totals[split] = (len(DS), n_bad)
        print(f"{split}: {len(DS)} structures, {n_bad} mismatches", flush=True)

    out = "docs/plans/diag/niggli_match_mismatches.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "index", "name", "problems"])
        w.writeheader()
        w.writerows(rows)
    print(f"TOTAL: {sum(n for n, _ in totals.values())} structures, "
          f"{sum(b for _, b in totals.values())} mismatches (listed in {out})")


if __name__ == "__main__":
    main()
