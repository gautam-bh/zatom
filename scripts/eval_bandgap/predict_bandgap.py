"""Stage 4: band gap prediction + conditioning-accuracy metrics.

Predicts the band gap of each (relaxed) generated structure with a pretrained
surrogate and scores it against the target gap the structure was conditioned on
(from the manifest), giving MAE / RMSE / correlation per target and overall.

Backends (neither ships in the zatom container — use a separate env):
  - matgl  (default): MEGNet-MP-2019.4.1-BandGap-mfi at PBE fidelity, matching the
    PBE gaps in matbench_mp_gap that the LoRA model was conditioned on.
        pip install matgl
  - alignn: mp_gappbe_alignn (also PBE gaps on MP).
        pip install alignn

Ground-truth-quality gaps for a small subset come from dft_validate.py instead.

    python scripts/eval_bandgap/predict_bandgap.py \
        --manifest $SCRATCH/zatom_eval/bandgap_gen/manifest_relaxed.csv \
        --backend matgl
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest", required=True,
        help="manifest_relaxed.csv (or manifest_ehull.csv) with target_bandgap + relaxed_cif",
    )
    p.add_argument("--backend", choices=["matgl", "alignn"], default="matgl")
    p.add_argument("--out", default=None, help="Output dir (default: manifest's dir)")
    p.add_argument(
        "--use-unrelaxed", action="store_true",
        help="Predict on the raw generated CIFs (column 'cif') instead of relaxed ones",
    )
    p.add_argument(
        "--tolerance", type=float, default=0.5,
        help="Band gap tolerance in eV for the 'hit rate' metric",
    )
    return p.parse_args()


def make_matgl_predictor():
    import matgl
    import torch

    model = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
    pbe_state = torch.tensor([0])  # multi-fidelity model: 0 = PBE

    def predict(structure: Structure) -> float:
        return float(model.predict_structure(structure=structure, state_attr=pbe_state))

    return predict


def make_alignn_predictor():
    import torch
    from alignn.graphs import Graph
    from alignn.pretrained import get_figshare_model
    from jarvis.core.atoms import pmg_to_atoms

    model = get_figshare_model("mp_gappbe_alignn")
    model.eval()

    def predict(structure: Structure) -> float:
        atoms = pmg_to_atoms(structure)
        g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=8.0, max_neighbors=12)
        with torch.no_grad():
            out = model([g, lg])
        return float(out.detach().cpu().numpy().ravel()[0])

    return predict


def accuracy_metrics(df: pd.DataFrame, tolerance: float) -> dict:
    """df needs columns target_bandgap + pred_bandgap, NaNs already dropped."""
    if len(df) == 0:
        return {"n": 0}
    err = df["pred_bandgap"] - df["target_bandgap"]
    metrics = {
        "n": int(len(df)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mean_error": float(err.mean()),
        f"within_{tolerance}eV": float((err.abs() <= tolerance).mean()),
    }
    if df["target_bandgap"].nunique() > 1:
        metrics["pearson_r"] = float(df["target_bandgap"].corr(df["pred_bandgap"]))
        metrics["spearman_rho"] = float(
            df["target_bandgap"].corr(df["pred_bandgap"], method="spearman")
        )
    return metrics


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    out_dir = Path(args.out) if args.out else Path(args.manifest).parent

    cif_col = "cif" if args.use_unrelaxed else "relaxed_cif"
    if cif_col not in manifest.columns:
        raise SystemExit(
            f"Column '{cif_col}' not in {args.manifest} — run mlff_relax.py first "
            "or pass --use-unrelaxed with the sun_filter manifest."
        )

    predict = make_matgl_predictor() if args.backend == "matgl" else make_alignn_predictor()

    preds = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc=f"{args.backend} band gap"):
        pred = None
        if pd.notna(row.get(cif_col)):
            try:
                pred = predict(Structure.from_file(row[cif_col]))
            except Exception as e:  # noqa: BLE001
                print(f"Prediction failed for {row['id']}: {e}")
        preds.append(pred)
    manifest["pred_bandgap"] = preds
    manifest.to_csv(out_dir / "manifest_bandgap.csv", index=False)

    scored = manifest[manifest["pred_bandgap"].notna() & manifest["target_bandgap"].notna()]
    summary = {
        "backend": args.backend,
        "structures": "unrelaxed" if args.use_unrelaxed else "relaxed",
        "overall": accuracy_metrics(scored, args.tolerance),
        "per_target": {
            str(t): accuracy_metrics(scored[scored["target_bandgap"] == t], args.tolerance)
            for t in sorted(scored["target_bandgap"].unique())
        },
    }
    if "metastable" in scored.columns:
        summary["metastable_only"] = accuracy_metrics(
            scored[scored["metastable"].fillna(False)], args.tolerance
        )

    (out_dir / "bandgap_accuracy.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
