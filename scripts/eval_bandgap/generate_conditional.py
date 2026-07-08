"""Generate band gap-conditioned crystal samples from a LoRA fine-tuned Zatom checkpoint.

Sweeps a set of target band gaps, samples `num_per_target` structures for each via
classifier-free guidance, and writes CIFs plus a manifest CSV recording each structure's
target band gap (needed downstream to score conditioning accuracy).

Run inside the zatom container (needs torch + the zatom package), on a GPU node:

    shifter python scripts/eval_bandgap/generate_conditional.py \
        --ckpt logs/finetune_fm/runs/<run>/checkpoints/last.ckpt \
        --out-dir $SCRATCH/zatom_eval/bandgap_gen \
        --targets 0.5 1.0 2.0 3.0 4.0 \
        --num-per-target 200 --batch-size 50 --cfg-scale 3.0
"""

import argparse
import json
import os
from pathlib import Path

import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import open_dict  # noqa: E402

from zatom import register_custom_omegaconf_resolvers  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="Path to the LoRA fine-tuned checkpoint")
    p.add_argument("--out-dir", required=True, help="Output directory for CIFs + manifest")
    p.add_argument(
        "--targets", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0, 4.0],
        help="Target band gaps in eV",
    )
    p.add_argument("--num-per-target", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=100, help="ODE integration steps")
    p.add_argument("--architecture", default="tft_160M")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return p.parse_args()


def build_model(args):
    """Instantiate the lit module from the lora_bandgap experiment config and load weights."""
    register_custom_omegaconf_resolvers()
    initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3")
    cfg = compose(
        config_name="train_fm",
        overrides=[
            "experiment=lora_bandgap",
            f"model/architecture={args.architecture}",
            # Compose API has no hydra runtime context, so anything that resolves
            # ${hydra:...} must be overridden explicitly.
            "name=bandgap_generation",
            f"paths.viz_dir={args.out_dir}",
            f"paths.output_dir={args.out_dir}",
        ],
    )

    # Match eval_fm.py: datasets configs are consumed as plain dicts, not instantiated
    with open_dict(cfg):
        for dataset in cfg.model.datasets:
            cfg.model.datasets[dataset].pop("_target_", None)

    model = instantiate(cfg.model)
    model.configure_model()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)  # nosec
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        print(f"WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")

    model = model.to(args.device)
    model.eval()
    return model


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    cif_dir = out_dir / "cifs"
    cif_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args)

    dataset = "matbench"
    num_nodes_bincount = model.num_nodes_bincount[dataset]
    spacegroups_bincount = model.spacegroups_bincount[dataset]
    assert num_nodes_bincount is not None, "matbench num_nodes.pt bincount not loaded"
    mask_token_index = model.hparams.sampling.mask_token_index

    rows = []
    sample_id = 0
    for target in args.targets:
        remaining = args.num_per_target
        while remaining > 0:
            bs = min(args.batch_size, remaining)
            band_gap = torch.full((bs,), float(target), device=args.device)
            out, batch = model.sample_and_decode(
                num_nodes_bincount=num_nodes_bincount,
                spacegroups_bincount=spacegroups_bincount,
                batch_size=bs,
                cfg_scale=args.cfg_scale,
                dataset_index=model.dataset_to_index[dataset],
                dataset_idx=model.dataset_to_idx[dataset],
                steps=args.steps,
                band_gap=band_gap,
            )

            # Decode per-structure arrays (mirrors ZatomLitModule.sample_generation_examples)
            start_idx = 0
            for idx_in_batch, num_atom in enumerate(batch["num_atoms"].tolist()):
                atom_types = out["atom_types"].narrow(0, start_idx, num_atom).clone()
                atom_types[atom_types == 0] = 1
                atom_types[atom_types == mask_token_index] = 1
                frac_coords = out["frac_coords"].narrow(0, start_idx, num_atom)
                lengths = out["lengths_scaled"][idx_in_batch] * float(num_atom) ** (1 / 3)
                angles = torch.rad2deg(out["angles_radians"][idx_in_batch])
                start_idx += num_atom

                cif_path = cif_dir / f"gen_{sample_id:06d}.cif"
                row = {
                    "id": f"gen_{sample_id:06d}",
                    "target_bandgap": target,
                    "num_atoms": num_atom,
                    "cif": str(cif_path),
                    "decode_ok": False,
                }
                try:
                    from pymatgen.core import Lattice, Structure

                    lengths_np = lengths.flatten().detach().cpu().numpy()
                    angles_np = angles.flatten().detach().cpu().numpy()
                    structure = Structure(
                        Lattice.from_parameters(*lengths_np, *angles_np),
                        atom_types.detach().cpu().numpy().tolist(),
                        frac_coords.detach().cpu().numpy(),
                        coords_are_cartesian=False,
                    )
                    structure.to(filename=str(cif_path))
                    row["decode_ok"] = True
                except Exception as e:  # noqa: BLE001 - garbage lattices are expected
                    print(f"Decode failed for sample {sample_id}: {e}")
                rows.append(row)
                sample_id += 1
            remaining -= bs
        print(f"Target {target} eV: generated {args.num_per_target} structures")

    import pandas as pd

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    (out_dir / "generation_meta.json").write_text(
        json.dumps(
            {
                "ckpt": os.path.abspath(args.ckpt),
                "targets": args.targets,
                "num_per_target": args.num_per_target,
                "cfg_scale": args.cfg_scale,
                "steps": args.steps,
                "seed": args.seed,
                "architecture": args.architecture,
            },
            indent=2,
        )
    )
    n_ok = int(manifest["decode_ok"].sum())
    print(f"\nDone: {n_ok}/{len(manifest)} structures decoded to CIF -> {out_dir}")


if __name__ == "__main__":
    main()
