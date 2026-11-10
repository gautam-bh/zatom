"""Diagnose large `lengths_scaled_loss` spikes in the LoRA band gap finetune.

Loads the real pretrained 160M checkpoint into the LoRA model, runs N matbench training
batches forward (no grad), and records per sample: true scaled lengths, predicted x_1,
interpolation time t, the InverseTimeFactor weight, and the squared error. Writes a CSV
and prints how the batch loss decomposes into the error and the time weight.

Usage (repo root, GPU):
    shifter --image=registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1 --env=PROJECT_ROOT=$PWD \
        python docs/plans/diag/lengths_loss.py <pretrained.ckpt> [n_batches] [batch_size]

Set ZERO_BG_EMBED=1 to zero the band gap embedder's output layer (the model is then exactly
the pretrained one). Both variants use seed 0, so they see the same batches and t draws.
"""

import csv
import os
import sys

import torch

sys.path.insert(0, ".")
from tests.predeploy.conftest import attach_stub_trainer, build_model, compose_cfg  # noqa: E402
from tests.predeploy.launcher import PROD_LAUNCHER, train_overrides  # noqa: E402
from zatom.flow import interpolants  # noqa: E402

CKPT = sys.argv[1]
N_BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else 40
BS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
ROWS = []

_orig = interpolants.SDEMetricInterpolant.compute_loss


def recording_compute_loss(self, path, pred, *args, **kwargs):
    if self.key == "lengths_scaled":
        x1 = path.x_1[self.key][:, 0, :]
        p = pred[self.key][:, 0, :]
        t = path.t[self.key].reshape(-1)
        tf = self.time_factor(path.t[self.key]).reshape(-1)
        periodic = path.x_1["token_is_periodic"].any(-1)
        err2 = ((p - x1) ** 2).mean(-1)
        for i in range(len(x1)):
            if periodic[i]:
                ROWS.append(
                    dict(
                        t=t[i].item(), tf=tf[i].item(), err2=err2[i].item(),
                        weighted=(tf[i] * err2[i]).item(),
                        true_max=x1[i].max().item(), pred_max=p[i].max().item(),
                        true=x1[i].tolist(), pred=p[i].tolist(),
                    )
                )
    return _orig(self, path, pred, *args, **kwargs)


interpolants.SDEMetricInterpolant.compute_loss = recording_compute_loss


def main():
    import hydra

    torch.manual_seed(0)
    zero_embed = os.environ.get("ZERO_BG_EMBED") == "1"

    from zatom.callbacks.lora_finetuning import LoRABandGapFinetuning

    overrides = [
        o for o in train_overrides(PROD_LAUNCHER)
        if o.split("=")[0] in ("experiment", "model/architecture")
    ]
    cfg = compose_cfg(
        [*overrides, "~logger", "data.datamodule.num_workers.train=0", "model.compile=false",
         f"data.datamodule.batch_size.train={BS}"],
        tiny=False, cpu=False,
    )
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)
    datamodule.setup("fit")
    model = build_model(cfg)
    attach_stub_trainer(model, datamodule, ckpt_path=CKPT, pretrained_ckpt_path=CKPT)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.on_load_checkpoint(ckpt)
    res = model.load_state_dict(ckpt["state_dict"], strict=False)
    assert not res.unexpected_keys, res.unexpected_keys[:5]
    del ckpt
    LoRABandGapFinetuning().freeze_before_training(model)
    if zero_embed:
        from zatom.models.architectures.transformer.band_gap_embedder import BandGapEmbedder

        embedders = [m for m in model.modules() if isinstance(m, BandGapEmbedder)]
        assert len(embedders) == 1, len(embedders)
        torch.nn.init.zeros_(embedders[0].mlp[-1].weight)
        torch.nn.init.zeros_(embedders[0].mlp[-1].bias)
    model.log = lambda *a, **k: None
    model = model.cuda().train()

    batch_losses = []
    it = iter(datamodule.train_dataloader())
    with torch.no_grad():
        for b in range(N_BATCHES):
            n0 = len(ROWS)
            batch = next(it).cuda()
            loss = model.training_step(batch, b)
            rows = ROWS[n0:]
            batch_losses.append((loss.item(), max(r["weighted"] for r in rows)))
            print(f"batch {b:3d} loss={loss.item():9.3f}  max tf*err2={batch_losses[-1][1]:9.2f}", flush=True)

    out = "docs/plans/diag/lengths_loss" + ("_zero_embed" if zero_embed else "") + ".csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ROWS[0]))
        w.writeheader()
        w.writerows(ROWS)

    r = sorted(ROWS, key=lambda r: -r["weighted"])
    tot = sum(x["weighted"] for x in ROWS)
    print(f"\n{len(ROWS)} samples (incl. multiplicity copies). Sum tf*err2 = {tot:.1f}")
    top = r[: max(1, len(r) // 100)]
    print(f"top 1% of samples carry {sum(x['weighted'] for x in top) / tot:.1%} of the weighted lengths loss")
    print("top 10 by tf*err2:")
    for x in r[:10]:
        print(f"  t={x['t']:.3f} tf={x['tf']:6.1f} err2={x['err2']:8.3f} true={[round(v,2) for v in x['true']]} pred={[round(v,2) for v in x['pred']]}")
    big = [x for x in ROWS if x["true_max"] > 10]
    small = [x for x in ROWS if x["true_max"] <= 10]
    mean = lambda xs, k: sum(x[k] for x in xs) / max(1, len(xs))
    print(f"true max len > 10: n={len(big)} mean err2={mean(big,'err2'):.3f} mean weighted={mean(big,'weighted'):.2f}")
    print(f"true max len <=10: n={len(small)} mean err2={mean(small,'err2'):.3f} mean weighted={mean(small,'weighted'):.2f}")
    hi_t = [x for x in ROWS if x["tf"] >= 50]
    print(f"tf>=50 (t>~0.86): n={len(hi_t)} mean err2={mean(hi_t,'err2'):.3f} mean weighted={mean(hi_t,'weighted'):.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
