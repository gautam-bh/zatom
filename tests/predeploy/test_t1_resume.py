"""T1-7: resubmitting a run resumes from the right checkpoint and keeps its LoRA weights (I-2)."""

from pathlib import Path

from tests.predeploy.conftest import attach_stub_trainer, build_model
from zatom.train_fm import find_latest_step_checkpoint


def test_resume_picks_latest_step_checkpoint(tmp_path):
    for name in (
        "0-5000.ckpt",
        "1-10000.ckpt",
        "epoch=1-step=9999.ckpt",
        "1-10000-v1.ckpt",  # versioned duplicate (enable_version_counter)
        "lora-bandgap-epoch@3-step@20000.ckpt",  # best-model checkpoint
        "last.ckpt",
    ):
        (tmp_path / name).touch()
    assert Path(find_latest_step_checkpoint(str(tmp_path))).name == "1-10000.ckpt"


def test_resume_handles_logger_style_names(tmp_path):
    for name in ("epoch=0-step=5000.ckpt", "epoch=2-step=15000.ckpt", "epoch=10-step=900.ckpt"):
        (tmp_path / name).touch()
    # Numeric, not lexicographic: epoch 10 beats epoch 2
    assert Path(find_latest_step_checkpoint(str(tmp_path))).name == "epoch=10-step=900.ckpt"


def test_resume_empty_dir(tmp_path):
    (tmp_path / "last.ckpt").touch()
    assert find_latest_step_checkpoint(str(tmp_path)) is None


def test_resume_keeps_lora_weights(tiny_cfg, datamodule, tmp_path):
    """A resumed run is not an init run: optimizer, loops and LoRA weights must survive."""
    model = build_model(tiny_cfg)
    resume = tmp_path / "0-5000.ckpt"
    pretrained = tmp_path / "pretrained.ckpt"
    resume.touch()
    pretrained.touch()
    attach_stub_trainer(
        model, datamodule, ckpt_path=str(resume), pretrained_ckpt_path=str(pretrained)
    )

    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    lora_key = next(k for k in state_dict if k.endswith("lora_B.weight"))
    state_dict[lora_key].fill_(0.123)
    checkpoint = {"state_dict": state_dict, "loops": {"x": 1}, "optimizer_states": [{"s": 1}]}

    model.on_load_checkpoint(checkpoint)
    assert checkpoint["loops"] == {"x": 1}
    assert checkpoint["optimizer_states"] == [{"s": 1}]
    assert (checkpoint["state_dict"][lora_key] == 0.123).all()
