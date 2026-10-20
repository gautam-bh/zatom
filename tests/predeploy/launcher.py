"""Read the Hydra overrides and SBATCH settings straight out of the Perlmutter launchers.

T0 tests compose the config from what the launchers actually pass, so a renamed key or a
batch-size drift between a launcher and the experiment config fails on a laptop instead of
after a queue wait.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_DIR = REPO_ROOT / "scripts" / "perlmutter" / "regular"
PROD_LAUNCHER = LAUNCH_DIR / "finetune_ddp_tft_lora_bandgap.sh"
SMOKE_LAUNCHER = LAUNCH_DIR / "smoke_lora_bandgap.sh"
LORA_LAUNCHERS = (PROD_LAUNCHER, SMOKE_LAUNCHER, LAUNCH_DIR / "diag_lora_bandgap_mem.sh")

# Stand-ins for values the launchers get from SLURM, the shell, or command substitution
FIXED = {
    "USER": "testuser",
    "SLURM_JOB_ID": "12345",
    "DEFAULT_RUN_DATE": "2026-01-01_00-00-00",
}

_VAR = re.compile(r"\$\{(\w+)(?::(\d+):(\d+)|:-([^}]*)|:\?[^}]*)?\}|\$(\w+)")


def sbatch_directives(path: Path) -> dict:
    """`#SBATCH --key=value` lines as {key: value}."""
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"#SBATCH\s+--([\w-]+)=(\S+)", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _expand(value: str, env: dict) -> str:
    def sub(m):
        name = m.group(1) or m.group(5)
        if m.group(2) is not None:  # ${VAR:offset:length}
            start, length = int(m.group(2)), int(m.group(3))
            return env[name][start : start + length]
        if name in env:
            return env[name]
        if m.group(4) is not None:  # ${VAR:-default}
            return _expand(m.group(4), env)
        raise KeyError(f"launcher references undefined ${name}")

    value = _VAR.sub(sub, value)
    arith = re.fullmatch(r"\$\(\((.*)\)\)", value)
    if arith:
        ints = {k: int(v) for k, v in env.items() if str(v).isdigit()}  # bash uses bare names
        return str(int(eval(arith.group(1), {"__builtins__": {}}, ints)))  # nosec - launcher math
    return value


def launcher_env(path: Path, pretrained_ckpt: str = "/tmp/pretrained.ckpt") -> dict:
    """Evaluate the launcher's top-level `NAME=value` assignments, in order."""
    directives = sbatch_directives(path)
    env = dict(FIXED)
    env.update(
        {
            "1": pretrained_ckpt,
            "SLURM_JOB_NUM_NODES": directives["nodes"],
            "SLURM_NTASKS_PER_NODE": directives["ntasks-per-node"],
        }
    )
    for line in path.read_text().splitlines():
        m = re.match(r"^(?:export\s+)?([A-Z_][A-Z0-9_]*)=(.*?)(?:\s+#.*)?$", line)
        if not m:
            continue
        name, raw = m.groups()
        if name in FIXED:
            continue
        raw = raw.strip()
        if re.match(r'"?\$\((?!\()', raw):  # command substitution, not $((arithmetic))
            if name == "DEFAULT_RUN_ID" or name.endswith("RUN_ID"):
                env[name] = "abcd1234"
            continue
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
            raw = raw[1:-1]
        if raw.startswith("${") and ":?" in raw:  # required positional arg
            raw = raw.split(":?")[0] + "}"
        env[name] = _expand(raw, env)
    return env


def train_overrides(path: Path, pretrained_ckpt: str = "/tmp/pretrained.ckpt") -> list:
    """The Hydra overrides the launcher passes to `zatom/train_fm.py`, shell-expanded."""
    text = path.read_text()
    m = re.search(r"zatom/train_fm\.py \\\n(.*?)\n\s*\"\n", text, re.S)
    assert m, f"no `train_fm.py` srun block found in {path.name}"
    env = launcher_env(path, pretrained_ckpt)
    overrides = []
    for line in m.group(1).splitlines():
        line = line.strip().rstrip("\\").strip()
        if not line or line.startswith("#"):
            continue
        line = _expand(line, env)
        key, _, value = line.partition("=")
        overrides.append(f"{key}={value.strip(chr(39))}")
    return overrides
