from __future__ import annotations

from pathlib import Path


def train_baseline(artifacts_dir: Path, epochs: int) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_file = artifacts_dir / "baseline_model.txt"
    out_file.write_text(f"trained baseline for {epochs} epochs\n", encoding="utf-8")
    return out_file
