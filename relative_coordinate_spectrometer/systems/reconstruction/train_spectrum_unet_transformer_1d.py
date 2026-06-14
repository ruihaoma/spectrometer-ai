import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    print("Missing dependency: pyyaml. Please install it with: py -m pip install pyyaml", file=sys.stderr)
    raise SystemExit(1)

try:
    import numpy as np
except ModuleNotFoundError:
    print("Missing dependency: numpy. Please install it with: py -m pip install numpy", file=sys.stderr)
    raise SystemExit(1)

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError:
    print("Missing dependency: torch. Please install PyTorch before training.", file=sys.stderr)
    raise SystemExit(1)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("Missing dependency: matplotlib. Please install it before training.", file=sys.stderr)
    raise SystemExit(1)


SCRIPT_NAME = "train_spectrum_unet_transformer_1d.py"
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Support direct execution from the project directory.
from shared.data_loader.npy_spectrum_dataset import NpySpectrumDataset  # noqa: E402
from shared.spectrum_losses import composite_spectrum_loss  # noqa: E402
from shared.spectrum_unet_transformer_1d import SpectrumUNetTransformer1D  # noqa: E402


def read_text_auto(path):
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def load_yaml(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(read_text_auto(path))
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data, path


def project_path(path_text):
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def rel_path(path):
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def choose_device(device_name):
    if str(device_name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def make_run_dirs(config):
    output_cfg = config["output"]
    run_dir = project_path(output_cfg.get("runs_dir", "results/reproduction_runs")) / output_cfg.get(
        "run_name", "spectrum_unet_transformer_1d_v1"
    )
    example_dir = run_dir / "prediction_examples"
    run_dir.mkdir(parents=True, exist_ok=True)
    example_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, example_dir


def make_datasets(config):
    data_cfg = config["data"]
    dataset_dir = project_path(data_cfg["dataset_dir"])
    length = int(data_cfg["wavelength"]["point_count"])
    channels = len(data_cfg["input_channels"])
    kwargs = {
        "dataset_dir": dataset_dir,
        "x_file": data_cfg.get("x_file", "x.npy"),
        "y_file": data_cfg.get("y_file", "y.npy"),
        "wavelength_file": data_cfg.get("wavelength_file", "wavelength_nm.npy"),
        "split_file": data_cfg.get("split_file", "split.json"),
        "expected_channels": channels,
        "expected_length": length,
        "normalize": True,
    }
    return (
        dataset_dir,
        NpySpectrumDataset(split="train", **kwargs),
        NpySpectrumDataset(split="val", **kwargs),
    )


def make_model(config):
    model_cfg = config["model"]
    return SpectrumUNetTransformer1D(
        in_channels=int(model_cfg.get("in_channels", 4)),
        out_length=int(model_cfg.get("out_length", 2501)),
        base_channels=int(model_cfg.get("base_channels", 32)),
        trans_heads=int(model_cfg.get("trans_heads", 4)),
        trans_layers=int(model_cfg.get("trans_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        output_activation=str(model_cfg.get("output_activation", "none")),
    )


def move_batch(batch, device):
    return batch["x"].to(device, non_blocking=True), batch["y"].to(device, non_blocking=True)



def load_pretrained_checkpoint_if_needed(model, config, device):
    training_cfg = config.get("training", {})
    ckpt_path = training_cfg.get("pretrained_checkpoint", "")
    if not ckpt_path:
        print("[INFO] No pretrained_checkpoint specified. Training from scratch.")
        return model

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"pretrained_checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    print(f"[INFO] Loaded pretrained checkpoint from: {ckpt_path}")
    return model


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    totals = {"loss": 0.0, "weighted_l1": 0.0, "grad_l1": 0.0, "mse": 0.0}
    count = 0
    for batch in loader:
        x, y = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        losses = composite_spectrum_loss(pred, y)
        losses["loss"].backward()
        optimizer.step()

        batch_size = x.shape[0]
        count += batch_size
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    totals = {"loss": 0.0, "weighted_l1": 0.0, "grad_l1": 0.0, "mse": 0.0}
    count = 0
    for batch in loader:
        x, y = move_batch(batch, device)
        pred = model(x)
        losses = composite_spectrum_loss(pred, y)
        batch_size = x.shape[0]
        count += batch_size
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def save_loss_curve(history, path):
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_model_summary(path, config, dataset_dir, train_size, val_size, model, device):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    text = f"""# SpectrumUNetTransformer1D training summary

- script: {SCRIPT_NAME}
- created_at: {datetime.now().isoformat(timespec="seconds")}
- dataset_dir: {rel_path(dataset_dir)}
- train_size: {train_size}
- val_size: {val_size}
- device: {device}
- input_shape: [B, 4, 2501]
- output_shape: [B, 2501]
- model: {config['model']['name']}
- total_params: {total_params}
- trainable_params: {trainable_params}
- loss: weighted_l1 + 0.1 * grad_l1 + 0.05 * mse
- optimizer: AdamW
- epochs: {config['training']['epochs']}
- batch_size: {config['training']['batch_size']}
- learning_rate: {config['training']['learning_rate']}
- weight_decay: {config['training']['weight_decay']}
"""
    path.write_text(text, encoding="utf-8", newline="\n")


def save_checkpoint(path, model, optimizer, config, epoch, best_val_loss):
    torch.save(
        {
            "epoch": int(epoch),
            "best_val_loss": float(best_val_loss),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train SpectrumUNetTransformer1D.")
    parser.add_argument("--config", default="configs/train/relative_calib_mixed_v1_80k_train.yaml")
    args = parser.parse_args()

    config, config_path = load_yaml(args.config)
    set_seed(config["training"].get("seed", 42))
    device = choose_device(config["training"].get("device", "auto"))
    run_dir, _example_dir = make_run_dirs(config)

    dataset_dir, train_ds, val_ds = make_datasets(config)
    if len(train_ds) == 0:
        raise ValueError("train split is empty")
    if len(val_ds) == 0:
        raise ValueError("val split is empty")

    batch_size = int(config["training"].get("batch_size", 4))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = make_model(config).to(device)
    model = load_pretrained_checkpoint_if_needed(model, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 0.0002)),
        weight_decay=float(config["training"].get("weight_decay", 0.0001)),
    )
    scheduler_cfg = config.get("scheduler", {}) or {}
    scheduler = None
    if str(scheduler_cfg.get("name", "")).lower() == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_cfg.get("mode", "min"),
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 15)),
            threshold=float(scheduler_cfg.get("threshold", 1e-4)),
            cooldown=int(scheduler_cfg.get("cooldown", 3)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
        )

    first_batch = next(iter(train_loader))
    print("x batch shape:", list(first_batch["x"].shape))
    print("y batch shape:", list(first_batch["y"].shape))
    with torch.no_grad():
        first_pred = model(first_batch["x"].to(device))
    print("model output shape:", list(first_pred.shape))

    best_val_loss = float("inf")
    best_epoch = None
    history = []
    epochs = int(config["training"].get("epochs", 100))
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_loss(model, val_loader, device)
        train_loss = train_metrics["loss"]
        val_loss = val_metrics["loss"]
        if scheduler is not None:
            scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
                "train_components": train_metrics,
                "val_components": val_metrics,
            }
        )
        print(f"epoch {epoch:03d}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={current_lr:.8g}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(run_dir / "best_model.pt", model, optimizer, config, epoch, best_val_loss)

    save_checkpoint(run_dir / "last_model.pt", model, optimizer, config, epochs, best_val_loss)
    save_loss_curve(history, run_dir / "loss_curve.png")
    write_model_summary(run_dir / "model_summary.md", config, dataset_dir, len(train_ds), len(val_ds), model, device)

    metrics = {
        "script_name": SCRIPT_NAME,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": rel_path(config_path),
        "dataset_dir": rel_path(dataset_dir),
        "device": str(device),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
        "outputs": {
            "best_model": rel_path(run_dir / "best_model.pt"),
            "last_model": rel_path(run_dir / "last_model.pt"),
            "loss_curve": rel_path(run_dir / "loss_curve.png"),
            "model_summary": rel_path(run_dir / "model_summary.md"),
        },
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("training completed")
    print("best_model:", rel_path(run_dir / "best_model.pt"))
    print("last_model:", rel_path(run_dir / "last_model.pt"))
    print("metrics:", rel_path(run_dir / "metrics.json"))
    print("loss_curve:", rel_path(run_dir / "loss_curve.png"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
