#!/usr/bin/env python3
"""
Plot predictions vs actual values for the best model (CNN-BiLSTM Embedding).
Uses data from training_logs/cnn_bilstm_embedding_20260101_054609_run2.
Saves to images/best_model_predictions.png (same style as AMP_PhysioChem_AI plot_best_model.py).
Run from project root: python scripts/plot_best_model_predictions.py
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from src.embedding_datasets import EmbeddingPeptideDataset, embedding_collate_fn
from src.embedding_models import get_embedding_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_checkpoint(checkpoint_path, config_path):
    device = get_device()
    with open(config_path, "r") as f:
        config = json.load(f)
    model_config = config["model_config"]
    model = get_embedding_model(config["model_type"], **model_config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model, config, device


def make_predictions(model, dataset, device, batch_size=16):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=embedding_collate_fn
    )
    predictions, actuals = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)
            outputs = model(input_ids)
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0).flatten()
    actuals = np.concatenate(actuals, axis=0).flatten()
    return predictions, actuals


def plot_predictions_vs_actual(y_true, y_pred, model_name, output_path, show_stats=True):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pearson_r, pearson_p = pearsonr(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors="black", linewidth=0.5, zorder=3)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction", zorder=2)
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot(y_true, p(y_true), "b-", lw=2, label=f"Regression Line (slope={z[0]:.3f})", zorder=2, alpha=0.7)
    ax.set_xlabel("Actual Half-Life (log scale)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Predicted Half-Life (log scale)", fontsize=14, fontweight="bold")
    ax.set_title(f"Model Performance: {model_name}", fontsize=16, fontweight="bold", pad=20)
    # Metrics (R², MAE, RMSE, Pearson r) are still computed and returned,
    # but we intentionally do NOT draw the metrics legend box on the figure
    # to keep the plot clean and focused on the scatter and reference lines.
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return {"r2": r2, "mae": mae, "rmse": rmse, "pearson_r": pearson_r, "n": len(y_true)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot predicted vs actual for best model")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="Dataset split: train, val, or test (default: test)")
    args = parser.parse_args()
    split = args.split

    model_name = "cnn_bilstm_embedding_20260101_054609_run2"
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "Half_Life_cnn_bilstm_embedding_2.pt"
    config_path = PROJECT_ROOT / "training_logs" / model_name / "training_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    data_path_key = "train" if split == "train" else "validation" if split == "val" else "test"
    data_path = config["data_paths"][data_path_key]
    data_path = PROJECT_ROOT / data_path
    output_dir = PROJECT_ROOT / "images"
    output_path = output_dir / ("best_model_predictions.png" if split == "test" else f"best_model_predictions_{split}.png")

    print("Loading best model:", model_name)
    print("Checkpoint:", checkpoint_path)
    print("Data ({})".format(split), data_path)
    print("Output:", output_path)
    print()

    model, config, device = load_model_checkpoint(str(checkpoint_path), str(config_path))
    print("Model loaded on", device)

    dataset = EmbeddingPeptideDataset(str(data_path), max_length=config["max_length"])
    print("Loaded", len(dataset), "samples ({}, seq len ≤ 100)".format(split))

    predictions, actuals = make_predictions(
        model, dataset, device, batch_size=config.get("batch_size", 16)
    )
    print("Predictions completed for", len(predictions), "samples")

    output_dir.mkdir(parents=True, exist_ok=True)
    # For training split, hide stats box; keep it for val/test
    show_stats = split != "train"
    stats = plot_predictions_vs_actual(actuals, predictions, model_name, str(output_path), show_stats=show_stats)
    print("\nPlot saved to", output_path)
    print("R² = {:.4f}  MAE = {:.4f}  RMSE = {:.4f}  n = {}".format(
        stats["r2"], stats["mae"], stats["rmse"], stats["n"]))
    print("Done.")


if __name__ == "__main__":
    main()
