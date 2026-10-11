import os
import argparse
import random
from typing import Dict, Any
import yaml
import numpy as np
import torch
from torch.optim import AdamW

from models import Radiant, RadiantConfig
from data import load_npz_dataset, build_dataloaders
from evaluate import (
    compute_metrics,
    apply_guarded_temporal_decision_rule,
    print_evaluation_summary,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> RadiantConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return RadiantConfig(
        metric_dim=cfg.get("metric_dim", 7),
        log_dim=cfg.get("log_dim", 6),
        trace_dim=cfg.get("trace_dim", 2),
        embed_dim=cfg.get("embed_dim", 64),
        gpt2_dim=cfg.get("gpt2_dim", 768),
        num_hosts=cfg.get("num_hosts", 12),
        gpt2_layers=cfg.get("gpt2_layers", 3),
        freeze_gpt2=cfg.get("freeze_gpt2", True),
        moe_rank=cfg.get("moe_rank", 4),
        moe_alpha=cfg.get("moe_alpha", 8.0),
        moe_dropout=cfg.get("moe_dropout", 0.05),
        moe_num_experts=cfg.get("moe_num_experts", 4),
        routing_uncertainty_threshold=cfg.get("routing_uncertainty_threshold", 0.40),
        service_prior_strength=cfg.get("service_prior_strength", 0.60),
        service_prior_mode=cfg.get("service_prior_mode", "cyclic"),
        cls_hidden_dim=cfg.get("cls_hidden_dim", 128),
        pred_weight=cfg.get("pred_weight", 0.1),
        moe_balance_weight=cfg.get("moe_balance_weight", 0.05),
    )


def train_one_epoch(model, dataloader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    count = 0

    for batch in dataloader:
        metrics = batch["metrics"].to(device)
        logs = batch["logs"].to(device)
        traces = batch["traces"].to(device)
        host_labels = batch["host_labels"].to(device)

        optimizer.zero_grad()
        outputs = model(metrics, logs, traces, labels=host_labels)
        loss = outputs["total_loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = metrics.size(0)
        total_loss += loss.item() * batch_size
        count += batch_size

    return total_loss / max(1, count)


@torch.no_grad()
def evaluate(model, dataloader, device) -> Dict[str, Any]:
    model.eval()
    all_scores = []
    all_targets = []

    for batch in dataloader:
        metrics = batch["metrics"].to(device)
        logs = batch["logs"].to(device)
        traces = batch["traces"].to(device)
        labels = batch["window_label"].cpu().numpy()

        outputs = model(metrics, logs, traces)
        scores = outputs["window_probs"].cpu().numpy()

        all_scores.extend(scores)
        all_targets.extend(labels)

    scores_arr = np.asarray(all_scores)
    targets_arr = np.asarray(all_targets)

    # Apply guarded temporal post-processing rule (C=3, M=2, theta=0.309, theta_h=0.4625)
    guarded_decisions = apply_guarded_temporal_decision_rule(
        scores_arr,
        threshold=0.309,
        bypass_threshold=0.4625,
        window_size=3,
        min_confirmations=2,
    )

    metrics = compute_metrics(
        scores_arr,
        targets_arr,
        threshold=0.5,
        post_processed_decisions=guarded_decisions,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Radiant: Online Multimodal Fault Diagnosis")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--train_file", type=str, required=True, help="Path to preprocessed train NPZ file")
    parser.add_argument("--val_file", type=str, required=True, help="Path to preprocessed validation NPZ file")
    parser.add_argument("--test_file", type=str, default=None, help="Path to preprocessed test NPZ file (optional)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[*] Running on device: {device}")

    # 1. Load configuration and dataset
    print(f"[*] Loading configuration from: {args.config}")
    config = load_config(args.config)

    print(f"[*] Loading training dataset from: {args.train_file}")
    train_dataset, adj = load_npz_dataset(args.train_file)
    print(f"[*] Loading validation dataset from: {args.val_file}")
    val_dataset, _ = load_npz_dataset(args.val_file)

    test_dataset = None
    if args.test_file and os.path.exists(args.test_file):
        print(f"[*] Loading test dataset from: {args.test_file}")
        test_dataset, _ = load_npz_dataset(args.test_file)

    loaders = build_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
    )

    if test_dataset is not None:
        train_loader, val_loader, test_loader = loaders
    else:
        train_loader, val_loader = loaders
        test_loader = None

    # 2. Build Radiant model
    print("[*] Initializing Radiant architecture...")
    model = Radiant(config, adjacency_matrix=adj).to(device)

    # 3. Setup optimizer (trainable parameters only)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"[*] Total parameters: {total_count / 1e6:.2f}M | Trainable parameters: {trainable_count / 1e6:.2f}M")

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)

    # 4. Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_f1 = 0.0

    print(f"[*] Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        curr_f1 = val_metrics.get("f1_guarded", val_metrics["f1_raw"])
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val F1 (Guarded): {curr_f1:.4f} | "
            f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}"
        )

        if curr_f1 > best_f1:
            best_f1 = curr_f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, "radiant_best.pt"))

    # 5. Final Evaluation on test set (if provided)
    eval_target_loader = test_loader if test_loader is not None else val_loader
    eval_split_name = "Test Split" if test_loader is not None else "Validation Split"
    print(f"\n[*] Evaluating best model on {eval_split_name}:")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "radiant_best.pt"), map_location=device))
    final_metrics = evaluate(model, eval_target_loader, device)
    print_evaluation_summary(final_metrics)


if __name__ == "__main__":
    main()
