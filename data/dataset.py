import os
from typing import Dict, Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MultimodalTimeSeriesDataset(Dataset):
    """
    Standard dataset container for multimodal time-series fault diagnosis in microservices.

    Expected input tensor geometries:
      - metrics: [Batch, Seq_Len, Num_Services, Metric_Dim]
      - logs:    [Batch, Seq_Len, Num_Services, Log_Dim]
      - traces:  [Batch, Seq_Len, Num_Services, Num_Services, Trace_Dim]
      - window_labels: [Batch] (binary anomaly labels: 0 for normal, 1 for anomalous)
      - host_labels:   [Batch, Num_Services] (service-level fault attribution indicators)
    """

    def __init__(
        self,
        metrics: Union[np.ndarray, torch.Tensor],
        logs: Union[np.ndarray, torch.Tensor],
        traces: Union[np.ndarray, torch.Tensor],
        window_labels: Union[np.ndarray, torch.Tensor],
        host_labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        assert len(metrics) == len(logs) == len(traces) == len(window_labels), (
            f"Mismatched dataset lengths: metrics ({len(metrics)}), logs ({len(logs)}), "
            f"traces ({len(traces)}), window_labels ({len(window_labels)})"
        )

        self.metrics = torch.as_tensor(metrics, dtype=torch.float32)
        self.logs = torch.as_tensor(logs, dtype=torch.float32)
        self.traces = torch.as_tensor(traces, dtype=torch.float32)
        self.window_labels = torch.as_tensor(window_labels, dtype=torch.long)

        if host_labels is not None:
            self.host_labels = torch.as_tensor(host_labels, dtype=torch.float32)
        else:
            num_samples, _, num_services = self.metrics.shape[:3]
            self.host_labels = torch.zeros((num_samples, num_services), dtype=torch.float32)

    def __len__(self) -> int:
        return self.metrics.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "metrics": self.metrics[idx],
            "logs": self.logs[idx],
            "traces": self.traces[idx],
            "window_label": self.window_labels[idx],
            "host_labels": self.host_labels[idx],
        }


def load_npz_dataset(file_path: str) -> Tuple[MultimodalTimeSeriesDataset, Optional[torch.Tensor]]:
    """
    Loads preprocessed multimodal telemetry dataset and optional topology adjacency matrix from an NPZ file.

    The NPZ file should contain the following arrays:
      - 'metrics': array of shape [N, T, S, D_m]
      - 'logs': array of shape [N, T, S, D_l]
      - 'traces': array of shape [N, T, S, S, D_t]
      - 'window_labels': array of shape [N]
      - 'host_labels' (optional): array of shape [N, S]
      - 'adj' or 'adjacency_matrix' (optional): topology graph matrix of shape [S, S]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    with np.load(file_path, allow_pickle=True) as data:
        metrics = data["metrics"]
        logs = data["logs"]
        traces = data["traces"]
        window_labels = data["window_labels"]
        host_labels = data["host_labels"] if "host_labels" in data else None

        adj = None
        if "adj" in data:
            adj = torch.from_numpy(data["adj"]).float()
        elif "adjacency_matrix" in data:
            adj = torch.from_numpy(data["adjacency_matrix"]).float()

    dataset = MultimodalTimeSeriesDataset(
        metrics=metrics,
        logs=logs,
        traces=traces,
        window_labels=window_labels,
        host_labels=host_labels,
    )
    return dataset, adj


def build_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
    """Constructs standard PyTorch DataLoaders for training, validation, and testing."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        return train_loader, val_loader, test_loader

    return train_loader, val_loader
