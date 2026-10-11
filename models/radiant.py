from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import MetricEncoder, LogEncoder, TraceGraphEncoder
from .moe_layer import ServiceAwareMoELoRALayer
from .losses import RadiantLoss


@dataclass
class RadiantConfig:
    """Hyperparameter configuration for Radiant."""
    # Telemetry dimensions
    metric_dim: int = 7
    log_dim: int = 6
    trace_dim: int = 2
    embed_dim: int = 64
    gpt2_dim: int = 768
    num_hosts: int = 12

    # Backbone parameters
    gpt2_layers: int = 3
    freeze_gpt2: bool = True
    train_ln: bool = True
    train_wpe: bool = True

    # MoE-LoRA Dynamic Capacity Allocation
    moe_rank: int = 4
    moe_alpha: float = 8.0
    moe_dropout: float = 0.05
    moe_num_experts: int = 4
    routing_uncertainty_threshold: float = 0.40  # tau_k
    service_prior_strength: float = 0.60         # lambda
    service_prior_mode: str = "cyclic"
    router_temperature: float = 1.0

    # Diagnostic Deviation Classifier & Loss Weights
    cls_hidden_dim: int = 128
    pred_weight: float = 0.1
    moe_balance_weight: float = 0.05


def _load_gpt2_backbone(config: RadiantConfig) -> nn.Module:
    """Loads a truncated 3-layer GPT-2 backbone, falling back to random init if offline."""
    from transformers import GPT2Config, GPT2Model
    try:
        gpt2 = GPT2Model.from_pretrained("gpt2", local_files_only=False)
    except Exception:
        gpt2_conf = GPT2Config(
            n_embd=config.gpt2_dim,
            n_layer=config.gpt2_layers,
            n_head=12,
            vocab_size=50257,
        )
        gpt2 = GPT2Model(gpt2_conf)

    # Retain only the first N layers
    gpt2.h = nn.ModuleList(list(gpt2.h)[:config.gpt2_layers])

    if config.freeze_gpt2:
        for param in gpt2.parameters():
            param.requires_grad = False
        if config.train_ln:
            for name, param in gpt2.named_parameters():
                if "ln" in name:
                    param.requires_grad = True
        if config.train_wpe:
            for name, param in gpt2.named_parameters():
                if "wpe" in name:
                    param.requires_grad = True

    return gpt2


def _apply_service_aware_moe(gpt2: nn.Module, config: RadiantConfig) -> nn.Module:
    """Injects Service-Aware MoE LoRA dynamic adapter into the last retained Transformer block."""
    last_block = gpt2.h[-1]
    last_block.attn.c_attn = ServiceAwareMoELoRALayer(
        original_layer=last_block.attn.c_attn,
        rank=config.moe_rank,
        alpha=config.moe_alpha,
        dropout=config.moe_dropout,
        num_experts=config.moe_num_experts,
        routing_uncertainty_threshold=config.routing_uncertainty_threshold,
        router_temperature=config.router_temperature,
        num_hosts=config.num_hosts,
        service_prior_mode=config.service_prior_mode,
        service_prior_strength=config.service_prior_strength,
    )
    return gpt2


class Radiant(nn.Module):
    """
    Radiant: Bounded-Latency Online Multimodal Fault Diagnosis Framework for Cloud Microservices.

    Pipeline:
      1. Multimodal Observability Modeling: encodes Metrics, Logs, and Traces (via SpatialGAT)
         into unified 768-dimensional service representations.
      2. Behavioral Evolution Learning: predicts expected future service behavior via frozen GPT-2
         equipped with a service-conditioned dynamic capacity allocation MoE-LoRA adapter.
      3. 768-Dimensional Deviation Space Diagnosis: compares predicted expected states against
         observed telemetry, formulating online fault detection and root-cause localization
         directly in the high-dimensional latent deviation space.
    """

    def __init__(
        self,
        config: RadiantConfig,
        adjacency_matrix: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.config = config

        if adjacency_matrix is None:
            adj = torch.ones(config.num_hosts, config.num_hosts, dtype=torch.float32)
            adj.fill_diagonal_(0.0)
        else:
            adj = adjacency_matrix.clone().float()
            adj.fill_diagonal_(0.0)
        self.register_buffer("adj", adj)

        # 1. Multimodal encoders
        self.metric_encoder = MetricEncoder(config.metric_dim, config.embed_dim)
        self.log_encoder = LogEncoder(config.log_dim, config.embed_dim)
        self.trace_encoder = TraceGraphEncoder(
            trace_dim=config.trace_dim,
            embed_dim=config.embed_dim,
            num_hosts=config.num_hosts,
        )

        # 2. Multimodal early fusion projection -> [Batch, Seq_Len, Num_Services, 768]
        self.fusion_proj = nn.Sequential(
            nn.Linear(config.embed_dim * 3, config.gpt2_dim),
            nn.LayerNorm(config.gpt2_dim),
            nn.GELU(),
        )

        # 3. Behavioral evolution backbone with dynamic MoE-LoRA
        gpt2_raw = _load_gpt2_backbone(config)
        self.gpt2 = _apply_service_aware_moe(gpt2_raw, config)

        # 4. Latent deviation prediction and diagnosis heads
        self.pred_head = nn.Linear(config.gpt2_dim, config.gpt2_dim)
        self.deviation_encoder = nn.Sequential(
            nn.Linear(config.gpt2_dim, config.cls_hidden_dim),
            nn.LayerNorm(config.cls_hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.cls_hidden_dim, config.cls_hidden_dim // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(config.cls_hidden_dim // 2, 2),
        )

        # 5. Training criterion
        self.loss_fn = RadiantLoss(
            cls_weight=1.0,
            pred_weight=config.pred_weight,
            moe_balance_weight=config.moe_balance_weight,
        )

    def _set_service_ids(self, service_ids: torch.Tensor) -> None:
        for m in self.gpt2.modules():
            if isinstance(m, ServiceAwareMoELoRALayer):
                m.set_runtime_service_ids(service_ids)

    def _collect_moe_loss(self) -> torch.Tensor:
        losses = []
        for m in self.gpt2.modules():
            if isinstance(m, ServiceAwareMoELoRALayer):
                loss = m.load_balance_loss()
                if loss is not None:
                    losses.append(loss)
        if not losses:
            return torch.zeros((), device=next(self.parameters()).device)
        return torch.stack(losses).mean()

    def _clear_cache(self) -> None:
        for m in self.gpt2.modules():
            if isinstance(m, ServiceAwareMoELoRALayer):
                m.clear_routing_cache()

    def forward(
        self,
        metrics: torch.Tensor,
        logs: torch.Tensor,
        traces: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Radiant.
        Inputs:
          - metrics: [Batch, Seq_Len, Num_Services, Metric_Dim]
          - logs:    [Batch, Seq_Len, Num_Services, Log_Dim]
          - traces:  [Batch, Seq_Len, Num_Services, Num_Services, Trace_Dim]
          - labels (optional): [Batch, Num_Services] service-level fault indicators
        """
        B, T, N = metrics.shape[:3]

        # 1. Multimodal Encoding
        h_m = self.metric_encoder(metrics)
        h_l = self.log_encoder(logs)
        h_t = self.trace_encoder(traces, self.adj)

        # 2. Multimodal Early Fusion -> [B, T, N, 768]
        fused = torch.cat([h_m, h_l, h_t], dim=-1)
        fused = self.fusion_proj(fused)

        # Permute to sequence per service: [B*N, T, 768]
        gpt_in = fused.permute(0, 2, 1, 3).contiguous().view(B * N, T, self.config.gpt2_dim)

        service_ids = torch.arange(N, device=metrics.device).repeat(B)
        self._set_service_ids(service_ids)

        # 3. Backbone Forward
        gpt_out = self.gpt2(inputs_embeds=gpt_in).last_hidden_state  # [B*N, T, 768]

        # 4. Behavioral Evolution Prediction & Latent Deviation Computation
        # Predict the current expected latent state from historical context
        pred_last = self.pred_head(gpt_out[:, -2, :])               # Predicted state h_hat(t)
        actual_last = gpt_in[:, -1, :]                               # Observed state h(t)
        deviation = (pred_last - actual_last).view(B, N, self.config.gpt2_dim)  # Deviation d_i(t) in R^768

        # 5. Deviation Space Diagnosis
        anomaly_feat = self.deviation_encoder(deviation)
        logits = self.classifier(anomaly_feat)  # [B, N, 2]
        probs = F.softmax(logits, dim=-1)

        # Window-level anomaly score is the maximum service anomaly probability
        service_fault_probs = probs[..., 1]     # [B, N]
        window_probs = service_fault_probs.max(dim=1).values  # [B]

        outputs = {
            "logits": logits,
            "probs": probs,
            "service_fault_probs": service_fault_probs,
            "window_probs": window_probs,
            "deviation": deviation,
        }

        # 6. Training Loss
        if labels is not None:
            moe_loss = self._collect_moe_loss()
            loss_dict = self.loss_fn(
                cls_logits=logits.view(B * N, 2),
                cls_targets=labels.view(B * N).long(),
                pred_last=pred_last,
                actual_last=actual_last,
                moe_balance_loss=moe_loss,
            )
            outputs.update(loss_dict)

        self._clear_cache()
        return outputs
