import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


def build_service_prior(
    num_hosts: int,
    num_experts: int,
    mode: str = "cyclic",
    custom_matrix: Optional[List[List[float]]] = None,
) -> torch.Tensor:
    """
    Constructs the structural service-to-expert prior bias matrix p_i in R^{num_hosts x num_experts}.
    Encodes service topological specializations into routing guidance.
    """
    if custom_matrix is not None:
        prior = torch.tensor(custom_matrix, dtype=torch.float32)
        if prior.shape != (num_hosts, num_experts):
            raise ValueError(f"custom_matrix shape mismatch: expected {(num_hosts, num_experts)}, got {tuple(prior.shape)}")
        return prior

    prior = torch.zeros(num_hosts, num_experts, dtype=torch.float32)
    normalized = mode.lower()
    if normalized == "cyclic":
        for h in range(num_hosts):
            prior[h, h % num_experts] = 1.0
        return prior

    if normalized == "block":
        block_size = max(1, (num_hosts + num_experts - 1) // num_experts)
        for h in range(num_hosts):
            prior[h, min(num_experts - 1, h // block_size)] = 1.0
        return prior

    raise ValueError(f"Unsupported service_prior mode: {mode}")


class ServiceAwareMoELoRALayer(nn.Module):
    """
    Service-Aware Mixture-of-Experts Low-Rank Adaptation (MoE-LoRA) with Dynamic Capacity Allocation.

    Mathematical formulation (Paper Section V):
      - Routing logit with structural prior: u_i(t) = g(h_i(t)) + lambda * p_i
      - Routing confidence: c_i(t) = max_j softmax(u_i(t))_j
      - Routing uncertainty proxy: gamma_i(t) = 1 - c_i(t)
      - Dynamic capacity allocation:
          k_i(t) = 2 if gamma_i(t) > tau_k else 1
    """

    def __init__(
        self,
        original_layer: nn.Module,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.05,
        num_experts: int = 4,
        router_hidden: int = 64,
        routing_uncertainty_threshold: float = 0.40,
        router_temperature: float = 1.0,
        num_hosts: int = 12,
        service_prior_mode: str = "cyclic",
        service_prior_strength: float = 0.60,
        service_prior_matrix: Optional[List[List[float]]] = None,
    ) -> None:
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.num_experts = num_experts
        self.routing_uncertainty_threshold = float(routing_uncertainty_threshold)
        self.router_temperature = max(1e-4, float(router_temperature))
        self.service_prior_strength = max(0.0, float(service_prior_strength))

        # Detect in/out dimensions (supports nn.Linear and HuggingFace Conv1D)
        if hasattr(original_layer, "nf"):
            d_in = original_layer.weight.shape[0]
            d_out = original_layer.nf
            self.is_conv1d = True
        else:
            d_in = original_layer.in_features
            d_out = original_layer.out_features
            self.is_conv1d = False

        self.d_in = d_in
        self.d_out = d_out

        # Specialized LoRA parameter matrices: A in R^{num_experts x d_in x rank}, B in R^{num_experts x rank x d_out}
        self.lora_A = nn.Parameter(torch.empty(num_experts, d_in, rank))
        self.lora_B = nn.Parameter(torch.empty(num_experts, rank, d_out))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Learnable gating router
        self.router = nn.Sequential(
            nn.Linear(d_in, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, num_experts),
        )

        # Topological service prior buffer
        prior = build_service_prior(
            num_hosts=num_hosts,
            num_experts=self.num_experts,
            mode=service_prior_mode,
            custom_matrix=service_prior_matrix,
        )
        self.register_buffer("service_prior", prior)

        self._reset_parameters()
        self._runtime_service_ids: Optional[torch.Tensor] = None
        self.last_router_probs: Optional[torch.Tensor] = None
        self._router_probs_for_loss: Optional[torch.Tensor] = None

    def _reset_parameters(self) -> None:
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.lora_A[i], a=5 ** 0.5)
            nn.init.zeros_(self.lora_B[i])

    def set_runtime_service_ids(self, service_ids: torch.Tensor) -> None:
        self._runtime_service_ids = service_ids.detach()

    def _route(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch_Services, Seq_Len, D] or [Batch_Services, D]
        pooled = x.mean(dim=1) if x.dim() == 3 else x
        logits = self.router(pooled) / self.router_temperature

        # Inject service-conditioned topological prior bias: u_i(t) = g(h_i) + lambda * p_i
        if (
            self.service_prior_strength > 0
            and self._runtime_service_ids is not None
            and self._runtime_service_ids.numel() == logits.shape[0]
        ):
            prior_bias = self.service_prior[self._runtime_service_ids.to(logits.device)]
            logits = logits + self.service_prior_strength * prior_bias

        probs = F.softmax(logits, dim=-1)
        self._router_probs_for_loss = probs

        # Dynamic capacity allocation: gamma_i(t) = 1 - max_j(probs)_j
        conf, top1_idx = torch.max(probs, dim=-1, keepdim=True)
        uncertainty = 1.0 - conf

        # If uncertainty > tau_k: allocate k=2 experts; otherwise allocate k=1 expert
        sparse_probs = torch.zeros_like(probs)
        top2_probs, top2_indices = torch.topk(probs, k=min(2, self.num_experts), dim=-1)

        for b in range(probs.size(0)):
            if uncertainty[b].item() > self.routing_uncertainty_threshold and self.num_experts >= 2:
                # Activate 2 experts
                indices = top2_indices[b]
                p = top2_probs[b]
                p_norm = p / p.sum().clamp_min(1e-9)
                sparse_probs[b, indices] = p_norm
            else:
                # Activate 1 expert
                idx = top1_idx[b]
                sparse_probs[b, idx] = 1.0

        self.last_router_probs = sparse_probs.detach()
        return sparse_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Base layer forward
        if self.is_conv1d:
            size_out = x.size()[:-1] + (self.d_out,)
            orig_out = torch.addmm(self.original.bias, x.view(-1, x.size(-1)), self.original.weight)
            orig_out = orig_out.view(size_out)
        else:
            orig_out = self.original(x)

        # 2. Dynamic MoE LoRA adaptation
        probs = self._route(x)  # [Batch_Services, num_experts]
        x_drop = self.dropout(x)

        # Compute expert delta weights: A @ B for each expert
        expert_weights = torch.bmm(self.lora_A, self.lora_B) * self.scaling
        mixed_weights = torch.einsum("be,eio->bio", probs, expert_weights)

        if x.dim() == 3:
            lora_out = torch.bmm(x_drop, mixed_weights)
        else:
            lora_out = torch.bmm(x_drop.unsqueeze(1), mixed_weights).squeeze(1)

        return orig_out + lora_out

    def load_balance_loss(self) -> Optional[torch.Tensor]:
        if self._router_probs_for_loss is None:
            return None
        density = self._router_probs_for_loss.mean(dim=0)
        return (self.num_experts * torch.sum(density ** 2)) - 1.0

    def clear_routing_cache(self) -> None:
        self.last_router_probs = None
        self._router_probs_for_loss = None
        self._runtime_service_ids = None
