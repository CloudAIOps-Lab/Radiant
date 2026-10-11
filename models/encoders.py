import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):
    """
    Encodes univariate/multivariate service telemetry (metrics or logs) into a latent subspace.
    Architecture: Linear -> LayerNorm -> GELU.
    """

    def __init__(self, raw_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Seq_Len, Num_Services, Raw_Dim] -> [Batch, Seq_Len, Num_Services, Embed_Dim]
        return self.proj(x)


class MetricEncoder(ModalityEncoder):
    """Encodes continuous resource metrics (CPU, Memory, Network, Disk)."""
    pass


class LogEncoder(ModalityEncoder):
    """Encodes structured and parsed log event distribution vectors."""
    pass


class SpatialGAT(nn.Module):
    """
    Spatial Graph Attention Layer operating over dynamic microservice call topologies.
    Aggregates incoming inter-service dependencies guided by the service invocation adjacency matrix.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Seq_Len, Num_Services, Embed_Dim]
        # adj: [Num_Services, Num_Services] service invocation topology matrix
        B, T, N, D = x.shape
        residual = x

        x_flat = x.view(B * T, N, D)
        q = self.w_q(x_flat).view(B * T, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x_flat).view(B * T, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x_flat).view(B * T, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply topological mask: allow valid directed call edges and self-loops
        adj_mask = (adj > 0).view(1, 1, N, N)
        eye_mask = torch.eye(N, dtype=torch.bool, device=x.device).view(1, 1, N, N)
        valid_mask = adj_mask | eye_mask

        scores = scores.masked_fill(~valid_mask, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B * T, N, D)
        out = self.out_proj(context).view(B, T, N, D)

        return self.norm(residual + self.dropout(out))


class TraceGraphEncoder(nn.Module):
    """
    Trace Graph Encoder for distributed tracing spans and invocation call topologies.
    Projects edge attributes (latency, request counts, error flags) and applies multi-layer
    SpatialGAT to propagate structural telemetry across caller-callee microservice graphs.
    """

    def __init__(
        self,
        trace_dim: int,
        embed_dim: int,
        num_hosts: int,
        gat_heads: int = 4,
        gat_dropout: float = 0.1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_hosts = num_hosts
        self.edge_proj = nn.Linear(trace_dim, embed_dim)
        self.gat_layers = nn.ModuleList([
            SpatialGAT(embed_dim=embed_dim, num_heads=gat_heads, dropout=gat_dropout)
            for _ in range(num_layers)
        ])

    def forward(self, traces: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # traces: [Batch, Seq_Len, Num_Services, Num_Services, Trace_Dim]
        # adj: [Num_Services, Num_Services]
        edge_feat = self.edge_proj(traces)  # [Batch, Seq_Len, Num_Services, Num_Services, Embed_Dim]
        node_feat = edge_feat.mean(dim=2)   # Aggregate incoming call edges: [Batch, Seq_Len, Num_Services, Embed_Dim]

        for gat in self.gat_layers:
            node_feat = gat(node_feat, adj)

        return node_feat
