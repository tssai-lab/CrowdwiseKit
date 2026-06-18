from __future__ import annotations

"""Optional PyTorch implementation of the neural StyleGCL-Net modules.

The default CLI uses the NumPy reproduction so the project runs in restricted
environments. Install torch to use these classes for the paper-style trainable
model.
"""


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "stylegcl.torch_model requires PyTorch. Install torch to use the "
            "trainable neural implementation."
        ) from exc
    return torch, nn, F


class SemanticDisentangler:
    """Factory wrapper to avoid importing torch at module import time."""

    @staticmethod
    def build(input_dim: int = 768, hidden_dim: int = 64, num_heads: int = 8):
        torch, nn, F = _require_torch()

        class _SemanticDisentangler(nn.Module):
            def __init__(self):
                super().__init__()
                self.answer_proj = nn.Linear(input_dim, hidden_dim)
                self.task_proj = nn.Linear(input_dim, hidden_dim)
                self.cross_attn = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    batch_first=True,
                )
                self.content = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.style = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                )

            def forward(self, answer_emb, task_emb):
                answer = self.answer_proj(answer_emb).unsqueeze(1)
                task = self.task_proj(task_emb).unsqueeze(1)
                attended, _weights = self.cross_attn(query=answer, key=task, value=task)
                attended = attended.squeeze(1)
                residual = answer.squeeze(1) - attended
                content = F.normalize(self.content(attended), dim=-1)
                style_raw = self.style(residual)
                leakage = (style_raw * content).sum(dim=-1, keepdim=True) * content
                style = F.normalize(style_raw - leakage, dim=-1)
                reconstruction = self.decoder(content)
                return content, style, reconstruction

        return _SemanticDisentangler()


class WorkerGAT:
    @staticmethod
    def build(dim: int = 64, hidden_dim: int = 128, heads: int = 4, layers: int = 2):
        torch, nn, F = _require_torch()

        class _GATLayer(nn.Module):
            def __init__(self, in_dim, out_dim, heads):
                super().__init__()
                self.heads = heads
                self.out_dim = out_dim
                self.linear = nn.Linear(in_dim, out_dim * heads, bias=False)
                self.attn = nn.Parameter(torch.empty(heads, 2 * out_dim))
                nn.init.xavier_uniform_(self.attn)

            def forward(self, x, adjacency):
                n = x.size(0)
                h = self.linear(x).view(n, self.heads, self.out_dim)
                src = h.unsqueeze(1).expand(n, n, self.heads, self.out_dim)
                dst = h.unsqueeze(0).expand(n, n, self.heads, self.out_dim)
                pair = torch.cat([src, dst], dim=-1)
                logits = (pair * self.attn.view(1, 1, self.heads, -1)).sum(dim=-1)
                logits = F.leaky_relu(logits, negative_slope=0.2)
                mask = adjacency.bool().unsqueeze(-1) | torch.eye(n, device=x.device).bool().unsqueeze(-1)
                logits = logits.masked_fill(~mask, -1e9)
                alpha = torch.softmax(logits, dim=1)
                out = (alpha.unsqueeze(-1) * dst).sum(dim=1)
                return F.elu(out.flatten(start_dim=1))

        class _WorkerGAT(nn.Module):
            def __init__(self):
                super().__init__()
                modules = []
                in_dim = dim
                for layer_idx in range(layers):
                    out_dim = hidden_dim // heads if layer_idx < layers - 1 else dim // heads
                    modules.append(_GATLayer(in_dim, out_dim, heads))
                    in_dim = out_dim * heads
                self.layers = nn.ModuleList(modules)

            def forward(self, x, adjacency):
                for layer in self.layers:
                    x = layer(x, adjacency)
                return F.normalize(x, dim=-1)

        return _WorkerGAT()


def style_losses(content, style, reconstruction, task_emb, worker_index, temperature: float = 0.5):
    torch, _nn, F = _require_torch()
    recon_loss = F.mse_loss(reconstruction, task_emb)
    orth_loss = ((content * style).sum(dim=-1) ** 2).mean()
    consistency_terms = []
    for worker in torch.unique(worker_index):
        mask = worker_index == worker
        if int(mask.sum()) > 1:
            local = style[mask]
            consistency_terms.append(((local - local.mean(dim=0, keepdim=True)) ** 2).mean())
    consistency_loss = torch.stack(consistency_terms).mean() if consistency_terms else style.new_tensor(0.0)
    sim = style @ style.t() / temperature
    labels = torch.arange(style.size(0), device=style.device)
    contrastive = F.cross_entropy(sim, labels)
    return {
        "content_reconstruction": recon_loss,
        "orthogonal": orth_loss,
        "style_consistency": consistency_loss,
        "info_nce": contrastive,
    }
