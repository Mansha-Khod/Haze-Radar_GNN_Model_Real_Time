# inference_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseGATv2Layer(nn.Module):
    """
    Dense (inference-only) GATv2-like layer.
    This computes attention over all nodes but uses an adjacency mask
    to zero-out non-edges. Works without torch_geometric.
    NOTE: O(N^2) memory/time. Fine for tens-hundreds of nodes.
    """
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.0, concat=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # Linear projection W
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        # Attention mechanism: we'll use a small MLP per head
        # We'll parametrize as a single Linear for speed: maps 2*out_dim -> 1, applied per head
        self.att = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att.weight)

    def forward(self, x, adj_mask):
        """
        x: (N, in_dim)
        adj_mask: (N, N) bool or 0/1 mask where adj_mask[i,j]=1 means j -> i edge exists
                  (i.e., messages from j to i are considered)
        Returns: output (N, heads*out_dim) if concat, else (N, out_dim)
        """
        N = x.shape[0]
        h = self.lin(x)  # (N, heads*out_dim)
        h = h.view(N, self.heads, self.out_dim)  # (N, heads, out_dim)

        # Compute pairwise attention scores in a dense way:
        # expand and compute e_ij for all i,j
        h_i = h.unsqueeze(1).expand(N, N, self.heads, self.out_dim)  # (N, N, heads, out)
        h_j = h.unsqueeze(0).expand(N, N, self.heads, self.out_dim)  # (N, N, heads, out)

        # concat along feature dim
        cat = torch.cat([h_i, h_j], dim=-1)  # (N, N, heads, 2*out)
        # collapse heads into batch dimension for linear op efficiency
        cat_flat = cat.view(N * N * self.heads, 2 * self.out_dim)
        e = self.att(cat_flat)  # (N*N*heads, 1)
        e = e.view(N, N, self.heads)  # (N, N, heads)
        e = self.leaky_relu(e)

        # Mask with adjacency: adj_mask is (N,N) with 1 where edge exists j->i
        if adj_mask.dtype != torch.bool:
            adj_bool = adj_mask != 0
        else:
            adj_bool = adj_mask

        # Set scores for non-edges to -inf before softmax
        minus_inf = -9e15
        e_masked = torch.where(adj_bool.unsqueeze(-1), e, torch.full_like(e, minus_inf))

        # Softmax over source nodes j (dim=1) for each target i and each head
        attn = torch.softmax(e_masked, dim=1)  # (N, N, heads)

        # Apply dropout if set
        if self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        # Compute weighted sum of h_j for each target node i
        # attn: (N, N, heads), h_j: (N, heads, out)
        # We need for each i: sum_j attn[i,j,head] * h[j,head,:]
        # Do batched matmul: for each head, attn[:, :, head] @ h[:, head, :]
        outputs = []
        for head in range(self.heads):
            a = attn[:, :, head]  # (N, N)
            hj = h[:, head, :]    # (N, out)
            out_head = torch.matmul(a, hj)  # (N, out)
            outputs.append(out_head)

        out = torch.stack(outputs, dim=1)  # (N, heads, out)
        if self.concat:
            out = out.view(N, self.heads * self.out_dim)  # (N, heads*out)
        else:
            out = out.mean(dim=1)  # (N, out)

        return out


class InferenceRealtimeHazeGNN(nn.Module):
    """
    Inference-only version of the GNN — pure PyTorch (no torch_geometric).
    Must be loaded with converted state_dict produced by convert_checkpoint.py.
    """
    def __init__(self, in_feats, hidden, out_feats, num_heads=4, dropout=0.2):
        super().__init__()
        self.hidden = hidden
        self.out_feats = out_feats
        self.layer1 = DenseGATv2Layer(in_feats, hidden // num_heads, heads=num_heads,
                                     dropout=dropout, concat=True)
        # After concatenation, hidden dimension is num_heads * (hidden // num_heads) == hidden
        self.ln1 = nn.LayerNorm(hidden)
        self.layer2 = DenseGATv2Layer(hidden, hidden // num_heads, heads=num_heads,
                                     dropout=dropout, concat=True)
        self.ln2 = nn.LayerNorm(hidden)

        # prediction head and uncertainty head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_feats)
        )
        self.unc_head = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, out_feats),
            nn.Softplus()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_mask):
        # x: (N, in_feats), adj_mask: (N, N)
        h = self.layer1(x, adj_mask)
        h = self.ln1(h)
        h = F.elu(h)
        h = self.dropout(h)

        h2 = self.layer2(h, adj_mask)
        h2 = self.ln2(h2)
        h = h + h2
        h = F.elu(h)

        pred = self.pred_head(h)
        unc = self.unc_head(h)
        return pred, unc
