import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool
import math

def calculate_irf_weights(graphs, num_relations):
    """Calculates Inverse Relation Frequency with Sequential Exemption."""
    edge_counts = torch.zeros(num_relations)
    for data in graphs:
        unique, counts = torch.unique(data.edge_attr, return_counts=True)
        for u, c in zip(unique, counts):
            edge_counts[u] += c
            
    total_edges = edge_counts.sum()
    irf_weights = torch.zeros(num_relations)
    
    for i in range(num_relations):
        if edge_counts[i] > 0: 
            irf_weights[i] = math.log((total_edges / edge_counts[i]) + 1)
        else: 
            irf_weights[i] = 1.0 
            
    # --- NEW: Sequential Protection ---
    # The sequential relation (index 0) must form the backbone. 
    # We assign it the maximum penalty value found so it is never squashed.
    if num_relations > 1:
        max_irf = irf_weights[1:].max()
        irf_weights[0] = max_irf
            
    weight_sum = irf_weights.sum()
    if weight_sum > 0:
        irf_weights = irf_weights / weight_sum
        
    return irf_weights

def semantic_contrastive_loss(node_embeddings, edge_index, edge_attr):
    semantic_mask = edge_attr.squeeze(-1) > 0
    sem_edges = edge_index[:, semantic_mask]
    
    if sem_edges.size(1) == 0: 
        return node_embeddings.sum() * 0.0 
        
    u = node_embeddings[sem_edges[0]]
    v = node_embeddings[sem_edges[1]]
    
    cos_sim = F.cosine_similarity(u, v, dim=-1)
    loss = F.mse_loss(cos_sim, torch.ones_like(cos_sim))
    return loss

class SarcasmGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_embed_dim, num_relations, irf_weights):
        super(SarcasmGNN, self).__init__()
        self.edge_embedding = torch.nn.Embedding(num_relations, edge_embed_dim)
        torch.nn.init.orthogonal_(self.edge_embedding.weight)
        self.register_buffer('irf_weights', irf_weights)
        
        self.gat1 = GATv2Conv(input_dim, hidden_dim, heads=2, edge_dim=edge_embed_dim, dropout=0.3, concat=False)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, heads=1, edge_dim=edge_embed_dim, dropout=0.3, concat=False)
        
        self.skip_proj = torch.nn.Linear(input_dim, hidden_dim)
        
        # Jumping Knowledge Dimension
        jk_dim = input_dim + hidden_dim + hidden_dim
        
        # --- NEW: Classifier Input Doubled for Dual Pooling (Max + Mean) ---
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(jk_dim * 2, hidden_dim), 
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4), 
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, edge_weight, batch):
        raw_edge_feats = self.edge_embedding(edge_attr.squeeze(-1))
        normalized_edge_feats = F.normalize(raw_edge_feats, p=2, dim=-1)
        
        irf_scalars = self.irf_weights[edge_attr.squeeze(-1)].unsqueeze(-1)
        empirical_scalars = edge_weight.unsqueeze(-1)
        scaled_edge_feats = normalized_edge_feats * irf_scalars * empirical_scalars
        
        h0 = x 
        x_skip = self.skip_proj(x)
        
        x = F.dropout(x, p=0.3, training=self.training)
        x_gat = self.gat1(x, edge_index, edge_attr=scaled_edge_feats)
        h1 = F.relu(x_gat + x_skip)
        
        h1_drop = F.dropout(h1, p=0.3, training=self.training)
        h2 = self.gat2(h1_drop, edge_index, edge_attr=scaled_edge_feats)
        h2 = F.relu(h2)
        
        jk_node_embeddings = torch.cat([h0, h1, h2], dim=-1)
        
        # --- NEW: Dual-Aggregation Pooling ---
        x_max = global_max_pool(jk_node_embeddings, batch)
        x_mean = global_mean_pool(jk_node_embeddings, batch)
        x_pool = torch.cat([x_max, x_mean], dim=-1) # Concatenate features
        
        out = self.classifier(x_pool)
        return out, h2