import torch
from torch_geometric.data import Data
import concurrent.futures
import logging
import global_state
from api import get_node_data
from embedder import nlp

def build_graph_from_title(title, label, embedder, return_concepts=False):
    doc = nlp(title.lower())
    valid_pos = {"NOUN", "VERB", "ADJ", "PROPN"}
    concepts = [token.lemma_.lower().strip().replace(" ", "_") for token in doc if token.pos_ in valid_pos and not token.is_stop]
    concepts = list(dict.fromkeys(concepts)) 
    
    if not concepts:
        concepts = ["placeholder"]
        
    x = embedder.embed_nodes(concepts)
    edge_index = []
    edge_attr = []
    edge_weight = [] # NEW: Tracking empirical ConceptNet weights
    
    # 1. Base Sequential Edges (Default weight: 1.0)
    for i in range(len(concepts) - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i]) 
        edge_attr.extend([[0], [0]])
        edge_weight.extend([1.0, 1.0])
        
    # 2. Parallel Node Pre-fetching
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(get_node_data, concept) for concept in concepts]
        concurrent.futures.wait(futures) 
        
    # 3. Local Memory Graph Intersection
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            c_i = concepts[i]
            c_j = concepts[j]
            
            with global_state.cache_lock:
                node_i_cache = global_state.conceptnet_cache.get(c_i, {})
                
            if c_j in node_i_cache:
                for rel_label, weight in node_i_cache[c_j]:
                    rel_id = global_state.get_relation_id(rel_label)
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    edge_attr.extend([[rel_id], [rel_id]])
                    # Capture the empirical weight for both directions
                    edge_weight.extend([weight, weight]) 
                
    if not edge_index:
        edge_index = [[0, 0]]
        edge_attr = [[0]]
        edge_weight = [1.0]
        
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    
    # --- NEW: L_1 Topological Normalisation ---
    # Guarantees the sum of all edge weights in this specific graph equals 1.0
    weight_sum = edge_weight.sum()
    if weight_sum > 0:
        edge_weight = edge_weight / weight_sum
        
    y = torch.tensor([label], dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_weight=edge_weight, y=y)
    
    if return_concepts: 
        return data, concepts
    return data