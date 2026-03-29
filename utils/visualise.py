import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import torch
import logging
import math
import os
from torch_geometric.utils import to_networkx
import utils.global_state as global_state

weight_history = {}

def save_gnn_graph(data, concepts, headline, true_label, pred_label, save_path):
    # Convert to a directed graph temporarily so NetworkX respects the edge curvature directions
    G = to_networkx(data, to_undirected=False)
    mapping = {i: concept for i, concept in enumerate(concepts)}
    G = nx.relabel_nodes(G, mapping)

    seq_only, semantic_only, both = [], [], []
    seq_widths, semantic_widths, both_widths = [], [], []
    edge_labels = {}

    # Track seen undirected pairs to prevent drawing duplicate curves in both directions
    seen_pairs = set()

    for u, v in G.edges():
        # FIX 1: Safely hash the undirected pair without type-sorting conflicts
        pair_key = frozenset([u, v])
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        
        # FIX 2: Safely initialise state flags to prevent UnboundLocalError
        is_sequential = False
        is_conceptnet = False
        semantic_links = []
        
        # Check if both nodes are mapped concept strings
        if isinstance(u, str) and isinstance(v, str) and u in concepts and v in concepts:
            idx_u, idx_v = concepts.index(u), concepts.index(v)
            c_u = u.lower().strip().replace(" ", "_")
            c_v = v.lower().strip().replace(" ", "_")
            
            with global_state.cache_lock:
                node_u_cache = global_state.conceptnet_cache.get(c_u, {})
                node_v_cache = global_state.conceptnet_cache.get(c_v, {})
                
            if c_v in node_u_cache:
                semantic_links.extend(node_u_cache[c_v])
            if c_u in node_v_cache:
                semantic_links.extend(node_v_cache[c_u])
                
            is_conceptnet = len(semantic_links) > 0
            is_sequential = abs(idx_u - idx_v) == 1
        else:
            # If nodes are raw token integers (unmapped), assess their sequential adjacency safely
            try:
                idx_u = concepts.index(u) if u in concepts else int(u)
                idx_v = concepts.index(v) if v in concepts else int(v)
                is_sequential = abs(idx_u - idx_v) == 1
            except (ValueError, TypeError):
                is_sequential = False
            
        rel_label = ""
        max_weight = 0.0
        
        if is_conceptnet:
            unique_rels = list(set([rel[0] for rel in semantic_links]))
            sem_labels = " | ".join(unique_rels)
            max_weight = max([rel[1] for rel in semantic_links])
            
            # Logarithmic squashing for width
            dynamic_width = 1.0 + (math.log1p(max_weight) * 0.7)
            
            if is_sequential:
                rel_label = f"Seq + {sem_labels}"
                both.append((u, v))
                both_widths.append(dynamic_width + 0.5) 
            else:
                rel_label = sem_labels
                semantic_only.append((u, v))
                semantic_widths.append(dynamic_width)
        elif is_sequential:
            seq_only.append((u, v))
            seq_widths.append(1.0) 
            
        if rel_label:
            if max_weight > 0:
                edge_labels[(u, v)] = f"{rel_label}\n({max_weight:.1f})"
            else:
                edge_labels[(u, v)] = rel_label
                
    plt.figure(figsize=(12, 8))
    
    # Increase k to push nodes further apart, preventing text overlap
    pos = nx.spring_layout(G.to_undirected(), seed=42, k=1.4)
    
    # 1. Draw Curved Edges First (so they sit behind the text)
    if seq_only:
        nx.draw_networkx_edges(G, pos, edgelist=seq_only, width=seq_widths, alpha=0.3, style='solid', edge_color='#7f8c8d', arrows=False)
        
    if semantic_only:
        nx.draw_networkx_edges(G, pos, edgelist=semantic_only, width=semantic_widths, alpha=0.75, edge_color='#e67e22', connectionstyle='arc3,rad=0.25', arrows=True, arrowstyle='-')
        
    if both:
        nx.draw_networkx_edges(G, pos, edgelist=both, width=both_widths, alpha=0.85, edge_color='#8e44ad', connectionstyle='arc3,rad=0.15', arrows=True, arrowstyle='-')
    
    # 2. Draw Edge Labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color='#d35400', alpha=0.85, label_pos=0.4)

    # 3. Draw Typography Nodes
    bbox_props = dict(boxstyle="round,pad=0.4", fc="#fdfefe", ec="#2980b9", lw=1.5, alpha=0.95)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', font_color='#154360', bbox=bbox_props)
    
    match_colour = "#27ae60" if true_label == pred_label else "#c0392b"
    plt.title(f"{headline[:70]}...\nTrue: {true_label} | Pred: {pred_label}", fontsize=13, fontweight='bold', color=match_colour, pad=20)
    plt.axis('off')
    
    l1 = mlines.Line2D([], [], color='#7f8c8d', linestyle='solid', linewidth=1.0, label='Sequential Backbone')
    l2 = mlines.Line2D([], [], color='#e67e22', linewidth=2.0, label='Semantic Prior (ConceptNet)')
    l3 = mlines.Line2D([], [], color='#8e44ad', linewidth=3.0, label='Hybrid (Sequential + Semantic)')
    plt.legend(handles=[l1, l2, l3], loc='lower right', frameon=True, facecolor='#f8f9f9', edgecolor='#bdc3c7')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def track_and_log_weights(model, vocab, epoch):
    logging.info(f"\n--- Relation Weights (Epoch {epoch+1}) ---")
    model.eval()
    with torch.no_grad():
        for rel_label, rel_id in vocab.items():
            if rel_id < model.edge_embedding.num_embeddings:
                embedding_vector = model.edge_embedding.weight[rel_id]
                irf_multiplier = model.irf_weights[rel_id].item()
                effective_weight = torch.norm(embedding_vector * irf_multiplier).item()
                
                if rel_label not in weight_history:
                    weight_history[rel_label] = []
                    
                weight_history[rel_label].append(effective_weight)
                
                if epoch > 0:
                    delta = effective_weight - weight_history[rel_label][-2]
                    sign = "+" if delta >= 0 else ""
                    logging.info(f"{rel_label:16s} | Weight: {effective_weight:.4f} | Delta: {sign}{delta:.4f}")
                else:
                    logging.info(f"{rel_label:16s} | Weight: {effective_weight:.4f} | Delta: Initialised")
    logging.info("------------------------------------\n")

def plot_weight_trajectories(history, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sorted_history = sorted(history.items(), key=lambda item: item[1][-1], reverse=True)
    
    for rel_label, weights in sorted_history:
        final_weight = weights[-1]
        display_label = f"{rel_label} ({final_weight:.4f})"
        
        ax.plot(range(1, len(weights) + 1), weights, marker=None, label=display_label, linewidth=2.5)
        
    ax.set_title("Learned Semantic Prominence per Epoch", fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Effective L2 Norm (Scaled Weight)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., ncol=2, fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_training_metrics(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # --- Plot 1: Loss Progression ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker=None, color='#3498db', linewidth=2.5)
    plt.plot(epochs, history['test_loss'], label='Validation Loss', marker=None, color='#e74c3c', linewidth=2.5)
    
    plt.title('Loss Progression over Epochs (Convergence Curve)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_progression.png'), dpi=300)
    plt.close()
    
    # --- Plot 2: Performance Metrics Progression ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['test_acc'], label='Accuracy', marker=None, color='#2ecc71', linewidth=2.5)
    plt.plot(epochs, history['test_prec'], label='Precision', marker=None, color='#9b59b6', linewidth=2.5)
    plt.plot(epochs, history['test_rec'], label='Recall', marker=None, color='#f1c40f', linewidth=2.5)
    plt.plot(epochs, history['test_f1'], label='F1-Score', marker=None, color='#e67e22', linewidth=2.5)
    
    plt.title('Validation Statistics Progression over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score (0.0 - 1.0)', fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right', ncol=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_progression.png'), dpi=300)
    plt.close()