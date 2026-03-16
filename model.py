import os
import sys
import time
import requests
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn.functional as F
from torch.nn import Linear
# from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GATConv, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import spacy
from transformers import AutoTokenizer, AutoModel
from gradio_client import Client
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from functools import wraps
import threading
import concurrent.futures
from tqdm import tqdm
import re
import logging

class ImmediateFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush() # 1. Flush Python's internal buffer to the OS
        os.fsync(self.stream.fileno()) # 2. Force the OS to write to the physical disk

# Set up logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        ImmediateFileHandler("output.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Initialise SpaCy for tokenisation and part-of-speech tagging
nlp = spacy.load("en_core_web_md")

logging.info("Initialising Hugging Face Space client for fallback...")
try:
    # hf_client = Client("cstr/conceptnet_normalized")
    hf_client = Client("http://127.0.0.1:7860/")
except Exception as e:
    logging.info(f"Warning: Failed to initialise Hugging Face fallback. Error: {e}")
    hf_client = None

# --- Cache Management & Memoisation ---
CACHE_FILE = "conceptnet_cache.json"
query_counter = 0
AUTOSAVE_INTERVAL = 100

# Threading locks for concurrent safety
cache_lock = threading.Lock()
api_lock = threading.Lock()
last_api_time = 0.0

def load_cache():
    """Loads the memoised predictions from a JSON file."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                raw_cache = json.load(f)
                logging.info(f"Successfully loaded {len(raw_cache)} memoised predictions from {CACHE_FILE}.")
                return {tuple(k.split('|')): v for k, v in raw_cache.items()}
        except Exception as e:
            logging.info(f"Warning: Could not load cache file. Error: {e}")
    return {}

def save_cache(cache):
    """Saves the memoised predictions to disk."""
    try:
        serialisable_cache = {"|".join(k): v for k, v in cache.items()}
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(serialisable_cache, f, indent=4)
        logging.info(f"[Cache] Successfully saved {len(cache)} predictions to {CACHE_FILE}.")
    except Exception as e:
        logging.info(f"[Cache] Warning: Failed to save cache. Error: {e}")

# Initialise global cache dictionary
conceptnet_cache = load_cache()
spacing_api = 0.6

def memoize_prediction(func):
    """Decorator to memoise ConceptNet predictions, trigger autosaves, and enforce rate limits."""
    @wraps(func)
    def wrapper(concept_a, concept_b, *args, **kwargs):
        global query_counter, last_api_time
        
        c_a = concept_a.lower().strip().replace(" ", "_")
        c_b = concept_b.lower().strip().replace(" ", "_")
        cache_key = tuple(sorted([c_a, c_b]))
        
        # Thread-safe O(1) Memory Lookup
        with cache_lock:
            if cache_key in conceptnet_cache:
                return conceptnet_cache[cache_key]
                
        # --- Strict Rate Limiting Block ---
        # Ensures that network calls are dispatched at most x per second across all threads
        with api_lock:
            current_time = time.time()
            elapsed = current_time - last_api_time
            if elapsed < spacing_api:
                time.sleep(spacing_api - elapsed)
            last_api_time = time.time()
            
        # Execute the actual network function (runs concurrently outside the api_lock)
        prediction_result = func(c_a, c_b, *args, **kwargs)
        
        # Thread-safe write and autosave
        with cache_lock:
            conceptnet_cache[cache_key] = prediction_result
            query_counter += 1
            if query_counter >= AUTOSAVE_INTERVAL:
                logging.info(f"\n[Autosave] Reached {AUTOSAVE_INTERVAL} network queries. Saving to disk...")
                save_cache(conceptnet_cache)
                query_counter = 0
            
        return prediction_result
    return wrapper

class TextEmbedder:
    """Handles text embedding using BERT, SpaCy GloVe, or random initialisation."""
    def __init__(self, method='bert'):
        self.method = method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.method == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
            self.dim = 768
        elif self.method == 'spacy_glove':
            self.dim = 300
        else:
            self.dim = 128 

    def embed_nodes(self, concepts):
        if not concepts:
            return torch.empty((0, self.dim))

        if self.method == 'bert':
            inputs = self.tokenizer(concepts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu()
            
        elif self.method == 'spacy_glove':
            embeddings = [nlp(c).vector for c in concepts]
            return torch.tensor(embeddings, dtype=torch.float)
            
        else:
            return torch.randn((len(concepts), self.dim), dtype=torch.float)
        
def is_valid_status(text: str) -> bool:
    """
    Returns True if 'results' found, False if 'No results'.
    Raises ValueError if the string format is invalid.
    """
    # Pattern definitions
    no_results_pattern = r"^⚠️ No results \(\d+\.?\d*s\)$"
    results_found_pattern = r"^✅ \d+ results in \d+\.?\d*s$"

    if re.match(results_found_pattern, text):
        return True
    
    if re.match(no_results_pattern, text):
        return False

    # If neither match, the input is malformed
    raise ValueError(f"Invalid status format: '{text}'")

def _fallback_huggingface(c_a, c_b, weight_threshold):
    """Helper function to execute the Hugging Face API fallback."""
    if not hf_client:
        logging.info(f"[HF Fallback] Hugging Face client not available. Cannot query '{c_a}' <-> '{c_b}'.")
        return False
    
    try:
        hf_input_a = c_a.replace("_", " ")
        hf_input_b = c_b.replace("_", " ")
        result = hf_client.predict(
            start_node=hf_input_a,
            start_lang="en",        
            relation="",
            end_node=hf_input_b,
            end_lang="en",          
            limit=1,
            api_name="/run_query"
        )
        logging.info(f"[HF Query Result] '{c_a}' <-> '{c_b}': {result}")
        if is_valid_status(result[1]) and result[0].get("data", [])[0][3] >= weight_threshold:
            logging.info(f"[HF Fallback] Link discovered: '{c_a}' <-> '{c_b}'")
            return True
        else:  
            logging.info(f"[HF Fallback] No link for '{c_a}' <-> '{c_b}'")
            return False
    except Exception as e:
        logging.info(f"[HF Fallback] Error querying '{c_a}' <-> '{c_b}': {e}")
            
    logging.info(f"[Network] No edge found for '{c_a}' <-> '{c_b}' after all attempts.")
    return False

@memoize_prediction
def query_conceptnet(c_a, c_b, weight_threshold=1.0):
    """Queries ConceptNet 5 API, falling back to Hugging Face if needed."""
    url = f"https://api.conceptnet.io/query?node=/c/en/{c_a}&other=/c/en/{c_b}"
    headers = {'User-Agent': 'SarcasmGNN_Research_Bot/1.0'}
    
    try:
        response = requests.get(url, headers=headers, timeout=3)
        if response.status_code == 200:
            data = response.json()
            for edge in data.get('edges', []):
                if edge.get('weight', 0.0) >= weight_threshold:
                    logging.info(f"[ConceptNet API] Link discovered: '{c_a}' <-> '{c_b}' (Weight: {edge.get('weight')})")
                    return True
            logging.info(f"[ConceptNet API] No significant link for '{c_a}' <-> '{c_b}'")
            return False
        elif response.status_code in [429, 502, 503]:
            logging.info(f"[ConceptNet API] Rate limit/Error {response.status_code}. Falling back to Hugging Face...")
            return _fallback_huggingface(c_a, c_b, weight_threshold)
    except requests.exceptions.RequestException as e:
        logging.info(f"[ConceptNet API] Network exception ({e}). Falling back to Hugging Face...")
        return _fallback_huggingface(c_a, c_b, weight_threshold)
        
    return False

def build_graph_from_title(title, label, embedder, return_concepts=False):
    """Builds a PyG Data object from a headline using parallel edge evaluation."""
    doc = nlp(title.lower())
    
    valid_pos = {"NOUN", "VERB", "ADJ", "PROPN"}
    concepts = [token.lemma_ for token in doc if token.pos_ in valid_pos and not token.is_stop]
    concepts = list(dict.fromkeys(concepts)) 
    
    if not concepts:
        concepts = ["placeholder"]
        
    x = embedder.embed_nodes(concepts)
    edge_index = []
    
    # Sequential edges
    for i in range(len(concepts) - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i]) 
        
    # --- Parallel ConceptNet Prior Edges ---
    query_tasks = []
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            query_tasks.append((concepts[i], concepts[j], i, j))

    if query_tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all concept pairs to the thread pool
            future_to_edge = {
                executor.submit(query_conceptnet, q[0], q[1]): (q[2], q[3]) 
                for q in query_tasks
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_edge):
                i, j = future_to_edge[future]
                try:
                    if future.result():
                        edge_index.append([i, j])
                        edge_index.append([j, i])
                except Exception as e:
                    logging.info(f"Error executing parallel query: {e}")
                
    if not edge_index:
        edge_index = [[0, 0]] # Fixed isolated node tensor dimension
        
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    y = torch.tensor([label], dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    if return_concepts:
        return data, concepts
    return data

class SarcasmGNN(torch.nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, input_dim, hidden_dim=64, heads=4):
        super(SarcasmGNN, self).__init__()
        # Multi-head attention allows the model to learn multiple relationship types simultaneously
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.4)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=0.4)
        self.classifier = Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = global_max_pool(x, batch)
        
        out = self.classifier(x)
        return out 

def evaluate(model, loader, criterion):
    """Evaluates the model, calculating Loss, Accuracy, Precision, Recall, and F1."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            probs = torch.sigmoid(out.squeeze(-1))
            loss = criterion(probs, data.y)
            total_loss += loss.item()
            
            predictions = (probs >= 0.5).float()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) if all_labels else 0
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return total_loss / len(loader), accuracy, precision, recall, f1

def save_gnn_graph(data, concepts, headline, true_label, pred_label, save_path):
    """Saves a topology visualisation highlighting the mathematical edges."""
    G = to_networkx(data, to_undirected=True)
    mapping = {i: concept for i, concept in enumerate(concepts)}
    G = nx.relabel_nodes(G, mapping)

    seq_only, semantic_only, both = [], [], []

    for u, v in G.edges():
        if u in concepts and v in concepts:
            idx_u, idx_v = concepts.index(u), concepts.index(v)
            
            c_u = u.lower().strip().replace(" ", "_")
            c_v = v.lower().strip().replace(" ", "_")
            cache_key = tuple(sorted([c_u, c_v]))
            
            # Use the cache to definitively check if it's a ConceptNet edge
            is_conceptnet = conceptnet_cache.get(cache_key, False)
            is_sequential = abs(idx_u - idx_v) == 1
            
            # Prioritise displaying the ConceptNet relationship even if adjacent
            if is_sequential and is_conceptnet:
                both.append((u, v))
            elif is_sequential:
                seq_only.append((u, v))
            elif is_conceptnet:
                semantic_only.append((u, v))
                
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42, k=0.9)
    
    nx.draw_networkx_nodes(G, pos, node_color='#d4e6f1', node_size=2500, alpha=0.9, edgecolors='#2874a6')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_color='#1b4f72')
    
    # Plot mutually exclusive edge sets with distinct visual hierarchies
    nx.draw_networkx_edges(G, pos, edgelist=seq_only, width=1.5, alpha=0.4, style='dashed', edge_color='grey')
    nx.draw_networkx_edges(G, pos, edgelist=semantic_only, width=2.5, alpha=0.8, edge_color='#e67e22')
    nx.draw_networkx_edges(G, pos, edgelist=both, width=4.0, alpha=0.9, edge_color='#8e44ad')
    
    match_colour = "green" if true_label == pred_label else "red"
    plt.title(f"{headline[:60]}...\nTrue: {true_label} | Pred: {pred_label}", fontsize=12, fontweight='bold', color=match_colour)
    plt.axis('off')
    
    l1 = mlines.Line2D([], [], color='grey', linestyle='dashed', linewidth=1.5, label='Sequential Only')
    l2 = mlines.Line2D([], [], color='#e67e22', linewidth=2.5, label='Semantic Only (Prior)')
    l3 = mlines.Line2D([], [], color='#8e44ad', linewidth=4.0, label='Both (Sequential + Semantic)')
    plt.legend(handles=[l1, l2, l3], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    logging.info("Loading dataset...")
    try:
        df = pd.read_json("Sarcasm_Headlines_Dataset_v2.json", lines=True)
    except FileNotFoundError:
        logging.info("Dataset not found. Please ensure 'Sarcasm_Headlines_Dataset_v2.json' is in the directory.")
        exit()
    
    df_subset = df.sample(400, random_state=42) 
    
    train_df, test_df = train_test_split(df_subset, test_size=0.2, random_state=42, stratify=df_subset['is_sarcastic'])
    logging.info(f"Training samples: {len(train_df)} | Testing samples: {len(test_df)}")

    embedder = TextEmbedder(method='bert') 
    
    logging.info("\nConstructing training graphs... (API rate limits and fallbacks apply)")
    train_graphs = [build_graph_from_title(row['headline'], row['is_sarcastic'], embedder) 
                    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Training Graphs")]

    logging.info("\nConstructing testing graphs...")
    test_graphs = [build_graph_from_title(row['headline'], row['is_sarcastic'], embedder) 
                   for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing Graphs")]

    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SarcasmGNN(input_dim=embedder.dim, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    logging.info("\nInitiating training...")
    epochs = 10
    for epoch in range(epochs): 
        model.train()
        train_loss = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            probs = torch.sigmoid(out.squeeze(-1))
            loss = criterion(probs, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion)
        
        logging.info(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Acc: {test_acc*100:.1f}% | Prec: {test_prec:.3f} | Rec: {test_rec:.3f} | F1: {test_f1:.3f}")
        
    # --- Post-Training Visualisation Block ---
    logging.info("\nTraining complete. Generating visualisations for analysis...")
    os.makedirs("visualisations", exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Save 3 Train samples
        for idx, (_, row) in enumerate(train_df.head(3).iterrows()):
            data, concepts = build_graph_from_title(row['headline'], row['is_sarcastic'], embedder, return_concepts=True)
            batch = torch.zeros(data.x.size(0), dtype=torch.long)
            pred_prob = torch.sigmoid(model(data.x, data.edge_index, batch).squeeze(-1)).item()
            pred_label = 1 if pred_prob >= 0.5 else 0
            save_gnn_graph(data, concepts, row['headline'], row['is_sarcastic'], pred_label, os.path.join("visualisations", f"train_sample_{idx}.png"))
            
        # Save 3 Test samples
        for idx, (_, row) in enumerate(test_df.head(3).iterrows()):
            data, concepts = build_graph_from_title(row['headline'], row['is_sarcastic'], embedder, return_concepts=True)
            batch = torch.zeros(data.x.size(0), dtype=torch.long)
            pred_prob = torch.sigmoid(model(data.x, data.edge_index, batch).squeeze(-1)).item()
            pred_label = 1 if pred_prob >= 0.5 else 0
            save_gnn_graph(data, concepts, row['headline'], row['is_sarcastic'], pred_label, os.path.join("visualisations", f"test_sample_{idx}.png"))
            
    logging.info("Visualisations saved successfully in the 'visualisations' directory.")

    # --- Persist the Final Cache ---
    save_cache(conceptnet_cache)