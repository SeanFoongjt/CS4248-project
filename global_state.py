import os
import json
import threading
import logging

CACHE_FILE = "conceptnet_node_cache.json" # Updated filename
AUTOSAVE_INTERVAL = 20 # Lowered because node queries are larger/denser
spacing_api = 0.001

# Threading Locks
cache_lock = threading.Lock()
api_lock = threading.Lock()
vocab_lock = threading.Lock()
active_queries_lock = threading.Lock() # NEW: Prevents Cache Stampedes

# State Tracking
query_counter = 0
last_api_time = 0.0
RELATION_VOCAB = {"sequential": 0}
relation_counter = 1

# Tracks words currently being fetched by a thread: { "word": threading.Event() }
active_queries = {} 

def get_relation_id(rel_label):
    global relation_counter
    with vocab_lock:
        if rel_label not in RELATION_VOCAB:
            RELATION_VOCAB[rel_label] = relation_counter
            relation_counter += 1
        return RELATION_VOCAB[rel_label]

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            logging.info(f"Successfully loaded {len(cache)} nodes from local memory.")
            return cache
        except Exception as e:
            logging.info(f"Warning: Could not load cache file. Error: {e}")
    return {}

def save_cache(cache):
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=4)
    except Exception as e:
        logging.info(f"[Cache] Warning: Failed to save cache. Error: {e}")

conceptnet_cache = load_cache()