import time
import requests
import logging
from bs4 import BeautifulSoup
import threading
import concurrent.futures
import utils.global_state as global_state

logging.info("[Initialisation] Node-Centric API loaded. HF Fallback disabled for bulk extraction.")

def get_node_data(concept, weight_threshold=1.0, verbose=False):
    """
    Thread-safe orchestrator for node-centric queries. 
    Implements a strict barrier to prevent Cache Stampedes with verbose logging.
    """
    c_safe = concept.lower().strip().replace(" ", "_")
    
    with global_state.cache_lock:
        if c_safe in global_state.conceptnet_cache:
            if verbose:
                logging.info(f"[Cache Hit] Node '{c_safe}' loaded instantly from local memory.")
            return global_state.conceptnet_cache[c_safe]
            
    if verbose:
        logging.info(f"[Cache Miss] Node '{c_safe}' not found. Entering concurrency barrier...")
    with global_state.active_queries_lock:
        if c_safe in global_state.active_queries:
            event = global_state.active_queries[c_safe]
            is_owner = False
            if verbose:
                logging.info(f"[Concurrency] Conflict detected. Thread yielding ownership of '{c_safe}'.")
        else:
            event = threading.Event()
            global_state.active_queries[c_safe] = event
            is_owner = True
            if verbose:
                logging.info(f"[Concurrency] Thread claimed ownership of '{c_safe}'.")
            
    if not is_owner:
        if verbose:
            logging.info(f"[Concurrency] Thread suspended: Waiting for owner to fetch '{c_safe}'.")
        event.wait() 
        if verbose:
            logging.info(f"[Concurrency] Thread awakened: Fetching '{c_safe}' from updated cache.")
        with global_state.cache_lock:
            return global_state.conceptnet_cache.get(c_safe, {})

    if verbose:        
        logging.info(f"[Network] Owner thread initiating HTTP scrape for node '{c_safe}'...")
    scraped_data = _scrape_concept_node(c_safe, weight_threshold)
    
    with global_state.cache_lock:
        global_state.conceptnet_cache[c_safe] = scraped_data
        global_state.query_counter += 1
        if verbose:
            logging.info(f"[Cache Write] Committed {len(scraped_data)} target edges for '{c_safe}' to memory buffer.")
        
        if global_state.query_counter >= global_state.AUTOSAVE_INTERVAL:
            if verbose:
                logging.info(f"[Cache Persist] Autosave threshold ({global_state.AUTOSAVE_INTERVAL}) reached. Flushing to disk...")
            global_state.save_cache(global_state.conceptnet_cache)
            global_state.query_counter = 0
            
    with global_state.active_queries_lock:
        del global_state.active_queries[c_safe]
        event.set() 
        if verbose:
            logging.info(f"[Concurrency] Lock released. Wake-up signal broadcasted for '{c_safe}'.")
        
    return scraped_data

def _process_partition(rel_url, concept, weight_threshold, headers, shared_node_edges, local_lock, verbose=False):
    """Worker function to process a single feature box partition in parallel."""
    # Synchronised rate limiting
    with global_state.api_lock:
        current_time = time.time()
        elapsed = current_time - global_state.last_api_time
        if elapsed < global_state.spacing_api: 
            delay = global_state.spacing_api - elapsed
            if verbose:
                logging.info(f"[Rate Limit] Throttling partition thread for {delay:.2f}s...")
            time.sleep(delay)
        global_state.last_api_time = time.time()
        
    partition_url = f"https://conceptnet.io{rel_url}" if not rel_url.startswith('http') else rel_url
    if verbose:
        logging.info(f"[Scraper-Worker] Fetching partition data from {partition_url}...")
    
    try:
        part_resp = requests.get(partition_url, headers=headers, timeout=10) 
        if part_resp.status_code == 200:
            part_soup = BeautifulSoup(part_resp.text, 'html.parser')
            edge_rows = part_soup.find_all('tr', class_='edge-main')
            
            # Step 1: Calculate the local supremum to minimise lock contention
            local_updates = {} 
            
            for row in edge_rows:
                start_a = row.find('td', class_='edge-start')
                end_a = row.find('td', class_='edge-end')
                rel_td = row.find('td', class_='edge-rel')
                
                if not start_a or not end_a or not rel_td: continue
                
                start_a = start_a.find('a')
                end_a = end_a.find('a')
                if not start_a or not end_a: continue
                    
                rel_label_span = rel_td.find('span', class_='rel-label')
                rel_label = rel_label_span.text.strip() if rel_label_span else "RelatedTo"
                
                start_href = start_a['href'].rstrip('/')
                end_href = end_a['href'].rstrip('/')
                
                target_href = end_href if start_href.endswith(f"/c/en/{concept}") else start_href
                
                if target_href.startswith('/c/en/'):
                    target_word = target_href.replace('/c/en/', '').split('/')[0]
                    weight_div = rel_td.find('div', class_='weight')
                    
                    if weight_div:
                        try:
                            weight = float(weight_div.text.replace('Weight:', '').strip())
                            if weight >= weight_threshold:
                                if target_word not in local_updates:
                                    local_updates[target_word] = {}
                                if rel_label not in local_updates[target_word] or weight > local_updates[target_word][rel_label]:
                                    local_updates[target_word][rel_label] = weight
                        except ValueError:
                            pass
                            
            # Step 2: Apply local updates to the global supremum dictionary securely
            with local_lock:
                for target_word, rels in local_updates.items():
                    if target_word not in shared_node_edges:
                        shared_node_edges[target_word] = {}
                    for rel_label, weight in rels.items():
                        if rel_label not in shared_node_edges[target_word] or weight > shared_node_edges[target_word][rel_label]:
                            shared_node_edges[target_word][rel_label] = weight
                            
            if verbose:
                logging.info(f"[Scraper-Worker] Partition processed. {len(local_updates)} nodes aggregated.")
        else:
            if verbose:
                logging.info(f"[Scraper-Worker] Error: Partition URL returned status {part_resp.status_code}.")
    except requests.exceptions.RequestException as e:
        if verbose:
            logging.info(f"[Scraper-Worker] Partition Network Error: {e}")

def _scrape_concept_node(concept, weight_threshold, verbose=False):
    """Hits the main page, discovers limit=1000 partition URLs, and dispatches worker threads."""
    base_url = f"https://conceptnet.io/c/en/{concept}"
    headers = {'User-Agent': 'SarcasmGNN_Research_Bot/2.0'}
    
    shared_node_edges = {} 
    local_lock = threading.Lock() # Protects shared_node_edges during multi-threaded reduction
    
    try:
        if verbose:
            logging.info(f"[Scraper-Master] Requesting base topology for '{concept}' -> {base_url}")
        response = requests.get(base_url, headers=headers, timeout=5)
        
        if response.status_code != 200:
            if verbose:
                logging.info(f"[Scraper-Master] Warning: ConceptNet returned status {response.status_code} for '{concept}'.")
            return {}
            
        soup = BeautifulSoup(response.text, 'html.parser')
        feature_boxes = soup.find_all('div', class_='feature-box')
        
        partition_urls = []
        for box in feature_boxes:
            h2_a = box.find('h2').find('a')
            if not h2_a or not h2_a.has_attr('href'): continue
            rel_url = h2_a['href']
            if "limit=" in rel_url:
                partition_urls.append(rel_url)
                
        if verbose:
            logging.info(f"[Scraper-Master] Discovered {len(partition_urls)} valid partitions. Dispatching worker threads...")
        
        # Dispatch partition scraping concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(_process_partition, rel_url, concept, weight_threshold, headers, shared_node_edges, local_lock, verbose)
                for rel_url in partition_urls
            ]
            concurrent.futures.wait(futures)
                                
        # Flatten the reduced dictionaries back into the expected list of tuples
        final_node_edges = {
            target: [(rel, w) for rel, w in rels.items()]
            for target, rels in shared_node_edges.items()
        }
        
        total_unique_targets = len(final_node_edges)
        if verbose:
            logging.info(f"[Scraper-Master] Success. Flattened {total_unique_targets} strictly unique semantic targets for '{concept}'.")
        return final_node_edges
        
    except requests.exceptions.RequestException as e:
        if verbose:
            logging.info(f"[Network Error] Fatal exception during master scrape of '{concept}': {e}")
        return {}