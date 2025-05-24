from typing import List, Dict, Any, Tuple, Optional, Set
import networkx as nx
import numpy as np
from collections import defaultdict
import yaml
from sandbox.utils import get_embedding, cosine_similarity, normalize_scores
import openai


def normalize_scores_new(scores: Dict[Any, float]) -> Dict[Any, float]:

    if not scores:
        return {}
    try:
        max_score = max(scores.values())
        min_score = min(scores.values())
        score_range = max_score - min_score
        if score_range == 0:
            norm_val = 1.0 if max_score > 0 else 0.0
            return {k: norm_val for k in scores}
        else:
            return {k: (v - min_score) / score_range for k, v in scores.items()}
    except Exception as e:
        print(f"Warning: Score normalization failed: {e}. Returning original scores.")
        return scores


def find_init_topk_edges(
    agent_list: List[Dict[str, Any]],
    top_k: int = 5,
) -> Tuple[Dict[int, List[int]], Dict[Tuple[int, int], float]]:

    if top_k <= 0:
        print("Warning: top_k must be positive. Returning empty results.")
        return {}, {}

    G = nx.DiGraph()
    source_nodes_with_edges = set()
    nodes_in_system = set()

    if not agent_list:
        return {}, {}

    for agent_info in agent_list:
        source_id = agent_info.get("id")
        if source_id is None:
            continue

        nodes_in_system.add(source_id)
        G.add_node(source_id) 

        neighbors = agent_info.get("neighborhood", [])
        if neighbors:
            source_nodes_with_edges.add(source_id)

        for target_id in neighbors:
            nodes_in_system.add(target_id)
            if target_id not in G:
                 G.add_node(target_id)
            G.add_edge(source_id, target_id)

    if not G.edges:
        print("Warning: The constructed graph has no edges. Returning empty results.")
        return {}, {}

    try:
        edge_centrality = nx.edge_betweenness_centrality(G, normalized=True)
    except Exception as e:
        print(f"Error calculating edge betweenness centrality: {e}. Returning empty results.")
        return {}, {}
    edge_scores = edge_centrality 

    if not edge_scores:
        print("Warning: No edge scores could be calculated (edge_centrality was empty). Returning empty results.")
        return {}, {}

    best_outgoing_edge_for_source: Dict[int, Tuple[Tuple[int, int], float]] = {}
    for (u, v), score in edge_scores.items():
        if u in source_nodes_with_edges:
            if u not in best_outgoing_edge_for_source or score > best_outgoing_edge_for_source[u][1]:
                best_outgoing_edge_for_source[u] = ((u, v), score)

    final_top_edges: Set[Tuple[int, int]] = set()

    guaranteed_candidates = sorted(best_outgoing_edge_for_source.values(), key=lambda item: item[1], reverse=True)

    for edge_tuple, score in guaranteed_candidates:
        if len(final_top_edges) < top_k:
            final_top_edges.add(edge_tuple)
        else:
            break 

    if len(final_top_edges) < top_k:
        all_edges_sorted_by_score = sorted(edge_scores.items(), key=lambda item: item[1], reverse=True)

        for edge_tuple, score in all_edges_sorted_by_score:
            if len(final_top_edges) >= top_k:
                break

            final_top_edges.add(edge_tuple)


    output_dict: Dict[int, List[int]] = defaultdict(list)
    for u, v in final_top_edges:
        output_dict[u].append(v)

    return dict(output_dict), edge_scores



def find_init_topk_edges_vanilla(
    agent_list,
    top_k=3,
):

    if top_k <= 0:
        print("Warning: top_k must be positive.")
        return [], {}

    G = nx.DiGraph()
    if not agent_list:
        print("Warning: agent_list is empty.")
        return [], {}

    edges = []
    for agent_info in agent_list:
        source_id = agent_info.get("id")
        if source_id is None:
            print(f"Warning: Agent info missing 'id': {agent_info}")
            continue
        G.add_node(source_id)

        neighbors = agent_info.get("neighborhood", [])
        for target_id in neighbors:
            G.add_edge(source_id, target_id)
            edges.append((source_id, target_id))

    if not G.edges:
        print("Graph has no edges.")
        return [], {}


    try:
        edge_centrality = nx.edge_betweenness_centrality(G, normalized=True)
    except Exception as e:
        print(f"Error calculating edge betweenness centrality: {e}")
        return [], {}

    edge_scores = normalize_scores(edge_centrality)

    if not edge_scores:
        print("No edge scores could be calculated.")
        return [], {}

    sorted_edges = sorted(edge_scores.items(), key=lambda item: item[1], reverse=True)
    top_edges = [edge for edge, score in sorted_edges[:top_k]]

    output_dict: Dict[int, List[int]] = defaultdict(list)
    for u, v in top_edges:
        output_dict[u].append(v)

    return dict(output_dict), edge_scores




def update_topk_edges(
    goal_embedding: List[str],
    # data,
    agent_list,
    log,
    initial_static_scores: Dict[Tuple[int, int], float],
    top_k: int,
) -> Dict[int, List[int]]:

    with open("config/defense.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    weights = config["weights"]
    similarity_threshold = config["similarity_threshold"]
    window_size = None

    if top_k <= 0:
        print("Warning: top_k must be positive.")
        return {}
    if not all(k in weights for k in ['static', 'frequency', 'relevance']):
         raise ValueError("Weights dictionary must contain keys 'static', 'frequency', 'relevance'.")
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Weights must sum to 1.0.")

    G = nx.DiGraph()
    all_edges = set()
    if agent_list:
        for agent_info in agent_list:
            source_id = agent_info["id"]
            G.add_node(source_id)
            for target_id in agent_info.get("neighborhood", []):
                 G.add_edge(source_id, target_id)
                 all_edges.add((source_id, target_id))
    else:
         return {}

    if not all_edges:
         print("Graph has no edges based on agent_list.")
         return {}

    if goal_embedding is None:
        print("Error: Could not compute embedding for misinfo_goal. Aborting relevance calculation.")
        weights['relevance'] = 0.0

    edge_frequency: Dict[Tuple[int, int], int] = defaultdict(int)
    edge_max_relevance: Dict[Tuple[int, int], float] = defaultdict(float)

    log_to_process = log
    if window_size is not None and window_size > 0:
        log_to_process = log[-window_size:]
        print(f"Processing last {len(log_to_process)} log entries (window size: {window_size})")

    num_embeddings_calculated = 0

    with open("config/api_keys.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    api_key = config["openai_api_key"]
    base_url = config["openai_base_url"]
    openai_client = openai.OpenAI(api_key=api_key, base_url=base_url)

    for entry in log_to_process:
        if entry.get("action_type") == "send_message":
            source = entry.get("subjective")
            targets = entry.get("objective", [])
            message_content = entry.get("context", "")

            if source is None or not targets or not message_content:
                continue

            max_sim_message = 0.0
            if goal_embedding is not None and weights['relevance'] > 0:
                try:
                    import nltk
                    from nltk.tokenize import sent_tokenize
                except:
                    print("Warning: nltk library not found. Sentence tokenization will fail.")
                    print("Install using: pip install nltk")
                    import re
                    def simple_sent_tokenize(text):
                        sentences = re.split(r'(?<=[.?!])\s+', text)
                        return [s for s in sentences if s] # Remove empty strings
                    sent_tokenize = simple_sent_tokenize
                    print("Using simple regex-based sentence tokenizer as fallback.")
                sentences = sent_tokenize(message_content)

                for sentence in sentences:
                    sentence_embedding = get_embedding(sentence, openai_client)
                    if sentence_embedding is not None and len(goal_embedding):
                        sim = 0.0
                        for goal in goal_embedding:
                            num_embeddings_calculated += 1
                            sim += cosine_similarity(sentence_embedding, goal)
                        sim /= len(goal_embedding) 
                        if sim > similarity_threshold: 
                            max_sim_message = max(max_sim_message, sim)

            for target in targets:
                edge = (source, target)
                if G.has_edge(source, target):
                    edge_frequency[edge] += 1
                    edge_max_relevance[edge] = max(edge_max_relevance.get(edge, 0.0), max_sim_message)

    print(f"Calculated {num_embeddings_calculated} sentence embeddings for relevance scoring.")

    norm_frequency_scores = normalize_scores({e: float(f) for e, f in edge_frequency.items()})
    norm_relevance_scores = edge_max_relevance 

    edge_dynamic_scores: Dict[Tuple[int, int], float] = {}
    for u, v in G.edges():
        edge = (u, v)
        s_static = initial_static_scores.get(edge, 0.0) 
        s_freq = norm_frequency_scores.get(edge, 0.0)
        s_rel = norm_relevance_scores.get(edge, 0.0)

        score = (weights['static'] * s_static +
                 weights['frequency'] * s_freq +
                 weights['relevance'] * s_rel)
        edge_dynamic_scores[edge] = score

    sorted_edges = sorted(edge_dynamic_scores.items(), key=lambda item: item[1], reverse=True)
    top_edges_list = [edge for edge, score in sorted_edges[:top_k]]

    output_dict: Dict[int, List[int]] = defaultdict(list)
    for u, v in top_edges_list:
        output_dict[u].append(v)

    return dict(output_dict)