"""
Evaluation Script for GraphRAG-based Human Activity Recognition
Updated for 180 queries (30 subjects × 6 activities)

This script:
1. Loads all 180 query files and ground truth labels
2. Randomly selects 60 queries (10 per activity) for balanced testing
3. Uses existing GraphRAG cache (no retraining needed)
4. Runs selected queries through GraphRAG
5. Extracts predicted activity labels from LLM responses
6. Compares with ground truth and calculates accuracy
7. Shows detailed results and confusion matrix
"""

import os
import re
import logging
import ollama
import numpy as np
import random
from collections import defaultdict
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# Configuration - same as no_openai_key_at_all.py
WORKING_DIR = "./nano_graphrag_cache_ollama_TEST"
MODEL = "gpt-oss:120b"
QUERY_DIR = "/home/m202/M202/Richard/query"
GROUND_TRUTH_FILE = os.path.join(QUERY_DIR, "ground_truth.txt")

# Sampling configuration
QUERIES_PER_ACTIVITY = 10  # Sample 10 queries per activity
RANDOM_SEED = 42  # Set seed for reproducibility (change to get different samples)

# Initialize embedding model
EMBED_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", cache_folder=WORKING_DIR, device="cuda"
)


@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)


async def ollama_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # remove kwargs that are not supported by ollama
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)

    ollama_client = ollama.AsyncClient()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    
    response = await ollama_client.chat(model=MODEL, messages=messages, **kwargs)
    result = response["message"]["content"]
    
    # Cache the response if having
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": MODEL}})
    
    return result


def extract_activity_label(response_text):
    """
    Extract activity label from LLM response
    
    Looks for patterns like:
    - "represents the activity of Walking"
    - "the sensor data most likely represents Walking"
    - "Conclusion: Walking"
    - "Activity: Walking"
    
    Returns:
        str: One of the 6 activity labels, or "Unknown"
    """
    # Define valid activities (with variations)
    activities = {
        'Walking': ['walking', 'walk'],
        'Walking_Upstairs': ['walking upstairs', 'upstairs', 'walking_upstairs', 'walk upstairs'],
        'Walking_Downstairs': ['walking downstairs', 'downstairs', 'walking_downstairs', 'walk downstairs'],
        'Sitting': ['sitting', 'sit', 'seated'],
        'Standing': ['standing', 'stand'],
        'Laying': ['laying', 'lying', 'lie', 'lying down', 'laying down']
    }
    
    # Convert response to lowercase for matching
    response_lower = response_text.lower()
    
    # Try to find activity mentions with confidence scores
    activity_scores = {}
    
    for activity, variations in activities.items():
        for variation in variations:
            # Look for patterns indicating this is the answer
            patterns = [
                rf'\b{re.escape(variation)}\b',  # Exact word match
                rf'represents.*?\b{re.escape(variation)}\b',  # "represents ... walking"
                rf'activity.*?(?:is|of).*?\b{re.escape(variation)}\b',  # "activity is/of walking"
                rf'conclusion.*?\b{re.escape(variation)}\b',  # "conclusion: walking"
                rf'\*\*{re.escape(variation)}\*\*',  # **Walking** (bold)
            ]
            
            for pattern in patterns:
                if re.search(pattern, response_lower):
                    activity_scores[activity] = activity_scores.get(activity, 0) + 1
    
    # Return the activity with highest score
    if activity_scores:
        return max(activity_scores, key=activity_scores.get)
    
    # If no clear match, try to find the last mentioned activity
    for activity, variations in activities.items():
        for variation in variations:
            if variation in response_lower:
                return activity
    
    return "Unknown"


def load_ground_truth(filepath):
    """
    Load ground truth from file
    
    Format: query_6_Sitting.txt\tSitting
    
    Returns:
        dict: {filename: true_label}
    """
    ground_truth = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) == 2:
                filename, label = parts
                ground_truth[filename] = label
    
    return ground_truth


def group_queries_by_activity(ground_truth):
    """
    Group query files by activity label
    
    Args:
        ground_truth: dict of {filename: label}
    
    Returns:
        dict: {activity_label: [list of query filenames]}
    """
    grouped = defaultdict(list)
    
    for filename, label in ground_truth.items():
        grouped[label].append(filename)
    
    return grouped


def sample_queries(grouped_queries, queries_per_activity, random_seed=None):
    """
    Randomly sample queries from each activity
    
    Args:
        grouped_queries: dict of {activity: [filenames]}
        queries_per_activity: number of queries to sample per activity
        random_seed: random seed for reproducibility
    
    Returns:
        list: selected query filenames
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    selected = []
    
    for activity, filenames in sorted(grouped_queries.items()):
        # Sample queries for this activity
        if len(filenames) < queries_per_activity:
            print(f"WARNING: Only {len(filenames)} queries available for {activity}, requested {queries_per_activity}")
            sampled = filenames  # Use all available
        else:
            sampled = random.sample(filenames, queries_per_activity)
        
        selected.extend(sampled)
    
    return selected


def evaluate_graphrag():
    """
    Main evaluation function
    """
    print("="*80)
    print("GraphRAG Human Activity Recognition Evaluation")
    print("Updated for 180 Queries (30 subjects × 6 activities)")
    print("="*80)
    
    # Load ground truth
    print(f"\n1. Loading ground truth from {GROUND_TRUTH_FILE}...")
    ground_truth = load_ground_truth(GROUND_TRUTH_FILE)
    print(f"   Loaded {len(ground_truth)} total query-label pairs")
    
    # Group by activity
    print(f"\n2. Grouping queries by activity...")
    grouped_queries = group_queries_by_activity(ground_truth)
    
    print(f"   Query distribution:")
    for activity in sorted(grouped_queries.keys()):
        print(f"   - {activity}: {len(grouped_queries[activity])} queries")
    
    # Sample queries
    print(f"\n3. Randomly sampling {QUERIES_PER_ACTIVITY} queries per activity (seed={RANDOM_SEED})...")
    selected_queries = sample_queries(grouped_queries, QUERIES_PER_ACTIVITY, RANDOM_SEED)
    print(f"   Selected {len(selected_queries)} queries for testing")
    
    # Show sample distribution
    selected_by_activity = defaultdict(int)
    for query_file in selected_queries:
        label = ground_truth[query_file]
        selected_by_activity[label] += 1
    
    print(f"\n   Selected query distribution:")
    for activity in sorted(selected_by_activity.keys()):
        print(f"   - {activity}: {selected_by_activity[activity]} queries")
    
    # Save selected queries list
    with open("selected_queries.txt", 'w') as f:
        f.write("Selected Queries for Evaluation\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n")
        f.write(f"Queries per Activity: {QUERIES_PER_ACTIVITY}\n")
        f.write("="*80 + "\n\n")
        for query_file in sorted(selected_queries):
            f.write(f"{query_file}\t{ground_truth[query_file]}\n")
    
    print(f"   ✓ Saved selected queries to: selected_queries.txt")
    
    # Initialize GraphRAG with existing cache
    print(f"\n4. Initializing GraphRAG with existing cache at {WORKING_DIR}...")
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=ollama_model_if_cache,
        cheap_model_func=ollama_model_if_cache,
        embedding_func=local_embedding,
    )
    print("   GraphRAG initialized successfully")
    
    # Process selected queries
    print(f"\n5. Processing {len(selected_queries)} selected queries from {QUERY_DIR}...")
    results = []
    correct = 0
    total = 0
    
    # Sort selected queries for consistent ordering
    selected_queries_sorted = sorted(selected_queries)
    
    for i, query_file in enumerate(selected_queries_sorted, 1):
        true_label = ground_truth[query_file]
        query_path = os.path.join(QUERY_DIR, query_file)
        
        if not os.path.exists(query_path):
            print(f"   [{i}/{len(selected_queries)}] SKIP: {query_file} (file not found)")
            continue
        
        # Load query text
        with open(query_path, 'r') as f:
            query_text = f.read().strip()
        
        # Run query through GraphRAG
        print(f"   [{i}/{len(selected_queries)}] {query_file:30} True: {true_label:20}", end=" ")
        
        try:
            response = rag.query(query_text, param=QueryParam(mode="local"))
            
            # Extract predicted label
            predicted_label = extract_activity_label(response)
            
            # Check if correct
            is_correct = (predicted_label == true_label)
            if is_correct:
                correct += 1
                status = "✓"
            else:
                status = "✗"
            
            total += 1
            
            print(f"Pred: {predicted_label:20} {status}")
            
            # Store result
            results.append({
                'query_file': query_file,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'correct': is_correct,
                'response': response
            })
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            results.append({
                'query_file': query_file,
                'true_label': true_label,
                'predicted_label': 'Error',
                'correct': False,
                'response': str(e)
            })
    
    # Calculate accuracy
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nTotal Queries Tested: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Confusion matrix
    print("\n" + "="*80)
    print("CONFUSION MATRIX")
    print("="*80)
    
    activities = ['Walking', 'Walking_Upstairs', 'Walking_Downstairs', 'Sitting', 'Standing', 'Laying']
    confusion = defaultdict(lambda: defaultdict(int))
    
    for result in results:
        if result['predicted_label'] != 'Error':
            confusion[result['true_label']][result['predicted_label']] += 1
    
    # Print confusion matrix
    print("\n{:<20}".format("True \\ Predicted"), end="")
    for activity in activities:
        print(f"{activity[:10]:<12}", end="")
    print()
    print("-" * (20 + 12 * len(activities)))
    
    for true_activity in activities:
        print(f"{true_activity:<20}", end="")
        for pred_activity in activities:
            count = confusion[true_activity][pred_activity]
            print(f"{count:<12}", end="")
        print()
    
    # Show errors
    print("\n" + "="*80)
    print("INCORRECT PREDICTIONS")
    print("="*80)
    
    errors = [r for r in results if not r['correct']]
    if errors:
        print(f"\nFound {len(errors)} errors:\n")
        for error in errors:
            print(f"File: {error['query_file']}")
            print(f"  True: {error['true_label']}")
            print(f"  Predicted: {error['predicted_label']}")
            print(f"  Response excerpt: {error['response'][:150]}...")
            print()
    else:
        print("\nNo errors! Perfect accuracy! 🎉")
    
    # Per-activity accuracy
    print("\n" + "="*80)
    print("PER-ACTIVITY ACCURACY")
    print("="*80)
    
    for activity in activities:
        activity_results = [r for r in results if r['true_label'] == activity]
        if activity_results:
            activity_correct = sum(1 for r in activity_results if r['correct'])
            activity_total = len(activity_results)
            activity_accuracy = (activity_correct / activity_total * 100)
            print(f"{activity:<20}: {activity_correct}/{activity_total} ({activity_accuracy:.1f}%)")
    
    # Save detailed results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    output_file = "evaluation_results.txt"
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GraphRAG Human Activity Recognition Evaluation Results\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n")
        f.write(f"Queries per Activity: {QUERIES_PER_ACTIVITY}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Queries Tested: {total}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Incorrect: {total - correct}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for result in results:
            f.write(f"Query File: {result['query_file']}\n")
            f.write(f"True Label: {result['true_label']}\n")
            f.write(f"Predicted Label: {result['predicted_label']}\n")
            f.write(f"Correct: {'Yes' if result['correct'] else 'No'}\n")
            f.write(f"\nFull Response:\n{result['response']}\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    print(f"✓ Detailed results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nTo run with different random sample, change RANDOM_SEED in the script.")
    
    return accuracy, results


if __name__ == "__main__":
    accuracy, results = evaluate_graphrag()