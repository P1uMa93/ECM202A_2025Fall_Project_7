"""
Baseline Evaluation Script - WITHOUT GraphRAG
Direct Ollama LLM with Training Document Context

This script:
1. Uses the SAME 60 queries as GraphRAG evaluation (same random seed)
2. Loads training document as context
3. Sends: [Training Doc] + [Query] to Ollama directly
4. Extracts predicted activity labels
5. Compares with GraphRAG results
"""

import os
import re
import logging
import ollama
import random
from collections import defaultdict

logging.basicConfig(level=logging.WARNING)

# Configuration
MODEL = "gpt-oss:120b"  
QUERY_DIR = "/home/m202/M202/Richard/query"
TRAINING_DOC = "/home/m202/M202/Richard/training/test_4.txt"
GROUND_TRUTH_FILE = os.path.join(QUERY_DIR, "ground_truth.txt")

# Sampling configuration - MUST MATCH GraphRAG evaluation
QUERIES_PER_ACTIVITY = 10
RANDOM_SEED = 42  # Same seed ensures same query selection

# System prompt for activity classification
SYSTEM_PROMPT = """You are an expert in Human Activity Recognition from sensor data.
Given statistical features from accelerometer and gyroscope sensors, classify the activity.

The possible activities are:
1. Walking
2. Walking_Upstairs
3. Walking_Downstairs
4. Sitting
5. Standing
6. Laying

Provide your reasoning and conclude with a clear activity prediction."""


async def query_ollama(training_context, query_text):
    """
    Send training document + query to Ollama
    
    Args:
        training_context: Training document text
        query_text: Query text
    
    Returns:
        str: LLM response
    """
    # Construct prompt: training document + query
    full_prompt = f"""{training_context}

================================================================================
QUERY - Please classify the following sensor data:
================================================================================

{query_text}

Based on the training examples above and the sensor statistics provided, what activity does this sensor data represent? Provide your reasoning and state the activity clearly."""

    ollama_client = ollama.AsyncClient()
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": full_prompt}
    ]
    
    response = await ollama_client.chat(model=MODEL, messages=messages)
    return response["message"]["content"]


def extract_activity_label(response_text):
    """
    Extract activity label from LLM response
    Same logic as GraphRAG evaluation
    """
    activities = {
        'Walking': ['walking', 'walk'],
        'Walking_Upstairs': ['walking upstairs', 'upstairs', 'walking_upstairs', 'walk upstairs'],
        'Walking_Downstairs': ['walking downstairs', 'downstairs', 'walking_downstairs', 'walk downstairs'],
        'Sitting': ['sitting', 'sit', 'seated'],
        'Standing': ['standing', 'stand'],
        'Laying': ['laying', 'lying', 'lie', 'lying down', 'laying down']
    }
    
    response_lower = response_text.lower()
    
    # Count matches for each activity
    activity_scores = {}
    
    for activity, variations in activities.items():
        for variation in variations:
            patterns = [
                rf'\b{re.escape(variation)}\b',
                rf'represents.*?\b{re.escape(variation)}\b',
                rf'activity.*?(?:is|of).*?\b{re.escape(variation)}\b',
                rf'conclusion.*?\b{re.escape(variation)}\b',
                rf'\*\*{re.escape(variation)}\*\*',
            ]
            
            for pattern in patterns:
                if re.search(pattern, response_lower):
                    activity_scores[activity] = activity_scores.get(activity, 0) + 1
    
    if activity_scores:
        return max(activity_scores, key=activity_scores.get)
    
    # Fallback: find any mention
    for activity, variations in activities.items():
        for variation in variations:
            if variation in response_lower:
                return activity
    
    return "Unknown"


def load_ground_truth(filepath):
    """Load ground truth labels"""
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
    """Group queries by activity"""
    grouped = defaultdict(list)
    for filename, label in ground_truth.items():
        grouped[label].append(filename)
    return grouped


def sample_queries(grouped_queries, queries_per_activity, random_seed=None):
    """Sample queries with same logic as GraphRAG evaluation"""
    if random_seed is not None:
        random.seed(random_seed)
    
    selected = []
    for activity, filenames in sorted(grouped_queries.items()):
        if len(filenames) < queries_per_activity:
            print(f"WARNING: Only {len(filenames)} queries available for {activity}")
            sampled = filenames
        else:
            sampled = random.sample(filenames, queries_per_activity)
        selected.extend(sampled)
    
    return selected


async def evaluate_baseline():
    """Main evaluation function"""
    print("="*80)
    print("BASELINE Evaluation - Direct Ollama WITHOUT GraphRAG")
    print(f"Model: {MODEL}")
    print("="*80)
    
    # Load training document
    print(f"\n1. Loading training document: {TRAINING_DOC}...")
    if not os.path.exists(TRAINING_DOC):
        print(f"ERROR: Training document not found at {TRAINING_DOC}")
        return
    
    with open(TRAINING_DOC, 'r') as f:
        training_context = f.read()
    
    print(f"   ✓ Loaded training document ({len(training_context)} characters)")
    
    # Load ground truth
    print(f"\n2. Loading ground truth from {GROUND_TRUTH_FILE}...")
    ground_truth = load_ground_truth(GROUND_TRUTH_FILE)
    print(f"   Loaded {len(ground_truth)} total query-label pairs")
    
    # Group by activity
    print(f"\n3. Grouping queries by activity...")
    grouped_queries = group_queries_by_activity(ground_truth)
    
    # Sample queries - SAME SEED AS GRAPHRAG
    print(f"\n4. Sampling {QUERIES_PER_ACTIVITY} queries per activity (seed={RANDOM_SEED})...")
    selected_queries = sample_queries(grouped_queries, QUERIES_PER_ACTIVITY, RANDOM_SEED)
    print(f"   Selected {len(selected_queries)} queries (SAME as GraphRAG evaluation)")
    
    # Save selected queries
    with open("selected_queries_baseline.txt", 'w') as f:
        f.write("Selected Queries for Baseline Evaluation (NO GraphRAG)\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n")
        f.write(f"Model: {MODEL}\n")
        f.write("="*80 + "\n\n")
        for query_file in sorted(selected_queries):
            f.write(f"{query_file}\t{ground_truth[query_file]}\n")
    
    # Process queries
    print(f"\n5. Processing {len(selected_queries)} queries with direct Ollama...")
    print(f"   Format: [Training Doc] + [Query] → Ollama {MODEL}")
    print()
    
    results = []
    correct = 0
    total = 0
    
    selected_queries_sorted = sorted(selected_queries)
    
    for i, query_file in enumerate(selected_queries_sorted, 1):
        true_label = ground_truth[query_file]
        query_path = os.path.join(QUERY_DIR, query_file)
        
        if not os.path.exists(query_path):
            print(f"   [{i}/{len(selected_queries)}] SKIP: {query_file} (not found)")
            continue
        
        # Load query
        with open(query_path, 'r') as f:
            query_text = f.read().strip()
        
        print(f"   [{i}/{len(selected_queries)}] {query_file:30} True: {true_label:20}", end=" ")
        
        try:
            # Query Ollama with training context
            response = await query_ollama(training_context, query_text)
            
            # Extract prediction
            predicted_label = extract_activity_label(response)
            
            # Check correctness
            is_correct = (predicted_label == true_label)
            if is_correct:
                correct += 1
                status = "✓"
            else:
                status = "✗"
            
            total += 1
            print(f"Pred: {predicted_label:20} {status}")
            
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
    
    # Calculate metrics
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Print summary
    print("\n" + "="*80)
    print("BASELINE RESULTS (NO GRAPHRAG)")
    print("="*80)
    print(f"\nTotal Queries: {total}")
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
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    output_file = "evaluation_results_no_graphrag.txt"
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BASELINE Evaluation Results - Direct Ollama WITHOUT GraphRAG\n")
        f.write(f"Model: {MODEL}\n")
        f.write(f"Training Document: {TRAINING_DOC}\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Queries: {total}\n")
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
    print("BASELINE EVALUATION COMPLETE")
    print("="*80)
    print(f"\nTo compare with GraphRAG, run: python compare_results.py")
    
    return accuracy, results


if __name__ == "__main__":
    import asyncio
    asyncio.run(evaluate_baseline())
