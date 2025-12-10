"""
Analyze GraphRAG Evaluation Results

This script:
1. Parses evaluation_results.txt
2. Identifies "Unknown" predictions (where LLM couldn't answer)
3. Calculates Unknown rate
4. Calculates accuracy excluding Unknown responses
5. Outputs detailed analysis to a new file
"""

import re
from collections import defaultdict


def parse_evaluation_results(filepath):
    """
    Parse evaluation results file
    
    Returns:
        list of dict: Each dict contains query info and results
    """
    results = []
    current_result = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for query file marker
        if line.startswith("Query File:"):
            # If we have a previous result, save it
            if current_result:
                results.append(current_result)
            
            # Start new result
            current_result = {}
            current_result['query_file'] = line.replace("Query File:", "").strip()
            i += 1
            
            # Get true label
            if i < len(lines) and lines[i].strip().startswith("True Label:"):
                current_result['true_label'] = lines[i].strip().replace("True Label:", "").strip()
                i += 1
            
            # Get predicted label
            if i < len(lines) and lines[i].strip().startswith("Predicted Label:"):
                current_result['predicted_label'] = lines[i].strip().replace("Predicted Label:", "").strip()
                i += 1
            
            # Get correct status
            if i < len(lines) and lines[i].strip().startswith("Correct:"):
                correct_str = lines[i].strip().replace("Correct:", "").strip()
                current_result['correct'] = (correct_str.lower() == 'yes')
                i += 1
            
            # Skip empty line
            if i < len(lines) and not lines[i].strip():
                i += 1
            
            # Get full response
            if i < len(lines) and lines[i].strip().startswith("Full Response:"):
                i += 1
                response_lines = []
                
                # Read until separator or end
                while i < len(lines) and not lines[i].strip().startswith("---"):
                    response_lines.append(lines[i])
                    i += 1
                
                current_result['response'] = ''.join(response_lines).strip()
            
            continue
        
        i += 1
    
    # Don't forget the last result
    if current_result:
        results.append(current_result)
    
    return results


def analyze_results(results):
    """
    Analyze results and calculate metrics
    
    Returns:
        dict: Analysis metrics
    """
    total_queries = len(results)
    
    # Count categories
    unknown_count = 0
    answered_count = 0
    correct_count = 0
    correct_excluding_unknown = 0
    
    # Track per-activity stats
    activity_stats = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'unknown': 0,
        'answered': 0,
        'correct_answered': 0
    })
    
    for result in results:
        true_label = result['true_label']
        predicted_label = result['predicted_label']
        is_correct = result['correct']
        
        # Update activity stats
        activity_stats[true_label]['total'] += 1
        
        # Check if Unknown
        if predicted_label == 'Unknown':
            unknown_count += 1
            activity_stats[true_label]['unknown'] += 1
        else:
            answered_count += 1
            activity_stats[true_label]['answered'] += 1
            
            if is_correct:
                correct_excluding_unknown += 1
                activity_stats[true_label]['correct_answered'] += 1
        
        # Overall correct (including unknown as incorrect)
        if is_correct:
            correct_count += 1
            activity_stats[true_label]['correct'] += 1
    
    # Calculate rates
    unknown_rate = (unknown_count / total_queries * 100) if total_queries > 0 else 0
    answered_rate = (answered_count / total_queries * 100) if total_queries > 0 else 0
    
    # Accuracy including Unknown (Unknown counts as incorrect)
    accuracy_with_unknown = (correct_count / total_queries * 100) if total_queries > 0 else 0
    
    # Accuracy excluding Unknown (only consider answered queries)
    accuracy_excluding_unknown = (correct_excluding_unknown / answered_count * 100) if answered_count > 0 else 0
    
    return {
        'total_queries': total_queries,
        'unknown_count': unknown_count,
        'answered_count': answered_count,
        'correct_total': correct_count,
        'correct_excluding_unknown': correct_excluding_unknown,
        'unknown_rate': unknown_rate,
        'answered_rate': answered_rate,
        'accuracy_with_unknown': accuracy_with_unknown,
        'accuracy_excluding_unknown': accuracy_excluding_unknown,
        'activity_stats': dict(activity_stats)
    }


def generate_analysis_report(results, analysis, output_file):
    """
    Generate detailed analysis report
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("GraphRAG Evaluation Results - Detailed Analysis\n")
        f.write("Focus: Unknown Rate and Accuracy Excluding Unknown\n")
        f.write("="*80 + "\n\n")
        
        # Overall Statistics
        f.write("OVERALL STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Queries: {analysis['total_queries']}\n")
        f.write(f"Answered (LLM provided prediction): {analysis['answered_count']}\n")
        f.write(f"Unknown (LLM could not answer): {analysis['unknown_count']}\n\n")
        
        f.write(f"Unknown Rate: {analysis['unknown_rate']:.2f}%\n")
        f.write(f"Answer Rate: {analysis['answered_rate']:.2f}%\n\n")
        
        f.write("-"*80 + "\n\n")
        
        # Accuracy Metrics
        f.write("ACCURACY METRICS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"1. Accuracy INCLUDING Unknown (Unknown = Incorrect):\n")
        f.write(f"   Correct: {analysis['correct_total']}/{analysis['total_queries']}\n")
        f.write(f"   Accuracy: {analysis['accuracy_with_unknown']:.2f}%\n\n")
        
        f.write(f"2. Accuracy EXCLUDING Unknown (Only Answered Queries):\n")
        f.write(f"   Correct: {analysis['correct_excluding_unknown']}/{analysis['answered_count']}\n")
        f.write(f"   Accuracy: {analysis['accuracy_excluding_unknown']:.2f}%\n\n")
        
        f.write(f"Improvement when excluding Unknown: {analysis['accuracy_excluding_unknown'] - analysis['accuracy_with_unknown']:.2f}%\n\n")
        
        f.write("-"*80 + "\n\n")
        
        # Per-Activity Analysis
        f.write("PER-ACTIVITY ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        activities = sorted(analysis['activity_stats'].keys())
        
        # Table header
        f.write(f"{'Activity':<20} {'Total':<8} {'Answered':<10} {'Unknown':<10} {'Correct':<10} {'Acc (all)':<12} {'Acc (ans)':<12}\n")
        f.write("-"*80 + "\n")
        
        for activity in activities:
            stats = analysis['activity_stats'][activity]
            
            # Calculate accuracies
            acc_all = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            acc_ans = (stats['correct_answered'] / stats['answered'] * 100) if stats['answered'] > 0 else 0
            
            f.write(f"{activity:<20} "
                   f"{stats['total']:<8} "
                   f"{stats['answered']:<10} "
                   f"{stats['unknown']:<10} "
                   f"{stats['correct_answered']:<10} "
                   f"{acc_all:<11.1f}% "
                   f"{acc_ans:<11.1f}%\n")
        
        f.write("\n" + "-"*80 + "\n\n")
        
        # Unknown Queries Details
        f.write("UNKNOWN QUERIES (LLM Could Not Answer)\n")
        f.write("="*80 + "\n\n")
        
        unknown_queries = [r for r in results if r['predicted_label'] == 'Unknown']
        
        if unknown_queries:
            f.write(f"Total Unknown: {len(unknown_queries)}\n\n")
            
            # Group by activity
            unknown_by_activity = defaultdict(list)
            for query in unknown_queries:
                unknown_by_activity[query['true_label']].append(query['query_file'])
            
            for activity in sorted(unknown_by_activity.keys()):
                f.write(f"{activity}:\n")
                for query_file in unknown_by_activity[activity]:
                    f.write(f"  - {query_file}\n")
                f.write("\n")
        else:
            f.write("No Unknown queries! LLM provided predictions for all queries.\n\n")
        
        f.write("-"*80 + "\n\n")
        
        # Correct Predictions (Excluding Unknown)
        f.write("CORRECT PREDICTIONS (Answered Queries Only)\n")
        f.write("="*80 + "\n\n")
        
        correct_answered = [r for r in results if r['correct'] and r['predicted_label'] != 'Unknown']
        
        f.write(f"Total Correct (Answered): {len(correct_answered)}\n\n")
        
        # Group by activity
        correct_by_activity = defaultdict(list)
        for query in correct_answered:
            correct_by_activity[query['true_label']].append(query['query_file'])
        
        for activity in sorted(correct_by_activity.keys()):
            f.write(f"{activity}: {len(correct_by_activity[activity])} correct\n")
            for query_file in correct_by_activity[activity]:
                f.write(f"  - {query_file}\n")
            f.write("\n")
        
        f.write("-"*80 + "\n\n")
        
        # Incorrect Predictions (Excluding Unknown)
        f.write("INCORRECT PREDICTIONS (Answered Queries Only)\n")
        f.write("="*80 + "\n\n")
        
        incorrect_answered = [r for r in results if not r['correct'] and r['predicted_label'] != 'Unknown']
        
        f.write(f"Total Incorrect (Answered): {len(incorrect_answered)}\n\n")
        
        for query in incorrect_answered:
            f.write(f"File: {query['query_file']}\n")
            f.write(f"  True: {query['true_label']}\n")
            f.write(f"  Predicted: {query['predicted_label']}\n\n")
        
        f.write("-"*80 + "\n\n")
        
        # Summary
        f.write("SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Queries: {analysis['total_queries']}\n")
        f.write(f"Unknown Rate: {analysis['unknown_rate']:.2f}% ({analysis['unknown_count']} queries)\n")
        f.write(f"Answer Rate: {analysis['answered_rate']:.2f}% ({analysis['answered_count']} queries)\n\n")
        
        f.write(f"Accuracy (including Unknown as incorrect): {analysis['accuracy_with_unknown']:.2f}%\n")
        f.write(f"Accuracy (excluding Unknown): {analysis['accuracy_excluding_unknown']:.2f}%\n\n")
        
        f.write(f"When LLM provides an answer, accuracy is {analysis['accuracy_excluding_unknown']:.2f}%\n")
        f.write(f"The Unknown rate of {analysis['unknown_rate']:.2f}% indicates {analysis['unknown_count']} queries where GraphRAG could not find relevant information.\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF ANALYSIS\n")
        f.write("="*80 + "\n")


def main():
    """
    Main function
    """
    print("="*80)
    print("GraphRAG Evaluation Results Analysis")
    print("="*80)
    
    input_file = "evaluation_results.txt"
    output_file = "evaluation_analysis_unknown.txt"
    
    # Parse results
    print(f"\n1. Parsing evaluation results from {input_file}...")
    results = parse_evaluation_results(input_file)
    print(f"   ✓ Parsed {len(results)} query results")
    
    # Analyze
    print(f"\n2. Analyzing results...")
    analysis = analyze_results(results)
    
    print(f"\n   Summary:")
    print(f"   - Total Queries: {analysis['total_queries']}")
    print(f"   - Answered: {analysis['answered_count']} ({analysis['answered_rate']:.1f}%)")
    print(f"   - Unknown: {analysis['unknown_count']} ({analysis['unknown_rate']:.1f}%)")
    print(f"\n   - Accuracy (including Unknown): {analysis['accuracy_with_unknown']:.2f}%")
    print(f"   - Accuracy (excluding Unknown): {analysis['accuracy_excluding_unknown']:.2f}%")
    
    # Generate report
    print(f"\n3. Generating detailed analysis report...")
    generate_analysis_report(results, analysis, output_file)
    print(f"   ✓ Analysis saved to: {output_file}")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print(f"\n• Unknown Rate: {analysis['unknown_rate']:.2f}%")
    print(f"  - {analysis['unknown_count']} out of {analysis['total_queries']} queries could not be answered")
    print(f"  - LLM responded with 'Sorry, I'm not able to provide an answer'")
    print(f"\n• Answer Rate: {analysis['answered_rate']:.2f}%")
    print(f"  - {analysis['answered_count']} queries received predictions from the LLM")
    print(f"\n• Accuracy Comparison:")
    print(f"  - Including Unknown (as incorrect): {analysis['accuracy_with_unknown']:.2f}%")
    print(f"  - Excluding Unknown (only answered): {analysis['accuracy_excluding_unknown']:.2f}%")
    print(f"  - Improvement: +{analysis['accuracy_excluding_unknown'] - analysis['accuracy_with_unknown']:.2f}%")
    print(f"\n• When the LLM CAN answer, it is correct {analysis['accuracy_excluding_unknown']:.2f}% of the time")
    print(f"• The main issue is {analysis['unknown_rate']:.2f}% of queries get no answer from GraphRAG")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nDetailed report saved to: {output_file}")
    
    return analysis


if __name__ == "__main__":
    analysis = main()
