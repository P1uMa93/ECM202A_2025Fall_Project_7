"""
Calculate Per-Activity Accuracy from GraphRAG Evaluation Results

This script:
1. Parses evaluation_results.txt
2. Calculates accuracy for each activity separately
3. Shows confusion matrix (what each activity was predicted as)
4. Identifies most/least accurate activities
5. Outputs detailed per-activity analysis
"""

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
            
            continue
        
        i += 1
    
    # Don't forget the last result
    if current_result:
        results.append(current_result)
    
    return results


def calculate_per_activity_accuracy(results):
    """
    Calculate accuracy for each activity
    
    Returns:
        dict: Per-activity statistics
    """
    activities = ['Walking', 'Walking_Upstairs', 'Walking_Downstairs', 'Sitting', 'Standing', 'Laying']
    
    # Initialize statistics
    activity_stats = {}
    for activity in activities:
        activity_stats[activity] = {
            'total': 0,
            'correct': 0,
            'incorrect': 0,
            'unknown': 0,
            'accuracy': 0.0,
            'predictions': defaultdict(int)  # What this activity was predicted as
        }
    
    # Calculate statistics
    for result in results:
        true_label = result['true_label']
        predicted_label = result['predicted_label']
        is_correct = result['correct']
        
        if true_label not in activity_stats:
            continue
        
        # Update counts
        activity_stats[true_label]['total'] += 1
        
        if predicted_label == 'Unknown':
            activity_stats[true_label]['unknown'] += 1
            activity_stats[true_label]['incorrect'] += 1
        elif is_correct:
            activity_stats[true_label]['correct'] += 1
        else:
            activity_stats[true_label]['incorrect'] += 1
        
        # Track what it was predicted as
        activity_stats[true_label]['predictions'][predicted_label] += 1
    
    # Calculate accuracy
    for activity in activities:
        stats = activity_stats[activity]
        if stats['total'] > 0:
            stats['accuracy'] = (stats['correct'] / stats['total']) * 100
    
    return activity_stats


def generate_per_activity_report(results, activity_stats, output_file):
    """
    Generate detailed per-activity accuracy report
    """
    activities = ['Walking', 'Walking_Upstairs', 'Walking_Downstairs', 'Sitting', 'Standing', 'Laying']
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Per-Activity Accuracy Analysis\n")
        f.write("GraphRAG Human Activity Recognition Evaluation\n")
        f.write("="*80 + "\n\n")
        
        # Overall summary table
        f.write("ACCURACY SUMMARY BY ACTIVITY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Activity':<25} {'Total':<8} {'Correct':<10} {'Incorrect':<12} {'Unknown':<10} {'Accuracy':<10}\n")
        f.write("-"*80 + "\n")
        
        total_all = 0
        correct_all = 0
        
        for activity in activities:
            stats = activity_stats[activity]
            total_all += stats['total']
            correct_all += stats['correct']
            
            f.write(f"{activity:<25} "
                   f"{stats['total']:<8} "
                   f"{stats['correct']:<10} "
                   f"{stats['incorrect']:<12} "
                   f"{stats['unknown']:<10} "
                   f"{stats['accuracy']:<9.2f}%\n")
        
        overall_accuracy = (correct_all / total_all * 100) if total_all > 0 else 0
        f.write("-"*80 + "\n")
        f.write(f"{'OVERALL':<25} "
               f"{total_all:<8} "
               f"{correct_all:<10} "
               f"{total_all - correct_all:<12} "
               f"{'--':<10} "
               f"{overall_accuracy:<9.2f}%\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Rank by accuracy
        f.write("ACTIVITIES RANKED BY ACCURACY\n")
        f.write("="*80 + "\n\n")
        
        ranked = sorted(activities, key=lambda x: activity_stats[x]['accuracy'], reverse=True)
        
        f.write(f"{'Rank':<8} {'Activity':<25} {'Accuracy':<12} {'Correct/Total':<15}\n")
        f.write("-"*80 + "\n")
        
        for rank, activity in enumerate(ranked, 1):
            stats = activity_stats[activity]
            f.write(f"{rank:<8} "
                   f"{activity:<25} "
                   f"{stats['accuracy']:<11.2f}% "
                   f"{stats['correct']}/{stats['total']}\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Detailed per-activity breakdown
        f.write("DETAILED PER-ACTIVITY BREAKDOWN\n")
        f.write("="*80 + "\n\n")
        
        for activity in activities:
            stats = activity_stats[activity]
            
            f.write(f"Activity: {activity}\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Queries: {stats['total']}\n")
            f.write(f"Correct: {stats['correct']} ({stats['accuracy']:.2f}%)\n")
            f.write(f"Incorrect: {stats['incorrect']}\n")
            f.write(f"Unknown: {stats['unknown']}\n\n")
            
            # Show what this activity was predicted as
            f.write("Prediction Distribution:\n")
            for pred_label, count in sorted(stats['predictions'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / stats['total'] * 100) if stats['total'] > 0 else 0
                status = "✓" if pred_label == activity else "✗"
                f.write(f"  {status} {pred_label:<25} {count:>3} ({percentage:>5.1f}%)\n")
            
            f.write("\n")
            
            # Show actual queries for this activity
            f.write("Queries:\n")
            activity_results = [r for r in results if r['true_label'] == activity]
            
            for result in activity_results:
                status = "✓" if result['correct'] else "✗"
                f.write(f"  {status} {result['query_file']:<35} → {result['predicted_label']}\n")
            
            f.write("\n" + "="*80 + "\n\n")
        
        # Confusion insights
        f.write("COMMON MISCLASSIFICATIONS\n")
        f.write("="*80 + "\n\n")
        
        misclassifications = []
        for activity in activities:
            stats = activity_stats[activity]
            for pred_label, count in stats['predictions'].items():
                if pred_label != activity and pred_label != 'Unknown' and count > 0:
                    misclassifications.append((activity, pred_label, count))
        
        # Sort by count
        misclassifications.sort(key=lambda x: x[2], reverse=True)
        
        if misclassifications:
            f.write(f"{'True Activity':<25} {'Predicted As':<25} {'Count':<10}\n")
            f.write("-"*80 + "\n")
            
            for true_act, pred_act, count in misclassifications:
                f.write(f"{true_act:<25} {pred_act:<25} {count:<10}\n")
        else:
            f.write("No misclassifications (excluding Unknown).\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Best and worst
        f.write("BEST AND WORST PERFORMING ACTIVITIES\n")
        f.write("="*80 + "\n\n")
        
        best = max(activities, key=lambda x: activity_stats[x]['accuracy'])
        worst = min(activities, key=lambda x: activity_stats[x]['accuracy'])
        
        f.write(f"Best: {best}\n")
        f.write(f"  Accuracy: {activity_stats[best]['accuracy']:.2f}%\n")
        f.write(f"  Correct: {activity_stats[best]['correct']}/{activity_stats[best]['total']}\n\n")
        
        f.write(f"Worst: {worst}\n")
        f.write(f"  Accuracy: {activity_stats[worst]['accuracy']:.2f}%\n")
        f.write(f"  Correct: {activity_stats[worst]['correct']}/{activity_stats[worst]['total']}\n\n")
        
        f.write(f"Accuracy Gap: {activity_stats[best]['accuracy'] - activity_stats[worst]['accuracy']:.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF PER-ACTIVITY ANALYSIS\n")
        f.write("="*80 + "\n")


def print_summary_table(activity_stats):
    """
    Print summary table to console
    """
    activities = ['Walking', 'Walking_Upstairs', 'Walking_Downstairs', 'Sitting', 'Standing', 'Laying']
    
    print("\n" + "="*80)
    print("PER-ACTIVITY ACCURACY SUMMARY")
    print("="*80)
    print()
    print(f"{'Activity':<25} {'Total':<8} {'Correct':<10} {'Accuracy':<10}")
    print("-"*80)
    
    total_all = 0
    correct_all = 0
    
    for activity in activities:
        stats = activity_stats[activity]
        total_all += stats['total']
        correct_all += stats['correct']
        
        print(f"{activity:<25} {stats['total']:<8} {stats['correct']:<10} {stats['accuracy']:<9.2f}%")
    
    overall_accuracy = (correct_all / total_all * 100) if total_all > 0 else 0
    print("-"*80)
    print(f"{'OVERALL':<25} {total_all:<8} {correct_all:<10} {overall_accuracy:<9.2f}%")
    print()


def main():
    """
    Main function
    """
    print("="*80)
    print("Per-Activity Accuracy Calculator")
    print("GraphRAG Human Activity Recognition Evaluation")
    print("="*80)
    
    input_file = "evaluation_results.txt"
    output_file = "per_activity_accuracy.txt"
    
    # Parse results
    print(f"\n1. Parsing evaluation results from {input_file}...")
    results = parse_evaluation_results(input_file)
    print(f"   ✓ Parsed {len(results)} query results")
    
    # Calculate per-activity accuracy
    print(f"\n2. Calculating per-activity accuracy...")
    activity_stats = calculate_per_activity_accuracy(results)
    
    # Print summary table
    print_summary_table(activity_stats)
    
    # Find best and worst
    activities = ['Walking', 'Walking_Upstairs', 'Walking_Downstairs', 'Sitting', 'Standing', 'Laying']
    best = max(activities, key=lambda x: activity_stats[x]['accuracy'])
    worst = min(activities, key=lambda x: activity_stats[x]['accuracy'])
    
    print("="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print()
    print(f"Best Performing: {best}")
    print(f"  - Accuracy: {activity_stats[best]['accuracy']:.2f}%")
    print(f"  - Correct: {activity_stats[best]['correct']}/{activity_stats[best]['total']}")
    print()
    print(f"Worst Performing: {worst}")
    print(f"  - Accuracy: {activity_stats[worst]['accuracy']:.2f}%")
    print(f"  - Correct: {activity_stats[worst]['correct']}/{activity_stats[worst]['total']}")
    print()
    print(f"Accuracy Range: {activity_stats[worst]['accuracy']:.2f}% - {activity_stats[best]['accuracy']:.2f}%")
    print(f"Gap: {activity_stats[best]['accuracy'] - activity_stats[worst]['accuracy']:.2f}%")
    
    # Generate detailed report
    print(f"\n3. Generating detailed per-activity report...")
    generate_per_activity_report(results, activity_stats, output_file)
    print(f"   ✓ Report saved to: {output_file}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nDetailed per-activity analysis saved to: {output_file}")
    print()
    
    return activity_stats


if __name__ == "__main__":
    activity_stats = main()
