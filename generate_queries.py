"""
Generate query files for UCI HAR dataset
Creates 180 query files (30 subjects × 6 activities) with statistical descriptions
"""

import numpy as np
import os

def load_all_data(base_path):
    """Load both train and test data to get all 30 subjects."""
    print("Loading train and test data...")
    
    # Load feature names
    feature_names = []
    with open(os.path.join(base_path, "features.txt"), 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                feature_names.append(parts[1])
    
    # Load training data
    X_train = np.loadtxt(os.path.join(base_path, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(base_path, "train", "y_train.txt"), dtype=int)
    subject_train = np.loadtxt(os.path.join(base_path, "train", "subject_train.txt"), dtype=int)
    
    # Load test data
    X_test = np.loadtxt(os.path.join(base_path, "test", "X_test.txt"))
    y_test = np.loadtxt(os.path.join(base_path, "test", "y_test.txt"), dtype=int)
    subject_test = np.loadtxt(os.path.join(base_path, "test", "subject_test.txt"), dtype=int)
    
    # Combine train and test
    X_all = np.vstack([X_train, X_test])
    y_all = np.hstack([y_train, y_test])
    subject_all = np.hstack([subject_train, subject_test])
    
    print(f"✓ Loaded {X_all.shape[0]} total samples from {len(set(subject_all))} subjects")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    return X_all, y_all, subject_all, feature_names

def get_feature_description(feature_name):
    """Generate human-readable description of a feature."""
    descriptions = {
        'tBodyAcc-mean()-X': 'mean body acceleration in X direction',
        'tBodyAcc-mean()-Y': 'mean body acceleration in Y direction',
        'tBodyAcc-mean()-Z': 'mean body acceleration in Z direction',
        'tBodyAcc-std()-X': 'standard deviation of body acceleration in X direction',
        'tBodyAcc-std()-Y': 'standard deviation of body acceleration in Y direction',
        'tBodyAcc-std()-Z': 'standard deviation of body acceleration in Z direction',
        'tBodyAcc-max()-X': 'maximum body acceleration in X direction',
        'tBodyAcc-max()-Y': 'maximum body acceleration in Y direction',
        'tBodyAcc-max()-Z': 'maximum body acceleration in Z direction',
        'tGravityAcc-mean()-X': 'mean gravity acceleration in X direction',
        'tGravityAcc-mean()-Y': 'mean gravity acceleration in Y direction',
        'tGravityAcc-mean()-Z': 'mean gravity acceleration in Z direction',
        'tGravityAcc-std()-X': 'standard deviation of gravity acceleration in X direction',
        'tGravityAcc-std()-Y': 'standard deviation of gravity acceleration in Y direction',
        'tGravityAcc-std()-Z': 'standard deviation of gravity acceleration in Z direction',
        'tBodyAccMag-mean()': 'mean body acceleration magnitude',
        'tBodyAccMag-std()': 'standard deviation of body acceleration magnitude',
        'tGravityAccMag-mean()': 'mean gravity acceleration magnitude',
        'tGravityAccMag-std()': 'standard deviation of gravity acceleration magnitude',
        'tBodyAccJerkMag-mean()': 'mean jerk magnitude of body acceleration',
        'tBodyAccJerkMag-std()': 'standard deviation of jerk magnitude',
        'tBodyGyroMag-mean()': 'mean gyroscope magnitude',
        'tBodyGyroMag-std()': 'standard deviation of gyroscope magnitude',
        'angle(Y,gravityMean)': 'angle between Y axis and gravity mean',
        'angle(Z,gravityMean)': 'angle between Z axis and gravity mean',
        'angle(X,gravityMean)': 'angle between X axis and gravity mean'
    }
    
    return descriptions.get(feature_name, feature_name)

def generate_query_text(feature_data, feature_names, selected_features):
    """
    Generate query text in the specified format.
    feature_data: numpy array of shape (n_samples, n_features)
    """
    lines = []
    
    # Opening paragraph
    lines.append("This sensor reading contains measurements across six key features extracted from accelerometer and gyroscope data. The following paragraphs describe the statistical properties of each feature.")
    lines.append("")
    
    # For each of the 6 selected features
    for feat_idx in selected_features:
        feat_name = feature_names[feat_idx]
        feat_values = feature_data[:, feat_idx]
        
        # Calculate statistics
        mean_val = np.mean(feat_values)
        std_val = np.std(feat_values)
        median_val = np.median(feat_values)
        min_val = np.min(feat_values)
        max_val = np.max(feat_values)
        range_val = max_val - min_val
        q25 = np.percentile(feat_values, 25)
        q75 = np.percentile(feat_values, 75)
        iqr = q75 - q25
        
        # Get description
        description = get_feature_description(feat_name)
        
        # Generate paragraph
        paragraph = (
            f"For the feature {feat_name}, which represents {description}, "
            f"the mean value is {mean_val:.6f}, with standard deviation being {std_val:.6f}. "
            f"The median value is {median_val:.6f}. "
            f"The minimum recorded value is {min_val:.6f}, while the maximum is {max_val:.6f}, "
            f"giving a range of {range_val:.6f}. "
            f"The 25th percentile is {q25:.6f} and the 75th percentile is {q75:.6f}, "
            f"with an interquartile range of {iqr:.6f}."
        )
        
        lines.append(paragraph)
        lines.append("")
    
    # Closing question
    lines.append('What activity does this sensor data most likely represent? Choose from "Sitting", "Laying", "Walking", "Standing", "Walking_Downstairs", "Walking_Upstairs".')
    
    return "\n".join(lines)

def generate_all_queries(X_all, y_all, subject_all, feature_names, output_dir):
    """Generate all 180 query files."""
    
    # Select 6 features to use (based on user specification)
    # Note: features.txt uses 1-based numbering, but Python arrays are 0-based
    # So we subtract 1 from each feature number
    selected_features = [
        9,    # Feature 10: tBodyAcc-max()-X
        42,   # Feature 43: tGravityAcc-mean()-Z
        200,  # Feature 201: tBodyAccMag-mean()
        201,  # Feature 202: tBodyAccMag-std()
        214,  # Feature 215: tGravityAccMag-std()
        226   # Feature 227: tBodyAccJerkMag-mean()
    ]
    
    print(f"\nSelected features:")
    for idx in selected_features:
        print(f"  Feature {idx+1} (array index {idx}): {feature_names[idx]}")
    
    # Activity mapping
    activity_map = {
        1: 'Walking',
        2: 'Walking_Upstairs',
        3: 'Walking_Downstairs',
        4: 'Sitting',
        5: 'Standing',
        6: 'Laying'
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Ground truth list
    ground_truth = []
    
    # Generate queries for all 30 subjects
    all_subjects = sorted(set(subject_all))
    print(f"\nGenerating queries for {len(all_subjects)} subjects...")
    
    query_count = 0
    
    for subject in all_subjects:
        # Get all data for this subject
        subject_mask = subject_all == subject
        subject_activities = y_all[subject_mask]
        
        # Get unique activities for this subject
        unique_activities = sorted(set(subject_activities))
        
        for activity in unique_activities:
            # Get all samples for this subject-activity combination
            activity_mask = subject_activities == activity
            subject_indices = np.where(subject_mask)[0]
            activity_indices = subject_indices[activity_mask]
            
            # Extract feature data
            feature_data = X_all[activity_indices]
            
            # Generate query text
            query_text = generate_query_text(feature_data, feature_names, selected_features)
            
            # Generate filename
            activity_name = activity_map[activity]
            filename = f"query_{subject}_{activity_name}.txt"
            filepath = os.path.join(output_dir, filename)
            
            # Write query file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(query_text)
            
            # Add to ground truth
            ground_truth.append(f"{filename}\t{activity_name}")
            
            query_count += 1
            
            if query_count % 30 == 0:
                print(f"  Generated {query_count} queries...")
    
    print(f"✓ Generated {query_count} query files")
    
    # Write ground truth file
    gt_path = os.path.join(output_dir, "ground_truth.txt")
    with open(gt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(ground_truth))
    
    print(f"✓ Saved ground truth to {gt_path}")
    
    return query_count

def main():
    # File paths
    base_path = r"/home/m202/M202/Richard/UCI HAR Dataset"
    output_dir = "/home/m202/M202/Richard/query"
    
    print("="*80)
    print("UCI HAR QUERY GENERATOR")
    print("="*80)
    
    # Load all data (train + test)
    X_all, y_all, subject_all, feature_names = load_all_data(base_path)
    
    # Verify we have all 30 subjects
    unique_subjects = sorted(set(subject_all))
    print(f"\nSubjects found: {unique_subjects}")
    print(f"Total: {len(unique_subjects)} subjects")
    
    if len(unique_subjects) != 30:
        print(f"WARNING: Expected 30 subjects, found {len(unique_subjects)}")
    
    # Generate all queries
    query_count = generate_all_queries(X_all, y_all, subject_all, feature_names, output_dir)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nGenerated {query_count} query files")
    print(f"Output directory: {output_dir}")
    print(f"Ground truth file: {output_dir}/ground_truth.txt")
    
    print("\nQuery file format:")
    print("  query_<subject>_<activity>.txt")
    print("  Examples:")
    print("    query_1_Walking.txt")
    print("    query_1_Sitting.txt")
    print("    query_2_Walking_Upstairs.txt")
    
    print("\nExpected: 30 subjects × 6 activities = 180 files")
    print(f"Actual: {query_count} files")
    
    if query_count == 180:
        print("\n✓ SUCCESS: All 180 queries generated!")
    else:
        print(f"\n⚠ Note: Some subjects may not have all 6 activities in the dataset")

if __name__ == '__main__':
    main()