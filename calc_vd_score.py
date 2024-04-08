import numpy as np
from sklearn.metrics import roc_curve
import argparse
import json

def calculate_vul_det_score(predictions, ground_truth, target_fpr=0.005):
    """
    Calculate the vulnerability detection score (VD-S) given a tolerable FPR.
    
    Args:
    - predictions: List of model prediction probabilities for the positive class.
    - ground_truth: List of ground truth labels, where 1 means vulnerable class, and 0 means benign class.
    - target_fpr: The tolerable false positive rate.
    
    Returns:
    - vds: Calculated vulnerability detection score given the acceptable .
    - threshold: The classification threashold for vulnerable prediction.
    """
    
    # Calculate FPR, TPR, and thresholds using ROC curve
    fpr, tpr, thresholds = roc_curve(ground_truth, predictions)
    
    # Filter thresholds where FPR is less than or equal to the target FPR
    valid_indices = np.where(fpr <= target_fpr)[0]
    
    # Choose the threshold with the largest FPR that is still below the target FPR, if possible
    if len(valid_indices) > 0:
        idx = valid_indices[-1]  # Last index where FPR is below or equal to target FPR
    else:
        # If no such threshold exists (unlikely), default to the closest to the target FPR
        idx = np.abs(fpr - target_fpr).argmin()
        
    chosen_threshold = thresholds[idx]
    
    # Classify predictions based on the chosen threshold
    classified_preds = [1 if pred >= chosen_threshold else 0 for pred in predictions]
    
    # Calculate VD-S
    fn = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == 1 and classified_preds[i] == 0])
    tp = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == 1 and classified_preds[i] == 1])
    vds = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return vds, chosen_threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate vulnerability detection score.")
    parser.add_argument("--pred_file", type=str, help="Path to the file containing model predictions: predictions.txt")
    parser.add_argument("--test_file", type=str, help="Path to the file containing ground truth labels: test.jsonl")
    
    args = parser.parse_args()

    # Extract ground truth labels
    with open(args.test_file, 'r') as f:
        lines = f.readlines()
        idx2label = {} # idx to the ground truth label mapping
        for line in lines:
            data = json.loads(line)
            idx2label[data['idx']] = data['target']

    # Extract model predictions
    with open(args.pred_file, 'r') as f:
        pred_lines = f.readlines()
    pred_prob = {}
    for p in pred_lines:
        idx, _, prob = p.strip().split("\t") # the predictions.txt file should be in the format: idx\tlabel\tprobability
        idx = int(idx)
        prob = float(prob)
        pred_prob[idx] = prob
    ground_truth = []
    pred = []
    for idx in idx2label:
        ground_truth.append(idx2label[idx])
        pred.append(pred_prob[idx])

    vds, threshold = calculate_vul_det_score(pred, ground_truth, target_fpr=0.005)
    print(f"VD-S: {vds}, Picked FPR: {target_fpr}, Threshold: {threshold}")
