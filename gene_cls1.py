import datetime
import pickle
import argparse
from geneformer import Classifier
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Gene classification using Geneformer")
    parser.add_argument("--input_data_file", 
                       type=str, 
                       default="/fs/scratch/PCON0022/ch/Geneformer/examples/example_input_files/gc-30M_sample50k.dataset",
                       help="Path to the input dataset file")
    parser.add_argument("--output_prefix", 
                       type=str, 
                       default="tf_dosage_sens_test",
                       help="Prefix for output files")
    parser.add_argument("--gene_dict_path", 
                       type=str, 
                       default="/fs/scratch/PCON0022/ch/Geneformer/examples/example_input_files/dosage_sensitivity_TFs.pickle",
                       help="Path to the gene dictionary pickle file")
    parser.add_argument("--epochs", 
                       type=int, 
                       default=50,
                       help="Number of training epochs (default: 1 for gene classification)")
    return parser.parse_args()

args = parse_args()

print(f"Using arguments:")
print(f"  Input data file: {args.input_data_file}")
print(f"  Output prefix: {args.output_prefix}")
print(f"  Gene dict path: {args.gene_dict_path}")
print(f"  Training epochs: {args.epochs}")
print()

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
datestamp_min = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"

output_prefix = args.output_prefix
output_dir = f"/fs/scratch/PCON0022/ch/Geneformer/examples/outputs/{datestamp}"
os.makedirs(output_dir, exist_ok=True)
gene_dict_path = args.gene_dict_path
# Example input_data_file: https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/blob/main/example_input_files/gene_classification/dosage_sensitive_tfs/dosage_sensitivity_TFs.pickle
with open(gene_dict_path, "rb") as fp:
    gene_class_dict = pickle.load(fp)

# OF NOTE: token_dictionary_file must be set to the gc-30M token dictionary if using a 30M series model
# (otherwise the Classifier will use the current default model dictionary)
# 30M token dictionary: https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl

# Define custom training arguments with user-specified epochs
training_args = {
    "num_train_epochs": args.epochs
}

cc = Classifier(classifier="gene",
                gene_class_dict = gene_class_dict,
                max_ncells = 10_000,
                freeze_layers = 4,
                num_crossval_splits = 5,
                forward_batch_size=200,
                nproc=16,
                training_args=training_args)


# Example input_data_file for 30M model series: https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/gene_classification/dosage_sensitive_tfs/gc-30M_sample50k.dataset
cc.prepare_data(input_data_file=args.input_data_file,
                output_directory=output_dir,
                output_prefix=output_prefix)

# 6 layer 30M Geneformer model: https://huggingface.co/ctheodoris/Geneformer/blob/main/gf-6L-30M-i2048/model.safetensors
all_metrics = cc.validate(model_directory="/fs/scratch/PCON0022/ch/Geneformer/gf-6L-30M-i2048",
                          prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled.dataset",
                          id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
                          output_directory=output_dir,
                          output_prefix=output_prefix)

cc.plot_conf_mat(
    conf_mat_dict={"Geneformer": all_metrics["conf_matrix"]},
    output_directory=output_dir,
    output_prefix=output_prefix,
)

cc.plot_roc(
    roc_metric_dict={"Geneformer": all_metrics["all_roc_metrics"]},
    model_style_dict={"Geneformer": {"color": "red", "linestyle": "-"}},
    title="Dosage-sensitive vs -insensitive factors",
    output_directory=output_dir,
    output_prefix=output_prefix,
)

# Calculate detailed classification metrics
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score

print("="*80)
print("CROSS-VALIDATION METRICS ANALYSIS")
print("="*80)

# Extract per-fold metrics from Geneformer's all_metrics
fold_macro_f1_scores = all_metrics["macro_f1"]
fold_accuracies = all_metrics["acc"]

# Extract ROC metrics per fold for binary classification
fold_roc_aucs = []
if all_metrics["all_roc_metrics"] is not None:
    fold_roc_aucs = all_metrics["all_roc_metrics"]["all_roc_auc"]

print(f"Number of cross-validation folds: {len(fold_macro_f1_scores)}")
print()

print("=== PER-FOLD METRICS ===")
for fold_idx, (macro_f1, acc) in enumerate(zip(fold_macro_f1_scores, fold_accuracies), 1):
    print(f"Fold {fold_idx}:")
    print(f"  Macro F1 Score: {macro_f1:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    if fold_roc_aucs:
        print(f"  ROC AUC: {fold_roc_aucs[fold_idx-1]:.4f}")
    print()

print("=== CROSS-VALIDATION SUMMARY STATISTICS ===")
print(f"Macro F1 Score - Mean: {np.mean(fold_macro_f1_scores):.4f} ± {np.std(fold_macro_f1_scores):.4f}")
print(f"Accuracy - Mean: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
if fold_roc_aucs:
    print(f"ROC AUC - Mean: {np.mean(fold_roc_aucs):.4f} ± {np.std(fold_roc_aucs):.4f}")
    print(f"ROC AUC (Geneformer weighted): {all_metrics['all_roc_metrics']['roc_auc']:.4f} ± {all_metrics['all_roc_metrics']['roc_auc_sd']:.4f}")
print()

# Extract overall confusion matrix (summed across all folds)
conf_matrix = all_metrics["conf_matrix"]
print("=== AGGREGATED CONFUSION MATRIX (All Folds Combined) ===")
print("Confusion Matrix:")
print(conf_matrix)
print()

# Convert pandas DataFrame to numpy array and extract values
conf_matrix_array = conf_matrix.values
print("Confusion Matrix Shape:", conf_matrix_array.shape)
print("Confusion Matrix Array:")
print(conf_matrix_array)
print()

# For binary classification: 
# conf_matrix[0,0] = True Positives for class 0 (Dosage-sensitive TFs)
# conf_matrix[0,1] = False Negatives for class 0 (predicted as class 1)
# conf_matrix[1,0] = False Positives for class 0 (predicted as class 0 but actually class 1)
# conf_matrix[1,1] = True Negatives for class 0 (correctly predicted as class 1)

tp = conf_matrix_array[0, 0]  # True positives for Dosage-sensitive TFs
fn = conf_matrix_array[0, 1]  # False negatives for Dosage-sensitive TFs
fp = conf_matrix_array[1, 0]  # False positives for Dosage-sensitive TFs
tn = conf_matrix_array[1, 1]  # True negatives for Dosage-sensitive TFs

print(f"TP (True Positives): {tp}")
print(f"FN (False Negatives): {fn}")
print(f"FP (False Positives): {fp}")
print(f"TN (True Negatives): {tn}")
print()

# Calculate metrics for positive class (Dosage-sensitive TFs)
accuracy = (tp + tn) / (tp + tn + fp + fn)
balanced_accuracy = 0.5 * ((tp / (tp + fn)) + (tn / (tn + fp)))
recall_pos = tp / (tp + fn)  # Recall for positive class
precision_pos = tp / (tp + fp)  # Precision for positive class
f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos)  # F1 for positive class

# Calculate metrics for negative class (Dosage-insensitive TFs)
recall_neg = tn / (tn + fp)  # Recall for negative class
precision_neg = tn / (tn + fn)  # Precision for negative class
f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg)  # F1 for negative class

# Calculate macro-averaged metrics
recall_macro = (recall_pos + recall_neg) / 2
precision_macro = (precision_pos + precision_neg) / 2
f1_macro = (f1_pos + f1_neg) / 2

print("=== AGGREGATED POSITIVE CLASS METRICS (Dosage-sensitive TFs) ===")
print(f"Precision (Positive): {precision_pos:.4f}")
print(f"Recall (Positive): {recall_pos:.4f}")
print(f"F1 Score (Positive): {f1_pos:.4f}")
print()

print("=== AGGREGATED NEGATIVE CLASS METRICS (Dosage-insensitive TFs) ===")
print(f"Precision (Negative): {precision_neg:.4f}")
print(f"Recall (Negative): {recall_neg:.4f}")
print(f"F1 Score (Negative): {f1_neg:.4f}")
print()

print("=== AGGREGATED MACRO-AVERAGED METRICS ===")
print(f"Precision (Macro): {precision_macro:.4f}")
print(f"Recall (Macro): {recall_macro:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")
print()

print("=== AGGREGATED OVERALL METRICS ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print()

# Additional metrics for completeness
specificity = tn / (tn + fp)  # True negative rate
print(f"Specificity: {specificity:.4f}")

# Create a comprehensive summary dictionary
metrics_summary = {
    "cross_val_macro_f1_mean": np.mean(fold_macro_f1_scores),
    "cross_val_macro_f1_std": np.std(fold_macro_f1_scores),
    "cross_val_accuracy_mean": np.mean(fold_accuracies),
    "cross_val_accuracy_std": np.std(fold_accuracies),
    "aggregated_accuracy": accuracy,
    "aggregated_balanced_accuracy": balanced_accuracy,
    # Positive class metrics
    "aggregated_precision_positive": precision_pos,
    "aggregated_recall_positive": recall_pos,
    "aggregated_f1_score_positive": f1_pos,
    # Negative class metrics
    "aggregated_precision_negative": precision_neg,
    "aggregated_recall_negative": recall_neg,
    "aggregated_f1_score_negative": f1_neg,
    # Macro-averaged metrics
    "aggregated_precision_macro": precision_macro,
    "aggregated_recall_macro": recall_macro,
    "aggregated_f1_score_macro": f1_macro,
    "aggregated_specificity": specificity
}

if fold_roc_aucs:
    metrics_summary.update({
        "cross_val_roc_auc_mean": np.mean(fold_roc_aucs),
        "cross_val_roc_auc_std": np.std(fold_roc_aucs),
        "weighted_roc_auc": all_metrics["all_roc_metrics"]["roc_auc"],
        "weighted_roc_auc_sd": all_metrics["all_roc_metrics"]["roc_auc_sd"]
    })

print("\n" + "="*80)
print("COMPREHENSIVE METRICS SUMMARY:")
print("="*80)
for metric, value in metrics_summary.items():
    print(f"{metric}: {value:.4f}")

# Also calculate using sklearn for verification
print("\n=== SKLEARN VERIFICATION (Aggregated Confusion Matrix) ===")
# Create true and predicted labels from confusion matrix
y_true = []
y_pred = []
for i in range(conf_matrix_array.shape[0]):
    for j in range(conf_matrix_array.shape[1]):
        count = int(conf_matrix_array[i, j])
        y_true.extend([i] * count)
        y_pred.extend([j] * count)

print(f"sklearn Precision (macro): {precision_score(y_true, y_pred, average='macro'):.4f}")
print(f"sklearn Recall (macro): {recall_score(y_true, y_pred, average='macro'):.4f}")
print(f"sklearn F1 (macro): {f1_score(y_true, y_pred, average='macro'):.4f}")
print(f"sklearn Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"sklearn Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")

# Save detailed metrics to a file
metrics_output_file = f"{output_dir}/{output_prefix}_detailed_metrics.pkl"
with open(metrics_output_file, "wb") as f:
    detailed_metrics = {
        "per_fold_metrics": {
            "macro_f1_scores": fold_macro_f1_scores,
            "accuracies": fold_accuracies,
            "roc_aucs": fold_roc_aucs if fold_roc_aucs else None
        },
        "aggregated_metrics": metrics_summary,
        "aggregated_confusion_matrix": conf_matrix_array,
        "geneformer_all_metrics": all_metrics
    }
    pickle.dump(detailed_metrics, f)

print(f"\nDetailed metrics saved to: {metrics_output_file}")

def analyze_fold_predictions(output_dir, output_prefix, num_folds=5):
    """
    Analyze individual fold predictions if available.
    """
    print("\n" + "="*80)
    print("INDIVIDUAL FOLD ANALYSIS")
    print("="*80)
    
    fold_detailed_metrics = {}
    
    # Find the Geneformer output subdirectory
    import glob
    geneformer_subdir = None
    
    # Look for the pattern: *_geneformer_geneClassifier_{output_prefix}
    pattern = f"{output_dir}/*_geneformer_geneClassifier_{output_prefix}"
    matching_dirs = glob.glob(pattern)
    
    if matching_dirs:
        geneformer_subdir = matching_dirs[0]  # Take the first match
        print(f"Found Geneformer output directory: {geneformer_subdir}")
    else:
        print(f"No Geneformer output directory found matching pattern: {pattern}")
        print("Checking if predictions are in direct ksplit directories...")
        geneformer_subdir = output_dir
    
    for fold_idx in range(1, num_folds + 1):
        fold_dir = f"{geneformer_subdir}/ksplit{fold_idx}"
        predictions_file = f"{fold_dir}/{output_prefix}_pred_dict.pkl"
        
        if os.path.exists(predictions_file):
            print(f"\nAnalyzing Fold {fold_idx}:")
            print(f"Predictions file: {predictions_file}")
            
            try:
                # Load fold-specific predictions
                with open(predictions_file, "rb") as f:
                    pred_dict = pickle.load(f)
                
                y_pred_fold = pred_dict["pred_ids"]
                y_true_fold = pred_dict["label_ids"] 
                logits_fold = pred_dict["predictions"]
                
                # Calculate fold-specific confusion matrix and metrics
                from sklearn.metrics import confusion_matrix
                fold_conf_mat = confusion_matrix(y_true_fold, y_pred_fold)
                
                print(f"  Fold {fold_idx} Confusion Matrix:")
                print(f"  {fold_conf_mat}")
                
                # Calculate detailed metrics for this fold
                fold_tp = fold_conf_mat[0, 0]
                fold_fn = fold_conf_mat[0, 1] 
                fold_fp = fold_conf_mat[1, 0]
                fold_tn = fold_conf_mat[1, 1]
                
                fold_accuracy = (fold_tp + fold_tn) / (fold_tp + fold_tn + fold_fp + fold_fn)
                fold_precision_pos = fold_tp / (fold_tp + fold_fp) if (fold_tp + fold_fp) > 0 else 0
                fold_recall_pos = fold_tp / (fold_tp + fold_fn) if (fold_tp + fold_fn) > 0 else 0
                fold_f1_pos = 2 * (fold_precision_pos * fold_recall_pos) / (fold_precision_pos + fold_recall_pos) if (fold_precision_pos + fold_recall_pos) > 0 else 0
                
                fold_precision_neg = fold_tn / (fold_tn + fold_fn) if (fold_tn + fold_fn) > 0 else 0
                fold_recall_neg = fold_tn / (fold_tn + fold_fp) if (fold_tn + fold_fp) > 0 else 0
                fold_f1_neg = 2 * (fold_precision_neg * fold_recall_neg) / (fold_precision_neg + fold_recall_neg) if (fold_precision_neg + fold_recall_neg) > 0 else 0
                
                fold_macro_f1 = (fold_f1_pos + fold_f1_neg) / 2
                fold_macro_precision = (fold_precision_pos + fold_precision_neg) / 2
                fold_macro_recall = (fold_recall_pos + fold_recall_neg) / 2
                
                # Calculate balanced accuracy (average of recall for each class)
                fold_balanced_accuracy = (fold_recall_pos + fold_recall_neg) / 2
                
                # Calculate ROC AUC if binary classification
                fold_roc_auc = None
                if len(set(y_true_fold)) == 2:
                    from sklearn.metrics import roc_auc_score
                    # Convert logits to probabilities for positive class
                    fold_probs = [py_softmax(logit)[1] for logit in logits_fold]
                    fold_roc_auc = roc_auc_score(y_true_fold, fold_probs)
                
                print(f"  Fold {fold_idx} Metrics:")
                print(f"    Accuracy: {fold_accuracy:.4f}")
                print(f"    Balanced Accuracy: {fold_balanced_accuracy:.4f}")
                print(f"    Macro F1: {fold_macro_f1:.4f}")
                print(f"    Macro Precision: {fold_macro_precision:.4f}")
                print(f"    Macro Recall: {fold_macro_recall:.4f}")
                print(f"    Positive Class F1: {fold_f1_pos:.4f}")
                print(f"    Negative Class F1: {fold_f1_neg:.4f}")
                print(f"    Positive Class Precision: {fold_precision_pos:.4f}")
                print(f"    Positive Class Recall: {fold_recall_pos:.4f}")
                print(f"    Negative Class Precision: {fold_precision_neg:.4f}")
                print(f"    Negative Class Recall: {fold_recall_neg:.4f}")
                if fold_roc_auc is not None:
                    print(f"    ROC AUC: {fold_roc_auc:.4f}")
                
                # Store detailed metrics for this fold
                fold_detailed_metrics[f"fold_{fold_idx}"] = {
                    "confusion_matrix": fold_conf_mat,
                    "accuracy": fold_accuracy,
                    "balanced_accuracy": fold_balanced_accuracy,
                    "macro_f1": fold_macro_f1,
                    "macro_precision": fold_macro_precision,
                    "macro_recall": fold_macro_recall,
                    "precision_positive": fold_precision_pos,
                    "recall_positive": fold_recall_pos,
                    "f1_positive": fold_f1_pos,
                    "precision_negative": fold_precision_neg,
                    "recall_negative": fold_recall_neg,
                    "f1_negative": fold_f1_neg,
                    "roc_auc": fold_roc_auc,
                    "num_samples": len(y_true_fold)
                }
                
            except Exception as e:
                print(f"  Error loading predictions for fold {fold_idx}: {e}")
        else:
            print(f"\nFold {fold_idx}: Predictions file not found at {predictions_file}")
    
    if fold_detailed_metrics:
        print(f"\n" + "="*50)
        print("FOLD-BY-FOLD SUMMARY TABLE")
        print("="*50)
        
        # Create a summary table
        import pandas as pd
        
        summary_data = []
        for fold_name, metrics in fold_detailed_metrics.items():
            row = {
                "Fold": fold_name,
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "Balanced Acc": f"{metrics['balanced_accuracy']:.4f}",
                "Macro F1": f"{metrics['macro_f1']:.4f}",
                "Macro Prec": f"{metrics['macro_precision']:.4f}",
                "Macro Rec": f"{metrics['macro_recall']:.4f}",
                "Pos F1": f"{metrics['f1_positive']:.4f}",
                "Neg F1": f"{metrics['f1_negative']:.4f}",
                "Samples": metrics['num_samples']
            }
            if metrics['roc_auc'] is not None:
                row["ROC AUC"] = f"{metrics['roc_auc']:.4f}"
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Calculate statistics across folds
        accuracies = [metrics['accuracy'] for metrics in fold_detailed_metrics.values()]
        balanced_accuracies = [metrics['balanced_accuracy'] for metrics in fold_detailed_metrics.values()]
        macro_f1s = [metrics['macro_f1'] for metrics in fold_detailed_metrics.values()]
        macro_precisions = [metrics['macro_precision'] for metrics in fold_detailed_metrics.values()]
        macro_recalls = [metrics['macro_recall'] for metrics in fold_detailed_metrics.values()]
        roc_aucs = [metrics['roc_auc'] for metrics in fold_detailed_metrics.values() if metrics['roc_auc'] is not None]
        
        print(f"\n" + "="*50)
        print("CROSS-FOLD STATISTICS")
        print("="*50)
        print(f"Accuracy - Mean: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Balanced Accuracy - Mean: {np.mean(balanced_accuracies):.4f} ± {np.std(balanced_accuracies):.4f}")
        print(f"Macro F1 - Mean: {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
        print(f"Macro Precision - Mean: {np.mean(macro_precisions):.4f} ± {np.std(macro_precisions):.4f}")
        print(f"Macro Recall - Mean: {np.mean(macro_recalls):.4f} ± {np.std(macro_recalls):.4f}")
        if roc_aucs:
            print(f"ROC AUC - Mean: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
        
        # Save detailed fold metrics
        detailed_fold_metrics_file = f"{output_dir}/{output_prefix}_detailed_fold_metrics.pkl"
        with open(detailed_fold_metrics_file, "wb") as f:
            pickle.dump(fold_detailed_metrics, f)
        print(f"\nDetailed fold metrics saved to: {detailed_fold_metrics_file}")
        
        return fold_detailed_metrics
    else:
        print("\nNo individual fold predictions found. This might be because:")
        print("1. predict_eval=False was used in the validate() call")
        print("2. Predictions were not saved for individual folds")
        print("3. The cross-validation hasn't completed yet")
        return None

# Helper function from Geneformer for softmax calculation
def py_softmax(vector):
    e = np.exp(vector)
    return e / e.sum()

print(f"\nDetailed metrics saved to: {metrics_output_file}")

# Try to analyze individual fold predictions
try:
    fold_metrics = analyze_fold_predictions(output_dir, output_prefix, num_folds=cc.num_crossval_splits)
except Exception as e:
    print(f"\nError analyzing individual fold predictions: {e}")
    print("Continuing with aggregated metrics only...")