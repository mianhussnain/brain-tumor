"""
Evaluation script for testing model performance
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
import config
from data_loader import DataLoader
from model import BrainTumorModel
import seaborn as sns


def evaluate_model():
    """Comprehensive model evaluation"""
    
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    data_loader = DataLoader()
    (_, _), (_, _), (X_test, y_test) = data_loader.get_train_test_split()
    
    # Load model
    print("Loading model...")
    model = BrainTumorModel()
    model.load_model()
    
    # Make predictions
    print("Making predictions...")
    y_pred_proba = model.model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    print("\n" + "="*60)
    print("OVERALL METRICS")
    print("="*60)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Per-class metrics
    print("\n" + "="*60)
    print("PER-CLASS METRICS")
    print("="*60)
    
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=config.CLASSES,
        digits=4
    ))
    
    # Confusion matrix
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    
    cm = confusion_matrix(y_test, y_pred)
    print("\n" + str(cm))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=config.CLASSES,
        yticklabels=config.CLASSES,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(str(config.MODELS_DIR / 'evaluation_confusion_matrix.png'), dpi=300)
    print("\nConfusion matrix saved to: models/evaluation_confusion_matrix.png")
    plt.close()
    
    # ROC curves
    print("\n" + "="*60)
    print("ROC-AUC ANALYSIS")
    print("="*60)
    
    from tensorflow.keras.utils import to_categorical
    y_test_cat = to_categorical(y_test, config.NUM_CLASSES)
    
    plt.figure(figsize=(12, 8))
    
    for i in range(config.NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_test_cat[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{config.CLASSES[i]} (AUC = {roc_auc:.3f})')
        print(f"{config.CLASSES[i]}: AUC = {roc_auc:.4f}")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Test Set', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(config.MODELS_DIR / 'roc_curves.png'), dpi=300)
    print("\nROC curves saved to: models/roc_curves.png")
    plt.close()
    
    # Confidence analysis
    print("\n" + "="*60)
    print("CONFIDENCE ANALYSIS")
    print("="*60)
    
    max_confidences = np.max(y_pred_proba, axis=1)
    
    print(f"\nConfidence Statistics:")
    print(f"  Mean:   {np.mean(max_confidences):.4f}")
    print(f"  Median: {np.median(max_confidences):.4f}")
    print(f"  Min:    {np.min(max_confidences):.4f}")
    print(f"  Max:    {np.max(max_confidences):.4f}")
    print(f"  Std:    {np.std(max_confidences):.4f}")
    
    # Count predictions by confidence threshold
    print(f"\nPredictions by confidence threshold:")
    thresholds = [0.6, 0.75, 0.9]
    for thresh in thresholds:
        count = np.sum(max_confidences >= thresh)
        pct = 100 * count / len(max_confidences)
        print(f"  >= {thresh}: {count}/{len(max_confidences)} ({pct:.1f}%)")
    
    # Accuracy by confidence
    plt.figure(figsize=(12, 5))
    
    # Histogram of confidences
    plt.subplot(1, 2, 1)
    plt.hist(max_confidences, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(max_confidences), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(max_confidences):.3f}')
    plt.xlabel('Confidence Score', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Distribution of Prediction Confidence', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Accuracy vs confidence
    plt.subplot(1, 2, 2)
    sorted_indices = np.argsort(max_confidences)
    cumulative_correct = np.cumsum(y_pred[sorted_indices] == y_test[sorted_indices])
    cumulative_samples = np.arange(1, len(sorted_indices) + 1)
    cumulative_accuracy = cumulative_correct / cumulative_samples
    
    plt.plot(max_confidences[sorted_indices], cumulative_accuracy, linewidth=2)
    plt.xlabel('Prediction Confidence', fontsize=11)
    plt.ylabel('Cumulative Accuracy', fontsize=11)
    plt.title('Accuracy vs Confidence (Sorted)', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(str(config.MODELS_DIR / 'confidence_analysis.png'), dpi=300)
    print("\nConfidence analysis saved to: models/confidence_analysis.png")
    plt.close()
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nAll results saved to: {config.MODELS_DIR}")


if __name__ == "__main__":
    evaluate_model()
