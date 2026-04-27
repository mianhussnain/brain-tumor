"""
Main training script
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import config
from data_loader import DataLoader
from model import BrainTumorModel


def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history['loss'], label='Train Loss', linewidth=2)
    axes[0, 1].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Find exact metric keys (Keras sometimes appends _1, _2 to metric names)
    prec_key = next((k for k in history.keys() if k.startswith('precision') and not k.startswith('val_')), 'precision')
    val_prec_key = next((k for k in history.keys() if k.startswith('val_precision')), 'val_precision')
    rec_key = next((k for k in history.keys() if k.startswith('recall') and not k.startswith('val_')), 'recall')
    val_rec_key = next((k for k in history.keys() if k.startswith('val_recall')), 'val_recall')

    # Precision
    axes[1, 0].plot(history[prec_key], label='Train Precision', linewidth=2)
    axes[1, 0].plot(history[val_prec_key], label='Val Precision', linewidth=2)
    axes[1, 0].set_title('Model Precision', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(history[rec_key], label='Train Recall', linewidth=2)
    axes[1, 1].plot(history[val_rec_key], label='Val Recall', linewidth=2)
    axes[1, 1].set_title('Model Recall', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(config.MODELS_DIR / 'training_history.png'), dpi=300, bbox_inches='tight')
    print("Training history plot saved!")
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
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
    plt.savefig(str(config.MODELS_DIR / 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print("Confusion matrix plot saved!")
    plt.close()


def main():
    """Main training function"""
    
    print("\n" + "="*60)
    print("BRAIN TUMOR DETECTION SYSTEM - TRAINING")
    print("="*60)
    
    try:
        # Load data
        print("\n[1/5] Loading data...")
        data_loader = DataLoader()
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.get_train_test_split()
        
        # Build model
        print("\n[2/5] Building model...")
        model = BrainTumorModel()
        model.build_model()
        
        if model.model is None:
            raise RuntimeError("Model building failed - model is None")
        
        print("✓ Model built successfully")
        
        model.compile_model()
        
        if model.model is None:
            raise RuntimeError("Model compilation failed - model is None")
        
        print("✓ Model compiled successfully")
        
        # Get data generators
        print("\n[3/5] Preparing data generators...")
        train_generator, val_generator = data_loader.get_data_generators(
            X_train, y_train, X_val, y_val
        )
        
        # Calculate steps
        steps_per_epoch = len(X_train) // config.BATCH_SIZE
        validation_steps = len(X_val) // config.BATCH_SIZE
        
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        
        # Train initial model
        print("\n[4/5] Training initial model...")
        model.train(train_generator, val_generator, steps_per_epoch, validation_steps)
        
        # Fine-tune: Unfreeze and train more layers
        print("\n" + "="*60)
        print("FINE-TUNING PHASE")
        print("="*60)
        
        model.unfreeze_and_finetune(num_layers=30)
        
        # Train with unfrozen layers (reduced epochs)
        model.history = model.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=20,  # Fine-tuning with fewer epochs
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=model.get_callbacks(),
            verbose=1
        )
        
        # Save model
        print("\n[5/5] Saving model...")
        model.save_model()
        
        # Evaluate on test set
        test_results = model.evaluate(X_test, y_test)
        
        # Make predictions
        y_pred_proba = model.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Detailed evaluation
        print("\n" + "="*60)
        print("DETAILED EVALUATION")
        print("="*60)
        
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=config.CLASSES,
            digits=4
        ))
        
        # Per-class metrics
        print("\nPer-Class Metrics:")
        for idx, class_name in enumerate(config.CLASSES):
            mask = y_test == idx
            if mask.sum() > 0:
                class_acc = accuracy_score(y_test[mask], y_pred[mask])
                print(f"  {class_name}: {class_acc:.4f}")
        
        # Plot results
        print("\nGenerating plots...")
        if model.history:
            plot_training_history(model.history.history)
        
        plot_confusion_matrix(y_test, y_pred)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Model saved to: {config.FINAL_MODEL_PATH}")
        print(f"Best model saved to: {config.BEST_MODEL_PATH}")
        print(f"Logs and plots saved to: {config.MODELS_DIR}")
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
