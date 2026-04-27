import pickle
import numpy as np
import config
from train import plot_training_history, plot_confusion_matrix
from model import BrainTumorModel
from data_loader import DataLoader

def main():
    print("Loading training history...")
    try:
        with open(config.HISTORY_PATH, 'rb') as f:
            history = pickle.load(f)
        
        print("Generating training history plot...")
        plot_training_history(history)
    except FileNotFoundError:
        print(f"Error: Could not find {config.HISTORY_PATH}. Ensure training saved the history file.")
    except Exception as e:
        print(f"Error generating history plot: {e}")
    
    print("\nLoading test data for confusion matrix...")
    data_loader = DataLoader()
    _, _, (X_test, y_test) = data_loader.get_train_test_split()
    
    print("\nLoading saved model...")
    model = BrainTumorModel()
    try:
        model.load_model()
        
        print("\nGenerating predictions...")
        y_pred_proba = model.model.predict(X_test, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        print("\nGenerating confusion matrix plot...")
        plot_confusion_matrix(y_test, y_pred)
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
    
    print("\nScript completed. Check the 'models' folder for the new PNG files.")

if __name__ == "__main__":
    main()
