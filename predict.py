"""
Prediction and inference module
"""

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import config


class BrainTumorPredictor:
    """Make predictions on new images"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.class_names = config.CLASSES
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        
        if model_path is None:
            model_path = config.FINAL_MODEL_PATH
        
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            self.model = keras.models.load_model(str(model_path))
            print(f"Model loaded successfully from: {model_path}")
        except FileNotFoundError:
            print(f"Error: Model not found at {model_path}")
            raise
    
    def preprocess_image(self, image_path_or_array):
        """Preprocess image for prediction"""
        if isinstance(image_path_or_array, str):
            # Load from file path
            img = Image.open(image_path_or_array).convert('RGB')
        else:
            # Convert numpy array to PIL Image if needed
            if isinstance(image_path_or_array, np.ndarray):
                img = Image.fromarray((image_path_or_array * 255).astype(np.uint8))
            else:
                img = image_path_or_array
        
        # Resize to model input size
        img = img.resize((config.IMG_WIDTH, config.IMG_HEIGHT))
        
        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
    
    def predict(self, image_path_or_array):
        """
        Make prediction on an image
        
        Args:
            image_path_or_array: Path to image file or numpy array
        
        Returns:
            Dictionary with predictions and confidence scores
        """
        # Preprocess
        img_array, original_img = self.preprocess_image(image_path_or_array)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get top prediction
        predicted_class_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[predicted_class_idx]
        
        # Create result dictionary
        result = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'is_confident': confidence >= self.confidence_threshold,
            'all_predictions': {
                self.class_names[i]: float(predictions[i])
                for i in range(len(self.class_names))
            },
            'sorted_predictions': sorted(
                [(self.class_names[i], float(predictions[i])) for i in range(len(self.class_names))],
                key=lambda x: x[1],
                reverse=True
            )
        }
        
        return result
    
    def predict_batch(self, image_paths_or_arrays):
        """Make predictions on multiple images"""
        results = []
        for image in image_paths_or_arrays:
            result = self.predict(image)
            results.append(result)
        return results
    
    def get_prediction_confidence_level(self, confidence):
        """Get confidence level description"""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.75:
            return "High"
        elif confidence >= 0.6:
            return "Moderate"
        else:
            return "Low"


def format_prediction_result(result):
    """Format prediction result for display"""
    output = []
    output.append("="*60)
    output.append("PREDICTION RESULT")
    output.append("="*60)
    output.append(f"\nPredicted Class: {result['predicted_class'].upper()}")
    output.append(f"Confidence: {result['confidence']:.2%}")
    output.append(f"Confidence Level: {get_confidence_level(result['confidence'])}")
    
    if not result['is_confident']:
        output.append("⚠️  Warning: Low confidence - result should be verified by medical professional")
    
    output.append("\n" + "-"*60)
    output.append("Detailed Predictions (All Classes):")
    output.append("-"*60)
    
    for class_name, pred_conf in result['sorted_predictions']:
        bar_length = int(pred_conf * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        output.append(f"{class_name:12} | {bar} | {pred_conf:.2%}")
    
    return "\n".join(output)


def get_confidence_level(confidence):
    """Get confidence level description"""
    if confidence >= 0.9:
        return "Very High"
    elif confidence >= 0.75:
        return "High"
    elif confidence >= 0.6:
        return "Moderate"
    else:
        return "Low"
