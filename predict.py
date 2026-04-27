"""
Prediction and inference module
"""

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import config


class BrainTumorPredictor:
    """Make predictions on new images.

    Parameters
    ----------
    model_path : Path or str
        Path to the Keras model file.
    mode : str
        '4class' (default) — 4-class softmax classifier.
        'binary'           — binary sigmoid (Tumor / No Tumor).
    """

    def __init__(self, model_path=None, mode="4class"):
        self.model = None
        self.mode  = mode

        if mode == "binary":
            self.class_names = config.BINARY_CLASSES          # ["No Tumor", "Tumor"]
            self.confidence_threshold = config.CONFIDENCE_THRESHOLD
            if model_path is None:
                model_path = config.BINARY_MODEL_PATH
        else:
            self.class_names = config.CLASSES                 # 4-class list
            self.confidence_threshold = config.CONFIDENCE_THRESHOLD
            if model_path is None:
                model_path = config.FINAL_MODEL_PATH

        self.load_model(model_path)

    def load_model(self, model_path):
        """Load the trained model"""
        try:
            self.model = keras.models.load_model(str(model_path), compile=False)
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
                if image_path_or_array.dtype == np.uint8 or image_path_or_array.max() > 1.0:
                    img = Image.fromarray(image_path_or_array.astype(np.uint8)).convert('RGB')
                else:
                    img = Image.fromarray((image_path_or_array * 255).astype(np.uint8)).convert('RGB')
            else:
                img = image_path_or_array.convert('RGB')
        
        # Resize to model input size
        img = img.resize((config.IMG_WIDTH, config.IMG_HEIGHT))
        
        # Convert to array (no normalization; EfficientNet expects 0-255)
        img_array = np.array(img, dtype=np.float32)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
    
    def predict(self, image_path_or_array):
        """
        Make prediction on an image.

        Args:
            image_path_or_array: Path to image file or numpy array

        Returns:
            Dictionary with keys:
                predicted_class, confidence, is_confident,
                all_predictions, sorted_predictions
        """
        img_array, _ = self.preprocess_image(image_path_or_array)
        raw = self.model.predict(img_array, verbose=0)[0]

        if self.mode == "binary":
            # raw is a single sigmoid value: P(Tumor)
            p_tumor    = float(raw[0]) if hasattr(raw, "__len__") else float(raw)
            p_no_tumor = 1.0 - p_tumor

            predicted_class = "Tumor" if p_tumor >= 0.5 else "No Tumor"
            confidence      = p_tumor if p_tumor >= 0.5 else p_no_tumor

            all_predictions = {
                "No Tumor": p_no_tumor,
                "Tumor":    p_tumor,
            }
            sorted_predictions = sorted(
                all_predictions.items(), key=lambda x: x[1], reverse=True
            )
        else:
            # 4-class softmax
            predicted_class_idx = int(np.argmax(raw))
            predicted_class     = self.class_names[predicted_class_idx]
            confidence          = float(raw[predicted_class_idx])

            all_predictions = {
                self.class_names[i]: float(raw[i])
                for i in range(len(self.class_names))
            }
            sorted_predictions = sorted(
                all_predictions.items(), key=lambda x: x[1], reverse=True
            )

        return {
            "predicted_class":    predicted_class,
            "confidence":         confidence,
            "is_confident":       confidence >= self.confidence_threshold,
            "all_predictions":    all_predictions,
            "sorted_predictions": sorted_predictions,
        }
    
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
