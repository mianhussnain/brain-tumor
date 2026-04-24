"""
Model architecture and training
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from tensorflow.keras.optimizers import Adam
import config
import pickle


class BrainTumorModel:
    """Build and train brain tumor detection model using transfer learning"""
    
    def __init__(self):
        self.model = None
        self.history = None
    
    def build_model(self, input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS)):
        """
        Build model using EfficientNetB4 with transfer learning
        EfficientNetB4 provides best accuracy for medical imaging tasks
        """
        print("\n" + "="*60)
        print("BUILDING MODEL - EfficientNetB4 (Transfer Learning)")
        print("="*60)
        
        try:
            # Load pre-trained EfficientNetB4 model
            print("Loading EfficientNetB4 from ImageNet...")
            base_model = EfficientNetB4(
                input_shape=input_shape,
                include_top=False,
                weights='imagenet'
            )
            print("✓ EfficientNetB4 loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load with 'weights' parameter: {e}")
            print("Attempting alternative loading method...")
            base_model = EfficientNetB4(
                input_shape=input_shape,
                include_top=False
            )
        
        # Freeze initial layers for better transfer learning
        base_model.trainable = False
        print("✓ Initial layers frozen for transfer learning")
        
        try:
            # Build complete model
            print("Building complete model architecture...")
            model = models.Sequential([
                layers.Input(shape=input_shape),
                
                # Base model (EfficientNetB4)
                # EfficientNet has built-in rescaling, so no Rescaling layer is needed here
                base_model,
                
                # Global Average Pooling
                layers.GlobalAveragePooling2D(),
                
                # Dense layers with dropout
                layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                
                layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                
                # Output layer
                layers.Dense(config.NUM_CLASSES, activation='softmax')
            ])
            
            self.model = model
            print(f"✓ Model built successfully!")
            print(f"✓ Total parameters: {model.count_params():,}")
            
            return model
        except Exception as e:
            print(f"✗ Error building model: {e}")
            print("Error details:", str(e))
            raise
    
    def compile_model(self):
        """Compile the model with optimal settings"""
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        
        try:
            optimizer = Adam(learning_rate=config.LEARNING_RATE)
            
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=[
                    'accuracy',
                    keras.metrics.Precision(),
                    keras.metrics.Recall(),
                    keras.metrics.AUC()
                ]
            )
            print("✓ Model compiled successfully!")
        except Exception as e:
            print(f"✗ Error compiling model: {e}")
            raise
    
    def get_callbacks(self):
        """Setup training callbacks"""
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=config.REDUCE_LR_FACTOR,
                patience=config.REDUCE_LR_PATIENCE,
                min_lr=config.MIN_LR,
                verbose=1
            ),
            
            # Save best model
            ModelCheckpoint(
                filepath=str(config.BEST_MODEL_PATH),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=str(config.LOGS_DIR),
                histogram_freq=1,
                write_graph=True
            )
        ]
        return callbacks
    
    def train(self, train_generator, val_generator, steps_per_epoch, validation_steps):
        """Train the model"""
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        
        print("\n" + "="*60)
        print("STARTING MODEL TRAINING")
        print("="*60)
        
        callbacks = self.get_callbacks()
        
        try:
            self.history = self.model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=config.EPOCHS,
                validation_data=val_generator,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )
            
            print("\n" + "="*60)
            print("TRAINING COMPLETED")
            print("="*60)
        except Exception as e:
            print(f"✗ Error during training: {e}")
            raise
    
    def unfreeze_and_finetune(self, num_layers=30):
        """Unfreeze base model layers for fine-tuning"""
        print("\n" + "="*60)
        print("UNFREEZING BASE MODEL FOR FINE-TUNING")
        print("="*60)
        
        # Unfreeze some of the base model layers
        base_model = self.model.layers[0]  # EfficientNetB4 is now at index 0
        base_model.trainable = True
        
        # Freeze all layers except the last `num_layers`
        for layer in base_model.layers[:-num_layers]:
            layer.trainable = False
            
        # CRITICAL: Keep BatchNormalization layers frozen during fine-tuning to prevent weight destruction
        for layer in base_model.layers[-num_layers:]:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
        
        # Compile with lower learning rate for fine-tuning
        fine_tune_lr = config.LEARNING_RATE / 10
        optimizer = Adam(learning_rate=fine_tune_lr)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(),
                keras.metrics.Recall(),
                keras.metrics.AUC()
            ]
        )
        print(f"Model recompiled with learning rate: {fine_tune_lr}")
    
    def save_model(self):
        """Save the trained model"""
        self.model.save(str(config.FINAL_MODEL_PATH))
        print(f"\nModel saved to: {config.FINAL_MODEL_PATH}")
        
        # Save training history
        with open(config.HISTORY_PATH, 'wb') as f:
            pickle.dump(self.history.history, f)
        print(f"Training history saved to: {config.HISTORY_PATH}")
    
    def load_model(self):
        """Load a saved model"""
        self.model = keras.models.load_model(str(config.FINAL_MODEL_PATH))
        print(f"Model loaded from: {config.FINAL_MODEL_PATH}")
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        from tensorflow.keras.utils import to_categorical
        
        y_test_cat = to_categorical(y_test, config.NUM_CLASSES)
        
        print("\n" + "="*60)
        print("EVALUATING MODEL ON TEST SET")
        print("="*60)
        
        results = self.model.evaluate(X_test, y_test_cat, verbose=0)
        
        metric_names = ['Loss', 'Accuracy', 'Precision', 'Recall', 'AUC']
        print("\nTest Set Results:")
        for name, value in zip(metric_names, results):
            print(f"  {name}: {value:.4f}")
        
        return dict(zip(metric_names, results))
