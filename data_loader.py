"""
Data loading, preprocessing, and augmentation
"""

import os
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
import config


class DataLoader:
    """Handles data loading, preprocessing, and augmentation"""
    
    def __init__(self):
        self.img_width = config.IMG_WIDTH
        self.img_height = config.IMG_HEIGHT
        self.batch_size = config.BATCH_SIZE
        self.classes = config.CLASSES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
    
    def load_images_from_directory(self, directory_path, label):
        """Load all images from a directory and assign a label"""
        images = []
        labels = []
        
        image_dir = Path(directory_path) / label
        if not image_dir.exists():
            print(f"Warning: Directory {image_dir} does not exist")
            return images, labels
        
        for image_file in image_dir.glob("*.jpg"):
            try:
                # Load and preprocess image
                img = load_img(str(image_file), target_size=(self.img_width, self.img_height))
                img_array = img_to_array(img)  # EfficientNet expects [0, 255]
                
                images.append(img_array)
                labels.append(self.class_to_idx[label])
            except Exception as e:
                print(f"Error loading image {image_file}: {e}")
                continue
        
        return images, labels
    
    def load_all_data(self, data_dir):
        """Load all images from all classes"""
        all_images = []
        all_labels = []
        
        print("Loading images...")
        for class_name in self.classes:
            images, labels = self.load_images_from_directory(data_dir, class_name)
            all_images.extend(images)
            all_labels.extend(labels)
            print(f"  Loaded {len(images)} images for class: {class_name}")
        
        return np.array(all_images, dtype=np.float32), np.array(all_labels, dtype=np.int32)
    
    def get_train_test_split(self):
        """Load and split data into train and test sets"""
        print("\n" + "="*60)
        print("LOADING TRAINING DATA")
        print("="*60)
        train_images, train_labels = self.load_all_data(config.TRAIN_DIR)
        
        print("\n" + "="*60)
        print("LOADING TESTING DATA")
        print("="*60)
        test_images, test_labels = self.load_all_data(config.TEST_DIR)
        
        print("\n" + "="*60)
        print("DATA LOADING SUMMARY")
        print("="*60)
        print(f"Training images shape: {train_images.shape}")
        print(f"Training labels shape: {train_labels.shape}")
        print(f"Testing images shape: {test_images.shape}")
        print(f"Testing labels shape: {test_labels.shape}")
        
        # Print class distribution
        print("\nClass distribution in training data:")
        for idx, class_name in enumerate(self.classes):
            count = np.sum(train_labels == idx)
            print(f"  {class_name}: {count} samples")
        
        print("\nClass distribution in testing data:")
        for idx, class_name in enumerate(self.classes):
            count = np.sum(test_labels == idx)
            print(f"  {class_name}: {count} samples")
        
        # Split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            train_images, train_labels,
            test_size=config.VALIDATION_SPLIT,
            random_state=config.RANDOM_STATE,
            stratify=train_labels
        )
        
        print(f"\nTrain/Val split:")
        print(f"  Training: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Test: {test_images.shape[0]} samples")
        
        return (X_train, y_train), (X_val, y_val), (test_images, test_labels)
    
    def get_data_augmentation(self):
        """Get ImageDataGenerator with augmentation parameters"""
        train_augmentation = ImageDataGenerator(
            rotation_range=config.AUGMENTATION["rotation_range"],
            width_shift_range=config.AUGMENTATION["width_shift_range"],
            height_shift_range=config.AUGMENTATION["height_shift_range"],
            horizontal_flip=config.AUGMENTATION["horizontal_flip"],
            vertical_flip=config.AUGMENTATION["vertical_flip"],
            zoom_range=config.AUGMENTATION["zoom_range"],
            brightness_range=config.AUGMENTATION["brightness_range"],
            fill_mode=config.AUGMENTATION["fill_mode"]
        )
        
        # No augmentation for validation data, just rescaling
        val_augmentation = ImageDataGenerator()
        
        return train_augmentation, val_augmentation
    
    def get_data_generators(self, X_train, y_train, X_val, y_val):
        """Get training and validation data generators"""
        from tensorflow.keras.utils import to_categorical
        
        train_augmentation, val_augmentation = self.get_data_augmentation()
        
        # Convert labels to one-hot encoding
        y_train_cat = to_categorical(y_train, config.NUM_CLASSES)
        y_val_cat = to_categorical(y_val, config.NUM_CLASSES)
        
        train_generator = train_augmentation.flow(
            X_train, y_train_cat,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_generator = val_augmentation.flow(
            X_val, y_val_cat,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_generator, val_generator
