#!/usr/bin/env python
"""
Main entry point and launcher for Brain Tumor Detection System
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


def print_header():
    """Print application header"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║   🧠 BRAIN TUMOR DETECTION SYSTEM                       ║
    ║                                                          ║
    ║   AI-Powered Medical Imaging Analysis v1.0             ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Brain Tumor Detection System',
        epilog='For detailed help: python main.py --help'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    
    # Web app command
    subparsers.add_parser('app', help='Launch web application')
    subparsers.add_parser('web', help='Launch web application (alias)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('image', help='Path to image file')
    
    # Evaluate command
    subparsers.add_parser('evaluate', help='Evaluate model on test set')
    subparsers.add_parser('eval', help='Evaluate model on test set (alias)')
    
    # Check command
    subparsers.add_parser('check', help='Check environment setup')
    
    # Info command
    subparsers.add_parser('info', help='Show system information')
    
    # Guide command
    subparsers.add_parser('guide', help='Show usage guide')
    
    args = parser.parse_args()
    
    print_header()
    
    if not args.command:
        print_help_menu()
        return
    
    # Route commands
    if args.command in ['train']:
        run_train(args)
    elif args.command in ['app', 'web']:
        run_app()
    elif args.command in ['predict']:
        run_predict(args.image)
    elif args.command in ['evaluate', 'eval']:
        run_evaluate()
    elif args.command == 'check':
        run_check()
    elif args.command == 'info':
        run_info()
    elif args.command == 'guide':
        run_guide()


def print_help_menu():
    """Print help menu"""
    menu = """
    Available Commands:
    
    Training & Evaluation:
      python main.py train       Train the model
      python main.py evaluate    Evaluate on test set
      python main.py check       Verify environment setup
    
    Prediction & Web:
      python main.py app         Launch web application
      python main.py predict <image>   Predict on single image
    
    Information:
      python main.py info        Show system information
      python main.py guide       Show usage guide
      python main.py --help      Show detailed help
    
    Quick Start:
      1. Setup: python main.py check
      2. Train: python main.py train
      3. Web:   python main.py app
    
    For more information, see README.md or run: python main.py guide
    """
    print(menu)


def run_train(args):
    """Run training script"""
    print("\nStarting model training...")
    print("-" * 60)
    
    try:
        subprocess.run(['python', 'train.py'], check=True)
        print("\n✓ Training completed successfully!")
    except subprocess.CalledProcessError:
        print("\n✗ Training failed. Check error messages above.")
        sys.exit(1)


def run_app():
    """Run web application"""
    print("\nLaunching web application...")
    print("-" * 60)
    print("\nStreamlit app starting at: http://localhost:8501")
    print("Press Ctrl+C to stop the server.\n")
    
    try:
        subprocess.run(['streamlit', 'run', 'app.py'], check=True)
    except FileNotFoundError:
        print("✗ Streamlit not found. Install with: pip install streamlit")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nApplication stopped.")


def run_predict(image_path):
    """Run prediction on image"""
    print(f"\nMaking prediction on: {image_path}")
    print("-" * 60)
    
    from predict import BrainTumorPredictor, format_prediction_result
    
    if not Path(image_path).exists():
        print(f"✗ File not found: {image_path}")
        sys.exit(1)
    
    try:
        predictor = BrainTumorPredictor()
        result = predictor.predict(image_path)
        print(format_prediction_result(result))
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        sys.exit(1)


def run_evaluate():
    """Run evaluation script"""
    print("\nEvaluating model on test set...")
    print("-" * 60)
    
    try:
        subprocess.run(['python', 'evaluate.py'], check=True)
        print("\n✓ Evaluation completed successfully!")
    except subprocess.CalledProcessError:
        print("\n✗ Evaluation failed. Check error messages above.")
        sys.exit(1)


def run_check():
    """Run environment check"""
    print("Checking environment...")
    subprocess.run(['python', 'quick_start.py', '--check'])


def run_info():
    """Show system information"""
    print("System information...")
    subprocess.run(['python', 'quick_start.py', '--info'])


def run_guide():
    """Show usage guide"""
    subprocess.run(['python', 'quick_start.py', '--guide'])


if __name__ == "__main__":
    main()
