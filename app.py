"""
Streamlit Web Application for Brain Tumor Detection
Professional and user-friendly interface
"""

import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import config
from predict import BrainTumorPredictor, get_confidence_level
import os


# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    h2 {
        color: #2c3e50;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .metric-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-moderate {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


# Initialize session state
@st.cache_resource
def load_model():
    """Load the trained model once"""
    model_path = config.FINAL_MODEL_PATH
    if not os.path.exists(model_path):
        return None
    try:
        predictor = BrainTumorPredictor(model_path)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def create_prediction_chart(predictions):
    """Create a plotly chart for predictions"""
    classes = list(predictions.keys())
    confidences = [predictions[cls] * 100 for cls in classes]
    
    colors = ['#d62728' if conf == max(confidences) else '#1f77b4' for conf in confidences]
    
    fig = go.Figure(data=[
        go.Bar(
            y=classes,
            x=confidences,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{conf:.1f}%' for conf in confidences],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence Scores",
        xaxis_title="Confidence (%)",
        yaxis_title="Brain Tumor Type",
        height=400,
        margin=dict(l=150),
        showlegend=False,
        template="plotly_white"
    )
    
    return fig


def display_diagnosis_section(result):
    """Display diagnosis information"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Predicted Class",
            value=result['predicted_class'].upper(),
            delta=None
        )
    
    with col2:
        confidence = result['confidence']
        confidence_level = get_confidence_level(confidence)
        st.metric(
            label="Confidence",
            value=f"{confidence:.1%}",
            delta=confidence_level
        )
    
    with col3:
        status = "✅ High Confidence" if result['is_confident'] else "⚠️ Low Confidence"
        st.metric(
            label="Status",
            value="Valid" if result['is_confident'] else "Review",
            delta=status
        )
    
    # Detailed predictions chart
    st.plotly_chart(
        create_prediction_chart(result['all_predictions']),
        use_container_width=True
    )
    
    # Confidence warning
    if not result['is_confident']:
        st.warning(
            f"⚠️ **Low Confidence Warning**: "
            f"The model confidence ({result['confidence']:.1%}) is below the threshold. "
            f"This prediction should be reviewed by a medical professional.",
            icon="⚠️"
        )


def main():
    # Header
    st.title("🧠 Brain Tumor Detection System")
    st.markdown("### AI-Powered Medical Imaging Analysis")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 System Information")
        st.markdown("""
        **Model Details:**
        - Architecture: EfficientNetB4 (Transfer Learning)
        - Input Size: 224×224 pixels
        - Classes: 4 tumor types
        
        **Supported Classes:**
        - Glioma
        - Meningioma
        - No Tumor
        - Pituitary
        
        **Important:**
        ⚠️ This system is for **assistance only** and should not replace professional medical diagnosis.
        """)
        
        st.divider()
        
        st.markdown("""
        **Confidence Threshold:**
        - High: ≥ 90%
        - Moderate: 60-75%
        - Low: < 60%
        """)
    
    # Load model
    predictor = load_model()
    
    if predictor is None:
        st.error(
            "❌ Model not found. Please run the training script first:\n"
            "```bash\npython train.py\n```"
        )
        return
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis", "ℹ️ About"])
    
    # Tab 1: Single Prediction
    with tab1:
        st.header("Single Image Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader(
                "Choose a brain MRI image",
                type=["jpg", "jpeg", "png", "bmp"]
            )
        
        with col2:
            st.subheader("Test with Sample")
            use_sample = st.checkbox("Use sample image for testing")
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.info("Processing image...", icon="⏳")
                
                # Make prediction
                result = predictor.predict(np.array(image))
                
                st.success("Prediction complete!", icon="✅")
            
            # Display results
            st.divider()
            display_diagnosis_section(result)
            
            # Additional information
            st.divider()
            st.subheader("📝 Detailed Analysis")
            
            with st.expander("View All Predictions"):
                for class_name, conf in result['sorted_predictions']:
                    st.write(f"**{class_name}**: {conf:.2%}")
    
    # Tab 2: Batch Analysis
    with tab2:
        st.header("Batch Image Analysis")
        
        uploaded_files = st.file_uploader(
            "Upload multiple brain MRI images",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.info(f"📸 {len(uploaded_files)} images uploaded")
            
            if st.button("Process All Images", type="primary"):
                progress_bar = st.progress(0)
                results_list = []
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    # Update progress
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    
                    # Make prediction
                    image = Image.open(uploaded_file)
                    result = predictor.predict(np.array(image))
                    
                    results_list.append({
                        'filename': uploaded_file.name,
                        'result': result
                    })
                
                # Display results
                st.success("✅ All images processed!", icon="✅")
                st.divider()
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                predictions = [r['result']['predicted_class'] for r in results_list]
                avg_confidence = np.mean([r['result']['confidence'] for r in results_list])
                
                with col1:
                    st.metric("Total Images", len(results_list))
                with col2:
                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                with col3:
                    st.metric("High Confidence", sum(1 for r in results_list if r['result']['is_confident']))
                with col4:
                    st.metric("Most Common", max(set(predictions), key=predictions.count))
                
                # Detailed results
                st.subheader("Individual Results")
                
                for item in results_list:
                    with st.expander(f"📄 {item['filename']}"):
                        result = item['result']
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.write(f"**Predicted**: {result['predicted_class'].upper()}")
                            st.write(f"**Confidence**: {result['confidence']:.2%}")
                            st.write(f"**Status**: {'✅ High' if result['is_confident'] else '⚠️ Low'}")
                        
                        with col2:
                            st.plotly_chart(
                                create_prediction_chart(result['all_predictions']),
                                use_container_width=True
                            )
    
    # Tab 3: About
    with tab3:
        st.header("About This System")
        
        st.markdown("""
        ## 🔬 Model Architecture
        
        This system uses **EfficientNetB4**, a state-of-the-art deep learning model optimized for medical imaging:
        
        - **Base Model**: EfficientNetB4 (pre-trained on ImageNet)
        - **Training Method**: Transfer Learning with Fine-tuning
        - **Input Resolution**: 224×224 RGB images
        - **Output**: 4-class classification (Glioma, Meningioma, No Tumor, Pituitary)
        
        ### Why EfficientNetB4?
        - Exceptional accuracy for medical imaging tasks
        - Optimized for computational efficiency
        - Better generalization than standard CNNs
        - Proven performance on brain tumor datasets
        
        ## 📊 Dataset Information
        
        The model was trained on a comprehensive dataset of brain MRI images:
        - **Classes**: 4 tumor types
        - **Data Augmentation**: Rotation, zoom, flip, brightness adjustments
        - **Validation Strategy**: 80-20 train-validation split
        - **Test Set**: Independent test set for unbiased evaluation
        
        ## ⚕️ Important Disclaimers
        
        ⚠️ **CRITICAL NOTICE**:
        - This system is designed for **research and assistance purposes only**
        - It should **NOT** be used as a sole diagnostic tool
        - Always consult with qualified medical professionals for diagnosis
        - Results should be verified by radiologists and physicians
        - The system may have false positives or false negatives
        
        ## 🎯 Performance Metrics
        
        - Accuracy: High (check detailed report)
        - Precision & Recall: Optimized for all classes
        - AUC-ROC: Strong discrimination ability
        
        ## 📧 Support
        
        For issues or questions, contact the development team.
        """)
        
        st.divider()
        
        st.markdown("""
        <div style="text-align: center; color: gray; margin-top: 2rem;">
        <p>Brain Tumor Detection System v1.0</p>
        <p>Developed with advanced deep learning techniques</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
