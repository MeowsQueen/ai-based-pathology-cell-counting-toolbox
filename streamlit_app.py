import streamlit as st
import numpy as np
from PIL import Image
import io

# Import existing toolbox modules
from preprocessing.image_enhancement import preprocess_image
from models.inference import load_trained_model, predict_segmentation, predict_classification
from utils.helpers import get_default_class_names
from analysis.cell_detection import count_cells, calculate_cell_density
from visualization.visualization import plot_cell_counts, plot_cell_density_heatmap

# Cache model loading
@st.cache(allow_output_mutation=True)
def load_models(
    seg_model_path: str = "Cell Detection Toolbox/segmentation_model.keras",
    class_model_path: str = "Cell Detection Toolbox/classification_model.keras"
):
    segmentation_model = load_trained_model(seg_model_path, model_type="segmentation")
    classification_model = load_trained_model(class_model_path, model_type="classification")
    return segmentation_model, classification_model

# Main Streamlit app
def main():
    st.set_page_config(page_title="AI-Based Pathology Cell Counting Toolbox", layout="wide")
    st.title("AI-Based Pathology Cell Counting Toolbox")

    # Sidebar: module selection and file upload
    module = st.sidebar.selectbox(
        "Select Module", 
        [
            "Image Preprocessing & Enhancement", 
            "Cell Detection & Counting", 
            "AI-Powered Quantitative Analysis", 
            "Visualization Tools", 
            "Model Training & Optimization"
        ]
    )
    uploaded_file = st.sidebar.file_uploader("Upload Pathology Image", type=["png", "jpg", "tif", "tiff"])

    # Load models once
    seg_model, class_model = load_models()
    class_names = get_default_class_names()

    if uploaded_file:
        # Read and display original image
        image = Image.open(io.BytesIO(uploaded_file.read()))
        image_np = np.array(image)
        st.image(image_np, caption="Original Image", use_column_width=True)

        # Preprocess image
        processed = preprocess_image(
            image_np, enhance=True, denoise=True, detect_regions=True
        )

        if module == "Image Preprocessing & Enhancement":
            st.header("Preprocessing & Enhancement")
            st.image(processed, caption="Preprocessed Image", use_column_width=True)

        elif module == "Cell Detection & Counting":
            st.header("Cell Detection & Counting")
            # Segmentation & classification
            seg_mask = predict_segmentation(seg_model, processed)
            class_probs = predict_classification(class_model, processed)

            # Count cells by class
            counts = count_cells(seg_mask, class_names=class_names)
            st.subheader("Cell Counts by Class")
            fig_counts = plot_cell_counts(counts, class_names=class_names)
            st.pyplot(fig_counts)
            st.write(counts)

        elif module == "AI-Powered Quantitative Analysis":
            st.header("Quantitative Analysis")
            # Compute counts and density
            seg_mask = predict_segmentation(seg_model, processed)
            counts = count_cells(seg_mask, class_names=class_names)
            density = calculate_cell_density(seg_mask, seg_mask.size)
            st.subheader("Quantitative Metrics")
            st.write({"Cell Density (cells/pixel)": density})

        elif module == "Visualization Tools":
            st.header("Visualization Tools")
            seg_mask = predict_segmentation(seg_model, processed)
            # Select class for heatmap
            selected = st.selectbox("Select Cell Type for Heatmap", class_names)
            class_id = class_names.index(selected)
            fig_heatmap = plot_cell_density_heatmap(image_np, seg_mask, class_id)
            st.pyplot(fig_heatmap)

        elif module == "Model Training & Optimization":
            st.header("Model Training & Optimization")
            st.markdown(
                "- **Data Augmentation:** Enhance dataset with synthetic variations.  \
"
                "- **Self-Learning:** Integrate pathologist feedback loops.  \
"
                "- **Edge AI:** Optimize models for on-device inference."
            )
            st.info("For training pipelines, see `train_model_example.py` and `training` modules.")

    else:
        st.info("Please upload a pathology image to get started.")

if __name__ == "__main__":
    main()
