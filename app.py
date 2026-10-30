"""Streamlit research prototype for medical image classification."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from config import DATA_ROOT, MODEL_ROOT, RESULTS_ROOT
from src.explainability import grad_cam, overlay_heatmap
from src.models import SUPPORTED_ARCHITECTURES
from src.prediction import load_checkpoint, predict_image
from src.preprocessing import open_rgb_image

TASKS = {
    "Thoracic diseases (NIH ChestX-ray14)": ("chest_xray14", "multilabel", DATA_ROOT / "raw" / "chest_xray14"),
    "Diabetic retinopathy (APTOS)": ("aptos_retinopathy", "multiclass", DATA_ROOT / "raw" / "aptos"),
    "Skin lesion classification (ISIC 2016 Task 3)": ("isic_skin_lesion", "multiclass", DATA_ROOT / "raw" / "isic2016"),
}

st.set_page_config(page_title="Medical Image Analysis Research Prototype", page_icon="🩻", layout="wide")
st.markdown("""
<style>
  .block-container {max-width: 1180px; padding-top: 2rem;}
  .disclaimer {padding: 0.9rem 1rem; border-left: 4px solid #b45a43; background: #fff4ef; color: #4c322d; border-radius: 4px;}
  div[data-testid="stMetric"] {background: #f5f7fa; padding: 14px 16px; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

st.title("Deep Learning Based Medical Image Analysis For Early Disease Detection")
st.caption("Academic research prototype | Image classification with transfer learning")
st.markdown(
    '<div class="disclaimer"><strong>Research use only.</strong> Outputs are experimental model estimates, not a diagnosis. '
    'They are not clinically validated and must not replace assessment by a qualified healthcare professional.</div>',
    unsafe_allow_html=True,
)

page = st.sidebar.radio("Navigate", ["Overview", "Image analysis", "Dataset status", "Model results", "Methodology"])
task_label = st.sidebar.selectbox("Research task", list(TASKS))
task, task_type, default_data_root = TASKS[task_label]
architecture = st.sidebar.selectbox("Model architecture", list(SUPPORTED_ARCHITECTURES), index=2)
checkpoint_path = Path(MODEL_ROOT) / task / architecture / "model.pt"
PROJECT_ROOT = Path(__file__).resolve().parent


def display_project_path(path: Path) -> str:
    """Show a portable project-relative path without exposing the host username."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name

if page == "Overview":
    st.header("Project dashboard")
    st.write("The report defines three independent image-analysis use cases. Select a task in the sidebar to inspect its local data and model status.")
    columns = st.columns(3)
    columns[0].metric("Research tasks", "3")
    columns[1].metric("Available trained models", str(sum(
        (Path(MODEL_ROOT) / code / model / "model.pt").is_file()
        for code, _, _ in TASKS.values() for model in SUPPORTED_ARCHITECTURES
    )))
    columns[2].metric("Dataset folders detected", str(sum(path.exists() for _, _, path in TASKS.values())))
    st.subheader("Current task")
    st.write(f"**{task_label}**")
    st.write(f"Local data folder: `{display_project_path(default_data_root)}`")
    st.write(f"Checkpoint: `{display_project_path(checkpoint_path)}`")
    if not checkpoint_path.is_file():
        st.info("No trained checkpoint is present for this task and architecture. The prototype will not show a prediction until a real checkpoint has been trained.")
    st.subheader("Scope")
    st.write("The application supports image upload, model-backed probabilities, optional Grad-CAM visualizations, and locally generated evaluation results when available.")

elif page == "Image analysis":
    st.header("Analyze an image")
    st.write(f"Selected task: **{task_label}**")
    uploaded = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"])
    if uploaded is not None:
        try:
            if uploaded.size > 20 * 1024 * 1024:
                raise ValueError("Image exceeds the 20 MB upload limit")
            image_bytes = uploaded.getvalue()
            image = open_rgb_image(image_bytes)
            st.image(image, caption=f"Uploaded image · {image.width} × {image.height}", use_container_width=True)
            prediction_key = f"{task}:{architecture}:{hashlib.sha256(image_bytes).hexdigest()}"
            if checkpoint_path.is_file():
                if st.button("Run model inference", type="primary"):
                    with st.spinner("Running the trained model..."):
                        model, checkpoint, device = load_checkpoint(checkpoint_path)
                        prediction = predict_image(image, model, checkpoint, device)
                        prediction["tensor"] = prediction["tensor"].detach().cpu()
                    st.session_state["image_prediction_key"] = prediction_key
                    st.session_state["image_prediction"] = prediction

                prediction = (
                    st.session_state.get("image_prediction")
                    if st.session_state.get("image_prediction_key") == prediction_key
                    else None
                )
                if prediction is not None:
                    st.subheader("Model output")
                    st.caption(f"Experimental output from {architecture} · checkpoint-backed inference")
                    if task_type == "multilabel":
                        selected = prediction["prediction"] or ["No labels met the configured threshold"]
                        st.write("Thresholded labels: " + ", ".join(selected))
                    else:
                        st.write("Highest probability class: **" + prediction["prediction"] + "**")
                    probability_frame = pd.DataFrame({"Class": prediction["classes"], "Probability": prediction["probabilities"]})
                    st.dataframe(probability_frame, hide_index=True, use_container_width=True)
                    if st.checkbox("Show optional Grad-CAM attention overlay"):
                        with st.spinner("Generating an attention visualization..."):
                            model, checkpoint, device = load_checkpoint(checkpoint_path)
                            top_index = int(np.argmax(prediction["probabilities"]))
                            cam = grad_cam(model, prediction["tensor"].to(device), top_index, checkpoint["architecture"])
                        st.image(overlay_heatmap(prediction["image"], cam), caption="Grad-CAM attention visualization; not a segmentation mask", use_container_width=True)
            else:
                st.warning("The image is valid and previewed. Inference is unavailable because no trained checkpoint exists for the selected task/model.")
        except (ValueError, OSError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Inference failed safely: {exc}")
    else:
        st.session_state.pop("image_prediction_key", None)
        st.session_state.pop("image_prediction", None)
        st.info("Upload one supported 2D raster image (JPG, PNG, TIFF, or BMP).")
    st.caption("Uploaded image bytes are processed in memory and are not written to the project data folders.")

elif page == "Dataset status":
    st.header("Dataset and data quality")
    st.write(f"Default folder: `{display_project_path(default_data_root)}`")
    if not default_data_root.exists():
        st.warning("Dataset folder is not present. Add the approved dataset locally and run `python audit_data.py --task " + task + "` before training.")
    else:
        audit_path = Path(RESULTS_ROOT) / task / "dataset_audit.json"
        if audit_path.is_file():
            try:
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                a, b, c = st.columns(3)
                a.metric("Resolved records", audit.get("records_resolved", 0))
                b.metric("Readable images", audit.get("image_audit", {}).get("readable", 0))
                c.metric("Missing references", len(audit.get("missing_image_references", [])))
                st.write("Class counts")
                st.bar_chart(audit.get("label_counts", {}))
                st.json(audit.get("image_audit", {}))
            except (json.JSONDecodeError, OSError) as exc:
                st.error(f"Could not read dataset audit: {exc}")
        else:
            st.info("Dataset files are present, but no audit has been generated. Run the audit command before training.")
    st.markdown("Dataset download, license acceptance, and access credentials are handled by the user. Do not place private or identifiable patient data in this repository.")

elif page == "Model results":
    st.header("Held-out model results")
    result_path = Path(RESULTS_ROOT) / task / architecture / "metrics.json"
    if not result_path.is_file():
        st.info("No held-out evaluation result is available for this task/model. Train the model locally before reporting performance.")
    else:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            metrics = result.get("metrics", {})
            fields = [name for name in ("accuracy", "subset_accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro", "roc_auc_macro_ovr") if metrics.get(name) is not None]
            if fields:
                cols = st.columns(min(4, len(fields)))
                for index, name in enumerate(fields):
                    cols[index % len(cols)].metric(name.replace("_", " ").title(), f"{metrics[name]:.3f}")
            st.caption(f"Held-out records: {result.get('split_counts', {}).get('test', 'unknown')} · Checkpoint: {result.get('checkpoint', 'unknown')}")
            st.subheader("Per-class metrics")
            st.dataframe(pd.DataFrame.from_dict(metrics.get("per_class", {}), orient="index"), use_container_width=True)
            matrix = metrics.get("confusion_matrix")
            if matrix is not None:
                st.write("Confusion matrix (rows = actual, columns = predicted)")
                st.dataframe(pd.DataFrame(matrix, index=metrics.get("class_order"), columns=metrics.get("class_order")), use_container_width=True)
            matrices = metrics.get("confusion_matrices")
            if matrices:
                selected_class = st.selectbox("Class confusion matrix", list(matrices))
                st.dataframe(pd.DataFrame(matrices[selected_class], index=["actual negative", "actual positive"], columns=["predicted negative", "predicted positive"]), use_container_width=True)
            for artifact_name in ("training_curve.png", "class_metrics.png", "confusion_matrix.png"):
                artifact = Path(RESULTS_ROOT) / task / architecture / artifact_name
                if artifact.is_file():
                    st.image(str(artifact), caption=artifact_name.replace("_", " ").replace(".png", "").title(), use_container_width=True)
            st.json({key: value for key, value in result.items() if key != "metrics"})
        except (json.JSONDecodeError, OSError) as exc:
            st.error(f"Could not read model metrics: {exc}")

else:
    st.header("Methodology and limitations")
    st.markdown("""
    - The report names NIH ChestX-ray14, the Kaggle APTOS retinopathy dataset, and the ISIC Skin Lesion Archive.
    - Classification models use pre-trained ResNet or DenseNet backbones; the task determines the output and loss.
    - Patient or lesion identifiers are used for grouped data splitting when supplied in metadata.
    - Validation loss selects checkpoints. The test split is reserved for final evaluation.
    - Grad-CAM is an optional model-attention visualization, not a segmentation or clinical explanation.
    - No model is clinically validated. Outputs must not be used for diagnosis or treatment decisions.
    """)
    st.write("Run `python audit_data.py --task <task>` and then `python train.py --task <task> --architecture <architecture>` after arranging licensed data in `data/raw/`.")

