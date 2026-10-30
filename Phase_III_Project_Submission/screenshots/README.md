# Genuine application captures

All images in this folder were captured from the working local Streamlit application on 30 October 2026. The inference and Grad-CAM captures use an official ISIC 2016 Task 3 test image with the trained ResNet-18 checkpoint. The screenshot source is the CC0 ISIC challenge release; no patient identifiers are shown.

- `dashboard.png`: overview and live local availability summary. Displayed paths are project-relative.
- `analysis.png`: actual image preview and checkpoint-backed class probabilities.
- `model_results.png`: application metrics for the 379-image held-out test partition.
- `application_screenshots/gradcam.png`: Grad-CAM overlay from the live app; it is an attention visualization, not a segmentation mask.
