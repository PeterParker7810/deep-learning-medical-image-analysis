# Working Prototype Demo Recording Script

**Status:** No video recording has been produced yet. The local app, ISIC 2016 data audit, ResNet-18 checkpoint, held-out metrics, image inference, and Grad-CAM are available for a real capture. NIH/APTOS data and public deployment remain pending.

## Before recording

1. Install the project requirements and a compatible PyTorch/torchvision build.
2. Start from the repository root with `streamlit run app.py`.
3. Confirm that the local browser page loads and that the application shows the real dataset/checkpoint status.
4. Use the official ISIC 2016 Task 3 sample currently under local data and its actual trained checkpoint. Do not show private patient data or describe the score as a diagnosis.
5. Start screen capture at 1920 x 1080, 30 fps, with microphone narration optional. Save the actual recording as `Working Prototype & Demo Video.mov`.

## Suggested 3-4 minute narration and sequence

1. **Introduction (0:00-0:25):** State the project title, academic research scope, and three report-defined use cases: chest X-ray findings, diabetic-retinopathy grading, and skin-lesion classification. State that it is not clinically validated and is not for diagnosis.
2. **Launch and overview (0:25-0:55):** Show the Streamlit landing dashboard. Point out detected research tasks, dataset status, and whether a real checkpoint is available. Narrate the values shown on screen; do not imply missing assets are present.
3. **Dataset status (0:55-1:25):** Open Dataset status and show the ISIC audit: 1,275 readable records, benign/malignant counts, no exact duplicates after cleanup. State NIH storage and APTOS Kaggle access are pending.
4. **Image analysis (1:25-2:15):** Open Image analysis, upload the official ISIC sample, show the preview, run the real ResNet-18 inference, and point out the class probabilities. Explain this is an experimental output, not a diagnosis.
5. **Model results (2:15-3:00):** Show actual test metrics and confusion matrix from the report, explain the preserved official test split and the lower malignant precision/F1, and state only one backbone has been trained.
6. **Grad-CAM and methodology (3:00-3:30):** Enable the actual Grad-CAM attention overlay, state it is not a segmentation mask, mention data split and training-only class weighting, then close with the research-only disclaimer.

## Final recording checklist

- [ ] The recording shows the real running app, not slides or a mockup.
- [ ] On-screen content matches the current dataset/model state.
- [ ] Any sample medical image is authorized and has no identifying metadata.
- [ ] No predictions or performance scores are invented.
- [ ] The research-only / non-diagnostic notice is visible or narrated.
- [ ] Save the unedited original video in this submission directory; note recording date and software in a short metadata file.
