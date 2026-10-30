# Approved Report Requirements and Implementation Checklist

## Source reviewed

The complete text, tables, references, and embedded media in `report.docx` were inspected. The document identifies itself as a Phase-I report and describes scope and objectives, not a completed experimental protocol. ISIC 2016 Task 3 data, metadata, trained ResNet-18 checkpoint, experiment outputs, and working app flow are now present locally. NIH and APTOS data remain unavailable. The four embedded media files in the report are university marks and do not specify a technical architecture.

The project title in the user-provided master instructions is **Deep Learning Based Medical Image Analysis For Early Disease Detection**. The report cover uses “Deep Learning-Based Medical Image Analysis for Early Disease Diagnosis,” while its declaration uses another title. The implementation will follow the title explicitly supplied in the master instructions and retain the report as-is.

## Requirement-to-implementation checklist

| ID | Approved requirement | Planned implementation | Evidence/status |
|---|---|---|---|
| R-01 | Analyze 2D medical images for early disease support | Three independent task configurations: chest X-ray, retinal fundus, and dermoscopic lesion images | Implement in `src/config.py` and task-specific data adapters |
| R-02 | Thoracic disease task using NIH ChestX-ray14 | Parse dataset metadata and pipe-delimited multi-label findings; train a multi-label image classifier | `src/data/chest_xray14.py`; requires user-supplied NIH files |
| R-03 | Diabetic retinopathy using Kaggle APTOS Retinopathy data | Read image ID and severity label from the supplied CSV; train a five-grade classifier when those labels are present | `src/data/aptos.py`; exact dataset release/layout must be confirmed from actual files |
| R-04 | Skin-lesion classification using ISIC Archive | Selected official ISIC 2016 Task 3 binary benign/malignant classification release; official train/test retained | `prepare_isic2016.py`, `src/data/isic.py`; CC0; audit: 1,275 readable images, zero unresolved/missing/unreadable/duplicate records |
| R-05 | CNNs and transfer learning with pre-trained models | Provide a documented model registry for ResNet and DenseNet transfer learning; a compact CNN baseline may be used for controlled comparison | `src/models.py`; exact approved backbone is not fixed by the report |
| R-06 | U-Net and segmentation are discussed in the literature | Keep segmentation out of the trained pipeline unless supplied data includes masks and the task is explicitly selected; provide optional Grad-CAM overlays for classifier attention visualization | `src/explainability.py`; Grad-CAM is a practical interpretation aid, not a segmentation output |
| R-07 | Image preprocessing and model output must be consistent | Centralize validation, RGB conversion, resize, normalization, and inference transforms | `src/preprocessing.py` |
| R-08 | Use transfer learning to help with limited labeled data | Support frozen-backbone warm-up and fine-tuning with configurable epochs, optimizer, learning rate, and batch size | `src/training.py`, `config.py` |
| R-09 | Avoid train/test leakage and protect patient independence | Split by patient/lesion group whenever metadata provides one; otherwise use deterministic stratified splitting and document the limitation | `src/data/splitting.py` |
| R-10 | Account for imbalanced labels | Report class frequencies; use configurable loss weighting based on training data only | `src/training.py`, `src/evaluation.py` |
| R-11 | Evaluate accuracy, precision, recall, F1, and ROC-AUC | Calculate supported metrics per class and macro averages; include confusion matrices and sensitivity/specificity where defined | `src/evaluation.py`; values only from actual held-out predictions |
| R-12 | Compare model performance with literature benchmarks | Keep published results separate from this project's measured results and record dataset/split comparability | `reports/` and generated results metadata |
| R-13 | Upload an image and show prediction probability | Streamlit task selector, image validation and preview, checkpoint-backed inference, class probabilities, and clear unavailable-model errors | `app.py`; real ISIC ResNet-18 output verified on official test image |
| R-14 | Show contributing region where applicable | Optional Grad-CAM heatmap for supported classifier backbones | `src/explainability.py`; disclose it is not a validated lesion mask |
| R-15 | System assists trained health professionals and is not a replacement | Put research-only and non-diagnostic language in app and documentation | UI and `README.md` |
| R-16 | Use public, anonymized data and protect privacy | Exclude raw datasets, credentials and identifying metadata; explicitly track only the trained ISIC checkpoint needed by the working app; ignore other checkpoints by default | `.gitignore`, data access documentation |
| R-17 | Provide reproducible training and evaluation | Seed Python/NumPy/PyTorch; save config, label map, split manifest, checkpoint, and metric JSON | `src/reproducibility.py`, `src/training.py` |
| R-18 | Test data, preprocessing, models, evaluation, and app behavior | Unit and pipeline tests for available code; integration test with generated non-medical fixtures only for mechanics, clearly not experimental evidence | `tests/` |
| R-19 | Prepare Phase III reports and demo | Create reports from actual dataset/model/test/app evidence; mark remaining external deliverables pending | `Phase_III_Project_Submission/`; six updated PDFs, genuine dashboard/analysis/results/Grad-CAM captures, recording script; MOV pending |

## Explicit implementation decisions

1. The report does not name one final backbone. The implementation will make ResNet and DenseNet options configurable, because both are explicitly discussed in the report. It will not assert either is the best model until a valid experiment is run.
2. The report discusses U-Net but supplies no segmentation masks or segmentation target. It will not label a classifier heatmap as segmentation. Grad-CAM may be offered as an optional visual explanation.
3. The report does not identify the exact APTOS/ISIC releases, splits, or label subset. ISIC 2016 Task 3 was selected as a report-compatible implementation choice and its official test partition is preserved. APTOS continues to require user-side Kaggle rules acceptance and download.
4. The report does not specify a train/validation/test ratio, hyperparameter values, or performance target. These will be explicit configuration defaults, reported as implementation choices, and not described as approved report requirements.
5. The report includes no experimental results. One actual ISIC ResNet-18 held-out run is now reported; no result is claimed for NIH, APTOS, or untrained backbone options.

## Acceptance checklist

- [x] ISIC layouts, labels, image counts, dimensions, duplicates, and CC0 metadata verified from official files.
- [x] ISIC adapter loads actual data; NIH/APTOS adapters need source data validation.
- [x] Official ISIC test split preserved; no patient/lesion IDs exist in Task 3 labels.
- [x] ISIC ResNet-18 training and validation completed for 8 epochs; NIH/APTOS training is pending data access.
- [x] Checkpoint reload and real ISIC image inference verified.
- [x] Held-out metrics, split counts and config metadata saved from actual run.
- [x] Streamlit launches; genuine prediction and Grad-CAM behavior verified; missing data/checkpoint paths handled.
- [x] Automated tests executed: 13 passed; `compileall` and `pip check` passed.
- [x] Phase III PDFs contain actual ISIC results; genuine dashboard/analysis screenshots saved; MOV remains pending.
- [x] Public deployment is pending because no deployment account/credentials are available.

## Blocking inputs for experimental completion

- NIH ChestX-ray14 full dataset download awaits adequate storage.
- APTOS 2019 data download awaits user sign-in and competition-rules acceptance.
- No expected model comparison set, split policy, training schedule, or target metric is prescribed in the Phase-I report.
- No deployment account, URL, or recording output location is provided.

These gaps prevent reporting the other two datasets' results, claiming model comparison across all named tasks, and providing a public deployment URL. ISIC dataset/model/app evidence is available and labeled as a single-task experiment.
