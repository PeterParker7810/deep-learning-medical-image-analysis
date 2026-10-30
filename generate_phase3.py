"""Generate Phase III PDFs from the verified local audit and experiment artifacts."""
from pathlib import Path
import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Phase_III_Project_Submission"
OUT.mkdir(exist_ok=True)
(OUT / "screenshots" / "application_screenshots").mkdir(parents=True, exist_ok=True)
(OUT / "supporting_materials").mkdir(exist_ok=True)
ART = ROOT / "reports" / "generated" / "isic_skin_lesion" / "resnet18"
AUDIT = ROOT / "reports" / "generated" / "isic_skin_lesion" / "dataset_audit.json"
metrics = json.loads((ART / "metrics.json").read_text(encoding="utf-8")) if (ART / "metrics.json").exists() else {}
scores = metrics.get("metrics", {})
audit = json.loads(AUDIT.read_text(encoding="utf-8")) if AUDIT.exists() else {}
TITLE = "Deep Learning Based Medical Image Analysis For Early Disease Detection"
NAVY, TEAL, PALE = colors.HexColor("#17324D"), colors.HexColor("#137C8B"), colors.HexColor("#EAF2F5")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="DocTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20, leading=25, textColor=NAVY, alignment=TA_CENTER, spaceAfter=14))
styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=14, leading=18, textColor=NAVY, spaceBefore=11, spaceAfter=6))
styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=9, leading=13, spaceAfter=6))
styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=7.5, leading=10, textColor=colors.HexColor("#465563")))

def p(text, style="Bodyx"):
    return Paragraph(str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), styles[style])

def table(rows, widths=None):
    data = [[p(cell, "Smallx") for cell in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#B9C8D2")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t

def footer(canvas, doc):
    canvas.saveState(); w, _ = A4
    canvas.setStrokeColor(TEAL); canvas.line(18*mm, 15*mm, w-18*mm, 15*mm)
    canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#536473"))
    canvas.drawString(18*mm, 10*mm, "Academic research prototype | Not for clinical diagnosis")
    canvas.drawRightString(w-18*mm, 10*mm, f"Page {doc.page}"); canvas.restoreState()

def build(filename, subtitle, sections):
    doc = SimpleDocTemplate(str(OUT/filename), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=17*mm, bottomMargin=21*mm, title=subtitle, author="Academic project submission")
    story = [Spacer(1, 18*mm), p(TITLE, "DocTitle"), p(subtitle, "H1x"), p("Phase III project submission | 30 October 2026", "Bodyx"), p("Student / institution: enter the approved cover details. No identity or approval has been invented.", "Smallx")]
    for heading, content in sections:
        story.append(p(heading, "H1x"))
        for item in content if isinstance(content, list) else [content]:
            if isinstance(item, tuple) and item[0] == "table": story.extend([table(item[1], item[2] if len(item)>2 else None), Spacer(1, 3*mm)])
            elif isinstance(item, tuple) and item[0] == "break":
                story.append(PageBreak())
            elif isinstance(item, tuple) and item[0] == "image":
                path = Path(item[1])
                if path.exists():
                    img = Image(str(path)); img._restrictSize(165*mm, 175*mm); story.extend([img, p(item[2], "Smallx"), Spacer(1, 3*mm)])
            else: story.append(p(item))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)

def fmt(value, digits=4):
    return "Pending" if value is None else f"{value:.{digits}f}"

def metric_rows():
    return [["Metric", "Held-out official test result"], ["Accuracy", fmt(scores.get("accuracy"))], ["Macro precision", fmt(scores.get("precision_macro"))], ["Macro recall / sensitivity", fmt(scores.get("recall_macro"))], ["Macro F1-score", fmt(scores.get("f1_macro"))], ["Macro ROC-AUC (OVR)", fmt(scores.get("roc_auc_macro_ovr"))]]

source_isic = 'Official ISIC Challenge 2016 Task 3 release, CC0: https://challenge.isic-archive.com/data/ (retrieved and recorded in local metadata). The report names ISIC but does not specify a year/release; Task 3 binary benign/malignant was selected and is documented as an implementation choice.'
blocked = 'NIH ChestX-ray14 source files are not available locally, and approximately 49.8 GB free space is insufficient for the full release plus extraction alongside current workspace contents. APTOS 2019 files are also absent; the user currently cannot access Kaggle to accept the competition rules and download them. These model paths are implemented but neither task has been trained or evaluated.'

build("Performance Results.pdf", "Performance Results", [
    ("Experiment status", ["One genuine experiment has completed: ISIC 2016 Task 3 binary benign/malignant classification with ImageNet-pretrained ResNet-18. Seed 42; 738 training, 158 validation, and 379 untouched official test images. Validation loss selected the checkpoint (0.6302 at epoch 4 of 8); the official test partition was held out from model selection. A subset of non-identifying audit and experiment artifacts is included for app reproducibility.", "This is an academic research prototype, not a clinically validated diagnostic system. Results must not guide medical care."]),
    ("Held-out test metrics", [("table", metric_rows(), [78*mm, 88*mm])]),
    ("Class-wise outcomes", [("table", [["Class", "Precision", "Sensitivity", "Specificity", "F1", "Support"], *[[name, fmt(v.get("precision")), fmt(v.get("recall_sensitivity")), fmt(v.get("specificity")), fmt(v.get("f1")), str(v.get("support"))] for name, v in scores.get("per_class", {}).items()]], [39*mm,25*mm,25*mm,25*mm,22*mm,20*mm])]),
    ("Confusion matrix", [f"Class order benign, malignant. Matrix: {scores.get('confusion_matrix', 'Pending')} (rows are reference labels; columns are predicted labels).", ("image", ART/"confusion_matrix.png", "Figure 1. Confusion matrix computed from the official test set.")]),
    ("Training and validation", [("image", ART/"training_curve.png", "Figure 2. Per-epoch training history; model selection used validation loss."), ("image", ART/"class_metrics.png", "Figure 3. Class-wise held-out test metrics."), f"Best validation loss: {metrics.get('best_validation_loss', 'recorded in metrics.json')}. Training epochs: {metrics.get('epochs_completed', 'recorded in history.json')}. Checkpoint selection did not use test metrics."]),
    ("Interpretation and model choice", ["ResNet-18 is the only trained candidate, so no comparative winner is claimed. Overall accuracy can obscure minority malignant errors: malignant precision and F1 are materially lower than benign performance, and the test set contains fewer malignant examples. The checkpoint is the selected model only for this run, chosen by validation loss; the measured scores do not demonstrate clinical utility.", blocked])])

build("Dataset & Results Analysis.pdf", "Dataset & Results Analysis", [
    ("Dataset sources and scope", [source_isic, "The supplied Phase-I report also identifies NIH ChestX-ray14 and Kaggle APTOS Retinopathy. Those remain unavailable for local experiments as described below. The three tasks have separate labels and models; their data are not combined.", blocked]),
    ("ISIC 2016 Task 3 dataset profile", [("table", [["Audit item", "Observed value"], ["Original listed records", "1,279 (900 train + 379 official test)"], ["Retained records", "1,275 (896 train + 379 official test)"], ["Labels", "Benign 1,028; malignant 247"], ["Training partition", "896: benign 724; malignant 172"], ["Official test partition", "379: benign 304; malignant 75"], ["Formats", "JPEG"], ["Most frequent dimensions", "1024 x 768 (659 images)"], ["Unreadable / missing references", "0 / 0"], ["Exact duplicate images", "0 after removing 4 redundant training copies"], ["Patient identifiers", "Not supplied in Task 3 labels"]], [62*mm,103*mm])]),
    ("Data quality and split policy", ["Four byte-identical training copies were identified; one repeated within training and three overlapped official test records. Labels matched. Redundant training copies were removed while all official test records were retained. No exact duplicates remain in the training/test manifest. Patient-level separation cannot be verified because patient/lesion IDs are absent; the source official train/test split is preserved and a stratified validation subset is drawn only from the official training partition.", "The binary training set is imbalanced (724 benign, 172 malignant). Training-only class weighting is used. Image dimensions vary; preprocessing resizes to 224 x 224 and applies ImageNet normalization. Training augmentation is enabled; validation, test and inference use deterministic transforms."]),
    ("Exploratory and model results", [("image", ART/"class_metrics.png", "Figure 1. Class-wise metrics for the held-out official test partition."), ("table", metric_rows(), [78*mm,88*mm]), "Confusion matrix (benign, malignant order): " + str(scores.get("confusion_matrix", "pending")) + ". The matrix reports both false positives and false negatives; see Performance Results for counts and curves."]),
    ("Conclusions and limitations", ["Measured performance is specific to this release, split, preprocessing and single ResNet-18 run. It is not validated on other institutions or devices. No clinical claim is supported. NIH and APTOS findings cannot be reported until their actual files are supplied and audited."])])

build("Deployed_Application.pdf", "Deployed Application", [
    ("Deployment status", ["Public deployment URL: PENDING. No deployment platform account or credentials were available. The repository is prepared for deployment, but a public service has not been launched.", "The working application was run locally at http://127.0.0.1:8511 during verification. This local address is not a public deployment URL. The source repository is private at https://github.com/PeterParker7810/deep-learning-medical-image-analysis."]),
    ("Overview and verified functionality", ["Streamlit research prototype with Overview, Image analysis, Dataset status, Model results and Methodology sections. In the image workflow, an image is validated, previewed, preprocessed, passed to the selected trained checkpoint and shown with class probabilities. A Grad-CAM attention overlay is available for the verified ResNet-18 checkpoint; it is not a segmentation mask or clinical explanation.", "The application has been exercised using a genuine image from the official ISIC test partition and the actual trained checkpoint. Dataset availability and metrics reflect generated local experiment artifacts."]),
    ("Architecture and access", ["Run `python -m pip install -r requirements.txt`, place authorized dataset files in the documented local layout, run `python audit_data.py --task isic_skin_lesion --data-root data/raw/isic2016 --manifest metadata.csv`, then `streamlit run app.py`. The checkpoint is stored at `models/saved_models/isic_skin_lesion/resnet18/model.pt` locally. The app runs inference locally; image bytes are held in memory by the application code."]),
    ("Limitations", ["NIH and APTOS pages show dataset-specific pending status. Public deployment is pending. The app is a research prototype, not clinically validated; predictions must not replace professional medical assessment. Avoid identifiable patient images."])])

build("Screenshots or Technical Documentation.pdf", "Application Screens and Workflow Documentation", [
    ("Evidence policy", ["The screenshots directory contains genuine captures only. The analysis view was exercised with an actual held-out ISIC image, real ResNet-18 inference and Grad-CAM. Additional dashboard, dataset and results captures are included when saved; no mock interface is substituted."]),
    ("Screenshots", [("image", OUT/"screenshots"/"dashboard.png", "Figure 1. Application overview captured from the working Streamlit app."), ("break",), ("image", OUT/"screenshots"/"analysis.png", "Figure 2. Genuine image upload, checkpoint-backed prediction and class probabilities."), ("break",), ("image", OUT/"screenshots"/"model_results.png", "Figure 3. Genuine model results page showing the actual ISIC test evaluation."), ("break",), ("image", OUT/"screenshots"/"application_screenshots"/"gradcam.png", "Figure 4. Grad-CAM view from the working app and actual model checkpoint."), ("break",), ("image", ART/"confusion_matrix.png", "Figure 5. Actual model results visualization; numeric results are also tabulated in Performance Results." )]),
    ("Workflow", ["Select ISIC skin-lesion task and ResNet-18; upload an allowed image; inspect the preview; run model inference; review class probabilities; optionally enable Grad-CAM. Invalid files are rejected before inference. Outputs are experimental and not diagnostic.", "The included screenshots are saved separately in the submission `screenshots/` folders."]),
    ("Demo recording", ["No MOV is included unless an actual screen recording is captured. The recording script and checklist are supplied under supporting_materials. A real prototype can be demonstrated locally using the above workflow."])])

build("Technical Documentation.pdf", "Technical Documentation", [
    ("Abstract, problem statement and objectives", ["The approved Phase-I report investigates early disease detection support for thoracic findings in chest radiographs, diabetic-retinopathy severity in retinal fundus images, and skin lesions in dermoscopic images. This implementation provides separate deep-learning image classification pipelines and a local interactive research app. It is not a diagnostic product.", "The report is Phase I rather than an exact final experimental protocol. It does not determine releases, exact model selection, training hyperparameters, split proportions or performance targets. These choices are recorded as implementation decisions rather than attributed to the report."]),
    ("Scope and implementation checklist", ["Requirement traceability is maintained in IMPLEMENTATION_CHECKLIST.md. ISIC 2016 Task 3 was selected because the report says ISIC without identifying a release. NIH and APTOS remain coded but await actual local dataset access. U-Net is not used because no segmentation masks or segmentation target are specified."]),
    ("Requirements and technology", [("table", [["Category", "Implementation"], ["Runtime", "Python 3.11, virtual environment"], ["ML", "PyTorch, torchvision, scikit-learn"], ["UI", "Streamlit"], ["Data/visualization", "Pillow, NumPy, Pandas, Matplotlib"], ["Hardware verified", "Windows; NVIDIA RTX 3050 Laptop GPU, CUDA available; 6 GB VRAM"], ["Storage / deployment", "Datasets and checkpoint remain local; deployment URL pending"]], [47*mm,118*mm])]),
    ("System architecture and data flow", ["Dataset metadata adapter -> validated image records -> deterministic train/validation/test partition -> common image transform -> transfer-learning classifier -> training-only class weights and validation-loss checkpoint selection -> held-out evaluation -> serialized checkpoint and metrics -> Streamlit prediction and optional Grad-CAM.", "ISIC official test records remain fixed; validation is split from source training records. No patient identifiers are present in the selected task label files. The independent test set is not used for selection."]),
    ("Algorithms and evaluation", ["ResNet-18, ResNet-50, and DenseNet-121 are configurable transfer-learning classifiers. The completed experiment uses pretrained ResNet-18 and binary softmax cross-entropy. Metrics include accuracy, macro precision/recall/F1, one-vs-rest ROC-AUC, per-class sensitivity/specificity and confusion matrix. Training set class weights address the observed imbalance. Seed: recorded in experiment metadata."]),
    ("Module descriptions", ["`src/data/` loads task metadata and labels; `src/datasets.py` exposes the PyTorch dataset; `src/preprocessing.py` validates inputs and defines transforms; `src/models.py` builds classifier backbones; `src/training.py` trains and checkpoints; `src/evaluation.py` computes metrics; `src/prediction.py` performs inference; `src/explainability.py` creates Grad-CAM; `src/pipeline.py` orchestrates the experiment; `app.py` is the Streamlit UI; `train.py` and `audit_data.py` are command-line entry points."]),
    ("Security, error handling and limitations", ["Raw image data and checkpoint files are excluded from ordinary Git tracking. The app enforces upload size/type/readability checks and handles missing model artifacts. No API keys are required. Medical images should remain de-identified. Dataset shift, label quality, class imbalance and unknown patient separation limit generalization. Outputs are experimental and not clinically validated."]),
    ("Future work and conclusion", ["Obtain the complete NIH release after provisioning adequate storage; accept APTOS rules and download via the user's Kaggle account; audit both datasets; train using approved splits; add comparative backbone runs and repeat-seed uncertainty analysis; review dataset-specific licensing; deploy only with suitable privacy controls and hosting resources."]),
    ("References", ["Supplied Phase-I project report, `report.docx` (local source specification; intentionally not published because it contains academic identity details).", "ISIC Challenge data: https://challenge.isic-archive.com/data/", "NIH Clinical Center ChestX-ray dataset: https://nihcc.app.box.com/v/ChestXray-NIHCC", "APTOS 2019 competition data and rules: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data", "PyTorch installation selector: https://pytorch.org/get-started/locally/"])])

build("Testing Report.pdf", "Testing Report", [
    ("Objectives and environment", ["Tests cover metadata and data loading, preprocessing, split integrity, model initialization, metrics, checkpoint save/load, end-to-end training mechanics, real dataset audit and model-backed inference. Windows, Python 3.11, PyTorch 2.15.0+cu132, torchvision 0.29.2+cu132, RTX 3050 Laptop GPU (CUDA)."]),
    ("Executed test cases", [("table", [["ID", "Test scope", "Expected result", "Actual result", "Status"], ["T-01", "Fixture metadata adapter", "Validated records and labels", "Passed", "PASS"], ["T-02", "Grouped and predefined split", "No leakage; official test preserved", "Passed", "PASS"], ["T-03", "Image transform", "Consistent tensor shape", "Passed", "PASS"], ["T-04", "Model forward/checkpoint reload", "Correct output shape", "Passed", "PASS"], ["T-05", "Metric calculation", "Finite metrics and confusion matrix", "Passed", "PASS"], ["T-06", "One-epoch fixture pipeline", "Train and checkpoint execute", "Passed", "PASS"], ["T-07", "ISIC source audit", "Readable data, manifest and duplicates checked", "1,275 records; no unreadables/missing/duplicates", "PASS"], ["T-08", "GPU model prediction", "Checkpoint loads and gives probabilities", "ISIC held-out image predicted; probabilities returned", "PASS"], ["T-09", "Grad-CAM inference", "Valid overlay dimensions", "224x224 map; source-sized overlay", "PASS"], ["T-10", "Streamlit inference and state", "Result persists and Grad-CAM renders", "Verified in browser", "PASS"], ["T-11", "Full pytest suite", "All tests pass", "13 passed in 6.99 s", "PASS"], ["T-12", "Compilation / dependencies", "No compile or dependency errors", "compileall passed; pip check clean", "PASS"], ["T-13", "NIH and APTOS full-data tests", "Audit and train", "Not executed: source datasets unavailable", "PENDING"]], [12*mm,39*mm,40*mm,55*mm,17*mm])]),
    ("Method and summary", ["Synthetic fixtures validate software mechanics only; they are not medical evidence. The real ISIC audit, training, held-out evaluation, CUDA inference, app prediction and Grad-CAM were separately exercised. NIH/APTOS integration remains untested until the full sources are available. Final command: `.venv/Scripts/python.exe -m pytest -q --basetemp=.pytest-tmp` (13 passed in 6.99 s). Bytecode compilation succeeded and `pip check` reported no broken requirements.", "Known outstanding deliverables: public deployment, NIH/APTOS source data, and a recorded .mov demo. A concise recording script and checklist are provided."])])

print(f"Created six PDFs under {OUT}")

