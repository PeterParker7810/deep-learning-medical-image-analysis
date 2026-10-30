# Phase III Project Submission

**Project:** Deep Learning Based Medical Image Analysis For Early Disease Detection  
**Report basis:** supplied Phase-I project report. Exact student/institution details should be entered from the approved cover page.

## Deliverables

1. `Performance Results.pdf` - actual ISIC 2016 Task 3 ResNet-18 held-out metrics, confusion matrix, class metrics, and training curves.
2. `Dataset & Results Analysis.pdf` - source selection, audit, counts, preprocessing, split policy, and measured findings.
3. `Deployed_Application.pdf` - local application and deployment status. Public URL is pending; no URL is invented.
4. `Screenshots or Technical Documentation.pdf` - genuine dashboard, image inference, model results, and Grad-CAM captures.
5. `Technical Documentation.pdf` - requirements, implementation, architecture, methodology, limitations, and references.
6. `Testing Report.pdf` - executed unit/mechanics tests and actual dataset/model/application evidence.
7. `Working Prototype & Demo Video.mov` - not recorded. See `supporting_materials/demo_recording_script.md` for a real recording script and checklist.

## Scope and remaining inputs

The Phase-I report names NIH ChestX-ray14, APTOS diabetic-retinopathy data, and ISIC. Only the official CC0 ISIC 2016 Task 3 release was available for this run. NIH download needs more workspace storage; APTOS files are absent, and the user currently cannot access Kaggle to accept the competition rules. The delivered application includes the real trained ISIC checkpoint and does not claim clinical validation. Public deployment is pending.

The latest repository test run passed (13 passed using a workspace-local pytest temporary directory); compilation and dependency checks passed. The ISIC test metrics were recomputed from the saved checkpoint and preserved test split and matched the recorded metrics exactly. Read each report for the exact split, metrics and limitations; no other task results are claimed.
