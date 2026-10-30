# Local Dataset Layouts

Do not commit the datasets. The `.gitignore` excludes files under `data/raw/` and `data/processed/` by default.

## NIH ChestX-ray14

Recommended layout:

```text
data/raw/chest_xray14/
  Data_Entry_2017.csv                 # or Data_Entry_2017_v2020.csv
  images/                              # images may also be nested below this folder
    *.png
```

The metadata needs `Image Index` and `Finding Labels`; if present, `Patient ID` is used for group-aware splitting. Otherwise the loader attempts the NIH `patient<number>` filename pattern. `No Finding` becomes an all-zero pathology vector. The dataset's source page requests attribution; retain the original metadata and cite the dataset paper in academic work.

## Kaggle APTOS retinopathy

To obtain the gated 2019 release, sign in to Kaggle, open the [APTOS 2019 data page](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data), review and accept the competition rules in your own account, then download the competition files. Do not provide Kaggle credentials to the project or commit them. Keep the labeled training files in the layout below.

```text
data/raw/aptos/
  train.csv                            # id_code, diagnosis
  train_images/
    <id_code>.png
```

The adapter reads the actual `diagnosis` values 0-4 described by the APTOS 2019 competition. The report does not identify a specific release, so confirm that your supplied files match this schema and the competition rules before training. No patient identifier is assumed.

## ISIC 2016 Challenge Task 3 (selected release)

```text
data/raw/isic2016/
  ISBI2016_ISIC_Part3_Training_Data.zip
  ISBI2016_ISIC_Part3_Training_GroundTruth.csv
  ISBI2016_ISIC_Part3_Test_Data.zip
  ISBI2016_ISIC_Part3_Test_GroundTruth.csv
  metadata.csv                         # generated from the four official files
  ISBI2016_ISIC_Part3_Training_Data/
  ISBI2016_ISIC_Part3_Test_Data/
```

The project selects Task 3 from the official 2016 ISBI challenge because it matches the report's skin-lesion classification scope, has small labeled train and test partitions, and is listed CC0. It labels lesions benign or malignant. Run `python prepare_isic2016.py` to safely extract the files, validate identifier coverage, create the explicit CC0 manifest, and create a readability/hash audit. The official test partition remains held out; validation is carved from the official training partition. The release does not supply patient/lesion IDs in its Task 3 label tables, so patient-level separation cannot be verified.

For any other ISIC release, use a separate manifest with `image_path`, `diagnosis`, `license`, and optional `patient_id`, `lesion_id`, and `split` columns. ISIC licenses vary by image, so preserve and check the exact license/attribution for the chosen collection.

The project does not automatically download datasets or accept third-party terms. Verify source access and licensing yourself before placing data here.

