# Early Detection of Alzheimer's Disease Using Machine Learning and Deep Learning

## Graduation Project

**Department of Biomedical Engineering**  
**Jordan University of Science and Technology — JUST**

This project investigates the early detection of Alzheimer's disease from brain MRI using a combination of **medical image processing, anatomical biomarkers, machine learning, and deep learning**.

The project focuses particularly on structures in the medial temporal lobe, including the **hippocampus and amygdala**, which are strongly associated with Alzheimer's-related neurodegeneration.

---

## Project Objectives

The main objectives of this project are to:

- Develop an MRI-based pipeline for early Alzheimer's disease detection.
- Apply medical image preprocessing and anatomical segmentation.
- Extract hippocampal and amygdala biomarkers.
- Compare Machine Learning and Deep Learning approaches.
- Prevent data leakage through patient-level evaluation.
- Provide explainable model predictions.
- Develop an interactive prediction interface using Streamlit.
- Provide reproducible code and documented experiments.

---

# Datasets

## 1. Kaggle MRI Dataset

The initial dataset contained:

**40,000 MRI images**

distributed among four classes:

- Non-Demented
- Very Mild Demented
- Mild Demented
- Moderate Demented

An integrity audit was performed before training.

### Duplicate Detection

Exact duplicate images were identified using image hashes.

After duplicate removal:

**39,538 unique images remained.**

For the primary binary task:

- Normal: 9,984 images
- Early Demented: 19,869 images

Total:

**29,853 MRI images**

The Early Demented group combines:

- Very Mild Demented
- Mild Demented

The Moderate Demented class was excluded from the primary early-detection task.

Because reliable patient identifiers were unavailable in the Kaggle dataset, the Kaggle data was used primarily for **large-scale MRI representation learning and CNN pretraining**, rather than for the final patient-level clinical evaluation.

---

## 2. OASIS MRI Dataset

The OASIS dataset was used for patient-level anatomical analysis and evaluation.

Clinical metadata included:

- Age
- Sex
- MMSE
- Clinical Dementia Rating (CDR)
- Estimated Total Intracranial Volume (eTIV)
- Normalized Whole Brain Volume

For this project:

- CDR = 0 → Normal
- CDR = 0.5 or 1 → Early Demented

A data-integrity audit identified conflicting duplicate MRI scans and missing MRI data. These subjects were excluded before modeling.

---

# Overall System Pipeline

```text
Brain MRI
   |
   v
Data Integrity Audit
   |
   v
MRI Preprocessing
   |
   v
FreeSurfer / SynthSeg
   |
   v
Anatomical Segmentation
   |
   +-------------------------+
   |                         |
   v                         v
Hippocampus              Amygdala
   |                         |
   +------------+------------+
                |
                v
      Anatomical Feature Extraction
                |
                v
      Intracranial Volume Normalization
                |
                v
          Machine Learning
                |
                v
           Compact RBF-SVM
                |
                v
      Normal / Early Demented
```

A separate Deep Learning branch was also developed using ResNet18.

---

# Anatomical Segmentation

MRI segmentation was performed using **FreeSurfer SynthSeg**.

The primary anatomical structures used by the final Machine Learning model were:

| Structure | FreeSurfer Label |
|---|---:|
| Left Hippocampus | 17 |
| Right Hippocampus | 53 |
| Left Amygdala | 18 |
| Right Amygdala | 54 |

Volumes were normalized using estimated intracranial volume to reduce variation related to head size.

---

# FreeView Visualization

FreeView is used to visually inspect:

- Original T1-weighted MRI
- SynthSeg segmentation
- Hippocampal structures
- Amygdala structures
- Anatomical localization quality

FreeView provides an important quality-control and demonstration component of the project.

---

# Deep Learning Branch

A **ResNet18** convolutional neural network pretrained on ImageNet was adapted for MRI classification.

The network was first trained using the larger Kaggle dataset.

### Kaggle Development Result

Best image-level validation balanced accuracy:

**95.74%**

This result represents Kaggle image-level development validation and must not be interpreted as final patient-level clinical performance.

A patient-level multi-slice model was subsequently developed using OASIS MRI data.

The final selected CNN configuration used three anatomically localized MRI slices:

- Peak hippocampal slice - 1
- Peak hippocampal slice
- Peak hippocampal slice + 1

The CNN branch was retained as a Deep Learning comparison and for explainability experiments.

---

# Grad-CAM Explainability

Grad-CAM was implemented to visualize image regions influencing the CNN predictions.

Grad-CAM provides visual explanations for the Deep Learning branch by highlighting spatial regions contributing to the prediction.

Grad-CAM and FreeView serve different purposes:

- **FreeView:** anatomical segmentation and visualization
- **Grad-CAM:** CNN prediction explainability

---

# Machine Learning Branch

Multiple anatomical Machine Learning models were evaluated using normalized MRI-derived features.

Experiments included:

- Logistic Regression
- Support Vector Machine
- Random Forest

Anatomical feature groups included:

- Hippocampus
- Amygdala
- Ventricular measurements
- Anatomical asymmetry features

The strongest and most stable final approach was the:

## Compact Medial Temporal Lobe SVM

Candidate features:

- Left Hippocampus
- Right Hippocampus
- Left Amygdala
- Right Amygdala

The final feature-selection pipeline retained three anatomical features.

Final SVM configuration:

```text
Kernel: RBF
C: 1
Gamma: scale
Class weighting: balanced
```

---

# Final Current Performance

The primary project performance is reported using **Repeated Nested Patient-Level Cross-Validation**.

### Compact MTL-SVM

**Balanced Accuracy**

```text
88.3% ± 8.6%
```

**ROC-AUC**

```text
84.1% ± 15.9%
```

A development-stage parameter selection run produced:

```text
Balanced Accuracy = 88.5%
```

However, the repeated nested cross-validation result is used as the primary estimate because it provides a more conservative evaluation of model generalization.

---

# Important Evaluation Principle

The project distinguishes between:

- image-level validation,
- patient-level cross-validation,
- development data,
- exposed evaluation data,
- and truly unseen patient evaluation.

This distinction is important because MRI slices from the same patient can be highly correlated.

Therefore, the project prioritizes **patient-level evaluation** to reduce data leakage and overestimated performance.

---

# Streamlit Application

An interactive Streamlit interface was developed to demonstrate the prediction workflow.

The application includes:

- MRI upload
- Prediction output
- Score visualization
- Anatomical information
- Grad-CAM explainability
- Project information

The prediction scores displayed by the interface represent model outputs and should not be interpreted as calibrated clinical diagnostic probabilities.

The application is intended as a research prototype and not as a clinical diagnostic system.

---

# Repository Structure

```text
FINAL_PROJECT_2027/
|
|-- src/
|   |-- app.py
|   |-- app_inference.py
|   |-- freeze_final_svm.py
|   |-- compact_mtl_nested_cv.py
|   |-- nested_mtl_cv.py
|   |-- build_mtl_features.py
|   |-- train_kaggle.py
|   |-- train_final_model.py
|   |-- model.py
|   |-- patient_model.py
|   |-- gradcam.py
|   |-- make_gradcam_demo.py
|   |-- extract_oasis_rois.py
|   |-- hippocampus_roi.py
|   |-- qc_hippocampus.py
|   |-- archive/
|
|-- tests/
|
|-- models/
|
|-- results/
|
|-- .gitignore
|
|-- requirements.txt
|
`-- README.md
```

The `src/archive/` directory contains experimental scripts retained for research traceability but not required for the main pipeline.

---

# Major Technologies

- Python
- PyTorch
- Torchvision
- Scikit-learn
- Pandas
- NumPy
- NiBabel
- FreeSurfer
- SynthSeg
- FreeView
- Streamlit
- Matplotlib
- Grad-CAM

---

# Running the Streamlit Application

From the project root:

```powershell
python -m streamlit run src/app.py
```

The interface will open in the local web browser.

---

# Reproducibility

The repository separates:

- raw datasets,
- processed MRI data,
- trained models,
- source code,
- evaluation results,
- experimental scripts.

Large medical datasets and model checkpoints are intentionally excluded from GitHub using `.gitignore`.

Clinical MRI data is not distributed through this repository.

---

# Research Limitations

Several limitations should be considered:

1. The available patient-level OASIS cohort is substantially smaller than the Kaggle image dataset.

2. Kaggle MRI images do not contain reliable patient identifiers, preventing strict patient-level evaluation on that dataset.

3. Deep Learning performance can be unstable when training on small patient cohorts.

4. The system remains a research prototype and has not undergone prospective clinical validation.

5. MRI acquisition differences and population differences may affect model generalization.

---

# Future Work

Future work may include:

- Expanding the OASIS patient cohort.
- External validation using additional MRI datasets.
- Improved MRI registration and preprocessing.
- Larger patient-level Deep Learning datasets.
- Multimodal integration with clinical variables.
- Longitudinal MRI analysis.
- Model calibration.
- Independent clinical validation.

---

# Disclaimer

This project is developed for academic and research purposes.

The system is **not a medical device and should not be used for clinical diagnosis or treatment decisions**.

---

## Biomedical Engineering Graduation Project

**Jordan University of Science and Technology**

**Early Detection of Alzheimer's Disease Using Machine Learning and Deep Learning**