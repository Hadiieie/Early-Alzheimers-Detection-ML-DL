from pathlib import Path
import sys
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app_inference import load_final, predict_files


st.set_page_config(
    page_title="Alzheimer Early Detection",
    page_icon="🧠",
    layout="wide"
)


@st.cache_resource
def get_model():
    return load_final()


model, device, threshold = get_model()


st.title("🧠 Alzheimer's Early Detection")
st.caption(
    "Hippocampus-focused MRI analysis | Biomedical Engineering Graduation Project"
)

prediction_tab, anatomy_tab, about_tab = st.tabs(
    ["Prediction", "Anatomy & Explainability", "About"]
)


with prediction_tab:

    st.subheader("Patient MRI Prediction")

    st.write(
        "Upload three consecutive hippocampal MRI slices "
        "in anatomical order."
    )

    names = ["Peak - 1", "Peak", "Peak + 1"]
    columns = st.columns(3)
    files = []

    for column, name in zip(columns, names):
        with column:
            file = st.file_uploader(
                name,
                type=["png", "jpg", "jpeg"],
                key=name
            )

            files.append(file)

            if file:
                st.image(
                    Image.open(file),
                    caption=name
                )


    if all(files):

        if st.button("Run Prediction", type="primary"):

            with st.spinner("Analyzing hippocampal MRI slices..."):

                label, early, score = predict_files(
                    files,
                    model,
                    device,
                    threshold
                )

            st.divider()

            if label == "Early_Demented":
                st.warning("Prediction: EARLY DEMENTED")
            else:
                st.success("Prediction: NORMAL")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Early Demented Score",
                f"{early * 100:.1f}%"
            )

            c2.metric(
                "Normal Score",
                f"{(1 - early) * 100:.1f}%"
            )

            c3.metric(
                "Decision Threshold",
                f"{threshold:.2f}"
            )

            st.caption(
                "Model scores are neural-network outputs and should not "
                "be interpreted as calibrated probabilities of diagnosis."
            )


with anatomy_tab:

    st.subheader("Anatomical Localization")

    st.write(
        "The pipeline uses FreeSurfer SynthSeg to identify the "
        "left and right hippocampus before selecting the MRI slices."
    )

    st.code(
        "3D T1 MRI\n"
        "↓\n"
        "SynthSeg\n"
        "↓\n"
        "Left Hippocampus (17)\n"
        "Right Hippocampus (53)\n"
        "↓\n"
        "Peak hippocampal slice ± 1"
    )

    st.subheader("Grad-CAM Explainability")

    gradcam = (
        ROOT
        / "results"
        / "EXPLAINABILITY"
        / "gradcam_OAS1_0001_MR1.png"
    )

    if gradcam.exists():
        st.image(
            str(gradcam),
            caption="Grad-CAM visualization on three hippocampal slices"
        )
    else:
        st.info("Grad-CAM figure not found.")

    st.info(
        "FreeView is used separately for the live anatomical "
        "segmentation demonstration during the presentation."
    )


with about_tab:

    st.subheader("Project Pipeline")

    st.markdown(
        """
**Primary task:** Normal vs Early Demented

**MRI region:** Bilateral hippocampal / medial temporal region

**Pretraining:** ImageNet → Kaggle MRI dataset

**Patient-level analysis:** Three anatomically consistent OASIS slices

**Anatomical segmentation:** FreeSurfer SynthSeg

**Explainability:** Grad-CAM

**Visualization:** FreeView

**Deployment:** Streamlit
"""
    )

    st.warning(
        "Research and educational use only. "
        "This system is not a clinical diagnostic device."
    )