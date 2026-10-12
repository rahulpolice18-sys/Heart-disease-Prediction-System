
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import traceback

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

BASE_DIR = Path(__file__).parent


# ==========================================
# LIGHT THEME
# ==========================================
st.markdown("""
<style>
.stApp {
    background-color: #F3F7FC;
    color: #243247;
}

header[data-testid="stHeader"] {
    background-color: #F3F7FC;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #153E75 !important;
}

.stApp p,
.stApp label,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stMarkdownContainer"] p {
    color: #243247 !important;
}

.hero-description {
    color: #43566E !important;
    font-size: 16px;
    line-height: 1.8;
}

div[data-testid="stForm"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D6E2F0;
    border-radius: 16px;
    padding: 24px;
}

div[data-testid="stWidgetLabel"] p {
    color: #243247 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] *,
div[data-baseweb="select"] input {
    color: #243247 !important;
}

div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[role="listbox"] {
    background-color: #FFFFFF !important;
}

li[role="option"] {
    color: #243247 !important;
}

div[data-testid="stNumberInput"] input {
    background-color: #FFFFFF !important;
    color: #243247 !important;
    border: 1px solid #CBD5E1 !important;
}

div[data-testid="stSlider"] label,
div[data-testid="stSlider"] [data-testid="stMarkdownContainer"] {
    color: #243247 !important;
}

div[data-testid="stFormSubmitButton"] > button {
    background-color: #1769AA !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 650 !important;
    min-height: 44px;
}

div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #10528A !important;
    color: #FFFFFF !important;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL COMPONENTS
# ==========================================
@st.cache_resource
def load_components():

    model_path = BASE_DIR / "heart_model.pkl"
    scaler_path = BASE_DIR / "scaler.pkl"
    selector_path = BASE_DIR / "selector.pkl"

    # Check that all model files exist
    required_files = [
        model_path,
        scaler_path,
        selector_path
    ]

    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Missing file: {file_path.name}. "
                f"Expected location: {file_path}"
            )

        if file_path.stat().st_size == 0:
            raise ValueError(
                f"{file_path.name} is empty."
            )

    # Load the model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {model_path.name}. "
            f"Error type: {type(e).__name__}. "
            f"Details: {repr(e)}"
        ) from e

    # Load the scaler
    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {scaler_path.name}. "
            f"Error type: {type(e).__name__}. "
            f"Details: {repr(e)}"
        ) from e

    # Load the feature selector
    try:
        selector = joblib.load(selector_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {selector_path.name}. "
            f"Error type: {type(e).__name__}. "
            f"Details: {repr(e)}"
        ) from e

    return model, scaler, selector


try:
    model, scaler, selector = load_components()

except Exception as e:
    st.error("Unable to load model files.")

    st.code(
        f"Error type: {type(e).__name__}\n"
        f"Error details: {repr(e)}"
    )

    st.info(
        "Check the exact error above. It identifies whether "
        "a file is missing, inaccessible, or incompatible "
        "with the installed Python libraries."
    )

    st.stop()


# ==========================================
# HERO SECTION
# ==========================================
hero_text, hero_image = st.columns(
    [1.15, 1],
    vertical_alignment="center"
)

with hero_text:

    st.markdown(
        """
        <h1 style="
            color: #153E75;
            font-size: 36px;
            font-weight: 750;
            line-height: 1.3;
        ">
            ❤️ Heart Disease Prediction
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.write(
        "Your heart continuously pumps blood, supplying "
        "oxygen and essential nutrients throughout your body."
    )

    st.write(
        "Maintaining heart health through healthy habits "
        "and regular medical checkups can help reduce "
        "cardiovascular risks."
    )

with hero_image:

    image_path = BASE_DIR / "heart_banner.png"

    if image_path.is_file():
        st.image(
            str(image_path),
            use_container_width=True
        )
    else:
        st.warning(
            "Heart image not found. Upload "
            "heart_banner.png to the same folder as app.py."
        )

st.divider()


# ==========================================
# PATIENT HEALTH INFORMATION
# ==========================================
st.markdown("## Patient Health Information")

st.write(
    "Enter the patient's details below to generate "
    "a machine-learning prediction."
)


# ==========================================
# DROPDOWN OPTIONS
# Verify encodings against your training dataset.
# ==========================================
chest_pain_options = {
    "Typical angina (0)": 0,
    "Atypical angina (1)": 1,
    "Non-anginal pain (2)": 2,
    "Asymptomatic (3)": 3
}

restecg_options = {
    "Normal (0)": 0,
    "ST-T wave abnormality (1)": 1,
    "Left ventricular hypertrophy (2)": 2
}

slope_options = {
    "Upsloping (0)": 0,
    "Flat (1)": 1,
    "Downsloping (2)": 2
}

thal_options = {
    "Unknown / not specified (0)": 0,
    "Normal (1)": 1,
    "Fixed defect (2)": 2,
    "Reversible defect (3)": 3
}


# ==========================================
# PATIENT INPUT FORM
# ==========================================
with st.form("heart_prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.slider(
            "Age",
            min_value=20,
            max_value=100,
            value=50
        )

        sex_label = st.selectbox(
            "Sex",
            ["Female (0)", "Male (1)"]
        )

        cp_label = st.selectbox(
            "Chest Pain Type",
            list(chest_pain_options.keys())
        )

        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80,
            max_value=200,
            value=120
        )

        chol = st.number_input(
            "Cholesterol (mg/dl)",
            min_value=100,
            max_value=600,
            value=200
        )

    with col2:

        fbs_label = st.selectbox(
            "Fasting Blood Sugar >120 mg/dl",
            ["No (0)", "Yes (1)"]
        )

        restecg_label = st.selectbox(
            "Resting ECG Result",
            list(restecg_options.keys())
        )

        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=60,
            max_value=220,
            value=150
        )

        exang_label = st.selectbox(
            "Exercise-Induced Angina",
            ["No (0)", "Yes (1)"]
        )

    with col3:

        oldpeak = st.slider(
            "Oldpeak",
            min_value=0.0,
            max_value=6.0,
            value=1.0,
            step=0.1
        )

        slope_label = st.selectbox(
            "ST Segment Slope",
            list(slope_options.keys())
        )

        ca = st.selectbox(
            "Number of Major Vessels",
            [0, 1, 2, 3]
        )

        thal_label = st.selectbox(
            "Thal Test Category",
            list(thal_options.keys())
        )

    submitted = st.form_submit_button(
        "Predict Heart Disease Risk",
        use_container_width=True
    )


# ==========================================
# PREDICTION
# ==========================================
if submitted:

    # Convert displayed labels to numeric values
    sex = 1 if sex_label.startswith("Male") else 0

    cp = chest_pain_options[cp_label]

    fbs = 1 if fbs_label.startswith("Yes") else 0

    restecg = restecg_options[restecg_label]

    exang = 1 if exang_label.startswith("Yes") else 0

    slope = slope_options[slope_label]

    thal = thal_options[thal_label]

    # Preserve the original 13 input features and order
    columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ]

    input_data = pd.DataFrame(
        [[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]],
        columns=columns
    )

    try:
        # Step 1: Apply feature selection
        input_selected = selector.transform(input_data)

        # Step 2: Apply scaling
        input_scaled = scaler.transform(input_selected)

        # Step 3: Generate prediction
        prediction = model.predict(input_scaled)

        st.divider()
        st.markdown("## Prediction Result")

        if prediction[0] == 1:

            st.error(
                "Potential Heart Disease Risk Detected. "
                "Please consult a qualified healthcare professional."
            )

        else:

            st.success(
                "The model did not identify a positive "
                "heart disease result for these inputs."
            )

        st.caption(
            "For educational purposes only. This prediction "
            "is not a medical diagnosis and does not replace "
            "professional medical evaluation."
        )

    except Exception as e:

        st.error("Prediction failed.")

        st.code(
            f"Error type: {type(e).__name__}\n"
            f"Error details: {repr(e)}"
        )

        st.info(
            "The input features, feature selector, scaler, "
            "and model must follow the same preprocessing "
            "order and feature encoding used during training."
        )


# ==========================================
# FOOTER
# ==========================================
st.divider()

st.markdown(
    """
    <div style="
        text-align: center;
        color: #64748B;
        padding: 10px;
        font-size: 13px;
    ">
        Heart Disease Prediction | Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)
