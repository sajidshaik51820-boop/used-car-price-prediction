import streamlit as st
import streamlit.components.v1 as components
import os
import joblib
import pandas as pd
import time
from datetime import datetime


# ============================================================
# USED CAR AI
# HOME → PRICE PREDICTION → CALCULATING → RESULT
# ============================================================

st.set_page_config(
    page_title="Used Car AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "vehicle_data" not in st.session_state:
    st.session_state.vehicle_data = None

if "price_animation_done" not in st.session_state:
    st.session_state.price_animation_done = False


# ============================================================
# PROJECT PATH
# ============================================================

# app.py and models folder are in the SAME repository/folder
APP_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    APP_DIR,
    "models",
    "xgboost_car_price_model.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.isfile(MODEL_PATH):
        return None

    try:
        return joblib.load(MODEL_PATH)

    except Exception as e:
        print("MODEL LOAD ERROR:", e)
        return None


model = load_model()


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

* {
    box-sizing: border-box;
}

html,
body,
[class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {

    background:
        linear-gradient(
            rgba(0,0,0,0.66),
            rgba(0,0,0,0.88)
        ),
        url("https://images.unsplash.com/photo-1555215695-3004980ad54e?auto=format&fit=crop&w=2200&q=90")
        center center / cover fixed no-repeat !important;

    color: white;
}

.block-container {

    max-width: 1400px !important;

    padding-top: 25px !important;
    padding-bottom: 35px !important;

    padding-left: 35px !important;
    padding-right: 35px !important;
}

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* ============================================================
BUTTONS
============================================================ */

div.stButton > button {

    width: 100%;
    min-height: 55px;

    border-radius: 15px;

    background:
        linear-gradient(
            135deg,
            #2588ff,
            #315ee9
        );

    color: white;

    font-size: 15px;
    font-weight: 800;

    border: 1px solid rgba(255,255,255,0.20);

    box-shadow:
        0 0 22px rgba(20,110,255,0.40),
        0 10px 25px rgba(0,0,0,0.45);

    transition: all 0.25s ease;
}

div.stButton > button:hover {

    transform: translateY(-3px);

    color: white;

    border-color: rgba(255,255,255,0.45);

    box-shadow:
        0 0 35px rgba(20,130,255,0.75),
        0 15px 30px rgba(0,0,0,0.55);
}


/* ============================================================
TOP BADGES
============================================================ */

.top-area {

    width: 100%;

    display: flex;

    justify-content: space-between;
    align-items: flex-start;

    margin-bottom: 12px;
}

.top-badge {

    min-width: 190px;

    padding: 14px 18px;

    background:
        rgba(7,7,7,0.72);

    border:
        1px solid rgba(255,255,255,0.22);

    border-radius: 15px;

    backdrop-filter: blur(14px);

    box-shadow:
        0 8px 25px rgba(0,0,0,0.40);
}

.badge-main {

    font-size: 14px;

    font-weight: 800;

    letter-spacing: 0.3px;
}

.badge-sub {

    color: #eeeeee;

    font-size: 12px;

    margin-top: 4px;
}

.blue {
    color: #2196ff;
}

.green {
    color: #35e29a;
}


/* ============================================================
HOME HERO
============================================================ */

.hero {

    text-align: center;

    margin-top: 8px;
}

.hero-icon {

    width: 94px;
    height: 94px;

    margin: 0 auto 10px auto;

    border-radius: 50%;

    display: flex;

    align-items: center;
    justify-content: center;

    border: 2px solid #188cff;

    background:
        radial-gradient(
            circle,
            rgba(0,110,255,0.30),
            rgba(0,0,0,0.58)
        );

    box-shadow:
        0 0 25px rgba(0,130,255,0.55);

    font-size: 43px;
}

.hero-title {

    font-size: clamp(58px, 7vw, 96px);

    font-weight: 900;

    letter-spacing: 1px;

    line-height: 0.95;

    color: white;

    text-shadow:
        0 5px 25px rgba(0,0,0,0.85);
}

.hero-title span {

    color: #237fff;

    text-shadow:
        0 0 20px rgba(20,120,255,0.45);
}

.hero-subtitle-row {

    display: flex;

    align-items: center;
    justify-content: center;

    gap: 18px;

    margin-top: 18px;
}

.hero-line {

    width: 135px;
    height: 2px;

    background: #258eff;
}

.hero-subtitle {

    color: #2491ff;

    font-size: 22px;

    font-weight: 800;
}

.hero-description {

    max-width: 700px;

    margin: 18px auto;

    color: #f1f1f1;

    font-size: 16px;

    line-height: 1.6;
}


/* ============================================================
FEATURE CARDS
============================================================ */

.features {

    display: grid;

    grid-template-columns:
        repeat(4,1fr);

    gap: 28px;

    max-width: 1100px;

    margin: 34px auto 28px auto;
}

.feature-card {

    min-height: 175px;

    padding: 22px 18px;

    text-align: center;

    background:
        rgba(8,8,8,0.68);

    border:
        1px solid rgba(255,255,255,0.23);

    border-radius: 16px;

    backdrop-filter: blur(13px);

    box-shadow:
        0 10px 30px rgba(0,0,0,0.42);

    transition: 0.25s ease;
}

.feature-card:hover {

    transform: translateY(-5px);

    border-color:
        rgba(50,145,255,0.60);
}

.feature-icon {

    height: 48px;

    display: flex;

    align-items: center;
    justify-content: center;

    font-size: 37px;

    margin-bottom: 9px;
}

.feature-title {

    color: white;

    font-size: 17px;

    font-weight: 800;

    margin-bottom: 9px;
}

.feature-text {

    color: #e4e4e4;

    font-size: 13px;

    line-height: 1.6;
}


/* ============================================================
HOW IT WORKS
============================================================ */

.how-title-row {

    display: flex;

    justify-content: center;
    align-items: center;

    gap: 20px;

    margin-top: 18px;
}

.how-line {

    width: 125px;
    height: 2px;

    background: #328aff;
}

.how-title {

    font-size: 30px;

    font-weight: 800;
}

.how-subtitle {

    text-align: center;

    margin-top: 5px;
    margin-bottom: 18px;

    color: #eeeeee;

    font-size: 14px;
}


/* ============================================================
STEPS
============================================================ */

.steps {

    display: grid;

    grid-template-columns:
        repeat(4,1fr);

    gap: 32px;

    max-width: 1220px;

    margin: 0 auto;
}

.step {

    position: relative;

    min-height: 142px;

    padding:
        13px 20px 16px 20px;

    background:
        rgba(8,8,8,0.67);

    border:
        1px solid rgba(255,255,255,0.24);

    border-radius: 15px;

    backdrop-filter: blur(12px);
}

.step-number {

    width: 43px;
    height: 43px;

    border-radius: 50%;

    margin:
        0 auto 8px auto;

    display: flex;

    justify-content: center;
    align-items: center;

    font-weight: 800;

    background:
        rgba(0,0,0,0.75);

    border:
        2px solid #2995ff;

    color: #2995ff;
}

.step-title {

    text-align: center;

    font-size: 15px;

    font-weight: 800;

    margin-bottom: 7px;
}

.step-text {

    text-align: center;

    color: #eeeeee;

    font-size: 12px;

    line-height: 1.55;
}

.step-arrow {

    position: absolute;

    right: -29px;

    top: 64px;

    font-size: 29px;

    color: #2995ff;
}


/* ============================================================
TRUST
============================================================ */

.trust {

    max-width: 820px;

    margin: 22px auto 10px auto;

    min-height: 88px;

    display: flex;

    align-items: center;
    justify-content: center;

    gap: 20px;

    padding: 15px 25px;

    background:
        rgba(8,8,8,0.72);

    border:
        1px solid rgba(255,255,255,0.25);

    border-radius: 15px;

    backdrop-filter: blur(12px);
}

.trust-icon {
    font-size: 46px;
}

.trust-title {

    color: #2492ff;

    font-size: 18px;

    font-weight: 800;

    margin-bottom: 6px;
}

.trust-text {

    color: white;

    font-size: 13px;
}


/* ============================================================
PREDICTION PAGE
============================================================ */

.prediction-container {

    max-width: 1050px;

    margin: auto;
}

.prediction-header {

    text-align: center;

    margin-bottom: 22px;
}

.prediction-icon {

    width: 75px;
    height: 75px;

    margin: 0 auto 12px auto;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 50%;

    border: 2px solid #188cff;

    background:
        radial-gradient(
            circle,
            rgba(0,110,255,0.30),
            rgba(0,0,0,0.55)
        );

    box-shadow:
        0 0 25px rgba(0,130,255,0.55);

    font-size: 34px;
}

.prediction-title {

    font-size: 42px;

    font-weight: 900;
}

.prediction-title span {
    color: #237fff;
}

.prediction-subtitle {

    color: #dddddd;

    font-size: 15px;

    margin-top: 8px;
}

.prediction-card {

    max-width: 1050px;

    margin: auto;

    background:
        rgba(5,8,12,0.84);

    border:
        1px solid rgba(255,255,255,0.20);

    border-radius: 22px;

    padding: 30px;

    backdrop-filter: blur(15px);

    box-shadow:
        0 15px 45px rgba(0,0,0,0.55);
}

.section-label {

    color: #2995ff;

    font-size: 18px;

    font-weight: 800;

    margin-bottom: 18px;
}


/* ============================================================
INPUTS
============================================================ */

label {

    color: #eeeeee !important;

    font-weight: 600 !important;
}

div[data-baseweb="select"] > div {

    background:
        rgba(255,255,255,0.08) !important;

    border:
        1px solid rgba(255,255,255,0.20) !important;

    border-radius: 12px !important;
}

div[data-baseweb="input"] > div {

    background:
        rgba(255,255,255,0.08) !important;

    border:
        1px solid rgba(255,255,255,0.20) !important;

    border-radius: 12px !important;
}

input {
    color: white !important;
}


/* ============================================================
CALCULATING PAGE
============================================================ */

.calculating-box {

    max-width: 800px;

    margin: 100px auto;

    padding: 55px 35px;

    text-align: center;

    background:
        rgba(5,8,12,0.84);

    border:
        1px solid rgba(35,127,255,0.45);

    border-radius: 25px;

    backdrop-filter: blur(15px);

    box-shadow:
        0 0 50px rgba(25,120,255,0.25);
}

.calculating-car {

    font-size: 75px;

    animation:
        carMove 1.2s infinite alternate;
}

@keyframes carMove {

    from {
        transform: translateX(-20px);
    }

    to {
        transform: translateX(20px);
    }
}

.calculating-title {

    margin-top: 20px;

    font-size: 30px;

    font-weight: 900;
}

.calculating-text {

    color: #cccccc;

    margin-top: 10px;

    font-size: 14px;
}


/* ============================================================
RESULT PAGE
============================================================ */

.result-page {

    max-width: 1000px;

    margin: auto;

    text-align: center;
}

.result-icon {

    font-size: 65px;

    margin-bottom: 10px;
}

.result-page-title {

    font-size: 44px;

    font-weight: 900;
}

.result-page-title span {
    color: #39e58b;
}

.result-card {

    margin: 15px auto 25px auto;

    padding: 40px;

    max-width: 700px;

    border-radius: 25px;

    background:
        linear-gradient(
            135deg,
            rgba(15,100,255,0.22),
            rgba(0,0,0,0.70)
        );

    border:
        1px solid rgba(35,127,255,0.60);

    box-shadow:
        0 0 40px rgba(25,120,255,0.30);
}

.result-label {

    color: #bbbbbb;

    font-size: 15px;

    letter-spacing: 1px;

    margin-bottom: 10px;
}

.result-price {

    color: #39e58b;

    font-size: 55px;

    font-weight: 900;

    text-shadow:
        0 0 20px rgba(50,220,130,0.30);
}

.result-note {

    color: #dddddd;

    font-size: 13px;

    margin-top: 10px;
}


/* ============================================================
SUMMARY
============================================================ */

.summary-card {

    max-width: 950px;

    margin: 25px auto;

    padding: 25px;

    background:
        rgba(5,8,12,0.78);

    border:
        1px solid rgba(255,255,255,0.18);

    border-radius: 18px;

    backdrop-filter: blur(12px);
}

.summary-title {

    color: #2995ff;

    font-size: 19px;

    font-weight: 800;

    margin-bottom: 16px;
}

.summary-text {

    color: white;

    font-size: 14px;

    line-height: 2;
}


/* ============================================================
RESPONSIVE
============================================================ */

@media (max-width: 950px) {

    .features,
    .steps {

        grid-template-columns:
            repeat(2,1fr);
    }

    .step-arrow {
        display: none;
    }
}

@media (max-width: 650px) {

    .block-container {

        padding-left: 15px !important;

        padding-right: 15px !important;
    }

    .top-area {

        gap: 10px;
    }

    .top-badge {

        min-width: 0;

        padding: 10px;
    }

    .hero-title {

        font-size: 48px;
    }

    .hero-subtitle {

        font-size: 15px;
    }

    .hero-line {

        width: 45px;
    }

    .features,
    .steps {

        grid-template-columns: 1fr;

        gap: 15px;
    }

    .prediction-title {

        font-size: 32px;
    }

    .prediction-card {

        padding: 20px;
    }

    .result-price {

        font-size: 40px;
    }
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HOME PAGE
# ============================================================

def show_home():

    st.html(
        """
        <div class="top-area">

            <div class="top-badge">

                <div class="badge-main">
                    🧠 <span class="blue">AI-POWERED</span>
                </div>

                <div class="badge-sub">
                    Vehicle Valuation
                </div>

            </div>

            <div class="top-badge">

                <div class="badge-main">
                    📈 <span class="green">XGBOOST MODEL</span>
                </div>

                <div class="badge-sub">
                    Machine Learning
                </div>

            </div>

        </div>
        """
    )

    st.html(
        """
        <div class="hero">

            <div class="hero-icon">
                🚗
            </div>

            <div class="hero-title">
                USED CAR <span>AI</span>
            </div>

            <div class="hero-subtitle-row">

                <div class="hero-line"></div>

                <div class="hero-subtitle">
                    Smart Used Car Price Prediction
                </div>

                <div class="hero-line"></div>

            </div>

            <div class="hero-description">
                Get an intelligent estimate of your used car's
                current market value using a trained XGBoost
                machine learning model.
            </div>

        </div>
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)

    start_col = st.columns([1, 2, 1])[1]

    with start_col:

        if st.button(
            "🚀  START PREDICTION",
            key="start_prediction",
            use_container_width=True
        ):

            st.session_state.page = "prediction"
            st.session_state.price_animation_done = False

            st.rerun()

    st.html(
        """
        <div class="features">

            <div class="feature-card">

                <div class="feature-icon">
                    🧠
                </div>

                <div class="feature-title">
                    XGBoost AI
                </div>

                <div class="feature-text">
                    Real machine learning<br>
                    price prediction.
                </div>

            </div>

            <div class="feature-card">

                <div class="feature-icon">
                    📊
                </div>

                <div class="feature-title">
                    Data Driven
                </div>

                <div class="feature-text">
                    Uses your trained<br>
                    vehicle dataset.
                </div>

            </div>

            <div class="feature-card">

                <div class="feature-icon">
                    🚘
                </div>

                <div class="feature-title">
                    Vehicle Analysis
                </div>

                <div class="feature-text">
                    Multiple vehicle<br>
                    characteristics.
                </div>

            </div>

            <div class="feature-card">

                <div class="feature-icon">
                    💰
                </div>

                <div class="feature-title">
                    Market Value
                </div>

                <div class="feature-text">
                    Get an estimated<br>
                    market price.
                </div>

            </div>

        </div>
        """
    )

    st.html(
        """
        <div class="how-title-row">

            <div class="how-line"></div>

            <div class="how-title">
                How It Works
            </div>

            <div class="how-line"></div>

        </div>

        <div class="how-subtitle">
            From vehicle information to AI-powered price prediction.
        </div>
        """
    )

    st.html(
        """
        <div class="steps">

            <div class="step">

                <div class="step-number">
                    01
                </div>

                <div class="step-title">
                    🚘 Enter Vehicle
                </div>

                <div class="step-text">
                    Enter your vehicle<br>
                    information.
                </div>

                <div class="step-arrow">
                    →
                </div>

            </div>

            <div class="step">

                <div class="step-number">
                    02
                </div>

                <div class="step-title">
                    📋 Add Details
                </div>

                <div class="step-text">
                    Provide engine,<br>
                    mileage and usage.
                </div>

                <div class="step-arrow">
                    →
                </div>

            </div>

            <div class="step">

                <div class="step-number">
                    03
                </div>

                <div class="step-title">
                    🧠 AI Analysis
                </div>

                <div class="step-text">
                    XGBoost analyzes<br>
                    your vehicle data.
                </div>

                <div class="step-arrow">
                    →
                </div>

            </div>

            <div class="step">

                <div class="step-number">
                    04
                </div>

                <div class="step-title">
                    💰 Get Price
                </div>

                <div class="step-text">
                    Receive the model's<br>
                    estimated value.
                </div>

            </div>

        </div>
        """
    )

    st.html(
        """
        <div class="trust">

            <div class="trust-icon">
                🛡️
            </div>

            <div>

                <div class="trust-title">
                    Machine Learning Based
                </div>

                <div class="trust-text">
                    Powered by your trained XGBoost regression pipeline.
                </div>

            </div>

        </div>
        """
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

def show_prediction():

    st.html(
        """
        <div class="prediction-header">

            <div class="prediction-icon">
                🚗
            </div>

            <div class="prediction-title">
                PRICE <span>PREDICTION</span>
            </div>

            <div class="prediction-subtitle">
                Enter your vehicle details to estimate its current market value.
            </div>

        </div>
        """
    )

    back_col = st.columns([1, 4, 1])[0]

    with back_col:

        if st.button(
            "← BACK TO HOME",
            key="back_home"
        ):

            st.session_state.page = "home"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="prediction-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-label">🚘 Vehicle Information</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # ROW 1
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        brand = st.selectbox(
            "Brand",
            [
                "Hyundai",
                "Maruti Suzuki",
                "Tata",
                "Mahindra",
                "Honda",
                "Toyota",
                "Kia",
                "Ford",
                "Volkswagen",
                "Renault",
                "Nissan",
                "Skoda",
                "MG",
                "Other"
            ],
            key="brand"
        )

    with col2:

        model_name = st.text_input(
            "Model",
            value="Creta",
            key="model_name"
        )


    # ========================================================
    # ROW 2
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        manufacturing_year = st.number_input(
            "Manufacturing Year",
            min_value=1990,
            max_value=2026,
            value=2021,
            step=1,
            key="manufacturing_year"
        )

    with col2:

        kilometers = st.number_input(
            "Kilometers Driven",
            min_value=0,
            max_value=1000000,
            value=45000,
            step=1000,
            key="kilometers"
        )


    # ========================================================
    # ROW 3
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        fuel_type = st.selectbox(
            "Fuel Type",
            [
                "Petrol",
                "Diesel",
                "CNG",
                "Electric",
                "Hybrid"
            ],
            key="fuel_type"
        )

    with col2:

        transmission = st.selectbox(
            "Transmission",
            [
                "Manual",
                "Automatic"
            ],
            key="transmission"
        )


    # ========================================================
    # ROW 4
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        seller_type = st.selectbox(
            "Seller Type",
            [
                "Individual",
                "Dealer",
                "Trustmark Dealer"
            ],
            key="seller_type"
        )

    with col2:

        seats = st.number_input(
            "Number of Seats",
            min_value=2,
            max_value=15,
            value=5,
            step=1,
            key="seats"
        )


    # ========================================================
    # ROW 5
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        mileage = st.number_input(
            "Mileage (km/l)",
            min_value=1.0,
            max_value=100.0,
            value=17.0,
            step=0.1,
            key="mileage"
        )

    with col2:

        engine = st.number_input(
            "Engine (CC)",
            min_value=500.0,
            max_value=10000.0,
            value=1497.0,
            step=1.0,
            key="engine"
        )


    # ========================================================
    # ROW 6
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        max_power = st.number_input(
            "Max Power (bhp)",
            min_value=20.0,
            max_value=1000.0,
            value=115.0,
            step=1.0,
            key="max_power"
        )

    with col2:

        owners = st.selectbox(
            "Number of Owners",
            [1, 2, 3, 4],
            key="owners"
        )


    st.markdown("<br>", unsafe_allow_html=True)

    predict_col = st.columns([1, 2, 1])[1]

    with predict_col:

        predict_clicked = st.button(
            "💰  START PREDICTION",
            key="predict_price",
            use_container_width=True
        )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_clicked:

        if model is None:

            st.error(
                "❌ XGBoost model not found.\n\n"
                f"Expected model location:\n"
                f"{MODEL_PATH}\n\n"
                "Please make sure the model file exists."
            )

            return


        if not model_name.strip():

            st.warning(
                "Please enter the vehicle model."
            )

            return


        # ====================================================
        # VEHICLE AGE
        # ====================================================

        current_year = datetime.now().year

        vehicle_age = (
            current_year -
            int(manufacturing_year)
        )

        vehicle_age = max(
            vehicle_age,
            0
        )


        # ====================================================
        # KM PER YEAR
        # ====================================================

        km_per_year = (
            kilometers /
            (vehicle_age + 1)
        )


        # ====================================================
        # INPUT DATA
        # ====================================================

        input_data = pd.DataFrame({

            "brand": [
                brand
            ],

            "model": [
                model_name.strip()
            ],

            "vehicle_age": [
                vehicle_age
            ],

            "km_driven": [
                kilometers
            ],

            "km_per_year": [
                km_per_year
            ],

            "seller_type": [
                seller_type
            ],

            "fuel_type": [
                fuel_type
            ],

            "transmission_type": [
                transmission
            ],

            "mileage": [
                mileage
            ],

            "engine": [
                engine
            ],

            "max_power": [
                max_power
            ],

            "seats": [
                seats
            ]

        })


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        try:

            prediction_result = model.predict(
                input_data
            )

            predicted_price = float(
                prediction_result[0]
            )

            predicted_price = max(
                predicted_price,
                0
            )

        except Exception as e:

            st.error(
                "❌ Prediction failed\n\n"
                f"{e}"
            )

            return


        # ====================================================
        # SAVE RESULT
        # ====================================================

        st.session_state.prediction = predicted_price

        st.session_state.vehicle_data = {

            "brand": brand,

            "model": model_name.strip(),

            "year": int(
                manufacturing_year
            ),

            "km": int(
                kilometers
            ),

            "fuel": fuel_type,

            "transmission": transmission,

            "seller": seller_type,

            "seats": int(
                seats
            ),

            "mileage": float(
                mileage
            ),

            "engine": float(
                engine
            ),

            "power": float(
                max_power
            ),

            "owners": int(
                owners
            ),

            "age": vehicle_age
        }

        st.session_state.price_animation_done = False

        st.session_state.page = "calculating"

        st.rerun()


# ============================================================
# CALCULATING PAGE
# ============================================================

def show_calculating():

    if st.session_state.prediction is None:

        st.session_state.page = "prediction"

        st.rerun()

        return


    st.html(
        """
        <div class="calculating-box">

            <div class="calculating-car">
                🚗
            </div>

            <div class="calculating-title">
                AI IS ANALYZING YOUR CAR
            </div>

            <div class="calculating-text">
                Processing your vehicle information
                with the trained XGBoost model...
            </div>

        </div>
        """
    )


    progress = st.progress(0)

    status = st.empty()


    messages = [

        "🔍 Reading vehicle information...",

        "📊 Processing vehicle specifications...",

        "🧠 Running XGBoost prediction...",

        "📈 Estimating market value...",

        "💰 Finalizing estimated price..."

    ]


    for i, message in enumerate(messages):

        status.markdown(
            f"""
            <div style="
                text-align:center;
                color:#2995ff;
                font-weight:700;
                margin-top:15px;
            ">
                {message}
            </div>
            """,
            unsafe_allow_html=True
        )

        progress.progress(
            int(
                ((i + 1) /
                len(messages)) *
                100
            )
        )

        time.sleep(0.55)


    st.session_state.page = "result"

    st.session_state.price_animation_done = False

    st.rerun()


# ============================================================
# RESULT PAGE
# ============================================================

def show_result():

    prediction = st.session_state.prediction

    vehicle = st.session_state.vehicle_data


    if prediction is None or vehicle is None:

        st.session_state.page = "prediction"

        st.rerun()

        return


    # ========================================================
    # HEADER
    # ========================================================

    st.html(
        """
        <div class="result-page">

            <div class="result-icon">
                🚗
            </div>

            <div class="result-page-title">
                YOUR <span>CAR VALUE</span>
            </div>

            <div style="
                color:#dddddd;
                margin-top:8px;
                font-size:15px;
            ">
                XGBoost estimated market value
            </div>

        </div>
        """
    )


    # ========================================================
    # EXACT MODEL PREDICTION
    # ========================================================

    final_price = float(prediction)

    final_lakhs = final_price / 100000


    # ========================================================
    # REAL BROWSER PRICE ANIMATION
    # ========================================================

    components.html(
        f"""
<!DOCTYPE html>

<html>

<head>

<style>

* {{
    box-sizing: border-box;
}}

html,
body {{
    margin: 0;
    padding: 0;
    background: transparent;
    font-family: Arial, sans-serif;
    overflow: hidden;
}}

.result-card {{
    margin: 15px auto 25px auto;

    padding: 40px;

    max-width: 700px;

    border-radius: 25px;

    background:
        linear-gradient(
            135deg,
            rgba(15,100,255,0.22),
            rgba(0,0,0,0.70)
        );

    border:
        1px solid rgba(35,127,255,0.60);

    box-shadow:
        0 0 40px rgba(25,120,255,0.30);

    text-align: left;
}}

.result-label {{
    color: #bbbbbb;

    font-size: 15px;

    letter-spacing: 1px;

    margin-bottom: 10px;
}}

.result-price {{
    color: #39e58b;

    font-size: 55px;

    font-weight: 900;

    line-height: 1.2;

    white-space: nowrap;

    text-shadow:
        0 0 20px
        rgba(50,220,130,0.30);

    transition:
        transform 0.15s ease;
}}

.price-counting {{
    animation:
        priceGlow 0.8s
        ease-in-out
        infinite alternate;
}}

@keyframes priceGlow {{

    from {{
        transform: scale(1);

        text-shadow:
            0 0 10px
            rgba(57,229,139,0.20);
    }}

    to {{
        transform: scale(1.03);

        text-shadow:
            0 0 30px
            rgba(57,229,139,0.55);
    }}
}}

.result-note {{
    color: #dddddd;

    font-size: 13px;

    margin-top: 10px;
}}

@media (max-width: 650px) {{

    .result-card {{
        padding: 25px;
    }}

    .result-price {{
        font-size: 40px;
    }}

}}

</style>

</head>


<body>


<div class="result-card">

    <div class="result-label">
        ESTIMATED MARKET VALUE
    </div>

    <div
        id="price"
        class="result-price price-counting"
    >
        ₹ 0.00 Lakhs
    </div>

    <div class="result-note">
        AI-based estimated value for your vehicle
    </div>

</div>


<script>

(function() {{

    const priceElement =
        document.getElementById("price");


    const target =
        {final_lakhs:.10};


    const duration =
        2500;


    const startTime =
        performance.now();


    function animate(currentTime) {{

        const elapsed =
            currentTime - startTime;


        let progress =
            elapsed / duration;


        if (progress > 1) {{
            progress = 1;
        }}


        /*
         * Smooth ease-out.
         */

        const eased =
            1 -
            Math.pow(
                1 - progress,
                3
            );


        const currentPrice =
            target * eased;


        priceElement.textContent =
            "₹ " +
            currentPrice.toFixed(2) +
            " Lakhs";


        if (progress < 1) {{

            requestAnimationFrame(
                animate
            );

        }} else {{

            /*
             * Force the exact final
             * model prediction.
             */

            priceElement.textContent =
                "₹ " +
                target.toFixed(2) +
                " Lakhs";


            priceElement.classList.remove(
                "price-counting"
            );

        }}

    }}


    /*
     * GUARANTEED START VALUE
     */

    priceElement.textContent =
        "₹ 0.00 Lakhs";


    /*
     * START ANIMATION
     */

    requestAnimationFrame(
        animate
    );

}})();

</script>


</body>

</html>
        """,
        height=230,
        scrolling=False
    )


    # ========================================================
    # VEHICLE SUMMARY
    # ========================================================

    st.html(
        f"""
        <div class="summary-card">

            <div class="summary-title">
                🚘 Vehicle Summary
            </div>

            <div class="summary-text">

                <b>
                    {vehicle["brand"]} {vehicle["model"]}
                </b>

                &nbsp; • &nbsp;

                {vehicle["year"]}

                &nbsp; • &nbsp;

                {vehicle["km"]:,} km

                &nbsp; • &nbsp;

                {vehicle["fuel"]}

                &nbsp; • &nbsp;

                {vehicle["transmission"]}

                &nbsp; • &nbsp;

                {vehicle["owners"]} Owner

                &nbsp; • &nbsp;

                {vehicle["seller"]}

            </div>

        </div>
        """
    )


    # ========================================================
    # AI ANALYSIS
    # ========================================================

    st.html(
        f"""
        <div class="summary-card">

            <div class="summary-title">
                🧠 AI Analysis
            </div>

            <div class="summary-text">

                Vehicle Age:
                <b>{vehicle["age"]} years</b>

                &nbsp; • &nbsp;

                Mileage:
                <b>{vehicle["mileage"]:.1f} km/l</b>

                &nbsp; • &nbsp;

                Engine:
                <b>{vehicle["engine"]:.0f} CC</b>

                &nbsp; • &nbsp;

                Power:
                <b>{vehicle["power"]:.0f} bhp</b>

                &nbsp; • &nbsp;

                Seats:
                <b>{vehicle["seats"]}</b>

            </div>

        </div>
        """
    )


    # ========================================================
    # BUTTONS
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(
        [1, 1.2, 1]
    )


    # ========================================================
    # EDIT DETAILS
    # ========================================================

    with col1:

        if st.button(
            "← EDIT DETAILS",
            key="edit_details",
            use_container_width=True
        ):

            st.session_state.page = "prediction"

            st.session_state.price_animation_done = False

            st.rerun()


    # ========================================================
    # NEW PREDICTION
    # ========================================================

    with col2:

        if st.button(
            "🔄 NEW PREDICTION",
            key="new_prediction",
            use_container_width=True
        ):

            st.session_state.prediction = None

            st.session_state.vehicle_data = None

            st.session_state.price_animation_done = False

            st.session_state.page = "prediction"

            st.rerun()


    # ========================================================
    # HOME
    # ========================================================

    with col3:

        if st.button(
            "🏠 HOME",
            key="result_home",
            use_container_width=True
        ):

            st.session_state.prediction = None

            st.session_state.vehicle_data = None

            st.session_state.price_animation_done = False

            st.session_state.page = "home"

            st.rerun()


# ============================================================
# PAGE ROUTER
# ============================================================

if st.session_state.page == "home":

    show_home()

elif st.session_state.page == "prediction":

    show_prediction()

elif st.session_state.page == "calculating":

    show_calculating()

elif st.session_state.page == "result":

    show_result()
