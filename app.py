import streamlit as st
import pandas as pd
import re

from src.pipeline import analyze_text, analyze_dataframe
from src.ingest import extract_upload
from src.model_store import load_sentiment_model, MODEL_PATH


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CMASSIP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# SESSION STATE
# ============================================================

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

if "page" not in st.session_state:
    st.session_state.page = "Overview"

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "batch" not in st.session_state:
    st.session_state.batch = None

if "batch_name" not in st.session_state:
    st.session_state.batch_name = None

if "batch_signature" not in st.session_state:
    st.session_state.batch_signature = None


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def get_model():
    return load_sentiment_model()


# ============================================================
# THEME
# ============================================================

dark = st.session_state.theme == "Dark"

if dark:

    BG = "#080D18"
    SURFACE = "#111827"
    SURFACE_2 = "#182235"
    SURFACE_3 = "#202C42"

    TEXT = "#F8FAFC"
    MUTED = "#9CA9BD"

    BORDER = "#2B3952"

    ACCENT = "#8B7CFF"
    ACCENT_BG = "#29234F"

    SUCCESS_BG = "#102C24"
    SUCCESS_BORDER = "#23644D"

else:

    BG = "#F5F7FB"
    SURFACE = "#FFFFFF"
    SURFACE_2 = "#F8FAFC"
    SURFACE_3 = "#EEF2F7"

    TEXT = "#172033"
    MUTED = "#667085"

    BORDER = "#D8DFEA"

    ACCENT = "#5B4FE9"
    ACCENT_BG = "#EEECFF"

    SUCCESS_BG = "#ECFDF5"
    SUCCESS_BORDER = "#A7F3D0"


# ============================================================
# CSS
# ============================================================

st.markdown(
    f"""
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    html,
    body,
    [data-testid="stAppViewContainer"],
    [data-testid="stApp"] {{
        background: {BG} !important;
    }}

    .stApp {{
        background: {BG} !important;
        color: {TEXT} !important;
    }}

    [data-testid="stHeader"] {{
        height: 0 !important;
        background: transparent !important;
    }}

    section[data-testid="stSidebar"] {{
        display: none !important;
    }}

    .main .block-container {{
        max-width: 1380px !important;
        padding-top: 0.7rem !important;
        padding-bottom: 3rem !important;
    }}


    /* ======================================================
       TEXT
       ====================================================== */

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6 {{
        color: {TEXT} !important;
    }}

    .stApp p,
    .stApp li,
    .stApp label {{
        color: {TEXT};
    }}

    [data-testid="stCaptionContainer"] p {{
        color: {MUTED} !important;
    }}


    /* ======================================================
       TOP NAV
       ====================================================== */

    .nav-title {{
        color: {TEXT};
        font-size: 1.45rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.1;
    }}

    .nav-subtitle {{
        color: {MUTED};
        font-size: 0.72rem;
        margin-top: 3px;
    }}

    .nav-divider {{
        margin-top: 0.45rem;
        margin-bottom: 1.3rem;
        border-bottom: 1px solid {BORDER};
    }}


    /* ======================================================
       NAV BUTTONS
       ====================================================== */

    div[data-testid="stHorizontalBlock"] .nav-button button {{
        width: 100%;
    }}

    .stButton > button {{
        background: {SURFACE} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 9px !important;
        min-height: 2.5rem !important;
        font-weight: 650 !important;
        box-shadow: none !important;
    }}

    .stButton > button:hover {{
        border-color: {ACCENT} !important;
        color: {ACCENT} !important;
        background: {SURFACE_2} !important;
    }}


    /* ======================================================
       CONTAINERS / CARDS
       ====================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 15px !important;
    }}


    /* ======================================================
       TEXT INPUT
       ====================================================== */

    .stTextInput input {{
        background: {SURFACE_2} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 9px !important;
    }}

    .stTextInput input:focus {{
        border-color: {ACCENT} !important;
        box-shadow: 0 0 0 1px {ACCENT} !important;
    }}

    .stTextInput input::placeholder {{
        color: {MUTED} !important;
        opacity: 1 !important;
    }}


    /* ======================================================
       TEXT AREA
       ====================================================== */

    .stTextArea textarea {{
        background: {SURFACE_2} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
    }}

    .stTextArea textarea:focus {{
        border-color: {ACCENT} !important;
        box-shadow: 0 0 0 1px {ACCENT} !important;
    }}

    .stTextArea textarea::placeholder {{
        color: {MUTED} !important;
        opacity: 1 !important;
    }}


    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    [data-testid="stFileUploader"] {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 12px !important;
        padding: 0.8rem !important;
    }}

    [data-testid="stFileUploaderDropzone"] {{
        background: {SURFACE_2} !important;
        border: 1px dashed {BORDER} !important;
        border-radius: 10px !important;
    }}

    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] div {{
        color: {TEXT} !important;
    }}

    [data-testid="stFileUploaderDropzoneInstructions"] {{
        color: {MUTED} !important;
    }}

    [data-testid="stFileUploaderDropzone"] button {{
        background: {SURFACE_3} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
    }}


    /* ======================================================
       SELECT BOX
       ====================================================== */

    [data-baseweb="select"] > div {{
        background: {SURFACE_2} !important;
        color: {TEXT} !important;
        border-color: {BORDER} !important;
    }}

    [data-baseweb="select"] span {{
        color: {TEXT} !important;
    }}


    /* ======================================================
       METRICS
       ====================================================== */

    [data-testid="stMetric"] {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 13px !important;
        padding: 0.85rem 1rem !important;
    }}

    [data-testid="stMetricLabel"] {{
        color: {MUTED} !important;
    }}

    [data-testid="stMetricValue"] {{
        color: {TEXT} !important;
    }}


    /* ======================================================
       TABS
       ====================================================== */

    .stTabs [data-baseweb="tab-list"] {{
        gap: 22px;
        border-bottom: 1px solid {BORDER};
    }}

    .stTabs [data-baseweb="tab"] {{
        color: {MUTED} !important;
    }}

    .stTabs [aria-selected="true"] {{
        color: {ACCENT} !important;
    }}


    /* ======================================================
       DATAFRAME
       ====================================================== */

    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
    }}


    /* ======================================================
       EXPANDER
       ====================================================== */

    [data-testid="stExpander"] {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 12px !important;
    }}


    /* ======================================================
       ALERTS
       ====================================================== */

    [data-testid="stAlert"] {{
        border-radius: 10px !important;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# NAVIGATION
# ============================================================

nav1, nav2, nav3, nav4, nav5, spacer, light_col, dark_col = st.columns(
    [1.65, 1.1, 1.35, 1.0, 1.0, 0.7, 0.8, 0.8],
    vertical_alignment="center",
)


# BRAND

with nav1:

    st.markdown(
        '<div class="nav-title">🧠 CMASSIP</div>'
        '<div class="nav-subtitle">Multilingual Language Intelligence</div>',
        unsafe_allow_html=True,
    )


# NAVIGATION BUTTONS

with nav2:

    if st.button(
        "🏠 Overview",
        use_container_width=True,
        type=(
            "primary"
            if st.session_state.page == "Overview"
            else "secondary"
        ),
    ):
        st.session_state.page = "Overview"
        st.rerun()


with nav3:

    if st.button(
        "🔍 Analyze",
        use_container_width=True,
        type=(
            "primary"
            if st.session_state.page == "Analyze"
            else "secondary"
        ),
    ):
        st.session_state.page = "Analyze"
        st.rerun()


with nav4:

    if st.button(
        "📊 Batch",
        use_container_width=True,
        type=(
            "primary"
            if st.session_state.page == "Batch Analysis"
            else "secondary"
        ),
    ):
        st.session_state.page = "Batch Analysis"
        st.rerun()


with nav5:

    if st.button(
        "📈 Insights",
        use_container_width=True,
        type=(
            "primary"
            if st.session_state.page == "Insights"
            else "secondary"
        ),
    ):
        st.session_state.page = "Insights"
        st.rerun()


with spacer:
    st.write("")


with light_col:

    if st.button(
        "☀️",
        help="Light mode",
        use_container_width=True,
    ):

        st.session_state.theme = "Light"
        st.rerun()


with dark_col:

    if st.button(
        "🌙",
        help="Dark mode",
        use_container_width=True,
    ):

        st.session_state.theme = "Dark"
        st.rerun()


st.markdown(
    '<div class="nav-divider"></div>',
    unsafe_allow_html=True,
)


# ============================================================
# REVIEW SPLITTER
# ============================================================

def split_document_into_reviews(text):

    if not text:
        return []

    text = (
        str(text)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u00a0", " ")
        .strip()
    )

    # Handles:
    # Review 01 — English
    # Review 01 - English
    # Review 01: English
    # Review 01 | English
    # ### Review 01 — English
    # Review No. 01
    # Review #01

    pattern = re.compile(
        r"(?im)"
        r"^[ \t#>*_-]*"
        r"review"
        r"(?:[ \t]+(?:no\.?|number))?"
        r"[ \t]*#?"
        r"[ \t]*(\d+)"
        r"[ \t]*"
        r"(?:[-–—:|]+[ \t]*[A-Za-z][^\n]*)?"
        r"[ \t]*$"
    )

    matches = list(
        pattern.finditer(text)
    )

    reviews = []

    if matches:

        for i, match in enumerate(matches):

            number = int(
                match.group(1)
            )

            start = match.end()

            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(text)

            review_text = text[
                start:end
            ].strip()

            review_text = re.sub(
                r"^[#>*_\-\s]+",
                "",
                review_text,
            ).strip()

            if review_text:

                reviews.append(
                    {
                        "review_id": number,
                        "text": review_text,
                    }
                )

        return reviews

    # Fallback for documents without headings

    paragraphs = [
        p.strip()
        for p in re.split(
            r"\n\s*\n",
            text,
        )
        if p.strip()
    ]

    return [
        {
            "review_id": i,
            "text": paragraph,
        }
        for i, paragraph in enumerate(
            paragraphs,
            start=1,
        )
    ]


# ============================================================
# BATCH DOCUMENT ANALYSIS
# ============================================================

def analyze_document_reviews(
    text,
    target=None,
    model=None,
):

    reviews = split_document_into_reviews(
        text
    )

    rows = []

    for review in reviews:

        result = analyze_text(
            review["text"],
            target=target or None,
            model=model,
        )

        stance = result.get(
            "stance",
            {}
        ) or {}

        aspects = result.get(
            "aspects",
            []
        ) or []

        aspect_names = []
        aspect_sentiments = []

        for aspect in aspects:

            name = str(
                aspect.get(
                    "aspect",
                    "",
                )
            ).strip()

            if not name:
                continue

            aspect_names.append(name)

            aspect_sentiments.append(
                f"{name}="
                f"{aspect.get('sentiment', 'Neutral')}"
            )

        rows.append(
            {
                "Review": review["review_id"],
                "Language": result.get(
                    "language",
                    "unknown",
                ),
                "Sentiment": result.get(
                    "overall_sentiment",
                    "Neutral",
                ),
                "Confidence": round(
                    float(
                        result.get(
                            "overall_confidence",
                            0,
                        )
                    ),
                    3,
                ),
                "Valence": round(
                    float(
                        result.get(
                            "valence",
                            0,
                        )
                    ),
                    3,
                ),
                "Arousal": round(
                    float(
                        result.get(
                            "arousal",
                            0,
                        )
                    ),
                    3,
                ),
                "Stance": stance.get(
                    "label",
                    "Unknown",
                ),
                "Stance Confidence": round(
                    float(
                        stance.get(
                            "confidence",
                            0,
                        )
                    ),
                    3,
                ),
                "Aspects": "; ".join(
                    aspect_names
                ),
                "Aspect Sentiments": "; ".join(
                    aspect_sentiments
                ),
                "Evidence": " | ".join(
                    result.get(
                        "evidence",
                        [],
                    )
                ),
                "Review Text": result.get(
                    "text",
                    review["text"],
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# OVERVIEW
# ============================================================

if st.session_state.page == "Overview":

    st.caption(
        "AI-DRIVEN LANGUAGE TECHNOLOGIES"
    )

    st.title(
        "From opinions to intelligence."
    )

    st.write(
        "CMASSIP continuously analyzes multilingual "
        "opinions at aspect level, identifies target-aware "
        "stance, and provides evidence and affect signals."
    )

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        with st.container(border=True):

            st.subheader("🌐")
            st.markdown("### Multilingual")
            st.caption(
                "Automatic language detection "
                "and multilingual processing."
            )

    with c2:

        with st.container(border=True):

            st.subheader("🎯")
            st.markdown("### Aspect Sentiment")
            st.caption(
                "Understand sentiment toward "
                "specific aspects."
            )

    with c3:

        with st.container(border=True):

            st.subheader("🧭")
            st.markdown("### Target Stance")
            st.caption(
                "Target-aware stance analysis."
            )

    with c4:

        with st.container(border=True):

            st.subheader("📐")
            st.markdown("### Affect")
            st.caption(
                "Valence and arousal signals."
            )

    st.header(
        "How CMASSIP works"
    )

    steps = st.columns(5)

    workflow = [
        ("01", "Input", "Text or document"),
        ("02", "Language", "Detect language"),
        ("03", "Aspects", "Find opinions"),
        ("04", "Stance", "Analyze target"),
        ("05", "Evidence", "Ground output"),
    ]

    for column, item in zip(
        steps,
        workflow,
    ):

        with column:

            with st.container(border=True):

                st.caption(item[0])

                st.markdown(
                    f"**{item[1]}**"
                )

                st.caption(
                    item[2]
                )


# ============================================================
# ANALYZE
# ============================================================

elif st.session_state.page == "Analyze":

    st.title(
        "Analyze"
    )

    st.caption(
        "Analyze individual text or documents."
    )

    text_tab, document_tab = st.tabs(
        [
            "✍️ Text",
            "📄 Document",
        ]
    )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    with text_tab:

        text = st.text_area(
            "Enter text",
            height=170,
            placeholder=(
                "Example: The camera is excellent, "
                "but the battery drains very quickly."
            ),
        )

        target = st.text_input(
            "Optional stance target",
            placeholder="Example: smartphone",
        )

        if st.button(
            "Analyze Text",
            type="primary",
            use_container_width=True,
            disabled=not text.strip(),
        ):

            with st.spinner(
                "Analyzing..."
            ):

                st.session_state.last_result = analyze_text(
                    text,
                    target=target or None,
                    model=get_model(),
                )

    # --------------------------------------------------------
    # DOCUMENT
    # --------------------------------------------------------

    with document_tab:

        st.subheader(
            "Upload document"
        )

        st.caption(
            "PDF • DOCX • DOC • TXT"
        )

        uploaded = st.file_uploader(
            "Choose document",
            type=[
                "pdf",
                "docx",
                "doc",
                "txt",
            ],
            key="single_document",
        )

        document_target = st.text_input(
            "Optional stance target",
            placeholder="Example: smartphone",
            key="document_target",
        )

        if uploaded:

            st.success(
                f"Selected: {uploaded.name}"
            )

            if st.button(
                "Analyze Document",
                type="primary",
                use_container_width=True,
            ):

                with st.spinner(
                    "Analyzing document..."
                ):

                    text = extract_upload(
                        uploaded
                    )

                    st.session_state.last_result = analyze_text(
                        text,
                        target=document_target or None,
                        model=get_model(),
                    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if st.session_state.last_result:

        r = st.session_state.last_result

        st.divider()

        st.header(
            "Analysis Result"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Language",
            r.get(
                "language",
                "Unknown",
            ),
        )

        c2.metric(
            "Overall Sentiment",
            r.get(
                "overall_sentiment",
                "Unknown",
            ),
        )

        c3.metric(
            "Valence",
            f"{float(r.get('valence', 0)):.2f}",
        )

        c4.metric(
            "Arousal",
            f"{float(r.get('arousal', 0)):.2f}",
        )

        st.subheader(
            "Aspect-Based Sentiment Analysis"
        )

        st.caption(
            "ABSA engine: "
            + str(
                r.get(
                    "absa_mode",
                    "reference",
                )
            )
        )

        if r.get("aspects"):

            st.dataframe(
                pd.DataFrame(
                    r["aspects"]
                ),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No reliable aspect-opinion pairs "
                "were extracted."
            )

        left, right = st.columns(2)

        with left:

            st.subheader(
                "🎯 Target-aware Stance"
            )

            stance = r.get(
                "stance",
                {},
            ) or {}

            st.write(
                "**Target:** "
                + str(
                    stance.get(
                        "target"
                    )
                    or "Not specified"
                )
            )

            confidence = stance.get(
                "confidence"
            )

            if confidence is not None:

                st.write(
                    "**Stance:** "
                    + str(
                        stance.get(
                            "label",
                            "Unknown",
                        )
                    )
                    + f" ({float(confidence):.0%})"
                )

            else:

                st.write(
                    "**Stance:** "
                    + str(
                        stance.get(
                            "label",
                            "Unknown",
                        )
                    )
                )

            if stance.get("method"):

                st.caption(
                    stance["method"]
                )

        with right:

            st.subheader(
                "🔎 Evidence"
            )

            evidence = r.get(
                "evidence",
                [],
            )

            if evidence:

                for item in evidence:

                    st.write(
                        f"• {item}"
                    )

            else:

                st.info(
                    "No aspect-level evidence available."
                )

        st.subheader(
            "📐 Dimensional Affect"
        )

        valence = float(
            r.get(
                "valence",
                0,
            )
        )

        arousal = float(
            r.get(
                "arousal",
                0,
            )
        )

        st.progress(
            max(
                0.0,
                min(
                    1.0,
                    valence,
                ),
            ),
            text=f"Valence {valence:.2f}",
        )

        st.progress(
            max(
                0.0,
                min(
                    1.0,
                    arousal,
                ),
            ),
            text=f"Arousal {arousal:.2f}",
        )

        with st.expander(
            "Model / Method Information"
        ):

            st.write(
                r.get(
                    "model_note",
                    "",
                )
            )


# ============================================================
# BATCH ANALYSIS
# ============================================================

elif st.session_state.page == "Batch Analysis":

    st.title(
        "Batch Analysis"
    )

    st.caption(
        "Analyze multiple reviews independently."
    )

    # --------------------------------------------------------
    # UPLOAD CARD
    # --------------------------------------------------------

    with st.container(border=True):

        st.subheader(
            "📁 Upload batch file"
        )

        st.write(
            "Upload a document containing multiple reviews "
            "or a CSV/XLSX dataset."
        )

        st.caption(
            "Supported: PDF • DOCX • DOC • TXT • CSV • XLSX"
        )

        batch_file = st.file_uploader(
            "Choose batch file",
            type=[
                "pdf",
                "docx",
                "doc",
                "txt",
                "csv",
                "xlsx",
            ],
            key="batch_uploader",
        )

    # --------------------------------------------------------
    # FILE
    # --------------------------------------------------------

    if batch_file:

        signature = (
            batch_file.name,
            batch_file.size,
        )

        if (
            st.session_state.batch_signature
            != signature
        ):

            st.session_state.batch = None
            st.session_state.batch_name = None
            st.session_state.batch_signature = signature

        st.success(
            f"Selected: {batch_file.name}"
        )

        filename = batch_file.name.lower()

        # ====================================================
        # DOCUMENT
        # ====================================================

        if filename.endswith(
            (
                ".pdf",
                ".docx",
                ".doc",
                ".txt",
            )
        ):

            try:

                document_text = extract_upload(
                    batch_file
                )

                reviews = split_document_into_reviews(
                    document_text
                )

                st.subheader(
                    "Document detected"
                )

                count_col, info_col = st.columns(
                    [1, 3]
                )

                with count_col:

                    st.metric(
                        "Reviews detected",
                        len(reviews),
                    )

                with info_col:

                    if len(reviews) > 1:

                        st.success(
                            f"{len(reviews)} reviews "
                            "are ready for independent analysis."
                        )

                    elif len(reviews) == 1:

                        st.warning(
                            "Only one review was detected. "
                            "Check that your document uses headings "
                            "such as 'Review 01 — English'."
                        )

                    else:

                        st.error(
                            "No review text was detected."
                        )

                # ------------------------------------------------
                # PREVIEW
                # ------------------------------------------------

                if reviews:

                    with st.expander(
                        "Preview detected reviews"
                    ):

                        preview = []

                        for review in reviews:

                            preview.append(
                                {
                                    "Review": review[
                                        "review_id"
                                    ],
                                    "Length": len(
                                        review["text"]
                                    ),
                                    "Preview": (
                                        review["text"][:160]
                                        + (
                                            "..."
                                            if len(
                                                review["text"]
                                            ) > 160
                                            else ""
                                        )
                                    ),
                                }
                            )

                        st.dataframe(
                            pd.DataFrame(
                                preview
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

                target = st.text_input(
                    "Optional stance target",
                    placeholder="Example: smartphone",
                    key="batch_document_target",
                )

                if st.button(
                    "🚀 Analyze All Reviews",
                    type="primary",
                    use_container_width=True,
                    disabled=not reviews,
                ):

                    with st.spinner(
                        f"Analyzing {len(reviews)} "
                        "reviews independently..."
                    ):

                        result = analyze_document_reviews(
                            document_text,
                            target=target or None,
                            model=get_model(),
                        )

                        st.session_state.batch = result
                        st.session_state.batch_name = (
                            batch_file.name
                        )

                        st.success(
                            f"Analysis completed for "
                            f"{len(result)} reviews."
                        )

            except Exception as exc:

                st.error(
                    "Document processing failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        # ====================================================
        # CSV / XLSX
        # ====================================================

        else:

            try:

                if filename.endswith(".csv"):

                    df = pd.read_csv(
                        batch_file
                    )

                else:

                    df = pd.read_excel(
                        batch_file
                    )

                st.subheader(
                    "Dataset preview"
                )

                st.dataframe(
                    df.head(10),
                    use_container_width=True,
                    hide_index=True,
                )

                text_col = st.selectbox(
                    "Text column",
                    list(df.columns),
                    key="dataset_text_col",
                )

                target_col = st.selectbox(
                    "Optional target column",
                    ["None"] + list(df.columns),
                    key="dataset_target_col",
                )

                if st.button(
                    "🚀 Analyze Dataset",
                    type="primary",
                    use_container_width=True,
                ):

                    with st.spinner(
                        f"Analyzing {len(df)} records..."
                    ):

                        result = analyze_dataframe(
                            df,
                            text_col,
                            (
                                None
                                if target_col == "None"
                                else target_col
                            ),
                            get_model(),
                        )

                        st.session_state.batch = result
                        st.session_state.batch_name = (
                            batch_file.name
                        )

                        st.success(
                            f"Analysis completed for "
                            f"{len(result)} records."
                        )

            except Exception as exc:

                st.error(
                    "Dataset processing failed: "
                    f"{type(exc).__name__}: {exc}"
                )

    # ========================================================
    # RESULTS
    # ========================================================

    if st.session_state.batch is not None:

        result = st.session_state.batch

        st.divider()

        st.header(
            "Batch Results"
        )

        if result.empty:

            st.warning(
                "No results were generated."
            )

        else:

            sentiment_col = (
                "Sentiment"
                if "Sentiment" in result.columns
                else "sentiment"
            )

            language_col = (
                "Language"
                if "Language" in result.columns
                else "language"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Records",
                len(result),
            )

            c2.metric(
                "Languages",
                result[
                    language_col
                ].nunique(),
            )

            c3.metric(
                "Positive",
                int(
                    (
                        result[
                            sentiment_col
                        ] == "Positive"
                    ).sum()
                ),
            )

            c4.metric(
                "Negative",
                int(
                    (
                        result[
                            sentiment_col
                        ] == "Negative"
                    ).sum()
                ),
            )

            st.subheader(
                "Results table"
            )

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True,
                height=450,
            )

            st.download_button(
                "⬇️ Download Results CSV",
                result.to_csv(
                    index=False
                ).encode(
                    "utf-8-sig"
                ),
                "cmassip_results.csv",
                "text/csv",
                use_container_width=True,
            )

            st.write("")

            chart1, chart2 = st.columns(2)

            with chart1:

                st.subheader(
                    "Sentiment Distribution"
                )

                st.bar_chart(
                    result[
                        sentiment_col
                    ].value_counts()
                )

            with chart2:

                st.subheader(
                    "Language Distribution"
                )

                st.bar_chart(
                    result[
                        language_col
                    ].value_counts()
                )

            if st.button(
                "Clear Batch Results",
                use_container_width=True,
            ):

                st.session_state.batch = None
                st.session_state.batch_name = None
                st.rerun()


# ============================================================
# INSIGHTS
# ============================================================

elif st.session_state.page == "Insights":

    st.title(
        "Insights"
    )

    st.caption(
        "Explore patterns across your analyzed batch."
    )

    if st.session_state.batch is None:

        st.info(
            "No batch results yet. "
            "Go to Batch Analysis and analyze a file first."
        )

    else:

        df = st.session_state.batch

        if df.empty:

            st.warning(
                "There are no records to analyze."
            )

        else:

            sentiment_col = (
                "Sentiment"
                if "Sentiment" in df.columns
                else "sentiment"
            )

            language_col = (
                "Language"
                if "Language" in df.columns
                else "language"
            )

            confidence_col = (
                "Confidence"
                if "Confidence" in df.columns
                else "confidence"
            )

            stance_col = (
                "Stance"
                if "Stance" in df.columns
                else "stance"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Records",
                len(df),
            )

            c2.metric(
                "Languages",
                df[
                    language_col
                ].nunique(),
            )

            c3.metric(
                "Positive",
                int(
                    (
                        df[
                            sentiment_col
                        ] == "Positive"
                    ).sum()
                ),
            )

            c4.metric(
                "Negative",
                int(
                    (
                        df[
                            sentiment_col
                        ] == "Negative"
                    ).sum()
                ),
            )

            st.divider()

            left, right = st.columns(2)

            with left:

                st.subheader(
                    "Sentiment"
                )

                st.bar_chart(
                    df[
                        sentiment_col
                    ].value_counts()
                )

            with right:

                st.subheader(
                    "Languages"
                )

                st.bar_chart(
                    df[
                        language_col
                    ].value_counts()
                )

            if confidence_col in df.columns:

                confidence = pd.to_numeric(
                    df[
                        confidence_col
                    ],
                    errors="coerce",
                ).mean()

                if pd.notna(confidence):

                    st.subheader(
                        "Average Confidence"
                    )

                    st.progress(
                        max(
                            0.0,
                            min(
                                1.0,
                                float(confidence),
                            ),
                        ),
                        text=f"{confidence:.1%}",
                    )

            if stance_col in df.columns:

                st.subheader(
                    "Stance"
                )

                st.bar_chart(
                    df[
                        stance_col
                    ].value_counts()
                )


# ============================================================
# PROJECT
# ============================================================

elif st.session_state.page == "Project":

    st.title(
        "CMASSIP Project"
    )

    st.caption(
        "Continuous Multilingual Aspect Sentiment "
        "& Stance Intelligence Platform"
    )

    with st.container(border=True):

        st.header(
            "Project Overview"
        )

        st.write(
            "CMASSIP is an AI-driven language technology "
            "platform designed for multilingual aspect "
            "sentiment, target-aware stance, valence, "
            "arousal and evidence analysis."
        )

    st.header(
        "Project Information"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        with st.container(border=True):

            st.caption("DOMAIN")

            st.write(
                "AI-Driven Language Technologies"
            )

    with c2:

        with st.container(border=True):

            st.caption("CORE")

            st.write(
                "ABSA + Stance + Affect"
            )

    with c3:

        with st.container(border=True):

            st.caption("INPUT")

            st.write(
                "Multilingual Text & Documents"
            )

    st.header(
        "Research Objectives"
    )

    objectives = [
        (
            "O1",
            "Problem & Engineering Contract",
            "Define multilingual aspect, opinion, holder, "
            "stance, valence and arousal requirements.",
        ),
        (
            "O2",
            "Reference Baselines",
            "Build reproducible reference implementations, "
            "fixtures, tests and baseline measurements.",
        ),
        (
            "O3",
            "Joint Intelligence",
            "Support aspect/opinion extraction together "
            "with valence and arousal prediction.",
        ),
        (
            "O4",
            "Transfer & Calibration",
            "Evaluate cross-domain transfer, calibration, "
            "subgroups and evidence review.",
        ),
        (
            "O5",
            "Acceptance & Impact",
            "Validate acceptance criteria, negative cases, "
            "reproducibility and operating guidance.",
        ),
    ]

    for code, title, description in objectives:

        with st.container(border=True):

            st.caption(code)

            st.subheader(
                title
            )

            st.write(
                description
            )

    st.header(
        "Platform Capabilities"
    )

    capabilities = [
        "🌐 Multilingual language detection",
        "🎯 Aspect and opinion extraction",
        "💭 Aspect-based sentiment analysis",
        "🧭 Target-aware stance analysis",
        "📐 Valence and arousal estimation",
        "🔎 Evidence-grounded outputs",
        "📊 Batch document analysis",
        "📈 Evaluation and insights",
    ]

    for capability in capabilities:

        st.write(
            capability
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CMASSIP • Continuous Multilingual Aspect Sentiment "
    "& Stance Intelligence Platform"
)