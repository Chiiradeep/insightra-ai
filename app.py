import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import os
import json
import sqlite3
from google import genai

from tools import (
    set_dataset,
    get_dataset_info,
    get_numeric_columns,
    get_categorical_columns,
    get_column_average,
    get_column_sum,
    get_column_max,
    get_column_min,
    get_top_row,
    get_bottom_row,
    get_unique_values,
    get_value_counts,
    get_missing_values,
    group_and_aggregate,
    get_dataset_preview
)

from report_generator import create_pdf_report


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Insightra AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Prediction history is persisted in a local SQLite database beside this app.
# Set INSIGHTRA_HISTORY_DB to choose a different database location if needed.
HISTORY_DB_PATH = os.environ.get(
    "INSIGHTRA_HISTORY_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "insightra_prediction_history.db"),
)


def _connect_history_db():
    """Open and initialize the local prediction-history database."""
    connection = sqlite3.connect(HISTORY_DB_PATH, timeout=10)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            source TEXT,
            model TEXT,
            target TEXT,
            task TEXT,
            rows_count INTEGER,
            metrics_json TEXT NOT NULL DEFAULT '{}',
            preview_csv TEXT,
            dedupe_key TEXT UNIQUE
        )
    """)
    connection.commit()
    return connection


def _history_record_to_db(record):
    preview = record.get("preview")
    preview_csv = preview.to_csv(index=False) if isinstance(preview, pd.DataFrame) else None
    metrics_json = json.dumps(record.get("metrics", {}) or {}, default=str)
    return (
        str(record.get("timestamp", "—")), str(record.get("source", "—")),
        str(record.get("model", "—")), str(record.get("target", "—")),
        str(record.get("task", "—")), int(record.get("rows", 0) or 0),
        metrics_json, preview_csv, record.get("dedupe_key"),
    )


def _load_persistent_prediction_history(limit=100):
    """Load recent records from SQLite and reconstruct saved prediction previews."""
    connection = _connect_history_db()
    try:
        rows = connection.execute("""
            SELECT timestamp, source, model, target, task, rows_count,
                   metrics_json, preview_csv, dedupe_key
            FROM prediction_history ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()
    finally:
        connection.close()
    history = []
    for timestamp, source, model, target, task, rows_count, metrics_json, preview_csv, dedupe_key in rows:
        try:
            metrics = json.loads(metrics_json or "{}")
        except (TypeError, ValueError):
            metrics = {}
        preview = None
        if preview_csv:
            try:
                from io import StringIO
                preview = pd.read_csv(StringIO(preview_csv))
            except Exception:
                preview = None
        history.append({
            "timestamp": timestamp, "source": source, "model": model,
            "target": target, "task": task, "rows": rows_count, "metrics": metrics,
            "preview": preview, "dedupe_key": dedupe_key,
        })
    return history


def _clear_persistent_prediction_history():
    connection = _connect_history_db()
    try:
        connection.execute("DELETE FROM prediction_history")
        connection.commit()
    finally:
        connection.close()


if "insightra_prediction_history" not in st.session_state:
    try:
        st.session_state["insightra_prediction_history"] = _load_persistent_prediction_history()
    except Exception as exc:
        st.session_state["insightra_prediction_history"] = []
        st.warning(f"Persistent prediction history could not be loaded: {exc}")


def record_prediction_history(record, dedupe_key=None):
    """Save a prediction/model record to SQLite and refresh session history."""
    history = st.session_state.setdefault("insightra_prediction_history", [])
    if dedupe_key and any(item.get("dedupe_key") == dedupe_key for item in history):
        return
    record["dedupe_key"] = dedupe_key
    connection = _connect_history_db()
    try:
        connection.execute("""
            INSERT OR IGNORE INTO prediction_history
            (timestamp, source, model, target, task, rows_count, metrics_json, preview_csv, dedupe_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, _history_record_to_db(record))
        connection.commit()
    finally:
        connection.close()
    history.insert(0, record)
    del history[100:]


# ============================================================
# CUSTOM CSS — MODERN INSIGHTRA AI UI
# ============================================================

st.markdown("""
<style>
/* INSIGHTRA AI — MINT / EMERALD / TEAL / BLUE THEME */
.stApp {
    background:
        radial-gradient(circle at 4% 0%, rgba(16,185,129,.10), transparent 24%),
        radial-gradient(circle at 96% 6%, rgba(20,184,166,.10), transparent 25%),
        linear-gradient(135deg, #fffdf7 0%, #f7fbf8 52%, #effcf9 100%);
    color: #17221d;
}
[data-testid="stAppViewContainer"] > .main { background: transparent; }
.block-container { max-width: 1450px; padding-top: 1.8rem; padding-bottom: 3rem; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12352d 0%, #14532d 56%, #115e59 100%);
    border-right: 0;
}
section[data-testid="stSidebar"] .block-container { padding: 1.4rem 1rem; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.17); }
.sidebar-brand { padding: 8px 4px 16px 4px; }
.sidebar-brand-title { font-size:1.35rem; font-weight:800; letter-spacing:-.02em; color:#fff; }
.sidebar-brand-subtitle { color:#bbf7d0; font-size:.82rem; margin-top:4px; }
.sidebar-section { color:#dcfce7; font-size:.78rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em; margin:18px 0 8px 2px; }
.engine-pill { border:1px solid rgba(110,231,183,.42); background:rgba(16,185,129,.17); border-radius:12px; padding:10px 12px; color:#bbf7d0; font-size:.82rem; font-weight:700; }
.engine-caption { color:#a7f3d0; font-size:.75rem; margin-top:7px; }
.hero { border:1px solid rgba(16,185,129,.18); border-radius:26px; padding:30px 32px; background:linear-gradient(135deg,#fffef9 0%,#f0fdf4 52%,#ecfdf5 100%); box-shadow:0 18px 50px rgba(20,83,45,.10); margin-bottom:20px; position:relative; overflow:hidden; }
.hero::after { content:""; position:absolute; width:260px; height:260px; right:-100px; top:-120px; border-radius:50%; background:linear-gradient(135deg,rgba(16,185,129,.18),rgba(20,184,166,.15)); }
.hero-kicker { color:#059669; font-size:.76rem; font-weight:850; letter-spacing:.12em; text-transform:uppercase; }
.hero-title { font-size:2.3rem; line-height:1.1; font-weight:850; margin:6px 0 8px 0; color:#13231c; }
.hero-subtitle { color:#52665d; font-size:.98rem; max-width:720px; line-height:1.6; }
.status-chip { display:inline-block; padding:9px 14px; border-radius:999px; background:#ecfdf5; border:1px solid #6ee7b7; color:#047857; font-size:.78rem; font-weight:800; white-space:nowrap; }
.section-heading { font-size:1.25rem; font-weight:850; margin:22px 0 10px 0; color:#17221d; }
.section-caption { color:#65776f; font-size:.86rem; margin-bottom:12px; }
.dataset-strip { border-radius:16px; border:1px solid #a7f3d0; background:linear-gradient(90deg,#f0fdf4,#ecfdf5); padding:12px 16px; margin:8px 0 18px 0; }
.dataset-name { font-weight:850; color:#047857; }
.dataset-meta { color:#65776f; font-size:.80rem; margin-top:3px; }
div[data-testid="stMetric"] { background:rgba(255,255,255,.96); border:1px solid #d1fae5; border-radius:18px; padding:16px 18px; min-height:112px; box-shadow:0 8px 25px rgba(20,83,45,.07); }
div[data-testid="stMetric"]:hover { border-color:#6ee7b7; box-shadow:0 12px 30px rgba(5,150,105,.12); transform:translateY(-2px); }
div[data-testid="stMetricLabel"] { color:#65776f; font-weight:750; }
div[data-testid="stMetricValue"] { color:#17221d; font-weight:850; }
button[data-baseweb="tab"] { font-weight:800; color:#65776f; padding-left:12px; padding-right:12px; }
button[data-baseweb="tab"][aria-selected="true"] { color:#059669; }
div[data-baseweb="tab-highlight"] { background:linear-gradient(90deg,#10b981,#14b8a6,#3b82f6); height:3px; }
[data-testid="stFileUploader"] { background:rgba(255,255,255,.98); border:1px dashed #6ee7b7; border-radius:14px; padding:8px; }
[data-testid="stFileUploaderDropzone"] { background:linear-gradient(135deg,#fbfffc,#ecfdf5); border-radius:10px; }
[data-testid="stFileUploaderDropzone"] button { background:linear-gradient(135deg,#059669,#14b8a6) !important; color:white !important; border:0 !important; border-radius:9px !important; font-weight:800 !important; }
[data-testid="stFileUploaderDropzone"] small { color:#65776f !important; }
div[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; border:1px solid #d1fae5; box-shadow:0 6px 20px rgba(20,83,45,.05); }
.stButton > button { border-radius:11px; border:1px solid #a7f3d0; background:linear-gradient(135deg,#f0fdf4,#ecfdf5); color:#047857; font-weight:750; box-shadow:0 5px 15px rgba(5,150,105,.07); }
.stButton > button:hover { border-color:#10b981; background:linear-gradient(135deg,#dcfce7,#ccfbf1); transform:translateY(-1px); }
.empty-card { border:1px solid #a7f3d0; border-radius:24px; padding:32px; background:linear-gradient(135deg,#fffef9,#f0fdf4 55%,#ecfdf5); text-align:center; margin:16px 0; box-shadow:0 15px 40px rgba(20,83,45,.08); }
.empty-icon { font-size:3rem; }
.empty-title { font-size:1.55rem; font-weight:850; margin-top:8px; color:#17221d; }
.empty-text { color:#65776f; max-width:650px; margin:8px auto 0 auto; line-height:1.6; }
.workflow-card { border:1px solid #d1fae5; border-radius:18px; padding:18px; min-height:130px; background:rgba(255,255,255,.94); box-shadow:0 8px 24px rgba(20,83,45,.06); }
.workflow-icon { font-size:1.5rem; }
.workflow-title { font-weight:850; margin-top:8px; color:#17221d; }
.workflow-text { color:#65776f; font-size:.82rem; line-height:1.5; margin-top:5px; }
label,.stMarkdown,.stText,p { color:#34443d; }
[data-testid="stAlert"] { border-radius:14px; }
.section-title {
    color: #173b2e;
    font-size: 1.22rem;
    font-weight: 850;
    letter-spacing: -.02em;
    margin: 1.1rem 0 .65rem 0;
    padding: .25rem 0 .55rem .75rem;
    border-left: 4px solid #10b981;
    border-bottom: 1px solid #d1fae5;
}
.insight-box {
    background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
    border: 1px solid #bbf7d0;
    border-radius: 16px;
    padding: 18px 20px;
    margin: 8px 0 16px 0;
    box-shadow: 0 8px 24px rgba(20,83,45,.06);
}
[data-testid="stChatMessage"] {
    border: 1px solid #d1fae5;
    border-radius: 14px;
    padding: 12px 15px;
    background: rgba(255,255,255,.88);
}
[data-testid="stChatInput"] {
    border-color: #a7f3d0;
    border-radius: 14px;
}
div[data-testid="stTabs"] { margin-top: 1rem; }
@media (max-width: 768px) {
    .block-container { padding-top: 1rem; padding-left: 1rem; padding-right: 1rem; }
    .hero { padding: 22px 20px; border-radius: 20px; }
    .hero-title { font-size: 1.8rem; }
    .empty-card { padding: 24px 18px; }
}

/* PREMIUM DASHBOARD FINISH — spacing, focus states, responsive surfaces */
:root { color-scheme: light; }
html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.stApp { background-attachment: fixed; }
.block-container { padding-left: clamp(1rem, 3vw, 2.4rem); padding-right: clamp(1rem, 3vw, 2.4rem); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #e2f5eb; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] { background: rgba(255,255,255,.97); }
[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
[data-testid="stSidebar"] [data-testid="stFileUploader"] small { color: #24463a !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background: #f3fbf6; border-color: #a7f3d0; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button { background: #087f5b !important; }
[data-testid="stMetric"] { transition: box-shadow .18s ease, transform .18s ease, border-color .18s ease; }
[data-testid="stMetricLabel"] p { font-size: .84rem; letter-spacing: .01em; }
[data-testid="stMetricValue"] { letter-spacing: -.035em; }
[data-testid="stTabs"] [role="tablist"] { gap: .35rem; border-bottom: 1px solid #d9eee3; }
[data-testid="stTabs"] button[role="tab"] { border-radius: 10px 10px 0 0; min-height: 2.7rem; }
[data-testid="stTabs"] button[role="tab"]:focus-visible,
.stButton button:focus-visible, .stDownloadButton button:focus-visible { outline: 3px solid rgba(16,185,129,.38); outline-offset: 2px; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button { min-height: 2.55rem; transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { box-shadow: 0 8px 20px rgba(5,150,105,.13); }
[data-testid="stDataFrame"], [data-testid="stTable"] { background: #fff; }
[data-testid="stExpander"] { border: 1px solid #d7eee2; border-radius: 14px; background: rgba(255,255,255,.8); }
[data-testid="stChatMessage"] { box-shadow: 0 5px 18px rgba(20,83,45,.035); }
[data-testid="stAlert"] { border-width: 1px; }
[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg,#059669,#14b8a6); }
.stTextInput input, .stTextArea textarea, .stNumberInput input, [data-baseweb="select"] > div { border-radius: 10px; }
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus { border-color: #10b981; box-shadow: 0 0 0 1px #10b981; }
.hero, .empty-card { isolation: isolate; }
.hero-title { letter-spacing: -.045em; }
.section-title { border-radius: 0 8px 8px 0; }
.dataset-strip { overflow-wrap: anywhere; }
@media (max-width: 900px) {
  .block-container { padding-top: 1rem; }
  div[data-testid="stHorizontalBlock"] { gap: .8rem; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition-duration: .01ms !important; animation-duration: .01ms !important; scroll-behavior: auto !important; }
}


/* HARD CONTRAST OVERRIDE — Streamlit version-safe sidebar text */
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] .stCaption p {
    color: #f0fdf4 !important;
    -webkit-text-fill-color: #f0fdf4 !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] label p,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
    color: #24463a !important;
    -webkit-text-fill-color: #24463a !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button *,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] label,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] label * {
    color: #24463a !important;
    -webkit-text-fill-color: #24463a !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* CONTRAST FIX: readable sidebar navigation and uploader labels */
[data-testid="stSidebar"] [data-testid="stPageLink"] a,
[data-testid="stSidebar"] [data-testid="stPageLink"] a span,
[data-testid="stSidebar"] [data-testid="stPageLink"] p,
[data-testid="stSidebar"] nav a,
[data-testid="stSidebar"] nav a span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a {
    color: #f0fdf4 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #f0fdf4 !important;
}
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover,
[data-testid="stSidebar"] nav a:hover {
    color: #6ee7b7 !important;
    -webkit-text-fill-color: #6ee7b7 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] label {
    color: #24463a !important;
    -webkit-text-fill-color: #24463a !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"],
[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"] span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">🤖 Insightra AI</div>
            <div class="sidebar-brand-subtitle">Intelligent Data Analysis Agent</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-section">Workspace</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📂 Choose your CSV dataset",
        type=["csv"],
        help="Upload any CSV dataset from Kaggle, a college project, or your own work."
    )

    st.markdown('<div class="sidebar-section">AI Engine</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="engine-pill">🟢 Gemini AI · Online</div>'
        '<div class="engine-caption">Python · Pandas · Gemini</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.caption("Insightra AI • Analyze · Visualize · Ask · Report")


# ============================================================
# HERO HEADER
# ============================================================

hero_left, hero_right = st.columns([5, 1.25], vertical_alignment="center")

with hero_left:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">AI DATA WORKSPACE</div>
            <div class="hero-title">Insightra AI</div>
            <div class="hero-subtitle">
                Turn any CSV dataset into clear insights, interactive visualizations,
                natural-language analysis, and a professional AI-generated report.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with hero_right:
    st.markdown(
        '<div style="text-align:center; margin-top:8px;">'
        '<span class="status-chip">● Agent Online</span>'
        '<div style="color:#94a3b8;font-size:.75rem;margin-top:8px;">Gemini AI</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# NO DATASET
# ============================================================

if uploaded_file is None:

    st.markdown(
        """
        <div class="empty-card">
            <div class="empty-icon">✨</div>
            <div class="empty-title">Your AI data workspace is ready</div>
            <div class="empty-text">
                Start by uploading a CSV from the sidebar. Insightra AI will
                automatically inspect the dataset and unlock exploration,
                visualizations, AI analysis, and report generation.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    workflow = [
        ("📂", "1 · Upload", "Add any CSV dataset from your project, Kaggle, or your own work."),
        ("🔎", "2 · Explore", "Understand columns, statistics, missing values, and data quality."),
        ("📊", "3 · Visualize", "Create charts that make trends and patterns easier to understand."),
        ("🤖", "4 · Ask AI", "Ask questions in plain English and generate an AI analysis report.")
    ]

    for col, (icon, title, desc) in zip((c1, c2, c3, c4), workflow):
        with col:
            st.markdown(
                f"""
                <div class="workflow-card">
                    <div class="workflow-icon">{icon}</div>
                    <div class="workflow-title">{title}</div>
                    <div class="workflow-text">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.stop()


# ============================================================
# LOAD DATASET
# ============================================================


try:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    data = None

    for encoding in encodings:
        try:
            uploaded_file.seek(0)
            data = pd.read_csv(uploaded_file, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue

    if data is None:
        st.error("Unable to read this CSV file. Please check its encoding.")
        st.stop()

    # Keep your existing code below this line!
    # Clean and make headers unique...

    # Clean and make headers unique. Duplicate CSV headers can make
    # data[column_name] return a DataFrame instead of a Series.
    cleaned_columns = []
    used_columns = set()

    for column_index, raw_column in enumerate(data.columns, start=1):
        base_name = str(raw_column).replace("\ufeff", "").strip()
        if not base_name or base_name.lower().startswith("unnamed:"):
            base_name = f"Column {column_index}"

        unique_name = base_name
        suffix = 2
        while unique_name in used_columns:
            unique_name = f"{base_name}_{suffix}"
            suffix += 1

        cleaned_columns.append(unique_name)
        used_columns.add(unique_name)

    data.columns = cleaned_columns

    # Detect numeric-looking text columns such as "43,032,450".
    # Otherwise Pandas reads them as text and Smart Chart rejects them.
    for column in data.columns:
        if pd.api.types.is_object_dtype(data[column].dtype):
            original = data[column]
            cleaned = (
                original.astype("string")
                .str.strip()
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("₹", "", regex=False)
            )
            parsed = pd.to_numeric(cleaned, errors="coerce")
            non_empty = original.notna() & original.astype(str).str.strip().ne("")
            usable_count = int(parsed[non_empty].notna().sum())
            non_empty_count = int(non_empty.sum())

            # Preserve categorical columns unless at least 90% of populated
            # values are numeric after cleaning.
            if non_empty_count > 0 and usable_count / non_empty_count >= 0.90:
                data[column] = parsed

except Exception as e:

    st.error(
        f"Unable to read the CSV file: {e}"
    )

    st.stop()


# ============================================================
# SET DATASET
# ============================================================

set_dataset(data)


# ============================================================
# DATASET INFORMATION
# ============================================================

rows = data.shape[0]

columns = data.shape[1]

numeric_columns = data.select_dtypes(
    include="number"
).columns.tolist()

categorical_columns = data.select_dtypes(
    exclude="number"
).columns.tolist()

missing_values = int(
    data.isnull().sum().sum()
)


# ============================================================
# DATASET STATUS
# ============================================================

file_size_kb = len(uploaded_file.getvalue()) / 1024

st.markdown(
    f"""
    <div class="dataset-strip">
        <div class="dataset-name">📁 {uploaded_file.name}</div>
        <div class="dataset-meta">
            {rows:,} rows · {columns} columns · {file_size_kb:.1f} KB ·
            {len(numeric_columns)} numeric · {len(categorical_columns)} categorical
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-heading">📊 Dataset Overview</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-caption">A quick health check before you start exploring the data.</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("📄 Rows", f"{rows:,}")

with m2:
    st.metric("🧩 Columns", f"{columns:,}")

with m3:
    st.metric("🔢 Numeric Columns", f"{len(numeric_columns):,}")

with m4:
    st.metric("⚠️ Missing Values", f"{missing_values:,}")


# ============================================================
# MAIN TABS
# ============================================================

# ============================================================

# ============================================================
# MAIN TABS

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🔎 Explore",
        "📊 Visualize",
        "🤖 AI Analyst",
        "📄 Reports",
        "🧪 Data Lab"
    ]
)


# ============================================================
# TAB 1 — EXPLORE
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">'
        'Dataset Preview'
        '</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        data.head(10),
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">'
        '📈 Statistical Summary'
        '</div>',
        unsafe_allow_html=True
    )

    if numeric_columns:

        st.dataframe(
            data[numeric_columns].describe().T,
            use_container_width=True
        )

    else:

        st.info(
            "No numeric columns found."
        )

    st.markdown(
        '<div class="section-title">'
        '📋 Column Information'
        '</div>',
        unsafe_allow_html=True
    )

    column_info = pd.DataFrame({

        "Column":
        data.columns,

        "Data Type":
        data.dtypes.astype(str),

        "Missing Values":
        data.isnull().sum().values,

        "Unique Values":
        data.nunique().values

    })

    st.dataframe(
        column_info,
        use_container_width=True
    )


# ============================================================
# TAB 2 — VISUALIZE
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">📊 Data Visualization</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # SMART AI CHART
    # ========================================================

    st.markdown("### 🤖 Smart Chart Generator")
    st.write(
        "Describe the chart you want in simple words and Insightra AI "
        "will choose the appropriate columns and chart type."
    )

    smart_chart_question = st.text_input(
        "What would you like to visualize?",
        placeholder="Example: Show Maths vs Python"
    )

    if st.button("✨ Generate Smart Chart", use_container_width=True):

        if not smart_chart_question.strip():
            st.warning("Please describe the chart you want.")
        elif not numeric_columns:
            st.warning("At least one numeric column is required.")
        else:
            with st.spinner("Insightra AI is choosing the best chart..."):

                chart_prompt = f"""
You are a data visualization assistant.

The uploaded dataset has these columns:
{list(data.columns)}

Numeric columns:
{numeric_columns}

Categorical columns:
{categorical_columns}

User request:
{smart_chart_question}

Choose the best visualization using ONLY the available columns.

Allowed chart types:
- bar
- line
- scatter
- histogram
- pie

Return ONLY valid JSON in exactly this format:
{{
  "chart_type": "bar",
  "x_column": "column_name",
  "y_column": "column_name",
  "reason": "short explanation"
}}

Rules:
1. Never invent column names.
2. For scatter, both x_column and y_column must be numeric.
3. For histogram, use y_column as the numeric column and set x_column to "".
4. For pie, use x_column as the category column and y_column as "".
5. For bar, x_column can be categorical and y_column must be numeric.
6. For line, x_column can be a suitable ordered column and y_column must be numeric.
7. If the request mentions two numeric columns and asks for a relationship or comparison, prefer scatter.
8. Return JSON only. No markdown.
"""

                try:
                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=chart_prompt
                    )

                    import json
                    import re

                    raw_response = response.text.strip()

                    # Remove accidental markdown code fences.
                    raw_response = re.sub(
                        r"^```(?:json)?\s*|\s*```$",
                        "",
                        raw_response,
                        flags=re.IGNORECASE
                    ).strip()

                    chart_config = json.loads(raw_response)

                    chart_type = str(
                        chart_config.get("chart_type", "")
                    ).lower().strip()

                    x_column = str(
                        chart_config.get("x_column", "")
                    ).strip()

                    y_column = str(
                        chart_config.get("y_column", "")
                    ).strip()

                    reason = str(
                        chart_config.get("reason", "")
                    ).strip()

                    valid_chart_types = {
                        "bar",
                        "line",
                        "scatter",
                        "histogram",
                        "pie"
                    }

                    if chart_type not in valid_chart_types:
                        raise ValueError("AI selected an invalid chart type.")

                    if x_column and x_column not in data.columns:
                        raise ValueError(
                            f"AI selected an unavailable X column: {x_column}"
                        )

                    if y_column and y_column not in data.columns:
                        raise ValueError(
                            f"AI selected an unavailable Y column: {y_column}"
                        )

                    if chart_type in ["bar", "line"]:
                        if not y_column or y_column not in numeric_columns:
                            raise ValueError(
                                "Bar and line charts require a numeric Y column."
                            )

                    if chart_type == "scatter":
                        if (
                            x_column not in numeric_columns
                            or y_column not in numeric_columns
                        ):
                            raise ValueError(
                                "Scatter plots require two numeric columns."
                            )

                    if chart_type == "histogram":
                        if y_column not in numeric_columns:
                            raise ValueError(
                                "Histogram requires a numeric column."
                            )

                    if chart_type == "pie":
                        if not x_column:
                            raise ValueError(
                                "Pie chart requires a category column."
                            )

                        if data[x_column].nunique() > 20:
                            raise ValueError(
                                "The selected category has too many unique values for a pie chart."
                            )

                    st.success(
                        f"📊 Chart selected: {chart_type.title()}"
                    )

                    if reason:
                        st.caption(f"💡 {reason}")

                    # ------------------------------------------------
                    # RENDER THE AI-SELECTED CHART
                    # ------------------------------------------------

                    if chart_type == "bar":

                        chart_data = data.groupby(
                            x_column,
                            dropna=False
                        )[y_column].sum().sort_values(
                            ascending=False
                        )

                        fig, ax = plt.subplots(figsize=(9, 5))

                        chart_data.plot.bar(ax=ax)

                        ax.set_title(
                            f"{y_column} by {x_column}"
                        )

                        ax.set_xlabel(x_column)
                        ax.set_ylabel(y_column)

                        plt.xticks(rotation=45, ha="right")
                        plt.tight_layout()

                        st.pyplot(fig)

                    elif chart_type == "line":

                        fig, ax = plt.subplots(figsize=(9, 5))

                        ax.plot(
                            data[x_column],
                            data[y_column],
                            marker="o"
                        )

                        ax.set_title(
                            f"{y_column} over {x_column}"
                        )

                        ax.set_xlabel(x_column)
                        ax.set_ylabel(y_column)

                        plt.xticks(rotation=45, ha="right")
                        plt.tight_layout()

                        st.pyplot(fig)

                    elif chart_type == "scatter":

                        fig, ax = plt.subplots(figsize=(9, 5))

                        ax.scatter(
                            data[x_column],
                            data[y_column]
                        )

                        ax.set_xlabel(x_column)
                        ax.set_ylabel(y_column)

                        ax.set_title(
                            f"{x_column} vs {y_column}"
                        )

                        plt.tight_layout()

                        st.pyplot(fig)

                    elif chart_type == "histogram":

                        fig, ax = plt.subplots(figsize=(9, 5))

                        ax.hist(
                            data[y_column].dropna(),
                            bins=10
                        )

                        ax.set_title(
                            f"Distribution of {y_column}"
                        )

                        ax.set_xlabel(y_column)
                        ax.set_ylabel("Frequency")

                        plt.tight_layout()

                        st.pyplot(fig)

                    elif chart_type == "pie":

                        counts = (
                            data[x_column]
                            .dropna()
                            .value_counts()
                        )

                        fig, ax = plt.subplots(figsize=(7, 7))

                        ax.pie(
                            counts.values,
                            labels=counts.index,
                            autopct="%1.1f%%"
                        )

                        ax.set_title(
                            f"{x_column} Distribution"
                        )

                        st.pyplot(fig)

                except json.JSONDecodeError:
                    st.error(
                        "Insightra AI returned an invalid chart configuration. "
                        "Please try the request again."
                    )

                except Exception as e:
                    st.error(
                        f"Smart Chart error: {e}"
                    )

    st.markdown("---")

    # ========================================================
    # MANUAL VISUALIZATION
    # ========================================================

    # Give visualization widgets a dataset-specific key. This prevents
    # Streamlit from reusing a previous dataset's selected column.
    # Use the actual uploaded file identity, not only column names/row count.
    file_bytes = uploaded_file.getvalue()
    file_signature = hashlib.md5(file_bytes).hexdigest()
    dataset_signature = f"{uploaded_file.name}_{file_signature}"

    # Clear stale manual-chart selections whenever the uploaded file changes.
    previous_file_signature = st.session_state.get("_insightra_file_signature")
    if previous_file_signature != dataset_signature:
        for key in list(st.session_state.keys()):
            if key.startswith("manual_"):
                del st.session_state[key]
        st.session_state["_insightra_file_signature"] = dataset_signature

    st.markdown("### 🎛️ Manual Visualization")

    chart_type = st.selectbox(
        "Choose Visualization",
        [
            "Bar Chart",
            "Line Chart",
            "Scatter Plot",
            "Histogram",
            "Pie Chart",
            "Correlation Heatmap"
        ]
    )

    if chart_type == "Bar Chart":

        x_column = st.selectbox(
            "X-axis",
            data.columns,
            key=f"manual_bar_x_{dataset_signature}"
        )

        if numeric_columns:

            y_key = f"manual_bar_y_{dataset_signature}"

            # Reset a stale selection from a previous CSV.
            if y_key in st.session_state and st.session_state[y_key] not in numeric_columns:
                del st.session_state[y_key]

            y_column = st.selectbox(
                "Y-axis",
                numeric_columns,
                index=0,
                key=y_key
            )

            # Final safety check: never send a stale/missing column to pandas.
            # If Streamlit ever returns an old widget value, replace it with a
            # valid column from the CURRENT dataframe before plotting.
            current_columns = data.columns.tolist()
            current_numeric = [c for c in current_columns if c in numeric_columns]

            if not current_columns or not current_numeric:
                st.warning("This dataset does not contain a usable numeric column for a bar chart.")
                st.stop()

            if x_column not in current_columns:
                x_column = current_columns[0]

            if y_column not in current_numeric:
                y_column = current_numeric[0]

            # X and Y must be different columns. If the same column is selected
            # for both axes, selecting [x_column, y_column] creates duplicate
            # labels and plot_data[y_column] becomes a DataFrame, not a Series.
            if x_column == y_column:
                alternative_x_columns = [column for column in current_columns if column != y_column]
                if alternative_x_columns:
                    x_column = alternative_x_columns[0]
                else:
                    st.warning("This dataset needs at least two columns to create a bar chart. Please upload a CSV with a category column and a numeric column.")
                    st.stop()

            # Validate against the exact dataframe used for this chart.
            # Avoid pandas.DataFrame.plot.bar here: direct Matplotlib plotting
            # makes the selected columns explicit and easier to validate.
            if x_column not in data.columns:
                st.warning("The selected X-axis column is not in the current CSV. Please choose it again.")
                st.stop()

            if y_column not in data.columns or y_column not in numeric_columns:
                st.warning("The selected Y-axis column is not a numeric column in the current CSV. Please choose a valid column.")
                st.stop()

            # Select the two columns separately so each is guaranteed to be 1-D.
            plot_data = pd.DataFrame({
                x_column: data[x_column].copy(),
                y_column: pd.to_numeric(data[y_column].copy(), errors="coerce"),
            })
            plot_data = plot_data.dropna(subset=[y_column])

            if plot_data.empty:
                st.warning(f"Column '{y_column}' has no usable numeric values to plot.")
                st.stop()

            # Group categories to keep duplicate X values readable.
            plot_data[x_column] = plot_data[x_column].fillna("Missing").astype(str)
            chart_data = plot_data.groupby(x_column, dropna=False)[y_column].sum().sort_values(ascending=False)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(chart_data.index.astype(str), chart_data.values)
            ax.set_title(f"{y_column} by {x_column}")
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            ax.set_title(
                f"{y_column} by {x_column}"
            )

            st.pyplot(fig)

        else:

            st.warning(
                "A numeric column is required."
            )

    elif chart_type == "Line Chart":

        if numeric_columns:

            x_column = st.selectbox(
                "X-axis",
                data.columns,
                key=f"manual_line_x_{dataset_signature}"
            )

            y_column = st.selectbox(
                "Y-axis",
                numeric_columns,
                key=f"manual_line_y_{dataset_signature}"
            )

            fig, ax = plt.subplots()

            ax.plot(
                data[x_column],
                data[y_column],
                marker="o"
            )

            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)

            ax.set_title(
                f"{y_column} over {x_column}"
            )

            st.pyplot(fig)

        else:

            st.warning(
                "A numeric column is required."
            )

    elif chart_type == "Scatter Plot":

        if len(numeric_columns) >= 2:

            x_column = st.selectbox(
                "X-axis",
                numeric_columns,
                key=f"manual_scatter_x_{dataset_signature}"
            )

            y_column = st.selectbox(
                "Y-axis",
                numeric_columns,
                index=1,
                key=f"manual_scatter_y_{dataset_signature}"
            )

            fig, ax = plt.subplots()

            ax.scatter(
                data[x_column],
                data[y_column]
            )

            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)

            ax.set_title(
                f"{x_column} vs {y_column}"
            )

            st.pyplot(fig)

        else:

            st.warning(
                "At least two numeric columns are required."
            )

    elif chart_type == "Histogram":

        if numeric_columns:

            column = st.selectbox(
                "Select Column",
                numeric_columns,
                key=f"manual_histogram_{dataset_signature}"
            )

            fig, ax = plt.subplots()

            ax.hist(
                data[column].dropna(),
                bins=10
            )

            ax.set_title(
                f"Distribution of {column}"
            )

            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")

            st.pyplot(fig)

        else:

            st.warning(
                "No numeric columns available."
            )

    elif chart_type == "Pie Chart":

        categorical_options = [
            col
            for col in data.columns
            if data[col].nunique() <= 20
        ]

        if categorical_options:

            column = st.selectbox(
                "Select Category",
                categorical_options,
                key=f"manual_pie_{dataset_signature}"
            )

            counts = data[column].value_counts()

            fig, ax = plt.subplots()

            ax.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.1f%%"
            )

            ax.set_title(
                f"{column} Distribution"
            )

            st.pyplot(fig)

        else:

            st.warning(
                "No suitable categorical columns found."
            )

    elif chart_type == "Correlation Heatmap":

        if len(numeric_columns) >= 2:

            correlation = data[
                numeric_columns
            ].corr()

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            sns.heatmap(
                correlation,
                annot=True,
                cmap="coolwarm",
                ax=ax
            )

            ax.set_title(
                "Correlation Heatmap"
            )

            st.pyplot(fig)

        else:

            st.warning(
                "At least two numeric columns are required."
            )


# ============================================================
# TAB 3 — AI ANALYST
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">'
        '🤖 Insightra AI Analyst'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Chat with your dataset and ask follow-up questions naturally."
    )

    # ========================================================
    # AUTOMATIC DATASET INSIGHTS
    # ========================================================

    st.markdown("---")

    st.markdown("### ✨ Automatic Dataset Insights")

    st.write(
        "Let Insightra AI automatically inspect your dataset "
        "and summarize the most important findings."
    )

    if st.button(
        "✨ Generate Dataset Insights",
        use_container_width=True
    ):

        with st.spinner("Analyzing your dataset..."):

            total_rows = data.shape[0]
            total_columns = data.shape[1]

            numeric_cols = data.select_dtypes(
                include="number"
            ).columns.tolist()

            categorical_cols = data.select_dtypes(
                exclude="number"
            ).columns.tolist()

            total_missing = int(
                data.isnull().sum().sum()
            )

            total_duplicates = int(
                data.duplicated().sum()
            )

            numeric_summary = {}

            for column in numeric_cols:

                numeric_summary[column] = {
                    "average": round(
                        float(data[column].mean()),
                        2
                    ),
                    "minimum": round(
                        float(data[column].min()),
                        2
                    ),
                    "maximum": round(
                        float(data[column].max()),
                        2
                    )
                }

            categorical_summary = {}

            for column in categorical_cols:

                non_null_values = (
                    data[column].dropna()
                )

                if len(non_null_values) > 0:

                    top_value = (
                        non_null_values
                        .value_counts()
                        .index[0]
                    )

                else:

                    top_value = "None"

                categorical_summary[column] = {
                    "unique_values":
                        int(data[column].nunique()),

                    "top_value":
                        str(top_value)
                }

            dataset_summary = f"""
DATASET OVERVIEW

Rows: {total_rows}
Columns: {total_columns}

Numeric Columns:
{numeric_cols}

Categorical Columns:
{categorical_cols}

Missing Values:
{total_missing}

Duplicate Rows:
{total_duplicates}

NUMERIC STATISTICS

{numeric_summary}

CATEGORICAL STATISTICS

{categorical_summary}
"""

            insight_prompt = f"""
You are Insightra AI, an intelligent AI Data Analyst.

Analyze the following dataset summary:

{dataset_summary}

Provide a useful beginner-friendly analysis.

Use exactly these sections:

## 📊 Dataset Overview

Explain the size and structure of the dataset.

## 🔍 Key Insights

Identify important patterns from the provided statistics.

Mention important numbers and columns.

## ⚠️ Data Quality

Discuss missing values and duplicate rows.

## 💡 Recommendations

Give practical suggestions for further analysis.

IMPORTANT:

Use only the information provided.

Do not invent facts.

Do not assume information that is not present.

Mention actual numbers when useful.

Keep the explanation simple.

Do not explain internal AI tools.
"""

            try:

                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=insight_prompt
                )

                ai_insights = response.text

                st.session_state["ai_insights"] = ai_insights

                st.success(
                    "✨ Dataset Insights Generated"
                )

                st.markdown(
                    '<div class="insight-box">',
                    unsafe_allow_html=True
                )

                st.markdown(ai_insights)

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:

                st.error(
                    f"Gemini error: {e}"
                )

    # ========================================================
    # CHAT-STYLE DATA ANALYST
    # ========================================================

    st.markdown("---")

    st.markdown("### 💬 Chat with Insightra AI")

    st.write(
        "Ask questions naturally and continue with follow-up questions."
    )

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    # IMPORTANT:
    # Do not store a Gemini Chat object in session_state.
    # Streamlit reruns can leave that object closed.
    # We store only text history and create a fresh Gemini
    # chat for each request.

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    clear_col, status_col = st.columns([1, 3])

    with clear_col:

        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True
        ):

            st.session_state["chat_messages"] = []

            st.rerun()

    with status_col:

        st.caption(
            "🟢 Insightra AI is connected to your uploaded dataset"
        )

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    for message in st.session_state["chat_messages"]:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    user_message = st.chat_input(
        "Ask Insightra about your dataset..."
    )

    if user_message:

        st.session_state["chat_messages"].append(
            {
                "role": "user",
                "content": user_message
            }
        )

        with st.chat_message("user"):

            st.markdown(user_message)

        # Keep recent conversation as text so follow-up
        # questions work without storing a Gemini Chat object.

        conversation_history = ""

        for message in st.session_state["chat_messages"][-10:]:

            role = message["role"].upper()
            content = message["content"]

            conversation_history += (
                f"\n{role}: {content}\n"
            )

        tools = [
            get_dataset_info,
            get_numeric_columns,
            get_categorical_columns,
            get_column_average,
            get_column_sum,
            get_column_max,
            get_column_min,
            get_top_row,
            get_bottom_row,
            get_unique_values,
            get_value_counts,
            get_missing_values,
            group_and_aggregate,
            get_dataset_preview
        ]

        agent_instructions = f"""
You are Insightra AI, an intelligent AI Data Analyst.

You are analyzing the dataset currently uploaded by the user.

DATASET COLUMNS:
{list(data.columns)}

NUMERIC COLUMNS:
{numeric_columns}

CATEGORICAL COLUMNS:
{categorical_columns}

IMPORTANT RULES:

1. Understand the user's question before selecting a tool.
2. Use the available Python tools to calculate answers.
3. Never guess numerical results.
4. Never invent column names.
5. Always use actual dataset column names.
6. Use get_dataset_info when dataset structure is needed.
7. Use get_numeric_columns for numeric columns.
8. Use get_categorical_columns for categorical columns.
9. Use get_dataset_preview when sample rows are useful.
10. Use get_column_average for average questions.
11. Use get_column_sum for total questions.
12. Use get_column_max for maximum questions.
13. Use get_column_min for minimum questions.
14. Use get_top_row for highest-value row questions.
15. Use get_bottom_row for lowest-value row questions.
16. Use get_value_counts for frequency questions.
17. Use get_unique_values for category questions.
18. Use get_missing_values for missing-data questions.
19. Use group_and_aggregate for grouped analysis.
20. group_and_aggregate supports:
sum, average, mean, maximum, minimum, count.
21. Use the conversation history to understand follow-up questions.
22. If the dataset cannot answer the question, clearly say so.
23. Give simple, human-friendly answers.
24. Mention important numbers and values.
25. Do not explain internal tool calling.

You are a professional but beginner-friendly AI Data Analyst.
"""

        full_prompt = f"""
{agent_instructions}

RECENT CONVERSATION:
{conversation_history}

CURRENT USER QUESTION:

{user_message}

Analyze the dataset and return a clear answer.
"""

        try:

            with st.chat_message("assistant"):

                with st.spinner("Insightra AI is analyzing..."):

                    # Fresh Gemini chat for this request only.
                    # It is intentionally NOT stored in session_state.

                    fresh_chat = client.chats.create(
                        model="gemini-3.6-flash",
                        config={
                            "tools": tools
                        }
                    )

                    response = fresh_chat.send_message(
                        full_prompt
                    )

                    answer = response.text

                    st.markdown(answer)

            st.session_state["chat_messages"].append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception as e:

            error_message = f"AI Analyst error: {e}"

            st.error(error_message)

            st.session_state["chat_messages"].append(
                {
                    "role": "assistant",
                    "content": error_message
                }
            )


# ============================================================
# TAB 4 — REPORTS
# ============================================================

with tab4:

    st.markdown(
        '<div class="section-title">'
        '📄 AI Data Analysis Report'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Generate a complete professional PDF report with "
        "dataset statistics, data quality checks, AI findings, "
        "and practical recommendations."
    )

    # ========================================================
    # REPORT PREVIEW
    # ========================================================

    st.markdown("### 📋 Report Contents")

    report_items = [
        "Dataset overview",
        "Data quality analysis",
        "Numeric statistical summary",
        "AI-generated key findings",
        "AI-generated recommendations",
        "Important patterns and observations"
    ]

    for item in report_items:
        st.markdown(f"✅ {item}")

    st.markdown("---")

    # ========================================================
    # LOCAL ANALYSIS
    # ========================================================

    st.markdown("### 🔎 Analysis Summary")

    report_missing = int(data.isnull().sum().sum())
    report_duplicates = int(data.duplicated().sum())

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric("Rows", rows)

    with summary_col2:
        st.metric("Missing Values", report_missing)

    with summary_col3:
        st.metric("Duplicate Rows", report_duplicates)

    st.markdown("---")

    # ========================================================
    # GENERATE COMPLETE AI REPORT
    # ========================================================

    if st.button(
        "🤖 Generate AI Analysis Report",
        use_container_width=True
    ):

        with st.spinner(
            "Insightra AI is analyzing the dataset and preparing your report..."
        ):

            try:

                # ------------------------------------------------
                # Build reliable local statistics first.
                # Gemini receives calculated facts instead of raw
                # data, which keeps API usage smaller.
                # ------------------------------------------------

                numeric_report = {}

                for column in numeric_columns:
                    series = data[column].dropna()

                    if len(series) > 0:
                        numeric_report[column] = {
                            "average": round(float(series.mean()), 2),
                            "minimum": round(float(series.min()), 2),
                            "maximum": round(float(series.max()), 2),
                            "median": round(float(series.median()), 2)
                        }

                categorical_report = {}

                for column in categorical_columns:
                    values = data[column].dropna()

                    if len(values) > 0:
                        counts = values.value_counts()

                        categorical_report[column] = {
                            "unique_values": int(values.nunique()),
                            "most_common": str(counts.index[0]),
                            "most_common_count": int(counts.iloc[0])
                        }

                # ------------------------------------------------
                # Simple local anomaly detection using IQR.
                # This does NOT require another AI call.
                # ------------------------------------------------

                anomaly_report = {}

                for column in numeric_columns:

                    series = data[column].dropna()

                    if len(series) >= 4:

                        q1 = float(series.quantile(0.25))
                        q3 = float(series.quantile(0.75))
                        iqr = q3 - q1

                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr

                        anomaly_count = int(
                            ((series < lower_bound) |
                             (series > upper_bound)).sum()
                        )

                        if anomaly_count > 0:
                            anomaly_report[column] = {
                                "anomalies": anomaly_count,
                                "lower_bound": round(lower_bound, 2),
                                "upper_bound": round(upper_bound, 2)
                            }

                report_facts = f"""
DATASET:
Name: {uploaded_file.name}
Rows: {rows}
Columns: {columns}

NUMERIC COLUMNS:
{numeric_columns}

CATEGORICAL COLUMNS:
{categorical_columns}

MISSING VALUES:
{report_missing}

DUPLICATE ROWS:
{report_duplicates}

NUMERIC STATISTICS:
{numeric_report}

CATEGORICAL STATISTICS:
{categorical_report}

POTENTIAL NUMERIC ANOMALIES:
{anomaly_report}
"""

                # ------------------------------------------------
                # One Gemini call for the complete report analysis.
                # ------------------------------------------------

                report_prompt = f"""
You are Insightra AI, a professional AI Data Analyst.

Create a clear, beginner-friendly analysis for a professional
data analysis report.

Use ONLY the facts provided below.

{report_facts}

Return the report content using EXACTLY these sections:

## Executive Summary
Give a short overall summary of the dataset.

## Key Findings
List the most important findings using actual numbers.

## Data Quality
Discuss missing values, duplicate rows, and potential anomalies.
Do not claim a problem if the supplied facts do not show one.

## Important Patterns
Explain useful patterns visible from the supplied statistics.

## Recommendations
Give practical next steps for analyzing or improving the dataset.

RULES:
1. Never invent numbers.
2. Never invent columns.
3. Do not assume what the dataset represents.
4. Use actual values from the supplied facts.
5. If there are no missing values, say that clearly.
6. If there are no detected anomalies, say that clearly.
7. Keep the language professional but easy to understand.
8. Do not mention Gemini, prompts, tools, or internal processing.
"""

                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=report_prompt
                )

                ai_report = response.text.strip()

                # Store it so it can be reused without another
                # Gemini request during the same session.
                st.session_state["ai_report"] = ai_report

                # Also keep compatibility with the existing
                # report-generation workflow.
                st.session_state["ai_insights"] = ai_report

                st.success(
                    "🎉 AI analysis completed successfully!"
                )

                st.markdown("### 🤖 AI Report Preview")

                st.markdown(
                    '<div class="insight-box">',
                    unsafe_allow_html=True
                )

                st.markdown(ai_report)

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

                # ------------------------------------------------
                # Show local anomaly results separately.
                # ------------------------------------------------

                st.markdown("### 🔍 Local Anomaly Check")

                if anomaly_report:

                    for column, details in anomaly_report.items():

                        st.warning(
                            f"**{column}**: "
                            f"{details['anomalies']} potential "
                            f"outlier(s) detected using the IQR method."
                        )

                else:

                    st.success(
                        "No potential numeric outliers were detected "
                        "using the IQR method."
                    )

            except Exception as e:

                st.error(
                    f"AI report generation error: {e}"
                )

    # ========================================================
    # PDF GENERATION
    # ========================================================

    st.markdown("---")

    st.markdown("### 📥 Create PDF")

    if "ai_report" not in st.session_state:

        st.info(
            "Generate the AI Analysis Report above first. "
            "Then you can create the final PDF."
        )

    else:

        st.success(
            "✅ AI report is ready to be included in the PDF."
        )

        if st.button(
            "📄 Generate Full PDF Report",
            use_container_width=True
        ):

            with st.spinner(
                "Creating your professional PDF report..."
            ):

                try:

                    ai_insights = st.session_state["ai_report"]

                    dataset_name = uploaded_file.name

                    pdf_filename = (
                        "insightra_ai_analysis_report.pdf"
                    )

                    create_pdf_report(
                        filename=pdf_filename,
                        dataset_name=dataset_name,
                        data=data,
                        ai_insights=ai_insights
                    )

                    st.success(
                        "🎉 Professional AI report generated!"
                    )

                    with open(
                        pdf_filename,
                        "rb"
                    ) as pdf_file:

                        pdf_bytes = pdf_file.read()

                    st.download_button(
                        label="📥 Download AI Analysis PDF",
                        data=pdf_bytes,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        use_container_width=True
                    )

                except Exception as e:

                    st.error(
                        f"PDF generation error: {e}"
                    )


# ============================================================
# TAB 5 — DATA LAB: QUALITY, CLEANING, INSIGHTS, PREDICTION
# ============================================================

with tab5:

    st.markdown(
        '<div class="section-title">🧪 Insightra Data Lab</div>',
        unsafe_allow_html=True
    )
    st.write(
        "Check data quality, prepare a clean downloadable CSV, discover patterns, "
        "and train a basic prediction model — all locally on your uploaded dataset."
    )

    quality_tab, clean_tab, insight_tab, recommend_tab, automl_tab, prediction_tab, explain_tab, performance_tab = st.tabs(
        [
            "🩺 Data Quality",
            "🧹 Clean & Export",
            "💡 Insight Explorer",
            "💡 Model Recommendations",
            "🤖 AI AutoML",
            "🔮 Prediction Lab",
            "🧠 Explainability",
            "📊 Model Dashboard"
        ]
    )

    # --------------------------------------------------------
    # 1. AI DATA QUALITY CHECKER (local, reproducible checks)
    # --------------------------------------------------------
    with quality_tab:
        st.markdown("### 🩺 Data Quality Checker")
        duplicate_count = int(data.duplicated().sum())
        total_cells = max(int(data.shape[0] * data.shape[1]), 1)
        missing_pct = float(data.isna().sum().sum() / total_cells * 100)
        constant_columns = [
            str(c) for c in data.columns
            if data[c].nunique(dropna=False) <= 1
        ]
        missing_by_column = data.isna().sum()
        missing_columns = int((missing_by_column > 0).sum())

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
        q2.metric("Duplicate rows", f"{duplicate_count:,}")
        q3.metric("Columns with missing data", f"{missing_columns:,}")
        q4.metric("Missing data", f"{missing_pct:.1f}%")

        quality_rows = []
        for col in data.columns:
            series = data[col]
            miss = int(series.isna().sum())
            unique_count = int(series.nunique(dropna=True))
            row = {
                "Column": str(col),
                "Data type": str(series.dtype),
                "Missing": miss,
                "Missing %": round(miss / max(len(data), 1) * 100, 2),
                "Unique values": unique_count,
                "Constant column": unique_count <= 1,
            }
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                if len(numeric) >= 4:
                    lower = numeric.quantile(0.25)
                    upper = numeric.quantile(0.75)
                    iqr = upper - lower
                    outlier_count = int(((numeric < lower - 1.5 * iqr) | (numeric > upper + 1.5 * iqr)).sum())
                else:
                    outlier_count = 0
                row["Potential IQR outliers"] = outlier_count
            else:
                row["Potential IQR outliers"] = "—"
            quality_rows.append(row)

        st.dataframe(pd.DataFrame(quality_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Outliers are flagged with the standard 1.5×IQR rule. They are only potential outliers; "
            "review them before removing any values."
        )

        issues = []
        if int(data.isna().sum().sum()) > 0:
            issues.append(f"There are {int(data.isna().sum().sum()):,} missing cells across {missing_columns} column(s).")
        if duplicate_count:
            issues.append(f"There are {duplicate_count:,} fully duplicated row(s).")
        if constant_columns:
            issues.append("Constant or nearly empty columns: " + ", ".join(constant_columns) + ".")
        if issues:
            st.warning("**Quality checks to review**\n\n" + "\n\n".join(f"- {issue}" for issue in issues))
        else:
            st.success("No missing cells, duplicate rows, or constant columns were detected.")

    # --------------------------------------------------------
    # 2. CLEAN DATA + DOWNLOAD CSV
    # --------------------------------------------------------
    with clean_tab:
        st.markdown("### 🧹 Clean & Export Dataset")
        st.write("Choose the cleaning actions below. Your original uploaded dataset remains unchanged.")
        drop_duplicates = st.checkbox("Remove fully duplicated rows", value=True, key="lab_drop_duplicates")
        missing_strategy = st.selectbox(
            "How should missing values be handled?",
            ["Leave unchanged", "Drop rows with missing values", "Fill numeric with median and text with most common value", "Fill all missing values with 0 / 'Unknown'"],
            key="lab_missing_strategy"
        )
        strip_text = st.checkbox("Trim extra spaces from text values", value=True, key="lab_strip_text")

        cleaned = data.copy()
        before_rows = len(cleaned)
        if strip_text:
            for col in cleaned.select_dtypes(include=["object", "string"]).columns:
                cleaned[col] = cleaned[col].map(lambda value: value.strip() if isinstance(value, str) else value)
        if drop_duplicates:
            cleaned = cleaned.drop_duplicates().copy()
        if missing_strategy == "Drop rows with missing values":
            cleaned = cleaned.dropna().copy()
        elif missing_strategy == "Fill numeric with median and text with most common value":
            for col in cleaned.columns:
                if cleaned[col].isna().any():
                    if pd.api.types.is_numeric_dtype(cleaned[col]):
                        median = cleaned[col].median()
                        cleaned[col] = cleaned[col].fillna(median if pd.notna(median) else 0)
                    else:
                        modes = cleaned[col].mode(dropna=True)
                        fill_value = modes.iloc[0] if not modes.empty else "Unknown"
                        cleaned[col] = cleaned[col].fillna(fill_value)
        elif missing_strategy == "Fill all missing values with 0 / 'Unknown'":
            for col in cleaned.columns:
                if cleaned[col].isna().any():
                    fill_value = 0 if pd.api.types.is_numeric_dtype(cleaned[col]) else "Unknown"
                    cleaned[col] = cleaned[col].fillna(fill_value)

        c1, c2, c3 = st.columns(3)
        c1.metric("Original rows", f"{before_rows:,}")
        c2.metric("Cleaned rows", f"{len(cleaned):,}")
        c3.metric("Rows removed", f"{before_rows - len(cleaned):,}")
        st.markdown("#### Preview of cleaned dataset")
        st.dataframe(cleaned.head(20), use_container_width=True, hide_index=True)
        st.caption(f"Remaining missing cells: {int(cleaned.isna().sum().sum()):,}")
        csv_bytes = cleaned.to_csv(index=False).encode("utf-8-sig")
        base_filename = os.path.splitext(os.path.basename(uploaded_file.name))[0]
        st.download_button(
            "📥 Download cleaned CSV",
            data=csv_bytes,
            file_name=f"{base_filename}_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_cleaned_csv"
        )

    # --------------------------------------------------------
    # 3. AI INSIGHT EXPLORER
    # --------------------------------------------------------
    with insight_tab:
        st.markdown("### 💡 AI Insight Explorer")
        st.write("Find notable numeric relationships, skew, category patterns, and possible outliers.")
        numeric_data = data.select_dtypes(include="number")
        categorical_data = data.select_dtypes(exclude="number")
        insights = []

        if data.empty:
            insights.append("The dataset has no rows to analyze.")
        else:
            insights.append(f"The dataset contains {len(data):,} rows and {len(data.columns):,} columns.")
            missing_total = int(data.isna().sum().sum())
            if missing_total:
                insights.append(f"{missing_total:,} cells are missing ({missing_total / max(data.size, 1) * 100:.1f}% of all cells).")
            else:
                insights.append("No missing cells were found.")
            dupes = int(data.duplicated().sum())
            insights.append(f"{dupes:,} fully duplicated row(s) were detected.")

            if not numeric_data.empty:
                non_constant_numeric = [c for c in numeric_data.columns if numeric_data[c].nunique(dropna=True) > 1]
                if len(non_constant_numeric) >= 2:
                    corr = numeric_data[non_constant_numeric].corr(numeric_only=True)
                    pairs = []
                    for i, col_a in enumerate(corr.columns):
                        for col_b in corr.columns[i + 1:]:
                            value = corr.loc[col_a, col_b]
                            if pd.notna(value):
                                pairs.append((abs(float(value)), float(value), str(col_a), str(col_b)))
                    pairs.sort(reverse=True)
                    if pairs:
                        strength, value, col_a, col_b = pairs[0]
                        direction = "positive" if value > 0 else "negative"
                        insights.append(f"Strongest numeric correlation: {col_a} and {col_b} (r = {value:.2f}, {direction}). Correlation does not prove causation.")
                skew_rows = []
                for col in numeric_data.columns:
                    vals = numeric_data[col].dropna()
                    if len(vals) >= 3 and vals.nunique() > 1:
                        skew_value = float(vals.skew())
                        if abs(skew_value) >= 1:
                            skew_rows.append((abs(skew_value), str(col), skew_value))
                skew_rows.sort(reverse=True)
                if skew_rows:
                    _, col, value = skew_rows[0]
                    side = "right" if value > 0 else "left"
                    insights.append(f"{col} is strongly {side}-skewed (skewness {value:.2f}); its distribution may be asymmetric.")
                outlier_candidates = []
                for col in numeric_data.columns:
                    vals = numeric_data[col].dropna()
                    if len(vals) >= 4:
                        q1v, q3v = vals.quantile(.25), vals.quantile(.75)
                        iqr = q3v - q1v
                        if iqr > 0:
                            count = int(((vals < q1v - 1.5 * iqr) | (vals > q3v + 1.5 * iqr)).sum())
                            if count:
                                outlier_candidates.append((count, str(col)))
                outlier_candidates.sort(reverse=True)
                if outlier_candidates:
                    count, col = outlier_candidates[0]
                    insights.append(f"{col} has {count:,} potential IQR outlier(s); inspect these values before making changes.")
            else:
                insights.append("No numeric columns were detected, so correlation and numeric distribution checks were skipped.")

            if not categorical_data.empty:
                category_candidates = []
                for col in categorical_data.columns:
                    counts = data[col].dropna().astype(str).value_counts()
                    if not counts.empty:
                        category_candidates.append((int(counts.iloc[0]), str(col), str(counts.index[0]), len(data[col].dropna())))
                if category_candidates:
                    count, col, value, valid_count = max(category_candidates)
                    pct = count / max(valid_count, 1) * 100
                    insights.append(f"Most frequent category found: {value} in {col} ({count:,} rows; {pct:.1f}% of non-missing values).")

        for item in insights:
            st.markdown(f"- {item}")

        if not numeric_data.empty:
            st.markdown("#### Numeric correlation map")
            if numeric_data.shape[1] >= 2:
                corr = numeric_data.corr(numeric_only=True)
                fig, ax = plt.subplots(figsize=(min(11, max(5, len(corr.columns) * 0.8)), min(8, max(4, len(corr.columns) * 0.65))))
                sns.heatmap(corr, annot=len(corr.columns) <= 8, fmt=".2f", cmap="YlGnBu", center=0, ax=ax)
                ax.set_title("Numeric column correlations")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("Add at least two numeric columns to view a correlation map.")

        with st.expander("✨ Ask Gemini to explain these findings", expanded=False):
            st.caption("This optional action uses one Gemini request. Local statistics above do not use the API.")
            if st.button("Generate plain-English explanation", key="lab_explain_insights"):
                prompt = (
                    "You are Insightra AI, a beginner-friendly data analyst. Explain these computed findings "
                    "in simple language. Do not invent facts or claim correlation proves causation. "
                    "Give 3-5 practical next steps.\n\n"
                    f"Dataset columns: {list(data.columns)}\n"
                    f"Computed findings:\n- " + "\n- ".join(insights)
                )
                try:
                    with st.spinner("Insightra AI is explaining the findings..."):
                        response = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
                    explanation = (response.text or "").strip()
                    if explanation:
                        st.session_state["lab_insight_explanation"] = explanation
                    else:
                        st.warning("Gemini returned an empty explanation. Please try again later.")
                except Exception as exc:
                    st.error(f"Could not generate the explanation: {exc}")
            if st.session_state.get("lab_insight_explanation"):
                st.markdown(st.session_state["lab_insight_explanation"])

    # --------------------------------------------------------
    # AI MODEL RECOMMENDATIONS — DATASET PROFILE + ALGORITHM ADVICE
    # --------------------------------------------------------
    with recommend_tab:
        st.markdown("### 💡 AI Model Recommendations")
        st.write(
            "Profile your uploaded dataset and get practical baseline algorithm suggestions. "
            "Recommendations use the dataset's shape, column types, missing values, target type, "
            "and class balance; they are guidance, not a guarantee of model quality."
        )

        row_count, col_count = data.shape
        numeric_cols = list(data.select_dtypes(include=["number"]).columns)
        bool_cols = list(data.select_dtypes(include=["bool"]).columns)
        datetime_cols = list(data.select_dtypes(include=["datetime", "datetimetz"]).columns)
        categorical_cols = [c for c in data.columns if c not in numeric_cols and c not in datetime_cols]
        missing_cells = int(data.isna().sum().sum())
        total_cells = max(int(row_count * col_count), 1)
        missing_pct = missing_cells / total_cells * 100
        duplicate_rows = int(data.duplicated().sum())
        small_cardinality_cols = [str(c) for c in data.columns if data[c].nunique(dropna=True) <= 1]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{row_count:,}")
        m2.metric("Columns", f"{col_count:,}")
        m3.metric("Missing cells", f"{missing_pct:.1f}%")
        m4.metric("Duplicate rows", f"{duplicate_rows:,}")

        with st.expander("🔎 Dataset profile", expanded=True):
            profile = pd.DataFrame({
                "Column": [str(c) for c in data.columns],
                "Data type": [str(data[c].dtype) for c in data.columns],
                "Non-missing": [int(data[c].notna().sum()) for c in data.columns],
                "Missing": [int(data[c].isna().sum()) for c in data.columns],
                "Missing %": [round(float(data[c].isna().mean() * 100), 2) for c in data.columns],
                "Unique values": [int(data[c].nunique(dropna=True)) for c in data.columns],
            })
            st.dataframe(profile, use_container_width=True, hide_index=True)
            st.caption(
                f"Numeric columns: {len(numeric_cols)} · Categorical/other columns: {len(categorical_cols)} · "
                f"Date/time columns: {len(datetime_cols)} · Constant/near-empty columns (one or fewer unique values): {len(small_cardinality_cols)}"
            )

        if col_count < 2:
            st.info("Upload a dataset with at least two columns to select input features and a prediction target.")
        else:
            target_col = st.selectbox(
                "Choose the column you want to predict", list(data.columns), key="recommend_target"
            )
            candidate = data[target_col].dropna()
            unique_count = int(candidate.nunique())
            numeric_target = pd.to_numeric(candidate, errors="coerce")
            numeric_share = float(numeric_target.notna().mean()) if len(candidate) else 0.0
            target_is_numeric = pd.api.types.is_numeric_dtype(candidate) or numeric_share >= 0.95
            suggested_task = "Regression" if target_is_numeric and unique_count > 10 else "Classification"
            task_choice = st.radio(
                "Prediction goal", ["Auto-detect", "Regression", "Classification"],
                horizontal=True, key="recommend_task"
            )
            task_type = suggested_task if task_choice == "Auto-detect" else task_choice
            feature_cols = [c for c in data.columns if c != target_col]
            feature_count = len(feature_cols)
            usable_rows = int(data[target_col].notna().sum())
            st.markdown(f"**Detected target type:** {suggested_task} · **Selected task:** {task_type} · **Usable target rows:** {usable_rows:,}")

            notes = []
            if row_count < 100:
                notes.append("Small dataset: results can vary substantially with the train/test split. Prefer simple baselines and cross-validation where practical.")
            elif row_count < 1000:
                notes.append("Small-to-medium dataset: compare simple linear/tree baselines with a Random Forest; use a held-out test set.")
            if missing_pct > 0:
                notes.append(f"Missing values affect {missing_pct:.1f}% of cells. Use an imputation pipeline and inspect columns with high missingness before training.")
            if duplicate_rows:
                notes.append(f"There are {duplicate_rows:,} duplicate rows. Check whether they are genuine repeated observations before removing them.")
            if small_cardinality_cols:
                notes.append("Review constant or single-value columns: they usually add little predictive information.")
            if datetime_cols:
                notes.append("Date/time columns may need feature engineering (for example, year/month/day); do not treat raw timestamps as ordinary categories without checking the model pipeline.")
            if len(numeric_cols) == 0:
                notes.append("No numeric columns were detected. Categorical encoding may be essential, and some algorithms may be unsuitable until preprocessing is applied.")

            st.markdown("#### Recommended starting algorithms")
            if task_type == "Regression":
                recommendations = [
                    ("Linear Regression", "Good interpretable baseline when relationships are approximately linear and numeric scaling/outliers are considered."),
                    ("Decision Tree Regressor", "Captures non-linear patterns and is easy to explain, but can overfit."),
                    ("Random Forest Regressor", "Useful general-purpose baseline for mixed patterns and non-linear relationships; can be less interpretable."),
                ]
                if row_count < 50:
                    recommendations.insert(0, ("Dummy Regressor", "Essential simple baseline to check whether complex models improve over predicting a constant value."))
            else:
                recommendations = [
                    ("Logistic Regression", "Interpretable classification baseline; numeric scaling and categorical encoding may help."),
                    ("Decision Tree Classifier", "Captures non-linear rules and is relatively easy to explain, but may overfit."),
                    ("Random Forest Classifier", "Strong general-purpose baseline for many tabular datasets, though less interpretable than a small tree."),
                ]
                if row_count < 50:
                    recommendations.insert(0, ("Dummy Classifier", "Essential baseline to compare against the most-frequent or simple-strategy prediction."))

            if task_type == "Classification" and len(candidate):
                class_counts = candidate.astype(str).value_counts(dropna=False)
                class_share = class_counts / class_counts.sum()
                imbalance_ratio = float(class_counts.max() / max(class_counts.min(), 1)) if len(class_counts) else 1.0
                majority_pct = float(class_share.max() * 100) if len(class_share) else 0.0
                c1, c2, c3 = st.columns(3)
                c1.metric("Target classes", f"{len(class_counts):,}")
                c2.metric("Largest class", f"{majority_pct:.1f}%")
                c3.metric("Largest/smallest ratio", f"{imbalance_ratio:.1f}×")
                with st.expander("View target class balance", expanded=True):
                    class_table = class_counts.rename_axis("Class").reset_index(name="Count")
                    class_table["Share %"] = (class_table["Count"] / class_table["Count"].sum() * 100).round(2)
                    st.dataframe(class_table, use_container_width=True, hide_index=True)
                    st.bar_chart(class_table.set_index("Class")["Count"])
                if imbalance_ratio >= 3:
                    notes.append("Class imbalance detected (largest class is at least 3× the smallest). Consider stratified splitting when possible, class weights, and metrics such as macro/weighted F1, precision, recall, and a confusion matrix—not accuracy alone.")
                if len(class_counts) > 20:
                    notes.append("The selected target has many distinct classes. Confirm that classification is intended; if it is a continuous measurement, regression may be more suitable.")
            elif task_type == "Regression":
                st.caption("For regression, compare MAE and RMSE (lower is better) and R² (higher is generally better). Inspect target outliers and the target distribution.")

            for idx, (model_name, reason) in enumerate(recommendations, 1):
                with st.container(border=True):
                    st.markdown(f"**{idx}. {model_name}**")
                    st.write(reason)
            st.markdown("#### Data preparation checks")
            if notes:
                for note in notes:
                    st.markdown(f"- {note}")
            else:
                st.success("No obvious dataset-size, missingness, or duplicate-row warning was detected by these basic checks. Still inspect leakage and validate the model on held-out data.")

            st.info("These suggestions are rule-based, not an automatically trained or guaranteed best model. Use the AI AutoML tab to evaluate candidate models on a held-out test set.")
            if st.button("➡️ Continue to AI AutoML", key="recommend_to_automl"):
                st.success("Open the AI AutoML tab to train and compare models using the selected target.")

    # --------------------------------------------------------
    # 4. AI AUTOML — COMPARE MODELS AUTOMATICALLY
    # --------------------------------------------------------
    with automl_tab:
        st.markdown("### 🤖 AI AutoML")
        st.write(
            "Choose a target column and Insightra will prepare the data, train several "
            "baseline models, compare their held-out test performance, and recommend "
            "the strongest result. Training runs locally; your dataset is not sent to Gemini."
        )
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.dummy import DummyClassifier, DummyRegressor
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LinearRegression, LogisticRegression
            from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            automl_sklearn_available = True
        except ImportError:
            automl_sklearn_available = False

        if not automl_sklearn_available:
            st.warning("AI AutoML needs scikit-learn. Install it in your project's active virtual environment: `pip install scikit-learn`")
        elif len(data) < 20:
            st.info("Upload a dataset with at least 20 rows so AutoML can create a useful train/test split.")
        elif len(data.columns) < 2:
            st.info("The dataset needs at least one feature column and one target column.")
        else:
            auto_target = st.selectbox(
                "What should AutoML predict?", list(data.columns), key="automl_target"
            )
            auto_task = st.radio(
                "Task type", ["Auto-detect", "Regression (number)", "Classification (category)"],
                horizontal=True, key="automl_task"
            )
            auto_features = [c for c in data.columns if c != auto_target]
            selected_auto_features = st.multiselect(
                "Input features", auto_features, default=auto_features, key="automl_features",
                help="Remove ID columns or columns that reveal the answer to avoid misleading scores."
            )
            auto_test_size = st.slider(
                "Test set size", min_value=0.2, max_value=0.4, value=0.2, step=0.05,
                key="automl_test_size"
            )
            st.caption("AutoML compares models on the same held-out test set. Results are baselines, not guarantees of future performance.")

            if st.button("🚀 Run AutoML comparison", type="primary", use_container_width=True, key="automl_run"):
                try:
                    if not selected_auto_features:
                        st.error("Choose at least one input feature.")
                    else:
                        auto_frame = data[selected_auto_features + [auto_target]].copy()
                        auto_frame = auto_frame.dropna(subset=[auto_target])
                        raw_y = auto_frame[auto_target]
                        if auto_task == "Regression (number)":
                            auto_y = pd.to_numeric(raw_y, errors="coerce")
                            valid_y = auto_y.notna()
                            auto_frame = auto_frame.loc[valid_y].copy()
                            auto_y = auto_y.loc[valid_y]
                            auto_is_regression = True
                        elif auto_task == "Classification (category)":
                            auto_y = raw_y.astype(str)
                            auto_is_regression = False
                        else:
                            numeric_target = pd.to_numeric(raw_y, errors="coerce")
                            numeric_share = float(numeric_target.notna().mean()) if len(raw_y) else 0.0
                            auto_is_regression = (
                                pd.api.types.is_numeric_dtype(raw_y) or numeric_share >= 0.95
                            ) and raw_y.nunique(dropna=True) > 10
                            if auto_is_regression:
                                auto_y = numeric_target
                                valid_y = auto_y.notna()
                                auto_frame = auto_frame.loc[valid_y].copy()
                                auto_y = auto_y.loc[valid_y]
                            else:
                                auto_y = raw_y.astype(str)

                        if len(auto_frame) < 20 or auto_y.nunique(dropna=True) < 2:
                            st.error("The target needs at least 20 usable rows and at least two distinct values.")
                        else:
                            auto_X = auto_frame[selected_auto_features].copy()
                            auto_num = auto_X.select_dtypes(include="number").columns.tolist()
                            auto_cat = [c for c in auto_X.columns if c not in auto_num]
                            auto_transforms = []
                            if auto_num:
                                auto_transforms.append(("numeric", Pipeline(steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler())
                                ]), auto_num))
                            if auto_cat:
                                auto_transforms.append(("categorical", Pipeline(steps=[
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                                ]), auto_cat))
                            if not auto_transforms:
                                st.error("No usable input columns were found.")
                            else:
                                auto_preprocessor = ColumnTransformer(transformers=auto_transforms, remainder="drop")
                                auto_stratify = None
                                if not auto_is_regression:
                                    counts = auto_y.value_counts()
                                    if counts.min() >= 2 and int(len(auto_y) * auto_test_size) >= len(counts):
                                        auto_stratify = auto_y
                                Xtr, Xte, ytr, yte = train_test_split(
                                    auto_X, auto_y, test_size=auto_test_size, random_state=42,
                                    stratify=auto_stratify
                                )
                                if auto_is_regression:
                                    auto_candidates = [
                                        ("Mean baseline", DummyRegressor(strategy="mean")),
                                        ("Linear Regression", LinearRegression()),
                                        ("Decision Tree", DecisionTreeRegressor(max_depth=8, random_state=42)),
                                        ("Random Forest", RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)),
                                    ]
                                else:
                                    auto_candidates = [
                                        ("Majority baseline", DummyClassifier(strategy="most_frequent")),
                                        ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced")),
                                        ("Decision Tree", DecisionTreeClassifier(max_depth=8, random_state=42, class_weight="balanced")),
                                        ("Random Forest", RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1, class_weight="balanced_subsample")),
                                    ]

                                auto_results = []
                                auto_pipelines = {}
                                progress = st.progress(0, text="Preparing model comparison...")
                                for idx, (model_name, estimator) in enumerate(auto_candidates, start=1):
                                    try:
                                        pipe = Pipeline(steps=[
                                            ("preprocess", auto_preprocessor),
                                            ("model", estimator)
                                        ])
                                        pipe.fit(Xtr, ytr)
                                        pred = pipe.predict(Xte)
                                        if auto_is_regression:
                                            mae = float(mean_absolute_error(yte, pred))
                                            rmse = float(mean_squared_error(yte, pred) ** 0.5)
                                            r2 = float(r2_score(yte, pred)) if len(yte) >= 2 else float("nan")
                                            auto_results.append({"Model": model_name, "MAE ↓": mae, "RMSE ↓": rmse, "R² ↑": r2, "Status": "OK"})
                                            auto_pipelines[model_name] = (pipe, pred)
                                        else:
                                            acc = float(accuracy_score(yte, pred))
                                            weighted_f1 = float(f1_score(yte, pred, average="weighted", zero_division=0))
                                            auto_results.append({"Model": model_name, "Accuracy ↑": acc, "Weighted F1 ↑": weighted_f1, "Status": "OK"})
                                            auto_pipelines[model_name] = (pipe, pred)
                                    except Exception as model_exc:
                                        auto_results.append({"Model": model_name, "Status": f"Skipped: {str(model_exc)[:120]}"})
                                    progress.progress(idx / len(auto_candidates), text=f"Evaluated {idx} of {len(auto_candidates)} models...")
                                progress.empty()
                                results_df = pd.DataFrame(auto_results)
                                valid_results = [r for r in auto_results if r.get("Status") == "OK"]
                                if not valid_results:
                                    st.error("None of the models could be trained. Check the target and input columns, then try again.")
                                else:
                                    if auto_is_regression:
                                        valid_results.sort(key=lambda r: (r["MAE ↓"], r["RMSE ↓"]))
                                    else:
                                        valid_results.sort(key=lambda r: (r["Weighted F1 ↑"], r["Accuracy ↑"]), reverse=True)
                                    best = valid_results[0]
                                    st.success(f"Recommended baseline: **{best['Model']}**")
                                    if auto_is_regression:
                                        k1, k2, k3 = st.columns(3)
                                        k1.metric("Best MAE", f"{best['MAE ↓']:.4g}")
                                        k2.metric("Best RMSE", f"{best['RMSE ↓']:.4g}")
                                        k3.metric("Best R²", f"{best['R² ↑']:.3f}" if pd.notna(best['R² ↑']) else "N/A")
                                        st.caption("Regression ranking uses the lowest MAE first; lower MAE/RMSE is better. R² can be negative.")
                                    else:
                                        k1, k2 = st.columns(2)
                                        k1.metric("Best accuracy", f"{best['Accuracy ↑'] * 100:.1f}%")
                                        k2.metric("Best weighted F1", f"{best['Weighted F1 ↑']:.3f}")
                                        st.caption("Classification ranking uses weighted F1 first, then accuracy. Scores may be misleading if the dataset has leakage or imbalance.")
                                    st.markdown("#### Model leaderboard")
                                    st.dataframe(pd.DataFrame(valid_results).drop(columns=["Status"], errors="ignore"), use_container_width=True, hide_index=True)
                                    failed = [r for r in auto_results if r.get("Status") != "OK"]
                                    if failed:
                                        with st.expander("Models that could not run"):
                                            st.dataframe(pd.DataFrame(failed), use_container_width=True, hide_index=True)
                                    best_pipe, best_predictions = auto_pipelines[best["Model"]]
                                    # Keep the trained pipeline available across Streamlit reruns.
                                    st.session_state["insightra_saved_model_bundle"] = {
                                        "pipeline": best_pipe,
                                        "feature_columns": list(selected_auto_features),
                                        "target_column": str(auto_target),
                                        "task": "regression" if auto_is_regression else "classification",
                                        "model_name": str(best["Model"]),
                                        # Reference values make the explanations readable and consistent.
                                        "reference_data": auto_X.copy(),
                                    }
                                    preview = pd.DataFrame({"Actual": yte.to_numpy(), "Predicted": best_predictions})
                                    from datetime import datetime
                                    automl_metrics = (
                                        {"MAE": best.get("MAE ↓"), "RMSE": best.get("RMSE ↓"), "R²": best.get("R² ↑")}
                                        if auto_is_regression else
                                        {"Accuracy": best.get("Accuracy ↑"), "Weighted F1": best.get("Weighted F1 ↑")}
                                    )
                                    record_prediction_history({
                                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "source": "AI AutoML test evaluation",
                                        "model": str(best["Model"]),
                                        "target": str(auto_target),
                                        "task": "Regression" if auto_is_regression else "Classification",
                                        "rows": int(len(preview)),
                                        "metrics": automl_metrics,
                                        "preview": preview.copy(),
                                    }, dedupe_key="automl:" + datetime.now().isoformat(timespec="microseconds"))
                                    st.markdown(f"#### Best model test predictions — {best['Model']}")
                                    st.dataframe(preview.head(50), use_container_width=True, hide_index=True)
                                    st.download_button(
                                        "📥 Download AutoML leaderboard",
                                        data=pd.DataFrame(valid_results).drop(columns=["Status"], errors="ignore").to_csv(index=False).encode("utf-8-sig"),
                                        file_name="insightra_automl_leaderboard.csv", mime="text/csv", key="download_automl_leaderboard"
                                    )
                                    st.download_button(
                                        "📥 Download best-model test predictions",
                                        data=preview.to_csv(index=False).encode("utf-8-sig"),
                                        file_name="insightra_automl_predictions.csv", mime="text/csv", key="download_automl_predictions"
                                    )
                                    st.info("AutoML picks the best score among these baseline models only. It does not guarantee the model will perform well on new data; review leakage, data quality, and test-set size before relying on the result.")
                except Exception as exc:
                    st.error(f"AutoML comparison failed: {exc}")

        # Saved-model tools stay visible after Streamlit reruns.
        saved_bundle = st.session_state.get("insightra_saved_model_bundle")
        if saved_bundle:
            st.markdown("---")
            st.markdown("#### 💾 Save and reuse your best model")
            st.caption(
                f"Current saved model: {saved_bundle['model_name']} · "
                f"Target: {saved_bundle['target_column']} · "
                f"Task: {saved_bundle['task'].title()}"
            )
            try:
                import io
                from joblib import dump
                model_buffer = io.BytesIO()
                dump(saved_bundle, model_buffer)
                st.download_button(
                    "📦 Download trained model (.joblib)",
                    data=model_buffer.getvalue(),
                    file_name="insightra_saved_model.joblib",
                    mime="application/octet-stream",
                    key="download_insightra_saved_model",
                    help="Keep this file private and load it only in a trusted Python environment."
                )
            except Exception as save_exc:
                st.error(f"Could not package the model for download: {save_exc}")

            st.markdown("##### Predict on a new CSV")
            st.write(
                "Upload a new CSV containing the same input feature columns used during training. "
                "The saved pipeline applies its original preprocessing automatically."
            )
            new_prediction_file = st.file_uploader(
                "Upload new data for prediction", type=["csv"], key="insightra_new_prediction_csv"
            )
            if new_prediction_file is not None:
                try:
                    new_data = pd.read_csv(new_prediction_file)
                    required_features = saved_bundle["feature_columns"]
                    missing_features = [c for c in required_features if c not in new_data.columns]
                    if missing_features:
                        st.error("This CSV is missing required feature columns: " + ", ".join(map(str, missing_features)))
                    elif new_data.empty:
                        st.warning("The uploaded CSV has no rows to predict.")
                    else:
                        prediction_inputs = new_data[required_features].copy()
                        prediction_values = saved_bundle["pipeline"].predict(prediction_inputs)
                        prediction_output = new_data.copy()
                        prediction_output["Predicted " + saved_bundle["target_column"]] = prediction_values
                        from datetime import datetime
                        uploaded_bytes = new_prediction_file.getvalue()
                        prediction_fingerprint = hashlib.sha256(uploaded_bytes).hexdigest()[:20]
                        record_prediction_history({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "source": "Prediction on new CSV",
                            "model": str(saved_bundle["model_name"]),
                            "target": str(saved_bundle["target_column"]),
                            "task": str(saved_bundle["task"]).title(),
                            "rows": int(len(prediction_output)),
                            "metrics": {"Status": "Predictions generated"},
                            "preview": prediction_output.head(50).copy(),
                        }, dedupe_key="newcsv:" + prediction_fingerprint + ":" + str(saved_bundle["model_name"]))
                        st.success(f"Generated {len(prediction_output)} predictions using {saved_bundle['model_name']}.")
                        st.dataframe(prediction_output.head(50), use_container_width=True, hide_index=True)

                        # Plain-language, model-agnostic local explanations:
                        # replace one input at a time with a typical training value and
                        # describe how the prediction changes. This is a sensitivity check,
                        # not proof of causation.
                        st.markdown("#### 💡 Understand a prediction")
                        st.caption(
                            "Choose a row to see which inputs most change its prediction when each is "
                            "replaced, one at a time, with a typical value from the training data. "
                            "These are approximate sensitivity explanations—not causes or guarantees."
                        )
                        explain_row_number = st.selectbox(
                            "Prediction row to explain",
                            options=list(range(len(prediction_inputs))),
                            format_func=lambda i: f"Row {i + 1}",
                            key="automl_explain_row",
                        )
                        explained_input = prediction_inputs.iloc[[explain_row_number]].copy()
                        original_prediction = saved_bundle["pipeline"].predict(explained_input)[0]
                        reference_data = saved_bundle.get("reference_data")
                        explanation_rows = []
                        if isinstance(reference_data, pd.DataFrame) and not reference_data.empty:
                            for feature in required_features:
                                try:
                                    series = reference_data[feature].dropna()
                                    if series.empty:
                                        continue
                                    if pd.api.types.is_numeric_dtype(series):
                                        typical_value = float(series.median())
                                    else:
                                        modes = series.mode(dropna=True)
                                        if modes.empty:
                                            continue
                                        typical_value = modes.iloc[0]
                                    changed_input = explained_input.copy()
                                    changed_input.loc[:, feature] = typical_value
                                    changed_prediction = saved_bundle["pipeline"].predict(changed_input)[0]
                                    if saved_bundle["task"] == "regression":
                                        try:
                                            delta = float(changed_prediction) - float(original_prediction)
                                            explanation_rows.append({
                                                "Input feature": str(feature),
                                                "This row's value": str(explained_input.iloc[0][feature]),
                                                "Typical training value": str(typical_value),
                                                "Prediction change if replaced": delta,
                                                "Plain-language explanation": (
                                                    f"Replacing this value with {typical_value} would "
                                                    f"{'increase' if delta > 0 else 'decrease' if delta < 0 else 'not change'} "
                                                    f"the prediction by about {abs(delta):.4g}."
                                                    if delta != 0 else "Replacing this value leaves the prediction unchanged."
                                                ),
                                            })
                                        except (TypeError, ValueError):
                                            continue
                                    else:
                                        changed_label = str(changed_prediction)
                                        original_label = str(original_prediction)
                                        explanation_rows.append({
                                            "Input feature": str(feature),
                                            "This row's value": str(explained_input.iloc[0][feature]),
                                            "Typical training value": str(typical_value),
                                            "Prediction if replaced": changed_label,
                                            "Plain-language explanation": (
                                                f"Replacing this value changes the predicted class to {changed_label}."
                                                if changed_label != original_label else
                                                f"Replacing this value leaves the predicted class as {original_label}."
                                            ),
                                        })
                                except Exception:
                                    continue
                        if explanation_rows:
                            explanation_df = pd.DataFrame(explanation_rows)
                            if saved_bundle["task"] == "regression":
                                explanation_df["Absolute change"] = explanation_df["Prediction change if replaced"].abs()
                                explanation_df = explanation_df.sort_values("Absolute change", ascending=False).drop(columns=["Absolute change"])
                            st.markdown(f"**Original prediction:** {original_prediction}")
                            st.dataframe(explanation_df, use_container_width=True, hide_index=True)
                            st.caption(
                                "Read this as 'what-if' information: only one feature is changed at a time. "
                                "Correlated features, unusual data, and model limitations can affect the result."
                            )
                        else:
                            st.info("An explanation could not be generated for these columns. The prediction itself is still available.")

                        st.download_button(
                            "📥 Download predictions for new data",
                            data=prediction_output.to_csv(index=False).encode("utf-8-sig"),
                            file_name="insightra_new_data_predictions.csv",
                            mime="text/csv",
                            key="download_insightra_new_predictions"
                        )
                except Exception as predict_exc:
                    st.error(f"Could not generate predictions for this CSV: {predict_exc}")

    # --------------------------------------------------------
    # 5. AI PREDICTION LAB
    # --------------------------------------------------------
    with prediction_tab:
        st.markdown("### 🔮 AI Prediction Lab")
        st.write("Train a baseline machine-learning model using your uploaded CSV. The model runs locally; data is not sent to Gemini.")
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.impute import SimpleImputer
            from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder
            sklearn_available = True
        except ImportError:
            sklearn_available = False

        if not sklearn_available:
            st.warning("The Prediction Lab needs scikit-learn. Install it in your project's active virtual environment with: `pip install scikit-learn`")
        elif len(data) < 10:
            st.info("Upload a dataset with at least 10 rows to train and evaluate a simple model.")
        elif len(data.columns) < 2:
            st.info("The dataset needs at least one feature column and one target column.")
        else:
            target_column = st.selectbox("Choose the value you want to predict (target)", list(data.columns), key="lab_prediction_target")
            task_type = st.radio("Prediction type", ["Auto-detect", "Regression (number)", "Classification (category)"], horizontal=True, key="lab_prediction_task")
            available_features = [str(c) for c in data.columns if c != target_column]
            selected_features = st.multiselect("Feature columns", available_features, default=available_features, key="lab_prediction_features")
            test_fraction = st.slider("Test data size", min_value=0.2, max_value=0.4, value=0.2, step=0.05, key="lab_prediction_test_size")
            st.caption("The test split is held out for evaluation. Results are a baseline, not a guarantee of future performance.")

            if st.button("🚀 Train and evaluate model", use_container_width=True, key="lab_train_model"):
                try:
                    if not selected_features:
                        st.error("Choose at least one feature column.")
                    else:
                        model_data = data[selected_features + [target_column]].copy()
                        model_data = model_data.dropna(subset=[target_column])
                        if len(model_data) < 10:
                            st.error("There are fewer than 10 rows with a target value after removing missing targets.")
                        else:
                            y = model_data[target_column]
                            if task_type == "Regression (number)":
                                y = pd.to_numeric(y, errors="coerce")
                                valid = y.notna()
                                model_data = model_data.loc[valid].copy()
                                y = y.loc[valid]
                                is_regression = True
                            elif task_type == "Classification (category)":
                                y = y.astype(str)
                                is_regression = False
                            else:
                                is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 10
                                if not is_regression:
                                    y = y.astype(str)

                            if len(model_data) < 10 or y.nunique(dropna=True) < 2:
                                st.error("The target needs at least 10 usable rows and at least two distinct target values.")
                            else:
                                X = model_data[selected_features].copy()
                                numeric_features = X.select_dtypes(include="number").columns.tolist()
                                categorical_features = [c for c in X.columns if c not in numeric_features]
                                transformers = []
                                if numeric_features:
                                    transformers.append(("numeric", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_features))
                                if categorical_features:
                                    transformers.append(("categorical", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features))
                                preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
                                model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1) if is_regression else RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced_subsample", n_jobs=-1)
                                pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
                                stratify = None
                                if not is_regression:
                                    class_counts = y.value_counts()
                                    if len(class_counts) >= 2 and class_counts.min() >= 2 and int(len(y) * test_fraction) >= len(class_counts):
                                        stratify = y
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=test_fraction, random_state=42, stratify=stratify
                                )
                                pipeline.fit(X_train, y_train)
                                predictions = pipeline.predict(X_test)
                                st.success("Model trained and evaluated successfully.")
                                if is_regression:
                                    rmse = float(mean_squared_error(y_test, predictions) ** 0.5)
                                    r2 = float(r2_score(y_test, predictions)) if len(y_test) >= 2 else float("nan")
                                    p1, p2, p3 = st.columns(3)
                                    p1.metric("Mean absolute error", f"{mean_absolute_error(y_test, predictions):.4g}")
                                    p2.metric("Root mean squared error", f"{rmse:.4g}")
                                    p3.metric("R² score", f"{r2:.3f}" if pd.notna(r2) else "N/A")
                                    metric_note = "MAE/RMSE are errors in the target's units; lower is generally better. R² can be negative on a small or difficult test set."
                                else:
                                    accuracy = float(accuracy_score(y_test, predictions))
                                    f1 = float(f1_score(y_test, predictions, average="weighted", zero_division=0))
                                    p1, p2 = st.columns(2)
                                    p1.metric("Accuracy", f"{accuracy * 100:.1f}%")
                                    p2.metric("Weighted F1 score", f"{f1:.3f}")
                                    metric_note = "Accuracy is the share of correct test predictions. Weighted F1 balances precision and recall across classes."
                                st.caption(metric_note)
                                prediction_preview = pd.DataFrame({"Actual": y_test.to_numpy(), "Predicted": predictions})
                                from datetime import datetime
                                if is_regression:
                                    lab_metrics = {"MAE": float(mean_absolute_error(y_test, predictions)), "RMSE": rmse, "R²": r2}
                                else:
                                    lab_metrics = {"Accuracy": accuracy, "Weighted F1": f1}
                                record_prediction_history({
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "source": "AI Prediction Lab test evaluation",
                                    "model": "Random Forest",
                                    "target": str(target_column),
                                    "task": "Regression" if is_regression else "Classification",
                                    "rows": int(len(prediction_preview)),
                                    "metrics": lab_metrics,
                                    "preview": prediction_preview.copy(),
                                }, dedupe_key="lab:" + datetime.now().isoformat(timespec="microseconds"))
                                st.markdown("#### Test-set prediction preview")
                                st.dataframe(prediction_preview.head(50), use_container_width=True, hide_index=True)
                                st.download_button(
                                    "📥 Download test predictions",
                                    data=prediction_preview.to_csv(index=False).encode("utf-8-sig"),
                                    file_name="insightra_test_predictions.csv",
                                    mime="text/csv",
                                    key="download_prediction_preview"
                                )
                                st.info("This is a baseline model. Check for leakage, class imbalance, and whether the selected features would really be available at prediction time before relying on the results.")
                except Exception as exc:
                    st.error(f"Could not train the model: {exc}")

        # Prediction history viewer
        st.markdown("---")
        st.markdown("### 🕘 Prediction History")
        history = st.session_state.get("insightra_prediction_history", [])
        if not history:
            st.info("No prediction runs recorded yet. Run AutoML, evaluate a model, or predict on a new CSV to start your history.")
        else:
            history_rows = []
            for item in history:
                metric_text = ", ".join(
                    f"{name}: {value:.4g}" if isinstance(value, (int, float)) and pd.notna(value)
                    else f"{name}: {value}"
                    for name, value in item.get("metrics", {}).items()
                ) or "—"
                history_rows.append({
                    "Date & time": item.get("timestamp", "—"),
                    "Activity": item.get("source", "—"),
                    "Model": item.get("model", "—"),
                    "Target": item.get("target", "—"),
                    "Task": item.get("task", "—"),
                    "Rows": item.get("rows", 0),
                    "Metrics / status": metric_text,
                })
            st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)
            selected_history_index = st.selectbox(
                "View a recorded prediction preview",
                options=list(range(len(history))),
                format_func=lambda i: f"{history[i].get('timestamp', '—')} · {history[i].get('source', 'Run')} · {history[i].get('model', 'Model')}",
                key="insightra_history_selection",
            )
            selected_history = history[selected_history_index]
            history_preview = selected_history.get("preview")
            if isinstance(history_preview, pd.DataFrame):
                st.dataframe(history_preview, use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 Download selected history result",
                    data=history_preview.to_csv(index=False).encode("utf-8-sig"),
                    file_name="insightra_prediction_history_result.csv",
                    mime="text/csv",
                    key="download_selected_prediction_history",
                )
            history_export = pd.DataFrame(history_rows).to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 Export history summary (CSV)",
                data=history_export,
                file_name="insightra_prediction_history.csv",
                mime="text/csv",
                key="download_prediction_history_summary",
            )
            if st.button("🗑️ Clear prediction history", key="clear_insightra_prediction_history"):
                try:
                    _clear_persistent_prediction_history()
                    st.session_state["insightra_prediction_history"] = []
                    st.success("Prediction history cleared from this session and the local database.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not clear saved history: {exc}")
            st.caption("History is saved locally in SQLite and can be restored after restarting the app. The most recent 100 records are shown.")

    # --------------------------------------------------------
    # MODEL EXPLAINABILITY — global importance and local what-if effects
    # --------------------------------------------------------
    with explain_tab:
        st.markdown("### 🧠 Model Explainability")
        st.write(
            "Understand which input features matter to the currently trained AutoML model, "
            "and how changing one feature at a time can change a prediction. Explanations "
            "are model-based estimates, not proof of cause and effect."
        )
        explain_bundle = st.session_state.get("insightra_saved_model_bundle")
        if not explain_bundle:
            st.info("Run AI AutoML first. Its best trained model will then be available here for explanation.")
        else:
            explain_pipe = explain_bundle.get("pipeline")
            explain_features = list(explain_bundle.get("feature_columns", []))
            explain_reference = explain_bundle.get("reference_data")
            explain_model_name = explain_bundle.get("model_name", "Saved model")
            explain_task = explain_bundle.get("task", "regression")
            if not isinstance(explain_reference, pd.DataFrame) or explain_reference.empty or not explain_features:
                st.warning("The saved model does not contain reference rows needed to create explanations. Run AutoML again.")
            else:
                st.caption(f"Explaining: {explain_model_name} · Target: {explain_bundle.get('target_column', '—')} · Task: {str(explain_task).title()}")
                try:
                    fitted_preprocessor = explain_pipe.named_steps.get("preprocess")
                    fitted_estimator = explain_pipe.named_steps.get("model")
                    transformed_names = list(fitted_preprocessor.get_feature_names_out())
                    transformed_importance = None
                    importance_kind = None
                    if hasattr(fitted_estimator, "feature_importances_"):
                        transformed_importance = pd.Series(fitted_estimator.feature_importances_, index=transformed_names)
                        importance_kind = "Tree-based feature importance"
                    elif hasattr(fitted_estimator, "coef_"):
                        coef = fitted_estimator.coef_
                        import numpy as np
                        coef_array = np.asarray(coef)
                        if coef_array.ndim == 1:
                            raw_importance = np.abs(coef_array)
                        else:
                            raw_importance = np.mean(np.abs(coef_array), axis=0)
                        transformed_importance = pd.Series(raw_importance, index=transformed_names)
                        importance_kind = "Absolute model coefficients"
                    if transformed_importance is not None:
                        grouped_importance = {}
                        for transformed_name, score in transformed_importance.items():
                            # ColumnTransformer names are typically numeric__Age or categorical__City_X.
                            stripped_name = str(transformed_name).split("__", 1)[-1]
                            source_feature = next((str(col) for col in explain_features if stripped_name == str(col) or stripped_name.startswith(str(col) + "_") or stripped_name.startswith(str(col) + "=")), stripped_name)
                            grouped_importance[source_feature] = grouped_importance.get(source_feature, 0.0) + float(abs(score))
                        importance_df = pd.DataFrame([{"Feature": k, "Importance": v} for k, v in grouped_importance.items()])
                        importance_df = importance_df.sort_values("Importance", ascending=False).reset_index(drop=True)
                        if float(importance_df["Importance"].sum()) > 0:
                            importance_df["Relative importance (%)"] = importance_df["Importance"] / importance_df["Importance"].sum() * 100
                        st.markdown("#### Global feature importance")
                        st.caption(importance_kind + ". Scores are relative within this model and should not be interpreted as causal effects.")
                        st.bar_chart(importance_df.head(15).set_index("Feature")["Importance"], use_container_width=True)
                        st.dataframe(importance_df.head(20).round(4), use_container_width=True, hide_index=True)
                        st.download_button(
                            "📥 Download feature importance (CSV)",
                            data=importance_df.to_csv(index=False).encode("utf-8-sig"),
                            file_name="insightra_feature_importance.csv", mime="text/csv",
                            key="download_insightra_feature_importance"
                        )
                    else:
                        st.info("This estimator does not expose native coefficients or tree importance. Local what-if explanations are still available below.")

                    st.markdown("#### Explain an individual prediction")
                    explain_source = data.copy() if all(col in data.columns for col in explain_features) else explain_reference.copy()
                    explain_source = explain_source[explain_features].copy()
                    if explain_source.empty:
                        st.info("No rows are available to explain yet.")
                    else:
                        row_choice = st.selectbox(
                            "Choose a row to explain", options=list(range(min(len(explain_source), 500))),
                            format_func=lambda i: f"Row {i + 1}", key="insightra_explain_row"
                        )
                        chosen_row = explain_source.iloc[[row_choice]].copy()
                        base_prediction = explain_pipe.predict(chosen_row)[0]
                        base_confidence = None
                        if explain_task == "classification" and hasattr(explain_pipe, "predict_proba"):
                            try:
                                probabilities = explain_pipe.predict_proba(chosen_row)[0]
                                classes = list(explain_pipe.named_steps["model"].classes_)
                                predicted_index = classes.index(base_prediction)
                                base_confidence = float(probabilities[predicted_index])
                            except Exception:
                                base_confidence = None
                        if explain_task == "regression":
                            st.metric("Baseline prediction", f"{float(base_prediction):.4g}")
                        else:
                            st.metric("Predicted class", str(base_prediction))
                            if base_confidence is not None:
                                st.caption(f"Model confidence for this predicted class: {base_confidence:.1%}. This is not a guarantee of correctness.")
                        reference = explain_reference[explain_features]
                        effect_rows = []
                        for feature in explain_features:
                            modified = chosen_row.copy()
                            original_value = chosen_row.iloc[0][feature]
                            series = reference[feature].dropna()
                            if series.empty:
                                continue
                            if pd.api.types.is_numeric_dtype(series):
                                replacement = float(series.median())
                            else:
                                modes = series.mode(dropna=True)
                                replacement = modes.iloc[0] if not modes.empty else original_value
                            try:
                                modified.at[modified.index[0], feature] = replacement
                                changed_prediction = explain_pipe.predict(modified)[0]
                                if explain_task == "regression":
                                    delta = float(changed_prediction) - float(base_prediction)
                                    effect_rows.append({
                                        "Feature": str(feature), "Original value": str(original_value),
                                        "Comparison value": str(replacement), "Prediction after change": float(changed_prediction),
                                        "Change vs original prediction": delta,
                                        "Interpretation": "Prediction increases" if delta > 1e-12 else ("Prediction decreases" if delta < -1e-12 else "No change"),
                                    })
                                else:
                                    changed_confidence = None
                                    if hasattr(explain_pipe, "predict_proba"):
                                        try:
                                            probs = explain_pipe.predict_proba(modified)[0]
                                            classes = list(explain_pipe.named_steps["model"].classes_)
                                            changed_confidence = float(probs[classes.index(base_prediction)])
                                        except Exception:
                                            pass
                                    same_class = bool(changed_prediction == base_prediction)
                                    effect_rows.append({
                                        "Feature": str(feature), "Original value": str(original_value),
                                        "Comparison value": str(replacement), "Prediction after change": str(changed_prediction),
                                        "Original-class confidence change": (changed_confidence - base_confidence) if changed_confidence is not None and base_confidence is not None else None,
                                        "Interpretation": "Predicted class changed" if not same_class else ("Original-class confidence decreased" if changed_confidence is not None and base_confidence is not None and changed_confidence < base_confidence else "Original class unchanged"),
                                    })
                            except Exception:
                                continue
                        if effect_rows:
                            effects_df = pd.DataFrame(effect_rows)
                            st.caption("Each row changes one feature to a typical reference value (median for numeric columns, most common value for categorical columns), leaving the other inputs unchanged. This is a what-if comparison, not a causal claim.")
                            st.dataframe(effects_df, use_container_width=True, hide_index=True)
                            st.download_button(
                                "📥 Download this prediction explanation (CSV)",
                                data=effects_df.to_csv(index=False).encode("utf-8-sig"),
                                file_name="insightra_prediction_explanation.csv", mime="text/csv",
                                key="download_insightra_local_explanation"
                            )
                        else:
                            st.info("Could not calculate feature changes for this row. Try a different row or retrain AutoML.")
                except Exception as explain_exc:
                    st.error(f"Could not create model explanations: {explain_exc}")
                    st.caption("Try rerunning AutoML to refresh the saved model, then reopen this tab.")

    # --------------------------------------------------------
    # MODEL PERFORMANCE DASHBOARD — summarizes recorded runs
    # --------------------------------------------------------
    with performance_tab:
        st.markdown("### 📊 Model Performance Dashboard")
        st.write(
            "Compare recorded AutoML and Prediction Lab evaluations. Metrics are shown only "
            "when they were actually calculated for a run; different task types should not be "
            "compared using unrelated metrics."
        )

        dashboard_history = st.session_state.get("insightra_prediction_history", [])
        metric_rows = []
        for run in dashboard_history:
            run_metrics = run.get("metrics", {}) or {}
            for metric_name, metric_value in run_metrics.items():
                try:
                    numeric_value = float(metric_value)
                except (TypeError, ValueError):
                    continue
                if pd.notna(numeric_value):
                    metric_rows.append({
                        "Timestamp": run.get("timestamp", "—"),
                        "Activity": run.get("source", "—"),
                        "Model": run.get("model", "Unknown model"),
                        "Target": run.get("target", "—"),
                        "Task": run.get("task", "Unknown"),
                        "Rows": run.get("rows", 0),
                        "Metric": str(metric_name),
                        "Value": numeric_value,
                    })

        if not metric_rows:
            st.info("No numeric model metrics yet. Run AI AutoML or evaluate a model in Prediction Lab, and its results will appear here.")
        else:
            metrics_df = pd.DataFrame(metric_rows)
            c1, c2, c3 = st.columns(3)
            c1.metric("Recorded evaluations", int(metrics_df["Timestamp"].nunique()))
            c2.metric("Models represented", int(metrics_df["Model"].nunique()))
            c3.metric("Metrics available", int(metrics_df["Metric"].nunique()))

            available_tasks = sorted(metrics_df["Task"].dropna().astype(str).unique().tolist())
            selected_task = st.selectbox(
                "Filter by task type",
                options=["All tasks"] + available_tasks,
                key="insightra_dashboard_task_filter",
            )
            task_df = metrics_df if selected_task == "All tasks" else metrics_df[metrics_df["Task"] == selected_task]
            available_metrics = sorted(task_df["Metric"].dropna().astype(str).unique().tolist())

            if not available_metrics:
                st.info("No metrics are available for this task filter.")
            else:
                preferred_order = ["Accuracy", "Weighted F1", "MAE", "RMSE", "R²"]
                default_metric = next((m for m in preferred_order if m in available_metrics), available_metrics[0])
                selected_metric = st.selectbox(
                    "Metric to compare",
                    options=available_metrics,
                    index=available_metrics.index(default_metric),
                    key="insightra_dashboard_metric_filter",
                )
                selected_df = task_df[task_df["Metric"] == selected_metric].copy()
                # Aggregate repeat runs by model so the chart stays readable.
                comparison = (
                    selected_df.groupby("Model", as_index=False)["Value"]
                    .mean()
                    .sort_values("Value", ascending=selected_metric in {"MAE", "RMSE"})
                )

                best_is_lower = selected_metric in {"MAE", "RMSE"}
                best_row = comparison.iloc[0]
                value_text = f"{best_row['Value']:.4g}"
                left, right = st.columns([2, 1])
                with left:
                    st.markdown("#### Model comparison")
                    st.bar_chart(comparison.set_index("Model"), y="Value", use_container_width=True)
                with right:
                    st.markdown("#### Leading result")
                    st.metric("Top model for selected metric", str(best_row["Model"]))
                    st.metric(selected_metric, value_text)
                    st.caption(
                        "Lower is better for MAE/RMSE. Higher is better for accuracy, F1, and R². "
                        "This summary uses the recorded values and averages repeated runs by model."
                    )

                st.markdown("#### Recorded metric details")
                detail_table = selected_df.sort_values("Value", ascending=best_is_lower)[
                    ["Timestamp", "Activity", "Model", "Target", "Task", "Rows", "Value"]
                ]
                st.dataframe(detail_table, use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 Download performance data (CSV)",
                    data=metrics_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="insightra_model_performance.csv",
                    mime="text/csv",
                    key="download_insightra_model_performance",
                )
            st.caption("These metrics are based on recorded test evaluations. They are useful for comparison, but they do not guarantee performance on future data.")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        font-size:12px;
        padding:10px;
    ">
        🤖 Insightra AI • Python + Pandas + Gemini + ML
        <br>
        Intelligent Data Analysis Agent
    </div>
    """,
    unsafe_allow_html=True
)