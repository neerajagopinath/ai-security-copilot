"""
Streamlit Demo Application — AI Security Copilot
--------------------------------------------------
A clean, professional UI for source code vulnerability detection.

Usage:
  streamlit run app.py
"""

import streamlit as st
import requests
import json
import time

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Security Copilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark theme overrides */
    .main {
        background-color: #0f1117;
    }

    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 50%, #1a1f2e 100%);
        border: 1px solid #2d3748;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse at top, rgba(99, 102, 241, 0.1) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8, #38bdf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin: 0;
        font-weight: 400;
    }

    /* Cards */
    .result-card {
        background: #1e2330;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .vuln-card {
        border-left: 4px solid #ef4444;
    }
    .safe-card {
        border-left: 4px solid #22c55e;
    }
    .warning-card {
        border-left: 4px solid #f59e0b;
    }

    /* Metric badges */
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-red { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
    .badge-green { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
    .badge-yellow { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
    .badge-blue { background: rgba(56,189,248,0.15); color: #38bdf8; border: 1px solid rgba(56,189,248,0.3); }

    /* Code area */
    .stTextArea textarea {
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
        font-size: 0.85rem !important;
        background: #0d1117 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #818cf8, #6366f1) !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0d1117 !important;
        border-right: 1px solid #1e2330 !important;
    }

    /* Pattern items */
    .pattern-item {
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.2);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.88rem;
        color: #fbbf24;
    }

    /* Recommendation items */
    .rec-item {
        background: rgba(56,189,248,0.06);
        border-left: 3px solid #38bdf8;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.88rem;
        color: #bae6fd;
    }

    /* Disclaimer */
    .disclaimer-box {
        background: rgba(148,163,184,0.05);
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 1rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #2d3748;
    }

    /* Status indicator */
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .dot-green { background: #22c55e; }
    .dot-yellow { background: #f59e0b; }
    .dot-red { background: #ef4444; }
    .dot-gray { background: #6b7280; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------
API_BASE_URL = "http://127.0.0.1:8000"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ AI Security Copilot")
    st.markdown("---")

    st.markdown("### ⚙️ Configuration")

    language = st.selectbox(
        "Programming Language",
        ["C/C++", "Python", "Java", "JavaScript", "Go", "Rust"],
        index=0,
        help="Select the programming language of your code snippet.",
    )

    model_choice = st.selectbox(
        "Analysis Model",
        ["bilstm", "graphcodebert"],
        index=0,
        format_func=lambda x: {
            "bilstm": "🧠 Bi-LSTM (Demo Checkpoint)",
            "graphcodebert": "🤖 GraphCodeBERT (GPU Required)",
        }[x],
        help="Select the ML model for vulnerability detection.",
    )

    st.markdown("---")

    # API status check
    st.markdown("### 🔌 API Status")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if response.status_code == 200:
            health = response.json()
            bilstm_status = health.get("bilstm_status", "unknown")
            status_color = {
                "trained": "dot-green",
                "demo": "dot-yellow",
                "fallback": "dot-red",
            }.get(bilstm_status, "dot-gray")
            st.markdown(
                f'<span class="status-dot {status_color}"></span>'
                f'API Connected',
                unsafe_allow_html=True,
            )
            st.caption(f"Bi-LSTM: `{bilstm_status}` | Device: `{health.get('device', 'cpu')}`")
        else:
            st.error("API responded with error")
    except requests.exceptions.ConnectionError:
        st.error("❌ API not reachable")
        st.caption("Start API: `uvicorn api.main:app --reload`")
    except Exception:
        st.warning("⚠️ API status unknown")

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown(
        """
        **AI Security Copilot** detects potential vulnerabilities
        in source code using:
        - 🧠 **Bi-LSTM** trained on Devign dataset
        - 📋 **Rule-based** security pattern detection
        - 💡 **Secure fix** suggestions

        > ⚠️ For educational and demonstration use only.
        Not a replacement for professional security review.
        """
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

# Hero header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🛡️ AI Security Copilot</div>
    <div class="hero-subtitle">
        Source Code Vulnerability Detection & Secure Fix Recommendation
        using Bi-LSTM + GraphCodeBERT on the Devign Dataset
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sample code examples
# ---------------------------------------------------------------------------
SAMPLE_CODES = {
    "Vulnerable: Buffer Overflow": """void copy_user_input(char *user_data) {
    char buffer[64];
    strcpy(buffer, user_data);  // No bounds check!
    printf("Data: %s\\n", buffer);
}""",
    "Vulnerable: Command Injection": """void run_command(char *user_input) {
    char cmd[256];
    sprintf(cmd, "ls %s", user_input);
    system(cmd);  // Dangerous: user input in system()
}""",
    "Vulnerable: SQL Injection": """void query_user(char *username) {
    char query[512];
    sprintf(query, "SELECT * FROM users WHERE name='%s'", username);
    db_execute(query);  // User input in SQL!
}""",
    "Safer: Bounded Copy": """void safe_copy(char *user_data) {
    char buffer[64];
    strncpy(buffer, user_data, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\\0';  // Null terminate
    printf("Data: %s\\n", buffer);
}""",
    "Custom Code": "",
}

col_left, col_right = st.columns([2, 1])

with col_right:
    st.markdown('<div class="section-header">Quick Examples</div>', unsafe_allow_html=True)
    sample_choice = st.selectbox(
        "Load sample code",
        list(SAMPLE_CODES.keys()),
        index=0,
        label_visibility="collapsed",
    )

with col_left:
    st.markdown('<div class="section-header">Source Code Input</div>', unsafe_allow_html=True)

# Code input
default_code = SAMPLE_CODES.get(sample_choice, "")
code_input = st.text_area(
    "Paste your source code here:",
    value=default_code,
    height=280,
    placeholder="// Paste your C/C++ or other source code here...\n// The model will analyze it for potential vulnerabilities.",
    label_visibility="collapsed",
)

# Character count
char_count = len(code_input)
max_chars = 50000
count_color = "#ef4444" if char_count > max_chars else "#64748b"
st.markdown(
    f'<div style="text-align:right; font-size:0.75rem; color:{count_color};">'
    f'{char_count:,} / {max_chars:,} characters</div>',
    unsafe_allow_html=True,
)

st.markdown("")

# Analyze button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze_clicked = st.button("🔍 Analyze for Vulnerabilities", use_container_width=True)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
if analyze_clicked:
    if not code_input.strip():
        st.warning("⚠️ Please enter some source code before analyzing.")
    elif len(code_input) > 50000:
        st.error(f"❌ Code too long ({len(code_input):,} chars). Maximum is 50,000 characters.")
    else:
        with st.spinner("🔍 Analyzing code for vulnerabilities..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/analyze",
                    json={
                        "code": code_input,
                        "model": model_choice,
                        "language": language,
                    },
                    timeout=30,
                )

                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get("prediction", "Unknown")
                    prob = result.get("vulnerability_probability", 0.0)
                    confidence = result.get("confidence", 0.0)
                    category = result.get("potential_category", "Unknown")
                    patterns = result.get("suspicious_patterns", [])
                    explanation = result.get("explanation", "")
                    recommendations = result.get("recommendations", [])
                    suggested_code = result.get("suggested_code", "")
                    disclaimer = result.get("disclaimer", "")
                    model_mode = result.get("model_mode", "fallback")

                    # ---------------------
                    # Main result card
                    # ---------------------
                    is_vuln = prediction == "Potentially Vulnerable"
                    card_class = "vuln-card" if is_vuln else "safe-card"
                    prediction_icon = "🚨" if is_vuln else "✅"
                    prediction_color = "#ef4444" if is_vuln else "#22c55e"
                    badge_class = "badge-red" if is_vuln else "badge-green"

                    mode_labels = {
                        "trained": ("🎓 Trained Model", "badge-green"),
                        "demo": ("⚗️ Demo Model", "badge-yellow"),
                        "fallback": ("📋 Rule-Based Only", "badge-yellow"),
                    }
                    mode_label, mode_badge = mode_labels.get(
                        model_mode, ("Unknown", "badge-blue")
                    )

                    st.markdown(f"""
                    <div class="result-card {card_class}">
                        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;">
                            <div>
                                <span style="font-size:1.6rem;">{prediction_icon}</span>
                                <span style="font-size:1.3rem; font-weight:700; color:{prediction_color}; margin-left:8px;">
                                    {prediction}
                                </span>
                            </div>
                            <div style="display:flex; gap:8px; flex-wrap:wrap;">
                                <span class="metric-badge {badge_class}">{prediction}</span>
                                <span class="metric-badge {mode_badge}">{mode_label}</span>
                            </div>
                        </div>
                        <div style="margin-top:1rem; display:flex; gap:2rem; flex-wrap:wrap;">
                            <div>
                                <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.5px;">Vulnerability Probability</div>
                                <div style="font-size:1.5rem; font-weight:700; color:{prediction_color};">{prob:.1%}</div>
                            </div>
                            <div>
                                <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.5px;">Confidence</div>
                                <div style="font-size:1.5rem; font-weight:700; color:#94a3b8;">{confidence:.1%}</div>
                            </div>
                            <div>
                                <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.5px;">Potential Category</div>
                                <div style="font-size:1rem; font-weight:600; color:#e2e8f0; margin-top:4px;">{category}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ---------------------
                    # Tabs for details
                    # ---------------------
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "🔎 Suspicious Patterns",
                        "📝 Explanation",
                        "💡 Recommendations",
                        "🔧 Suggested Fix",
                    ])

                    with tab1:
                        if patterns:
                            st.markdown(
                                f'<div style="color:#94a3b8; font-size:0.9rem; margin-bottom:0.75rem;">'
                                f'Found <strong>{len(patterns)}</strong> security pattern(s):</div>',
                                unsafe_allow_html=True,
                            )
                            for pattern in patterns:
                                st.markdown(
                                    f'<div class="pattern-item">⚠️ {pattern}</div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.success("✅ No suspicious security patterns detected by rule-based analysis.")
                        st.markdown(
                            '<div style="font-size:0.78rem; color:#475569; margin-top:1rem;">'
                            '⚠️ Rule-based findings are separate from ML model predictions.</div>',
                            unsafe_allow_html=True,
                        )

                    with tab2:
                        st.markdown(
                            f'<div class="result-card" style="line-height:1.7; color:#cbd5e1;">'
                            f'{explanation}</div>',
                            unsafe_allow_html=True,
                        )

                    with tab3:
                        if recommendations:
                            for i, rec in enumerate(recommendations, 1):
                                st.markdown(
                                    f'<div class="rec-item">💡 <strong>Rec {i}:</strong> {rec}</div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.info("No specific recommendations generated.")

                    with tab4:
                        if suggested_code:
                            st.markdown(
                                '<div style="color:#94a3b8; font-size:0.88rem; margin-bottom:0.5rem;">'
                                '⚠️ Automated suggestions — requires developer review before use:</div>',
                                unsafe_allow_html=True,
                            )
                            st.code(suggested_code, language="c")
                        else:
                            st.info(
                                "No automated code suggestion available. "
                                "Refer to recommendations above for guidance."
                            )

                    # Disclaimer
                    st.markdown(
                        f'<div class="disclaimer-box">⚖️ <strong>Disclaimer:</strong> {disclaimer}</div>',
                        unsafe_allow_html=True,
                    )

                elif response.status_code == 422:
                    error_detail = response.json().get("detail", "Validation error")
                    st.error(f"❌ Input validation error: {error_detail}")
                else:
                    st.error(f"❌ API error: HTTP {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot connect to the API server.\n\n"
                    "Please start the API first:\n"
                    "```\nuvicorn api.main:app --host 127.0.0.1 --port 8000 --reload\n```"
                )
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The server may be overloaded.")
            except Exception as exc:
                st.error(f"❌ Unexpected error: {str(exc)}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#475569; font-size:0.8rem;">
        🛡️ <strong>AI Security Copilot</strong> — College Project MVP |
        Bi-LSTM + GraphCodeBERT | Devign Dataset |
        Built with PyTorch, FastAPI & Streamlit
        <br><em>For educational and demonstration purposes only.</em>
    </div>
    """,
    unsafe_allow_html=True,
)
