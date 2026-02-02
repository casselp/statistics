import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import io
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# 1. Page Configuration
st.set_page_config(page_title="Pro Stat Applet", layout="centered")

st.markdown("""
    <style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Pro Probability Applet")

# --- RESET LOGIC ---
if 'reset_key' not in st.session_state:
    st.session_state.reset_key = 0

def reset_app():
    st.session_state.reset_key += 1
    st.rerun()

# --- 1. DISTRIBUTION SETTINGS ---
st.header("1. Distribution Settings")
col_dist, col_type, col_mode = st.columns([2, 2, 2])

with col_dist:
    dist_name = st.selectbox("Distribution", ["Normal", "Student's t", "Chi-Square"], key=f"dist_{st.session_state.reset_key}")
    
with col_type:
    calc_mode = st.radio("Goal", ["Find Probability", "Find Critical Value (Inverse)"], key=f"calc_{st.session_state.reset_key}")

with col_mode:
    prob_mode = st.selectbox("Interval Type", ["Unidirectional", "AND", "OR"], key=f"prob_{st.session_state.reset_key}")

p_col1, p_col2, p_col3 = st.columns(3)
with p_col1:
    if dist_name in ["Normal", "Student's t"]:
        mu = st.number_input("Mean (μ)", value=0.0, step=0.1, format="%.3f", key=f"mu_{st.session_state.reset_key}")
with p_col2:
    if dist_name in ["Normal", "Student's t"]:
        sigma = st.number_input("Std Dev (σ)", value=1.0, step=0.1, min_value=0.01, format="%.3f", key=f"sig_{st.session_state.reset_key}")
with p_col3:
    if dist_name in ["Student's t", "Chi-Square"]:
        n = st.number_input("Sample Size (n)", value=10, min_value=2, key=f"n_{st.session_state.reset_key}")
        df = n - 1

show_comp = False
if dist_name == "Student's t":
    show_comp = st.checkbox("Show Normal Ref (Dashed)", key=f"comp_{st.session_state.reset_key}")

st.divider()

# --- 2. ANALYSIS PARAMETERS ---
st.header("2. Analysis Parameters")
b_col1, b_col2 = st.columns(2)

try:
    if dist_name in ["Normal", "Student's t"]:
        dist = stats.norm(mu, sigma) if dist_name == "Normal" else stats.t(df, mu, sigma)
    else:
        dist = stats.chi2(df)

    v1, v2 = None, None
    input_label = "Probability (0 to 1)" if "Inverse" in calc_mode else "Value (x)"

    with b_col1:
        if prob_mode == "Unidirectional":
            op1 = st.selectbox("Direction", ["≤", "≥"], key=f"op1_{st.session_state.reset_key}")
            val1_in = st.number_input(input_label, value=0.05 if "Inverse" in calc_mode else 0.0, format="%.3f", step=0.001)
            v1 = dist.ppf(val1_in) if "Inverse" in calc_mode and op1 == "≤" else (dist.ppf(1-val1_in) if "Inverse" in calc_mode else val1_in)
        elif prob_mode == "AND":
            v1_in = st.number_input(f"Lower {input_label}", value=0.025 if "Inverse" in calc_mode else -1.0, format="%.3f")
            v2_in = st.number_input(f"Upper {input_label}", value=0.975 if "Inverse" in calc_mode else 1.0, format="%.3f")
            v1, v2 = (dist.ppf(v1_in), dist.ppf(v2_in)) if "Inverse" in calc_mode else (v1_in, v2_in)
        elif prob_mode == "OR":
            op1 = st.selectbox("First Direction", ["≤", "≥"])
            v1_in = st.number_input(f"First {input_label}", value=0.025 if "Inverse" in calc_mode else -2.0, format="%.3f")
            v1 = dist.ppf(v1_in) if "Inverse" in calc_mode and op1 == "≤" else (dist.ppf(1-v1_in) if "Inverse" in calc_mode else v1_in)

    with b_col2:
        if prob_mode == "OR":
            op2 = st.selectbox("Second Direction", ["≤", "≥"], index=1)
            v2_in = st.number_input(f"Second {input_label}", value=0.025 if "Inverse" in calc_mode else 2.0, format="%.3f")
            v2 = dist.ppf(v2_in) if "Inverse" in calc_mode and op2 == "≤" else (dist.ppf(1-v2_in) if "Inverse" in calc_mode else v2_in)

    # --- 3. PLOTTING ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x_plot = np.linspace(mu - 4.5*sigma, mu + 4.5*sigma, 500) if dist_name != "Chi-Square" else np.linspace(0.001, stats.chi2.ppf(0.999, df), 500)
    y_plot = dist.pdf(x_plot)
    ax.plot(x_plot, y_plot, color='blue', lw=2)
    
    ticks = [v for v in [v1, v2] if v is not None]
    
    # Dual Axis Labels for Normal Distribution
    if dist_name == "Normal":
        labels = [f"x={v:.2f}\nz={((v-mu)/sigma):.2f}" for v in ticks]
    else:
        labels = [f"{v:.3f}" for v in ticks]

    # Shading Logic
    if prob_mode == "Unidirectional":
        mask = (x_plot <= v1) if op1 == "≤" else (x_plot >= v1)
        res_prob = dist.cdf(v1) if op1 == "≤" else 1 - dist.cdf(v1)
        ax.fill_between(x_plot, y_plot, where=mask, color='orange', alpha=0.5)
        res_text = f"P(x {op1} {v1:.3f}) = {res_prob:.4f}"
    elif prob_mode == "AND":
        ax.fill_between(x_plot, y_plot, where=(x_plot >= v1) & (x_plot <= v2), color='orange', alpha=0.5)
        res_prob = dist.cdf(v2) - dist.cdf(v1)
        res_text = f"P({v1:.3f} ≤ x ≤ {v2:.3f}) = {res_prob:.4f}"
    else: # OR
        m1 = (x_plot <= v1) if op1 == "≤" else (x_plot >= v1)
        m2 = (x_plot <= v2) if op2 == "≤" else (x_plot >= v2)
        ax.fill_between(x_plot, y_plot, where=m1 | m2, color='orange', alpha=0.5)
        p1 = dist.cdf(v1) if op1 == "≤" else 1 - dist.cdf(v1)
        p2 = dist.cdf(v2) if op2 == "≤" else 1 - dist.cdf(v2)
        res_prob = p1 + p2
        res_text = f"P(x {op1} {v1:.3f}) OR P(x {op2} {v2:.3f}) = {res_prob:.4f}"

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([])
    st.pyplot(fig)

    # --- 4. RESULTS & SUMMARY ---
    st.success(f"**Result:** {res_text}")
    
    if dist_name == "Normal":
        z_col1, z_col2 = st.columns(2)
        z_col1.metric("Z-score (v1)", f"{(v1-mu)/sigma:.3f}")
        if v2 is not None: z_col2.metric("Z-score (v2)", f"{(v2-mu)/sigma:.3f}")

    st.subheader("📋 Distribution Summary")
    summary_df = pd.DataFrame({
        "Stat": ["Mean", "Variance", "Median", "IQR (25th-75th)"],
        "Value": [f"{dist.mean():.3f}", f"{dist.var():.3f}", f"{dist.median():.3f}", f"{dist.ppf(0.25):.2f} to {dist.ppf(0.75):.2f}"]
    })
    st.table(summary_df)

    # --- 5. EXPORTS ---
    st.divider()
    e1, e2, e3 = st.columns(3)
    with e1:
        img_buf = io.BytesIO(); fig.savefig(img_buf, format="png")
        st.download_button("💾 Save Image", img_buf.getvalue(), "plot.png")
    with e2:
        pdf_buf = io.BytesIO()
        with PdfPages(pdf_buf) as pdf:
            rpt = plt.figure(figsize=(8.5, 11))
            plt.figtext(0.1, 0.9, f"Report: {dist_name}", fontsize=14, fontweight='bold')
            plt.figtext(0.1, 0.85, res_text, family='monospace')
            pdf.savefig(rpt); plt.close(rpt)
        st.download_button("📄 Save PDF", pdf_buf.getvalue(), "report.pdf")
    with e3:
        st.button("🔄 Reset App", on_click=reset_app)

except Exception as e:
    st.info("Awaiting valid inputs. Probabilities must be between 0 and 1.")