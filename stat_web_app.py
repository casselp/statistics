import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import io
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# 1. Page Configuration
st.set_page_config(page_title="Pro Stat Applet", layout="centered")

# --- CUSTOM CSS FOR BETTER MOBILE VISIBILITY ---
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
    # ADDED: Inverse Lookup Mode
    calc_mode = st.radio("Goal", ["Find Probability", "Find Critical Value (Inverse)"], key=f"calc_{st.session_state.reset_key}")

with col_mode:
    prob_mode = st.selectbox("Interval Type", ["Unidirectional", "AND", "OR"], key=f"prob_{st.session_state.reset_key}")

# Parameters
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

# --- 2. SPECIFY INPUTS (PROBABILITY OR CRITICAL VALUE) ---
st.header("2. Analysis Parameters")
b_col1, b_col2 = st.columns(2)

try:
    if dist_name in ["Normal", "Student's t"]:
        dist = stats.norm(mu, sigma) if dist_name == "Normal" else stats.t(df, mu, sigma)
    else:
        dist = stats.chi2(df)

    v1, v2 = None, None
    input_label = "Probability (0 to 1)" if calc_mode == "Find Critical Value (Inverse)" else "Boundary Value (x)"

    with b_col1:
        if prob_mode == "Unidirectional":
            op1 = st.selectbox("Direction", ["≤", "≥"], key=f"op1_{st.session_state.reset_key}")
            val1_input = st.number_input(input_label, value=0.05 if "Inverse" in calc_mode else 0.0, format="%.3f", step=0.01 if "Inverse" in calc_mode else 0.001)
            
            if "Inverse" in calc_mode:
                v1 = dist.ppf(val1_input) if op1 == "≤" else dist.ppf(1 - val1_input)
            else:
                v1 = val1_input

        elif prob_mode == "AND":
            val1_input = st.number_input(f"Lower {input_label}", value=0.025 if "Inverse" in calc_mode else -1.0, format="%.3f")
            val2_input = st.number_input(f"Upper {input_label}", value=0.975 if "Inverse" in calc_mode else 1.0, format="%.3f")
            
            if "Inverse" in calc_mode:
                v1, v2 = dist.ppf(val1_input), dist.ppf(val2_input)
            else:
                v1, v2 = val1_input, val2_input

        elif prob_mode == "OR":
            op1 = st.selectbox("First Direction", ["≤", "≥"], key="op1_or")
            val1_input = st.number_input(f"First {input_label}", value=0.025 if "Inverse" in calc_mode else -2.0, format="%.3f")
            if "Inverse" in calc_mode:
                v1 = dist.ppf(val1_input) if op1 == "≤" else dist.ppf(1 - val1_input)
            else:
                v1 = val1_input

    with b_col2:
        if prob_mode == "OR":
            op2 = st.selectbox("Second Direction", ["≤", "≥"], index=1, key="op2_or")
            val2_input = st.number_input(f"Second {input_label}", value=0.025 if "Inverse" in calc_mode else 2.0, format="%.3f")
            if "Inverse" in calc_mode:
                v2 = dist.ppf(val2_input) if op2 == "≤" else dist.ppf(1 - val2_input)
            else:
                v2 = val2_input

    # --- 3. PLOTTING ---
    fig, ax = plt.subplots(figsize=(10, 4.5))
    
    # Define X range for plot
    if dist_name != "Chi-Square":
        x_plot = np.linspace(mu - 4.5*sigma, mu + 4.5*sigma, 500)
    else:
        x_plot = np.linspace(0.001, stats.chi2.ppf(0.999, df), 500)

    y_plot = dist.pdf(x_plot)
    ax.plot(x_plot, y_plot, color='blue', lw=2)
    
    if show_comp and dist_name == "Student's t":
        ax.plot(x_plot, stats.norm.pdf(x_plot, mu, sigma), color='gray', ls='--', alpha=0.5)

    # Shading and Results
    ticks = []
    if prob_mode == "Unidirectional":
        ticks = [v1]
        mask = (x_plot <= v1) if op1 == "≤" else (x_plot >= v1)
        res_prob = dist.cdf(v1) if op1 == "≤" else 1 - dist.cdf(v1)
        ax.fill_between(x_plot, y_plot, where=mask, color='orange', alpha=0.5)
        res_text = f"P(x {op1} {v1:.3f}) = {res_prob:.4f}"
    elif prob_mode == "AND":
        ticks = [v1, v2]
        ax.fill_between(x_plot, y_plot, where=(x_plot >= v1) & (x_plot <= v2), color='orange', alpha=0.5)
        res_prob = dist.cdf(v2) - dist.cdf(v1)
        res_text = f"P({v1:.3f} ≤ x ≤ {v2:.3f}) = {res_prob:.4f}"
    else: # OR
        ticks = [v1, v2]
        m1 = (x_plot <= v1) if op1 == "≤" else (x_plot >= v1)
        m2 = (x_plot <= v2) if op2 == "≤" else (x_plot >= v2)
        ax.fill_between(x_plot, y_plot, where=m1 | m2, color='orange', alpha=0.5)
        p1 = dist.cdf(v1) if op1 == "≤" else 1 - dist.cdf(v1)
        p2 = dist.cdf(v2) if op2 == "≤" else 1 - dist.cdf(v2)
        res_prob = p1 + p2
        res_text = f"P(x {op1} {v1:.3f}) OR P(x {op2} {v2:.3f}) = {res_prob:.4f}"

    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.3f}" for t in ticks], rotation=45)
    ax.set_yticks([])
    st.pyplot(fig)

    # --- 4. SUMMARY METRICS & Z-SCORES ---
    st.success(f"**Main Result:** {res_text}")

    # ADDED: Automatic Z-Scores for Normal
    if dist_name == "Normal":
        z_col1, z_col2 = st.columns(2)
        z1 = (v1 - mu) / sigma
        z_col1.metric("Z-score (v1)", f"{z1:.3f}")
        if v2 is not None:
            z2 = (v2 - mu) / sigma
            z_col2.metric("Z-score (v2)", f"{z2:.3f}")

    # ADDED: Summary Table
    st.subheader("📋 Distribution Summary")
    summary_data = {
        "Stat": ["Mean", "Variance", "Median", "25th Perc.", "75th Perc."],
        "Value": [
            f"{dist.mean():.3f}", 
            f"{dist.var():.3f}", 
            f"{dist.median():.3f}", 
            f"{dist.ppf(0.25):.3f}", 
            f"{dist.ppf(0.75):.3f}"
        ]
    }
    st.table(pd.DataFrame(summary_data))

    # --- 5. EXPORTS & RESET ---
    st.divider()
    e1, e2, e3 = st.columns(3)
    
    with e1:
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png")
        st.download_button("💾 Save Image", img_buf.getvalue(), "plot.png")
    
    with e2:
        pdf_buf = io.BytesIO()
        with PdfPages(pdf_buf) as pdf:
            report = plt.figure(figsize=(8.5, 11))
            plt.figtext(0.1, 0.9, f"Stat Report: {dist_name}", fontsize=16, fontweight='bold')
            plt.figtext(0.1, 0.8, res_text, fontsize=12, family='monospace')
            pdf.savefig(report)
            plt.close(report)
        st.download_button("📄 Save PDF", pdf_buf.getvalue(), "report.pdf")

    with e3:
        st.button("🔄 Reset App", on_click=reset_app)

except Exception as e:
    st.info("Check parameters. For Inverse Mode, probabilities must be between 0 and 1.")