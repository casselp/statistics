import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import io
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# 1. Page Configuration
st.set_page_config(page_title="Pro Stat Applet", layout="centered")

# Custom Accessibility Colors
COLOR_MEAN = "#2E7D32" # Dark Green
COLOR_SD = "#C62828"   # Dark Red
COLOR_SHADE = "#FF9800" # Amber/Orange

# Reset Logic
if 'reset_key' not in st.session_state:
    st.session_state.reset_key = 0

def reset_app():
    st.session_state.reset_key += 1
    st.rerun()

st.title("📊 Pro Probability Applet")

# --- 1. DISTRIBUTION SETTINGS ---
st.header("1. Distribution Settings")
col_dist, col_goal, col_type = st.columns(3)

with col_dist:
    dist_name = st.selectbox("Distribution", ["Normal", "Student's t", "Chi-Square"], index=0, key=f"dist_{st.session_state.reset_key}")
with col_goal:
    calc_mode = st.radio("Goal", ["Find Probability", "Find Critical Value (Inverse)"], index=0, key=f"goal_{st.session_state.reset_key}")
with col_type:
    prob_mode = st.selectbox("Interval Type", ["Unidirectional", "AND", "OR"], index=0, key=f"type_{st.session_state.reset_key}")

# Parameter Defaults
p_col1, p_col2, p_col3 = st.columns(3)
with p_col1:
    if dist_name in ["Normal", "Student's t"]:
        mu = st.number_input("Mean (μ)", value=0.0, format="%.3f", key=f"mu_{st.session_state.reset_key}")
    else: mu = 0.0
with p_col2:
    if dist_name in ["Normal", "Student's t"]:
        sigma = st.number_input("Std Dev (σ)", value=1.0, min_value=0.01, format="%.3f", key=f"sig_{st.session_state.reset_key}")
    else: sigma = 1.0
with p_col3:
    if dist_name in ["Student's t", "Chi-Square"]:
        def_n = 25
        n = st.number_input("Sample Size (n)", value=def_n, min_value=2, key=f"n_{st.session_state.reset_key}")
        df = n - 1

# Normal Comparison Logic for t-distribution
show_comp = False
if dist_name == "Student's t":
    show_comp = st.checkbox("Show Normal Comparison (Dashed Line)", key=f"comp_{st.session_state.reset_key}")

st.divider()

# --- 2. ANALYSIS PARAMETERS ---
st.header("2. Analysis Parameters")
b_col1, b_col2 = st.columns(2)

# Dynamic Defaults based on distribution
if dist_name in ["Normal", "Student's t"]:
    d_v1, d_v2 = -1.0, 1.0
else: # Chi-Square
    d_v1, d_v2 = 9.886, 45.559

try:
    if dist_name == "Normal": dist = stats.norm(mu, sigma)
    elif dist_name == "Student's t": dist = stats.t(df, mu, sigma)
    else: dist = stats.chi2(df)

    v1, v2 = None, None
    op1, op2 = "≤", "≥" # Internal math defaults

    with b_col1:
        if prob_mode == "Unidirectional":
            bound_choice = st.selectbox("Bound Type", ["Lower", "Upper"], index=0)
            op1 = "≥" if bound_choice == "Lower" else "≤"
            v1 = st.number_input("Value (x)", value=d_v1, format="%.3f", step=0.001)
        elif prob_mode == "AND":
            v1 = st.number_input("Lower Value", value=d_v1, format="%.3f")
            v2 = st.number_input("Upper Value", value=d_v2, format="%.3f")
        elif prob_mode == "OR":
            st.write("**Left-tail Bound (x)**")
            v1 = st.number_input("Value (x)", value=d_v1, format="%.3f", key="v1_or", label_visibility="collapsed")
            op1 = "≤"

    with b_col2:
        if prob_mode == "OR":
            st.write("**Right-tail Bound (x)**")
            v2 = st.number_input("Value (x)", value=d_v2, format="%.3f", key="v2_or", label_visibility="collapsed")
            op2 = "≥"

    # Inverse Logic override if selected
    if "Inverse" in calc_mode:
        st.warning("Note: In Inverse Mode, input value represents Area/Probability.")
        if v1 is not None: v1 = dist.ppf(v1) if op1 == "≤" else dist.ppf(1-v1)
        if v2 is not None: v2 = dist.ppf(v2) if op2 == "≤" else dist.ppf(1-v2)

    # --- 3. GRAPHICAL AREA ---
    fig, ax = plt.subplots(figsize=(10, 5))
    if dist_name != "Chi-Square":
        x_plot = np.linspace(mu - 4.5*sigma, mu + 4.5*sigma, 1000)
    else:
        x_plot = np.linspace(0.001, dist.ppf(0.999), 1000)
    
    y_plot = dist.pdf(x_plot)
    ax.plot(x_plot, y_plot, color='#1565C0', lw=2.5) # Strong blue

    # Vertical Bars & Mean Label
    if dist_name in ["Normal", "Student's t"]:
        ax.axvline(mu, color=COLOR_MEAN, lw=2, label=f'Mean: {mu:.2f}')
        for i in [1, 2, 3]:
            ax.axvline(mu + i*sigma, color=COLOR_SD, ls='--', lw=1, alpha=0.6)
            ax.axvline(mu - i*sigma, color=COLOR_SD, ls='--', lw=1, alpha=0.6)
        
        if show_comp:
            ax.plot(x_plot, stats.norm.pdf(x_plot, mu, sigma), color='gray', ls=':', label="Normal Comparison")

    # Shading and Calculation
    ticks = [v for v in [v1, v2] if v is not None]
    if prob_mode == "Unidirectional":
        mask = (x_plot >= v1) if op1 == "≥" else (x_plot <= v1)
        res_prob = 1 - dist.cdf(v1) if op1 == "≥" else dist.cdf(v1)
        ax.fill_between(x_plot, y_plot, where=mask, color=COLOR_SHADE, alpha=0.5)
    elif prob_mode == "AND":
        ax.fill_between(x_plot, y_plot, where=(x_plot >= v1) & (x_plot <= v2), color=COLOR_SHADE, alpha=0.5)
        res_prob = dist.cdf(v2) - dist.cdf(v1)
    else: # OR
        mask = (x_plot <= v1) | (x_plot >= v2)
        ax.fill_between(x_plot, y_plot, where=mask, color=COLOR_SHADE, alpha=0.5)
        res_prob = dist.cdf(v1) + (1 - dist.cdf(v2))

    # X-Axis Labels (Rotated 90 degrees)
    ax.set_xticks(ticks)
    if dist_name == "Normal":
        labels = [f"x={v:.2f}\nz={(v-mu)/sigma:.2f}" for v in ticks]
    else:
        labels = [f"{v:.3f}" for v in ticks]
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks([])
    ax.legend(prop={'size': 8})
    st.pyplot(fig)

    st.success(f"**Calculated Probability:** {res_prob:.4f}")

    # --- 4. EXPORTS ---
    st.divider()
    e1, e2, e3 = st.columns(3)
    with e1:
        img_buf = io.BytesIO(); fig.savefig(img_buf, format="png", dpi=300)
        st.download_button("💾 Download Image (Save to Photos)", img_buf.getvalue(), "stat_plot.png", "image/png")
    with e2:
        pdf_buf = io.BytesIO()
        with PdfPages(pdf_buf) as pdf:
            rpt = plt.figure(figsize=(8.5, 11))
            plt.figtext(0.1, 0.9, f"Statistical Report - {dist_name}", fontsize=14, fontweight='bold')
            plt.figtext(0.1, 0.85, f"Probability Result: {res_prob:.4f}")
            pdf.savefig(rpt); plt.close(rpt)
        st.download_button("📄 Download PDF", pdf_buf.getvalue(), "report.pdf")
    with e3:
        st.button("🔄 Reset to Defaults", on_click=reset_app)

except Exception as e:
    st.info("Awaiting valid inputs...")