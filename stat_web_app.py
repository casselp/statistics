import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import io
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# 1. Page Configuration
st.set_page_config(page_title="Pro Stat Applet", layout="centered")

# Accessibility Colors
COLOR_MEAN = "#2E7D32" # Green
COLOR_SD = "#C62828"   # Red
COLOR_SHADE = "#FF9800" # Orange

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
        mu = st.number_input("Mean (μ)", value=0.0, format="%.3f")
    else: mu = 0.0
with p_col2:
    if dist_name in ["Normal", "Student's t"]:
        sigma = st.number_input("Std Dev (σ)", value=1.0, min_value=0.01, format="%.3f")
    else: sigma = 1.0
with p_col3:
    if dist_name in ["Student's t", "Chi-Square"]:
        def_n = 25
        n = st.number_input("Sample Size (n)", value=def_n, min_value=2)
        df = n - 1

show_comp = False
if dist_name == "Student's t":
    show_comp = st.checkbox("Show Normal Comparison (Dashed Line)")

st.divider()

# --- 2. ANALYSIS PARAMETERS (With Conditional Defaults) ---
st.header("2. Analysis Parameters")
b_col1, b_col2 = st.columns(2)

is_inverse = "Inverse" in calc_mode

# Determine Dynamic Defaults based on Mode and Distribution
if not is_inverse:
    # Probability Mode Defaults
    d_v1 = -1.0 if dist_name != "Chi-Square" else 9.886
    d_v2 = 1.0 if dist_name != "Chi-Square" else 45.559
    val_label = "Value (x)"
else:
    # Critical Value (Inverse) Mode Defaults
    val_label = "Probability (0 to 1)"
    if prob_mode == "Unidirectional":
        d_v1 = 0.05
    elif prob_mode == "AND":
        d_v1, d_v2 = 0.025, 0.975
    else: # OR
        d_v1, d_v2 = 0.025, 0.025

try:
    if dist_name == "Normal": dist = stats.norm(mu, sigma)
    elif dist_name == "Student's t": dist = stats.t(df, mu, sigma)
    else: dist = stats.chi2(df)

    v1_raw, v2_raw = None, None
    v1_plot, v2_plot = None, None

    with b_col1:
        if prob_mode == "Unidirectional":
            bound_choice = st.selectbox("Bound Type", ["Lower", "Upper"], index=0)
            v1_raw = st.number_input(val_label, value=d_v1, format="%.3f", step=0.001)
            if is_inverse:
                # If Lower (x >= bound), we want the value where right-tail is val_label
                v1_plot = dist.ppf(1 - v1_raw) if bound_choice == "Lower" else dist.ppf(v1_raw)
            else:
                v1_plot = v1_raw
                
        elif prob_mode == "AND":
            v1_raw = st.number_input(f"Lower {val_label}", value=d_v1, format="%.3f")
            v2_raw = st.number_input(f"Upper {val_label}", value=d_v2, format="%.3f")
            if is_inverse:
                v1_plot, v2_plot = dist.ppf(v1_raw), dist.ppf(v2_raw)
            else:
                v1_plot, v2_plot = v1_raw, v2_raw
                
        elif prob_mode == "OR":
            st.write("**Left-tail Bound (x)**")
            v1_raw = st.number_input(val_label, value=d_v1, format="%.3f", key="v1_or", label_visibility="collapsed")
            if is_inverse:
                v1_plot = dist.ppf(v1_raw)
            else:
                v1_plot = v1_raw

    with b_col2:
        if prob_mode == "OR":
            st.write("**Right-tail Bound (x)**")
            v2_raw = st.number_input(val_label, value=d_v2, format="%.3f", key="v2_or", label_visibility="collapsed")
            if is_inverse:
                v2_plot = dist.ppf(1 - v2_raw)
            else:
                v2_plot = v2_raw

    # --- 3. GRAPHICAL AREA ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x_min = mu - 4.5*sigma if dist_name != "Chi-Square" else 0.001
    x_max = mu + 4.5*sigma if dist_name != "Chi-Square" else dist.ppf(0.999)
    x_plot = np.linspace(x_min, x_max, 1000)
    y_plot = dist.pdf(x_plot)
    
    ax.plot(x_plot, y_plot, color='#1565C0', lw=2.5)

    # Statistical Markers
    if dist_name in ["Normal", "Student's t"]:
        ax.axvline(mu, color=COLOR_MEAN, lw=2, label=f'Mean: {mu:.2f}')
        for i in [1, 2, 3]:
            ax.axvline(mu + i*sigma, color=COLOR_SD, ls='--', lw=1, alpha=0.4)
            ax.axvline(mu - i*sigma, color=COLOR_SD, ls='--', lw=1, alpha=0.4)
        if show_comp:
            ax.plot(x_plot, stats.norm.pdf(x_plot, mu, sigma), color='gray', ls=':', label="Normal Ref")

    # Shading & Calculation
    ticks = [v for v in [v1_plot, v2_plot] if v is not None]
    if prob_mode == "Unidirectional":
        mask = (x_plot >= v1_plot) if bound_choice == "Lower" else (x_plot <= v1_plot)
        res_val = 1 - dist.cdf(v1_plot) if bound_choice == "Lower" else dist.cdf(v1_plot)
        ax.fill_between(x_plot, y_plot, where=mask, color=COLOR_SHADE, alpha=0.5)
    elif prob_mode == "AND":
        ax.fill_between(x_plot, y_plot, where=(x_plot >= v1_plot) & (x_plot <= v2_plot), color=COLOR_SHADE, alpha=0.5)
        res_val = dist.cdf(v2_plot) - dist.cdf(v1_plot)
    else: # OR
        mask = (x_plot <= v1_plot) | (x_plot >= v2_plot)
        ax.fill_between(x_plot, y_plot, where=mask, color=COLOR_SHADE, alpha=0.5)
        res_val = dist.cdf(v1_plot) + (1 - dist.cdf(v2_plot))

    # X-Axis Labels (Rotated 90 for Mobile)
    ax.set_xticks(ticks)
    if dist_name == "Normal":
        labels = [f"x={v:.2f}\nz={(v-mu)/sigma:.2f}" for v in ticks]
    else:
        labels = [f"{v:.3f}" for v in ticks]
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks([])
    ax.legend(prop={'size': 8}, loc='upper right')
    st.pyplot(fig)

    # Result Metric
    if is_inverse:
        st.success(f"**Critical Value(s) Found:** {', '.join([f'{t:.3f}' for t in ticks])}")
    else:
        st.success(f"**Calculated Probability:** {res_val:.4f}")

    # --- 4. EXPORTS ---
    st.divider()
    e1, e2, e3 = st.columns(3)
    with e1:
        img_buf = io.BytesIO(); fig.savefig(img_buf, format="png", dpi=300)
        st.download_button("💾 Download Image", img_buf.getvalue(), "plot.png", "image/png", help="Long-press image in browser to 'Save to Photos'")
    with e2:
        pdf_buf = io.BytesIO()
        with PdfPages(pdf_buf) as pdf:
            rpt = plt.figure(figsize=(8.5, 11))
            plt.figtext(0.1, 0.9, f"Stat Report - {dist_name}", fontsize=14, fontweight='bold')
            plt.figtext(0.1, 0.85, f"Goal: {calc_mode} | Result: {res_val:.4f}")
            pdf.savefig(rpt); plt.close(rpt)
        st.download_button("📄 Download PDF", pdf_buf.getvalue(), "report.pdf")
    with e3:
        st.button("🔄 Reset to Defaults", on_click=reset_app)

except Exception as e:
    st.error("Please ensure Probability inputs are between 0 and 1 for Inverse mode.")