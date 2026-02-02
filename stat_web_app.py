import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import io
from matplotlib.backends.backend_pdf import PdfPages

st.set_page_config(page_title="Advanced Probability Applet", layout="centered")

st.title("📊 Probability Distributions")

# --- 1. DISTRIBUTION SETTINGS ---
st.header("1. Distribution Settings")
col_dist, col_type = st.columns(2)

with col_dist:
    dist_name = st.selectbox("Select Distribution", ["Normal", "Student's t", "Chi-Square"])
    
with col_type:
    mode = st.selectbox("Probability Mode", ["Unidirectional", "AND", "OR"])

# Parameters
p_col1, p_col2, p_col3 = st.columns(3)
with p_col1:
    if dist_name in ["Normal", "Student's t"]:
        mu = st.number_input("Mean (μ)", value=0.0, step=0.1, format="%.3f")
with p_col2:
    if dist_name in ["Normal", "Student's t"]:
        sigma = st.number_input("Std Dev (σ)", value=1.0, step=0.1, min_value=0.01, format="%.3f")
with p_col3:
    if dist_name in ["Student's t", "Chi-Square"]:
        n = st.number_input("Sample Size (n)", value=10, min_value=2)
        df = n - 1

# T-Distribution visual option
show_comp = False
if dist_name == "Student's t":
    show_comp = st.checkbox("Show Normal Comparison (Dashed Line)")

st.divider()

# --- 2. SPECIFY PROBABILITY (Fixed for 3 Decimals) ---
st.header("2. Specify Probability")
b_col1, b_col2 = st.columns(2)

# Use format="%.3f" to ensure 3 decimal places are displayed
# Use step=0.001 to allow fine-tuned increments
with b_col1:
    if mode == "Unidirectional":
        op1 = st.selectbox("Direction", ["≤", "≥"])
        v1 = st.number_input("Value", value=0.0, format="%.3f", step=0.001)
    elif mode == "AND":
        v1 = st.number_input("Lower Bound", value=-1.0, format="%.3f", step=0.001)
        v2 = st.number_input("Upper Bound", value=1.0, format="%.3f", step=0.001)
    elif mode == "OR":
        op1 = st.selectbox("First Direction", ["≤", "≥"])
        v1 = st.number_input("First Value", value=-2.0, format="%.3f", step=0.001)

with b_col2:
    if mode == "OR":
        op2 = st.selectbox("Second Direction", ["≤", "≥"], index=1)
        v2 = st.number_input("Second Value", value=2.0, format="%.3f", step=0.001)

# --- 3. PLOTTING AND CALCULATIONS ---
try:
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if dist_name in ["Normal", "Student's t"]:
        x = np.linspace(mu - 4.5*sigma, mu + 4.5*sigma, 500)
        dist = stats.norm(mu, sigma) if dist_name == "Normal" else stats.t(df, mu, sigma)
        ax.axvline(mu, color='blue', linestyle='-', lw=1.5, alpha=0.5, label="Mean")
        if show_comp:
            ax.plot(x, stats.norm.pdf(x, mu, sigma), color='gray', ls='--', alpha=0.6, label="Normal Ref")
    else:
        dist = stats.chi2(df)
        x = np.linspace(0.001, stats.chi2.ppf(0.999, df), 500)
        ax.axvline(df, color='blue', linestyle='-', lw=1.5, alpha=0.5, label="Mean (df)")

    y = dist.pdf(x)
    ax.plot(x, y, color='blue', lw=2.5, label=dist_name)
    ax.axhline(0, color='black', lw=1.5)

    ticks, prob, expr = [], 0.0, ""
    if mode == "Unidirectional":
        ticks = [v1]
        if op1 == "≤":
            prob, expr = dist.cdf(v1), f"P(x ≤ {v1:.3f})"
            ax.fill_between(x, y, where=(x <= v1), color='orange', alpha=0.5)
        else:
            prob, expr = 1 - dist.cdf(v1), f"P(x ≥ {v1:.3f})"
            ax.fill_between(x, y, where=(x >= v1), color='orange', alpha=0.5)
    elif mode == "AND":
        ticks = [v1, v2]
        prob, expr = dist.cdf(v2) - dist.cdf(v1), f"P({v1:.3f} ≤ x ≤ {v2:.3f})"
        ax.fill_between(x, y, where=(x >= v1) & (x <= v2), color='orange', alpha=0.5)
    elif mode == "OR":
        ticks = [v1, v2]
        p1 = dist.cdf(v1) if op1 == "≤" else 1 - dist.cdf(v1)
        p2 = dist.cdf(v2) if op2 == "≤" else 1 - dist.cdf(v2)
        prob, expr = p1 + p2, f"P(x {op1} {v1:.3f}) OR P(x {op2} {v2:.3f})"
        m1 = (x <= v1) if op1 == "≤" else (x >= v1)
        m2 = (x <= v2) if op2 == "≤" else (x >= v2)
        ax.fill_between(x, y, where=m1 | m2, color='orange', alpha=0.5)

    ax.set_xticks(ticks)
    # Ensure ticks show 3 decimals on the plot too
    ax.set_xticklabels([f"{t:.3f}" for t in ticks], rotation=45)
    ax.set_yticks([])
    ax.legend()
    st.pyplot(fig)

    st.success(f"**Result:** {expr} = **{prob:.4f}**")
    
    # Export Options
    st.divider()
    st.subheader("📤 Export Options")
    
    # PNG Download
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", dpi=300)
    st.download_button("Download Plot (PNG)", data=img_buf.getvalue(), file_name="stat_plot.png")

    # PDF Report Download
    pdf_buf = io.BytesIO()
    with PdfPages(pdf_buf) as pdf:
        report_fig = plt.figure(figsize=(8.5, 11))
        plt.figtext(0.1, 0.9, "Statistical Analysis Report", fontsize=16, fontweight='bold')
        summary_text = f"Result: {expr} = {prob:.4f}\nDist: {dist_name}\ndf: {df if 'df' in locals() else 'N/A'}"
        plt.figtext(0.1, 0.8, summary_text, fontsize=12, family='monospace', verticalalignment='top')
        ax_pdf = report_fig.add_axes([0.1, 0.2, 0.8, 0.4])
        ax_pdf.plot(x, y, color='blue')
        ax_pdf.fill_between(x, y, color='orange', alpha=0.5)
        ax_pdf.set_xticks(ticks)
        pdf.savefig(report_fig)
        plt.close(report_fig)

    st.download_button("Download Full Report (PDF)", data=pdf_buf.getvalue(), file_name="stat_report.pdf")

except Exception as e:
    st.info("Please enter valid boundary values.")