import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import io
from matplotlib.backends.backend_pdf import PdfPages

# 1. Page Configuration
st.set_page_config(page_title="Advanced Probability Applet", layout="centered")

st.title("📊 Probability Distributions")
st.markdown("Designed for Windows, iPad, and iPhone via Streamlit Cloud.")

# 2. Sidebar Inputs
with st.sidebar:
    st.header("1. Distribution Specs")
    dist_name = st.selectbox("Select Distribution", ["Normal", "Student's t", "Chi-Square"])
    
    if dist_name in ["Normal", "Student's t"]:
        mu = st.number_input("Mean (μ)", value=0.0, step=0.1)
        sigma = st.number_input("Std Dev (σ)", value=1.0, step=0.1, min_value=0.01)
    
    if dist_name in ["Student's t", "Chi-Square"]:
        n = st.number_input("Sample Size (n)", value=10, min_value=2, step=1)
        df = n - 1

    st.header("2. Probability Type")
    mode = st.selectbox("Type", ["Unidirectional", "AND", "OR"])
    
    st.header("3. Visual Options")
    show_comp = False
    if dist_name == "Student's t":
        show_comp = st.checkbox("Show Normal Comparison")

# 3. Probability Bounds UI
st.subheader("Specify Bounds")
col1, col2 = st.columns(2)

with col1:
    if mode == "Unidirectional":
        op1 = st.selectbox("Direction", ["≤", "≥"])
        v1 = st.number_input("Value", value=0.0)
    elif mode == "AND":
        v1 = st.number_input("Lower Bound", value=-1.0)
        v2 = st.number_input("Upper Bound", value=1.0)
    elif mode == "OR":
        op1 = st.selectbox("First Direction", ["≤", "≥"])
        v1 = st.number_input("First Value", value=-2.0)

with col2:
    if mode == "OR":
        op2 = st.selectbox("Second Direction", ["≤", "≥"], index=1)
        v2 = st.number_input("Second Value", value=2.0)

# 4. Calculation and Plotting
try:
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if dist_name in ["Normal", "Student's t"]:
        x = np.linspace(mu - 4.5*sigma, mu + 4.5*sigma, 500)
        dist = stats.norm(mu, sigma) if dist_name == "Normal" else stats.t(df, mu, sigma)
        
        # Visual Guides
        ax.axvline(mu, color='blue', linestyle='-', lw=1.5, alpha=0.5, label="Mean")
        for i in [1, 2, 3]:
            ax.axvline(mu + i*sigma, color='red', ls=':', lw=1, alpha=0.3)
            ax.axvline(mu - i*sigma, color='red', ls=':', lw=1, alpha=0.3)
            
        if show_comp:
            ax.plot(x, stats.norm.pdf(x, mu, sigma), color='gray', ls='--', alpha=0.5, label="Normal Ref")
    else:
        dist = stats.chi2(df)
        x = np.linspace(0.001, stats.chi2.ppf(0.999, df), 500)
        ax.axvline(df, color='blue', linestyle='-', lw=1.5, alpha=0.5, label="Mean (df)")

    y = dist.pdf(x)
    ax.plot(x, y, color='blue', lw=2.5, label=dist_name)
    ax.axhline(0, color='black', lw=1.5)

    # Shading logic
    ticks, prob, expr = [], 0.0, ""
    if mode == "Unidirectional":
        ticks = [v1]
        if op1 == "≤":
            prob, expr = dist.cdf(v1), f"P(x ≤ {v1})"
            ax.fill_between(x, y, where=(x <= v1), color='orange', alpha=0.5)
        else:
            prob, expr = 1 - dist.cdf(v1), f"P(x ≥ {v1})"
            ax.fill_between(x, y, where=(x >= v1), color='orange', alpha=0.5)
    elif mode == "AND":
        ticks = [v1, v2]
        prob, expr = dist.cdf(v2) - dist.cdf(v1), f"P({v1} ≤ x ≤ {v2})"
        ax.fill_between(x, y, where=(x >= v1) & (x <= v2), color='orange', alpha=0.5)
    elif mode == "OR":
        ticks = [v1, v2]
        p1 = dist.cdf(v1) if op1 == "≤" else 1 - dist.cdf(v1)
        p2 = dist.cdf(v2) if op2 == "≤" else 1 - dist.cdf(v2)
        prob, expr = p1 + p2, f"P(x {op1} {v1}) OR P(x {op2} {v2})"
        m1 = (x <= v1) if op1 == "≤" else (x >= v1)
        m2 = (x <= v2) if op2 == "≤" else (x >= v2)
        ax.fill_between(x, y, where=m1 | m2, color='orange', alpha=0.5)

    ax.set_xticks(ticks)
    ax.set_yticks([])
    ax.legend()
    st.pyplot(fig)

    # 5. Results Display
    st.success(f"**Result:** {expr} = **{prob:.4f}**")
    
    result_summary = f"Expression: {expr} = {prob:.4f}\n"
    if dist_name == "Normal":
        z_scores = [f"{(t-mu)/sigma:.3f}" for t in ticks]
        st.info(f"**Z-scores:** {', '.join(z_scores)}")
        result_summary += f"Z-scores: {', '.join(z_scores)}\nMean: {mu}, SD: {sigma}"
    else:
        st.info(f"**Parameters:** n={n}, df={df}")
        result_summary += f"Sample Size (n): {n}\nDegrees of Freedom (df): {df}"

    # 6. PDF and Image Generator (Updated for Cloud)
    st.divider()
    st.subheader("📤 Export Options")
    
    # PNG Download
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", dpi=300)
    st.download_button("Download Plot (PNG)", data=img_buf.getvalue(), file_name="stat_plot.png", mime="image/png")

    # PDF Download (Full Report)
    pdf_buf = io.BytesIO()
    with PdfPages(pdf_buf) as pdf:
        # We create a new figure specifically for the PDF layout
        report_fig = plt.figure(figsize=(8.5, 11))
        plt.figtext(0.1, 0.9, "Statistical Analysis Report", fontsize=16, fontweight='bold')
        plt.figtext(0.1, 0.8, result_summary, fontsize=12, family='monospace', verticalalignment='top')
        
        # Re-plot on the PDF page
        ax_pdf = report_fig.add_axes([0.1, 0.2, 0.8, 0.4])
        # Simplify the PDF plot for clarity
        ax_pdf.plot(x, y, color='blue')
        ax_pdf.fill_between(x, y, color='orange', alpha=0.5) # Simplified shading
        ax_pdf.set_xticks(ticks)
        ax_pdf.set_title(f"{dist_name} Distribution")
        
        pdf.savefig(report_fig)
        plt.close(report_fig)

    st.download_button(
        label="Download Full Report (PDF)",
        data=pdf_buf.getvalue(),
        file_name="stat_report.pdf",
        mime="application/pdf"
    )

except Exception as e:
    st.error(f"Error in inputs: {e}")