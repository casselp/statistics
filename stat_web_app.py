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
COLOR_MEAN = "#2E7D32" 
COLOR_SD = "#C62828"   
COLOR_SHADE = "#FF9800" 

if 'reset_key' not in st.session_state:
    st.session_state.reset_key = 0

def reset_app():
    st.session_state.reset_key += 1

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

is_inverse = "Inverse" in calc_mode

# Parameter Defaults Logic
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
        # Default n=25 for t, n=10 for Chi-Square
        def_n = 25 if dist_name == "Student's t" else 10
        n = st.number_input("Sample Size (n)", value=def_n, min_value=2, key=f"n_{st.session_state.reset_key}")
        df = n - 1

show_comp = False
if dist_name == "Student's t":
    show_comp = st.checkbox("Show Normal Comparison (Dashed Line)", key=f"comp_{st.session_state.reset_key}")

st.divider()

# --- 2. ANALYSIS PARAMETERS ---
st.header("2. Analysis Parameters")
b_col1, b_col2 = st.columns(2)

try:
    if dist_name == "Normal": dist = stats.norm(mu, sigma)
    elif dist_name == "Student's t": dist = stats.t(df, mu, sigma)
    else: dist = stats.chi2(df)

    v1_plot, v2_plot = None, None
    res_prob = 0.0

    # UI LOGIC FOR CHI-SQUARE INVERSE
    if is_inverse and dist_name == "Chi-Square":
        with b_col1:
            if prob_mode == "Unidirectional":
                st.markdown(r"$\alpha$ represents the area to the right of the critical value.")
                alpha = st.number_input(r"Probability, $\alpha$", value=0.050, format="%.3f", step=0.001)
                v1_plot = dist.ppf(1 - alpha)
                bound_choice = "Lower" # Math: area to right
            elif prob_mode == "AND":
                st.markdown(r"The area remaining in each tail is $1/2 \alpha$.")
                alpha = st.number_input(r"Probability, $\alpha$", value=0.050, format="%.3f", step=0.001)
                v1_plot, v2_plot = dist.ppf(alpha/2), dist.ppf(1 - alpha/2)
            elif prob_mode == "OR":
                st.markdown(r"The shaded area in each tail is $1/2 \alpha$.")
                alpha = st.number_input(r"Probability, $\alpha$", value=0.050, format="%.3f", step=0.001)
                v1_plot, v2_plot = dist.ppf(alpha/2), dist.ppf(1 - alpha/2)

    # UI LOGIC FOR NORMAL / T / PROBABILITY MODES
    else:
        with b_col1:
            if prob_mode == "Unidirectional":
                bound_choice = st.selectbox("Bound Type", ["Lower", "Upper"], index=1 if not is_inverse else 0)
                label = "Probability (0 to 1)" if is_inverse else "Value (x)"
                
                # Default Logic
                if not is_inverse:
                    d_val = 1.0 if dist_name != "Chi-Square" else 16.919
                else: d_val = 0.050
                
                v1_raw = st.number_input(label, value=d_val, format="%.3f", key=f"v1u_{st.session_state.reset_key}")
                if is_inverse:
                    v1_plot = dist.ppf(1 - v1_raw) if bound_choice == "Lower" else dist.ppf(v1_raw)
                else: v1_plot = v1_raw

            elif prob_mode == "AND":
                if is_inverse and dist_name != "Chi-Square":
                    conf_c = st.number_input("Level of Confidence (c)", value=0.950, format="%.3f")
                    alpha = 1 - conf_c
                    v1_plot, v2_plot = dist.ppf(alpha/2), dist.ppf(1 - alpha/2)
                else:
                    # Prob Mode defaults
                    if dist_name != "Chi-Square": d_low, d_high = -1.0, 1.0
                    else: d_low, d_high = 2.700, 19.023
                    v1_plot = st.number_input("Lower Value (x)", value=d_low, format="%.3f")
                    v2_plot = st.number_input("Upper Value (x)", value=d_high, format="%.3f")

            elif prob_mode == "OR":
                if is_inverse:
                    l_area = st.number_input("Left-tail Area", value=0.025, format="%.3f")
                    r_area = st.number_input("Right-tail Area", value=0.025, format="%.3f")
                    v1_plot, v2_plot = dist.ppf(l_area), dist.ppf(1 - r_area)
                else:
                    st.write("**Left-tail Bound (x)**")
                    d_low = -1.0 if dist_name != "Chi-Square" else 2.700
                    v1_plot = st.number_input("LB", value=d_low, format="%.3f", label_visibility="collapsed")

        with b_col2:
            if prob_mode == "OR" and not is_inverse:
                st.write("**Right-tail Bound (x)**")
                d_high = 1.0 if dist_name != "Chi-Square" else 19.023
                v2_plot = st.number_input("RB", value=d_high, format="%.3f", label_visibility="collapsed")

    # --- 3. GRAPHICAL AREA ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x_min = mu - 4.5*sigma if dist_name != "Chi-Square" else 0.001
    x_max = mu + 4.5*sigma if dist_name != "Chi-Square" else dist.ppf(0.999)
    x_plot = np.linspace(x_min, x_max, 1000)
    y_plot = dist.pdf(x_plot)
    ax.plot(x_plot, y_plot, color='#1565C0', lw=2.5)

    if dist_name in ["Normal", "Student's t"]:
        ax.axvline(mu, color=COLOR_MEAN, lw=2, label=f'Mean: {mu:.2f}')
        for i in [1, 2, 3]:
            ax.axvline(mu + i*sigma, color=COLOR_SD, ls='--', lw=1, alpha=0.4)
            ax.axvline(mu - i*sigma, color=COLOR_SD, ls='--', lw=1, alpha=0.4)
        if show_comp:
            ax.plot(x_plot, stats.norm.pdf(x_plot, mu, sigma), color='gray', ls=':', label="Normal Ref")

    # Shading and Prob Result
    if prob_mode == "Unidirectional":
        # Bound Choice: Lower means x >= val; Upper means x <= val
        mask = (x_plot >= v1_plot) if bound_choice == "Lower" else (x_plot <= v1_plot)
        res_prob = 1 - dist.cdf(v1_plot) if bound_choice == "Lower" else dist.cdf(v1_plot)
    elif prob_mode == "AND":
        mask = (x_plot >= v1_plot) & (x_plot <= v2_plot)
        res_prob = dist.cdf(v2_plot) - dist.cdf(v1_plot)
    else: # OR
        mask = (x_plot <= v1_plot) | (x_plot >= v2_plot)
        res_prob = dist.cdf(v1_plot) + (1 - dist.cdf(v2_plot))

    ax.fill_between(x_plot, y_plot, where=mask, color=COLOR_SHADE, alpha=0.5)
    
    ticks = [v for v in [v1_plot, v2_plot] if v is not None]
    ax.set_xticks(ticks)
    labels = [f"x={v:.3f}\nz={(v-mu)/sigma:.2f}" for v in ticks] if dist_name == "Normal" else [f"{v:.3f}" for v in ticks]
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks([])
    ax.legend(prop={'size': 8}, loc='upper right')
    
    st.pyplot(fig)

    if is_inverse: st.success(f"**Critical Value(s):** {', '.join([f'{t:.3f}' for t in ticks])}")
    else: st.success(f"**Calculated Probability:** {res_prob:.4f}")

    # --- 4. EXPORTS ---
    st.divider()
    st.subheader("📁 Save Results")
    
    fn_col1, fn_col2 = st.columns(2)
    with fn_col1:
        file_name = st.text_input("Enter filename (no extension):", value="stat_results")
    
    e1, e2, e3 = st.columns(3)
    with e1:
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button(f"💾 Download Image", img_buf.getvalue(), f"{file_name}.png", "image/png")
    
    with e2:
        pdf_buf = io.BytesIO()
        with PdfPages(pdf_buf) as pdf:
            # We create a new high-quality figure for the PDF to ensure the chart is included
            pdf_fig, pdf_ax = plt.subplots(figsize=(8, 6))
            pdf_ax.plot(x_plot, y_plot, color='#1565C0')
            pdf_ax.fill_between(x_plot, y_plot, where=mask, color=COLOR_SHADE, alpha=0.5)
            pdf_ax.set_xticks(ticks)
            pdf_ax.set_xticklabels(labels, rotation=90)
            pdf_ax.set_title(f"Distribution: {dist_name} | Goal: {calc_mode}")
            
            # Add text summary to the PDF
            plt.figtext(0.1, 0.02, f"Result: {res_prob:.4f} | Params: mu={mu}, sigma={sigma}, n={n if 'n' in locals() else 'N/A'}", fontsize=10)
            pdf.savefig(pdf_fig, bbox_inches='tight')
            plt.close(pdf_fig)
        st.download_button(f"📄 Download PDF", pdf_buf.getvalue(), f"{file_name}.pdf", "application/pdf")
    
    with e3:
        st.button("🔄 Reset to Defaults", on_click=reset_app)

except Exception as e:
    st.error(f"Error: {e}")