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

# --- SESSION STATE & RESET LOGIC ---
if 'reset_key' not in st.session_state:
    st.session_state.reset_key = 0

def reset_app():
    st.session_state.reset_key += 1

st.title("📊 Pro Probability Applet")

# --- 1. DISTRIBUTION SETTINGS ---
st.header("1. Distribution Settings")
col_dist, col_goal, col_type = st.columns(3)

with col_dist:
    dist_name = st.selectbox("Distribution", ["Normal", "Student's t", "Chi-Square"], 
                             index=0, key=f"dist_{st.session_state.reset_key}")
with col_goal:
    calc_mode = st.radio("Goal", ["Find Probability", "Find Critical Value (Inverse)"], 
                         index=0, key=f"goal_{st.session_state.reset_key}")
with col_type:
    prob_mode = st.selectbox("Interval Type", ["Unidirectional", "AND", "OR"], 
                             index=0, key=f"type_{st.session_state.reset_key}")

is_inverse = "Inverse" in calc_mode

# --- DYNAMIC DEFAULTS ENGINE ---
p_col1, p_col2, p_col3 = st.columns(3)

# Default Sample Sizes
default_n = 25 if dist_name == "Student's t" else 10

with p_col1:
    if dist_name in ["Normal", "Student's t"]:
        mu = st.number_input("Mean (μ)", value=0.000, format="%.3f", key=f"mu_{dist_name}_{st.session_state.reset_key}")
    else: mu = 0.0
with p_col2:
    if dist_name in ["Normal", "Student's t"]:
        sigma = st.number_input("Std Dev (σ)", value=1.000, min_value=0.01, format="%.3f", key=f"sig_{dist_name}_{st.session_state.reset_key}")
    else: sigma = 1.0
with p_col3:
    if dist_name in ["Student's t", "Chi-Square"]:
        n = st.number_input("Sample Size (n)", value=default_n, min_value=2, key=f"n_{dist_name}_{st.session_state.reset_key}")
        df = n - 1

show_comp = False
if dist_name == "Student's t":
    show_comp = st.checkbox("Show Normal Comparison (Dashed Line)", value=False, key=f"comp_{st.session_state.reset_key}")

st.divider()

# --- 2. ANALYSIS PARAMETERS ---
st.header("2. Analysis Parameters")
b_col1, b_col2 = st.columns(2)

# Specific Boundary Defaults based on Prompt requirements
if not is_inverse:
    if dist_name != "Chi-Square":
        v_low_def, v_high_def = -1.000, 1.000
        v_uni_def = 1.000
    else:
        v_low_def, v_high_def = 2.700, 19.023
        v_uni_def = 16.919
else:
    v_uni_def = 0.050
    v_low_def, v_high_def = 0.025, 0.025 

try:
    if dist_name == "Normal": dist = stats.norm(mu, sigma)
    elif dist_name == "Student's t": dist = stats.t(df, mu, sigma)
    else: dist = stats.chi2(df)

    v1_plot, v2_plot = None, None
    # STARTUP REQUIREMENT: Default Bound Type to Lower
    bound_choice = "Lower" 

    if is_inverse and dist_name == "Chi-Square":
        with b_col1:
            # UNIDIRECTIONAL: Phrase appears ABOVE input
            if prob_mode == "Unidirectional":
                st.markdown(r"$\alpha$ represents the area to the right of the critical value.")
                alpha_val = st.number_input(r"Probability, $\alpha$", value=0.050, format="%.3f", key=f"alpha_uni_{st.session_state.reset_key}")
                v1_plot = dist.ppf(1 - alpha_val)
                bound_choice = "Lower" 
            
            # AND/OR: Alpha input first, then explanatory phrase BELOW
            elif prob_mode == "AND":
                alpha_val = st.number_input(r"Probability, $\alpha$", value=0.050, format="%.3f", key=f"alpha_and_{st.session_state.reset_key}")
                st.info(r"The area remaining in each tail is $1/2 \alpha$.")
                v1_plot, v2_plot = dist.ppf(alpha_val/2), dist.ppf(1 - alpha_val/2)
                
            elif prob_mode == "OR":
                alpha_val = st.number_input(r"Probability, $\alpha$", value=0.050, format="%.3f", key=f"alpha_or_{st.session_state.reset_key}")
                st.info(r"The shaded area in each tail is $1/2 \alpha$.")
                v1_plot, v2_plot = dist.ppf(alpha_val/2), dist.ppf(1 - alpha_val/2)
    else:
        with b_col1:
            if prob_mode == "Unidirectional":
                bound_choice = st.selectbox("Bound Type", ["Lower", "Upper"], index=0,
                                            key=f"bt_{dist_name}_{calc_mode}_{st.session_state.reset_key}")
                label = "Probability (0 to 1)" if is_inverse else "Value (x)"
                v1_raw = st.number_input(label, value=v_uni_def, format="%.3f", key=f"v1u_{dist_name}_{calc_mode}_{st.session_state.reset_key}")
                if is_inverse:
                    v1_plot = dist.ppf(1 - v1_raw) if bound_choice == "Lower" else dist.ppf(v1_raw)
                else: v1_plot = v1_raw
            
            elif prob_mode == "AND":
                if is_inverse:
                    conf_c = st.number_input("Level of Confidence (c)", value=0.950, format="%.3f", key=f"conf_{dist_name}_{st.session_state.reset_key}")
                    alpha = 1 - conf_c
                    v1_plot, v2_plot = dist.ppf(alpha/2), dist.ppf(1 - alpha/2)
                else:
                    v1_plot = st.number_input("Lower Value (x)", value=v_low_def, format="%.3f", key=f"vlow_{dist_name}_{st.session_state.reset_key}")
                    v2_plot = st.number_input("Upper Value (x)", value=v_high_def, format="%.3f", key=f"vhigh_{dist_name}_{st.session_state.reset_key}")

            elif prob_mode == "OR":
                if is_inverse:
                    v1_plot_area = st.number_input("Left-tail Area", value=0.025, format="%.3f", key=f"larea_{dist_name}_{st.session_state.reset_key}")
                    v2_plot_area = st.number_input("Right-tail Area", value=0.025, format="%.3f", key=f"rarea_{dist_name}_{st.session_state.reset_key}")
                    v1_plot, v2_plot = dist.ppf(v1_plot_area), dist.ppf(1 - v2_plot_area)
                else:
                    st.write("**Left-tail Bound (x)**")
                    v1_plot = st.number_input("LB", value=v_low_def, format="%.3f", key=f"v1or_{dist_name}_{st.session_state.reset_key}", label_visibility="collapsed")

        with b_col2:
            if prob_mode == "OR" and not is_inverse:
                st.write("**Right-tail Bound (x)**")
                v2_plot = st.number_input("RB", value=v_high_def, format="%.3f", key=f"v2or_{dist_name}_{st.session_state.reset_key}", label_visibility="collapsed")

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

    if prob_mode == "Unidirectional":
        mask = (x_plot >= v1_plot) if bound_choice == "Lower" else (x_plot <= v1_plot)
        res_val = 1 - dist.cdf(v1_plot) if bound_choice == "Lower" else dist.cdf(v1_plot)
    elif prob_mode == "AND":
        mask = (x_plot >= v1_plot) & (x_plot <= v2_plot)
        res_val = dist.cdf(v2_plot) - dist.cdf(v1_plot)
    else: # OR
        mask = (x_plot <= v1_plot) | (x_plot >= v2_plot)
        res_val = dist.cdf(v1_plot) + (1 - dist.cdf(v2_plot))

    ax.fill_between(x_plot, y_plot, where=mask, color=COLOR_SHADE, alpha=0.5)
    
    ticks = [v for v in [v1_plot, v2_plot] if v is not None]
    ax.set_xticks(ticks)
    labels = [f"x={v:.3f}\nz={(v-mu)/sigma:.2f}" for v in ticks] if dist_name == "Normal" else [f"{v:.3f}" for v in ticks]
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks([])
    ax.legend(prop={'size': 8}, loc='upper right')
    
    st.pyplot(fig)

    if is_inverse: st.success(f"**Critical Value(s):** {', '.join([f'{t:.3f}' for t in ticks])}")
    else: st.success(f"**Calculated Probability:** {res_val:.4f}")

    # --- 4. EXPORTS ---
    st.divider()
    st.subheader("📁 Save Results")
    file_name = st.text_input("Enter filename (no extension):", value="stat_results")
    
    e1, e2, e3 = st.columns(3)
    with e1:
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("💾 Download Image", img_buf.getvalue(), f"{file_name}.png", "image/png")
    
    with e2:
        pdf_buf = io.BytesIO()
        with PdfPages(pdf_buf) as pdf:
            pdf_fig = plt.figure(figsize=(8.5, 11))
            plt.figtext(0.1, 0.9, f"Stat Report - {dist_name}", fontsize=14, fontweight='bold')
            plt.figtext(0.1, 0.85, f"Goal: {calc_mode} | Interval: {prob_mode}")
            plt.figtext(0.1, 0.82, f"Result: {res_val:.4f}")
            
            new_ax = pdf_fig.add_axes([0.1, 0.3, 0.8, 0.4])
            new_ax.plot(x_plot, y_plot, color='#1565C0')
            new_ax.fill_between(x_plot, y_plot, where=mask, color=COLOR_SHADE, alpha=0.5)
            new_ax.set_xticks(ticks)
            new_ax.set_xticklabels(labels, rotation=90)
            pdf.savefig(pdf_fig)
            plt.close(pdf_fig)
        st.download_button("📄 Download PDF", pdf_buf.getvalue(), f"{file_name}.pdf", "application/pdf")
    
    with e3:
        st.button("🔄 Reset to Defaults", on_click=reset_app)

except Exception as e:
    st.error(f"Error: {e}")