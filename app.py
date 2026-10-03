import streamlit as st
import pandas as pd
import numpy as np
import portable

st.set_page_config(
    page_title="Study Office — Early Warning System",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Data & Model Loading
# Data loaded directly from GitHub raw URL
# -----------------------------------------------------------------------------
DATA_BASE_URL = "https://raw.githubusercontent.com/aaubs/tonights-front-desk/main/"

@st.cache_data
def load_datasets():
    # Load 2025 cohort (historical / validation) and 2026 cohort (new students)
    val_df = pd.read_csv(DATA_BASE_URL + "val_week6.csv")
    new_df = pd.read_csv(DATA_BASE_URL + "new_week6.csv")
    return val_df, new_df

@st.cache_resource
def load_model():
    # Load the XGBoost pipeline stored in portable format (model/ directory)
    return portable.Model("model")

try:
    val_df, new_df = load_datasets()
    model = load_model()
except Exception as e:
    st.error(f"Error loading model or datasets: {e}")
    st.stop()

# Compute predictions once
val_df["risk"] = model.predict_proba(val_df)
new_df["risk"] = model.predict_proba(new_df)

# Target column identification
target_col = "dropped_out" if "dropped_out" in val_df.columns else "target"
intl_col = "international" if "international" in val_df.columns else "is_international"

# -----------------------------------------------------------------------------
# App Layout & Header
# -----------------------------------------------------------------------------
st.title("🎓 Student Retention & Early Warning System")
st.markdown(
    "Welcome to the Study Office Decision Support System. Use this tool to prioritize outreach "
    "for students at risk of dropping out and evaluate intervention policies."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 This Week's List (2026)",
    "📊 Policy Evaluator (2025 Cohort)",
    "🌐 Breakdown by Group",
    "💡 Decision Support & Risk Drivers"
])

# -----------------------------------------------------------------------------
# TAB 1: This Week's List
# -----------------------------------------------------------------------------
with tab1:
    st.header("Priority Outreach List — 2026 Cohort (Week 6)")
    st.write("Students are ranked by predicted dropout risk. The **top 40** prioritized for outreach are marked below.")
    
    # Sort students by risk score descending
    sorted_2026 = new_df.sort_values(by="risk", ascending=False).reset_index(drop=True)
    
    # Mark top 40
    sorted_2026["Top 40 Target"] = False
    sorted_2026.loc[:39, "Top 40 Target"] = True
    
    # Format display
    display_cols = [c for c in ["student_id", "id", "name", "risk", "Top 40 Target", intl_col] if c in sorted_2026.columns]
    if len(display_cols) < 2:
        display_cols = ["risk", "Top 40 Target"] + [c for c in sorted_2026.columns if c not in ["risk", "Top 40 Target"]]

    st.dataframe(
        sorted_2026[display_cols].style.format({"risk": "{:.1%}"}),
        use_container_width=True,
        height=500
    )

# -----------------------------------------------------------------------------
# TAB 2: Mistakes of a Rule (2025 Cohort Evaluation)
# -----------------------------------------------------------------------------
with tab2:
    st.header("Evaluating Intervention Rules on Historical Data (2025 Cohort)")
    st.caption("Adjust the slider to simulate setting a capacity limit on how many conversations the office can hold.")

    n_conversations = st.slider(
        "Number of conversations / outreach capacity:",
        min_value=5,
        max_value=min(200, len(val_df)),
        value=40,
        step=5
    )

    # Determine cut-off threshold from capacity
    threshold = val_df["risk"].nlargest(n_conversations).iloc[-1]
    val_df["pred_reach"] = val_df["risk"] >= threshold

    # Confusion matrix elements
    tp = int(((val_df["pred_reach"] == True) & (val_df[target_col] == 1)).sum())
    fp = int(((val_df["pred_reach"] == True) & (val_df[target_col] == 0)).sum())
    fn = int(((val_df["pred_reach"] == False) & (val_df[target_col] == 1)).sum())
    tn = int(((val_df["pred_reach"] == False) & (val_df[target_col] == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    st.subheader("Intervention Summary in Plain Words")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reached in time", f"{tp} students", help="True Positives: Dropouts correctly identified and contacted.")
    col2.metric("Worried for nothing", f"{fp} students", help="False Positives: Contacted but would not have dropped out.")
    col3.metric("Missed", f"{fn} students", help="False Negatives: Dropped out without receiving outreach.")
    col4.metric("Correctly left alone", f"{tn} students", help="True Negatives: On track and correctly not contacted.")

    st.markdown("---")
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Precision (Efficiency)", f"{precision:.1%}", help="Share of contacted students who actually needed help.")
    m_col2.metric("Recall (Coverage)", f"{recall:.1%}", help="Share of all actual dropouts successfully contacted.")

# -----------------------------------------------------------------------------
# TAB 3: Per Group Breakdown
# -----------------------------------------------------------------------------
with tab3:
    st.header("Performance Breakdown by Student Category")

    if intl_col in val_df.columns:
        groups = [
            ("International Students", val_df[val_df[intl_col] == 1]),
            ("Domestic Students", val_df[val_df[intl_col] == 0])
        ]

        for group_label, group_data in groups:
            gtp = int(((group_data["pred_reach"] == True) & (group_data[target_col] == 1)).sum())
            gfp = int(((group_data["pred_reach"] == True) & (group_data[target_col] == 0)).sum())
            gfn = int(((group_data["pred_reach"] == False) & (group_data[target_col] == 1)).sum())
            gtn = int(((group_data["pred_reach"] == False) & (group_data[target_col] == 0)).sum())

            g_prec = gtp / (gtp + gfp) if (gtp + gfp) > 0 else 0.0
            g_rec = gtp / (gtp + gfn) if (gtp + gfn) > 0 else 0.0

            st.subheader(f"📌 {group_label} ({len(group_data)} total)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Reached in time", f"{gtp}")
            c2.metric("Worried for nothing", f"{gfp}")
            c3.metric("Missed", f"{gfn}")
            c4.metric("Precision / Recall", f"{g_prec:.1%} / {g_rec:.1%}")
            st.divider()
    else:
        st.info("Group column not found in dataset.")

# -----------------------------------------------------------------------------
# TAB 4: Decision Support & Cost/Risk Drivers
# -----------------------------------------------------------------------------
with tab4:
    st.header("💡 Decision Support: Financial & Operational Impact")
    st.write("Customize cost parameters to calculate the net financial impact of intervention policy.")

    c_left, c_right = st.columns(2)
    with c_left:
        cost_outreach = st.number_input("Cost per intervention/conversation (€)", value=50, step=10)
    with c_right:
        cost_dropout = st.number_input("Cost / lost tuition per unassisted dropout (€)", value=2500, step=100)

    # Net cost calculation
    total_outreach_cost = (tp + fp) * cost_outreach
    total_missed_cost = fn * cost_dropout
    total_net_cost = total_outreach_cost + total_missed_cost

    st.subheader(f"Estimated Total Policy Cost: €{total_net_cost:,.2f}")
    st.write(f"- **Outreach Budget Spent:** €{total_outreach_cost:,.2f}")
    st.write(f"- **Cost of Unassisted Dropouts:** €{total_missed_cost:,.2f}")
