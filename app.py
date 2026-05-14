# app.py

import streamlit as st
import pandas as pd

from eda_utils import (
    load_data,
    clean_data,
    get_overview_stats,
    get_numeric_columns,
    get_categorical_columns,
    plot_numeric_distribution,
    plot_categorical_distribution,
    summarize_dataframe,  # for later LLM use
)


st.set_page_config(page_title="Ecommerce EDA", layout="wide")

st.title("Ecommerce EDA Dashboard (MVP)")


@st.cache_data
def load_and_clean(path_or_file):
    df = load_data(path_or_file)
    df = clean_data(df)
    return df


# --- Data loading section ---

st.sidebar.header("Data")

uploaded_file = st.sidebar.file_uploader("Upload ecommerce CSV", type=["csv"])

if uploaded_file is not None:
    df = load_and_clean(uploaded_file)
    st.sidebar.success("Using uploaded file.")
else:
    # Fallback to local file
    default_path = "data/ecommerce.csv"
    df = load_and_clean(default_path)
    st.sidebar.info(f"Using default dataset: {default_path}")

if df is None or df.empty:
    st.error("No data available. Please upload a valid CSV.")
    st.stop()

# --- Tabs ---

tabs = st.tabs(["Overview", "Univariate Analysis", "Relationships", "Summary Text (for LLM)"])

# === Overview Tab ===
with tabs[0]:
    st.subheader("Dataset overview")

    overview = get_overview_stats(df)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Shape (rows, columns)**")
        st.write(overview["shape"])

        st.markdown("**Column dtypes**")
        st.write(overview["dtypes"])

    with col2:
        st.markdown("**Numeric summary (describe)**")
        st.dataframe(overview["describe_numeric"])

    st.markdown("**Sample rows**")
    st.dataframe(df.head())


# === Univariate Analysis Tab ===
with tabs[1]:
    st.subheader("Univariate analysis")

    all_cols = df.columns.tolist()
    selected_col = st.selectbox("Select a column", all_cols)

    if selected_col:
        if pd.api.types.is_numeric_dtype(df[selected_col]):
            st.markdown(f"**Numeric distribution for `{selected_col}`**")
            fig = plot_numeric_distribution(df, selected_col)
            st.pyplot(fig)
        else:
            st.markdown(f"**Categorical distribution for `{selected_col}`**")
            fig = plot_categorical_distribution(df, selected_col, top_n=20)
            st.pyplot(fig)


# === Relationships Tab ===
with tabs[2]:
    st.subheader("Relationships / Bivariate analysis")

    num_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)

    st.markdown("**Numeric vs numeric (scatter)**")
    if len(num_cols) >= 2:
        x_num = st.selectbox("X-axis (numeric)", num_cols, key="x_num")
        y_num = st.selectbox("Y-axis (numeric)", num_cols, key="y_num")
        if x_num and y_num and x_num != y_num:
            st.scatter_chart(df[[x_num, y_num]].dropna())
    else:
        st.info("Not enough numeric columns for scatter plot.")

    st.markdown("---")
    st.markdown("**Categorical vs numeric (grouped bar by mean)**")
    if cat_cols and num_cols:
        cat_col = st.selectbox("Categorical column", cat_cols, key="cat_col")
        num_col = st.selectbox("Numeric column", num_cols, key="num_col")
        if cat_col and num_col:
            grouped = df[[cat_col, num_col]].dropna().groupby(cat_col)[num_col].mean().reset_index()
            grouped = grouped.sort_values(num_col, ascending=False)
            st.bar_chart(grouped.set_index(cat_col))
    else:
        st.info("Need at least one categorical and one numeric column.")


# === Summary Text Tab (for LLM later) ===
with tabs[3]:
    st.subheader("Compact data summary (for LLM)")

    max_cols = st.slider("Max columns to include in summary", 3, min(10, len(df.columns)), 8)
    summary_text = summarize_dataframe(df, max_cols=max_cols)

    st.markdown("**Generated summary:**")
    st.text(summary_text)

    st.info("You will later pass this summary to an LLM to get EDA insights and reports.")
