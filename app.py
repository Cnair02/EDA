# app.py

import pandas as pd
import streamlit as st

from eda_utils import (
    load_data,
    clean_data,
    get_overview_stats,
    get_numeric_columns,
    get_categorical_columns,
    plot_numeric_distribution,
    plot_categorical_distribution,
    summarize_dataframe,
    generate_eda_insights,
    get_summary_table, 
)


st.set_page_config(page_title="Ecommerce EDA + AI Insights", layout="wide")
st.title("EDA Dashboard with AI Insights (Gemini)")


# -----------------------------
# Cached loader with cleaning
# -----------------------------
@st.cache_data
def load_and_clean(path_or_file):
    df = load_data(path_or_file)
    df = clean_data(df)
    return df


# -----------------------------
# Data loading with error handling
# -----------------------------
st.sidebar.header("Data")

uploaded_file = st.sidebar.file_uploader("Upload ecommerce CSV", type=["csv"])

df = None
error_msg = None

try:
    if uploaded_file is not None:
        df = load_and_clean(uploaded_file)
        st.sidebar.success("Using uploaded file.")
    else:
        default_path = "data/ecommerce.csv"
        df = load_and_clean(default_path)
        st.sidebar.info(f"Using default dataset: {default_path}")
except FileNotFoundError:
    error_msg = "Default dataset not found. Please upload a CSV file."
except pd.errors.EmptyDataError:
    error_msg = "The uploaded file is empty. Please provide a valid CSV."
except pd.errors.ParserError:
    error_msg = "Could not parse the CSV file. Please check the file format."
except Exception as e:
    error_msg = f"Unexpected error while loading data: {e}"

if error_msg:
    st.error(error_msg)
    st.stop()

if df is None or df.empty:
    st.error("No usable data available after loading/cleaning. Please check your CSV.")
    st.stop()


# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs([
    "Overview",
    "Univariate Analysis",
    "Relationships",
    # "Summary Text (for LLM)",
    "AI EDA Insights",
])

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
        if not overview["describe_numeric"].empty:
            st.dataframe(overview["describe_numeric"])
        else:
            st.info("No numeric columns to summarize.")

    st.markdown("**Sample rows**")
    st.dataframe(df.head())


# === Univariate Analysis Tab ===
with tabs[1]:
    st.subheader("Univariate analysis")

    all_cols = df.columns.tolist()
    if not all_cols:
        st.warning("No columns available for univariate analysis.")
    else:
        selected_col = st.selectbox("Select a column", all_cols)

        if selected_col:
            try:
                if pd.api.types.is_numeric_dtype(df[selected_col]):
                    st.markdown(f"**Numeric distribution for `{selected_col}`**")
                    fig = plot_numeric_distribution(df, selected_col)
                    st.pyplot(fig)
                else:
                    st.markdown(f"**Categorical distribution for `{selected_col}`**")
                    top_n = st.slider("Number of top categories to show", 5, 30, 20)
                    fig = plot_categorical_distribution(df, selected_col, top_n=top_n)
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating univariate plot: {e}")


# === Relationships Tab ===
with tabs[2]:
    st.subheader("Relationships / Bivariate analysis")

    num_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)

    # Numeric vs numeric
    st.markdown("**Numeric vs numeric (scatter plot)**")
    if len(num_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            x_num = st.selectbox("X-axis (numeric)", num_cols, key="x_num_rel")
        with col2:
            y_num = st.selectbox("Y-axis (numeric)", num_cols, key="y_num_rel")

        if x_num and y_num and x_num != y_num:
            try:
                scatter_df = df[[x_num, y_num]].dropna()
                if not scatter_df.empty:
                    st.scatter_chart(scatter_df)
                else:
                    st.info("No data available for the selected numeric columns.")
            except Exception as e:
                st.error(f"Error generating scatter plot: {e}")
    else:
        st.info("Not enough numeric columns to create a scatter plot.")

    st.markdown("---")

    # Categorical vs numeric
    st.markdown("**Categorical vs numeric (mean by category)**")
    if cat_cols and num_cols:
        col3, col4 = st.columns(2)
        with col3:
            cat_col = st.selectbox("Categorical column", cat_cols, key="cat_col_rel")
        with col4:
            num_col = st.selectbox("Numeric column", num_cols, key="num_col_rel")

        if cat_col and num_col:
            try:
                grouped = (
                    df[[cat_col, num_col]]
                    .dropna()
                    .groupby(cat_col)[num_col]
                    .mean()
                    .reset_index()
                )
                grouped = grouped.sort_values(num_col, ascending=False)
                if not grouped.empty:
                    st.bar_chart(grouped.set_index(cat_col))
                else:
                    st.info("No data available after grouping; try different columns.")
            except Exception as e:
                st.error(f"Error generating grouped bar chart: {e}")
    else:
        st.info("Need at least one categorical and one numeric column to plot group means.")


# # === Summary Text Tab ===
# with tabs[3]:
#     st.subheader("Compact data summary (for LLM)")

#     try:
#         max_cols_default = min(8, len(df.columns))
#         max_cols = st.slider(
#             "Max columns to include in summary",
#             min_value=3,
#             max_value=min(15, len(df.columns)),
#             value=max_cols_default,
#         )

#         if len(df.columns) == 0:
#             st.warning("No columns available in the dataframe to summarize.")
#         else:
#             summary_text = summarize_dataframe(df, max_cols=max_cols)

#             st.markdown("**Generated summary:**")
#             st.text(summary_text)

#             st.caption(
#                 "This summary is designed to be passed to a Gemini (Google ADK) agent "
#                 "to generate EDA insights."
#             )
#     except Exception as e:
#         st.error(f"Error while generating summary text: {e}")


# === AI EDA Insights Tab ===
with tabs[3]:
    st.subheader("AI EDA Insights (Gemini)")

    if len(df.columns) == 0:
        st.warning("No columns available; cannot generate AI insights.")
    else:
        max_cols_default = min(8, len(df.columns))
        max_cols = st.slider(
            "Max columns to include in AI summary",
            min_value=3,
            max_value=min(15, len(df.columns)),
            value=max_cols_default,
            key="max_cols_ai",
        )
        summary_table_ai = get_summary_table(df, max_cols=max_cols)
        st.markdown("**Summary table used to build AI context:**")
        st.table(summary_table_ai)

        summary_text = summarize_dataframe(df, max_cols=max_cols)

        if st.button("Generate AI EDA insights"):
            with st.spinner("Calling Gemini for EDA insights..."):
                insights_md = generate_eda_insights(summary_text)
            st.markdown("**LLM-generated EDA insights:**")
            st.markdown(insights_md)
