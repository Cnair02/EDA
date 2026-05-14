# eda_utils.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path_or_file):
    """
    Load CSV data from a file path or a file-like object (e.g., Streamlit upload).
    """
    return pd.read_csv(path_or_file)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning for a generic ecommerce dataset.

    - Parse common date columns
    - Coerce numeric columns
    - Drop obviously invalid rows based on ID fields
    """
    df = df.copy()

    # Parse likely date columns if present
    for col in df.columns:
        if any(key in col.lower() for key in ["date", "order_date", "order_datetime"]):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass

    # Coerce obvious numeric columns
    numeric_keywords = ["price", "amount", "revenue", "sales", "qty", "quantity", "discount", "profit"]
    for col in df.columns:
        if any(key in col.lower() for key in numeric_keywords):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing core identifiers if present
    core_id_cols = [
        c for c in df.columns
        if any(k in c.lower() for k in ["order_id", "order id", "customer_id", "customer id"])
    ]
    for cid in core_id_cols:
        df = df[df[cid].notna()]

    return df


def get_overview_stats(df: pd.DataFrame) -> dict:
    """
    Return basic overview information for the dataframe.
    Useful if you want to show in Streamlit without recomputing.
    """
    overview = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "describe_numeric": df.describe(include="number").T,
    }
    return overview


def get_numeric_columns(df: pd.DataFrame):
    """
    Return list of numeric columns.
    """
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(df: pd.DataFrame):
    """
    Return list of non-numeric (categorical/text) columns.
    """
    return df.select_dtypes(exclude="number").columns.tolist()


def plot_numeric_distribution(df: pd.DataFrame, col: str):
    """
    Create a histogram and boxplot for a numeric column.
    Returns a matplotlib Figure (for Streamlit's st.pyplot).
    """
    data = df[col].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.histplot(data, kde=True, ax=axes[0])
    axes[0].set_title(f"Histogram of {col}")

    sns.boxplot(x=data, ax=axes[1])
    axes[1].set_title(f"Boxplot of {col}")

    plt.tight_layout()
    return fig


def plot_categorical_distribution(df: pd.DataFrame, col: str, top_n: int = 20):
    """
    Create a bar chart of the top N categories for a categorical column.
    Returns a matplotlib Figure.
    """
    vc = df[col].astype(str).value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=vc.index, y=vc.values, ax=ax)
    ax.set_title(f"Top {top_n} categories for {col}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return fig


def summarize_dataframe(df: pd.DataFrame, max_cols: int = 10) -> str:
    """
    Create a compact textual summary of up to max_cols columns.
    This will be used later as input to an LLM.
    """
    lines = []
    cols = df.columns[:max_cols]
    n_rows = len(df)

    for col in cols:
        series = df[col]
        non_null = series.notna().sum()
        missing_pct = 100 * (1 - non_null / n_rows)
        dtype = str(series.dtype)

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            line = (
                f"Column: {col} (numeric, dtype={dtype}) | "
                f"non_null={non_null}, missing_pct={missing_pct:.1f}% | "
                f"min={desc.get('min', None)}, max={desc.get('max', None)}, "
                f"mean={desc.get('mean', None)}"
            )
        else:
            vc = series.astype(str).value_counts(normalize=True).head(3)
            top_vals = [f"{idx}({pct*100:.1f}%)" for idx, pct in vc.items()]
            line = (
                f"Column: {col} (non-numeric, dtype={dtype}) | "
                f"non_null={non_null}, missing_pct={missing_pct:.1f}% | "
                f"top_values=[{', '.join(top_vals)}]"
            )

        lines.append(line)

    summary = f"Rows: {n_rows}\n" + "\n".join(lines)
    return summary
