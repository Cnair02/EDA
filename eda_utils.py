# eda_utils.py

import os
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types


## Setup secret keys

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("✅ Gemini API key setup complete.")
except Exception as e:
    print(f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


# -----------------------------
# Core EDA utilities
# -----------------------------

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
    if df is None or df.empty:
        return df

    df = df.copy()

    # Parse likely date columns if present
    for col in df.columns:
        if any(key in col.lower() for key in ["date", "order_date", "order_datetime"]):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                # Soft-fail on date parsing
                pass

    # Coerce obvious numeric columns
    numeric_keywords = ["price", "amount", "revenue", "sales", "qty", "quantity", "discount", "profit"]
    for col in df.columns:
        if any(key in col.lower() for key in numeric_keywords):
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass

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
    """
    if df is None or df.empty:
        return {"shape": (0, 0), "dtypes": {}, "describe_numeric": pd.DataFrame()}

    overview = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "describe_numeric": df.describe(include="number").T if not df.select_dtypes("number").empty else pd.DataFrame(),
    }
    return overview


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Return list of numeric columns.
    """
    if df is None or df.empty:
        return []
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Return list of non-numeric (categorical/text) columns.
    """
    if df is None or df.empty:
        return []
    return df.select_dtypes(exclude="number").columns.tolist()


def plot_numeric_distribution(df: pd.DataFrame, col: str):
    """
    Create a histogram and boxplot for a numeric column.
    Returns a matplotlib Figure.
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
    This will be used as input/context for the LLM.

    - Handles missing values
    - Distinguishes numeric vs non-numeric columns
    - Limits categorical detail to top 3 categories
    """
    if df is None or df.empty:
        return "Empty dataframe: no rows to summarize."

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
            col_min = desc.get("min", None)
            col_max = desc.get("max", None)
            col_mean = desc.get("mean", None)

            line = (
                f"Column: {col} (numeric, dtype={dtype}) | "
                f"non_null={non_null}, missing_pct={missing_pct:.1f}% | "
                f"min={col_min}, max={col_max}, mean={col_mean}"
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


# -----------------------------
# LLM (Google ADK + Gemini) for EDA insights
# -----------------------------

# Simple, concise prompt with exactly 4 items per section
EDA_INSIGHTS_INSTRUCTION = """
You are a senior data analyst for an ecommerce company.
You see only a compact summary of a dataframe (no raw rows).

Given the dataframe summary, do ALL of the following:

1. Data understanding
   - Briefly describe what this dataset likely represents and name 2–3 key metrics.

2. Key patterns (exactly 4 bullet points)
   - List 4 notable patterns you can infer from ranges, means, and category frequencies.

3. Data quality issues (exactly 4 bullet points)
   - List 4 potential issues (missingness, outliers, imbalance, strange values).

4. Recommended EDA steps (exactly 4 bullet points)
   - Suggest 4 concrete plots or analyses to run next.

5. Business questions (exactly 4 bullet points)
   - Propose 4 business questions this dataset could help answer.

Respond in clear markdown with headings and bullet lists only.
""".strip()

# Global objects
_APP_NAME = "ecommerce_eda_app"
_USER_ID = "eda_user"
_SESSION_ID = "eda_session"

# Pre-create a simple ADK agent for EDA insights
_session_service = InMemorySessionService()
_eda_agent = Agent(
    model="gemini-2.5-flash-lite",
    name="eda_insights_agent",
    instruction=EDA_INSIGHTS_INSTRUCTION,
    description="Generates EDA insights from a compact dataframe summary.",
)
_eda_runner = Runner(
    agent=_eda_agent,
    app_name="ecommerce_eda_app",
    session_service=_session_service,
)


async def _call_eda_agent_async(summary_text: str) -> str:
    """
    Internal async helper to:
    - Ensure a session exists (create if needed, with await)
    - Call the ADK agent with the summary text
    """
    # 1) Ensure session exists (the key fix: await create_session) [web:84][web:86]
    session = await _session_service.get_session(
        app_name=_APP_NAME,
        user_id=_USER_ID,
        session_id=_SESSION_ID,
    )
    if session is None:
        session = await _session_service.create_session(
            app_name=_APP_NAME,
            user_id=_USER_ID,
            session_id=_SESSION_ID,
        )

    # 2) Prepare user content
    user_content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=summary_text)],
    )

    last_text = ""

    # 3) Run the agent
    async for event in _eda_runner.run_async(
        user_id=_USER_ID,
        session_id=_SESSION_ID,
        new_message=user_content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            # Take the first text part as the final markdown
            last_text = event.content.parts[0].text or ""

    return last_text or "No insights generated by the EDA agent."

def get_summary_table(df: pd.DataFrame, max_cols: int = 10) -> pd.DataFrame:
    """
    Build a compact tabular summary of up to max_cols columns:
    - Column name
    - Dtype
    - Non-null count
    - Missing %
    - Min / Max / Mean (for numeric)
    - Top categories (for non-numeric)
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["column", "dtype", "non_null", "missing_pct", "min", "max", "mean", "top_values"]
        )

    rows = []
    cols = df.columns[:max_cols]
    n_rows = len(df)

    for col in cols:
        series = df[col]
        non_null = series.notna().sum()
        missing_pct = 100 * (1 - non_null / n_rows)
        dtype = str(series.dtype)

        row = {
            "column": col,
            "dtype": dtype,
            "non_null": non_null,
            "missing_pct": round(missing_pct, 1),
            "min": None,
            "max": None,
            "mean": None,
            "top_values": None,
        }

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            row["min"] = desc.get("min", None)
            row["max"] = desc.get("max", None)
            row["mean"] = desc.get("mean", None)
        else:
            vc = series.astype(str).value_counts(normalize=True).head(3)
            top_vals = [f"{idx} ({pct*100:.1f}%)" for idx, pct in vc.items()]
            row["top_values"] = ", ".join(top_vals)

        rows.append(row)

    return pd.DataFrame(rows)
    

def generate_eda_insights(summary_text: str) -> str:
    """
    Synchronous wrapper for Streamlit:
    - Validates env / summary
    - Calls the ADK agent via asyncio
    - Handles errors and returns a safe string
    """
    if not summary_text or summary_text.strip() == "":
        return "No summary text provided. Generate a summary before requesting insights."

    if not os.getenv("GOOGLE_API_KEY"):
        return (
            "GOOGLE_API_KEY is not set. Please configure it in your environment "
            "before generating EDA insights."
        )

    try:
        import asyncio

        return asyncio.run(_call_eda_agent_async(summary_text))
    except Exception as e:
        return f"Error while generating EDA insights with Gemini/ADK: {e}"
