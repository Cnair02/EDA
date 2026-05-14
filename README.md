# EDA Dashboard with AI Insights (Python, Pandas, Streamlit, Gemini/ADK)

An interactive exploratory data analysis (EDA) app for datasets, built with Python, pandas, and Streamlit, and enhanced with an AI assistant using Gemini via Google Agent Development Kit (ADK).  
Upload a CSV, explore the data visually, and generate structured EDA insights and business questions from a Gemini “senior data analyst” agent.

---

## 🔗 Live demo & code

- Live app: **https://eda-ai-analyzer.streamlit.app/**


## 1. Project overview

This project simulates what a data / business analyst does with an ecommerce dataset:

- **Classic EDA:** profile data, inspect distributions, and explore relationships using Python, pandas, seaborn, and Streamlit.  
- **AI assistant:** send a compact summary of the dataframe to a Gemini agent (via Google ADK) to generate:
  - 4 key patterns  
  - 4 potential data quality issues  
  - 4 recommended analyses/plots  
  - 4 business questions the dataset can answer

The goal is to demonstrate end‑to‑end analytical thinking with modern LLM tools, not just static dashboards.

---

## 2. Features

- **CSV upload or default dataset**
  - Upload any CSV.
  - Fallback to a bundled sample dataset (`data/ecommerce.csv`) for quick testing. 

- **Overview**
  - Shape, column data types.
  - Numeric summary (`describe()`).
  - Sample rows for a quick glance at the data. 

- **Univariate analysis**
  - Numeric columns: histograms + boxplots.
  - Categorical columns: top‑N category frequency bar charts.

- **Bivariate analysis**
  - Numeric vs numeric: scatter plots.
  - Categorical vs numeric: mean metric by category (bar chart).

- **Compact summary for LLM**
  - Generates a concise text summary with:
    - Number of rows.
    - Up to N columns: type, non‑null count, missing %, min/max/mean (numeric), top 3 categories (categorical). 

- **AI EDA Insights (Gemini / Google ADK)**
  - Gemini agent (via Google ADK) reads the summary and returns:
    - 4 notable patterns.
    - 4 potential data quality issues.
    - 4 recommended EDA steps.
    - 4 business questions in stakeholder‑friendly language.
  - Uses an ADK `Agent` + `Runner` with proper async session creation and error handling.

---

## 3. Tech stack

**Core:**

- Python  
- pandas  
- Streamlit  
- matplotlib, seaborn  

**AI / LLM:**

- Gemini (via `google-genai`)  
- Google Agent Development Kit (`google-adk-python`) 

**Other:**

- In‑memory session management with `InMemorySessionService`  
- Async agent execution wrapped in a sync helper for Streamlit

---
## 4. Usage guide

1. **Overview tab**
   - Inspect shape, data types, numeric summary, and sample rows.
   - Quickly confirm that dates and numeric fields parsed correctly.

2. **Univariate analysis**
   - Select a column and:
     - For numeric: review histogram and boxplot to check distribution, skew, and outliers.
     - For categorical: inspect the top categories and their frequencies. 

3. **Relationships**
   - Numeric vs numeric: look at scatter plots (e.g., quantity vs revenue).
   - Categorical vs numeric: compare mean revenue or quantity by region, product category, etc. 

4. **Summary Text (for LLM)**
   - Adjust `max_cols` and view the compact summary text.
   - This is exactly what gets passed to the Gemini agent as context.

5. **AI EDA Insights**
   - Choose `max_cols` and click **“Generate AI EDA insights”**.
   - The Gemini/ADK agent returns:
     - 4 patterns (distributions, dominant categories, ranges).
     - 4 potential data quality issues.
     - 4 recommended next EDA steps.
     - 4 business questions phrased for stakeholders. 

---

## 5. Example use cases

- **Data / Business Analyst:** Use the app to quickly profile new ecommerce datasets and get AI‑generated ideas for further analysis and stakeholder questions.  
- **Portfolio piece:** Demonstrate how you combine Python/pandas EDA with LLM agents for analysis acceleration and storytelling.  

---

## 6. Screenshots

_Add 2–3 screenshots here, for example:_

- Overview tab (shape, schema, summary).
  <img width="1235" height="726" alt="Screenshot 2026-05-14 at 2 01 19 PM" src="https://github.com/user-attachments/assets/4ceb9e4e-2f5a-4a27-8d5c-a6bf3d843551" />

- Univariate analysis on `price` or `revenue`.
  <img width="1128" height="778" alt="Screenshot 2026-05-14 at 2 02 06 PM" src="https://github.com/user-attachments/assets/283c9cf6-3b57-4687-b272-eef7ba0ffe30" />

  <img width="1128" height="778" alt="Screenshot 2026-05-14 at 2 02 52 PM" src="https://github.com/user-attachments/assets/8765854a-2981-4e4f-b66c-778d08702215" />


- AI EDA Insights tab showing Gemini’s markdown output.
 <img width="1128" height="778" alt="Screenshot 2026-05-14 at 2 03 43 PM" src="https://github.com/user-attachments/assets/7db789a2-3617-4d40-8ae5-bc2c1d087c58" />

---

## 7. Roadmap

Possible future enhancements:

- Connect to a **SQL backend** (DuckDB/Postgres/BigQuery) instead of only CSVs.   
- Add **time‑series EDA** (e.g., revenue by day/week/month, rolling averages).  
- Extend the AI layer into a multi‑agent system (e.g., separate agents for anomalies, segmentation, marketing recommendations).  
- Export AI‑generated EDA reports to Markdown/PDF for stakeholders.

---
