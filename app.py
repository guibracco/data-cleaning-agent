"""Streamlit interface for the Data Cleaning Agent."""

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from data_cleaning_agent import LightweightDataCleaningAgent

load_dotenv()

BASELINE_CLEANING_STEPS = [
    "Standardize column names (trim + lower_snake_case)",
    "Remove columns with more than 40% missing values",
    "Impute missing values (mean for numeric, mode for categorical)",
    "Remove duplicate rows",
    "Remove numeric outliers using IQR (conservative thresholds)",
    "Clean text columns (trim whitespace, collapse repeated spaces)",
    "Conservative datatype inconsistency checks and selective conversion",
]


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages by column."""
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_pct"])

    missing = df.isna().sum().sort_values(ascending=False)
    summary = missing.to_frame("missing_count")
    summary["missing_pct"] = (summary["missing_count"] / len(df) * 100).round(2)
    return summary


def render_summary_statistics(df: pd.DataFrame, label: str):
    """Render a tabbed summary view for a DataFrame."""
    st.subheader(f"{label} Summary Statistics")
    numeric_tab, categorical_tab, missing_tab, dtypes_tab = st.tabs(
        ["Numeric", "Categorical", "Missing Values", "Data Types"]
    )

    with numeric_tab:
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty:
            st.info("No numeric columns available.")
        else:
            st.dataframe(numeric_df.describe().transpose(), width="stretch")

    with categorical_tab:
        categorical_df = df.select_dtypes(exclude="number")
        if categorical_df.empty:
            st.info("No categorical/text columns available.")
        else:
            categorical_summary = categorical_df.describe(include="all").transpose()
            categorical_summary = categorical_summary.where(
                categorical_summary.notna(), ""
            )
            st.dataframe(
                categorical_summary,
                width="stretch",
            )

    with missing_tab:
        st.dataframe(get_missing_summary(df), width="stretch")

    with dtypes_tab:
        dtypes = pd.DataFrame(
            {"column": df.columns, "dtype": df.dtypes.astype(str)}
        ).set_index("column")
        st.dataframe(dtypes, width="stretch")


st.set_page_config(page_title="Data Cleaning Agent", page_icon="🧹", layout="wide")
st.title("🧹 Data Cleaning Agent")
st.caption("Upload a CSV file to preview quality metrics, run cleaning, and compare before/after summaries.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Unable to read the CSV file: {error}")
        st.stop()

    st.subheader("Original Dataset Preview")
    st.write(f"Shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    st.dataframe(df_raw.head(20), width="stretch")
    render_summary_statistics(df_raw, "Original Dataset")

    st.subheader("Cleaning Plan")
    st.write("Baseline steps that will run (plus any custom instructions):")
    for index, step in enumerate(BASELINE_CLEANING_STEPS, start=1):
        st.write(f"{index}. {step}")

    user_instructions = st.text_area(
        "Optional custom cleaning instructions",
        placeholder="Example: Keep CustomerID unchanged, but standardize order_date to datetime.",
    )

    if st.button("Clean Data", type="primary"):
        with st.spinner("Generating and executing cleaning code..."):
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            agent = LightweightDataCleaningAgent(model=llm, log=True)
            agent.invoke_agent(
                data_raw=df_raw,
                user_instructions=user_instructions.strip() or None,
            )
            df_cleaned = agent.get_data_cleaned()
            data_cleaner_error = agent.get_data_cleaner_error()
            cleaning_steps = agent.get_cleaning_steps()
            generated_code = agent.get_data_cleaner_function()

        if data_cleaner_error:
            st.error(data_cleaner_error)
        elif df_cleaned is None:
            st.error("The cleaning agent did not return a cleaned dataset.")
        else:
            st.success("Cleaning complete.")

            st.subheader("Executed Cleaning Steps")
            if cleaning_steps:
                for index, step in enumerate(cleaning_steps, start=1):
                    st.write(f"{index}. {step}")
            else:
                st.info("No explicit step log was returned for this run.")

            st.subheader("Cleaned Dataset Preview")
            st.write(f"Shape: {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")
            st.dataframe(df_cleaned.head(20), width="stretch")
            render_summary_statistics(df_cleaned, "Cleaned Dataset")

            csv = df_cleaned.to_csv(index=False)
            st.download_button(
                "Download Cleaned Data",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv",
            )

        with st.expander("Generated Cleaning Code"):
            if generated_code:
                st.code(generated_code, language="python")
            else:
                st.info("No code snippet available.")
