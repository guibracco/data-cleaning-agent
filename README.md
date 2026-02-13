# Data Cleaning Agent

An AI-powered data cleaning agent built with LangChain + LangGraph and delivered through Streamlit.
It generates pandas cleaning code, executes it, retries on failures, and now exposes clearer before/after data quality insights.

## Features

### Core agent workflow

The agent follows a simple workflow:
1. **Analyze**: Examines your dataset structure and identifies data quality issues
2. **Generate**: Uses an LLM to create custom Python cleaning code based on the data
3. **Execute**: Runs the generated code to clean your data
4. **Retry**: Automatically fixes errors if the generated code fails (up to 3 attempts)

This approach combines the flexibility of LLMs with the reliability of pandas operations.

### Baseline cleaning steps (now expanded)

1. Standardize column names (trim + snake_case)
2. Remove columns with >40% missing values
3. Impute missing values (mean for numeric, mode for categorical)
4. Remove duplicate rows
5. Remove outliers from numeric columns using conservative IQR filtering
6. Clean text fields (trim and normalize repeated spaces)
7. Perform conservative datatype inconsistency checks/conversions

### New UX improvements

1. Original dataset preview before cleaning
2. Original dataset summary statistics (numeric, categorical, missing values, dtypes)
3. Clear display of planned cleaning steps
4. Clear display of executed cleaning steps returned by the agent
5. Cleaned dataset summary statistics for before/after comparison
6. Optional custom cleaning instructions in the UI
7. Generated code visibility for transparency and debugging

## Setup

### Prerequisites

- Python 3.9+ (except 3.9.7 due to Streamlit compatibility)
- Poetry
- OpenAI API key

### Installation

1. Install dependencies:
```bash
poetry install
```

2. Configure environment variables:
```bash
cp .env.example .env
```

Add your OpenAI API key to `.env`:
```env
OPENAI_API_KEY=your-key-here
```

## Usage

### Streamlit app

```bash
poetry run streamlit run app.py
```

Flow:
1. Upload CSV
2. Inspect original preview + summary stats
3. Add optional instructions
4. Run cleaning and review executed steps
5. Compare cleaned preview + summary stats
6. Download cleaned CSV

### Python API

For programmatic use or integration into data pipelines:

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from data_cleaning_agent import LightweightDataCleaningAgent

# Initialize the agent with an LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = LightweightDataCleaningAgent(model=llm)

# Load your messy data
df = pd.read_csv("your_data.csv")

# Run the cleaning agent
agent.invoke_agent(data_raw=df)

# Get the cleaned dataset
cleaned_df = agent.get_data_cleaned()

# Save or use the cleaned data
cleaned_df.to_csv("cleaned_data.csv", index=False)
```

**Optional: Provide custom instructions**

```python
# Give specific cleaning instructions to the agent
agent.invoke_agent(
    data_raw=df,
    user_instructions="Remove columns with more than 30% missing values and standardize date formats"
)
```

**Optional: Cleaning steps and generated code**

```python
# Get the cleaning steps and generated code
cleaning_steps = agent.get_cleaning_steps()
generated_code = agent.get_data_cleaner_function()
```

## Docker

Build image:

```bash
docker build -t data-cleaning-agent .
```

Run container:

```bash
docker run --rm -p 8501:8501 --env-file .env data-cleaning-agent
```

Open `http://localhost:8501` in your browser.

## Project Structure

```text
data-cleaning-agent/
├── data_cleaning_agent/
│   ├── __init__.py
│   ├── data_cleaning_agent.py
│   └── utils.py
├── app.py
├── Dockerfile
├── .dockerignore
├── pyproject.toml
├── poetry.lock
└── README.md
```
