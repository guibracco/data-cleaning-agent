# Utility functions for lightweight data cleaning agent

import re
import logging
import pandas as pd
from langchain_core.output_parsers import BaseOutputParser

logger = logging.getLogger(__name__)


class PythonOutputParser(BaseOutputParser):
    """Extract Python code from LLM responses."""
    
    def parse(self, text: str):
        """Extract code from ```python``` blocks or return text as-is."""
        python_code_match = re.search(r'```python(.*?)```', text, re.DOTALL)
        if python_code_match:
            return python_code_match.group(1).strip()
        return text


def get_dataframe_summary(df: pd.DataFrame) -> str:
    """
    Generate a simple summary of a DataFrame for the LLM.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to summarize.
    
    Returns
    -------
    str
        A text summary of the DataFrame.
    """
    if len(df) == 0:
        missing_stats = pd.Series(0.0, index=df.columns, dtype=float)
    else:
        missing_stats = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    missing_summary = "\n".join([f"{col}: {val:.2f}%" for col, val in missing_stats.items()])
    
    column_types = "\n".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])
    
    summary = f"""
        Dataset Summary:
        ----------------
        Column Data Types:
        {column_types}

        Missing Value Percentage:
        {missing_summary}"""

    return summary.strip()


def execute_agent_code(
    state,
    data_key,
    code_snippet_key,
    result_key,
    error_key,
    agent_function_name,
    steps_key=None,
):
    """
    Execute the generated agent code on the data.
    
    Parameters
    ----------
    state : dict
        The current state containing data and code.
    data_key : str
        Key in state where the input data is stored.
    code_snippet_key : str
        Key in state where the generated code is stored.
    result_key : str
        Key to store the result in.
    error_key : str
        Key to store any error message in.
    agent_function_name : str
        Name of the function to execute from the generated code.
    steps_key : str, optional
        Key to store a list of executed cleaning steps in.
    
    Returns
    -------
    dict
        Dictionary with result, error, and optional cleaning-step keys.
    """
    logger.info("Executing agent code")
    
    data = state.get(data_key)
    agent_code = state.get(code_snippet_key)
    df = pd.DataFrame.from_dict(data)
    
    # Execute the LLM-generated code in isolated namespace
    # Note: exec() can be risky - only use with trusted LLM-generated code
    local_vars = {}
    global_vars = {}
    exec(agent_code, global_vars, local_vars)
    
    # Get the function from executed code
    agent_function = local_vars.get(agent_function_name)
    if not agent_function or not callable(agent_function):
        raise ValueError(f"Function '{agent_function_name}' not found in generated code.")
    
    # Run the function and handle errors
    agent_error = None
    result = None
    cleaning_steps = []
    try:
        result = agent_function(df)

        # Backward compatibility:
        # - DataFrame return: df
        # - Tuple return: (df_or_dict, cleaning_steps)
        # - Dict return: {"data_cleaned": ..., "cleaning_steps": ...}
        if isinstance(result, tuple) and len(result) == 2:
            result, cleaning_steps = result
        elif isinstance(result, dict) and "data_cleaned" in result:
            cleaning_steps = result.get("cleaning_steps", [])
            result = result["data_cleaned"]

        if isinstance(result, pd.DataFrame):
            result = result.to_dict()

        if cleaning_steps is None:
            cleaning_steps = []
        elif not isinstance(cleaning_steps, list):
            cleaning_steps = [str(cleaning_steps)]
        else:
            cleaning_steps = [str(step) for step in cleaning_steps]
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        agent_error = f"An error occurred during data cleaning: {str(e)}"

    response = {result_key: result, error_key: agent_error}
    if steps_key is not None:
        response[steps_key] = cleaning_steps
    return response


def fix_agent_code(state, code_snippet_key, error_key, llm, prompt_template, function_name, retry_count_key="retry_count"):
    """
    Fix errors in the generated agent code using the LLM.
    
    Parameters
    ----------
    state : dict
        The current state containing code and error information.
    code_snippet_key : str
        Key in state where the broken code is stored.
    error_key : str
        Key in state where the error message is stored.
    llm : LLM
        The language model to use for fixing the code.
    prompt_template : str
        Template for the fix prompt (should have {code_snippet}, {error}, {function_name} placeholders).
    function_name : str
        Name of the function being fixed.
    retry_count_key : str, optional
        Key in state for tracking retry count. Defaults to "retry_count".
    
    Returns
    -------
    dict
        Dictionary with updated code, cleared error, and incremented retry count.
    """
    logger.info("Fixing agent code")
    logger.debug(f"Retry count: {state.get(retry_count_key)}")
    
    code_snippet = state.get(code_snippet_key)
    error_message = state.get(error_key)
    
    # Create the fix prompt
    prompt = prompt_template.format(
        code_snippet=code_snippet,
        error=error_message,
        function_name=function_name,
    )
    
    # Get fixed code from LLM
    response = (llm | PythonOutputParser()).invoke(prompt)
    
    return {
        code_snippet_key: response,
        error_key: None,
        retry_count_key: state.get(retry_count_key) + 1
    }
