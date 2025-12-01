"""
Exception handler helper for SAP dashboard.

Provides:
- load_exception_csv(filepath): load and cache the exception report CSV
- get_columns_info(df): return a short description of columns and sample values
- extract_filters_from_llm(query, columns_info): ask LLM to return filters JSON (with fallback)
- apply_filters(df, filters): apply exact-column filters to dataframe
- suggest_charts_from_llm(query, data_sample, columns_info): ask LLM for chart suggestions (with fallback)

Usage:
Import these functions in your Streamlit app and call them before sending large data to the LLM.
"""

from functools import lru_cache
from typing import Dict, Any, List
import pandas as pd
import json
import re
import logging
from pathlib import Path
from pepsico_llm import invoke_llm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Default CSV path relative to this file's location
DEFAULT_CSV = str(Path(__file__).parent.parent.parent / 'data' / 'Sales Order Exception report 13 and 14 Nov 2025.csv')


@lru_cache(maxsize=2)
def load_exception_csv(filepath: str = DEFAULT_CSV) -> pd.DataFrame:
    """Load the exception CSV and cache it in memory for the process lifetime.

    Keeps loading logic tolerant to encodings and parsing errors.
    """
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding, on_bad_lines='skip', engine='python')
            logger.info(f"Loaded {len(df):,} rows from {filepath} using encoding {encoding}")
            return df
        except Exception:
            continue

    # Last resort, let pandas try default
    df = pd.read_csv(filepath, engine='python')
    logger.info(f"Loaded {len(df):,} rows from {filepath} using default encoding")
    return df


def get_columns_info(df: pd.DataFrame) -> str:
    """Return a short human-friendly description of dataframe columns and samples."""
    cols = df.columns.tolist()
    info_lines: List[str] = []
    info_lines.append(f"Columns ({len(cols)}): {', '.join(cols[:20])}{'...' if len(cols)>20 else ''}")

    # Helpful sample values for common columns
    for key in ['Plant', 'Plant(Location)', 'Material', 'Sales Order Number']:
        if key in df.columns:
            uniques = df[key].dropna().astype(str).unique()[:8].tolist()
            info_lines.append(f"Sample values - {key}: {uniques}")

    return '\n'.join(info_lines)


def _init_llm():
    # Commented out ChatOllama - using PepGenX API only
    # try:
    #     from langchain_ollama import ChatOllama
    #     from langchain_core.prompts import ChatPromptTemplate
    #     from langchain_core.output_parsers import JsonOutputParser
    #
    #     llm = ChatOllama(model="llama3.2", temperature=0)
    #     return llm, ChatPromptTemplate, JsonOutputParser
    # except Exception as e:
    #     logger.warning("LLM imports failed or Ollama client not available: %s", e)
    return None, None, None


def extract_filters_from_llm(query: str, columns_info: str) -> Dict[str, Any]:
    """Return a dict of filters extracted by the LLM using exact column names.

    If LLM is unavailable or fails, falls back to simple regex-based extraction (plant, material).
    """
    llm, ChatPromptTemplate, JsonOutputParser = _init_llm()

    prompt_text = (
        "You are a data assistant. Given this user query and the available columns, "
        "return a JSON object with a single key 'filters' whose value is an object mapping EXACT column names to filter values. "
        "If no filters can be found, return {\"filters\": {}}. Use only column names present in the schema.\n\n"
        "Available columns and samples:\n{columns_info}\n\nUser Query:\n{query}\n"
    )

    if llm and ChatPromptTemplate and JsonOutputParser:
        try:
            intent_prompt = ChatPromptTemplate.from_messages([
                ("system", prompt_text),
                ("user", "{query}")
            ])

            chain = intent_prompt | llm | JsonOutputParser()
            result = chain.invoke({"query": query, "columns_info": columns_info})
            # Expected structure: {"filters": {"Plant": "1007"}}
            if isinstance(result, dict) and 'filters' in result:
                return result['filters'] if result['filters'] is not None else {}
        except Exception as e:
            logger.warning("LLM filter extraction failed: %s", e)

    # If ChatOllama not available, try Pepsico LLM API as fallback
    try:
        payload = {
            "generation_model": "gpt-4o",
            "max_tokens": 700,
            "temperature": 0.0,
            "top_p": 0.01,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "tools": [],
            "tools_choice": "none",
            "system_prompt": prompt_text.format(columns_info=columns_info, query=query),
            "custom_prompt": [
                {"role": "user", "content": query}
            ],
            "model_provider_name": "openai"
        }
        resp = invoke_llm(payload)
        if isinstance(resp, dict) and resp.get('error'):
            logger.warning("Pepsico LLM error: %s", resp.get('error'))
        else:
            # If API returned JSON in 'response', try parse
            if isinstance(resp, dict) and 'response' in resp:
                try:
                    response_text = resp['response']
                    # Remove markdown code blocks if present
                    if '```json' in response_text:
                        response_text = response_text.split('```json')[1].split('```')[0].strip()
                    elif '```' in response_text:
                        response_text = response_text.split('```')[1].split('```')[0].strip()
                    parsed = json.loads(response_text)
                    if isinstance(parsed, dict) and 'filters' in parsed:
                        return parsed['filters'] if parsed['filters'] is not None else {}
                except Exception as e:
                    logger.warning(f"Failed to parse filter JSON: {e}")
                    pass
            elif isinstance(resp, dict) and 'filters' in resp:
                return resp['filters']
    except Exception as e:
        logger.warning("Pepsico LLM invocation failed: %s", e)

    # Fallback extraction
    fallback_filters: Dict[str, Any] = {}
    # plant extraction
    plant_match = re.search(r"plant\s*[:=]?\s*'?(\d{3,6})'?", query, flags=re.IGNORECASE)
    if plant_match:
        fallback_filters['Plant'] = plant_match.group(1)
    else:
        plant_match2 = re.search(r"plant\s*(?:number\s*)?(\d{3,6})", query, flags=re.IGNORECASE)
        if plant_match2:
            fallback_filters['Plant'] = plant_match2.group(1)

    # material extraction
    material_match = re.search(r"material\s*[:=]?\s*'?(\w{4,})'?", query, flags=re.IGNORECASE)
    if material_match:
        fallback_filters['Material'] = material_match.group(1)

    return fallback_filters


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply exact-column filters (equality) to dataframe."""
    if not filters:
        return df.copy()

    filtered = df.copy()
    for col, value in filters.items():
        if col in filtered.columns:
            filtered = filtered[filtered[col].astype(str) == str(value)]
        else:
            logger.warning("Requested filter column '%s' not found in dataframe", col)

    return filtered


def suggest_charts_from_llm(query: str, data_sample: Dict[str, Any], columns_info: str) -> Dict[str, Any]:
    """Ask the LLM for chart suggestions given a small data sample. Returns chart configs.

    Fallback returns simple suggestions computed locally.
    """
    llm, ChatPromptTemplate, JsonOutputParser = _init_llm()

    prompt_text = (
        "You are a data visualization expert. Given the user's query, a small data sample, and columns info, "
        "return JSON with 'charts' (list) and 'tables' (list). Each chart should include type, title, and relevant columns.\n\n"
        "Columns Info:\n{columns_info}\n\nData Sample:\n{data_sample}\n\nUser Query:\n{query}\n"
    )

    if llm and ChatPromptTemplate and JsonOutputParser:
        try:
            chart_prompt = ChatPromptTemplate.from_messages([
                ("system", prompt_text),
                ("user", "{query}")
            ])

            chain = chart_prompt | llm | JsonOutputParser()
            result = chain.invoke({"query": query, "data_sample": json.dumps(data_sample), "columns_info": columns_info})
            if isinstance(result, dict):
                return result
        except Exception as e:
            logger.warning("LLM chart suggestion failed: %s", e)

    # Try Pepsico LLM API as fallback
    try:
        payload = {
            "generation_model": "gpt-4o",
            "max_tokens": 1500,
            "temperature": 0.2,
            "top_p": 0.01,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "tools": [],
            "tools_choice": "none",
            "system_prompt": prompt_text.format(columns_info=columns_info, data_sample=json.dumps(data_sample), query=query),
            "custom_prompt": [
                {"role": "user", "content": f"{query}\n\nData Sample:\n{json.dumps(data_sample)}"}
            ],
            "model_provider_name": "openai"
        }
        resp = invoke_llm(payload)
        if isinstance(resp, dict) and resp.get('error'):
            logger.warning("Pepsico LLM error for charts: %s", resp.get('error'))
        else:
            if isinstance(resp, dict) and 'response' in resp:
                try:
                    response_text = resp['response']
                    # Remove markdown code blocks if present
                    if '```json' in response_text:
                        response_text = response_text.split('```json')[1].split('```')[0].strip()
                    elif '```' in response_text:
                        response_text = response_text.split('```')[1].split('```')[0].strip()
                    parsed = json.loads(response_text)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception as e:
                    logger.warning(f"Failed to parse chart suggestion JSON: {e}")
                    pass
            elif isinstance(resp, dict):
                return resp
    except Exception as e:
        logger.warning("Pepsico LLM chart invocation failed: %s", e)

    # Fallback heuristic suggestions
    charts: List[Dict[str, Any]] = []
    cols = data_sample.get('columns', [])
    if 'Plant' in cols or 'Plant(Location)' in cols:
        plant_col = 'Plant' if 'Plant' in cols else 'Plant(Location)'
        charts.append({
            'type': 'bar',
            'title': 'Exceptions by Plant',
            'x_column': plant_col,
            'y_column': 'count',
            'agg_function': 'count',
            'limit': 10
        })

    if 'Material' in cols:
        charts.append({
            'type': 'bar',
            'title': 'Top Materials with Exceptions',
            'x_column': 'Material',
            'y_column': 'count',
            'agg_function': 'count',
            'limit': 10
        })

    # Always include a sample table
    charts.append({
        'type': 'table',
        'title': 'Sample Exceptions',
        'columns': cols[:8],
        'limit': 50
    })

    return {'charts': charts, 'tables': []}


if __name__ == '__main__':
    # Simple CLI demo
    try:
        df = load_exception_csv()
    except FileNotFoundError:
        print(f"CSV not found. Please place '{DEFAULT_CSV}' in the working directory.")
        raise

    print('\n' + '='*10 + ' Columns Info ' + '='*10)
    print(get_columns_info(df))
    print('\n')

    sample = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'sample_rows': df.head(5).to_dict('records')
    }

    user_q = input("Enter a test query (e.g. Plant 1007): ")
    filters = extract_filters_from_llm(user_q, get_columns_info(df))
    print('-'*10 + ' Extracted Filters ' + '-'*10)
    print(json.dumps(filters, indent=2))
    print('\n')

    filtered = apply_filters(df, filters)
    print(f"Filtered rows: {len(filtered):,}")

    chart_suggestions = suggest_charts_from_llm(user_q, sample, get_columns_info(df))
    print('-'*10 + ' Chart Suggestions ' + '-'*10)
    print(json.dumps(chart_suggestions, indent=2))
    print('\n')
