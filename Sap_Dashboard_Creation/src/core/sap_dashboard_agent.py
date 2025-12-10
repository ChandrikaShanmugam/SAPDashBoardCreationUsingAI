"""
SAP Data Dashboard Generator using LLM
Dynamic dashboard creation based on natural language queries
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Dict, List
from pepsico_llm import invoke_llm
import database_schema as db_schema
import json
import logging
import difflib
import time
from datetime import datetime
import sys
import os
from pathlib import Path
import base64
from exception_handler import (
    load_exception_csv,
    get_columns_info,
    extract_filters_from_llm,
    apply_filters,
    suggest_charts_from_llm,
)
from prompt_manager import PromptTemplateManager

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Inject custom CSS for 30% sidebar and 70% main content
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 30vw;
        max-width: 30vw;
        width: 30vw;
    }
    .main .block-container {
        max-width: 70vw;
        margin-left: 32vw;
    }
    </style>
    """,
    unsafe_allow_html=True
)
logger = logging.getLogger(__name__)

# ===========================
# 1. DATA LOADING FUNCTIONS
# ===========================

@st.cache_data
def load_sap_data():
    """Load all SAP data files including COF inventory tables"""
    logger.info("=" * 80)
    logger.info("LOADING SAP DATA FILES (4 TABLES)")
    logger.info("=" * 80)
    start_time = time.time()
    
    def load_csv_with_encoding(filepath):
        """Try to load CSV with different encodings and handle parsing issues"""
        # Get text columns from schema instead of hardcoding
        string_columns = db_schema.get_text_columns(table="all")
        
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                # Try with error_bad_lines=False (pandas <2.0) or on_bad_lines='skip' (pandas >=2.0)
                return pd.read_csv(filepath, encoding=encoding, on_bad_lines='skip', engine='python',
                                 dtype={col: str for col in string_columns})
            except (UnicodeDecodeError, ValueError):
                try:
                    # Fallback: try without on_bad_lines parameter
                    return pd.read_csv(filepath, encoding=encoding, engine='python',
                                     dtype={col: str for col in string_columns})
                except:
                    continue
        # If all fail, try with error handling
        try:
            return pd.read_csv(filepath, encoding='latin-1', on_bad_lines='skip', engine='python',
                             dtype={col: str for col in string_columns})
        except:
            return pd.read_csv(filepath, encoding='latin-1', engine='python',
                             dtype={col: str for col in string_columns})
    
    try:
        # Commented out auth files - focusing on exception report only
        # logger.info("Loading: Authorized To Sell Yes.csv")
        # auth_yes = load_csv_with_encoding('Authorized To Sell Yes.csv')
        # logger.info(f"✓ Loaded {len(auth_yes):,} records from Authorized To Sell Yes.csv")
        # 
        # logger.info("Loading: Authorized to Sell No.csv")
        # auth_no = load_csv_with_encoding('Authorized to Sell No.csv')
        # logger.info(f"✓ Loaded {len(auth_no):,} records from Authorized to Sell No.csv")
        
        # Create empty dataframes for auth data (to avoid errors in code that references them)
        #auth_yes = pd.DataFrame()
        #auth_no = pd.DataFrame()
        
        data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
        
        # Load Sales Order Exception Report (Table 1)
        logger.info("Loading: Sales Order Exception report.csv (via helper)")
        exception_file = str(data_dir / 'Sales Order Exception report.csv')
        logger.info(f"Attempting to load from: {exception_file}")
        exception_report = load_exception_csv(exception_file)
        logger.info(f"✓ Loaded {len(exception_report):,} records from Sales Order Exception Report")
        logger.info(f"  Columns: {len(exception_report.columns)}")
        
        # Normalize Material numbers in exception report (strip leading zeros)
        if 'Material' in exception_report.columns:
            logger.info("Normalizing Material numbers in exception report (removing leading zeros)")
            exception_report['Material_Original'] = exception_report['Material'].copy()
            exception_report['Material'] = exception_report['Material'].astype(str).str.split('.').str[0].str.lstrip('0').replace('', '0')
            logger.info(f"  Sample normalized: {exception_report[['Material_Original', 'Material']].head(3).to_dict()}")

        # Use exception_report as the primary exceptions data
        so_exceptions = exception_report
        
        # Load A1P Location Sequence (Table 2)
        logger.info("Loading: A1P_Locn_Seq_EXPORT.csv")
        location_file = str(data_dir / 'A1P_Locn_Seq_EXPORT.csv')
        logger.info(f"Attempting to load from: {location_file}")
        location_sequence = load_csv_with_encoding(location_file)
        logger.info(f"✓ Loaded {len(location_sequence):,} records from A1P Location Sequence")
        logger.info(f"  Columns: {len(location_sequence.columns)}")
        
        # Normalize Material numbers in location sequence (strip leading zeros)
        if 'Material' in location_sequence.columns:
            logger.info("Normalizing Material numbers in location sequence (removing leading zeros)")
            location_sequence['Material_Original'] = location_sequence['Material'].copy()
            location_sequence['Material'] = location_sequence['Material'].astype(str).str.split('.').str[0].str.lstrip('0').replace('', '0')
            logger.info(f"  Sample normalized: {location_sequence[['Material_Original', 'Material']].head(3).to_dict()}")
        
        # Verify foreign key columns exist
        common_cols = ['Plant', 'Material']
        location_cols_exist = all(col in location_sequence.columns for col in ['Plant(Location)', 'Material'])
        sales_cols_exist = all(col in exception_report.columns for col in common_cols)
        
        if location_cols_exist and sales_cols_exist:
            logger.info("✓ Foreign key columns verified in both tables")
            logger.info(f"  Sales Order: {common_cols}")
            logger.info(f"  Location Sequence: ['Plant(Location)', 'Material']")
        else:
            logger.warning("⚠️ Foreign key columns not found in one or both tables")
        
        # Load COF Inventory Net Price (Table 3)
        logger.info("Loading: Cof_Inven_NetPrice_Material.csv")
        cof_inventory_file = str(data_dir / 'Cof_Inven_NetPrice_Material.csv')
        cof_inventory = load_csv_with_encoding(cof_inventory_file)
        logger.info(f"✓ Loaded {len(cof_inventory):,} records from COF Inventory Net Price")
        logger.info(f"  Columns: {len(cof_inventory.columns)}")
        
        # Normalize Material numbers in COF inventory (strip leading zeros)
        if 'Material' in cof_inventory.columns:
            logger.info("Normalizing Material numbers in COF inventory (removing leading zeros)")
            cof_inventory['Material_Original'] = cof_inventory['Material'].copy()
            cof_inventory['Material'] = cof_inventory['Material'].astype(str).str.split('.').str[0].str.lstrip('0').replace('', '0')
            logger.info(f"  Sample normalized: {cof_inventory[['Material_Original', 'Material']].head(3).to_dict()}")
        
        # Load COF Material Pricing (Table 4)
        logger.info("Loading: Cof_Inven_NetPrice_MaterialPrice.csv")
        cof_pricing_file = str(data_dir / 'Cof_Inven_NetPrice_MaterialPrice.csv')
        cof_pricing = load_csv_with_encoding(cof_pricing_file)
        logger.info(f"✓ Loaded {len(cof_pricing):,} records from COF Material Pricing")
        logger.info(f"  Columns: {len(cof_pricing.columns)}")
        
        # Normalize Material numbers in COF pricing (strip leading zeros)
        if 'Material' in cof_pricing.columns:
            logger.info("Normalizing Material numbers in COF pricing (removing leading zeros)")
            cof_pricing['Material_Original'] = cof_pricing['Material'].copy()
            cof_pricing['Material'] = cof_pricing['Material'].astype(str).str.split('.').str[0].str.lstrip('0').replace('', '0')
            logger.info(f"  Sample normalized: {cof_pricing[['Material_Original', 'Material']].head(3).to_dict()}")
        
        elapsed = time.time() - start_time
        logger.info(f"✓ All data loaded successfully in {elapsed:.2f} seconds")
        logger.info("=" * 80)
        
        data = {
            'exception_report': exception_report,
            'location_sequence': location_sequence,
            'cof_inventory': cof_inventory,
            'cof_pricing': cof_pricing
        }
        return data
        
    except Exception as e:
        logger.error(f"✗ Error loading data: {str(e)}")
        logger.exception("Full traceback:")
        raise

# ===========================
# 2. INTENT CLASSIFICATION
# ===========================

class IntentClassifier:
    """Classify user intent and extract dashboard requirements with column-aware filtering"""
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        logger.info("Initializing IntentClassifier with PepGenX API")
        self.data = data
        # Commented out ChatOllama - using PepGenX API only
        # try:
        #     self.llm = ChatOllama(model="llama3.2", temperature=0)
        #     logger.info("✓ LLM initialized successfully")
        # except Exception as e:
        #     logger.warning(f"LLM init failed, falling back to Pepsico LLM wrapper: {str(e)}")
        self.llm = None
        
        # Initialize prompt manager
        self.prompt_manager = PromptTemplateManager()
        
        # NOTE: Schema information is now dynamically injected via prompt_manager
        # No need to pre-generate columns_info - it's generated on-demand
        
        # Load prompts from template files
        # STAGE 1: FILTER EXTRACTION + AGGREGATION DETECTION (with cross-table support)
        self.intent_system_prompt = self.prompt_manager.get_template('filter_extraction')
        
        # STAGE 2: CHART GENERATION BASED ON FILTERED DATA  
        self.chart_system_prompt = self.prompt_manager.get_template('chart_generation')
        
        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SAP data analyst. Classify the user's query and determine what dashboard to create.
            
Available Data Sources and Their Columns:
{columns_info}

Return JSON with:
- intent: "authorized_to_sell", "exceptions", "plant_analysis", "material_analysis", or "overview"
- data_sources: list of required data sources ["auth_yes", "auth_no", "so_exceptions"]
- filters: object with EXACT column names as keys and filter values. Examples:
  * {{"Plant": "1007"}} - for specific plant
  * {{"Material": "000000000300005846"}} - for specific material
  * {{"Plant(Location)": "1007"}} - use EXACT column name from schema
- show_material_details: true/false - whether to show detailed material information

IMPORTANT: Use EXACT column names from the schema above when creating filters.

Example queries:
- "Show me plant 1007 authorized data" → {{"intent": "plant_analysis", "filters": {{"Plant(Location)": "1007"}}}}
- "Give me plant 1007 authorized data and material details" → {{"intent": "plant_analysis", "filters": {{"Plant(Location)": "1007"}}, "show_material_details": true}}
- "What are the sales exceptions?" → {{"intent": "exceptions", "filters": {{}}}}
"""),
            ("user", "{query}")
        ])
        
        # Chart Generation Prompt
        self.chart_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data visualization expert. Based on the user's query and the filtered data sample, suggest appropriate charts and tables.

Data Sample:
{data_sample}

Return JSON with:
- charts: list of chart configurations, each with:
  * type: "bar", "pie", "line", "scatter", "table"
  * title: chart title
  * x_column: column for x-axis (for bar, line, scatter)
  * y_column: column for y-axis (for bar, line, scatter) or "count" for count aggregation
  * group_by: column to group by (optional)
  * agg_function: "count", "sum", "mean", "max", "min" (for aggregations)
  * limit: number of top items to show (optional, default 10)
- tables: list of table configurations with:
  * columns: list of column names to display
  * title: table title
  * limit: number of rows (optional, default 50)

Example:
{{
  "charts": [
    {{"type": "bar", "title": "Material Count by Plant", "x_column": "Plant", "y_column": "count", "agg_function": "count", "limit": 10}},
    {{"type": "pie", "title": "Authorization Status", "group_by": "Auth Sell Flag Description", "agg_function": "count"}}
  ],
  "tables": [
    {{"columns": ["Material", "Material Descrption", "Plant", "Auth Sell Flag Description"], "title": "Material Details", "limit": 50}}
  ]
}}
"""),
            ("user", "{query}")
        ])

    def _map_column_name(self, requested: str, available_columns: List[str], cutoff: float = 0.7) -> str:
        """Map a possibly-misspelled or user-friendly column name to an actual column.

        Uses difflib.get_close_matches to find a candidate column name.
        Returns the matched column name or None if no good match found.
        """
        if not requested:
            return None

        # Exact match quick path
        if requested in available_columns:
            return requested

        # Normalize: strip, lower
        req = requested.strip().lower()
        candidates = {col: col.lower() for col in available_columns}

        # Direct lower-case match
        for col, low in candidates.items():
            if low == req:
                return col

        # Fuzzy match on words / close string similarity
        close = difflib.get_close_matches(req, list(candidates.values()), n=1, cutoff=cutoff)
        if close:
            # find original column name for matched lower string
            for col, low in candidates.items():
                if low == close[0]:
                    logger.info(f"Mapped requested column '{requested}' → '{col}' using fuzzy match")
                    return col

        # Try token-based matching (e.g., 'description' -> 'descrption')
        req_tokens = set(req.replace('/', ' ').replace('-', ' ').split())
        best = None
        best_score = 0
        for col in available_columns:
            col_tokens = set(col.lower().replace('/', ' ').replace('-', ' ').split())
            score = len(req_tokens & col_tokens) / max(1, len(req_tokens | col_tokens))
            if score > best_score:
                best_score = score
                best = col

        if best_score >= 0.5:
            logger.info(f"Mapped requested column '{requested}' → '{best}' using token overlap (score={best_score:.2f})")
            return best

        return None
    
    def classify(self, query: str, conversation_history: List[Dict] = None) -> Dict:
        """Extract filters from user query - STAGE 1"""
        logger.info("=" * 80)
        logger.info("STAGE 1: FILTER EXTRACTION FROM QUERY")
        logger.info("=" * 80)
        logger.info(f"User Query: '{query}'")
        
        # Add conversation history context if available
        history_context = ""
        if conversation_history:
            history_context = "\nConversation History:\n" + "\n".join([
                f"Previous Query {i+1}: {item['query']}\nApplied Filters: {item['filters']}"
                for i, item in enumerate(conversation_history[-3:])  # Last 3 interactions
            ])
            logger.info(f"Using conversation history: {len(conversation_history)} previous interactions")
        
        start_time = time.time()
        
        try:
            logger.info("Building LLM prompt with column information...")
            if self.llm:
                chain = self.intent_prompt | self.llm | JsonOutputParser()
                logger.info("Sending request to local LLM (ChatOllama)...")
                result = chain.invoke({
                    "query": query,
                    "columns_info": self.prompt_manager.get_columns_info()
                })
            else:
                # Use Pepsico LLM API with updated prompt template
                logger.info("Using Pepsico LLM API for filter extraction...")
                
                # Use the new format_filter_extraction_prompt method that includes:
                # - {columns_info}: All 84 columns (69 Sales Order + 15 Location)
                # - {relationship_info}: Foreign key relationships (Plant, Material)
                # - Cross-table filtering examples and guidance
                formatted_system_prompt = self.prompt_manager.format_filter_extraction_prompt(query)
                logger.info(f"System prompt length: {len(formatted_system_prompt)} chars")
                logger.info("✓ Prompt includes cross-table filtering guidance")
                
                # Include conversation history in the prompt
                enhanced_query = query
                if history_context:
                    enhanced_query = f"{query}\n\n{history_context}"
                
                payload = {
                    "generation_model": "gpt-4o",
                    "max_tokens": 500,
                    "temperature": 0.0,
                    "top_p": 0.01,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                    "tools": [],
                    "tools_choice": "none",
                    "system_prompt": formatted_system_prompt,
                    "custom_prompt": [
                        {"role": "user", "content": enhanced_query}
                    ],
                    "model_provider_name": "openai"
                }
                logger.info(f"Sending enhanced query with history: '{enhanced_query}'")
                resp = invoke_llm(payload)
                logger.info(f"Raw API response: {resp}")
                
                # Try to extract JSON from response
                if isinstance(resp, dict) and resp.get('error'):
                    raise Exception(resp['error'])
                    
                # If API returns JSON with 'response' field, parse it
                if isinstance(resp, dict) and 'response' in resp:
                    try:
                        response_text = resp['response']
                        # Remove markdown code blocks if present
                        if '```json' in response_text:
                            response_text = response_text.split('```json')[1].split('```')[0].strip()
                        elif '```' in response_text:
                            response_text = response_text.split('```')[1].split('```')[0].strip()
                        result = json.loads(response_text)
                        logger.info(f"Successfully parsed JSON: {result}")
                    except Exception as e:
                        logger.error(f"Failed to parse JSON: {e}")
                        result = {'filters': {}}
                else:
                    result = resp
            
            # Ensure result has filters key
            if 'filters' not in result:
                logger.warning("No 'filters' key in result, wrapping result")
                result = {'filters': result if isinstance(result, dict) else {}}
            
            elapsed = time.time() - start_time
            logger.info(f"✓ LLM Response received in {elapsed:.2f} seconds")
            logger.info("\n" + "=" * 30 + " Extracted Filters " + "=" * 30)
            logger.info("%s", json.dumps(result.get('filters', {}), indent=2))
            logger.info("" + "=" * 80 + "\n")
            logger.info("=" * 80)
            
            return result
            
        except Exception as e:
            logger.error(f"✗ Error in filter extraction: {str(e)}")
            logger.exception("Full traceback:")
            
            # Fallback: try to extract plant/material from query using regex
            logger.warning("Using fallback filter extraction...")
            query_lower = query.lower()
            fallback_filters = {}
            
            # Try to extract plant number
            import re
            plant_match = re.search(r'plant\s+(\d+)', query_lower)
            if plant_match:
                fallback_filters['Plant'] = plant_match.group(1)
            
            # Try to detect auth flag
            if 'auth flag active' in query_lower or 'authorized' in query_lower:
                fallback_filters['Auth Sell Flag Description'] = 'Yes'
            elif 'not authorized' in query_lower or 'auth flag no' in query_lower:
                fallback_filters['Auth Sell Flag Description'] = 'No'
            
            fallback_result = {'filters': fallback_filters}
            logger.info("\n" + "-" * 20 + " Fallback Filters " + "-" * 20)
            logger.info("%s", json.dumps(fallback_filters, indent=2))
            logger.info("" + "-" * 60 + "\n")
            logger.info("=" * 80)
            return fallback_result
    
    def generate_chart_config(self, query: str, filtered_data: pd.DataFrame, intent: str, conversation_history: List[Dict] = None) -> Dict:
        """Generate chart configuration based on filtered data - Stage 2"""
        logger.info("=" * 80)
        logger.info("STAGE 2: DYNAMIC CHART GENERATION")
        logger.info("=" * 80)
        logger.info(f"Filtered data shape: {filtered_data.shape}")
        
        # Add conversation history context
        history_context = ""
        if conversation_history:
            history_context = "\nConversation History:\n" + "\n".join([
                f"Previous Query {i+1}: {item['query']}\nApplied Filters: {item['filters']}"
                for i, item in enumerate(conversation_history[-3:])  # Last 3 interactions
            ])
            logger.info(f"Using conversation history for chart generation: {len(conversation_history)} previous interactions")
        
        start_time = time.time()
        
        try:
            # Create data sample for LLM
            sample_size = min(10, len(filtered_data))
            data_sample = {
                "shape": filtered_data.shape,
                "columns": filtered_data.columns.tolist(),
                "sample_rows": filtered_data.head(sample_size).to_dict('records'),
                "dtypes": filtered_data.dtypes.astype(str).to_dict()
            }
            
            logger.info("Sending data sample to LLM for chart recommendations...")
            if self.llm:
                chain = self.chart_prompt | self.llm | JsonOutputParser()
                result = chain.invoke({
                    "query": query,
                    "data_sample": json.dumps(data_sample, indent=2)
                })
            else:
                logger.info("Using Pepsico LLM API as fallback for chart recommendations...")
                
                # CRITICAL: Use columns from ACTUAL filtered data, not schema
                # This ensures LLM only references columns that exist in this specific dataset
                actual_columns = filtered_data.columns.tolist()
                all_cols = ', '.join(actual_columns)
                
                logger.info(f"Filtered data has {len(actual_columns)} columns:")
                logger.info(f"  Available columns: {all_cols[:200]}...")  # Show first 200 chars
                
                # Get chart columns from database schema as guidance
                chart_columns = db_schema.get_common_chart_columns()
                
                # Filter chart column suggestions to only include those in actual data
                bar_chart_cols = [c for c in chart_columns['for_bar_charts'] if c in actual_columns]
                pie_chart_cols = [c for c in chart_columns['for_pie_charts'] if c in actual_columns]
                agg_cols = [c for c in chart_columns['for_aggregation'] if c in actual_columns]
                detail_cols = [c for c in chart_columns['for_details'] if c in actual_columns]
                
                logger.info(f"  Aggregation columns available: {', '.join(agg_cols) if agg_cols else 'NONE'}")
                
                # Format the chart prompt with dynamic columns from ACTUAL data
                formatted_system_prompt = self.prompt_manager.format_template(
                    'chart_generation',
                    data_sample=json.dumps(data_sample, indent=2),
                    all_columns=all_cols,
                    bar_chart_columns=', '.join(bar_chart_cols) if bar_chart_cols else all_cols,
                    pie_chart_columns=', '.join(pie_chart_cols) if pie_chart_cols else all_cols,
                    aggregation_columns=', '.join(agg_cols) if agg_cols else 'Order Quantity Sales Unit',
                    detail_columns=', '.join(detail_cols) if detail_cols else ', '.join(actual_columns[:10])
                )
                
                # Include conversation history in the query
                enhanced_query = f"{query}\n\nData Sample:\n{json.dumps(data_sample, indent=2)}"
                if history_context:
                    enhanced_query = f"{query}\n\n{history_context}\n\nData Sample:\n{json.dumps(data_sample, indent=2)}"
                
                payload = {
                    "generation_model": "gpt-4o",
                    "max_tokens": 1500,
                    "temperature": 0.2,
                    "top_p": 0.01,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                    "tools": [],
                    "tools_choice": "none",
                    "system_prompt": formatted_system_prompt,
                    "custom_prompt": [
                        {"role": "user", "content": enhanced_query}
                    ],
                    "model_provider_name": "openai"
                }
                resp = invoke_llm(payload)
                if isinstance(resp, dict) and resp.get('error'):
                    raise Exception(resp['error'])
                if isinstance(resp, dict) and 'response' in resp:
                    try:
                        response_text = resp['response']
                        # Remove markdown code blocks if present
                        if '```json' in response_text:
                            response_text = response_text.split('```json')[1].split('```')[0].strip()
                        elif '```' in response_text:
                            response_text = response_text.split('```')[1].split('```')[0].strip()
                        result = json.loads(response_text)
                    except Exception as e:
                        logger.error(f"Failed to parse chart config JSON: {e}")
                        result = {'charts': [], 'tables': []}
                else:
                    result = resp
            
            elapsed = time.time() - start_time
            logger.info(f"✓ Chart config generated in {elapsed:.2f} seconds")
            logger.info("\n" + "=" * 30 + " Chart Configuration " + "=" * 30)
            logger.info("%s", json.dumps(result, indent=2))
            logger.info("" + "=" * 80 + "\n")
            
            # Validate that charts have valid column names
            if result and 'charts' in result:
                valid_charts = []
                for chart in result.get('charts', []):
                    x_col = chart.get('x_column')
                    y_col = chart.get('y_column')
                    group_by = chart.get('group_by')
                    
                    # Check if columns are None or not in data
                    # Map fuzzy column names to actual columns when possible
                    if x_col:
                        mapped_x = self._map_column_name(x_col, filtered_data.columns.tolist())
                        if mapped_x:
                            chart['x_column'] = mapped_x
                            x_col = mapped_x
                    if y_col:
                        mapped_y = self._map_column_name(y_col, filtered_data.columns.tolist())
                        if mapped_y:
                            chart['y_column'] = mapped_y
                            y_col = mapped_y
                    if group_by:
                        mapped_g = self._map_column_name(group_by, filtered_data.columns.tolist())
                        if mapped_g:
                            chart['group_by'] = mapped_g
                            group_by = mapped_g

                    if chart.get('type') == 'bar':
                        if x_col and x_col in filtered_data.columns:
                            valid_charts.append(chart)
                        else:
                            logger.warning(f"Skipping bar chart - invalid x_column: {x_col}")
                    elif chart.get('type') == 'pie':
                        if group_by and group_by in filtered_data.columns:
                            valid_charts.append(chart)
                        else:
                            logger.warning(f"Skipping pie chart - invalid group_by: {group_by}")
                    else:
                        valid_charts.append(chart)
                
                result['charts'] = valid_charts
                
                # If no valid charts, use fallback
                if not valid_charts:
                    logger.warning("No valid charts from LLM, using fallback")
                    return self._generate_fallback_charts(filtered_data)
            
            logger.info("=" * 80)
            return result
            
        except Exception as e:
            logger.error(f"✗ Error in chart generation: {str(e)}")
            logger.exception("Full traceback:")
            
            # Fallback: generate intelligent charts based on actual data columns
            return self._generate_fallback_charts(filtered_data)
    
    def _generate_fallback_charts(self, data: pd.DataFrame) -> Dict:
        """Generate fallback charts when LLM fails, using schema column names"""
        logger.info("Generating fallback charts with schema column names...")
        chart_cols = db_schema.get_common_chart_columns()
        charts = []
        
        # Chart 1: If we have Plant column, show distribution
        if 'Plant' in data.columns:
            charts.append({
                "type": "bar",
                "title": "Records by Plant",
                "x_column": "Plant",
                "y_column": "count",
                "agg_function": "count",
                "limit": 10
            })
        
        # Chart 2: If we have Material, show top materials
        if 'Material' in data.columns:
            charts.append({
                "type": "bar",
                "title": "Top Materials",
                "x_column": "Material",
                "y_column": "count",
                "agg_function": "count",
                "limit": 10
            })
        
        # Chart 3: If we have Auth flag, show pie chart
        if 'Auth Sell Flag Description' in data.columns:
            charts.append({
                "type": "pie",
                "title": "Authorization Status",
                "group_by": "Auth Sell Flag Description",
                "agg_function": "count"
            })
        
        # Always add a table with key columns from schema
        key_cols = [col for col in chart_cols['for_details'] if col in data.columns]
        if not key_cols:
            key_cols = data.columns.tolist()[:5]  # First 5 columns as last resort
        
        return {
            "charts": charts[:2],  # Max 2 charts
            "tables": [{
                "type": "table",
                "columns": key_cols,
                "title": "Data Details",
                "limit": 50
            }]
        }

# ===========================
# 3. DASHBOARD GENERATOR
# ===========================

class DashboardGenerator:
    """Generate dynamic dashboards based on intent with intelligent filtering"""
    
    def __init__(self, data: Dict[str, pd.DataFrame], classifier: 'IntentClassifier'):
        self.data = data
        self.classifier = classifier
        logger.info("DashboardGenerator initialized")
        logger.info(f"Available datasets: {list(data.keys())}")
        for name, df in data.items():
            logger.info(f"  - {name}: {len(df):,} records")
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply filters to dataframe using exact column names"""
        filtered_df = df.copy()
        
        if not filters:
            return filtered_df
        
        logger.info(f"Applying filters: {filters}")
        
        for col, value in filters.items():
            # Try to map fuzzy column names to actual dataframe columns (use classifier helper if available)
            mapped_col = None
            try:
                if hasattr(self, 'classifier') and self.classifier is not None:
                    mapped_col = self.classifier._map_column_name(col, filtered_df.columns.tolist())
                else:
                    mapped_col = None
            except Exception:
                mapped_col = None
            use_col = mapped_col if mapped_col else col

            if use_col in filtered_df.columns:
                # Handle both single values and lists
                if isinstance(value, list):
                    # For lists, use .isin() to match any value in the list
                    value_strings = [str(v) for v in value]
                    filtered_df = filtered_df[filtered_df[col].astype(str).isin(value_strings)]
                    logger.info(f"  ✓ Filtered by {col} in {value}, remaining rows: {len(filtered_df)}")
                else:
                    # For single values, use equality comparison
                    filtered_df = filtered_df[filtered_df[use_col].astype(str) == str(value)]
                    logger.info(f"  ✓ Filtered by {use_col} = {value}, remaining rows: {len(filtered_df)}")
            else:
                logger.warning(f"  ⚠️ Column '{col}' not found in dataframe (tried mapping to '{use_col}')")
        
        return filtered_df
    
    def _apply_cross_table_filters(self, filter_result: Dict) -> pd.DataFrame:
        """
        Apply filters with support for cross-table queries using foreign key relationships.
        
        Args:
            filter_result: Dict containing 'filters', 'requires_join', and 'join_on' keys
        
        Returns:
            Filtered DataFrame (joined if necessary)
        """
        filters = filter_result.get('filters', {})
        requires_join = filter_result.get('requires_join', False)
        join_on = filter_result.get('join_on', ['Plant', 'Material'])
        
        logger.info(f"Cross-table filtering: requires_join={requires_join}")
        
        # Start with Sales Order data
        sales_df = self.data['exception_report'].copy()
        
        if not requires_join:
            # Simple single-table filtering
            logger.info("Single-table query - filtering Sales Order table only")
            return self._apply_filters(sales_df, filters)
        
        # Cross-table query - need to join
        logger.info("Cross-table query detected - performing JOIN")
        location_df = self.data['location_sequence'].copy()
        
        # Separate filters by table
        sales_filters = {}
        location_filters = {}
        
        # Get column names from both tables
        sales_cols = set(sales_df.columns)
        # Note: Location table uses 'Plant(Location)' instead of 'Plant'
        location_cols = set(location_df.columns)
        
        for col, value in filters.items():
            if col in sales_cols:
                sales_filters[col] = value
                logger.info(f"  Filter for Sales Order table: {col} = {value}")
            elif col in location_cols:
                location_filters[col] = value
                logger.info(f"  Filter for Location table: {col} = {value}")
            else:
                logger.warning(f"  Column '{col}' not found in either table")
        
        # Apply filters to each table before joining
        if sales_filters:
            sales_df = self._apply_filters(sales_df, sales_filters)
            logger.info(f"Sales Order filtered: {len(sales_df):,} rows")
        
        if location_filters:
            location_df = self._apply_filters(location_df, location_filters)
            logger.info(f"Location Sequence filtered: {len(location_df):,} rows")
        
        # Perform the JOIN
        # Note: Location table uses 'Plant(Location)' instead of 'Plant'
        logger.info(f"Joining on: Sales.Plant = Location.Plant(Location) AND Sales.Material = Location.Material")
        
        merged_df = pd.merge(
            sales_df,
            location_df,
            left_on=['Plant', 'Material'],
            right_on=['Plant(Location)', 'Material'],
            how='inner',
            suffixes=('', '_location')
        )
        
        logger.info(f"✓ Join complete: {len(merged_df):,} rows")
        
        # Add a column to indicate this is joined data
        merged_df['_is_joined'] = True
        
        return merged_df
    
    def _determine_and_apply_filters(self, filters: Dict, user_query: str) -> pd.DataFrame:
        """
        Intelligently determine which table(s) to use based on filters and query keywords.
        Supports multi-table joins when needed (Sales Order + COF Inventory + COF Pricing).
        """
        logger.info("Determining which tables to use based on filters and query...")
        
        # Keywords that indicate COF pricing data is needed
        pricing_keywords = ['pricing', 'price', 'condition amount', 'condition currency', 'pricing unit', 'cost']
        needs_pricing = any(keyword in user_query.lower() for keyword in pricing_keywords)
        
        # Check which tables have the filter columns
        sales_df = self.data['exception_report']
        cof_inventory_df = self.data.get('cof_inventory')
        cof_pricing_df = self.data.get('cof_pricing')
        
        sales_cols = set(sales_df.columns)
        cof_inv_cols = set(cof_inventory_df.columns) if cof_inventory_df is not None else set()
        cof_price_cols = set(cof_pricing_df.columns) if cof_pricing_df is not None else set()
        
        # Categorize filters by table
        sales_filters = {}
        cof_inv_filters = {}
        cof_price_filters = {}
        
        for col, value in filters.items():
            if col in sales_cols:
                sales_filters[col] = value
            if col in cof_inv_cols:
                cof_inv_filters[col] = value
            if col in cof_price_cols:
                cof_price_filters[col] = value
        
        logger.info(f"Filter analysis:")
        logger.info(f"  - Sales Order filters: {sales_filters}")
        logger.info(f"  - COF Inventory filters: {cof_inv_filters}")
        logger.info(f"  - COF Pricing filters: {cof_price_filters}")
        logger.info(f"  - Query needs pricing data: {needs_pricing}")
        
        # Case 1: Query mentions pricing keywords - need to join with COF tables
        if needs_pricing and cof_inventory_df is not None and cof_pricing_df is not None:
            logger.info("🔗 Multi-table join required: Sales Order → COF Inventory → COF Pricing")
            
            # Step 1: Filter Sales Order table
            result_df = sales_df.copy()
            if sales_filters:
                result_df = self._apply_filters(result_df, sales_filters)
                logger.info(f"  Sales Order filtered: {len(result_df):,} rows")
            
            # Step 2: Join with COF Inventory (on Sold-to Name and Material)
            cof_inv_filtered = cof_inventory_df.copy()
            if cof_inv_filters:
                cof_inv_filtered = self._apply_filters(cof_inv_filtered, cof_inv_filters)
                logger.info(f"  COF Inventory filtered: {len(cof_inv_filtered):,} rows")
            
            # Join on Sold-to Name (if available in both tables)
            if 'Sold-to Name' in result_df.columns and 'Sold-to Name' in cof_inv_filtered.columns:
                logger.info("  Joining Sales Order ↔ COF Inventory on 'Sold-to Name'")
                result_df = pd.merge(
                    result_df,
                    cof_inv_filtered,
                    on='Sold-to Name',
                    how='inner',
                    suffixes=('', '_cof_inv')
                )
                logger.info(f"  After COF Inventory join: {len(result_df):,} rows")
            
            # Step 3: Join with COF Pricing (on Material)
            cof_price_filtered = cof_pricing_df.copy()
            if cof_price_filters:
                cof_price_filtered = self._apply_filters(cof_price_filtered, cof_price_filters)
                logger.info(f"  COF Pricing filtered: {len(cof_price_filtered):,} rows")
            
            # Determine material column name (might have suffix after previous join)
            material_col_left = 'Material' if 'Material' in result_df.columns else 'Material_cof_inv'
            material_col_right = 'Material'
            
            if material_col_left in result_df.columns and material_col_right in cof_price_filtered.columns:
                logger.info(f"  Joining Result ↔ COF Pricing on '{material_col_left}' = '{material_col_right}'")
                result_df = pd.merge(
                    result_df,
                    cof_price_filtered,
                    left_on=material_col_left,
                    right_on=material_col_right,
                    how='inner',
                    suffixes=('', '_cof_price')
                )
                logger.info(f"  After COF Pricing join: {len(result_df):,} rows")
            
            result_df['_is_joined'] = True
            return result_df
        
        # Case 2: Only COF data needed (no sales order filters)
        elif (cof_inv_filters or cof_price_filters) and not sales_filters:
            logger.info("COF-only query detected")
            # Return COF data (joined if needed)
            if cof_inv_filters and cof_price_filters:
                # Join COF Inventory + COF Pricing
                cof_inv_filtered = self._apply_filters(cof_inventory_df.copy(), cof_inv_filters)
                cof_price_filtered = self._apply_filters(cof_pricing_df.copy(), cof_price_filters)
                result_df = pd.merge(
                    cof_inv_filtered,
                    cof_price_filtered,
                    on='Material',
                    how='inner'
                )
                logger.info(f"COF tables joined: {len(result_df):,} rows")
                return result_df
            elif cof_inv_filters:
                return self._apply_filters(cof_inventory_df.copy(), cof_inv_filters)
            elif cof_price_filters:
                return self._apply_filters(cof_pricing_df.copy(), cof_price_filters)
        
        # Case 3: Default - Sales Order table only
        logger.info("Using Sales Order table only")
        return self._apply_filters(sales_df.copy(), filters)
    
    def _render_dynamic_chart(self, chart_config: Dict, data: pd.DataFrame):
        """Render a chart based on LLM-generated configuration"""
        chart_type = chart_config.get('type', 'table')
        title = chart_config.get('title', 'Chart')
        
        try:
            if chart_type == 'bar':
                x_col = chart_config.get('x_column')
                y_col = chart_config.get('y_column')
                agg_func = chart_config.get('agg_function', 'count')
                limit = chart_config.get('limit', 10)
                
                # Validate x_column exists (allow fuzzy mapping via classifier)
                if not x_col:
                    st.error(f"No x_column provided for bar chart")
                    return
                if x_col not in data.columns:
                    mapped = None
                    try:
                        mapped = self.classifier._map_column_name(x_col, data.columns.tolist())
                    except Exception:
                        mapped = None
                    if mapped:
                        x_col = mapped
                        chart_config['x_column'] = mapped
                    else:
                        st.error(f"Column '{chart_config.get('x_column')}' not found in data. Available columns: {', '.join(data.columns.tolist())}")
                        return
                
                # Create a copy and ensure x_column is string type to prevent number abbreviations
                data = data.copy()
                data[x_col] = data[x_col].astype(str)
                
                if y_col == 'count' or agg_func == 'count':
                    chart_data = data.groupby(x_col, dropna=False).size().reset_index(name='Count')
                    chart_data = chart_data.sort_values('Count', ascending=False).head(limit)
                    fig = px.bar(chart_data, x=x_col, y='Count', title=title)
                else:
                    # Validate y_column exists
                    if not y_col:
                        st.error(f"No y_column provided for bar chart")
                        return
                    if y_col not in data.columns:
                        mapped_y = None
                        try:
                            mapped_y = self.classifier._map_column_name(y_col, data.columns.tolist())
                        except Exception:
                            mapped_y = None
                        if mapped_y:
                            y_col = mapped_y
                            chart_config['y_column'] = mapped_y
                        else:
                            st.error(f"Column '{chart_config.get('y_column')}' not found in data. Available columns: {', '.join(data.columns.tolist())}")
                            return
                    chart_data = data.groupby(x_col, dropna=False)[y_col].agg(agg_func).reset_index()
                    chart_data = chart_data.sort_values(y_col, ascending=False).head(limit)
                    fig = px.bar(chart_data, x=x_col, y=y_col, title=title)
                
                # Force categorical axis and disable abbreviations for ALL columns
                fig.update_xaxes(type='category', tickmode='linear')
                fig.update_layout(xaxis={'categoryorder': 'total descending'})
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type in ['grouped_bar', 'stacked_bar']:
                x_col = chart_config.get('x_column')
                y_col = chart_config.get('y_column')
                color_by = chart_config.get('color_by')
                agg_func = chart_config.get('agg_function', 'sum')
                limit = chart_config.get('limit', 10)
                limit_groups = chart_config.get('limit_groups', 5)
                
                # Validate columns exist (allow fuzzy mapping)
                if not x_col:
                    st.error("No x_column provided for grouped/stacked bar chart")
                    return
                if x_col not in data.columns:
                    mapped_x = self.classifier._map_column_name(x_col, data.columns.tolist())
                    if mapped_x:
                        x_col = mapped_x
                        chart_config['x_column'] = mapped_x
                    else:
                        st.error(f"Column '{x_col}' not found in data")
                        return
                if not color_by:
                    st.error("No color_by provided for grouped/stacked bar chart")
                    return
                if color_by not in data.columns:
                    mapped_c = self.classifier._map_column_name(color_by, data.columns.tolist())
                    if mapped_c:
                        color_by = mapped_c
                        chart_config['color_by'] = mapped_c
                    else:
                        st.error(f"Column '{color_by}' not found in data")
                        return
                
                # Create a copy and convert to string
                data = data.copy()
                data[x_col] = data[x_col].astype(str)
                data[color_by] = data[color_by].astype(str)
                
                # Get top items for x-axis
                if y_col == 'count' or agg_func == 'count':
                    top_x = data[x_col].value_counts().head(limit).index.tolist()
                else:
                    if not y_col or y_col not in data.columns:
                        st.error(f"Column '{y_col}' not found in data")
                        return
                    top_x = data.groupby(x_col)[y_col].agg(agg_func).nlargest(limit).index.tolist()
                
                # Get top groups for color
                if y_col == 'count' or agg_func == 'count':
                    top_groups = data[color_by].value_counts().head(limit_groups).index.tolist()
                else:
                    top_groups = data.groupby(color_by)[y_col].agg(agg_func).nlargest(limit_groups).index.tolist()
                
                # Filter data
                filtered = data[data[x_col].isin(top_x) & data[color_by].isin(top_groups)]
                
                # Aggregate
                if y_col == 'count' or agg_func == 'count':
                    chart_data = filtered.groupby([x_col, color_by]).size().reset_index(name='Count')
                    y_col = 'Count'
                else:
                    chart_data = filtered.groupby([x_col, color_by])[y_col].agg(agg_func).reset_index()
                
                # Create chart
                barmode = 'group' if chart_type == 'grouped_bar' else 'stack'
                fig = px.bar(chart_data, x=x_col, y=y_col, color=color_by, 
                           title=title, barmode=barmode)
                fig.update_xaxes(type='category')
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == 'pie':
                group_col = chart_config.get('group_by')
                # Validate group_by column exists
                if not group_col:
                    st.error("No group_by provided for pie chart")
                    return
                if group_col not in data.columns:
                    mapped_g = self.classifier._map_column_name(group_col, data.columns.tolist())
                    if mapped_g:
                        group_col = mapped_g
                        chart_config['group_by'] = mapped_g
                    else:
                        st.error(f"Column '{group_col}' not found in data. Available columns: {', '.join(data.columns.tolist())}")
                        return
                
                # Convert to string to prevent abbreviations
                data = data.copy()
                data[group_col] = data[group_col].astype(str)
                
                pie_data = data.groupby(group_col, dropna=False).size().reset_index(name='Count')
                fig = px.pie(pie_data, names=group_col, values='Count', title=title)
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == 'table':
                columns = chart_config.get('columns', data.columns.tolist())
                limit = chart_config.get('limit', 50)
                display_cols = [col for col in columns if col in data.columns]
                st.subheader(title)
                st.dataframe(data[display_cols].head(limit), use_container_width=True)
                
        except Exception as e:
            logger.error(f"Error rendering chart: {str(e)}")
            st.error(f"Could not render {chart_type} chart: {str(e)}")
        
    def generate(self, filter_result: Dict, user_query: str, conversation_history: List[Dict] = None):
        """Generate dashboard using two-stage workflow: filters already extracted, now apply and visualize"""
        logger.info("=" * 80)
        logger.info("APPLYING FILTERS AND GENERATING DASHBOARD")
        logger.info("=" * 80)
        
        # Add conversation history context for chart generation
        history_context = ""
        if conversation_history:
            history_context = "\nConversation History:\n" + "\n".join([
                f"Previous Query {i+1}: {item['query']}\nApplied Filters: {item['filters']}"
                for i, item in enumerate(conversation_history[-3:])  # Last 3 interactions
            ])
            logger.info(f"Using conversation history for chart generation: {len(conversation_history)} previous interactions")
        
        filters = filter_result.get('filters', {})
        requires_join = filter_result.get('requires_join', False)
        
        logger.info(f"Filters to apply: {filters}")
        logger.info(f"Cross-table query: {requires_join}")
        
        start_time = time.time()
        
        # APPLY FILTERS TO DATA (Stage 1 complete) with multi-table support
        if requires_join:
            combined_data = self._apply_cross_table_filters(filter_result)
            logger.info(f"✓ Cross-table join performed")
        else:
            # Determine which table(s) to use based on filters and query
            combined_data = self._determine_and_apply_filters(filters, user_query)
        
        logger.info(f"Data filtered: {len(self.data['exception_report']):,} → {len(combined_data):,} records")
        
        if len(combined_data) == 0:
            st.warning("⚠️ No data found matching your filters.")
            st.info(f"**Filters Applied:** {json.dumps(filters, indent=2)}")
            return
        
        # Display header and filter summary
        st.header(f"🔍 Filtered Data Analysis")
        
        # Show cross-table query indicator
        if requires_join:
            st.success("✨ Cross-table query: Results from Sales Order + Location Sequence tables")
        
        if filters:
            filter_text = ", ".join([f"{k}={v}" for k, v in filters.items()])
            st.info(f"**Filters Applied:** {filter_text}")
            # Also show pretty JSON for clarity
            with st.expander("🧾 Filters (pretty)", expanded=False):
                st.code(json.dumps(filters, indent=2), language="json")
        else:
            st.info("**Showing all data (no filters applied)**")
        
        # Show basic metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Records", f"{len(combined_data):,}")
        col2.metric("Unique Materials", f"{combined_data['Material'].nunique() if 'Material' in combined_data.columns else 0:,}")
        col3.metric("Unique UPCs", f"{combined_data['UPC'].nunique() if 'UPC' in combined_data.columns else 0:,}")
        col4.metric("Unique Plants", f"{combined_data['Plant'].nunique() if 'Plant' in combined_data.columns else 0:,}")
        total_qty = combined_data['Order Quantity Sales Unit'].sum() if 'Order Quantity Sales Unit' in combined_data.columns else 0
        col5.metric("Total Quantity", f"{total_qty:,.0f}")
        
        # Handle custom aggregations from Stage 1
        aggregations = filter_result.get('aggregations', [])
        logger.info(f"📊 AGGREGATIONS RECEIVED: {aggregations}")
        logger.info(f"Number of aggregations: {len(aggregations)}")
        
        if aggregations and len(combined_data) > 0:
            st.markdown("---")
            st.subheader("📊 Requested Metrics")
            
            # Show aggregations in expandable section for debugging
            with st.expander("🔍 Aggregation Details (Debug)", expanded=False):
                st.json(aggregations)
            
            agg_cols = st.columns(min(len(aggregations), 4))
            
            for idx, agg in enumerate(aggregations):
                col_name = agg.get('column')
                func = agg.get('function', 'count')
                label = agg.get('label', f"{func.title()}")
                
                logger.info(f"Processing aggregation #{idx+1}: {label}, column={col_name}, function={func}")
                logger.info(f"  Full aggregation object: {agg}")
                
                try:
                    if func == 'count' and col_name is None:
                        # Total record count
                        value = len(combined_data)
                    elif func == 'count':
                        value = len(combined_data[col_name].dropna()) if col_name in combined_data.columns else 0
                    elif func == 'count_unique':
                        value = combined_data[col_name].nunique() if col_name in combined_data.columns else 0
                    elif func in ['sum', 'mean', 'max', 'min']:
                        # For numeric aggregations, check if column is numeric
                        # If not, use Order Quantity Sales Unit as the default numeric column
                        if col_name in combined_data.columns:
                            col_dtype = combined_data[col_name].dtype
                            if col_dtype in ['object', 'string']:
                                # Non-numeric column - user likely wants to aggregate BY this column
                                # Use Order Quantity Sales Unit as the target
                                logger.info(f"  Column '{col_name}' is non-numeric ({col_dtype}), interpreting as grouping column")
                                logger.info(f"  Using 'Order Quantity Sales Unit' as aggregation target")
                                if 'Order Quantity Sales Unit' in combined_data.columns:
                                    if func == 'sum':
                                        value = combined_data['Order Quantity Sales Unit'].sum()
                                    elif func == 'mean':
                                        value = combined_data['Order Quantity Sales Unit'].mean()
                                    elif func == 'max':
                                        value = combined_data['Order Quantity Sales Unit'].max()
                                    elif func == 'min':
                                        value = combined_data['Order Quantity Sales Unit'].min()
                                else:
                                    value = 0
                            else:
                                # Numeric column - aggregate directly
                                if func == 'sum':
                                    value = combined_data[col_name].sum()
                                elif func == 'mean':
                                    value = combined_data[col_name].mean()
                                elif func == 'max':
                                    value = combined_data[col_name].max()
                                elif func == 'min':
                                    value = combined_data[col_name].min()
                        else:
                            value = 0
                    else:
                        value = len(combined_data)
                    
                    # Format value
                    if isinstance(value, float):
                        formatted_value = f"{value:,.2f}"
                    else:
                        formatted_value = f"{value:,}"
                    
                    logger.info(f"  ✓ Displaying metric: {label} = {formatted_value}")
                    agg_cols[idx % 4].metric(label, formatted_value)
                except Exception as e:
                    logger.error(f"Error calculating aggregation {label}: {e}")
                    agg_cols[idx % 4].metric(label, "Error")
        
        # STAGE 2: GENERATE CHARTS BASED ON FILTERED DATA
        logger.info("=" * 80)
        logger.info("STAGE 2: CHART GENERATION FROM FILTERED DATA")
        logger.info("=" * 80)
        
        st.markdown("---")
        chart_config = self.classifier.generate_chart_config(user_query, combined_data, "analysis", conversation_history)
        
        # Render charts from Stage 2
        if chart_config.get('charts'):
            st.subheader("📊 Visualizations")
            cols = st.columns(2)
            for idx, chart in enumerate(chart_config['charts'][:4]):  # Limit to 4 charts
                with cols[idx % 2]:
                    self._render_dynamic_chart(chart, combined_data)
        
        # Render tables
        if chart_config.get('tables'):
            st.markdown("---")
            for table in chart_config['tables']:
                self._render_dynamic_chart(table, combined_data)
        
        # Download option
        st.markdown("---")
        csv = combined_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="filtered_data.csv",
            mime="text/csv"
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Dashboard generated in {elapsed:.2f} seconds")
        logger.info("=" * 80)
    
    def _create_auth_to_sell_dashboard(self, intent_result: Dict):
        """Dashboard for Authorized to Sell analysis"""
        st.header("📊 Authorized to Sell Analysis")
        
        auth_yes = self.data['auth_yes']
        auth_no = self.data['auth_no']
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_materials = len(auth_yes) + len(auth_no)
        auth_count = len(auth_yes)
        unauth_count = len(auth_no)
        auth_rate = (auth_count / total_materials * 100) if total_materials > 0 else 0
        
        col1.metric("Total Materials", f"{total_materials:,}")
        col2.metric("Authorized", f"{auth_count:,}", delta=f"{auth_rate:.1f}%")
        col3.metric("Not Authorized", f"{unauth_count:,}")
        col4.metric("Authorization Rate", f"{auth_rate:.1f}%")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie Chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Authorized', 'Not Authorized'],
                values=[auth_count, unauth_count],
                hole=0.4,
                marker_colors=['#00D45A', '#FF4B4B']
            )])
            fig_pie.update_layout(title="Authorization Status Distribution")
            st.plotly_chart(fig_pie, width='stretch')
        
        with col2:
            # Top Plants by Authorized Materials
            if 'Plant(Location)' in auth_yes.columns:
                plant_counts = auth_yes.groupby('Plant(Location)').size().reset_index(name='Count')
                plant_counts = plant_counts.sort_values('Count', ascending=False).head(10)
                
                fig_bar = px.bar(
                    plant_counts, 
                    x='Plant(Location)', 
                    y='Count',
                    title="Top 10 Plants - Authorized Materials",
                    color='Count',
                    color_continuous_scale='greens'
                )
                st.plotly_chart(fig_bar, width='stretch')
            else:
                st.info("Plant(Location) column not available. Showing first available column.")
                first_col = auth_yes.columns[0] if len(auth_yes.columns) > 0 else None
                if first_col:
                    st.write(f"Available columns: {', '.join(auth_yes.columns[:5])}")
        
        # Plant-wise breakdown
        st.subheader("Plant-wise Authorization Breakdown")
        
        if 'Plant(Location)' in auth_yes.columns and 'Plant(Location)' in auth_no.columns:
            auth_yes_by_plant = auth_yes.groupby('Plant(Location)').size().reset_index(name='Authorized')
            auth_no_by_plant = auth_no.groupby('Plant(Location)').size().reset_index(name='Not Authorized')
            
            plant_summary = pd.merge(
                auth_yes_by_plant, 
                auth_no_by_plant, 
                on='Plant(Location)', 
                how='outer'
            ).fillna(0)
            
            plant_summary['Total'] = plant_summary['Authorized'] + plant_summary['Not Authorized']
            plant_summary['Auth Rate %'] = (plant_summary['Authorized'] / plant_summary['Total'] * 100).round(2)
            plant_summary = plant_summary.sort_values('Total', ascending=False).head(20)
            
            st.dataframe(plant_summary, width='stretch')
        else:
            st.warning("Plant(Location) column not found in data. Available columns:")
            st.write("Auth Yes columns:", list(auth_yes.columns))
            st.write("Auth No columns:", list(auth_no.columns))
        
        # Material Search
        st.subheader("🔍 Search Material Authorization Status")
        material_search = st.text_input("Enter Material Code")
        
        if material_search:
            in_auth_yes = auth_yes[auth_yes['Material'].astype(str) == material_search]
            in_auth_no = auth_no[auth_no['Material'].astype(str) == material_search]
            
            if not in_auth_yes.empty:
                st.success(f"✅ Material {material_search} is AUTHORIZED to sell")
                st.dataframe(in_auth_yes)
            elif not in_auth_no.empty:
                st.error(f"❌ Material {material_search} is NOT AUTHORIZED to sell")
                st.dataframe(in_auth_no)
            else:
                st.warning(f"⚠️ Material {material_search} not found in database")
    
    def _create_exceptions_dashboard(self, intent_result: Dict):
        """Dashboard for Sales Order Exceptions"""
        st.header("⚠️ Sales Order Exceptions Analysis")
        
        exceptions = self.data['so_exceptions']
        
        # Display available columns for debugging
        st.write("Available columns:", list(exceptions.columns))
        
        # Clean data - check if column exists first
        if 'Sales Order Number' in exceptions.columns:
            exceptions = exceptions.dropna(subset=['Sales Order Number'])
        else:
            # Use first column or try alternate column names
            st.warning("'Sales Order Number' column not found. Using available data.")
            exceptions = exceptions.dropna(how='all')
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_exceptions = len(exceptions)
        unique_materials = exceptions['Material'].nunique() if 'Material' in exceptions.columns else 0
        unique_plants = exceptions['Plant'].nunique() if 'Plant' in exceptions.columns else 0
        total_quantity = exceptions['Order Quantity Sales Unit'].sum() if 'Order Quantity Sales Unit' in exceptions.columns else 0
        
        col1.metric("Total Exceptions", f"{total_exceptions:,}")
        col2.metric("Unique Materials", f"{unique_materials:,}")
        col3.metric("Affected Plants", f"{unique_plants:,}")
        col4.metric("Total Order Quantity", f"{total_quantity:,.0f}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Exceptions by Plant
            if 'Plant' in exceptions.columns:
                plant_exceptions = exceptions.groupby('Plant').size().reset_index(name='Count')
                plant_exceptions = plant_exceptions.sort_values('Count', ascending=False).head(10)
                
                fig_bar = px.bar(
                    plant_exceptions,
                    x='Plant',
                    y='Count',
                    title="Top 10 Plants with Exceptions",
                    color='Count',
                    color_continuous_scale='reds'
                )
                st.plotly_chart(fig_bar, width='stretch')
            else:
                st.info("Plant column not available in data")
        
        with col2:
            # Top Materials with Exceptions
            if 'Material' in exceptions.columns:
                material_exceptions = exceptions.groupby('Material').size().reset_index(name='Count')
                material_exceptions = material_exceptions.sort_values('Count', ascending=False).head(10)
                
                fig_bar2 = px.bar(
                    material_exceptions,
                    x='Material',
                    y='Count',
                    title="Top 10 Materials with Exceptions",
                    color='Count',
                    color_continuous_scale='oranges'
                )
                st.plotly_chart(fig_bar2, width='stretch')
            else:
                st.info("Material column not available in data")
        
        # Recent Exceptions
        st.subheader("Recent Sales Order Exceptions")
        # Show only available columns
        display_cols = [col for col in ['Sales Order Number', 'Requested Delivery Date', 
                                         'Sold-To Party', 'Material', 'Plant', 
                                         'Order Quantity Sales Unit', 'Sales Unit of Measure'] 
                        if col in exceptions.columns]
        if display_cols:
            exceptions_display = exceptions[display_cols].head(50)
        else:
            exceptions_display = exceptions.head(50)
        st.dataframe(exceptions_display, width='stretch')
        
        # Download option
        csv = exceptions.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Exception Report",
            data=csv,
            file_name="so_exceptions_export.csv",
            mime="text/csv"
        )
    
    def _create_plant_analysis_dashboard(self, intent_result: Dict):
        """Dashboard for Plant-wise Analysis"""
        st.header("🏭 Plant Analysis Dashboard")
        
        auth_yes = self.data['auth_yes']
        auth_no = self.data['auth_no']
        exceptions = self.data['so_exceptions'].dropna(subset=['Sales Order Number'])
        
        # Plant selector
        if 'Plant(Location)' in auth_yes.columns and 'Plant(Location)' in auth_no.columns:
            all_plants = sorted(list(set(auth_yes['Plant(Location)'].unique()) | 
                                    set(auth_no['Plant(Location)'].unique())))
        else:
            st.warning("Plant(Location) column not found. Available columns:")
            st.write("Auth Yes:", list(auth_yes.columns[:5]))
            st.write("Auth No:", list(auth_no.columns[:5]))
            all_plants = []
        selected_plants = st.multiselect("Select Plants", all_plants, default=all_plants[:5] if all_plants else [])
        
        if selected_plants:
            # Filter data
            auth_yes_filtered = auth_yes[auth_yes['Plant(Location)'].isin(selected_plants)] if 'Plant(Location)' in auth_yes.columns else auth_yes
            auth_no_filtered = auth_no[auth_no['Plant(Location)'].isin(selected_plants)] if 'Plant(Location)' in auth_no.columns else auth_no
            exceptions_filtered = exceptions[exceptions['Plant'].isin(selected_plants)] if 'Plant' in exceptions.columns else exceptions
            
            # Metrics per plant
            plant_metrics = []
            for plant in selected_plants:
                auth_count = len(auth_yes_filtered[auth_yes_filtered['Plant(Location)'] == plant])
                unauth_count = len(auth_no_filtered[auth_no_filtered['Plant(Location)'] == plant])
                exc_count = len(exceptions_filtered[exceptions_filtered['Plant'] == plant]) if 'Plant' in exceptions_filtered.columns else 0
                
                plant_metrics.append({
                    'Plant': plant,
                    'Authorized Materials': auth_count,
                    'Not Authorized': unauth_count,
                    'Exceptions': exc_count,
                    'Total Materials': auth_count + unauth_count,
                    'Auth Rate %': (auth_count / (auth_count + unauth_count) * 100) if (auth_count + unauth_count) > 0 else 0
                })
            
            df_metrics = pd.DataFrame(plant_metrics)
            st.dataframe(df_metrics, width='stretch')
            
            # Visualizations
            fig = px.bar(
                df_metrics,
                x='Plant',
                y=['Authorized Materials', 'Not Authorized', 'Exceptions'],
                title="Plant Comparison",
                barmode='group'
            )
            st.plotly_chart(fig, width='stretch')
    
    def _create_material_analysis_dashboard(self, intent_result: Dict):
        """Dashboard for Material-wise Analysis"""
        st.header("📦 Material Analysis Dashboard")
        st.info("Material-specific deep dive coming soon!")
    
    def _create_exceptions_dashboard_dynamic(self, exceptions: pd.DataFrame, user_query: str, filters: Dict, aggregations: List = None):
        """Dynamic exceptions dashboard with LLM-generated charts"""
        st.header("⚠️ Sales Order Exceptions Analysis")
        
        if aggregations is None:
            aggregations = []
        
        # If incoming filters were empty, attempt to extract filters from the user's query
        if not filters:
            columns_info = get_columns_info(self.data['exception_report'])
            llm_filters = extract_filters_from_llm(user_query, columns_info)
            if llm_filters:
                filters = llm_filters

        if filters:
            filter_text = ", ".join([f"{k}={v}" for k, v in filters.items()])
            st.info(f"**Filters Applied:** {filter_text}")

        # Apply filters locally (do not send whole dataframe to LLM)
        filtered_exceptions = apply_filters(self.data['exception_report'], filters)

        if len(filtered_exceptions) == 0:
            st.warning("⚠️ No exceptions found matching your filters.")
            return

        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Exceptions", f"{len(filtered_exceptions):,}")
        col2.metric("Unique Materials", f"{filtered_exceptions['Material'].nunique() if 'Material' in filtered_exceptions.columns else 0:,}")
        col3.metric("Unique UPCs", f"{filtered_exceptions['UPC'].nunique() if 'UPC' in filtered_exceptions.columns else 0:,}")
        col4.metric("Affected Plants", f"{filtered_exceptions['Plant'].nunique() if 'Plant' in filtered_exceptions.columns else 0:,}")
        total_qty = filtered_exceptions['Order Quantity Sales Unit'].sum() if 'Order Quantity Sales Unit' in filtered_exceptions.columns else 0
        col5.metric("Total Order Quantity", f"{total_qty:,.0f}")
        
        # Display custom aggregations if requested
        if aggregations and len(filtered_exceptions) > 0:
            st.markdown("---")
            st.subheader("📊 Requested Metrics")
            agg_cols = st.columns(min(len(aggregations), 4))
            
            for idx, agg in enumerate(aggregations[:4]):
                col_name = agg.get('column')
                func = agg.get('function', 'count')
                label = agg.get('label', f"{func.title()} of {col_name}")
                
                if col_name in filtered_exceptions.columns:
                    try:
                        if func == 'count':
                            value = len(filtered_exceptions)
                        elif func == 'count_unique':
                            value = filtered_exceptions[col_name].nunique()
                        elif func in ['sum', 'mean', 'max', 'min']:
                            # Check if column is numeric
                            col_dtype = filtered_exceptions[col_name].dtype
                            if col_dtype in ['object', 'string']:
                                # Non-numeric column - use Order Quantity Sales Unit as target
                                logger.info(f"  Column '{col_name}' is non-numeric, using 'Order Quantity Sales Unit' as target")
                                if 'Order Quantity Sales Unit' in filtered_exceptions.columns:
                                    if func == 'sum':
                                        value = filtered_exceptions['Order Quantity Sales Unit'].sum()
                                    elif func == 'mean':
                                        value = filtered_exceptions['Order Quantity Sales Unit'].mean()
                                    elif func == 'max':
                                        value = filtered_exceptions['Order Quantity Sales Unit'].max()
                                    elif func == 'min':
                                        value = filtered_exceptions['Order Quantity Sales Unit'].min()
                                else:
                                    value = 0
                            else:
                                # Numeric column - aggregate directly
                                if func == 'sum':
                                    value = filtered_exceptions[col_name].sum()
                                elif func == 'mean':
                                    value = filtered_exceptions[col_name].mean()
                                elif func == 'max':
                                    value = filtered_exceptions[col_name].max()
                                elif func == 'min':
                                    value = filtered_exceptions[col_name].min()
                        else:
                            value = len(filtered_exceptions)
                        
                        # Format based on value type
                        if isinstance(value, float):
                            formatted_value = f"{value:,.2f}"
                        else:
                            formatted_value = f"{value:,}"
                        
                        agg_cols[idx % 4].metric(label, formatted_value)
                    except Exception as e:
                        logger.error(f"Error calculating aggregation {label}: {e}")
                        agg_cols[idx % 4].metric(label, "Error")

        # Generate dynamic charts using a small data sample and column info
        st.markdown("---")
        sample_size = min(10, len(filtered_exceptions))
        data_sample = {
            'shape': filtered_exceptions.shape,
            'columns': filtered_exceptions.columns.tolist(),
            'sample_rows': filtered_exceptions.head(sample_size).to_dict('records'),
            'dtypes': filtered_exceptions.dtypes.astype(str).to_dict()
        }

        columns_info = get_columns_info(filtered_exceptions)
        chart_config = suggest_charts_from_llm(user_query, data_sample, columns_info)

        if chart_config.get('charts'):
            cols = st.columns(2)
            for idx, chart in enumerate(chart_config['charts'][:4]):
                with cols[idx % 2]:
                    self._render_dynamic_chart(chart, filtered_exceptions)

        # Tables
        if chart_config.get('tables'):
            st.markdown("---")
            for table in chart_config['tables']:
                self._render_dynamic_chart(table, filtered_exceptions)

        # Download filtered data
        csv = filtered_exceptions.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="filtered_exceptions.csv",
            mime="text/csv"
        )
    
    def _create_overview_dashboard(self, intent_result: Dict):
        """Overview Dashboard"""
        st.header("📈 SAP Data Overview")
        
        exception_report = self.data['exception_report']
        
        # Overall Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Records", f"{len(exception_report):,}")
        if 'Material' in exception_report.columns:
            col2.metric("Unique Materials", f"{exception_report['Material'].nunique():,}")
        if 'Plant' in exception_report.columns:
            col3.metric("Unique Plants", f"{exception_report['Plant'].nunique():,}")
        if 'Material Status Description' in exception_report.columns:
            active_count = len(exception_report[exception_report['Material Status Description'] == 'Active'])
            col4.metric("Active Materials", f"{active_count:,}")
        
        st.info("💡 Try asking specific questions like: 'Show me plant 7001 data' or 'I want active materials for plant 7001'")

# ===========================
# FOLLOW-UP QUESTION GENERATION
# ===========================

def generate_follow_up_questions(user_query: str, conversation_history: List[Dict], data_summary: Dict) -> List[str]:
    """Generate follow-up questions based on user query and conversation history"""
    logger.info("Generating follow-up questions...")
    
    # Create context from conversation history
    history_context = ""
    if conversation_history:
        history_context = "\nPrevious conversation:\n" + "\n".join([
            f"User: {item['query']}\nAssistant: Generated dashboard for {item['query']}"
            for item in conversation_history[-3:]  # Last 3 interactions
        ])
    
    # Create data summary context
    data_context = f"""
Available data summary:
- Total records: {data_summary.get('total_records', 'Unknown')}
- Tables: {', '.join(data_summary.get('tables', []))}
- Key columns: Plant, Material, Auth Sell Flag Description, Material Status, etc.
"""
    
    follow_up_prompt = f"""You are a SAP data analyst helping users explore their data. Based on the user's current query and conversation history, suggest 4 relevant follow-up questions that would help them gain deeper insights.

Current user query: "{user_query}"

{history_context}

{data_context}

Generate 4 follow-up questions that are:
1. Specific and actionable
2. Related to the current analysis
3. Help explore different aspects of the data
4. Use natural language

Return only a JSON array of 4 question strings, no additional text.

Example format:
["What are the top materials by quantity?", "Show me authorization issues by plant", "How many active materials do we have?", "What are the sales exceptions?"]"""

    try:
        payload = {
            "generation_model": "gpt-4o",
            "max_tokens": 300,
            "temperature": 0.7,
            "top_p": 0.9,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "tools": [],
            "tools_choice": "none",
            "system_prompt": "You are a helpful assistant that generates relevant follow-up questions for data analysis.",
            "custom_prompt": [
                {"role": "user", "content": follow_up_prompt}
            ],
            "model_provider_name": "openai"
        }
        
        resp = invoke_llm(payload)
        
        if isinstance(resp, dict) and 'response' in resp:
            response_text = resp['response']
            # Clean up response
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            questions = json.loads(response_text)
            if isinstance(questions, list) and len(questions) >= 4:
                logger.info(f"Generated {len(questions)} follow-up questions")
                return questions[:4]  # Return first 4
            else:
                logger.warning("Invalid follow-up questions format, using defaults")
                return get_default_follow_up_questions()
        else:
            logger.warning("No valid response for follow-up questions, using defaults")
            return get_default_follow_up_questions()
            
    except Exception as e:
        logger.error(f"Error generating follow-up questions: {str(e)}")
        return get_default_follow_up_questions()

def get_default_follow_up_questions() -> List[str]:
    """Return default follow-up questions"""
    return [
        "Show me authorized to sell details",
        "What are the sales exceptions?",
        "Give me plant-wise analysis",
        "Show overview of all data"
    ]

# ===========================
# 4. MAIN APPLICATION
# ===========================

def main():
    st.set_page_config(page_title="SAP Dashboard Agent", page_icon="📊", layout="wide")
    
    st.title("🤖 SAP Intelligent Dashboard Generator")
    st.markdown("Ask questions in natural language and get dynamic dashboards!")
    
    # Handle clear data flag before initializing session state
    if st.session_state.get('clear_all_data', False):
        # Clear all relevant session state
        keys_to_clear = [
            'conversation_history', 'interaction_count', 'follow_up_questions',
            'user_query_text_area', 'last_run_query', 'performance_metrics',
            'logs', 'api_calls', 'clear_all_data'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        # Set flag to skip normal initialization
        skip_initialization = True
    else:
        skip_initialization = False
    
    # Essential session state that must always exist (even after clearing)
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = {}
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    if 'api_calls' not in st.session_state:
        st.session_state.api_calls = []
    if 'user_query_text_area' not in st.session_state:
        st.session_state.user_query_text_area = ""

    # Sidebar - Developer Mode Toggle
    st.sidebar.header("⚙️ Settings")
    dev_mode = st.sidebar.checkbox("🔧 Developer Mode", value=False, help="Show API requests, console logs, and debug info")
    show_metrics = st.sidebar.checkbox("📊 Show Performance Metrics", value=False)
    
    # Session state initialization (skip if clearing data)
    if not skip_initialization:
        # Session state for logging
        if 'logs' not in st.session_state:
            st.session_state.logs = []
        if 'api_calls' not in st.session_state:
            st.session_state.api_calls = []
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = {}
        
        # Session state for conversation history and follow-up questions
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'interaction_count' not in st.session_state:
            st.session_state.interaction_count = 0
        if 'follow_up_questions' not in st.session_state:
            st.session_state.follow_up_questions = [
                "Show me authorized to sell details",  # Example Query 2 as requested
                "What are the sales exceptions?",
                "Give me plant-wise analysis",
                "Show overview of all data"
            ]
    
    # Initialize
    try:
        logger.info("🚀 Starting application initialization...")
        init_start = time.time()
        
        data = load_sap_data()
        classifier = IntentClassifier(data)
        dashboard_gen = DashboardGenerator(data, classifier)
        
        init_time = time.time() - init_start
        st.session_state.performance_metrics['initialization_time'] = init_time
        logger.info(f"✓ Application initialized in {init_time:.2f} seconds")
        
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        logger.error(f"Application initialization failed: {str(e)}")
        return
    
    # User Input
    st.sidebar.header("🎯 Ask Your Question")
    
    # Dynamic follow-up questions based on interaction count
    if st.session_state.get('interaction_count', 0) == 0:
        st.sidebar.markdown("### 💡 Example Queries:")
        display_questions = st.session_state.get('follow_up_questions', [
            "Show me authorized to sell details",
            "What are the sales exceptions?",
            "Give me plant-wise analysis",
            "Show overview of all data"
        ])  # Show all example queries initially
    else:
        st.sidebar.markdown("### 💡 Follow-up Questions:")
        display_questions = st.session_state.get('follow_up_questions', [])[:4]  # Show generated follow-up questions

    def set_example_query(example):
        st.session_state.user_query_text_area = example
        # Clear last run query so the same query can be run again
        if 'last_run_query' in st.session_state:
            st.session_state.last_run_query = None

    for question in display_questions:
        st.sidebar.button(question, key=question, on_click=set_example_query, args=(question,))

    user_query = st.sidebar.text_area(
        "Enter your query:",
        value=st.session_state.get('user_query_text_area', ''),
        placeholder="e.g., Show me authorized to sell details\nWhat are the sales exceptions?\nGive me plant-wise analysis",
        height=150,
        key="user_query_text_area"
    )
    
    # Generate button to trigger processing (aligned to right)
    col1, col2 = st.sidebar.columns([3, 3])
    with col1:
        st.write("")  # Empty space
    with col2:
        run_button = st.button("Generate", use_container_width=True)
    st.sidebar.caption("Enter a query then click 'Generate' (click an example to populate the query)")
    
    # Clear Data button
    if st.sidebar.button("🧹 Clear Data", use_container_width=True, help="Clear conversation history and reset to example queries"):
        # Set clear flag instead of modifying widget session state directly
        st.session_state.clear_all_data = True
        st.rerun()
    
    
    # Developer Mode Panel
    if dev_mode:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔧 Developer Tools")
        if st.sidebar.button("🗑️ Clear Logs"):
            st.session_state.logs = []
            st.session_state.api_calls = []
            st.rerun()
        
        if st.sidebar.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        if st.sidebar.button("💬 Clear Conversation History"):
            st.session_state.conversation_history = []
            st.session_state.interaction_count = 0
            st.session_state.follow_up_questions = [
                "Show me authorized to sell details",
                "What are the sales exceptions?",
                "Give me plant-wise analysis",
                "Show overview of all data"
            ]
            st.rerun()
    
    # Manage run state to avoid re-running on every rerender
    if 'last_run_query' not in st.session_state:
        st.session_state['last_run_query'] = None

    should_run = False
    if user_query and run_button:
        # run if user clicked Generate
        if st.session_state['last_run_query'] != user_query:
            should_run = True
            st.session_state['last_run_query'] = user_query

    # Process Query
    if should_run:
        # Show animated loading indicator with custom GIF (centered)
        loader_col1, loader_col2, loader_col3 = st.columns([2, 1, 2])
        with loader_col2:
            loader_placeholder = st.empty()
            # Get absolute path to Loader.gif (case-sensitive for Linux deployment)
            loader_path = Path(__file__).parent.parent.parent / "docs" / "Loader.gif"
            if loader_path.exists():
                with open(loader_path, "rb") as f:
                    loader_data = base64.b64encode(f.read()).decode()
                loader_placeholder.markdown(
                    f"""<div style="display: flex; justify-content: center;">
                    <img src="data:image/gif;base64,{loader_data}" width="100">
                    </div>""",
                    unsafe_allow_html=True
                )
            else:
                loader_placeholder.info("⏳ Loading...")
        
        try:
            query_start = time.time()
            
            # Update interaction count
            if 'interaction_count' not in st.session_state:
                st.session_state.interaction_count = 0
            st.session_state.interaction_count += 1
            
            # Prepare conversation history for LLM context
            if 'conversation_history' not in st.session_state:
                st.session_state.conversation_history = []
            conversation_history = st.session_state.conversation_history
            
            # Initialize other session state if needed
            if 'performance_metrics' not in st.session_state:
                st.session_state.performance_metrics = {}
            if 'api_calls' not in st.session_state:
                st.session_state.api_calls = []
            
            # STAGE 1: Extract Filters (with conversation history)
            filter_result = classifier.classify(user_query, conversation_history)
            classification_time = time.time() - query_start
            st.session_state.performance_metrics['filter_extraction_time'] = classification_time
            
            # Log API call
            st.session_state.api_calls.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'type': 'Stage 1: Filter Extraction',
                'query': user_query,
                'response': filter_result,
                'duration': f"{classification_time:.2f}s"
            })
            
            # Generate follow-up questions in parallel
            follow_up_start = time.time()
            data_summary = {
                'total_records': sum(len(df) for df in data.values()),
                'tables': list(data.keys())
            }
            new_follow_up_questions = generate_follow_up_questions(user_query, conversation_history, data_summary)
            follow_up_time = time.time() - follow_up_start
            st.session_state.performance_metrics['follow_up_generation_time'] = follow_up_time
            
            # Update follow-up questions for next interaction
            st.session_state.follow_up_questions = new_follow_up_questions
            
            # Log follow-up generation
            st.session_state.api_calls.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'type': 'Follow-up Questions Generation',
                'query': user_query,
                'response': new_follow_up_questions,
                'duration': f"{follow_up_time:.2f}s"
            })
            
            # STAGE 2: Generate Dashboard with Charts (with conversation history)
            dashboard_start = time.time()
            dashboard_gen.generate(filter_result, user_query, conversation_history)
            dashboard_time = time.time() - dashboard_start
            st.session_state.performance_metrics['dashboard_generation_time'] = dashboard_time
            
            total_time = time.time() - query_start
            st.session_state.performance_metrics['total_time'] = total_time
            
            # Update conversation history
            st.session_state.conversation_history.append({
                'query': user_query,
                'filters': filter_result.get('filters', {}),
                'timestamp': datetime.now().isoformat()
            })
            
            # Clear loader
            loader_placeholder.empty()
        
        except Exception as e:
            loader_placeholder.empty()
            raise
        
        # Show filter extraction results after generation
        with st.expander("🔍 Stage 1: Filter Extraction Results"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**User Query:**")
                st.code(user_query, language="text")
            with col2:
                st.markdown("**Extracted Filters:**")
                st.json(filter_result.get('filters', {}))
        
        st.success(f"✅ Dashboard generated in {total_time:.2f}s")
        
        # Show newly generated follow-up questions immediately
        if st.session_state.interaction_count > 0:
            st.markdown("---")
            st.markdown("### 💡 Suggested Follow-up Questions:")
            cols = st.columns(2)
            for i, question in enumerate(new_follow_up_questions[:4]):
                col_idx = i % 2
                with cols[col_idx]:
                    st.button(question, key=f"followup_{i}", on_click=set_example_query, args=(question,), use_container_width=True)
        
    else:
        # Do not auto-generate. Prompt user to click Generate.
        if user_query and not run_button:
            st.info("Click 'Generate' to build the dashboard.")
        else:
            st.info("No query submitted. Enter a query on the left and click 'Generate' to start.")
    
    # Developer Console (if enabled)
    if dev_mode:
        st.markdown("---")
        
        # Create tabs for different debug views
        debug_tabs = st.tabs(["🔍 API Requests", "📊 Performance", "📝 Console Logs", "💾 Data Info"])
        
        # Tab 1: API Requests
        with debug_tabs[0]:
            st.subheader("🔍 API Request History")
            if st.session_state.api_calls:
                for i, call in enumerate(reversed(st.session_state.api_calls[-10:])):  # Last 10 calls
                    with st.expander(f"Call #{len(st.session_state.api_calls)-i} - {call['timestamp']} ({call['duration']})"):
                        st.markdown(f"**Type:** {call['type']}")
                        st.markdown("**Request:**")
                        st.code(call['query'], language="text")
                        st.markdown("**Response:**")
                        st.json(call['response'])
            else:
                st.info("No API calls yet. Make a query to see details.")
        
        # Tab 2: Performance Metrics
        with debug_tabs[1]:
            st.subheader("📊 Performance Metrics")
            if st.session_state.performance_metrics:
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                with metrics_col1:
                    if 'initialization_time' in st.session_state.performance_metrics:
                        st.metric("Init Time", f"{st.session_state.performance_metrics['initialization_time']:.2f}s")
                
                with metrics_col2:
                    if 'filter_extraction_time' in st.session_state.performance_metrics:
                        st.metric("Stage 1: Filter Extraction", f"{st.session_state.performance_metrics['filter_extraction_time']:.2f}s")
                
                with metrics_col3:
                    if 'dashboard_generation_time' in st.session_state.performance_metrics:
                        st.metric("Dashboard Gen", f"{st.session_state.performance_metrics['dashboard_generation_time']:.2f}s")
                
                if 'total_time' in st.session_state.performance_metrics:
                    st.markdown("---")
                    st.markdown(f"**Total Query Processing Time:** `{st.session_state.performance_metrics['total_time']:.2f}s`")
                
                # Performance breakdown chart
                if 'filter_extraction_time' in st.session_state.performance_metrics:
                    perf_data = pd.DataFrame([
                        {'Stage': 'Stage 1: Filter Extraction', 'Time (s)': st.session_state.performance_metrics.get('filter_extraction_time', 0)},
                        {'Stage': 'Stage 2: Chart Generation + Render', 'Time (s)': st.session_state.performance_metrics.get('dashboard_generation_time', 0)}
                    ])
                    fig = px.bar(perf_data, x='Stage', y='Time (s)', title="Performance Breakdown")
                    st.plotly_chart(fig, width='stretch')
            else:
                st.info("No performance metrics yet.")
        
        # Tab 3: Console Logs
        with debug_tabs[2]:
            st.subheader("📝 Console Logs")
            st.markdown("Check your terminal for detailed logs or enable log streaming below:")
            
            log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
            
            st.code("""
Terminal logs are displayed in your console where you ran:
python3 -m streamlit run sap_dashboard_agent.py

Look for:
- Data loading progress
- LLM request/response details
- Performance timings
- Error traces
            """, language="bash")
        
        # Tab 4: Data Info
        with debug_tabs[3]:
            st.subheader("💾 Loaded Data Information")
            
            # Build data_info dynamically based on what's actually loaded
            data_info = {}
            
            # Check which datasets are available and add them to data_info
            if 'exception_report' in data and data['exception_report'] is not None:
                data_info['Sales Order Exception Report'] = {
                    'records': len(data['exception_report']),
                    'columns': list(data['exception_report'].columns),
                    'memory': f"{data['exception_report'].memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
                }
            
            if 'location_sequence' in data and data['location_sequence'] is not None:
                data_info['A1P Location Sequence'] = {
                    'records': len(data['location_sequence']),
                    'columns': list(data['location_sequence'].columns),
                    'memory': f"{data['location_sequence'].memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
                }
            
            for dataset_name, info in data_info.items():
                with st.expander(f"📁 {dataset_name}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Records", f"{info['records']:,}")
                    with col2:
                        st.metric("Memory Usage", info['memory'])
                    st.markdown("**Columns:**")
                    st.write(", ".join(info['columns']))
            
            # Total summary
            total_records = sum(len(df) for df in data.values())
            st.markdown("---")
            st.markdown(f"**Total Records Loaded:** `{total_records:,}`")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if dev_mode:
        st.sidebar.markdown("🔧 **Developer Mode Active**")

if __name__ == "__main__":
    main()
