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
import json
import logging
import time
from datetime import datetime
import sys
from exception_handler import (
    load_exception_csv,
    get_columns_info,
    extract_filters_from_llm,
    apply_filters,
    suggest_charts_from_llm,
)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ===========================
# 1. DATA LOADING FUNCTIONS
# ===========================

@st.cache_data
def load_sap_data():
    """Load all SAP data files"""
    logger.info("=" * 80)
    logger.info("LOADING SAP DATA FILES")
    logger.info("=" * 80)
    start_time = time.time()
    
    def load_csv_with_encoding(filepath):
        """Try to load CSV with different encodings and handle parsing issues"""
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                # Try with error_bad_lines=False (pandas <2.0) or on_bad_lines='skip' (pandas >=2.0)
                return pd.read_csv(filepath, encoding=encoding, on_bad_lines='skip', engine='python')
            except (UnicodeDecodeError, ValueError):
                try:
                    # Fallback: try without on_bad_lines parameter
                    return pd.read_csv(filepath, encoding=encoding, engine='python')
                except:
                    continue
        # If all fail, try with error handling
        try:
            return pd.read_csv(filepath, encoding='latin-1', on_bad_lines='skip', engine='python')
        except:
            return pd.read_csv(filepath, encoding='latin-1', engine='python')
    
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
        
        logger.info("Loading: Sales Order Exception report 13 and 14 Nov 2025.csv (via helper)")
        exception_report = load_exception_csv('Sales Order Exception report 13 and 14 Nov 2025.csv')
        logger.info(f"✓ Loaded {len(exception_report):,} records from Exception Report")

        # Use exception_report as the primary exceptions data
        so_exceptions = exception_report
        
        elapsed = time.time() - start_time
        logger.info(f"✓ All data loaded successfully in {elapsed:.2f} seconds")
        logger.info("=" * 80)
        
        data = {
            'exception_report': exception_report
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
        
        # Get column information from datasets
        self.columns_info = self._get_columns_info()
        
        # Define system prompts as strings for API calls
        self.intent_system_prompt = """You are a SAP data analyst. Classify the user's query and determine what dashboard to create.
            
Available Data Sources and Their Columns:
{columns_info}

🔑 CRITICAL BUSINESS TERMINOLOGY MAPPING:

**Understanding "Failed" / "Error" / "Issue":**
- "failed due to Authorized to Sell" = Auth Sell Flag Description = "No"
- "Authorized to Sell issue" = Auth Sell Flag Description = "No"
- "Authorized to Sell error" = Auth Sell Flag Description = "No"
- "auth to sell problem" = Auth Sell Flag Description = "No"
- "not authorized" = Auth Sell Flag Description = "No"
- ALL records in this dataset are exceptions/failures, filter by Auth Sell Flag = "No" to find authorization-related failures

**Column Name Mappings:**
- "sales order" = column "Sales Order Number"
- "location" / "plant" = column "Plant"
- "customer" / "customer account" / "sold-to" = column "Sold-To Party"
- "material description" = column "Material Descrption" (note: spelling in data)
- "material" / "material code" = column "Material"
- "quantity" / "case qty" / "order quantity" = column "Order Quantity Sales Unit"
- "status" / "material status" = column "Material Status Description"
- "auth flag" / "authorization" = column "Auth Sell Flag Description"

**Counting Logic:**
- "how many sales orders" = count_unique on "Sales Order Number"
- "how many materials" = count_unique on "Material" or "Material Descrption"
- "total records" = count on any column (returns row count)
- "total quantity" = sum on "Order Quantity Sales Unit"

Return JSON with:
- intent: "exceptions", "plant_analysis", "material_analysis", "customer_analysis", or "overview"
- data_sources: list of required data sources (currently only ["exception_report"] is available)
- filters: object with EXACT column names as keys and filter values. Examples:
  * {{"Plant": "7001"}} - for specific plant/location
  * {{"Material": "000000000300005846"}} - for specific material
  * {{"Material Status Description": "Active"}} - for material status
  * {{"Auth Sell Flag Description": "No"}} - for authorization failures
  * {{"Auth Sell Flag Description": "Yes"}} - for authorized items
  * {{"Sold-To Party": "0001234567"}} - for customer account
- show_material_details: true/false - whether to show detailed material information
- aggregations: list of aggregation requests. Examples:
  * {{"column": "Sales Order Number", "function": "count_unique", "label": "Total Sales Orders"}}
  * {{"column": "Order Quantity Sales Unit", "function": "sum", "label": "Total Case Qty"}}
  * {{"column": "Material Descrption", "function": "count_unique", "label": "Unique Materials"}}

Example queries with EXACT expected output:

1. "Provide total number of sales orders failed due to Authorized to Sell issue"
   → {{"intent": "exceptions", "filters": {{"Auth Sell Flag Description": "No"}}, "aggregations": [{{"column": "Sales Order Number", "function": "count_unique", "label": "Failed Sales Orders"}}]}}

2. "Provide total records due to Authorized to sell error"
   → {{"intent": "exceptions", "filters": {{"Auth Sell Flag Description": "No"}}, "aggregations": [{{"column": "Sales Order Number", "function": "count", "label": "Total Records"}}]}}

3. "How many failure material description for plant 1007 with case qty"
   → {{"intent": "plant_analysis", "filters": {{"Plant": "1007"}}, "aggregations": [{{"column": "Material Descrption", "function": "count_unique", "label": "Unique Materials"}}, {{"column": "Order Quantity Sales Unit", "function": "sum", "label": "Total Case Qty"}}]}}

4. "How many failure materials for customer 0001234567 with case qty"
   → {{"intent": "customer_analysis", "filters": {{"Sold-To Party": "0001234567"}}, "aggregations": [{{"column": "Material", "function": "count_unique", "label": "Unique Materials"}}, {{"column": "Order Quantity Sales Unit", "function": "sum", "label": "Total Case Qty"}}]}}

5. "Show me plant 7001 data"
   → {{"intent": "plant_analysis", "filters": {{"Plant": "7001"}}}}

6. "I want plant 7001 details for auth flag active"
   → {{"intent": "plant_analysis", "filters": {{"Plant": "7001", "Auth Sell Flag Description": "Yes"}}, "show_material_details": true}}

7. "What are the sales exceptions?"
   → {{"intent": "exceptions", "filters": {{}}}}

REMEMBER: When user mentions "failed due to auth" or "authorized to sell issue/error", ALWAYS add filter {{"Auth Sell Flag Description": "No"}}
"""
        
        self.chart_system_prompt = """You are a data visualization expert. Based on the user's query and the filtered data sample, suggest appropriate charts and tables.

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
"""
        
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
    
    def _get_columns_info(self) -> str:
        """Get comprehensive column information from all datasets with data types and sample values"""
        info = []
        
        # Focus on exception_report since that's what's loaded
        if 'exception_report' in self.data and len(self.data['exception_report']) > 0:
            df = self.data['exception_report']
            info.append("Exception Report Data (exception_report):")
            info.append(f"   Total Records: {len(df):,}")
            info.append("")
            info.append("   COLUMNS WITH DETAILS:")
            info.append("   " + "="*70)
            
            # Show detailed info for ALL columns
            for col in df.columns:
                dtype = str(df[col].dtype)
                unique_count = df[col].nunique()
                null_count = df[col].isnull().sum()
                
                info.append(f"   • {col}")
                info.append(f"     - Type: {dtype}")
                info.append(f"     - Unique Values: {unique_count:,}")
                info.append(f"     - Null Count: {null_count:,}")
                
                # Show sample values for each column
                if unique_count <= 10:
                    # Show all unique values if ≤10
                    unique_vals = df[col].dropna().unique().tolist()
                    info.append(f"     - All Values: {unique_vals}")
                else:
                    # Show top 5 sample values
                    sample_vals = df[col].dropna().unique()[:5].tolist()
                    info.append(f"     - Sample Values (first 5): {sample_vals}")
                
                # For numeric columns, show min/max/mean
                if df[col].dtype in ['int64', 'float64']:
                    try:
                        min_val = df[col].min()
                        max_val = df[col].max()
                        mean_val = df[col].mean()
                        info.append(f"     - Range: {min_val} to {max_val}, Mean: {mean_val:.2f}")
                    except:
                        pass
                
                info.append("")
        
        return "\n".join(info)
        
    def classify(self, query: str) -> Dict:
        """Classify user intent with column-aware filtering"""
        logger.info("=" * 80)
        logger.info("STAGE 1: INTENT CLASSIFICATION WITH FILTER EXTRACTION")
        logger.info("=" * 80)
        logger.info(f"User Query: '{query}'")
        
        start_time = time.time()
        
        try:
            logger.info("Building LLM prompt with column information...")
            if self.llm:
                chain = self.intent_prompt | self.llm | JsonOutputParser()
                logger.info("Sending request to local LLM (ChatOllama)...")
                result = chain.invoke({
                    "query": query,
                    "columns_info": self.columns_info
                })
            else:
                # Fallback to Pepsico internal LLM API
                logger.info("Using Pepsico LLM API as fallback for classification...")
                formatted_system_prompt = self.intent_system_prompt.format(columns_info=self.columns_info)
                logger.info(f"System prompt length: {len(formatted_system_prompt)} chars")
                logger.info(f"First 500 chars of system prompt: {formatted_system_prompt[:500]}")
                
                payload = {
                    "generation_model": "gpt-4o",
                    "max_tokens": 1000,
                    "temperature": 0.0,
                    "top_p": 0.01,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                    "tools": [],
                    "tools_choice": "none",
                    "system_prompt": formatted_system_prompt,
                    "custom_prompt": [
                        {"role": "user", "content": query}
                    ],
                    "model_provider_name": "openai"
                }
                logger.info(f"Sending query: '{query}'")
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
                        result = {'intent': 'overview', 'filters': {}}
                else:
                    result = resp
            
            elapsed = time.time() - start_time
            logger.info(f"✓ LLM Response received in {elapsed:.2f} seconds")
            logger.info(f"Intent Classification Result:")
            logger.info(json.dumps(result, indent=2))
            logger.info("=" * 80)
            
            return result
            
        except Exception as e:
            logger.error(f"✗ Error in LLM classification: {str(e)}")
            logger.exception("Full traceback:")
            
            # Fallback classification
            logger.warning("Using fallback classification...")
            query_lower = query.lower()
            
            if "authorized" in query_lower or "auth" in query_lower:
                fallback_result = {
                    "intent": "authorized_to_sell",
                    "data_sources": ["auth_yes", "auth_no"],
                    "visualizations": ["pie", "bar", "table", "metric"],
                    "filters": {},
                    "metrics": ["total_materials", "authorized_count", "unauthorized_count", "authorization_rate"]
                }
            elif "exception" in query_lower or "error" in query_lower:
                fallback_result = {
                    "intent": "exceptions",
                    "data_sources": ["so_exceptions", "exception_report"],
                    "visualizations": ["bar", "table", "metric", "line"],
                    "filters": {},
                    "metrics": ["total_exceptions", "exception_by_plant", "top_materials"]
                }
            elif "plant" in query_lower:
                fallback_result = {
                    "intent": "plant_analysis",
                    "data_sources": ["auth_yes", "auth_no", "so_exceptions"],
                    "visualizations": ["bar", "table", "metric"],
                    "filters": {},
                    "metrics": ["materials_by_plant", "authorization_rate_by_plant"]
                }
            else:
                fallback_result = {
                    "intent": "overview",
                    "data_sources": ["auth_yes", "auth_no", "so_exceptions"],
                    "visualizations": ["metric", "pie", "bar"],
                    "filters": {},
                    "metrics": ["total_overview"]
                }
            
            logger.info(f"Fallback result: {json.dumps(fallback_result, indent=2)}")
            logger.info("=" * 80)
            return fallback_result
    
    def generate_chart_config(self, query: str, filtered_data: pd.DataFrame, intent: str) -> Dict:
        """Generate chart configuration based on filtered data - Stage 2"""
        logger.info("=" * 80)
        logger.info("STAGE 2: DYNAMIC CHART GENERATION")
        logger.info("=" * 80)
        logger.info(f"Filtered data shape: {filtered_data.shape}")
        
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
                payload = {
                    "generation_model": "gpt-4o",
                    "max_tokens": 1500,
                    "temperature": 0.2,
                    "top_p": 0.01,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                    "tools": [],
                    "tools_choice": "none",
                    "system_prompt": self.chart_system_prompt.format(data_sample=json.dumps(data_sample, indent=2)),
                    "custom_prompt": [
                        {"role": "user", "content": f"{query}\n\nData Sample:\n{json.dumps(data_sample, indent=2)}"}
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
            logger.info(f"Chart Configuration:")
            logger.info(json.dumps(result, indent=2))
            logger.info("=" * 80)
            
            return result
            
        except Exception as e:
            logger.error(f"✗ Error in chart generation: {str(e)}")
            logger.exception("Full traceback:")
            
            # Fallback: basic charts
            return {
                "charts": [
                    {"type": "table", "title": "Data Overview", "limit": 50}
                ],
                "tables": []
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
            if col in filtered_df.columns:
                # Convert value to string for comparison
                filtered_df = filtered_df[filtered_df[col].astype(str) == str(value)]
                logger.info(f"  ✓ Filtered by {col} = {value}, remaining rows: {len(filtered_df)}")
            else:
                logger.warning(f"  ⚠️ Column '{col}' not found in dataframe")
        
        return filtered_df
    
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
                
                # Validate x_column exists
                if not x_col or x_col not in data.columns:
                    st.error(f"Column '{x_col}' not found in data. Available columns: {', '.join(data.columns.tolist())}")
                    return
                
                if y_col == 'count' or agg_func == 'count':
                    chart_data = data.groupby(x_col, dropna=False).size().reset_index(name='Count')
                    chart_data = chart_data.sort_values('Count', ascending=False).head(limit)
                    fig = px.bar(chart_data, x=x_col, y='Count', title=title)
                else:
                    # Validate y_column exists
                    if not y_col or y_col not in data.columns:
                        st.error(f"Column '{y_col}' not found in data. Available columns: {', '.join(data.columns.tolist())}")
                        return
                    chart_data = data.groupby(x_col, dropna=False)[y_col].agg(agg_func).reset_index()
                    chart_data = chart_data.sort_values(y_col, ascending=False).head(limit)
                    fig = px.bar(chart_data, x=x_col, y=y_col, title=title)
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == 'pie':
                group_col = chart_config.get('group_by')
                # Validate group_by column exists
                if not group_col or group_col not in data.columns:
                    st.error(f"Column '{group_col}' not found in data. Available columns: {', '.join(data.columns.tolist())}")
                    return
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
        
    def generate(self, intent_result: Dict, user_query: str):
        """Generate dashboard based on classified intent with intelligent filtering and dynamic charts"""
        logger.info("=" * 80)
        logger.info("GENERATING DYNAMIC DASHBOARD")
        logger.info("=" * 80)
        
        intent = intent_result.get('intent', 'overview')
        filters = intent_result.get('filters', {})
        show_material_details = intent_result.get('show_material_details', False)
        aggregations = intent_result.get('aggregations', [])
        
        logger.info(f"Dashboard Type: {intent}")
        logger.info(f"Filters: {filters}")
        logger.info(f"Show Material Details: {show_material_details}")
        logger.info(f"Aggregations: {aggregations}")
        
        start_time = time.time()
        
        # Determine which data to use based on intent
        if intent in ['plant_analysis', 'material_analysis', 'customer_analysis']:
            # Use exception_report data (auth data not loaded)
            combined_data = self._apply_filters(self.data['exception_report'], filters)
            
            # Show filter summary
            st.header(f"🔍 Filtered Data Analysis")
            if filters:
                filter_text = ", ".join([f"{k}={v}" for k, v in filters.items()])
                st.info(f"**Filters Applied:** {filter_text}")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records", f"{len(combined_data):,}")
            
            # Show unique counts for key columns
            if 'Material' in combined_data.columns:
                col2.metric("Unique Materials", f"{combined_data['Material'].nunique():,}")
            if 'Plant' in combined_data.columns:
                col3.metric("Unique Plants", f"{combined_data['Plant'].nunique():,}")
            if 'Material Status Description' in combined_data.columns:
                active_count = len(combined_data[combined_data['Material Status Description'] == 'Active'])
                col4.metric("Active Materials", f"{active_count:,}")
            
            # Display custom aggregations if requested
            if aggregations and len(combined_data) > 0:
                st.markdown("---")
                st.subheader("📊 Requested Metrics")
                agg_cols = st.columns(min(len(aggregations), 4))
                
                for idx, agg in enumerate(aggregations[:4]):
                    col_name = agg.get('column')
                    func = agg.get('function', 'count')
                    label = agg.get('label', f"{func.title()} of {col_name}")
                    
                    logger.info(f"Processing aggregation: {label}, column={col_name}, function={func}")
                    logger.info(f"  Column exists: {col_name in combined_data.columns}")
                    logger.info(f"  Data shape: {combined_data.shape}")
                    
                    if col_name in combined_data.columns:
                        try:
                            if func == 'count':
                                value = len(combined_data)
                            elif func == 'count_unique':
                                value = combined_data[col_name].nunique()
                                logger.info(f"  ✓ Unique values calculated: {value}")
                            elif func == 'sum':
                                value = combined_data[col_name].sum()
                            elif func == 'mean':
                                value = combined_data[col_name].mean()
                            elif func == 'max':
                                value = combined_data[col_name].max()
                            elif func == 'min':
                                value = combined_data[col_name].min()
                            else:
                                value = len(combined_data)
                            
                            # Format based on value type
                            if isinstance(value, float):
                                formatted_value = f"{value:,.2f}"
                            else:
                                formatted_value = f"{value:,}"
                            
                            logger.info(f"  ✓ Displaying metric: {label} = {formatted_value}")
                            agg_cols[idx % 4].metric(label, formatted_value)
                        except Exception as e:
                            logger.error(f"Error calculating aggregation {label}: {e}")
                            agg_cols[idx % 4].metric(label, "Error")
            
            # Generate dynamic charts using Stage 2 LLM
            if len(combined_data) > 0:
                st.markdown("---")
                chart_config = self.classifier.generate_chart_config(user_query, combined_data, intent)
                
                # Render charts
                if chart_config.get('charts'):
                    cols = st.columns(2)
                    for idx, chart in enumerate(chart_config['charts'][:4]):  # Limit to 4 charts
                        with cols[idx % 2]:
                            self._render_dynamic_chart(chart, combined_data)
                
                # Render tables
                if chart_config.get('tables') or show_material_details:
                    st.markdown("---")
                    if show_material_details:
                        st.subheader("📋 Material Details")
                        detail_cols = [col for col in ['Material', 'Material Descrption', 'Plant', 
                                                       'Material Status Description', 'Sales Order Number'] 
                                      if col in combined_data.columns]
                        st.dataframe(combined_data[detail_cols].head(100), use_container_width=True)
                    else:
                        for table in chart_config.get('tables', []):
                            self._render_dynamic_chart(table, combined_data)
            else:
                st.warning("⚠️ No data found matching your filters.")
                
        elif intent == 'exceptions':
            # Pass raw exception report into the dynamic handler; it will extract/apply filters safely
            self._create_exceptions_dashboard_dynamic(self.data['exception_report'], user_query, filters, aggregations)
        else:
            self._create_overview_dashboard(intent_result)
        
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
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Exceptions", f"{len(filtered_exceptions):,}")
        col2.metric("Unique Materials", f"{filtered_exceptions['Material'].nunique() if 'Material' in filtered_exceptions.columns else 0:,}")
        col3.metric("Affected Plants", f"{filtered_exceptions['Plant'].nunique() if 'Plant' in filtered_exceptions.columns else 0:,}")
        total_qty = filtered_exceptions['Order Quantity Sales Unit'].sum() if 'Order Quantity Sales Unit' in filtered_exceptions.columns else 0
        col4.metric("Total Order Quantity", f"{total_qty:,.0f}")
        
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
                        elif func == 'sum':
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
# 4. MAIN APPLICATION
# ===========================

def main():
    st.set_page_config(page_title="SAP Dashboard Agent", page_icon="📊", layout="wide")
    
    st.title("🤖 SAP Intelligent Dashboard Generator")
    st.markdown("Ask questions in natural language and get dynamic dashboards!")
    
    # Sidebar - Developer Mode Toggle
    st.sidebar.header("⚙️ Settings")
    dev_mode = st.sidebar.checkbox("🔧 Developer Mode", value=False, help="Show API requests, console logs, and debug info")
    show_metrics = st.sidebar.checkbox("📊 Show Performance Metrics", value=False)
    
    # Session state for logging
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    if 'api_calls' not in st.session_state:
        st.session_state.api_calls = []
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = {}
    
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
    user_query = st.sidebar.text_area(
        "Enter your query:",
        placeholder="e.g., Show me authorized to sell details\nWhat are the sales exceptions?\nGive me plant-wise analysis",
        height=100
    )
    
    # Example queries
    st.sidebar.markdown("### 💡 Example Queries:")
    example_queries = [
        "Show me authorized to sell details",
        "What are the sales exceptions?",
        "Give me plant-wise analysis",
        "Show overview of all data"
    ]
    
    for example in example_queries:
        if st.sidebar.button(example, key=example):
            user_query = example
    
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
    
    # Process Query
    if user_query:
        query_start = time.time()
        
        with st.spinner("🤔 Understanding your query..."):
            intent_result = classifier.classify(user_query)
            classification_time = time.time() - query_start
            st.session_state.performance_metrics['classification_time'] = classification_time
            
            # Log API call
            st.session_state.api_calls.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'type': 'LLM Classification',
                'query': user_query,
                'response': intent_result,
                'duration': f"{classification_time:.2f}s"
            })
        
        # Show intent analysis
        with st.expander("🔍 Query Analysis"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**User Query:**")
                st.code(user_query, language="text")
            with col2:
                st.markdown("**Classified Intent:**")
                st.json(intent_result)
        
        # Generate Dashboard
        dashboard_start = time.time()
        with st.spinner("📊 Generating dashboard..."):
            dashboard_gen.generate(intent_result, user_query)
            dashboard_time = time.time() - dashboard_start
            st.session_state.performance_metrics['dashboard_generation_time'] = dashboard_time
        
        total_time = time.time() - query_start
        st.session_state.performance_metrics['total_time'] = total_time
        
    else:
        # Default overview
        dashboard_gen.generate({"intent": "overview"}, "Show overview of all data")
    
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
                    if 'classification_time' in st.session_state.performance_metrics:
                        st.metric("LLM Classification", f"{st.session_state.performance_metrics['classification_time']:.2f}s")
                
                with metrics_col3:
                    if 'dashboard_generation_time' in st.session_state.performance_metrics:
                        st.metric("Dashboard Gen", f"{st.session_state.performance_metrics['dashboard_generation_time']:.2f}s")
                
                if 'total_time' in st.session_state.performance_metrics:
                    st.markdown("---")
                    st.markdown(f"**Total Query Processing Time:** `{st.session_state.performance_metrics['total_time']:.2f}s`")
                
                # Performance breakdown chart
                if 'classification_time' in st.session_state.performance_metrics:
                    perf_data = pd.DataFrame([
                        {'Stage': 'LLM Classification', 'Time (s)': st.session_state.performance_metrics.get('classification_time', 0)},
                        {'Stage': 'Dashboard Generation', 'Time (s)': st.session_state.performance_metrics.get('dashboard_generation_time', 0)}
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
            
            data_info = {
                'Authorized Yes': {
                    'records': len(data['auth_yes']),
                    'columns': list(data['auth_yes'].columns),
                    'memory': f"{data['auth_yes'].memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
                },
                'Authorized No': {
                    'records': len(data['auth_no']),
                    'columns': list(data['auth_no'].columns),
                    'memory': f"{data['auth_no'].memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
                },
                'SO Exceptions': {
                    'records': len(data['so_exceptions']),
                    'columns': list(data['so_exceptions'].columns),
                    'memory': f"{data['so_exceptions'].memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
                },
                'Exception Report': {
                    'records': len(data['exception_report']),
                    'columns': list(data['exception_report'].columns),
                    'memory': f"{data['exception_report'].memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
                }
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
