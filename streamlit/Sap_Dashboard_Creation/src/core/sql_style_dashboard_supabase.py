"""
SQL-Style Filtering Dashboard with Supabase Integration
========================================================
LLM generates filter parameters from natural language
Supabase executes the query on cloud database
Only small results sent back to LLM for natural language response

Key Concept:
- User: "Show me Plant 1006 materials not authorized"
- LLM: Generates {"plant": "1006", "auth_flag": "N"}
- Supabase: Filters data and returns count
- LLM: Converts count to natural language answer
"""

import streamlit as st
import pandas as pd
import json
from typing import Dict, List, Any
import plotly.express as px
import plotly.graph_objects as go
from langchain_ollama import ChatOllama
from datetime import datetime
from supabase import create_client, Client
from database_schema import (
    get_table_list, 
    get_table_description, 
    generate_schema_prompt,
    get_columns_for_table,
    TABLES
)

# Initialize LLM with faster settings
llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    num_predict=500,  # Increased for longer prompts
    timeout=30  # Add timeout to prevent hanging
)

# Supabase Configuration
# Get these from: Supabase Dashboard → Settings → API
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")  # Your Supabase project URL
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")  # Your Supabase anon key


class SupabaseConnector:
    """Manages Supabase database connection"""
    
    def __init__(self, url: str, key: str):
        self.supabase: Client = create_client(url, key)
    
    def test_connection(self) -> bool:
        """Test if Supabase connection is working"""
        try:
            # Try to query authorized_materials table
            result = self.supabase.table('authorized_materials').select('*').limit(1).execute()
            return True
        except Exception as e:
            st.error(f"Supabase connection failed: {e}")
            return False


class SQLStyleFilterGenerator:
    """Converts natural language to filter specifications"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_filter_spec(self, user_query: str, selected_table: str) -> Dict[str, Any]:
        """
        Convert natural language query to filter parameters
        Uses the selected table from user
        """
        
        # Get schema for selected table
        schema_info = generate_schema_prompt(selected_table)
        
        # Get available columns for better prompt
        available_cols = get_columns_for_table(selected_table)
        filterable_cols = [col for col in available_cols if col in ['plant', 'material', 'material_description', 'sales_order_number', 'customer_hierarchy', 'sold_to_name', 'exception_type', 'order_create_date', 'auth_flag']]
        
        # Format columns as a readable list with descriptions
        cols_list = []
        for col in filterable_cols:
            from database_schema import COLUMNS
            col_desc = COLUMNS.get(col, {}).get('description', col)
            cols_list.append(f"  - {col}: {col_desc}")
        
        cols_str = "\n".join(cols_list)
        
        # Generate dynamic examples based on table
        example_filter_col = "plant" if "plant" in filterable_cols else filterable_cols[0] if filterable_cols else "id"
        
        prompt = f"""Convert user question to JSON filter for the selected table.

DATABASE SCHEMA:
Table: {selected_table}
Available Filter Columns:
{cols_str}

USER QUESTION: "{user_query}"

REQUIRED JSON FORMAT:
{{"table": "{selected_table}", "filters": {{"column_name": "value"}}, "aggregations": ["count"], "visualizations": ["bar", "table"]}}

RULES:
1. filters is OBJECT (not array): {{"plant": "1006", "material_description": "CEMENT"}}
2. Only use columns listed above in Available Filter Columns
3. For text search (material_description, sold_to_name, etc.), extract keywords from question
4. For exact match (plant, material, sales_order_number), use exact values
5. aggregations: ["count", "group_by_plant", "group_by_material"]
6. visualizations: ["metric", "bar", "pie", "table", "line"]

EXAMPLES:
Q: "Show Plant 1006 details"
A: {{"table": "{selected_table}", "filters": {{"plant": "1006"}}, "aggregations": ["count", "group_by_material"], "visualizations": ["metric", "bar", "table"]}}

Q: "Find materials with CEMENT in Plant 1007"
A: {{"table": "{selected_table}", "filters": {{"plant": "1007", "material_description": "CEMENT"}}, "aggregations": ["count", "group_by_material"], "visualizations": ["bar", "table"]}}

Q: "Show sales orders for customer 7-ELEVEN"
A: {{"table": "{selected_table}", "filters": {{"sold_to_name": "7-ELEVEN"}}, "aggregations": ["count", "group_by_plant"], "visualizations": ["bar", "table"]}}

Q: "All records" or "Show everything"
A: {{"table": "{selected_table}", "filters": {{}}, "aggregations": ["count", "group_by_plant"], "visualizations": ["bar", "pie"]}}

Return ONLY JSON (no markdown, no explanation):"""
        
        try:
            # Log the prompt being sent to LLM
            print("\n" + "="*80)
            print("🤖 PROMPT SENT TO LLM:")
            print("="*80)
            print(prompt)
            print("="*80 + "\n")
            
            response = self.llm.invoke(prompt)
            
            # Extract JSON from response
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Log the raw LLM response
            print("\n" + "="*80)
            print("🤖 RAW LLM RESPONSE:")
            print("="*80)
            print(response_text)
            print("="*80 + "\n")
            
            # Find JSON in response - improved extraction with brace matching
            json_str = response_text.strip()
            
            # Remove markdown code blocks if present
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0].strip()
            elif '```' in json_str:
                json_str = json_str.split('```')[1].split('```')[0].strip()
            
            # Find the first { and match braces to find the correct closing }
            start_idx = json_str.find('{')
            
            if start_idx != -1:
                # Count braces to find matching closing brace
                brace_count = 0
                end_idx = start_idx
                
                for i in range(start_idx, len(json_str)):
                    if json_str[i] == '{':
                        brace_count += 1
                    elif json_str[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i
                            break
                
                json_str = json_str[start_idx:end_idx+1]
            
            print(f"\n🔍 Extracted JSON string:\n{json_str}\n")
            
            filter_spec = json.loads(json_str)
            
            # Log the parsed filter spec
            print("\n" + "="*80)
            print("📋 PARSED FILTER SPEC:")
            print("="*80)
            print(json.dumps(filter_spec, indent=2))
            print("="*80 + "\n")
            
            # Force the selected table (in case LLM ignores it)
            filter_spec['table'] = selected_table
            
            # Auto-add aggregations if only count is present
            aggregations = filter_spec.get('aggregations', ['count'])
            if aggregations == ['count'] or len(aggregations) == 1:
                filters = filter_spec.get('filters', {})
                
                # If filtering by plant, add group_by_material
                if 'plant' in filters and filters['plant']:
                    if 'group_by_material' not in aggregations:
                        aggregations.append('group_by_material')
                # If no plant filter, add group_by_plant
                else:
                    if 'group_by_plant' not in aggregations:
                        aggregations.append('group_by_plant')
                
                filter_spec['aggregations'] = aggregations
                print(f"\n✨ Auto-enhanced aggregations: {aggregations}\n")
            
            # Auto-add visualizations if not present
            if 'visualizations' not in filter_spec or not filter_spec['visualizations']:
                # Default visualizations based on aggregations
                default_viz = ['metric']
                if 'group_by_plant' in aggregations or 'group_by_material' in aggregations:
                    default_viz.extend(['bar', 'table'])
                if 'calculate_rate' in aggregations:
                    default_viz = ['bar', 'table']
                
                filter_spec['visualizations'] = default_viz
                print(f"\n✨ Auto-added visualizations: {default_viz}\n")
            
            return filter_spec
        
        except Exception as e:
            print(f"\n❌ ERROR in generate_filter_spec: {e}\n")
            # Return default filter on error
            return {
                "table": selected_table,
                "filters": {},
                "aggregations": ["count"],
                "intent": "aggregation"
            }


class SupabaseQueryExecutor:
    """Executes filter specifications on Supabase database"""
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
    
    def execute_query(self, filter_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute filter specification on Supabase
        Returns only aggregated results (small payload)
        """
        
        # Select table
        table_name = filter_spec.get('table', 'all')
        filters = filter_spec.get('filters', {})
        aggregations = filter_spec.get('aggregations', ['count'])
        
        results = {}
        
        try:
            # Fetch data based on table selection
            if table_name == 'all':
                # Query both authorized and not_authorized tables
                # Remove auth_flag from filters since we're querying both
                filters_no_auth = {k: v for k, v in filters.items() if k != 'auth_flag'}
                
                auth_data = self._fetch_data('authorized_materials', filters_no_auth)
                not_auth_data = self._fetch_data('not_authorized_materials', filters_no_auth)
                all_data = auth_data + not_auth_data
                df = pd.DataFrame(all_data)
                
                # Add auth_flag column if not present
                if 'auth_flag' not in df.columns and not df.empty:
                    # Mark which records came from which table
                    for i, record in enumerate(all_data):
                        if i < len(auth_data):
                            all_data[i]['auth_flag'] = 'Y'
                        else:
                            all_data[i]['auth_flag'] = 'N'
                    df = pd.DataFrame(all_data)
            else:
                data = self._fetch_data(table_name, filters)
                df = pd.DataFrame(data)
            
            # Execute aggregations
            if 'count' in aggregations:
                results['total_count'] = len(df)
            
            if 'group_by_plant' in aggregations and not df.empty:
                plant_counts = df['plant'].value_counts().to_dict()
                limit_val = filter_spec.get('limit', 10)
                plant_counts = dict(list(plant_counts.items())[:limit_val])
                results['by_plant'] = plant_counts
            
            if 'group_by_material' in aggregations and not df.empty:
                material_counts = df['material'].value_counts().to_dict()
                limit_val = filter_spec.get('limit', 10)
                material_counts = dict(list(material_counts.items())[:limit_val])
                results['by_material'] = material_counts
            
            if 'calculate_rate' in aggregations:
                # Calculate authorization rate per plant
                auth_df = pd.DataFrame(self._fetch_data('authorized_materials', {}))
                not_auth_df = pd.DataFrame(self._fetch_data('not_authorized_materials', {}))
                
                if not auth_df.empty and not not_auth_df.empty:
                    auth_counts = auth_df.groupby('plant').size()
                    total_counts = pd.concat([
                        auth_df['plant'],
                        not_auth_df['plant']
                    ]).value_counts()
                    
                    auth_rates = {}
                    for plant in total_counts.index:
                        auth = auth_counts.get(plant, 0)
                        total = total_counts.get(plant, 1)
                        auth_rates[plant] = round((auth / total * 100), 2)
                    
                    # Apply limit and sort
                    limit_val = filter_spec.get('limit', 10)
                    sort_order = filter_spec.get('sort', 'auth_rate ASC')
                    
                    if 'DESC' in sort_order:
                        auth_rates = dict(sorted(auth_rates.items(), key=lambda x: x[1], reverse=True)[:limit_val])
                    else:
                        auth_rates = dict(sorted(auth_rates.items(), key=lambda x: x[1])[:limit_val])
                    
                    results['authorization_rates'] = auth_rates
            
            # Add metadata
            results['metadata'] = {
                'query_executed_at': datetime.now().isoformat(),
                'filters_applied': filters,
                'records_matched': len(df),
                'table': table_name
            }
            
        except Exception as e:
            results['error'] = str(e)
            results['metadata'] = {
                'query_executed_at': datetime.now().isoformat(),
                'filters_applied': filters,
                'records_matched': 0,
                'table': table_name
            }
        
        return results
    
    def _fetch_data(self, table: str, filters: Dict[str, Any], limit: int = None) -> List[Dict]:
        """Fetch data from Supabase with filters"""
        
        # Build SQL query string for logging
        sql_query = f"SELECT * FROM {table}"
        where_clauses = []
        
        # Log the query being built
        print("\n" + "="*80)
        print(f"🗄️  BUILDING SUPABASE QUERY:")
        print("="*80)
        print(f"Table: {table}")
        print(f"Filters: {json.dumps(filters, indent=2)}")
        print("="*80 + "\n")
        
        # Start query - use count to get accurate total first
        query = self.supabase.table(table).select('*', count='exact')
        
        # Apply filters dynamically for all columns
        if 'plant' in filters and filters['plant']:
            print(f"  ✓ Adding filter: plant = '{filters['plant']}'")
            where_clauses.append(f"plant = '{filters['plant']}'")
            query = query.eq('plant', str(filters['plant']))
        
        if 'material' in filters and filters['material']:
            print(f"  ✓ Adding filter: material LIKE '%{filters['material']}%'")
            where_clauses.append(f"material ILIKE '%{filters['material']}%'")
            query = query.ilike('material', f"%{filters['material']}%")
        
        if 'material_description' in filters and filters['material_description']:
            print(f"  ✓ Adding filter: material_description LIKE '%{filters['material_description']}%'")
            where_clauses.append(f"material_description ILIKE '%{filters['material_description']}%'")
            query = query.ilike('material_description', f"%{filters['material_description']}%")
        
        if 'auth_flag' in filters and filters['auth_flag']:
            print(f"  ✓ Adding filter: auth_flag = '{filters['auth_flag']}'")
            where_clauses.append(f"auth_flag = '{filters['auth_flag']}'")
            query = query.eq('auth_flag', filters['auth_flag'])
        
        if 'customer_hierarchy' in filters and filters['customer_hierarchy']:
            print(f"  ✓ Adding filter: customer_hierarchy LIKE '%{filters['customer_hierarchy']}%'")
            where_clauses.append(f"customer_hierarchy ILIKE '%{filters['customer_hierarchy']}%'")
            query = query.ilike('customer_hierarchy', f"%{filters['customer_hierarchy']}%")
        
        if 'sold_to_name' in filters and filters['sold_to_name']:
            print(f"  ✓ Adding filter: sold_to_name LIKE '%{filters['sold_to_name']}%'")
            where_clauses.append(f"sold_to_name ILIKE '%{filters['sold_to_name']}%'")
            query = query.ilike('sold_to_name', f"%{filters['sold_to_name']}%")
        
        if 'sales_order_number' in filters and filters['sales_order_number']:
            print(f"  ✓ Adding filter: sales_order_number = '{filters['sales_order_number']}'")
            where_clauses.append(f"sales_order_number = '{filters['sales_order_number']}'")
            query = query.eq('sales_order_number', str(filters['sales_order_number']))
        
        if 'exception_type' in filters and filters['exception_type']:
            print(f"  ✓ Adding filter: exception_type = '{filters['exception_type']}'")
            where_clauses.append(f"exception_type = '{filters['exception_type']}'")
            query = query.eq('exception_type', filters['exception_type'])
        
        if 'order_create_date' in filters and filters['order_create_date']:
            print(f"  ✓ Adding filter: order_create_date = '{filters['order_create_date']}'")
            where_clauses.append(f"order_create_date = '{filters['order_create_date']}'")
            query = query.eq('order_create_date', filters['order_create_date'])
        
        # Build complete SQL query string
        if where_clauses:
            sql_query += "\nWHERE " + " AND ".join(where_clauses)
        
        # Apply limit - default to 5000 to avoid Supabase pagination issues
        if limit:
            sql_query += f"\nLIMIT {limit};"
            print(f"  ✓ Adding limit: {limit} rows")
            query = query.limit(limit)
        else:
            # Set a high default limit to fetch all data (Supabase default is only 1000)
            default_limit = 5000
            sql_query += f"\nLIMIT {default_limit};  -- Default limit to avoid pagination"
            print(f"  ⚠️  Using default limit: {default_limit} rows (to avoid pagination issues)")
            query = query.limit(default_limit)
        
        # Print the SQL query
        print("\n" + "="*80)
        print("📝 EQUIVALENT SQL QUERY:")
        print("="*80)
        print(sql_query)
        print("="*80 + "\n")
        
        # Execute query
        print(f"\n  🚀 Executing query on Supabase...")
        result = query.execute()
        
        # Show actual vs returned count
        actual_count = result.count if hasattr(result, 'count') else len(result.data)
        returned_count = len(result.data)
        
        if actual_count and actual_count > returned_count:
            print(f"  ⚠️  Total matching rows: {actual_count}, but only fetched: {returned_count}")
            print(f"  💡 Increase limit or implement pagination to fetch all rows")
        else:
            print(f"  ✅ Query returned {returned_count} rows")
        
        print("="*80 + "\n")
        
        return result.data


class NaturalLanguageResponder:
    """Converts query results back to natural language"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_response(self, user_query: str, results: Dict[str, Any]) -> str:
        """
        Convert structured results to natural language answer
        """
        
        prompt = f"""Answer concisely:

Question: "{user_query}"
Data: {json.dumps(results, indent=2)}

Give a clear 2-3 sentence answer with numbers:"""
        
        try:
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            return answer.strip()
        
        except Exception as e:
            return f"Error: {e}"


def create_visualization(results: Dict[str, Any], filter_spec: Dict[str, Any]):
    """Create visualizations based on LLM suggestions and query results"""
    
    visualizations = filter_spec.get('visualizations', ['metric', 'bar'])
    
    # Check what data we have
    has_plant_data = 'by_plant' in results and len(results['by_plant']) > 0
    has_material_data = 'by_material' in results and len(results['by_material']) > 0
    has_rate_data = 'authorization_rates' in results and len(results['authorization_rates']) > 0
    
    # Count chart visualizations (not table/metric)
    chart_viz = [v for v in visualizations if v in ['bar', 'pie', 'line']]
    
    # Create columns for multiple charts
    use_columns = len(chart_viz) > 1
    if use_columns:
        col1, col2 = st.columns(2)
        current_col = 0
    
    # Render each visualization type
    for viz_type in visualizations:
        
        if viz_type == "bar":
            # Use column if we have multiple charts
            container = col1 if use_columns and current_col % 2 == 0 else (col2 if use_columns else st.container())
            
            with container:
                if has_plant_data:
                    plants = list(results['by_plant'].keys())[:15]
                    counts = list(results['by_plant'].values())[:15]
                    
                    fig = px.bar(
                        x=plants,
                        y=counts,
                        labels={'x': 'Plant', 'y': 'Count'},
                        title='📊 Distribution by Plant',
                        color=counts,
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    if use_columns:
                        current_col += 1
                
                elif has_material_data:
                    materials = list(results['by_material'].keys())[:15]
                    counts = list(results['by_material'].values())[:15]
                    
                    fig = px.bar(
                        y=materials,
                        x=counts,
                        orientation='h',
                        labels={'x': 'Count', 'y': 'Material'},
                        title='📦 Top Materials',
                        color=counts,
                        color_continuous_scale='Greens'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    if use_columns:
                        current_col += 1
                
                elif has_rate_data:
                    plants = list(results['authorization_rates'].keys())
                    rates = list(results['authorization_rates'].values())
                    
                    fig = px.bar(
                        x=plants,
                        y=rates,
                        labels={'x': 'Plant', 'y': 'Authorization Rate (%)'},
                        title='📊 Authorization Rates by Plant',
                        color=rates,
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    if use_columns:
                        current_col += 1
        
        elif viz_type == "pie":
            container = col1 if use_columns and current_col % 2 == 0 else (col2 if use_columns else st.container())
            
            with container:
                if has_plant_data:
                    plants = list(results['by_plant'].keys())[:10]
                    counts = list(results['by_plant'].values())[:10]
                    
                    fig = px.pie(
                        values=counts,
                        names=plants,
                        title='🥧 Distribution by Plant'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    if use_columns:
                        current_col += 1
                
                elif has_material_data:
                    materials = list(results['by_material'].keys())[:8]
                    counts = list(results['by_material'].values())[:8]
                    
                    fig = px.pie(
                        values=counts,
                        names=materials,
                        title='🥧 Material Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    if use_columns:
                        current_col += 1
        
        elif viz_type == "line":
            container = col1 if use_columns and current_col % 2 == 0 else (col2 if use_columns else st.container())
            
            with container:
                if has_rate_data:
                    plants = list(results['authorization_rates'].keys())
                    rates = list(results['authorization_rates'].values())
                    
                    fig = px.line(
                        x=plants,
                        y=rates,
                        labels={'x': 'Plant', 'y': 'Authorization Rate (%)'},
                        title='📈 Authorization Rate Trend',
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    if use_columns:
                        current_col += 1
        
        elif viz_type == "table":
            st.markdown("#### 📋 Detailed Data Table")
            
            if has_plant_data:
                df_display = pd.DataFrame([
                    {'Plant': k, 'Count': v} 
                    for k, v in results['by_plant'].items()
                ])
                df_display = df_display.sort_values('Count', ascending=False)
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            elif has_material_data:
                df_display = pd.DataFrame([
                    {'Material': k, 'Count': v} 
                    for k, v in results['by_material'].items()
                ])
                df_display = df_display.sort_values('Count', ascending=False)
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            elif has_rate_data:
                df_display = pd.DataFrame([
                    {'Plant': k, 'Authorization Rate (%)': v} 
                    for k, v in results['authorization_rates'].items()
                ])
                df_display = df_display.sort_values('Authorization Rate (%)', ascending=False)
                st.dataframe(df_display, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(
        page_title="SAP Data Dashboard (Supabase)",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 SAP Data Dashboard with Supabase")
    st.markdown("""
    **Intelligent Dashboard Generator** - Ask questions in natural language and get instant visual insights!
    """)
    
    # Supabase credentials input (if not in secrets)
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.warning("⚠️ Supabase credentials not configured!")
        
        with st.expander("📋 Configure Supabase Connection", expanded=True):
            st.markdown("""
            **Get your Supabase credentials:**
            1. Go to your Supabase project dashboard
            2. Click **Settings** → **API**
            3. Copy the **Project URL** and **anon public** key
            """)
            
            supabase_url = st.text_input("Supabase URL", placeholder="https://xxxxx.supabase.co")
            supabase_key = st.text_input("Supabase Anon Key", type="password", placeholder="eyJhbGc...")
            
            if st.button("Connect to Supabase"):
                if supabase_url and supabase_key:
                    st.session_state.supabase_url = supabase_url
                    st.session_state.supabase_key = supabase_key
                    st.success("✅ Credentials saved for this session!")
                    st.rerun()
                else:
                    st.error("Please provide both URL and Key")
        
        return
    
    # Use credentials from secrets or session state
    url = SUPABASE_URL or st.session_state.get('supabase_url')
    key = SUPABASE_KEY or st.session_state.get('supabase_key')
    
    # Initialize Supabase connection
    try:
        connector = SupabaseConnector(url, key)
        
        if not connector.test_connection():
            st.error("Failed to connect to Supabase. Check your credentials.")
            return
        
        st.success("✅ Connected to Supabase!")
        
    except Exception as e:
        st.error(f"Error connecting to Supabase: {e}")
        return
    
    # Get table stats
    try:
        auth_count = len(connector.supabase.table('authorized_materials').select('id').execute().data)
        not_auth_count = len(connector.supabase.table('not_authorized_materials').select('id').execute().data)
        exception_count = len(connector.supabase.table('exceptions').select('id').execute().data)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Authorized Materials", f"{auth_count:,}")
        with col2:
            st.metric("Not Authorized", f"{not_auth_count:,}")
        with col3:
            st.metric("Exceptions", f"{exception_count:,}")
        with col4:
            total = auth_count + not_auth_count
            auth_rate = (auth_count / total * 100) if total > 0 else 0
            st.metric("Auth Rate", f"{auth_rate:.1f}%")
    except Exception as e:
        st.warning(f"Could not fetch table stats: {e}")
    
    st.markdown("---")
    
    # Sidebar - Table Selection
    st.sidebar.header("⚙️ Settings")
    
    # Get available tables
    available_tables = get_table_list()
    
    selected_table = st.sidebar.selectbox(
        "Select Data Source:",
        options=available_tables,
        index=0,
        key="table_selector"
    )
    
    st.sidebar.info(f"**Querying:** `{selected_table}`")
    
    # Developer mode toggle
    dev_mode = st.sidebar.checkbox("🔧 Developer Mode", value=False)
    
    st.markdown("---")
    
    # Initialize components
    filter_generator = SQLStyleFilterGenerator(llm)
    query_executor = SupabaseQueryExecutor(connector.supabase)
    responder = NaturalLanguageResponder(llm)
    
    # Sidebar - Query input
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Ask Your Question")
    
    user_query = st.sidebar.text_area(
        "Enter your query:",
        value=st.session_state.get('query', ''),
        placeholder="e.g., Show me Plant 1006 authorization details",
        height=100
    )
    
    # Example queries in sidebar
    st.sidebar.markdown("### 💡 Examples:")
    example_queries = [
        "Show me Plant 1006 details",
        "Which plants have lowest auth rates?",
        "How many materials in Plant 1001?",
        "Top 10 plants by materials",
        "Compare all plants"
    ]
    
    for example in example_queries:
        if st.sidebar.button(example, key=example, use_container_width=True):
            st.session_state.query = example
            st.rerun()
    
    if user_query:
        # Process query
        with st.spinner("🤔 Understanding your query..."):
            filter_spec = filter_generator.generate_filter_spec(user_query, selected_table)
        
        with st.spinner("📊 Fetching data from Supabase..."):
            results = query_executor.execute_query(filter_spec)
        
        # Check for errors
        if 'error' in results:
            st.error(f"❌ Query error: {results['error']}")
            return
        
        # Check if no data was returned
        if results.get('total_count', 0) == 0:
            st.warning("⚠️ No data found matching your query!")
            st.info("""
            **Possible reasons:**
            - The specified plant/material doesn't exist in the selected table
            - Try a different query or table selection
            - Check if data exists in the database
            """)
            
            # Show what was queried
            with st.expander("🔍 Query Details"):
                st.markdown(f"**Table:** `{selected_table}`")
                st.markdown(f"**Filters Applied:**")
                st.json(filter_spec.get('filters', {}))
            return
        
        # Show Query Analysis in expander (only in dev mode)
        if dev_mode:
            with st.expander("🔍 Query Analysis (Developer Mode)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**User Query:**")
                    st.code(user_query)
                    st.markdown("**Generated Filters:**")
                    st.json(filter_spec)
                with col2:
                    st.markdown("**Query Results:**")
                    st.json(results)
        
        # VISUAL DASHBOARD OUTPUT (like sap_dashboard_agent)
        st.markdown("---")
        
        # Generate natural language answer
        with st.spinner("✨ Generating insights..."):
            answer = responder.generate_response(user_query, results)
        
        # Show answer prominently
        st.success(answer)
        
        # Key Metrics Row
        st.markdown("### 📊 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Records", 
                f"{results.get('total_count', 0):,}",
                help="Total records matching your query"
            )
        
        with col2:
            if 'by_plant' in results:
                st.metric(
                    "Plants Found",
                    len(results['by_plant']),
                    help="Number of unique plants in results"
                )
            elif 'by_material' in results:
                st.metric(
                    "Materials Found",
                    len(results['by_material']),
                    help="Number of unique materials"
                )
            else:
                st.metric("Data Source", selected_table)
        
        with col3:
            filters_applied = results.get('metadata', {}).get('filters_applied', {})
            if filters_applied:
                filter_text = ", ".join([f"{k}={v}" for k, v in filters_applied.items()])
                st.metric("Filters Applied", len(filters_applied), help=filter_text)
            else:
                st.metric("Filters Applied", "None", help="No filters - showing all data")
        
        with col4:
            query_time = results.get('metadata', {}).get('query_executed_at', 'N/A')
            if query_time != 'N/A':
                st.metric("Query Time", "< 1s", help=f"Executed at {query_time}")
            else:
                st.metric("Status", "✅ Success")
        
        st.markdown("---")
        
        # Visualizations
        st.markdown("### 📈 Visual Analysis")
        
        create_visualization(results, filter_spec)
    
    else:
        # Show welcome message when no query
        st.info("👈 Enter a question in the sidebar to get started!")
        
        st.markdown("### 💡 What can you ask?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Plant Analysis:**
            - Show me Plant 1006 details
            - Which plants have most materials?
            - Compare Plant 1001 vs 1006
            
            **Authorization Queries:**
            - How many authorized materials?
            - Show not authorized in Plant 1019
            - Authorization rates by plant
            """)
        
        with col2:
            st.markdown("""
            **Material Queries:**
            - Find material 12345
            - Top 10 materials by plant
            - Materials in exceptions
            
            **General Queries:**
            - Show overview of all data
            - Total records in database
            - Exception summary
            """)


if __name__ == "__main__":
    main()
