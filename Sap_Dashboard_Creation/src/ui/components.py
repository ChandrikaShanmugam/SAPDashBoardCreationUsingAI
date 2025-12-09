"""
UI Components for SAP Dashboard Agent
Reusable Streamlit UI components
"""

import streamlit as st
from datetime import datetime
from typing import List, Dict, Any
import json


def render_sidebar_header():
    """Render the SAP Assistant header in sidebar"""
    st.title("🤖 SAP Assistant")
    st.markdown("Ask questions about your SAP data.")
    st.markdown("---")

def render_chat_history(chat_history: List[Dict[str, str]]):
    """Render conversation history"""
    if not chat_history:
        return
    
    st.markdown("### 💬 Conversation")
    with st.container(height=300):
        for i, msg in enumerate(chat_history):
            if msg['role'] == 'user':
                st.markdown(
                    f'<div class="chat-message user-message"><b>You:</b> {msg["content"]}</div>', 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-message assistant-message"><b>Assistant:</b> {msg["content"]}</div>', 
                    unsafe_allow_html=True
                )
    st.markdown("---")


def render_followup_questions(followup_questions: List[str]):
    """Render follow-up question buttons"""
    if not followup_questions:
        return
    
    st.markdown("#### Suggested Follow-ups")
    for idx, question in enumerate(followup_questions):
        if st.button(question, key=f"followup_{idx}"):
            st.session_state.user_input_value = question
            st.session_state.current_query = question
            st.session_state.process_query = True
            st.rerun()


def render_example_queries():
    """Render example query buttons"""
    st.markdown("#### 💡 Example Queries")
    example_queries = [
        "Show me authorized to sell details",
        "What are the sales exceptions?",
        "Give me plant-wise analysis",
        "Show overview of all data"
    ]
    
    for example in example_queries:
        if st.button(example, key=f"example_{example}"):
            st.session_state.user_input_value = example
            st.session_state.current_query = example
            st.session_state.process_query = True
            st.rerun()


def render_query_input() -> str:
    """Render the query input area and return the user query"""
    st.markdown("### 💬 Ask a Question")
    
    # Initialize input value in session state
    if 'user_input_value' not in st.session_state:
        st.session_state.user_input_value = ""
    if 'process_query' not in st.session_state:
        st.session_state.process_query = False
    if 'input_key_counter' not in st.session_state:
        st.session_state.input_key_counter = 0
    
    user_query = st.text_area(
        "Your question:",
        value=st.session_state.user_input_value,
        placeholder="e.g., Show me authorized to sell details",
        height=100,
        key=f"user_query_input_{st.session_state.input_key_counter}",
        label_visibility="collapsed"
    )
    
    # Store current query - text area value takes precedence unless it's empty during processing
    if user_query:
        st.session_state.current_query = user_query
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➤ Send", key="send_btn", use_container_width=True):
            if user_query:
                st.session_state.process_query = True
                st.rerun()
    with col2:
        if st.button("🗑️ Clear", key="clear_btn", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.followup_questions = []
            st.session_state.user_input_value = ""
            st.session_state.current_query = ""
            st.rerun()
    
    return user_query


def render_settings() -> tuple:
    """Render settings section and return dev_mode and show_metrics flags"""
    st.markdown("---")
    
    with st.expander("⚙️ Settings", expanded=False):
        dev_mode = st.checkbox(
            "🔧 Developer Mode", 
            value=False, 
            key="dev_mode_checkbox", 
            help="Show API requests, console logs, and debug info"
        )
        show_metrics = st.checkbox(
            "📊 Show Performance Metrics", 
            value=False, 
            key="show_metrics_checkbox"
        )
    
    return dev_mode, show_metrics


def render_sidebar_footer(dev_mode: bool = False):
    """Render sidebar footer with timestamp and dev mode indicator"""
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if dev_mode:
        st.sidebar.markdown("🔧 **Developer Mode Active**")


def render_performance_metrics(metrics: Dict[str, float]):
    """Render performance metrics in an expander"""
    if not metrics:
        return
    
    with st.expander("⏱️ Performance Metrics"):
        cols = st.columns(3)
        
        if 'initialization_time' in metrics:
            cols[0].metric("Initialization", f"{metrics['initialization_time']:.2f}s")
        if 'filter_extraction_time' in metrics:
            cols[1].metric("Filter Extraction", f"{metrics['filter_extraction_time']:.2f}s")
        if 'dashboard_generation_time' in metrics:
            cols[2].metric("Dashboard Generation", f"{metrics['dashboard_generation_time']:.2f}s")
        
        if 'total_time' in metrics:
            st.markdown(f"**Total Query Time:** `{metrics['total_time']:.2f}s`")


def render_debug_tabs(data: Dict[str, Any], api_calls: List[Dict], logs: List[str]):
    """Render debug information in tabs"""
    debug_tabs = st.tabs(["📝 Console Logs", "🔌 API Requests", "📚 Usage Guide", "💾 Data Info"])
    
    # Tab 1: Console Logs
    with debug_tabs[0]:
        st.subheader("📝 Console Output")
        st.info("View real-time logs from the application backend.")
        
        if logs:
            log_text = "\n".join(logs[-50:])  # Show last 50 logs
            st.code(log_text, language="log")
        else:
            st.write("No logs available yet. Logs will appear here as you interact with the app.")
    
    # Tab 2: API Requests
    with debug_tabs[1]:
        st.subheader("🔌 API Request/Response Log")
        if api_calls:
            for i, call in enumerate(reversed(api_calls[-5:])):  # Show last 5 calls
                with st.expander(f"{call['timestamp']} - {call['type']} ({call.get('duration', 'N/A')})"):
                    st.markdown(f"**Query:** `{call['query']}`")
                    st.markdown("**Response:**")
                    st.json(call['response'])
        else:
            st.write("No API calls logged yet.")
    
    # Tab 3: Usage Guide
    with debug_tabs[2]:
        st.subheader("📚 How to Use & Debug")
        st.markdown("""
        **Application Architecture:**
        - **Stage 1:** Filter Extraction + Follow-up Questions (Parallel)
        - **Stage 2:** Chart Generation
        
        **Developer Mode Features:**
        - View real-time console logs
        - See LLM API requests and responses
        - Monitor performance metrics
        - Inspect data loading status
        
        **Running Manually (Terminal):**
        """)
        st.code("""
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


def render_metrics_columns(data, col_mapping: Dict[str, str]):
    """Render metric columns for data overview"""
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Records", f"{len(data):,}")
    
    if col_mapping.get('material') and col_mapping['material'] in data.columns:
        col2.metric("Unique Materials", f"{data[col_mapping['material']].nunique():,}")
    
    if col_mapping.get('plant') and col_mapping['plant'] in data.columns:
        col3.metric("Unique Plants", f"{data[col_mapping['plant']].nunique():,}")
    
    if col_mapping.get('status') and col_mapping['status'] in data.columns:
        active_count = len(data[data[col_mapping['status']] == 'Active'])
        col4.metric("Active Materials", f"{active_count:,}")


def show_info_message(message: str):
    """Display an info message"""
    st.info(message)


def show_success_message(message: str):
    """Display a success message"""
    st.success(message)


def show_warning_message(message: str):
    """Display a warning message"""
    st.warning(message)


def show_error_message(message: str):
    """Display an error message"""
    st.error(message)
