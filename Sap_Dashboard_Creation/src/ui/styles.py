"""
Custom CSS styles for SAP Dashboard Agent
"""

def get_custom_css() -> str:
    """Return the custom CSS for the Streamlit application"""
    return """
    <style>
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    .chat-message {
        padding: 5px 7px;
        border-radius: 15px;
        margin: 2px 0;
        font-size: 14px;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: left;
    }
    .assistant-message {
        background-color: #f5f5f5;
        text-align: left;
    }
    [data-testid="stSidebar"] {
        min-width: 30%;
        max-width: 30%;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }
    /* Pill-style buttons for examples and follow-ups */
    .stButton button {
        width: 100%;
        border-radius: 25px !important;
        text-align: left;
        padding: 2px 8px !important;
        border: 2px solid #4A90E2 !important;
        background-color: white !important;
        color: #4A90E2 !important;
        font-size: 10px !important;
        margin-bottom: 2px !important;
        height: auto !important;
        min-height: 32px !important;
        line-height: 1.2 !important;
    }
    .stButton button:hover {
        background-color: #E3F2FD !important;
        border-color: #2E7BD6 !important;
    }
    /* Remove spacing and padding aggressively */
    .element-container {
        margin-bottom: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stVerticalBlock"] > div {
        gap: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stSidebar"] .element-container {
        padding: 0 !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] h4 {
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
        font-size: 0.95rem !important;
    }
    /* Reduce spacing around horizontal rules and other elements */
    [data-testid="stSidebar"] hr {
        margin-top: 0.7rem !important;
        margin-bottom: 0.7rem !important;
    }
    [data-testid="stSidebar"] h1 {
        margin-bottom: 0.3=7rem !important;
    }
    [data-testid="stSidebar"] p {
        margin-bottom: 0.7rem !important;
    }
    [data-testid="stSidebar"] .stMarkdown {
        margin-bottom: 0.2 !important;
    }
    </style>
    """
