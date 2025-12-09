# UI Module

This module contains all UI-related code for the SAP Dashboard Agent, separated from the business logic for better maintainability and reusability.

## Structure

```
ui/
├── __init__.py         # Package initialization and exports
├── components.py       # Reusable Streamlit UI components
├── styles.py          # CSS styling definitions
└── README.md          # This file
```

## Files

### `styles.py`
Contains all custom CSS styling for the application. The main function is:
- `get_custom_css()` - Returns the complete CSS string for the application

**Features:**
- Chat-like interface styling
- Sidebar layout (30-70 split)
- Pill-style button designs
- Custom spacing and padding

### `components.py`
Contains reusable Streamlit UI components as functions. These components can be easily imported and used throughout the application.

**Main Components:**

#### Sidebar Components
- `render_sidebar_header()` - SAP Assistant title and description
- `render_chat_history(chat_history)` - Displays conversation history
- `render_followup_questions(followup_questions)` - Shows suggested follow-up questions
- `render_example_queries()` - Displays example query buttons
- `render_query_input()` - Text input area for user queries
- `render_settings()` - Settings expander with dev mode and metrics toggles
- `render_sidebar_footer(dev_mode)` - Timestamp and dev mode indicator

#### Main Area Components
- `render_performance_metrics(metrics)` - Performance metrics display
- `render_debug_tabs(data, api_calls, logs)` - Developer debug tabs
- `render_metrics_columns(data, col_mapping)` - Data metric columns

#### Utility Components
- `show_info_message(message)` - Info notification
- `show_success_message(message)` - Success notification
- `show_warning_message(message)` - Warning notification
- `show_error_message(message)` - Error notification

## Usage

### In the main application file:

```python
# Import the UI components
from ui.styles import get_custom_css
from ui.components import (
    render_sidebar_header,
    render_chat_history,
    render_followup_questions,
    render_query_input,
    render_settings,
    render_debug_tabs,
    show_info_message
)

# Apply styles
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Use components
with st.sidebar:
    render_sidebar_header()
    render_chat_history(st.session_state.chat_history)
    user_query = render_query_input()
    dev_mode, show_metrics = render_settings()

# Show messages
show_info_message("Welcome to the dashboard!")
```

## Benefits of Separation

1. **Maintainability**: UI code is isolated from business logic
2. **Reusability**: Components can be reused across different parts of the app
3. **Testing**: UI components can be tested independently
4. **Scalability**: Easy to add new components without cluttering main file
5. **Collaboration**: Multiple developers can work on UI without conflicts
6. **Consistency**: Centralized styling ensures consistent look and feel

## Future Enhancements

- Add unit tests for UI components
- Create theme variants (light/dark mode)
- Add more reusable chart components
- Implement responsive design utilities
- Add accessibility features (ARIA labels, keyboard navigation)
