"""
UI Module for SAP Dashboard Agent
Contains UI components and styling
"""

from .styles import get_custom_css
from .components import (
    render_sidebar_header,
    render_chat_history,
    render_followup_questions,
    render_example_queries,
    render_query_input,
    render_settings,
    render_sidebar_footer,
    render_performance_metrics,
    render_debug_tabs,
    render_metrics_columns,
    show_info_message,
    show_success_message,
    show_warning_message,
    show_error_message
)

__all__ = [
    'get_custom_css',
    'render_sidebar_header',
    'render_chat_history',
    'render_followup_questions',
    'render_example_queries',
    'render_query_input',
    'render_settings',
    'render_sidebar_footer',
    'render_performance_metrics',
    'render_debug_tabs',
    'render_metrics_columns',
    'show_info_message',
    'show_success_message',
    'show_warning_message',
    'show_error_message'
]
