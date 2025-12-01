"""
Prompt Template Manager
=======================
Manages loading and formatting of prompt templates from files.
"""

from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class PromptTemplateManager:
    """Manages prompt templates for LLM interactions"""
    
    def __init__(self, prompt_dir: Path = None):
        """
        Initialize prompt manager
        
        Args:
            prompt_dir: Directory containing prompt template files
        """
        if prompt_dir is None:
            # Default to prompts directory relative to this file
            prompt_dir = Path(__file__).parent.parent / 'config' / 'prompts'
        
        self.prompt_dir = Path(prompt_dir)
        self.templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all prompt templates from directory"""
        if not self.prompt_dir.exists():
            logger.warning(f"Prompt directory not found: {self.prompt_dir}")
            return
        
        template_files = {
            'filter_extraction': 'filter_extraction_prompt.txt',
            'chart_generation': 'chart_generation_prompt.txt'
        }
        
        for name, filename in template_files.items():
            filepath = self.prompt_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.templates[name] = f.read()
                logger.info(f"Loaded template: {name} from {filename}")
            else:
                logger.warning(f"Template file not found: {filepath}")
    
    def get_template(self, template_name: str) -> str:
        """
        Get a prompt template by name
        
        Args:
            template_name: Name of the template (e.g., 'filter_extraction', 'chart_generation')
        
        Returns:
            Template string
        """
        return self.templates.get(template_name, "")
    
    def format_template(self, template_name: str, **kwargs) -> str:
        """
        Get and format a template with variables
        
        Args:
            template_name: Name of the template
            **kwargs: Variables to format into the template
        
        Returns:
            Formatted template string
        """
        template = self.get_template(template_name)
        if not template:
            logger.error(f"Template not found: {template_name}")
            return ""
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            return template
    
    def add_custom_template(self, name: str, template_content: str):
        """
        Add a custom template at runtime
        
        Args:
            name: Template name
            template_content: Template string
        """
        self.templates[name] = template_content
        logger.info(f"Added custom template: {name}")
