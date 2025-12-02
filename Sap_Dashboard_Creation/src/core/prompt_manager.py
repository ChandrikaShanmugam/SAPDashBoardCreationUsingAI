"""
Prompt Template Manager
=======================
Manages loading and formatting of prompt templates from files.
Includes database schema and relationship information.
"""

from pathlib import Path
from typing import Dict
import logging
import sys

# Add parent directory to path to import database_schema
sys.path.insert(0, str(Path(__file__).parent))
import database_schema as ds

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
    
    def get_relationship_info(self) -> str:
        """
        Get formatted relationship information for prompts
        
        Returns:
            Formatted string describing table relationships
        """
        fk = ds.get_foreign_key_relationships()
        common_cols = ds.get_common_columns()
        
        info_parts = []
        info_parts.append("Two tables are available:")
        info_parts.append("1. Sales Order Exception Report (69 columns) - Main sales order data")
        info_parts.append("2. A1P Location Sequence (15 columns) - Plant location and material sequence data")
        info_parts.append("")
        info_parts.append("Tables can be joined using:")
        info_parts.append(f"- Common columns: {', '.join(common_cols)}")
        info_parts.append(f"- Join condition: Sales Order.Plant = Location.Plant(Location) AND Sales Order.Material = Location.Material")
        info_parts.append("")
        
        relationship = fk.get("sales_order_to_location", {})
        if relationship:
            info_parts.append("Foreign Key Relationships:")
            for rel in relationship.get("relationships", []):
                info_parts.append(f"- {rel['from_column']} (Sales Order) → {rel['to_column']} (Location) [{rel['relationship_type']}]")
            info_parts.append("")
            info_parts.append(f"Description: {relationship.get('description', 'N/A')}")
        
        return "\n".join(info_parts)
    
    def get_columns_info(self) -> str:
        """
        Get formatted column information for both tables
        
        Returns:
            Formatted string with column information
        """
        info_parts = []
        
        # Sales Order columns
        info_parts.append("SALES ORDER EXCEPTION REPORT COLUMNS:")
        sales_cols = ds.get_all_sales_order_columns()
        info_parts.append(f"Total: {len(sales_cols)} columns")
        info_parts.append(f"Key columns: {', '.join(sales_cols[:10])}... (and {len(sales_cols) - 10} more)")
        info_parts.append("")
        
        # Location Sequence columns
        info_parts.append("A1P LOCATION SEQUENCE COLUMNS:")
        loc_cols = ds.get_all_location_sequence_columns()
        info_parts.append(f"Total: {len(loc_cols)} columns")
        info_parts.append(f"All columns: {', '.join(loc_cols)}")
        
        return "\n".join(info_parts)
    
    def format_filter_extraction_prompt(self, user_query: str) -> str:
        """
        Format the filter extraction prompt with schema information
        
        Args:
            user_query: User's natural language query
        
        Returns:
            Formatted prompt ready for LLM
        """
        template = self.get_template("filter_extraction")
        if not template:
            logger.error("Filter extraction template not found")
            return ""
        
        try:
            return template.format(
                columns_info=self.get_columns_info(),
                relationship_info=self.get_relationship_info(),
                user_query=user_query
            )
        except KeyError as e:
            logger.error(f"Missing template variable in filter_extraction: {e}")
            return template
