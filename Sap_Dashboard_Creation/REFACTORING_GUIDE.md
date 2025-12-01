# SAP Dashboard Refactoring - Prompt Management & Dynamic Schema

## Changes Made

### 1. **Prompt Templates** (`src/config/prompts/`)
Moved all LLM prompts from hardcoded strings to external template files for easy management:

- **`filter_extraction_prompt.txt`** - Stage 1: Filter extraction and aggregation detection
- **`chart_generation_prompt.txt`** - Stage 2: Chart and table generation

**Benefits:**
- Easy to update prompts without touching code
- Version control for prompt changes
- Can add new prompts by creating new files

### 2. **Dynamic Schema Generator** (`src/core/schema_generator.py`)
Replaced hardcoded `database_schema.py` with dynamic schema generation:

```python
from schema_generator import SchemaGenerator

# Initialize with your data
schema_gen = SchemaGenerator(data_dict)

# Get all columns
columns = schema_gen.get_all_columns()

# Get schema prompt for LLM
schema_prompt = schema_gen.generate_schema_prompt()

# Get chart-specific columns
chart_cols = schema_gen.get_common_chart_columns()
```

**Features:**
- Automatically detects columns from CSV files
- Generates sample values for each column
- Categorizes columns for different chart types
- Supports multiple datasets

### 3. **Prompt Template Manager** (`src/core/prompt_manager.py`)
Manages loading and formatting of prompt templates:

```python
from prompt_manager import PromptTemplateManager

prompt_mgr = PromptTemplateManager()

# Get a template
template = prompt_mgr.get_template('filter_extraction')

# Format with variables
formatted = prompt_mgr.format_template(
    'chart_generation',
    data_sample=sample,
    all_columns=columns
)

# Add custom templates at runtime
prompt_mgr.add_custom_template('my_prompt', 'template content')
```

## Adding New CSV Files

### Step 1: Add CSV to data directory
```bash
cp your_new_file.csv Sap_Dashboard_Creation/data/
```

### Step 2: Load in `load_sap_data()` function
```python
@st.cache_data
def load_sap_data():
    data_dir = Path(__file__).parent.parent.parent / 'data'
    
    # Add your new file
    new_data = load_exception_csv(str(data_dir / 'your_new_file.csv'))
    
    data = {
        'exception_report': exception_report,
        'new_dataset': new_data  # Add here
    }
    return data
```

### Step 3: Schema is Generated Automatically!
The `SchemaGenerator` will automatically:
- Detect all columns in the new CSV
- Generate sample values
- Create schema prompts for LLM
- Categorize columns for charts

**No hardcoding required!**

## Updating Prompts

### Edit Template Files
1. Navigate to `src/config/prompts/`
2. Edit the `.txt` files directly
3. Changes take effect on next app restart

### Add New Template
```python
# 1. Create new file in src/config/prompts/
# my_new_prompt.txt

# 2. Update PromptTemplateManager._load_templates()
template_files = {
    'filter_extraction': 'filter_extraction_prompt.txt',
    'chart_generation': 'chart_generation_prompt.txt',
    'my_new_prompt': 'my_new_prompt.txt'  # Add here
}

# 3. Use it
formatted = self.prompt_manager.get_template('my_new_prompt')
```

## Example: Adding Sales Data CSV

```python
# Step 1: Add file to data/
# sales_data.csv

# Step 2: Update load_sap_data()
def load_sap_data():
    data_dir = Path(__file__).parent.parent.parent / 'data'
    
    exception_report = load_exception_csv(str(data_dir / 'Sales Order Exception report 13 and 14 Nov 2025.csv'))
    sales_data = pd.read_csv(str(data_dir / 'sales_data.csv'))  # New!
    
    data = {
        'exception_report': exception_report,
        'sales_data': sales_data  # Add to dictionary
    }
    return data

# Step 3: SchemaGenerator handles the rest automatically!
# - Columns detected: ['Order ID', 'Customer', 'Amount', 'Date', ...]
# - Sample values generated
# - Chart columns categorized
# - Ready to use in prompts
```

## File Structure
```
Sap_Dashboard_Creation/
├── data/
│   ├── Sales Order Exception report 13 and 14 Nov 2025.csv
│   └── [your new CSVs here]
├── src/
│   ├── config/
│   │   └── prompts/
│   │       ├── filter_extraction_prompt.txt
│   │       └── chart_generation_prompt.txt
│   └── core/
│       ├── sap_dashboard_agent.py (main app)
│       ├── schema_generator.py (NEW - dynamic schema)
│       ├── prompt_manager.py (NEW - prompt templates)
│       ├── database_schema.py (OLD - can be deprecated)
│       └── exception_handler.py
```

## Migration Notes

### Before (Hardcoded)
```python
# Prompts hardcoded in sap_dashboard_agent.py
self.intent_system_prompt = """You are a SAP data analyst..."""

# Schema hardcoded in database_schema.py
CSV_COLUMNS = ["Sales Order Number", "Material", ...]
```

### After (Dynamic)
```python
# Prompts in template files
self.intent_system_prompt = self.prompt_manager.get_template('filter_extraction')

# Schema generated from actual data
self.schema_generator = SchemaGenerator(data)
self.columns_info = self.schema_generator.generate_schema_prompt()
```

## Benefits

1. **Easier Maintenance**: Update prompts without code changes
2. **Scalable**: Add new CSV files without schema hardcoding
3. **Flexible**: Customize prompts per use case
4. **Version Control**: Track prompt changes separately from code
5. **Dynamic**: Schema adapts to actual data structure

## Testing

Restart the application to test:
```bash
cd /Users/chandrika/Documents/SAPDashBoardCreationUsingAI/Sap_Dashboard_Creation
python3 -m streamlit run ./src/core/sap_dashboard_agent.py
```

All existing functionality should work as before!
