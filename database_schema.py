"""
Database Schema Configuration
==============================
Centralized configuration for all database tables and columns.
Update this file when schema changes - no need to modify main code!
"""

# Database Tables Configuration
TABLES = {
    "authorized_materials": {
        "has_auth_flag": True,
        "primary_use": "Authorized materials tracking"
    },
    "not_authorized_materials": {
        "has_auth_flag": True,
        "primary_use": "Not authorized materials tracking"
    },
    "exceptions": {
        "has_auth_flag": False,
        "primary_use": "Exception tracking"
    },
    "salesdata": {
        "has_auth_flag": False,
        "primary_use": "Sales order analysis"
    },
    "all": {
        "has_auth_flag": True,
        "primary_use": "Authorization rate analysis"
    }
}

# Column Definitions - Common across all tables
COLUMNS = {
    "plant": {
        "type": "text",
        "description": "Plant/Location code (e.g., '1001', '1006')",
        "filterable": True,
        "aggregatable": True
    },
    "material": {
        "type": "text",
        "description": "Material number/code",
        "filterable": True,
        "aggregatable": True
    },
    "material_description": {
        "type": "text",
        "description": "Material name/description",
        "filterable": True,
        "aggregatable": False
    },
    "auth_flag": {
        "type": "text",
        "description": "Authorization flag: 'Y' (authorized) or 'N' (not authorized)",
        "filterable": True,
        "aggregatable": True,
        "only_in_tables": ["authorized_materials", "not_authorized_materials"]
    },
    "sales_order_number": {
        "type": "text",
        "description": "Sales order ID/number",
        "filterable": True,
        "aggregatable": False
    },
    "sales_document_type": {
        "type": "text",
        "description": "Type of sales document",
        "filterable": True,
        "aggregatable": True
    },
    "order_create_date": {
        "type": "date",
        "description": "Date when order was created",
        "filterable": True,
        "aggregatable": False
    },
    "requested_delivery_date": {
        "type": "date",
        "description": "Requested delivery date",
        "filterable": True,
        "aggregatable": False
    },
    "sold_to_party": {
        "type": "text",
        "description": "Customer ID/number",
        "filterable": True,
        "aggregatable": True
    },
    "sold_to_name": {
        "type": "text",
        "description": "Customer name",
        "filterable": True,
        "aggregatable": False
    },
    "customer_hierarchy": {
        "type": "text",
        "description": "Customer group/hierarchy",
        "filterable": True,
        "aggregatable": True
    },
    "customer_hierarchy_level_1": {
        "type": "text",
        "description": "Top level customer hierarchy",
        "filterable": True,
        "aggregatable": True
    },
    "order_quantity_unit": {
        "type": "numeric",
        "description": "Quantity ordered",
        "filterable": True,
        "aggregatable": False
    },
    "sales_unit_of_measure": {
        "type": "text",
        "description": "Unit of measure (EA, KG, etc.)",
        "filterable": True,
        "aggregatable": True
    },
    "material_status_description": {
        "type": "text",
        "description": "Material status description",
        "filterable": True,
        "aggregatable": True
    },
    "exception_type": {
        "type": "text",
        "description": "Exception category/type",
        "filterable": True,
        "aggregatable": True,
        "only_in_tables": ["exceptions"]
    }
}


def get_table_list():
    """Get list of all available tables"""
    return list(TABLES.keys())


def get_table_description(table_name):
    """Get description of a specific table"""
    return TABLES.get(table_name, {})


def get_columns_for_table(table_name):
    """Get all columns available for a specific table"""
    if table_name == "all":
        # Return columns from both auth tables
        return [col for col, info in COLUMNS.items() 
                if "only_in_tables" not in info or 
                "authorized_materials" in info.get("only_in_tables", [])]
    
    # Return columns that don't have table restrictions or are in the specified table
    return [col for col, info in COLUMNS.items() 
            if "only_in_tables" not in info or 
            table_name in info.get("only_in_tables", [])]


def get_filterable_columns(table_name):
    """Get columns that can be used for filtering"""
    all_cols = get_columns_for_table(table_name)
    return [col for col in all_cols if COLUMNS[col].get("filterable", False)]


def generate_schema_prompt(table_name):
    """Generate schema description for LLM prompt"""
    table_info = TABLES.get(table_name, {})
    available_columns = get_columns_for_table(table_name)
    
    prompt_parts = []
    prompt_parts.append(f"Table: {table_name}")
    prompt_parts.append(f"Purpose: {table_info.get('description', 'N/A')}")
    prompt_parts.append("\nAvailable columns:")
    
    for col in available_columns:
        col_info = COLUMNS[col]
        prompt_parts.append(f"- {col}: {col_info['description']}")
    
    return "\n".join(prompt_parts)


def get_column_type(column_name):
    """Get the data type of a column"""
    return COLUMNS.get(column_name, {}).get("type", "text")
