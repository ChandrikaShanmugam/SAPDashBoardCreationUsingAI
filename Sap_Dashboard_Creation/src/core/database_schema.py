"""
Database Schema Configuration
==============================
Centralized configuration for SAP Exception Report columns.
EXACT column names from CSV - hardcoded for reliability!
"""

# EXACT CSV Column Names - DO NOT MODIFY without checking the CSV file
CSV_COLUMNS = [
    "Sales Order Number",
    "Sales Document Type",
    "Sales Order Line Item",
    "Order Create Date",
    "Requested Delivery Date",
    "Sold-To Party",
    "Sold-to Name",
    "Customer Hierarchy ",  # Note: trailing space in actual data
    "Customer Hierarchy Level 6 Text",
    "Customer Hierarchy Level 7 Text",
    "Material",
    "Material Descrption",  # Note: typo in actual data
    "Material Status Description",
    "Plant Material Status Description",
    "Auth Sell Flag Description",
    "Plant",
    "Order Quantity Sales Unit",
    "Sales Unit of Measure"
]

# Column Metadata - using EXACT names from CSV
COLUMNS = {
    "Sales Order Number": {
        "type": "text",
        "description": "Sales order ID/number",
        "filterable": True,
        "aggregatable": False,
        "common_use": "Identifying specific orders"
    },
    "Sales Document Type": {
        "type": "text",
        "description": "Type of sales document",
        "filterable": True,
        "aggregatable": True,
        "common_use": "Grouping by document type"
    },
    "Sales Order Line Item": {
        "type": "text",
        "description": "Line item number within sales order",
        "filterable": True,
        "aggregatable": False,
        "common_use": "Detailed line item analysis"
    },
    "Order Create Date": {
        "type": "date",
        "description": "Date when order was created",
        "filterable": True,
        "aggregatable": False,
        "common_use": "Time-based filtering"
    },
    "Requested Delivery Date": {
        "type": "date",
        "description": "Requested delivery date",
        "filterable": True,
        "aggregatable": False,
        "common_use": "Delivery timeline analysis"
    },
    "Sold-To Party": {
        "type": "text",
        "description": "Customer ID/account number",
        "filterable": True,
        "aggregatable": True,
        "common_use": "Customer-specific analysis"
    },
    "Sold-to Name": {
        "type": "text",
        "description": "Customer name",
        "filterable": True,
        "aggregatable": False,
        "common_use": "Customer identification"
    },
    "Customer Hierarchy ": {  # Note: trailing space
        "type": "text",
        "description": "Customer group/hierarchy",
        "filterable": True,
        "aggregatable": True,
        "common_use": "Customer hierarchy grouping"
    },
    "Customer Hierarchy Level 6 Text": {
        "type": "text",
        "description": "Customer hierarchy level 6",
        "filterable": True,
        "aggregatable": True,
        "common_use": "Detailed hierarchy analysis"
    },
    "Customer Hierarchy Level 7 Text": {
        "type": "text",
        "description": "Customer hierarchy level 7",
        "filterable": True,
        "aggregatable": True,
        "common_use": "Detailed hierarchy analysis"
    },
    "Material": {
        "type": "text",
        "description": "Material number/code",
        "filterable": True,
        "aggregatable": True,
        "common_use": "Material-specific analysis, top materials"
    },
    "Material Descrption": {  # Note: typo in actual data
        "type": "text",
        "description": "Material name/description (note: spelling in source data)",
        "filterable": True,
        "aggregatable": False,
        "common_use": "Material identification, reporting"
    },
    "Material Status Description": {
        "type": "text",
        "description": "Material status (Active, Inactive, etc.)",
        "filterable": True,
        "aggregatable": True,
        "common_use": "Filtering by material status"
    },
    "Plant Material Status Description": {
        "type": "text",
        "description": "Plant-specific material status",
        "filterable": True,
        "aggregatable": True,
        "common_use": "Plant-level material status analysis"
    },
    "Auth Sell Flag Description": {
        "type": "text",
        "description": "Authorization flag: 'Yes' (authorized) or 'No' (not authorized)",
        "filterable": True,
        "aggregatable": True,
        "common_use": "Authorization status filtering, auth rate calculations"
    },
    "Plant": {
        "type": "text",
        "description": "Plant/Location code (e.g., '1007', '7001')",
        "filterable": True,
        "aggregatable": True,
        "common_use": "Plant-specific analysis, location filtering"
    },
    "Order Quantity Sales Unit": {
        "type": "numeric",
        "description": "Quantity ordered in sales units",
        "filterable": True,
        "aggregatable": True,
        "common_use": "Quantity summation, volume analysis"
    },
    "Sales Unit of Measure": {
        "type": "text",
        "description": "Unit of measure (EA, CS, KG, etc.)",
        "filterable": True,
        "aggregatable": True,
        "common_use": "Unit-based grouping"
    }
}


def get_all_columns():
    """Get list of all column names in exact order from CSV"""
    return CSV_COLUMNS.copy()


def get_filterable_columns():
    """Get columns that can be used for filtering"""
    return [col for col in CSV_COLUMNS if COLUMNS[col].get("filterable", False)]


def get_aggregatable_columns():
    """Get columns that can be aggregated (sum, count, etc.)"""
    return [col for col in CSV_COLUMNS if COLUMNS[col].get("aggregatable", False)]


def generate_schema_prompt():
    """Generate detailed schema description for LLM prompt"""
    prompt_parts = []
    prompt_parts.append("SAP Exception Report - Available Columns:")
    prompt_parts.append("=" * 70)
    
    for col in CSV_COLUMNS:
        col_info = COLUMNS[col]
        prompt_parts.append(f"\n• {col}")
        prompt_parts.append(f"  Type: {col_info['type']}")
        prompt_parts.append(f"  Description: {col_info['description']}")
        prompt_parts.append(f"  Common Use: {col_info.get('common_use', 'General analysis')}")
    
    return "\n".join(prompt_parts)


def get_common_filters():
    """Get most commonly used filter columns"""
    return {
        "Plant": "Plant/Location filtering (e.g., '1007', '7001')",
        "Material": "Material-specific filtering",
        "Auth Sell Flag Description": "Authorization status ('Yes' or 'No')",
        "Material Status Description": "Material status filtering",
        "Sold-To Party": "Customer-specific filtering"
    }


def get_common_chart_columns():
    """Get columns commonly used in visualizations"""
    return {
        "for_bar_charts": ["Plant", "Material", "Material Status Description", "Auth Sell Flag Description"],
        "for_pie_charts": ["Auth Sell Flag Description", "Material Status Description", "Sales Unit of Measure"],
        "for_aggregation": ["Order Quantity Sales Unit"],
        "for_details": ["Material", "Material Descrption", "Plant", "Sales Order Number", "Order Quantity Sales Unit"]
    }


def get_column_type(column_name):
    """Get the data type of a column"""
    return COLUMNS.get(column_name, {}).get("type", "text")


def validate_column_name(column_name):
    """Check if a column name exists in the schema"""
    return column_name in CSV_COLUMNS
