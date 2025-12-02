"""
Database Schema Configuration
==============================
Centralized configuration for SAP Exception Report and A1P Location Sequence tables.
EXACT column names from CSV - hardcoded for reliability!
"""

# ==============================================================================
# NEW SCHEMA - TWO TABLES WITH FOREIGN KEY RELATIONSHIPS
# ==============================================================================

# Table 1: Sales Order Exception Report
SALES_ORDER_EXCEPTION_COLUMNS = [
    "Sales Order Number",
    "Sales Document Type",
    "Sales Order Line Item",
    "Order Create Date",
    "Created Time",
    "Requested Delivery Date",
    "Sold-To Party",
    "Sold-to Name",
    "Customer Hierarchy ",
    "Customer Hierarchy Level 6 Text",
    "Customer Hierarchy Level 7 Text",
    "Cust PO Number",
    "Cust PO Date",
    "Purchase Order Type",
    "Sales Line Item Status",
    "Reason for Rejection",
    "Reason for Rejection Description ",
    "Material",
    "Material Descrption",
    "Material Status",
    "Material Status Description",
    "Plant Material Status",
    "Plant Material Status Description",
    "Auth Sell Flag",
    "Auth Sell Flag Description",
    "Material Found",
    "INVENID (Order)",
    "UPC",
    "GTIN",
    "Pespsi Invenid",
    "Customer Material Number",
    "Item Category",
    "Plant",
    "Location ID",
    "Location Description",
    "Net Value Doc Currency",
    "EAN/UPC",
    "Route",
    "EAN Unit of Measure ",
    "EAN Category ",
    "System ID",
    "Year",
    "Quarter",
    "Week",
    "Sales Document type Description",
    "Created By",
    "Sales Organization",
    "Division",
    "Distribution Channel",
    "Order Reason",
    "Delivery Block",
    "Shipping Condition",
    "Sold-to City",
    "Sold-to Region",
    "Sold-to Street",
    "Customer Reference",
    "Customer Reference Date",
    "Overall Sales Order Status",
    "Overall Credit Status",
    "Overall Blocked Status",
    "Billing Block",
    "Order Quantity Sales Unit",
    "Sales Unit of Measure",
    "Order Quantity (BU)",
    "Base Unit of Measure",
    "Order Quantity (CS)",
    "Net Price Doc Currency",
    "Unique Order Qty (SU)",
    "Available Stock (F)"
]

# Table 2: A1P Location Sequence Export
A1P_LOCATION_SEQUENCE_COLUMNS = [
    "Plant(Location)",
    "Material",
    "Inven Id",
    "Location Id",
    "Inven Sequ Number",
    "Inven Group Sequ",
    "Inven Group Name",
    "Auth to sell flag",
    "Sequence Number",
    "Created By",
    "Created on",
    "Time",
    "Changed By",
    "Changed On",
    "Time of change"
]

# Foreign Key Relationships
FOREIGN_KEY_RELATIONSHIPS = {
    "sales_order_to_location": {
        "from_table": "Sales Order Exception Report",
        "to_table": "A1P Location Sequence",
        "relationships": [
            {
                "from_column": "Plant",
                "to_column": "Plant(Location)",
                "relationship_type": "many-to-one",
                "description": "Links sales orders to plant location data"
            },
            {
                "from_column": "Material",
                "to_column": "Material",
                "relationship_type": "many-to-one",
                "description": "Links sales order materials to location sequence data"
            }
        ],
        "composite_key": ["Plant", "Material"],
        "description": "Sales orders can be linked to location sequences via Plant and Material combination"
    }
}

# Table Metadata for Sales Order Exception Report
SALES_ORDER_EXCEPTION_SCHEMA = {
    "Sales Order Number": {"type": "text", "primary_key": True, "description": "Unique sales order identifier"},
    "Sales Document Type": {"type": "text", "description": "Type of sales document (e.g., ZOR)"},
    "Sales Order Line Item": {"type": "text", "description": "Line item number within sales order"},
    "Order Create Date": {"type": "date", "description": "Date when order was created (YYYYMMDD)"},
    "Created Time": {"type": "time", "description": "Time when order was created"},
    "Requested Delivery Date": {"type": "date", "description": "Requested delivery date"},
    "Sold-To Party": {"type": "text", "description": "Customer ID/account number"},
    "Sold-to Name": {"type": "text", "description": "Customer name"},
    "Customer Hierarchy ": {"type": "text", "description": "Customer hierarchy identifier"},
    "Customer Hierarchy Level 6 Text": {"type": "text", "description": "Customer hierarchy level 6"},
    "Customer Hierarchy Level 7 Text": {"type": "text", "description": "Customer hierarchy level 7"},
    "Cust PO Number": {"type": "text", "description": "Customer purchase order number"},
    "Cust PO Date": {"type": "date", "description": "Customer PO date"},
    "Purchase Order Type": {"type": "text", "description": "Type of purchase order (e.g., ZEDI)"},
    "Sales Line Item Status": {"type": "text", "description": "Status of the line item"},
    "Reason for Rejection": {"type": "text", "description": "Rejection reason code"},
    "Reason for Rejection Description ": {"type": "text", "description": "Rejection reason description"},
    "Material": {"type": "text", "foreign_key": True, "description": "Material number/code"},
    "Material Descrption": {"type": "text", "description": "Material name/description"},
    "Material Status": {"type": "text", "description": "Material status code"},
    "Material Status Description": {"type": "text", "description": "Material status description"},
    "Plant Material Status": {"type": "text", "description": "Plant-specific material status code"},
    "Plant Material Status Description": {"type": "text", "description": "Plant-specific material status"},
    "Auth Sell Flag": {"type": "text", "description": "Authorization flag (Y/N)"},
    "Auth Sell Flag Description": {"type": "text", "description": "Authorization description"},
    "Material Found": {"type": "text", "description": "Material found indicator"},
    "INVENID (Order)": {"type": "text", "description": "Inventory ID for order"},
    "UPC": {"type": "text", "description": "Universal Product Code"},
    "GTIN": {"type": "text", "description": "Global Trade Item Number"},
    "Pespsi Invenid": {"type": "text", "description": "PepsiCo inventory ID"},
    "Customer Material Number": {"type": "text", "description": "Customer-specific material number"},
    "Item Category": {"type": "text", "description": "Item category code"},
    "Plant": {"type": "text", "foreign_key": True, "description": "Plant/Location code"},
    "Location ID": {"type": "text", "description": "Location identifier"},
    "Location Description": {"type": "text", "description": "Location description"},
    "Net Value Doc Currency": {"type": "numeric", "description": "Net value in document currency"},
    "EAN/UPC": {"type": "text", "description": "EAN or UPC code"},
    "Route": {"type": "text", "description": "Route information"},
    "EAN Unit of Measure ": {"type": "text", "description": "EAN unit of measure"},
    "EAN Category ": {"type": "text", "description": "EAN category"},
    "System ID": {"type": "text", "description": "System identifier (e.g., A1P)"},
    "Year": {"type": "integer", "description": "Year"},
    "Quarter": {"type": "text", "description": "Quarter (e.g., Q4)"},
    "Week": {"type": "integer", "description": "Week number"},
    "Sales Document type Description": {"type": "text", "description": "Sales document type description"},
    "Created By": {"type": "text", "description": "User who created the order"},
    "Sales Organization": {"type": "text", "description": "Sales organization code"},
    "Division": {"type": "text", "description": "Division code"},
    "Distribution Channel": {"type": "text", "description": "Distribution channel code"},
    "Order Reason": {"type": "text", "description": "Reason for order"},
    "Delivery Block": {"type": "text", "description": "Delivery block indicator"},
    "Shipping Condition": {"type": "text", "description": "Shipping condition code"},
    "Sold-to City": {"type": "text", "description": "Customer city"},
    "Sold-to Region": {"type": "text", "description": "Customer region/state"},
    "Sold-to Street": {"type": "text", "description": "Customer street address"},
    "Customer Reference": {"type": "text", "description": "Customer reference"},
    "Customer Reference Date": {"type": "date", "description": "Customer reference date"},
    "Overall Sales Order Status": {"type": "text", "description": "Overall order status"},
    "Overall Credit Status": {"type": "text", "description": "Credit check status"},
    "Overall Blocked Status": {"type": "text", "description": "Block status"},
    "Billing Block": {"type": "text", "description": "Billing block indicator"},
    "Order Quantity Sales Unit": {"type": "numeric", "description": "Quantity in sales units"},
    "Sales Unit of Measure": {"type": "text", "description": "Sales unit (CS, EA, etc.)"},
    "Order Quantity (BU)": {"type": "numeric", "description": "Quantity in base units"},
    "Base Unit of Measure": {"type": "text", "description": "Base unit of measure"},
    "Order Quantity (CS)": {"type": "numeric", "description": "Quantity in cases"},
    "Net Price Doc Currency": {"type": "numeric", "description": "Net price in document currency"},
    "Unique Order Qty (SU)": {"type": "numeric", "description": "Unique order quantity in sales units"},
    "Available Stock (F)": {"type": "numeric", "description": "Available stock"}
}

# Table Metadata for A1P Location Sequence
A1P_LOCATION_SEQUENCE_SCHEMA = {
    "Plant(Location)": {"type": "text", "primary_key": True, "foreign_key_target": True, "description": "Plant/Location code"},
    "Material": {"type": "text", "primary_key": True, "foreign_key_target": True, "description": "Material number/code"},
    "Inven Id": {"type": "text", "description": "Inventory ID"},
    "Location Id": {"type": "text", "description": "Location identifier"},
    "Inven Sequ Number": {"type": "integer", "description": "Inventory sequence number"},
    "Inven Group Sequ": {"type": "integer", "description": "Inventory group sequence"},
    "Inven Group Name": {"type": "text", "description": "Inventory group name"},
    "Auth to sell flag": {"type": "text", "description": "Authorization to sell flag (Y/N)"},
    "Sequence Number": {"type": "integer", "description": "Sequence number"},
    "Created By": {"type": "text", "description": "User who created the record"},
    "Created on": {"type": "date", "description": "Creation date"},
    "Time": {"type": "time", "description": "Creation time"},
    "Changed By": {"type": "text", "description": "User who last changed the record"},
    "Changed On": {"type": "date", "description": "Last change date"},
    "Time of change": {"type": "time", "description": "Last change time"}
}


# ==============================================================================
# OLD SCHEMA - COMMENTED OUT FOR BACKWARD COMPATIBILITY
# ==============================================================================

# # EXACT CSV Column Names - DO NOT MODIFY without checking the CSV file
# CSV_COLUMNS = [
#     "Sales Order Number",
#     "Sales Document Type",
#     "Sales Order Line Item",
#     "Order Create Date",
#     "Requested Delivery Date",
#     "Sold-To Party",
#     "Sold-to Name",
#     "Customer Hierarchy ",  # Note: trailing space in actual data
#     "Customer Hierarchy Level 6 Text",
#     "Customer Hierarchy Level 7 Text",
#     "Material",
#     "Material Descrption",  # Note: typo in actual data
#     "Material Status Description",
#     "Plant Material Status Description",
#     "Auth Sell Flag Description",
#     "Plant",
#     "Order Quantity Sales Unit",
#     "Sales Unit of Measure"
# ]

# # Column Metadata - using EXACT names from CSV
# COLUMNS = {
#     "Sales Order Number": {
#         "type": "text",
#         "description": "Sales order ID/number",
#         "filterable": True,
#         "aggregatable": False,
#         "common_use": "Identifying specific orders"
#     },
#     "Sales Document Type": {
#         "type": "text",
#         "description": "Type of sales document",
#         "filterable": True,
#         "aggregatable": True,
#         "common_use": "Grouping by document type"
#     },
#     "Sales Order Line Item": {
#         "type": "text",
#         "description": "Line item number within sales order",
#         "filterable": True,
#         "aggregatable": False,
#         "common_use": "Detailed line item analysis"
#     },
#     "Order Create Date": {
#         "type": "date",
#         "description": "Date when order was created",
#         "filterable": True,
#         "aggregatable": False,
#         "common_use": "Time-based filtering"
#     },
#     "Requested Delivery Date": {
#         "type": "date",
#         "description": "Requested delivery date",
#         "filterable": True,
#         "aggregatable": False,
#         "common_use": "Delivery timeline analysis"
#     },
#     "Sold-To Party": {
#         "type": "text",
#         "description": "Customer ID/account number",
#         "filterable": True,
#         "aggregatable": True,
#         "common_use": "Customer-specific analysis"
#     },
#     "Sold-to Name": {
#         "type": "text",
#         "description": "Customer name",
#         "filterable": True,
#         "aggregatable": False,
#         "common_use": "Customer identification"
#     },
#     "Customer Hierarchy ": {  # Note: trailing space
#         "type": "text",
#         "description": "Customer group/hierarchy",
#         "filterable": True,
#         "aggregatable": True,
#         "common_use": "Customer hierarchy grouping"
#     },
#     "Customer Hierarchy Level 6 Text": {
#         "type": "text",
#         "description": "Customer hierarchy level 6",
#         "filterable": True,
#         "aggregatable": True,
#         "common_use": "Detailed hierarchy analysis"
#     },
#     "Customer Hierarchy Level 7 Text": {
#         "type": "text",
#         "description": "Customer hierarchy level 7",
#         "filterable": True,
#         "aggregatable": True,
#         "common_use": "Detailed hierarchy analysis"
#     },
#     "Material": {
#         "type": "text",
#         "description": "Material number/code",
#         "filterable": True,
#         "aggregatable": True,
#         "common_use": "Material-specific analysis, top materials"
#     },
#     "Material Descrption": {  # Note: typo in actual data
#         "type": "text",
#         "description": "Material name/description (note: spelling in source data)",
#         "filterable": True,
#         "aggregatable": False,
#         "common_use": "Material identification, reporting"
#     },
#     "Material Status Description": {
#         "type": "text",
#         "description": "Material status (Active, Inactive, etc.)",
#         "filterable": True,
#         "aggregatable": True,
#         "common_use": "Filtering by material status"
#     },
#     "Plant Material Status Description": {
#         "type": "text",
#         "description": "Plant-specific material status",
#         "filterable": True,
#         "aggregatable": True,
#         "common_use": "Plant-level material status analysis"
#     },
#     "Auth Sell Flag Description": {
#         "type": "text",
#         "description": "Authorization flag: 'Yes' (authorized) or 'No' (not authorized)",
#         "filterable": True,
#         "aggregatable": True,
#         "common_use": "Authorization status filtering, auth rate calculations"
#     },
#     "Plant": {
#         "type": "text",
#         "description": "Plant/Location code (e.g., '1007', '7001')",
#         "filterable": True,
#         "aggregatable": True,
#         "common_use": "Plant-specific analysis, location filtering"
#     },
#     "Order Quantity Sales Unit": {
#         "type": "numeric",
#         "description": "Quantity ordered in sales units",
#         "filterable": True,
#         "aggregatable": True,
#         "common_use": "Quantity summation, volume analysis"
#     },
#     "Sales Unit of Measure": {
#         "type": "text",
#         "description": "Unit of measure (EA, CS, KG, etc.)",
#         "filterable": True,
#         "aggregatable": True,
#         "common_use": "Unit-based grouping"
#     }
# }


# ==============================================================================
# UTILITY FUNCTIONS (Updated for New Schema)
# ==============================================================================

def get_all_sales_order_columns():
    """Get list of all Sales Order Exception Report column names"""
    return SALES_ORDER_EXCEPTION_COLUMNS.copy()


def get_all_location_sequence_columns():
    """Get list of all A1P Location Sequence column names"""
    return A1P_LOCATION_SEQUENCE_COLUMNS.copy()


def get_foreign_key_relationships():
    """Get foreign key relationship definitions"""
    return FOREIGN_KEY_RELATIONSHIPS


def get_common_columns():
    """Get columns that exist in both tables (for joining)"""
    return ["Plant", "Material"]


def generate_relationship_diagram():
    """Generate ASCII diagram showing table relationships"""
    diagram = """
    ┌─────────────────────────────────────┐
    │  Sales Order Exception Report       │
    │─────────────────────────────────────│
    │  * Sales Order Number (PK)          │
    │    Plant (FK) ────────────┐         │
    │    Material (FK) ─────────┼────┐    │
    │    ... other columns      │    │    │
    └───────────────────────────┘    │    │
                                     │    │
    ┌────────────────────────────────┼────┼───┐
    │  A1P Location Sequence         │    │   │
    │────────────────────────────────┼────┼───│
    │  * Plant(Location) (PK) ◄──────┘    │   │
    │  * Material (PK) ◄──────────────────┘   │
    │    Inven Id                              │
    │    ... other columns                     │
    └──────────────────────────────────────────┘
    
    Relationship:
    - Plant (Sales Order) → Plant(Location) (A1P Location)
    - Material (Sales Order) → Material (A1P Location)
    - Composite Foreign Key: (Plant, Material)
    """
    return diagram


# ==============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# ==============================================================================

def get_all_columns():
    """Get list of all column names (backward compatible - returns Sales Order columns)"""
    return SALES_ORDER_EXCEPTION_COLUMNS.copy()


def get_filterable_columns():
    """Get columns that can be used for filtering (backward compatible)"""
    # Return commonly filterable columns from Sales Order table
    return [col for col in SALES_ORDER_EXCEPTION_COLUMNS 
            if col in ["Plant", "Material", "Sales Document Type", "Sold-To Party", 
                      "Material Status Description", "Auth Sell Flag Description",
                      "Order Create Date", "Requested Delivery Date"]]


def get_aggregatable_columns():
    """Get columns that can be aggregated (backward compatible)"""
    # Return commonly aggregatable columns
    return ["Order Quantity Sales Unit", "Order Quantity (BU)", "Order Quantity (CS)",
            "Net Value Doc Currency", "Net Price Doc Currency", "Unique Order Qty (SU)",
            "Available Stock (F)"]


def generate_schema_prompt():
    """Generate detailed schema description for LLM prompt"""
    prompt_parts = []
    
    # Sales Order Exception Report Schema
    prompt_parts.append("=" * 80)
    prompt_parts.append("TABLE 1: Sales Order Exception Report")
    prompt_parts.append("=" * 80)
    for col in SALES_ORDER_EXCEPTION_COLUMNS:
        col_info = SALES_ORDER_EXCEPTION_SCHEMA.get(col, {})
        fk = " [FOREIGN KEY]" if col_info.get("foreign_key") else ""
        pk = " [PRIMARY KEY]" if col_info.get("primary_key") else ""
        prompt_parts.append(f"\n• {col}{pk}{fk}")
        prompt_parts.append(f"  Type: {col_info.get('type', 'text')}")
        prompt_parts.append(f"  Description: {col_info.get('description', 'N/A')}")
    
    # A1P Location Sequence Schema
    prompt_parts.append("\n\n" + "=" * 80)
    prompt_parts.append("TABLE 2: A1P Location Sequence Export")
    prompt_parts.append("=" * 80)
    for col in A1P_LOCATION_SEQUENCE_COLUMNS:
        col_info = A1P_LOCATION_SEQUENCE_SCHEMA.get(col, {})
        pk = " [PRIMARY KEY]" if col_info.get("primary_key") else ""
        fk_target = " [FK TARGET]" if col_info.get("foreign_key_target") else ""
        prompt_parts.append(f"\n• {col}{pk}{fk_target}")
        prompt_parts.append(f"  Type: {col_info.get('type', 'text')}")
        prompt_parts.append(f"  Description: {col_info.get('description', 'N/A')}")
    
    # Relationship Information
    prompt_parts.append("\n\n" + "=" * 80)
    prompt_parts.append("FOREIGN KEY RELATIONSHIPS")
    prompt_parts.append("=" * 80)
    prompt_parts.append(generate_relationship_diagram())
    
    return "\n".join(prompt_parts)


def get_common_filters():
    """Get most commonly used filter columns (backward compatible)"""
    return {
        "Plant": "Plant/Location filtering",
        "Material": "Material-specific filtering",
        "Auth Sell Flag Description": "Authorization status filtering",
        "Material Status Description": "Material status filtering",
        "Sold-To Party": "Customer-specific filtering"
    }


def get_common_chart_columns():
    """Get columns commonly used in visualizations (backward compatible)"""
    return {
        "for_bar_charts": ["Plant", "Material", "Material Status Description", 
                          "Auth Sell Flag Description", "Sales Document Type"],
        "for_pie_charts": ["Auth Sell Flag Description", "Material Status Description", 
                          "Sales Unit of Measure"],
        "for_aggregation": ["Order Quantity Sales Unit", "Order Quantity (CS)", 
                           "Net Value Doc Currency"],
        "for_details": ["Material", "Material Descrption", "Plant", "Sales Order Number", 
                       "Order Quantity Sales Unit"]
    }


def get_column_type(column_name, table="sales_order"):
    """Get the data type of a column
    
    Args:
        column_name: Name of the column
        table: Either "sales_order" or "location_sequence"
    """
    if table == "sales_order":
        return SALES_ORDER_EXCEPTION_SCHEMA.get(column_name, {}).get("type", "text")
    elif table == "location_sequence":
        return A1P_LOCATION_SEQUENCE_SCHEMA.get(column_name, {}).get("type", "text")
    return "text"


def validate_column_name(column_name, table="sales_order"):
    """Check if a column name exists in the schema
    
    Args:
        column_name: Name of the column to validate
        table: Either "sales_order" or "location_sequence"
    """
    if table == "sales_order":
        return column_name in SALES_ORDER_EXCEPTION_COLUMNS
    elif table == "location_sequence":
        return column_name in A1P_LOCATION_SEQUENCE_COLUMNS
    return False

