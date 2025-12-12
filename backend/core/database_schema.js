/**
 * Database Schema Configuration
 * ==============================
 * Centralized configuration for SAP Exception Report and A1P Location Sequence tables.
 * EXACT column names from CSV - hardcoded for reliability!
 */

// ==============================================================================
// NEW SCHEMA - TWO TABLES WITH FOREIGN KEY RELATIONSHIPS
// ==============================================================================

// Table 1: Sales Order Exception Report
const SALES_ORDER_EXCEPTION_COLUMNS = [
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
];

// Table 2: A1P Location Sequence Export
const A1P_LOCATION_SEQUENCE_COLUMNS = [
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
];

// Table 3: COF Inventory Net Price
const COF_INVEN_NET_PRICE_COLUMNS = [
    "Sold-To Party",
    "Sold-to Name",
    "COF",
    "Pespsi Invenid",
    "UPC",
    "Material",
    "Material Descrption"
];

// Table 4: COF Inventory Net Price Material (Material Pricing Data)
const COF_INVEN_NETPRICE_MATERIAL_COLUMNS = [
    "Sales Organization",
    "Distribution Channel",
    "Condition Type",
    "Condition Record No.",
    "Material",
    "Condition Amount",
    "Condition Currency",
    "Pricing Unit",
    "Unit of Measure",
    "Valid From",
    "Valid To"
];

// Foreign Key Relationships
const FOREIGN_KEY_RELATIONSHIPS = {
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
    },
    "sales_order_to_cof_inventory": {
        "from_table": "Sales Order Exception Report",
        "to_table": "COF Inventory Net Price",
        "relationships": [
            {
                "from_column": "Sold-To Party",
                "to_column": "Sold-To Party",
                "relationship_type": "many-to-one",
                "description": "Links sales orders to customer COF inventory pricing"
            },
            {
                "from_column": "Material",
                "to_column": "Material",
                "relationship_type": "many-to-one",
                "description": "Links sales order materials to COF inventory pricing"
            },
            {
                "from_column": "UPC",
                "to_column": "UPC",
                "relationship_type": "many-to-one",
                "description": "Links via UPC code to COF inventory pricing"
            },
            {
                "from_column": "Pespsi Invenid",
                "to_column": "Pespsi Invenid",
                "relationship_type": "many-to-one",
                "description": "Links via PepsiCo Inventory ID to COF pricing"
            }
        ],
        "composite_key": ["Sold-To Party", "Material"],
        "description": "Sales orders can be linked to COF inventory pricing via customer and material combination"
    },
    "cof_inventory_to_material": {
        "from_table": "COF Inventory Net Price",
        "to_table": "COF Inventory Net Price Material",
        "relationships": [
            {
                "from_column": "Material",
                "to_column": "Material",
                "relationship_type": "many-to-one",
                "description": "Material reference for pricing lookup"
            }
        ],
        "composite_key": ["Sales Organization", "Distribution Channel", "Material"],
        "description": "COF Inventory links to Material Pricing via Material number for condition amounts and pricing units"
    }
};

// Table Metadata for Sales Order Exception Report
const SALES_ORDER_EXCEPTION_SCHEMA = {
    "Sales Order Number": {"type": "text", "primary_key": true, "description": "Unique sales order identifier"},
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
    "Material": {"type": "text", "foreign_key": true, "description": "Material number/code (may have leading zeros, e.g., 000000000300008760 or 300008760)"},
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
    "Plant": {"type": "text", "foreign_key": true, "description": "Plant/Location code"},
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
};

// Functions
function get_all_sales_order_columns() {
    return SALES_ORDER_EXCEPTION_COLUMNS;
}

function get_all_location_sequence_columns() {
    return A1P_LOCATION_SEQUENCE_COLUMNS;
}

function get_all_cof_inventory_columns() {
    return COF_INVEN_NET_PRICE_COLUMNS;
}

function get_all_cof_material_columns() {
    return COF_INVEN_NETPRICE_MATERIAL_COLUMNS;
}

function get_foreign_key_relationships() {
    return FOREIGN_KEY_RELATIONSHIPS;
}

function get_text_columns(table = "sales_order") {
    const schema = table === "sales_order" ? SALES_ORDER_EXCEPTION_SCHEMA : {};
    return Object.keys(schema).filter(column => schema[column].type === "text");
}

function normalize_material_number(material) {
    if (!material) return "";
    // Remove leading zeros
    return material.toString().replace(/^0+/, '');
}

function materials_match(material1, material2) {
    if (!material1 || !material2) return false;
    const norm1 = normalize_material_number(material1);
    const norm2 = normalize_material_number(material2);
    return norm1 === norm2;
}

module.exports = {
    SALES_ORDER_EXCEPTION_COLUMNS,
    A1P_LOCATION_SEQUENCE_COLUMNS,
    COF_INVEN_NET_PRICE_COLUMNS,
    COF_INVEN_NETPRICE_MATERIAL_COLUMNS,
    FOREIGN_KEY_RELATIONSHIPS,
    SALES_ORDER_EXCEPTION_SCHEMA,
    get_all_sales_order_columns,
    get_all_location_sequence_columns,
    get_all_cof_inventory_columns,
    get_all_cof_material_columns,
    get_foreign_key_relationships,
    get_text_columns,
    normalize_material_number,
    materials_match
};