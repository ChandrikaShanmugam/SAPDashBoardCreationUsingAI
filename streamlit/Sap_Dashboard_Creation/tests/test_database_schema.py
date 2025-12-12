"""
Comprehensive Test Suite for Database Schema
=============================================
Tests schema access, retrieval, validation, and all utility functions.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "core"))

import database_schema as ds


def test_table_columns():
    """Test retrieving column lists from both tables"""
    print("\n" + "=" * 70)
    print("TEST 1: Table Columns Access")
    print("=" * 70)
    
    # Test Sales Order columns
    sales_cols = ds.get_all_sales_order_columns()
    assert isinstance(sales_cols, list), "Sales columns should be a list"
    assert len(sales_cols) == 69, f"Expected 69 columns, got {len(sales_cols)}"
    print(f"✅ Sales Order Table: {len(sales_cols)} columns")
    print(f"   Sample columns: {sales_cols[:3]}")
    
    # Test Location Sequence columns
    loc_cols = ds.get_all_location_sequence_columns()
    assert isinstance(loc_cols, list), "Location columns should be a list"
    assert len(loc_cols) == 15, f"Expected 15 columns, got {len(loc_cols)}"
    print(f"✅ Location Sequence Table: {len(loc_cols)} columns")
    print(f"   Sample columns: {loc_cols[:3]}")
    
    # Test backward compatibility
    all_cols = ds.get_all_columns()
    assert all_cols == sales_cols, "get_all_columns() should return sales order columns"
    print(f"✅ Backward compatibility: get_all_columns() works")
    
    return True


def test_foreign_key_relationships():
    """Test foreign key relationship definitions"""
    print("\n" + "=" * 70)
    print("TEST 2: Foreign Key Relationships")
    print("=" * 70)
    
    fk = ds.get_foreign_key_relationships()
    assert isinstance(fk, dict), "FK relationships should be a dictionary"
    assert "sales_order_to_location" in fk, "Should have sales_order_to_location key"
    
    relationship = fk["sales_order_to_location"]
    assert relationship["from_table"] == "Sales Order Exception Report"
    assert relationship["to_table"] == "A1P Location Sequence"
    assert len(relationship["relationships"]) == 2, "Should have 2 FK relationships"
    
    print(f"✅ From Table: {relationship['from_table']}")
    print(f"✅ To Table: {relationship['to_table']}")
    print(f"✅ Number of FK relationships: {len(relationship['relationships'])}")
    
    for rel in relationship["relationships"]:
        print(f"   - {rel['from_column']} → {rel['to_column']} ({rel['relationship_type']})")
    
    # Test composite key
    composite = relationship["composite_key"]
    assert composite == ["Plant", "Material"], "Composite key should be Plant and Material"
    print(f"✅ Composite Key: {composite}")
    
    return True


def test_common_columns():
    """Test common columns between tables"""
    print("\n" + "=" * 70)
    print("TEST 3: Common Columns")
    print("=" * 70)
    
    common = ds.get_common_columns()
    assert isinstance(common, list), "Common columns should be a list"
    assert "Plant" in common, "Plant should be a common column"
    assert "Material" in common, "Material should be a common column"
    assert len(common) == 2, "Should have 2 common columns"
    
    print(f"✅ Common columns: {common}")
    print("   These columns can be used for JOINs between tables")
    
    return True


def test_schema_validation():
    """Test column name validation"""
    print("\n" + "=" * 70)
    print("TEST 4: Schema Validation")
    print("=" * 70)
    
    # Test valid columns in Sales Order table
    valid_sales = [
        "Sales Order Number",
        "Plant",
        "Material",
        "Order Quantity Sales Unit",
        "Material Descrption"  # Note the typo - this is correct!
    ]
    
    for col in valid_sales:
        result = ds.validate_column_name(col, "sales_order")
        assert result == True, f"Column '{col}' should be valid in sales_order"
        print(f"✅ Valid in Sales Order: '{col}'")
    
    # Test valid columns in Location Sequence table
    valid_location = [
        "Plant(Location)",
        "Material",
        "Inven Id",
        "Auth to sell flag"
    ]
    
    for col in valid_location:
        result = ds.validate_column_name(col, "location_sequence")
        assert result == True, f"Column '{col}' should be valid in location_sequence"
        print(f"✅ Valid in Location: '{col}'")
    
    # Test invalid columns
    invalid_col = "NonExistentColumn"
    result = ds.validate_column_name(invalid_col, "sales_order")
    assert result == False, f"Column '{invalid_col}' should be invalid"
    print(f"✅ Invalid column correctly rejected: '{invalid_col}'")
    
    return True


def test_column_types():
    """Test column data type retrieval"""
    print("\n" + "=" * 70)
    print("TEST 5: Column Data Types")
    print("=" * 70)
    
    # Test Sales Order column types
    type_tests = [
        ("Sales Order Number", "sales_order", "text"),
        ("Order Quantity Sales Unit", "sales_order", "numeric"),
        ("Order Create Date", "sales_order", "date"),
        ("Created Time", "sales_order", "time"),
        ("Year", "sales_order", "integer"),
    ]
    
    for col, table, expected_type in type_tests:
        actual_type = ds.get_column_type(col, table)
        assert actual_type == expected_type, f"{col} should be {expected_type}, got {actual_type}"
        print(f"✅ {col}: {actual_type}")
    
    # Test Location Sequence column types
    loc_type_tests = [
        ("Plant(Location)", "location_sequence", "text"),
        ("Inven Sequ Number", "location_sequence", "integer"),
        ("Created on", "location_sequence", "date"),
    ]
    
    for col, table, expected_type in loc_type_tests:
        actual_type = ds.get_column_type(col, table)
        assert actual_type == expected_type, f"{col} should be {expected_type}, got {actual_type}"
        print(f"✅ {col}: {actual_type}")
    
    return True


def test_schema_metadata():
    """Test schema metadata access"""
    print("\n" + "=" * 70)
    print("TEST 6: Schema Metadata")
    print("=" * 70)
    
    # Test Sales Order schema
    so_schema = ds.SALES_ORDER_EXCEPTION_SCHEMA
    assert isinstance(so_schema, dict), "Sales Order schema should be a dictionary"
    assert len(so_schema) == 69, f"Expected 69 entries, got {len(so_schema)}"
    
    # Test primary key
    assert so_schema["Sales Order Number"]["primary_key"] == True
    print(f"✅ Primary Key in Sales Order: Sales Order Number")
    
    # Test foreign keys
    assert so_schema["Plant"]["foreign_key"] == True
    assert so_schema["Material"]["foreign_key"] == True
    print(f"✅ Foreign Keys in Sales Order: Plant, Material")
    
    # Test Location Sequence schema
    loc_schema = ds.A1P_LOCATION_SEQUENCE_SCHEMA
    assert isinstance(loc_schema, dict), "Location schema should be a dictionary"
    assert len(loc_schema) == 15, f"Expected 15 entries, got {len(loc_schema)}"
    
    # Test composite primary key
    assert loc_schema["Plant(Location)"]["primary_key"] == True
    assert loc_schema["Material"]["primary_key"] == True
    assert loc_schema["Plant(Location)"]["foreign_key_target"] == True
    assert loc_schema["Material"]["foreign_key_target"] == True
    print(f"✅ Composite Primary Key in Location: Plant(Location), Material")
    
    return True


def test_filterable_columns():
    """Test filterable columns retrieval"""
    print("\n" + "=" * 70)
    print("TEST 7: Filterable Columns")
    print("=" * 70)
    
    filterable = ds.get_filterable_columns()
    assert isinstance(filterable, list), "Filterable columns should be a list"
    assert len(filterable) > 0, "Should have some filterable columns"
    
    # Check key filterable columns are present
    expected_filterable = ["Plant", "Material", "Sales Document Type"]
    for col in expected_filterable:
        assert col in filterable, f"'{col}' should be filterable"
    
    print(f"✅ Found {len(filterable)} filterable columns")
    print(f"   Sample: {filterable[:5]}")
    
    return True


def test_aggregatable_columns():
    """Test aggregatable columns retrieval"""
    print("\n" + "=" * 70)
    print("TEST 8: Aggregatable Columns")
    print("=" * 70)
    
    aggregatable = ds.get_aggregatable_columns()
    assert isinstance(aggregatable, list), "Aggregatable columns should be a list"
    assert len(aggregatable) > 0, "Should have some aggregatable columns"
    
    # Check key aggregatable columns are present
    expected_agg = ["Order Quantity Sales Unit", "Order Quantity (CS)"]
    for col in expected_agg:
        assert col in aggregatable, f"'{col}' should be aggregatable"
    
    print(f"✅ Found {len(aggregatable)} aggregatable columns")
    print(f"   Columns: {aggregatable}")
    
    return True


def test_common_filters():
    """Test common filter definitions"""
    print("\n" + "=" * 70)
    print("TEST 9: Common Filters")
    print("=" * 70)
    
    filters = ds.get_common_filters()
    assert isinstance(filters, dict), "Common filters should be a dictionary"
    
    expected_filters = ["Plant", "Material", "Auth Sell Flag Description"]
    for filt in expected_filters:
        assert filt in filters, f"'{filt}' should be in common filters"
    
    print(f"✅ Found {len(filters)} common filters:")
    for key, desc in filters.items():
        print(f"   - {key}: {desc}")
    
    return True


def test_chart_columns():
    """Test chart column recommendations"""
    print("\n" + "=" * 70)
    print("TEST 10: Chart Columns")
    print("=" * 70)
    
    chart_cols = ds.get_common_chart_columns()
    assert isinstance(chart_cols, dict), "Chart columns should be a dictionary"
    
    required_keys = ["for_bar_charts", "for_pie_charts", "for_aggregation", "for_details"]
    for key in required_keys:
        assert key in chart_cols, f"'{key}' should be in chart columns"
        assert isinstance(chart_cols[key], list), f"'{key}' should be a list"
    
    print(f"✅ Chart column categories:")
    for category, cols in chart_cols.items():
        print(f"   - {category}: {len(cols)} columns")
        print(f"     {cols[:3]}...")
    
    return True


def test_relationship_diagram():
    """Test relationship diagram generation"""
    print("\n" + "=" * 70)
    print("TEST 11: Relationship Diagram")
    print("=" * 70)
    
    diagram = ds.generate_relationship_diagram()
    assert isinstance(diagram, str), "Diagram should be a string"
    assert "Sales Order Exception Report" in diagram
    assert "A1P Location Sequence" in diagram
    assert "Plant" in diagram
    assert "Material" in diagram
    
    print("✅ Relationship diagram generated successfully:")
    print(diagram)
    
    return True


def test_schema_prompt():
    """Test schema prompt generation for LLM"""
    print("\n" + "=" * 70)
    print("TEST 12: Schema Prompt Generation")
    print("=" * 70)
    
    prompt = ds.generate_schema_prompt()
    assert isinstance(prompt, str), "Schema prompt should be a string"
    assert len(prompt) > 1000, "Prompt should be comprehensive"
    assert "TABLE 1: Sales Order Exception Report" in prompt
    assert "TABLE 2: A1P Location Sequence Export" in prompt
    assert "FOREIGN KEY RELATIONSHIPS" in prompt
    
    print(f"✅ Schema prompt generated: {len(prompt)} characters")
    print(f"   Contains table definitions and relationships")
    
    # Show first 500 characters
    print(f"\n   Preview (first 500 chars):")
    print(f"   {prompt[:500]}...")
    
    return True


def test_column_constants():
    """Test that column constant lists match schema dictionaries"""
    print("\n" + "=" * 70)
    print("TEST 13: Column Constants vs Schema Consistency")
    print("=" * 70)
    
    # Check Sales Order
    sales_cols = ds.SALES_ORDER_EXCEPTION_COLUMNS
    sales_schema = ds.SALES_ORDER_EXCEPTION_SCHEMA
    
    for col in sales_cols:
        assert col in sales_schema, f"Column '{col}' not in SALES_ORDER_EXCEPTION_SCHEMA"
    
    print(f"✅ All {len(sales_cols)} Sales Order columns have schema entries")
    
    # Check Location Sequence
    loc_cols = ds.A1P_LOCATION_SEQUENCE_COLUMNS
    loc_schema = ds.A1P_LOCATION_SEQUENCE_SCHEMA
    
    for col in loc_cols:
        assert col in loc_schema, f"Column '{col}' not in A1P_LOCATION_SEQUENCE_SCHEMA"
    
    print(f"✅ All {len(loc_cols)} Location Sequence columns have schema entries")
    
    return True


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "=" * 70)
    print("TEST 14: Edge Cases")
    print("=" * 70)
    
    # Test with empty/None/invalid inputs
    result = ds.validate_column_name("", "sales_order")
    assert result == False, "Empty string should be invalid"
    print("✅ Empty string correctly rejected")
    
    # Test with wrong table parameter
    result = ds.get_column_type("Plant", "invalid_table")
    assert result == "text", "Invalid table should return default type"
    print("✅ Invalid table returns default type")
    
    # Test column name with trailing space (known issue in data)
    result = ds.validate_column_name("Customer Hierarchy ", "sales_order")
    assert result == True, "Column with trailing space should be valid"
    print("✅ Column with trailing space handled correctly")
    
    # Test column name with typo (known in data)
    result = ds.validate_column_name("Material Descrption", "sales_order")
    assert result == True, "Column with typo should be valid (as per source data)"
    print("✅ Column with typo (from source) handled correctly")
    
    return True


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "=" * 80)
    print(" " * 20 + "DATABASE SCHEMA TEST SUITE")
    print("=" * 80)
    print(f"Testing schema from: {Path(__file__).parent.parent / 'src' / 'core' / 'database_schema.py'}")
    
    tests = [
        ("Table Columns Access", test_table_columns),
        ("Foreign Key Relationships", test_foreign_key_relationships),
        ("Common Columns", test_common_columns),
        ("Schema Validation", test_schema_validation),
        ("Column Data Types", test_column_types),
        ("Schema Metadata", test_schema_metadata),
        ("Filterable Columns", test_filterable_columns),
        ("Aggregatable Columns", test_aggregatable_columns),
        ("Common Filters", test_common_filters),
        ("Chart Columns", test_chart_columns),
        ("Relationship Diagram", test_relationship_diagram),
        ("Schema Prompt", test_schema_prompt),
        ("Column Constants Consistency", test_column_constants),
        ("Edge Cases", test_edge_cases),
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                errors.append(f"Test '{test_name}' returned False")
        except AssertionError as e:
            failed += 1
            errors.append(f"Test '{test_name}' failed: {str(e)}")
        except Exception as e:
            failed += 1
            errors.append(f"Test '{test_name}' error: {str(e)}")
    
    # Summary
    print("\n" + "=" * 80)
    print(" " * 30 + "TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if errors:
        print("\n" + "=" * 80)
        print("ERRORS:")
        for error in errors:
            print(f"  ❌ {error}")
    else:
        print("\n" + "🎉 " * 20)
        print(" " * 20 + "ALL TESTS PASSED!")
        print("🎉 " * 20)
    
    print("\n" + "=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
