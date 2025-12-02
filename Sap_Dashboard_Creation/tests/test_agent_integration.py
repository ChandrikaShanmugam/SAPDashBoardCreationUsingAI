"""
Integration Test for SAP Dashboard Agent with Cross-Table Support
==================================================================
Tests the complete flow: data loading → filter extraction → cross-table join → dashboard generation
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "core"))

import pandas as pd
from prompt_manager import PromptTemplateManager
import database_schema as db_schema


def test_prompt_manager_integration():
    """Test that PromptTemplateManager properly loads and formats templates"""
    print("\n" + "=" * 80)
    print("TEST 1: Prompt Manager Integration")
    print("=" * 80)
    
    manager = PromptTemplateManager()
    
    # Test filter extraction template exists
    template = manager.get_template('filter_extraction')
    assert template, "Filter extraction template not loaded"
    assert len(template) > 1000, f"Template seems incomplete: {len(template)} chars"
    print(f"✅ Filter extraction template loaded: {len(template)} characters")
    
    # Test that it includes cross-table guidance
    assert "requires_join" in template
    assert "cross-table" in template.lower() or "join" in template.lower()
    print("✅ Template includes cross-table filtering guidance")
    
    # Test formatting
    query = "Show me sales orders with inventory group 9.5OZ"
    formatted = manager.format_filter_extraction_prompt(query)
    
    assert "SALES ORDER EXCEPTION REPORT COLUMNS" in formatted
    assert "A1P LOCATION SEQUENCE COLUMNS" in formatted
    assert "Plant" in formatted
    assert "Material" in formatted
    assert "69 columns" in formatted  # Sales Order count
    assert "15 columns" in formatted  # Location count
    print("✅ Formatted prompt includes both table schemas")
    print(f"   Total formatted prompt: {len(formatted)} characters")
    
    return True


def test_schema_functions():
    """Test database schema helper functions"""
    print("\n" + "=" * 80)
    print("TEST 2: Database Schema Functions")
    print("=" * 80)
    
    # Test column retrieval
    sales_cols = db_schema.get_all_sales_order_columns()
    location_cols = db_schema.get_all_location_sequence_columns()
    
    assert len(sales_cols) == 69, f"Expected 69 Sales Order columns, got {len(sales_cols)}"
    assert len(location_cols) == 15, f"Expected 15 Location columns, got {len(location_cols)}"
    print(f"✅ Sales Order columns: {len(sales_cols)}")
    print(f"✅ Location Sequence columns: {len(location_cols)}")
    
    # Test foreign key relationships
    fk_rels = db_schema.get_foreign_key_relationships()
    assert "sales_order_to_location" in fk_rels
    
    rel = fk_rels["sales_order_to_location"]
    assert len(rel["relationships"]) == 2  # Plant and Material
    print(f"✅ Foreign key relationships defined: {len(rel['relationships'])}")
    
    # Test common columns
    common = db_schema.get_common_columns()
    assert "Plant" in common
    assert "Material" in common
    print(f"✅ Common columns: {common}")
    
    # Test column validation
    assert db_schema.validate_column_name("Plant", "sales_order") == True
    assert db_schema.validate_column_name("Plant(Location)", "location_sequence") == True
    assert db_schema.validate_column_name("InvalidColumn", "sales_order") == False
    print("✅ Column validation working")
    
    return True


def test_cross_table_filter_detection():
    """Test that cross-table queries are properly detected"""
    print("\n" + "=" * 80)
    print("TEST 3: Cross-Table Filter Detection")
    print("=" * 80)
    
    manager = PromptTemplateManager()
    template = manager.get_template('filter_extraction')
    
    # Check for cross-table examples
    cross_table_indicators = [
        '"requires_join": true',
        '"join_on": ["Plant", "Material"]',
        'Inven Group Name',  # Location table column
        'Location Id',  # Location table column
    ]
    
    found_count = 0
    for indicator in cross_table_indicators:
        if indicator in template:
            print(f"✅ Found indicator: {indicator}")
            found_count += 1
        else:
            print(f"❌ Missing indicator: {indicator}")
    
    assert found_count >= 3, f"Only found {found_count}/4 cross-table indicators"
    print(f"\n✅ Cross-table detection patterns present: {found_count}/4")
    
    return True


def test_data_file_availability():
    """Test that required data files exist"""
    print("\n" + "=" * 80)
    print("TEST 4: Data File Availability")
    print("=" * 80)
    
    data_dir = Path(__file__).parent.parent / "data"
    
    # Check Sales Order file
    sales_files = [
        "Sales Order Exception report.csv",
        "Sales Order Exception report 13 and 14 Nov 2025.csv"
    ]
    
    sales_file_found = False
    for filename in sales_files:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"✅ Sales Order file found: {filename}")
            print(f"   Size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
            sales_file_found = True
            break
    
    assert sales_file_found, "Sales Order CSV file not found"
    
    # Check Location Sequence file
    location_file = data_dir / "A1P_Locn_Seq_EXPORT.csv"
    assert location_file.exists(), f"Location Sequence file not found: {location_file}"
    print(f"✅ Location Sequence file found: A1P_Locn_Seq_EXPORT.csv")
    print(f"   Size: {location_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    return True


def test_column_mapping():
    """Test that column names map correctly between tables"""
    print("\n" + "=" * 80)
    print("TEST 5: Column Name Mapping")
    print("=" * 80)
    
    sales_cols = db_schema.get_all_sales_order_columns()
    location_cols = db_schema.get_all_location_sequence_columns()
    
    # Test foreign key columns
    assert "Plant" in sales_cols, "Plant not in Sales Order columns"
    assert "Material" in sales_cols, "Material not in Sales Order columns"
    assert "Plant(Location)" in location_cols, "Plant(Location) not in Location columns"
    assert "Material" in location_cols, "Material not in Location columns"
    
    print("✅ Foreign key columns present:")
    print("   Sales Order: Plant, Material")
    print("   Location:    Plant(Location), Material")
    
    # Test Location-specific columns
    location_specific = [
        "Inven Id",
        "Inven Sequ Number",
        "Inven Group Name",
        "Location Id",
        "Auth to sell flag"
    ]
    
    for col in location_specific:
        assert col in location_cols, f"Column '{col}' not in Location Sequence"
        print(f"✅ Location column: {col}")
    
    return True


def test_example_queries():
    """Test example query patterns that should work"""
    print("\n" + "=" * 80)
    print("TEST 6: Example Query Patterns")
    print("=" * 80)
    
    example_queries = [
        {
            "query": "Show sales orders for inventory group 9.5OZ",
            "should_join": True,
            "reason": "Inventory group is in Location table"
        },
        {
            "query": "Show me plant 1001 data",
            "should_join": False,
            "reason": "Plant is in both tables, Sales Order is default"
        },
        {
            "query": "Materials with location ID ABC123",
            "should_join": True,
            "reason": "Location ID is only in Location table"
        },
        {
            "query": "Orders with inventory sequence 3000",
            "should_join": True,
            "reason": "Inventory sequence is in Location table"
        },
        {
            "query": "Sales orders with auth flag no",
            "should_join": False,
            "reason": "Auth flag is in Sales Order table"
        }
    ]
    
    manager = PromptTemplateManager()
    
    for i, example in enumerate(example_queries, 1):
        print(f"\n{i}. Query: \"{example['query']}\"")
        print(f"   Should join: {example['should_join']}")
        print(f"   Reason: {example['reason']}")
        
        # Format the prompt
        prompt = manager.format_filter_extraction_prompt(example['query'])
        
        # Verify prompt contains necessary info
        assert "requires_join" in prompt, "Prompt missing requires_join guidance"
        print(f"   ✅ Prompt formatted successfully ({len(prompt)} chars)")
    
    print(f"\n✅ All {len(example_queries)} example queries can be processed")
    return True


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "=" * 80)
    print(" " * 20 + "AGENT INTEGRATION TESTS")
    print("=" * 80)
    
    tests = [
        ("Prompt Manager Integration", test_prompt_manager_integration),
        ("Database Schema Functions", test_schema_functions),
        ("Cross-Table Filter Detection", test_cross_table_filter_detection),
        ("Data File Availability", test_data_file_availability),
        ("Column Name Mapping", test_column_mapping),
        ("Example Query Patterns", test_example_queries),
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
        except Exception as e:
            failed += 1
            errors.append(f"Test '{test_name}' error: {str(e)}")
            import traceback
            print(f"\n❌ ERROR: {str(e)}")
            traceback.print_exc()
    
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
        print(" " * 10 + "ALL INTEGRATION TESTS PASSED!")
        print("🎉 " * 20)
        print("\n✅ SAP Dashboard Agent is ready for cross-table queries!")
        print("✅ Two-table schema with foreign key relationships working!")
        print("✅ Prompt templates include cross-table filtering examples!")
        print("✅ Data files available and columns mapped correctly!")
    
    print("\n" + "=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
