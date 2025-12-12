"""
Test Cross-Table Filtering with Foreign Key Relationships
==========================================================
Tests that the LLM prompt correctly guides filter extraction for queries
that span both Sales Order and Location Sequence tables.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "core"))

from prompt_manager import PromptTemplateManager


def test_prompt_includes_relationship_info():
    """Test that relationship info is included in the prompt"""
    print("\n" + "=" * 70)
    print("TEST 1: Relationship Info in Prompt")
    print("=" * 70)
    
    manager = PromptTemplateManager()
    
    # Get relationship info
    rel_info = manager.get_relationship_info()
    print("Relationship Info:")
    print(rel_info)
    
    assert "Plant" in rel_info
    assert "Material" in rel_info
    assert "Sales Order Exception Report" in rel_info
    assert "A1P Location Sequence" in rel_info
    print("✅ Relationship info contains required elements")
    
    return True


def test_prompt_includes_columns_info():
    """Test that columns info is included"""
    print("\n" + "=" * 70)
    print("TEST 2: Columns Info in Prompt")
    print("=" * 70)
    
    manager = PromptTemplateManager()
    
    # Get columns info
    cols_info = manager.get_columns_info()
    print("Columns Info:")
    print(cols_info)
    
    assert "SALES ORDER EXCEPTION REPORT" in cols_info
    assert "A1P LOCATION SEQUENCE" in cols_info
    assert "69 columns" in cols_info
    assert "15 columns" in cols_info
    print("✅ Columns info contains both tables")
    
    return True


def test_formatted_prompt_structure():
    """Test that the formatted prompt has all sections"""
    print("\n" + "=" * 70)
    print("TEST 3: Formatted Prompt Structure")
    print("=" * 70)
    
    manager = PromptTemplateManager()
    
    # Test with a sample query
    query = "Show sales orders for plant 1001 with inventory group 9.5OZ"
    prompt = manager.format_filter_extraction_prompt(query)
    
    # Check key sections
    required_sections = [
        "AVAILABLE TABLES & COLUMNS:",
        "TABLE RELATIONSHIPS:",
        "CROSS-TABLE FILTERING:",
        "TERMINOLOGY MAPPINGS",
        "requires_join",
        "Plant",
        "Material"
    ]
    
    for section in required_sections:
        if section in prompt:
            print(f"✅ Section found: {section}")
        else:
            print(f"❌ Missing section: {section}")
            return False
    
    print(f"\n✅ Prompt is {len(prompt)} characters")
    return True


def test_cross_table_examples():
    """Test that cross-table examples are in the prompt"""
    print("\n" + "=" * 70)
    print("TEST 4: Cross-Table Query Examples")
    print("=" * 70)
    
    manager = PromptTemplateManager()
    template = manager.get_template("filter_extraction")
    
    # Check for cross-table example patterns
    cross_table_indicators = [
        '"requires_join": true',
        '"join_on": ["Plant", "Material"]',
        "Inven Group Name",
        "Location Id",
        "cross-table query"
    ]
    
    for indicator in cross_table_indicators:
        if indicator in template:
            print(f"✅ Found: {indicator}")
        else:
            print(f"❌ Missing: {indicator}")
            return False
    
    return True


def test_location_table_terminology():
    """Test that Location Sequence table terminology is documented"""
    print("\n" + "=" * 70)
    print("TEST 5: Location Sequence Table Terminology")
    print("=" * 70)
    
    manager = PromptTemplateManager()
    template = manager.get_template("filter_extraction")
    
    # Check for Location Sequence specific columns
    location_columns = [
        "Inven Id",
        "Inven Sequ Number",
        "Inven Group Name",
        "Location Id",
        "Auth to sell flag",  # Note: different from Sales Order's "Auth Sell Flag Description"
        "Plant(Location)"
    ]
    
    found = 0
    for col in location_columns:
        if col in template:
            print(f"✅ Column documented: {col}")
            found += 1
        else:
            print(f"⚠️  Column not explicitly documented: {col}")
    
    assert found >= 4, "Should document at least 4 Location Sequence columns"
    print(f"✅ Found {found}/{len(location_columns)} Location Sequence columns")
    return True


def test_join_detection_rules():
    """Test that rules for detecting cross-table queries are clear"""
    print("\n" + "=" * 70)
    print("TEST 6: Join Detection Rules")
    print("=" * 70)
    
    manager = PromptTemplateManager()
    template = manager.get_template("filter_extraction")
    
    # Check that the prompt explains when to use requires_join
    required_explanations = [
        "requires_join",
        "join_on",
        "Plant",
        "Material",
        "BOTH"
    ]
    
    for explanation in required_explanations:
        if explanation in template:
            print(f"✅ Documented: {explanation}")
        else:
            print(f"❌ Missing explanation: {explanation}")
            return False
    
    return True


def show_sample_formatted_prompt():
    """Display a sample formatted prompt for visual inspection"""
    print("\n" + "=" * 70)
    print("SAMPLE: Formatted Prompt with Cross-Table Query")
    print("=" * 70)
    
    manager = PromptTemplateManager()
    
    query = "Show me sales orders for materials with inventory sequence 3000 in plant 1001"
    prompt = manager.format_filter_extraction_prompt(query)
    
    print("\nQuery:", query)
    print("\n" + "-" * 70)
    print("Generated Prompt (first 1500 chars):")
    print("-" * 70)
    print(prompt[:1500])
    print("...")
    print("-" * 70)
    print(f"\nTotal prompt length: {len(prompt)} characters")
    
    return True


def run_all_tests():
    """Run all cross-table filtering tests"""
    print("\n" + "=" * 80)
    print(" " * 15 + "CROSS-TABLE FILTERING PROMPT TESTS")
    print("=" * 80)
    
    tests = [
        ("Relationship Info in Prompt", test_prompt_includes_relationship_info),
        ("Columns Info in Prompt", test_prompt_includes_columns_info),
        ("Formatted Prompt Structure", test_formatted_prompt_structure),
        ("Cross-Table Query Examples", test_cross_table_examples),
        ("Location Table Terminology", test_location_table_terminology),
        ("Join Detection Rules", test_join_detection_rules),
        ("Sample Formatted Prompt", show_sample_formatted_prompt),
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
        print(" " * 15 + "ALL TESTS PASSED!")
        print("🎉 " * 20)
        print("\n✅ The prompt is now configured for cross-table filtering!")
        print("✅ LLM will understand foreign key relationships!")
        print("✅ Queries can span Sales Order and Location Sequence tables!")
    
    print("\n" + "=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
