"""
Test Prompt Manager with Foreign Key Relationships
===================================================
Demonstrates how the prompt manager includes relationship information
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "core"))

from prompt_manager import PromptTemplateManager


def test_relationship_info():
    """Test relationship information generation"""
    print("\n" + "=" * 80)
    print("TEST: Foreign Key Relationship Information")
    print("=" * 80)
    
    manager = PromptTemplateManager()
    
    relationship_info = manager.get_relationship_info()
    print(relationship_info)
    
    assert "Sales Order Exception Report" in relationship_info
    assert "A1P Location Sequence" in relationship_info
    assert "Plant" in relationship_info
    assert "Material" in relationship_info
    
    print("\n✅ Relationship information generated successfully")
    return True


def test_columns_info():
    """Test column information generation"""
    print("\n" + "=" * 80)
    print("TEST: Column Information")
    print("=" * 80)
    
    manager = PromptTemplateManager()
    
    columns_info = manager.get_columns_info()
    print(columns_info)
    
    assert "SALES ORDER EXCEPTION REPORT" in columns_info
    assert "A1P LOCATION SEQUENCE" in columns_info
    assert "69 columns" in columns_info
    assert "15 columns" in columns_info
    
    print("\n✅ Column information generated successfully")
    return True


def test_filter_extraction_prompt():
    """Test complete filter extraction prompt with relationships"""
    print("\n" + "=" * 80)
    print("TEST: Complete Filter Extraction Prompt")
    print("=" * 80)
    
    manager = PromptTemplateManager()
    
    # Test various query types
    test_queries = [
        "Show me plant 1007 data",
        "How many materials with inventory group name 9.5OZ NR 12/1 in plant 1001",
        "Show sales orders for inventory sequence 3000 with auth sell failures",
        "Get materials in location ID 245 that have authorization issues"
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {query}")
        print(f"{'=' * 80}")
        
        prompt = manager.format_filter_extraction_prompt(query)
        
        # Verify key elements are in the prompt
        assert "DATABASE SCHEMA & RELATIONSHIPS" in prompt
        assert "CROSS-TABLE FILTERING" in prompt
        assert "requires_join" in prompt
        assert query in prompt
        assert "Plant" in prompt
        assert "Material" in prompt
        
        print(f"✅ Prompt generated successfully ({len(prompt)} characters)")
        
        # Show a preview
        print(f"\nPrompt Preview (first 500 chars):")
        print(prompt[:500] + "...")
        print(f"\nPrompt Preview (last 500 chars):")
        print("..." + prompt[-500:])
    
    return True


def test_cross_table_examples():
    """Test that cross-table query examples are in the prompt"""
    print("\n" + "=" * 80)
    print("TEST: Cross-Table Query Examples")
    print("=" * 80)
    
    manager = PromptTemplateManager()
    
    prompt = manager.format_filter_extraction_prompt("test query")
    
    # Check for cross-table examples
    required_examples = [
        "inventory group name",
        "inventory sequence",
        "location ID",
        "requires_join",
        "join_on"
    ]
    
    for example in required_examples:
        if example in prompt:
            print(f"✅ Found example: '{example}'")
        else:
            print(f"❌ Missing example: '{example}'")
            return False
    
    print("\n✅ All cross-table examples present in prompt")
    return True


def demonstrate_prompt_usage():
    """Demonstrate real-world usage"""
    print("\n" + "=" * 80)
    print("DEMONSTRATION: Real-World Prompt Usage")
    print("=" * 80)
    
    manager = PromptTemplateManager()
    
    # Simulate realistic queries
    real_queries = [
        {
            "query": "Show me all failed sales orders for plant 1029",
            "description": "Simple single-table filter"
        },
        {
            "query": "How many materials have auth sell issues in inventory group 10OZ PL 1/24",
            "description": "Cross-table query with aggregation"
        },
        {
            "query": "Get sales orders for location 245 with inventory sequence 3000 that failed",
            "description": "Complex cross-table with multiple filters"
        }
    ]
    
    for i, test_case in enumerate(real_queries, 1):
        print(f"\n{'-' * 80}")
        print(f"Example {i}: {test_case['description']}")
        print(f"Query: \"{test_case['query']}\"")
        print(f"{'-' * 80}")
        
        prompt = manager.format_filter_extraction_prompt(test_case['query'])
        
        print(f"✅ Generated prompt: {len(prompt)} characters")
        print(f"   Contains schema info: {'✓' if 'DATABASE SCHEMA' in prompt else '✗'}")
        print(f"   Contains relationships: {'✓' if 'Foreign Key' in prompt else '✗'}")
        print(f"   Contains cross-table examples: {'✓' if 'requires_join' in prompt else '✗'}")
        print(f"   Contains user query: {'✓' if test_case['query'] in prompt else '✗'}")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("PROMPT MANAGER WITH FOREIGN KEY RELATIONSHIPS - TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Relationship Info Generation", test_relationship_info),
        ("Column Info Generation", test_columns_info),
        ("Filter Extraction Prompt", test_filter_extraction_prompt),
        ("Cross-Table Examples", test_cross_table_examples),
        ("Real-World Usage Demo", demonstrate_prompt_usage),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n❌ TEST FAILED: {test_name}")
        except Exception as e:
            failed += 1
            print(f"\n❌ TEST ERROR: {test_name}")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nThe prompt manager now includes:")
        print("  ✓ Foreign key relationship information")
        print("  ✓ Cross-table query examples")
        print("  ✓ Join condition guidance")
        print("  ✓ Table-specific column information")
    
    print("\n" + "=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
