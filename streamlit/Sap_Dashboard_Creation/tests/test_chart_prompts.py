#!/usr/bin/env python3
"""
Test Chart Generation Prompts for Multi-Dimensional Queries
===========================================================
Tests the prompt configuration for grouped bar and multi-dimensional charts
without calling the actual LLM API.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src' / 'core'))

import json
import pandas as pd
from prompt_manager import PromptTemplateManager
import database_schema as db_schema

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_subsection(title):
    print(f"\n--- {title} ---")

def test_chart_generation_prompt_content():
    """Test that chart generation prompt includes multi-dimensional guidance"""
    print_section("TEST 1: Chart Generation Prompt Content")
    
    manager = PromptTemplateManager()
    
    # Read the actual prompt template
    chart_prompt = manager.get_template('chart_generation')
    
    print_subsection("Checking for Key Sections")
    
    checks = {
        'MULTI-DIMENSIONAL QUERIES': 'MULTI-DIMENSIONAL' in chart_prompt,
        'grouped_bar chart type': 'grouped_bar' in chart_prompt,
        'stacked_bar chart type': 'stacked_bar' in chart_prompt,
        'color_by parameter': 'color_by' in chart_prompt,
        'limit_groups parameter': 'limit_groups' in chart_prompt,
    }
    
    all_passed = True
    for check_name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")
        if not result:
            all_passed = False
    
    return all_passed

def test_single_dimension_prompt():
    """Test prompt formatting for single dimension query"""
    print_section("TEST 2: Single Dimension Query Prompt")
    
    query = "Order Quantity by UPC details"
    print(f"Query: {query}")
    
    manager = PromptTemplateManager()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'UPC': ['12000203978', '78000019636', '78000394160'],
        'Material': ['300008760', '300004898', '300009919'],
        'Order Quantity Sales Unit': [7500, 4500, 3500],
        'Plant': ['1054', '1029', '1029']
    })
    
    data_sample = {
        "shape": sample_data.shape,
        "columns": sample_data.columns.tolist(),
        "sample_rows": sample_data.head(3).to_dict('records')
    }
    
    # Format prompt
    chart_columns = db_schema.get_common_chart_columns()
    all_cols = ', '.join(db_schema.get_all_columns())
    
    prompt = manager.format_template(
        'chart_generation',
        user_query=query,
        data_sample=json.dumps(data_sample, indent=2),
        all_columns=all_cols,
        bar_chart_columns=', '.join(chart_columns['for_bar_charts']),
        pie_chart_columns=', '.join(chart_columns['for_pie_charts']),
        aggregation_columns=', '.join(chart_columns['for_aggregation']),
        detail_columns=', '.join(chart_columns['for_details'])
    )
    
    print_subsection("Formatted Prompt Preview")
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Contains query: {'Order Quantity by UPC' in prompt}")
    print(f"Contains UPC column: {'UPC' in prompt}")
    print(f"Contains sample data: {'12000203978' in prompt}")
    
    # Check if multi-dimensional guidance is present
    print_subsection("Multi-Dimensional Guidance Check")
    has_multidim = 'MULTI-DIMENSIONAL' in prompt or 'grouped_bar' in prompt
    print(f"{'✅ PASS' if has_multidim else '❌ FAIL'}: Multi-dimensional guidance present")
    
    return has_multidim

def test_multi_dimension_prompt():
    """Test prompt formatting for multi-dimensional query"""
    print_section("TEST 3: Multi-Dimensional Query Prompt")
    
    query = "Order Quantity by UPC and Customer Hierarchy"
    print(f"Query: {query}")
    
    manager = PromptTemplateManager()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'UPC': ['12000203978', '78000019636'],
        'Customer Hierarchy Level 4': ['WALMART', 'TARGET'],
        'Order Quantity Sales Unit': [7500, 4500],
        'Material': ['300008760', '300004898'],
    })
    
    data_sample = {
        "shape": sample_data.shape,
        "columns": sample_data.columns.tolist(),
        "sample_rows": sample_data.head(2).to_dict('records')
    }
    
    # Format prompt
    chart_columns = db_schema.get_common_chart_columns()
    all_cols = ', '.join(db_schema.get_all_columns())
    
    prompt = manager.format_template(
        'chart_generation',
        user_query=query,
        data_sample=json.dumps(data_sample, indent=2),
        all_columns=all_cols,
        bar_chart_columns=', '.join(chart_columns['for_bar_charts']),
        pie_chart_columns=', '.join(chart_columns['for_pie_charts']),
        aggregation_columns=', '.join(chart_columns['for_aggregation']),
        detail_columns=', '.join(chart_columns['for_details'])
    )
    
    print_subsection("Formatted Prompt Analysis")
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Contains 'and' keyword: {' and ' in query}")
    print(f"Contains both dimensions:")
    print(f"  - UPC: {'UPC' in prompt}")
    print(f"  - Customer Hierarchy: {'Customer Hierarchy' in prompt}")
    
    print_subsection("Multi-Dimensional Features Check")
    checks = {
        'grouped_bar mentioned': 'grouped_bar' in prompt,
        'color_by mentioned': 'color_by' in prompt,
        'Multi-dimensional section': 'MULTI-DIMENSIONAL' in prompt,
    }
    
    all_passed = True
    for check_name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")
        if not result:
            all_passed = False
    
    return all_passed

def test_filter_extraction_prompt():
    """Test that filter extraction handles 'based on' queries"""
    print_section("TEST 4: Filter Extraction Prompt")
    
    manager = PromptTemplateManager()
    
    # Read the filter extraction prompt
    filter_prompt = manager.get_template('filter_extraction')
    
    print_subsection("Checking for 'based on' handling")
    
    checks = {
        'GROUPING/DISPLAY QUERIES section': 'GROUPING' in filter_prompt or 'DISPLAY' in filter_prompt,
        'UPC terminology': 'UPC' in filter_prompt or 'GTIN' in filter_prompt,
        '"based on" mentioned': 'based on' in filter_prompt,
        'Empty filters guidance': ('empty' in filter_prompt.lower() or 'EMPTY' in filter_prompt) and '{}' in filter_prompt,
    }
    
    all_passed = True
    for check_name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")
        if not result:
            all_passed = False
    
    return all_passed

def test_schema_text_columns():
    """Test that schema provides text columns correctly"""
    print_section("TEST 5: Schema Text Columns")
    
    text_cols = db_schema.get_text_columns("all")
    
    print_subsection("Text Columns Found")
    print(f"Total text columns: {len(text_cols)}")
    print(f"Columns: {', '.join(text_cols[:10])}...")
    
    critical_cols = ['UPC', 'Material', 'Plant']
    print_subsection("Critical Column Check")
    
    all_passed = True
    for col in critical_cols:
        present = col in text_cols
        status = "✅ PASS" if present else "❌ FAIL"
        print(f"{status}: {col} marked as text type")
        if not present:
            all_passed = False
    
    return all_passed

def main():
    """Run all tests"""
    print_section("MULTI-DIMENSIONAL CHART PROMPT TESTS")
    print("Testing prompt configuration without calling LLM API")
    
    results = {
        'Prompt Content': test_chart_generation_prompt_content(),
        'Single Dimension': test_single_dimension_prompt(),
        'Multi Dimension': test_multi_dimension_prompt(),
        'Filter Extraction': test_filter_extraction_prompt(),
        'Schema Text Columns': test_schema_text_columns(),
    }
    
    print_section("TEST SUMMARY")
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! Prompt configuration is correct.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    exit(main())
