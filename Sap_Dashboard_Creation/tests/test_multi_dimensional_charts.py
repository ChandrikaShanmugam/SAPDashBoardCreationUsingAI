#!/usr/bin/env python3
"""
Test Multi-Dimensional Chart Generation
=========================================
Tests the new grouped bar and multi-dimensional query functionality:
1. Single dimension queries (baseline)
2. Multi-dimensional queries with "and"
3. Grouped bar chart generation
4. Stacked bar chart generation
5. Data validation for grouped charts
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src' / 'core'))

import json
import pandas as pd
from prompt_manager import PromptTemplateManager
from pepsico_llm import invoke_llm
import database_schema as db_schema

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_subsection(title):
    print(f"\n--- {title} ---")

def test_single_dimension_query():
    """Test baseline: single dimension query should generate ONE chart"""
    print_section("TEST 1: Single Dimension Query")
    
    query = "Order Quantity by UPC details"
    print(f"Query: {query}")
    
    manager = PromptTemplateManager()
    llm = PepsiCoLLM()
    
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
    
    print_subsection("Calling LLM for chart generation")
    response = llm.generate_chart_config(prompt)
    
    print(f"\nLLM Response:")
    print(json.dumps(response, indent=2))
    
    # Validate
    charts = response.get('charts', [])
    tables = response.get('tables', [])
    
    print_subsection("Validation")
    print(f"Number of charts: {len(charts)}")
    print(f"Number of tables: {len(tables)}")
    
    if len(charts) == 1:
        print("✅ PASS: Single dimension query generated exactly ONE chart")
    else:
        print(f"❌ FAIL: Expected 1 chart, got {len(charts)}")
    
    if charts and charts[0].get('type') == 'bar':
        print("✅ PASS: Chart type is 'bar'")
    else:
        print(f"❌ FAIL: Expected bar chart, got {charts[0].get('type') if charts else 'none'}")
    
    return len(charts) == 1


def test_multi_dimensional_query():
    """Test multi-dimensional query should generate MULTIPLE charts including grouped"""
    print_section("TEST 2: Multi-Dimensional Query (UPC and Customer Hierarchy)")
    
    query = "Order Quantity by UPC and Customer Hierarchy summary"
    print(f"Query: {query}")
    
    manager = PromptTemplateManager()
    llm = PepsiCoLLM()
    
    # Create sample data with multiple dimensions
    sample_data = pd.DataFrame({
        'UPC': ['12000203978', '12000203978', '78000019636', '78000019636'],
        'Material': ['300008760', '300008760', '300004898', '300004898'],
        'Order Quantity Sales Unit': [3500, 4000, 2500, 2000],
        'Customer Hierarchy Level 6 Text': ['WALMART', 'TARGET', 'WALMART', 'KROGER'],
        'Sold-to Name': ['WALMART STORES', 'TARGET CORP', 'WALMART', 'KROGER CO'],
        'Plant': ['1054', '1054', '1029', '1029']
    })
    
    data_sample = {
        "shape": sample_data.shape,
        "columns": sample_data.columns.tolist(),
        "sample_rows": sample_data.head(4).to_dict('records')
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
    
    print_subsection("Calling LLM for chart generation")
    response = llm.generate_chart_config(prompt)
    
    print(f"\nLLM Response:")
    print(json.dumps(response, indent=2))
    
    # Validate
    charts = response.get('charts', [])
    tables = response.get('tables', [])
    
    print_subsection("Validation")
    print(f"Number of charts: {len(charts)}")
    print(f"Number of tables: {len(tables)}")
    
    # Check for individual dimension charts
    has_upc_chart = any(c.get('x_column') == 'UPC' and c.get('type') == 'bar' for c in charts)
    has_hierarchy_chart = any('Customer Hierarchy' in str(c.get('x_column', '')) and c.get('type') == 'bar' for c in charts)
    has_grouped_chart = any(c.get('type') in ['grouped_bar', 'stacked_bar'] for c in charts)
    
    print(f"\nChart breakdown:")
    for i, chart in enumerate(charts, 1):
        print(f"  Chart {i}: type={chart.get('type')}, x_column={chart.get('x_column')}, color_by={chart.get('color_by')}")
    
    if len(charts) >= 3:
        print(f"✅ PASS: Multi-dimensional query generated {len(charts)} charts (expected 3+)")
    else:
        print(f"❌ FAIL: Expected at least 3 charts, got {len(charts)}")
    
    if has_upc_chart:
        print("✅ PASS: Has individual UPC chart")
    else:
        print("❌ FAIL: Missing individual UPC chart")
    
    if has_hierarchy_chart:
        print("✅ PASS: Has individual Customer Hierarchy chart")
    else:
        print("❌ FAIL: Missing individual Customer Hierarchy chart")
    
    if has_grouped_chart:
        print("✅ PASS: Has grouped/stacked bar chart showing relationships")
    else:
        print("❌ FAIL: Missing grouped/stacked bar chart")
    
    # Check table includes both dimensions
    if tables:
        table_cols = tables[0].get('columns', [])
        has_upc_in_table = 'UPC' in table_cols
        has_hierarchy_in_table = any('Customer Hierarchy' in col for col in table_cols)
        
        if has_upc_in_table and has_hierarchy_in_table:
            print("✅ PASS: Table includes both UPC and Customer Hierarchy columns")
        else:
            print(f"❌ FAIL: Table missing dimensions. Columns: {table_cols}")
    
    return len(charts) >= 3 and has_upc_chart and has_hierarchy_chart and has_grouped_chart


def test_grouped_chart_configuration():
    """Test grouped bar chart has correct configuration"""
    print_section("TEST 3: Grouped Bar Chart Configuration")
    
    query = "Sales by plant and customer"
    print(f"Query: {query}")
    
    manager = PromptTemplateManager()
    llm = PepsiCoLLM()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'Plant': ['1054', '1054', '1029', '1029'],
        'Sold-to Name': ['WALMART', 'TARGET', 'WALMART', 'KROGER'],
        'Order Quantity Sales Unit': [10000, 8000, 7000, 5000]
    })
    
    data_sample = {
        "shape": sample_data.shape,
        "columns": sample_data.columns.tolist(),
        "sample_rows": sample_data.head(4).to_dict('records')
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
    
    print_subsection("Calling LLM for chart generation")
    response = llm.generate_chart_config(prompt)
    
    print(f"\nLLM Response:")
    print(json.dumps(response, indent=2))
    
    # Find grouped chart
    charts = response.get('charts', [])
    grouped_charts = [c for c in charts if c.get('type') in ['grouped_bar', 'stacked_bar']]
    
    print_subsection("Validation")
    
    if grouped_charts:
        grouped = grouped_charts[0]
        print(f"✅ Found grouped/stacked chart")
        print(f"   Type: {grouped.get('type')}")
        print(f"   X-column: {grouped.get('x_column')}")
        print(f"   Y-column: {grouped.get('y_column')}")
        print(f"   Color by: {grouped.get('color_by')}")
        print(f"   Limit: {grouped.get('limit', 'not set')}")
        print(f"   Limit groups: {grouped.get('limit_groups', 'not set')}")
        
        has_x_column = grouped.get('x_column') is not None
        has_y_column = grouped.get('y_column') is not None
        has_color_by = grouped.get('color_by') is not None
        
        if has_x_column and has_y_column and has_color_by:
            print("✅ PASS: Grouped chart has all required fields (x_column, y_column, color_by)")
        else:
            print(f"❌ FAIL: Grouped chart missing fields. x={has_x_column}, y={has_y_column}, color={has_color_by}")
        
        return has_x_column and has_y_column and has_color_by
    else:
        print("❌ FAIL: No grouped/stacked bar chart found")
        return False


def test_material_and_plant_query():
    """Test another multi-dimensional scenario"""
    print_section("TEST 4: Material and Plant Multi-Dimensional Query")
    
    query = "Material and UPC details by plant"
    print(f"Query: {query}")
    
    manager = PromptTemplateManager()
    llm = PepsiCoLLM()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'Plant': ['1054', '1054', '1029', '1029'],
        'Material': ['300008760', '300009428', '300044193', '300044660'],
        'Material Descrption': ['STRBK FRP PMNT', 'STRBK FRP CFFE', 'STRY ZS CNB', 'STRY ZS CNB CBL'],
        'UPC': ['12000203978', '12000203978', '12000240317', '12000240317'],
        'Order Quantity Sales Unit': [1, 1, 6, 6]
    })
    
    data_sample = {
        "shape": sample_data.shape,
        "columns": sample_data.columns.tolist(),
        "sample_rows": sample_data.head(4).to_dict('records')
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
    
    print_subsection("Calling LLM for chart generation")
    response = llm.generate_chart_config(prompt)
    
    print(f"\nLLM Response:")
    print(json.dumps(response, indent=2))
    
    # Validate
    charts = response.get('charts', [])
    
    print_subsection("Validation")
    print(f"Number of charts: {len(charts)}")
    
    # Should have charts for Material, UPC, and grouped view
    has_material_chart = any('Material' in str(c.get('x_column', '')) for c in charts)
    has_upc_chart = any(c.get('x_column') == 'UPC' for c in charts)
    has_grouped = any(c.get('type') in ['grouped_bar', 'stacked_bar'] for c in charts)
    
    print(f"\nChart types found:")
    for i, chart in enumerate(charts, 1):
        print(f"  Chart {i}: {chart.get('type')} - x={chart.get('x_column')}, color={chart.get('color_by')}")
    
    if len(charts) >= 2:
        print(f"✅ PASS: Generated {len(charts)} charts")
    else:
        print(f"⚠️  WARNING: Expected at least 2 charts, got {len(charts)}")
    
    if has_material_chart:
        print("✅ PASS: Has Material dimension chart")
    else:
        print("❌ FAIL: Missing Material dimension chart")
    
    if has_upc_chart:
        print("✅ PASS: Has UPC dimension chart")
    else:
        print("❌ FAIL: Missing UPC dimension chart")
    
    if has_grouped:
        print("✅ PASS: Has grouped chart showing relationships")
    else:
        print("⚠️  WARNING: No grouped chart found")
    
    return len(charts) >= 2


def main():
    print_section("MULTI-DIMENSIONAL CHART GENERATION TESTS")
    print("Testing new grouped bar and multi-dimensional functionality")
    
    results = []
    
    try:
        # Test 1: Single dimension (baseline)
        results.append(("Single Dimension Query", test_single_dimension_query()))
    except Exception as e:
        print(f"\n❌ Test 1 FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Single Dimension Query", False))
    
    try:
        # Test 2: Multi-dimensional query
        results.append(("Multi-Dimensional Query", test_multi_dimensional_query()))
    except Exception as e:
        print(f"\n❌ Test 2 FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Multi-Dimensional Query", False))
    
    try:
        # Test 3: Grouped chart configuration
        results.append(("Grouped Chart Config", test_grouped_chart_configuration()))
    except Exception as e:
        print(f"\n❌ Test 3 FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Grouped Chart Config", False))
    
    try:
        # Test 4: Material and Plant query
        results.append(("Material and Plant Query", test_material_and_plant_query()))
    except Exception as e:
        print(f"\n❌ Test 4 FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Material and Plant Query", False))
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{'='*80}")
    print(f"TOTAL: {passed}/{total} tests passed")
    print(f"{'='*80}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Multi-dimensional charts working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review output above.")
        return 1


if __name__ == "__main__":
    exit(main())
