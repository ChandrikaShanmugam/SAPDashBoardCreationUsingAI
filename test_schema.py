#!/usr/bin/env python3
"""Test database_schema module"""

import database_schema as db_schema

print("=" * 80)
print("TESTING DATABASE SCHEMA")
print("=" * 80)

print("\n1. All Column Names:")
print("-" * 80)
for col in db_schema.get_all_columns():
    print(f"  - {col}")

print("\n2. Common Filter Columns:")
print("-" * 80)
for col, desc in db_schema.get_common_filters().items():
    print(f"  - {col}: {desc}")

print("\n3. Chart Column Recommendations:")
print("-" * 80)
chart_cols = db_schema.get_common_chart_columns()
for category, cols in chart_cols.items():
    print(f"  {category}: {', '.join(cols)}")

print("\n4. Schema Prompt for LLM (first 500 chars):")
print("-" * 80)
prompt = db_schema.generate_schema_prompt()
print(prompt[:500] + "...")

print("\n5. Column Validation Test:")
print("-" * 80)
test_cols = ["Plant", "Material", "InvalidColumn", "Auth Sell Flag Description"]
for col in test_cols:
    valid = db_schema.validate_column_name(col)
    status = "✓" if valid else "✗"
    print(f"  {status} {col}: {'Valid' if valid else 'Invalid'}")

print("\n" + "=" * 80)
print("SCHEMA TEST COMPLETE")
print("=" * 80)
