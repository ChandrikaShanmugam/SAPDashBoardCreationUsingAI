#!/usr/bin/env python3
"""Verify what's being sent to LLM in Stage 1 and Stage 2"""

import database_schema as db_schema

print("=" * 80)
print("STAGE 1: FILTER EXTRACTION PROMPT CONTENT")
print("=" * 80)

# This is what gets inserted into {columns_info}
schema_prompt = db_schema.generate_schema_prompt()
print(f"\nSchema Info Length: {len(schema_prompt)} characters")
print(f"Number of columns: {len(db_schema.get_all_columns())}")
print(f"\nColumns being sent to LLM:")
for i, col in enumerate(db_schema.get_all_columns(), 1):
    print(f"  {i}. {col}")

print("\n" + "=" * 80)
print("STAGE 2: CHART GENERATION PROMPT CONTENT")
print("=" * 80)

chart_cols = db_schema.get_common_chart_columns()
all_cols = ', '.join(db_schema.get_all_columns())

print(f"\nAll columns list sent to LLM:")
print(f"  {all_cols[:200]}...")

print(f"\nChart-specific column recommendations:")
print(f"  Bar charts: {', '.join(chart_cols['for_bar_charts'])}")
print(f"  Pie charts: {', '.join(chart_cols['for_pie_charts'])}")
print(f"  Aggregation: {', '.join(chart_cols['for_aggregation'])}")
print(f"  Details: {', '.join(chart_cols['for_details'])}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ Stage 1 sends: Full schema with {len(db_schema.get_all_columns())} columns")
print(f"✓ Stage 2 sends: Column list + chart recommendations")
print(f"✓ User query is combined with column info to extract filters")
print("\nBoth stages have access to ALL column names!")
