#!/usr/bin/env python3
"""
Test all 6 user query scenarios to verify Stage 1 filter + aggregation extraction
"""

import json
import sys
from pepsico_llm import invoke_llm
import database_schema as db_schema

# Test queries
TEST_SCENARIOS = [
    {
        "id": 1,
        "query": "How many failure material description for location 1007 with case qty",
        "expected": {
            "filters": {"Plant": "1007", "Auth Sell Flag Description": "No"},
            "aggregations": [
                {"column": "Material Descrption", "function": "count_unique"},
                {"column": "Order Quantity Sales Unit", "function": "sum"}
            ]
        }
    },
    {
        "id": 2,
        "query": "How many failure material description for customer account 0001234567 with case qty",
        "expected": {
            "filters": {"Sold-To Party": "0001234567", "Auth Sell Flag Description": "No"},
            "aggregations": [
                {"column": "Material Descrption", "function": "count_unique"},
                {"column": "Order Quantity Sales Unit", "function": "sum"}
            ]
        }
    },
    {
        "id": 3,
        "query": "Provide total number of sales order failed due to Authorized to Sell issue",
        "expected": {
            "filters": {"Auth Sell Flag Description": "No"},
            "aggregations": [
                {"column": "Sales Order Number", "function": "count_unique"}
            ]
        }
    },
    {
        "id": 4,
        "query": "Provide total records due to Authorized to sell error",
        "expected": {
            "filters": {"Auth Sell Flag Description": "No"},
            "aggregations": [
                {"column": None, "function": "count"}
            ]
        }
    },
    {
        "id": 5,
        "query": "Provide total number failed records for active material but authorised to sell not active",
        "expected": {
            "filters": {
                "Material Status Description": "Material Active",
                "Auth Sell Flag Description": "No"
            },
            "aggregations": [
                {"column": None, "function": "count"}
            ]
        }
    },
    {
        "id": 6,
        "query": "Provide total number of sales order for active material but authorised to sell not active",
        "expected": {
            "filters": {
                "Material Status Description": "Material Active",
                "Auth Sell Flag Description": "No"
            },
            "aggregations": [
                {"column": "Sales Order Number", "function": "count_unique"}
            ]
        }
    },
    {
        "id": 7,
        "query": "I want plant 1007 details for auth flag active",
        "expected": {
            "filters": {"Plant": "1007", "Auth Sell Flag Description": "Yes"}
        }
    }
]

# Build the Stage 1 prompt (EXACT COPY from sap_dashboard_agent.py)
INTENT_SYSTEM_PROMPT = """You are a SAP data analyst. Extract filters and aggregation requests from the user's query using EXACT column names.
            
{columns_info}

**TERMINOLOGY MAPPINGS** (User Language → Column Names):
- "location" / "plant" / "facility" → "Plant"
- "customer account" / "customer" / "sold-to" / "sold to party" → "Sold-To Party"
- "material description" / "material desc" → "Material Descrption" (note: typo in data)
- "case qty" / "quantity" / "order quantity" / "qty" → "Order Quantity Sales Unit"
- "sales order" / "order" / "SO" → "Sales Order Number"
- "active material" / "material active" → Material Status Description = "Material Active"
- "inactive material" / "material inactive" / "material is not active" → Material Status Description = "Material is not active"

**FAILURE/ERROR TERMINOLOGY**:
- "failure" / "failed" / "error" / "issue" / "exception" → Usually means Auth Sell Flag Description = "No"
- "failed due to Authorized to Sell" → Auth Sell Flag Description = "No"
- "Authorized to Sell issue/error" → Auth Sell Flag Description = "No"
- "authorised to sell not active" → Auth Sell Flag Description = "No"
- "auth to sell problem" → Auth Sell Flag Description = "No"

**SUCCESS/AUTHORIZED TERMINOLOGY**:
- "auth flag active" / "authorized" / "authorised" → Auth Sell Flag Description = "Yes"
- "auth flag Yes" → Auth Sell Flag Description = "Yes"
- "authorization active" → Auth Sell Flag Description = "Yes"

**AGGREGATION DETECTION** (When user asks "How many", "Total number", "Count", "with qty"):
If query asks for counts/totals, include "aggregations" array:
- "How many materials" → count_unique on "Material"
- "How many sales orders" → count_unique on "Sales Order Number"
- "Total number of records" → count on any column (or omit column for row count)
- "Total quantity" / "sum of qty" → sum on "Order Quantity Sales Unit"
- "How many material description" → count_unique on "Material Descrption"
- "with case qty" / "with qty" / "with quantity" → ALWAYS add sum on "Order Quantity Sales Unit"
- If query mentions BOTH "how many [something]" AND "with case qty/quantity", return TWO aggregations:
  1. count_unique on the thing being counted
  2. sum on "Order Quantity Sales Unit"

Return JSON with:
- filters: object with EXACT column names as keys and filter values
- aggregations: array of aggregation requests (ONLY if user asks "how many", "total", "count", "sum")

Each aggregation object has:
- column: EXACT column name (or null for total record count)
- function: "count", "count_unique", "sum", "mean", "max", "min"
- label: human-readable label for the metric

Example queries with EXACT expected output:

1. "I want plant 1007 details for auth flag active"
   → {{"filters": {{"Plant": "1007", "Auth Sell Flag Description": "Yes"}}}}

2. "How many failure material description for location 1007 with case qty"
   → {{
     "filters": {{"Plant": "1007", "Auth Sell Flag Description": "No"}},
     "aggregations": [
       {{"column": "Material Descrption", "function": "count_unique", "label": "Unique Materials"}},
       {{"column": "Order Quantity Sales Unit", "function": "sum", "label": "Total Case Qty"}}
     ]
   }}

3. "How many failure material description for customer account 0001234567 with case qty"
   → {{
     "filters": {{"Sold-To Party": "0001234567", "Auth Sell Flag Description": "No"}},
     "aggregations": [
       {{"column": "Material Descrption", "function": "count_unique", "label": "Unique Materials"}},
       {{"column": "Order Quantity Sales Unit", "function": "sum", "label": "Total Case Qty"}}
     ]
   }}

4. "Total number of sales order failed due to Authorized to Sell issue"
   → {{
     "filters": {{"Auth Sell Flag Description": "No"}},
     "aggregations": [
       {{"column": "Sales Order Number", "function": "count_unique", "label": "Failed Sales Orders"}}
     ]
   }}

5. "Provide total records due to Authorized to sell error"
   → {{
     "filters": {{"Auth Sell Flag Description": "No"}},
     "aggregations": [
       {{"column": null, "function": "count", "label": "Total Records"}}
     ]
   }}

6. "Total number failed records for active material but authorised to sell not active"
   → {{
     "filters": {{"Material Status Description": "Material Active", "Auth Sell Flag Description": "No"}},
     "aggregations": [
       {{"column": null, "function": "count", "label": "Failed Active Materials"}}
     ]
   }}

7. "Total number of sales order for active material but authorised to sell not active"
   → {{
     "filters": {{"Material Status Description": "Material Active", "Auth Sell Flag Description": "No"}},
     "aggregations": [
       {{"column": "Sales Order Number", "function": "count_unique", "label": "Sales Orders"}}
     ]
   }}

8. "Show me plant 7001 data" (no aggregation request)
   → {{"filters": {{"Plant": "7001"}}}}
"""


def test_scenario(scenario):
    """Test a single scenario"""
    print(f"\n{'='*80}")
    print(f"SCENARIO {scenario['id']}")
    print(f"{'='*80}")
    print(f"Query: {scenario['query']}")
    print(f"\nExpected:")
    print('\n' + '-'*10 + ' Expected ' + '-'*10)
    print(json.dumps(scenario['expected'], indent=2))
    print('\n')
    
    # Build prompt
    schema_info = db_schema.generate_schema_prompt()
    formatted_prompt = INTENT_SYSTEM_PROMPT.format(columns_info=schema_info)
    
    # Call LLM
    payload = {
        "generation_model": "gpt-4o",
        "max_tokens": 700,
        "temperature": 0.0,
        "top_p": 0.01,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "tools": [],
        "tools_choice": "none",
        "system_prompt": formatted_prompt,
        "custom_prompt": [
            {"role": "user", "content": scenario['query']}
        ],
        "model_provider_name": "openai"
    }
    
    try:
        resp = invoke_llm(payload, timeout=30)
        
        if isinstance(resp, dict) and resp.get('error'):
            print(f"\n❌ LLM Error: {resp['error']}")
            return False
        
        if isinstance(resp, dict) and 'response' in resp:
            response_text = resp['response']
            # Remove markdown code blocks if present
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(response_text)
            print(f"\nActual LLM Response:")
            print('-'*10 + ' Result ' + '-'*10)
            print(json.dumps(result, indent=2))
            print('\n')
            
            # Check filters
            expected_filters = scenario['expected'].get('filters', {})
            actual_filters = result.get('filters', {})
            
            filters_match = expected_filters == actual_filters
            print(f"\n{'✅' if filters_match else '❌'} Filters: {'MATCH' if filters_match else 'MISMATCH'}")
            if not filters_match:
                print(f"  Expected: {expected_filters}")
                print(f"  Got: {actual_filters}")
            
            # Check aggregations
            expected_aggs = scenario['expected'].get('aggregations', [])
            actual_aggs = result.get('aggregations', [])
            
            if expected_aggs:
                aggs_match = len(expected_aggs) == len(actual_aggs)
                if aggs_match:
                    for exp, act in zip(expected_aggs, actual_aggs):
                        if exp['column'] != act.get('column') or exp['function'] != act.get('function'):
                            aggs_match = False
                            break
                
                print(f"{'✅' if aggs_match else '❌'} Aggregations: {'MATCH' if aggs_match else 'MISMATCH'}")
                if not aggs_match:
                    print(f"  Expected: {expected_aggs}")
                    print(f"  Got: {actual_aggs}")
            else:
                has_no_aggs = 'aggregations' not in result or len(actual_aggs) == 0
                print(f"{'✅' if has_no_aggs else '⚠️'} No aggregations expected: {'CORRECT' if has_no_aggs else 'Got unexpected aggs'}")
            
            return filters_match and (not expected_aggs or aggs_match)
            
        else:
            print(f"\n❌ Unexpected response format: {resp}")
            return False
            
    except Exception as e:
        print(f"\n❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("TESTING ALL SCENARIOS")
    print("="*80)
    
    results = []
    for scenario in TEST_SCENARIOS:
        success = test_scenario(scenario)
        results.append((scenario['id'], success))
    
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    for scenario_id, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"Scenario {scenario_id}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nTotal: {passed}/{total} passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
