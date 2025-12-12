#!/usr/bin/env python3
"""
Quick test for "Order Quantity based UPC details" query
"""
import sys
sys.path.insert(0, '/Users/chandrika/Documents/SAPDashBoardCreationUsingAI/Sap_Dashboard_Creation/src')
sys.path.insert(0, '/Users/chandrika/Documents/SAPDashBoardCreationUsingAI/Sap_Dashboard_Creation/src/core')

from core.prompt_manager import PromptManager
from core.pepsico_llm import PepsiCoLLM
import json

def test_upc_query():
    """Test the problematic UPC query"""
    print("=" * 80)
    print("Testing: Order Quantity based UPC details")
    print("=" * 80)
    
    # Initialize
    manager = PromptManager()
    llm = PepsiCoLLM()
    
    query = "Order Quantity based UPC details"
    
    # Generate filter extraction prompt
    print("\n1. Generating filter extraction prompt...")
    prompt = manager.format_filter_extraction_prompt(query)
    
    # Show relevant parts of the prompt
    print("\n2. Checking if prompt includes UPC mapping...")
    if "upc" in prompt.lower():
        print("   ✓ UPC terminology found in prompt")
    else:
        print("   ✗ UPC terminology NOT found in prompt")
    
    if "based on" in prompt.lower() or "grouping" in prompt.lower():
        print("   ✓ Grouping/display guidance found in prompt")
    else:
        print("   ✗ Grouping/display guidance NOT found in prompt")
    
    # Call LLM
    print("\n3. Calling LLM for filter extraction...")
    try:
        response = llm.extract_filters(prompt)
        print(f"\n4. LLM Response:")
        print(json.dumps(response, indent=2))
        
        # Validate response
        print("\n5. Validation:")
        if response.get("filters") == {}:
            print("   ✓ Filters are empty (correct for grouping/display query)")
        else:
            print(f"   ✗ Filters should be empty but got: {response.get('filters')}")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_upc_query()
