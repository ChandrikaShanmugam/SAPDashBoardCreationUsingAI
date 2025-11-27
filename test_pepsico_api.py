"""Quick test of PepGenX API integration"""
from pepsico_llm import invoke_llm
import json

# Test payload matching the API's expected format
test_payload = {
    "generation_model": "gpt-4o",
    "max_tokens": 100,
    "temperature": 0.0,
    "top_p": 0.01,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "tools": [],
    "tools_choice": "none",
    "system_prompt": "You are a helpful assistant.",
    "custom_prompt": [
        {"role": "user", "content": "Say 'Hello, API is working!' in one sentence."}
    ],
    "model_provider_name": "openai"
}

print("Testing PepGenX API integration...")
print("-" * 60)

response = invoke_llm(test_payload, timeout=30)

print("Response:")
print('\n' + '-'*10 + ' Response ' + '-'*10)
print(json.dumps(response, indent=2))
print('\n')

if 'error' in response:
    print("\n❌ API call failed!")
    print(f"Error: {response['error']}")
elif 'text' in response:
    print("\n✓ API responded successfully!")
    print(f"Text: {response['text']}")
else:
    print("\n✓ API responded with JSON:")
    print('-'*10 + ' Response ' + '-'*10)
    print(json.dumps(response, indent=2))
    print('\n')
