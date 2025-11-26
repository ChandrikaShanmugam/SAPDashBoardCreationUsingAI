# Data Flow: What Goes to LLM vs What Stays Local

## ❌ WRONG Approach (Don't Do This)
```python
# BAD: Sending all data to LLM
user_query = "Show me authorized materials"
all_data = load_1_million_records()  # 1.1M records
llm.invoke(f"Analyze this data: {all_data}")  # ❌ TOO EXPENSIVE!
```

## ✅ CORRECT Approach (What We Do)

### **Step 1: LLM Only Classifies Intent** 
```python
user_query = "Show me authorized to sell details"

# Only send the QUERY to LLM, not the data
llm_input = {
    "query": user_query,
    "available_data_sources": ["auth_yes", "auth_no", "exceptions"],
    "available_visualizations": ["pie", "bar", "table", "metric"]
}

# LLM returns classification (very small response)
llm_response = {
    "intent": "authorized_to_sell",
    "data_sources": ["auth_yes", "auth_no"],
    "visualizations": ["pie", "bar", "metric"],
    "metrics": ["total_count", "auth_rate"]
}
```

**LLM Sees:** ~100 tokens (just the query + schema info)
**LLM Returns:** ~50 tokens (JSON classification)
**Cost:** ~$0.0001 per query

---

### **Step 2: Python/Pandas Does the Heavy Work**
```python
# Load data LOCALLY (not sent to LLM)
auth_yes = pd.read_excel('Authorized To Sell Yes.csv')  # 301K records
auth_no = pd.read_excel('Authorized to Sell No.csv')   # 809K records

# Calculate metrics LOCALLY using Pandas
total = len(auth_yes) + len(auth_no)
auth_count = len(auth_yes)
auth_rate = (auth_count / total) * 100

# Create visualizations LOCALLY using Plotly
fig = px.pie(values=[auth_count, len(auth_no)], 
             names=['Authorized', 'Not Authorized'])
```

**Data Processing:** 100% local, no LLM involved
**Speed:** Fast (Pandas is optimized for big data)
**Cost:** $0

---

### **Step 3: (Optional) LLM Generates Insights on SUMMARY**
```python
# Only send SUMMARY statistics to LLM, not raw data
summary = {
    "total_materials": 1110952,
    "authorized": 301263,
    "not_authorized": 809689,
    "auth_rate": 27.1,
    "top_5_plants": [
        {"plant": "1001", "count": 15234},
        {"plant": "1006", "count": 12456},
        # ...
    ]
}

# Ask LLM to generate insights from summary
prompt = f"""
Analyze this summary and provide 3 key business insights:
{json.dumps(summary)}

Focus on:
1. What stands out
2. Potential issues
3. Recommendations
"""

insights = llm.invoke(prompt)
```

**LLM Sees:** ~200 tokens (summary only, not 1M records)
**LLM Returns:** ~300 tokens (insights text)
**Cost:** ~$0.001 per insight generation

---

## 📊 Complete Data Flow Diagram

```
┌─────────────────────────────────────────┐
│ User Query:                             │
│ "Show me authorized to sell details"    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  LLM (Llama 3.2 - Local)                │
│  Input: Query text (~50 tokens)         │
│  Output: Intent classification          │
│    {                                    │
│      "intent": "authorized_to_sell",    │
│      "data_sources": ["auth_yes"]       │
│    }                                    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Python/Pandas Processing               │
│  - Load CSVs (1.1M records)             │
│  - Filter, group, aggregate             │
│  - Calculate metrics                    │
│  - ALL DONE LOCALLY                     │
│  NO LLM INVOLVED ✅                      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Generate Visualizations                │
│  - Plotly creates charts                │
│  - Streamlit displays tables            │
│  ALL DONE LOCALLY ✅                     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  (Optional) Generate AI Insights        │
│  - Calculate summary stats locally      │
│  - Send ONLY summary to LLM (~200 tok)  │
│  - LLM returns insights (~300 tok)      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Display Dashboard to User              │
│  - Metrics cards                        │
│  - Interactive charts                   │
│  - Data tables                          │
│  - AI insights                          │
└─────────────────────────────────────────┘
```

---

## 💰 Cost Comparison

### **Scenario: Analyze 1.1M records**

#### **❌ BAD: Send All Data to LLM**
```
- Data size: 1.1M records × ~100 chars = ~110M characters
- Tokens: ~27.5M tokens
- Cost with GPT-4: ~$825 per query (!!)
- Time: Would timeout/fail
```

#### **✅ GOOD: Our Approach**
```
- Intent Classification: ~100 tokens input + 50 output = 150 tokens
- Cost with GPT-4: ~$0.0015 per query
- Cost with Local Llama: $0 (free!)
- Time: < 2 seconds

Optional AI Insights:
- Summary data: ~200 tokens input + 300 output = 500 tokens  
- Cost with GPT-4: ~$0.005 per insight
- Cost with Local Llama: $0 (free!)
```

**Savings: 99.999% reduction in cost!**

---

## 🔍 Detailed Example

### **Query: "Show me materials with exceptions that are not authorized"**

#### **Step 1: Intent Classification (LLM)**
```python
# What goes to LLM:
prompt = """
User query: "Show me materials with exceptions that are not authorized"

Available data:
- auth_yes: Materials authorized to sell
- auth_no: Materials not authorized
- exceptions: Sales order exceptions

What should I do?
"""

# LLM response (small):
{
    "intent": "cross_analysis",
    "data_sources": ["auth_no", "exceptions"],
    "join_on": "Material",
    "visualizations": ["table", "bar", "metric"]
}
```
**Tokens sent:** ~150
**Tokens received:** ~50

#### **Step 2: Data Processing (Python - No LLM)**
```python
# Load data locally
auth_no = pd.read_excel('Authorized to Sell No.csv')  # 809K records
exceptions = pd.read_excel('SOException Nov2025.csv')  # 26K records

# Join/merge locally using Pandas (FAST!)
result = exceptions.merge(
    auth_no, 
    left_on='Material', 
    right_on='Material', 
    how='inner'
)

# Calculate metrics locally
count = len(result)
unique_materials = result['Material'].nunique()
affected_plants = result['Plant'].nunique()
total_quantity = result['Order Quantity Sales Unit'].sum()
```
**LLM involvement:** ZERO
**Cost:** $0
**Time:** < 1 second

#### **Step 3: Generate Insights (LLM - Optional)**
```python
# Create summary (small!)
summary = {
    "total_exception_orders": count,
    "unique_materials": unique_materials,
    "percentage_of_exceptions": (count / len(exceptions) * 100),
    "top_3_materials": result['Material'].value_counts().head(3).to_dict(),
    "top_3_plants": result['Plant'].value_counts().head(3).to_dict()
}

# Ask LLM for insights (only send summary!)
prompt = f"""
Analysis: {count} exception orders found for unauthorized materials.
This represents {summary['percentage_of_exceptions']:.1f}% of all exceptions.

Top 3 problematic materials:
{summary['top_3_materials']}

Top 3 affected plants:
{summary['top_3_plants']}

Provide 3 business insights and recommendations:
"""

insights = llm.invoke(prompt)
```
**Tokens sent:** ~250
**Tokens received:** ~400
**Total tokens:** ~650

---

## 🎯 Key Principles

### **1. LLM for Intelligence, Not Heavy Lifting**
- ✅ Understand user intent
- ✅ Generate human-readable insights
- ✅ Suggest visualizations
- ❌ NOT for data processing
- ❌ NOT for calculations
- ❌ NOT for filtering/joining

### **2. Use Right Tool for Right Job**
- **LLM:** Natural language understanding, text generation
- **Pandas:** Data manipulation, filtering, aggregation
- **Plotly:** Visualization creation
- **Streamlit:** UI rendering

### **3. Summary First, Details Later**
```python
# Good: Hierarchical approach
1. LLM sees: "Analyze authorization data"
2. Python calculates: Summary stats
3. LLM generates: Insights from summary
4. User sees: Dashboard with insights

# Bad: Everything through LLM
1. LLM sees: All 1M records (SLOW, EXPENSIVE)
```

### **4. Cache Aggressively**
```python
@st.cache_data  # Cache data loading
def load_data():
    return pd.read_excel('data.csv')

# Data loaded once, reused for all queries
```

---

## 📈 Performance Metrics

### **Our Approach (Hybrid: LLM + Python)**
- Query understanding: < 1 second (LLM)
- Data loading: 2-3 seconds (Pandas, cached)
- Data processing: < 1 second (Pandas)
- Visualization: < 1 second (Plotly)
- Insights generation: 2-3 seconds (LLM, optional)
- **Total: < 5 seconds**
- **Cost: ~$0.001 or FREE with local LLM**

### **If We Sent All Data to LLM**
- Data upload: 30+ seconds
- LLM processing: Timeout/Error
- Cost: $100+ per query
- **Would not work!**

---

## 🔐 Privacy & Security Bonus

By keeping data local:
- ✅ Sensitive data never leaves your environment
- ✅ No data sent to OpenAI/Claude
- ✅ Compliant with data governance policies
- ✅ Fast (no network latency)
- ✅ Free (using local Llama)

---

## 💡 Smart Prompting Examples

### **Example 1: Intent Classification**
```python
# Efficient prompt (what we do)
prompt = f"""
Classify this query: "{user_query}"

Available intents:
- authorized_to_sell: Questions about material authorization
- exceptions: Questions about sales order issues
- plant_analysis: Questions about specific plants
- overview: General summaries

Return JSON: {{"intent": "...", "data_sources": [...], "metrics": [...]}}
"""
```
**Size:** ~200 tokens

### **Example 2: Insight Generation**
```python
# Send summary, not raw data
prompt = f"""
Key findings:
- 27% materials authorized (industry avg: 65%)
- Plant 1006: 342 exceptions (highest)
- Material X causes 15% of all exceptions

Provide 3 actionable insights for management:
"""
```
**Size:** ~150 tokens

### **❌ What NOT to Do**
```python
# DON'T send raw data
prompt = f"""
Analyze this data:
{dataframe.to_string()}  # ❌ Could be millions of characters!
"""
```

---

## 🎓 Summary

### **What LLM Does:**
1. Understand natural language query (~100 tokens)
2. Classify intent and suggest approach (~50 tokens)
3. Generate human insights from summary stats (~500 tokens)

**Total per query: ~650 tokens = $0.001 or FREE**

### **What Python/Pandas Does:**
1. Load 1.1M records from CSV
2. Filter, join, aggregate data
3. Calculate all metrics
4. Create visualizations

**All local, all fast, all free**

### **Result:**
- ⚡ Fast (< 5 seconds)
- 💰 Cheap ($0 with local LLM)
- 🎯 Accurate (Pandas for data, LLM for language)
- 🔒 Secure (data stays local)

---

## ✅ Best Practices Checklist

- [x] Use LLM only for natural language tasks
- [x] Process all data locally with Pandas
- [x] Send only summaries/aggregations to LLM
- [x] Cache data loading with `@st.cache_data`
- [x] Use local LLM (Llama) when possible
- [x] Calculate metrics before asking LLM for insights
- [x] Keep prompts small and focused
- [x] Never send raw data tables to LLM

---

**The magic is in the architecture: LLM for intelligence, Python for heavy lifting!**
