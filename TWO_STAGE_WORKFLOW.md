# Two-Stage LLM Workflow Implementation

## Overview
Your SAP Dashboard now implements a clean **two-stage workflow** where:
1. **Stage 1**: LLM extracts filters from natural language query
2. **Stage 2**: Data is filtered locally, then LLM generates chart configurations

## Workflow for: "I want plant 1007 details for auth flag active"

```
User Query: "I want plant 1007 details for auth flag active"
    ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 1: FILTER EXTRACTION (LLM)                        │
│ - Sends query + column schema to LLM                    │
│ - LLM returns: {"filters": {"Plant": "1007",            │
│                "Auth Sell Flag Description": "Yes"}}    │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ LOCAL DATA FILTERING (Python/Pandas)                    │
│ - Applies filters to 26,000 records                     │
│ - Returns 77 matching records                           │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 2: CHART GENERATION (LLM)                         │
│ - Sends: original query + filtered data sample          │
│ - LLM returns chart configs:                            │
│   [{"type": "bar", "x": "Material",                     │
│     "y": "Order Quantity Sales Unit",                   │
│     "agg": "sum", "limit": 10}]                         │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ CHART RENDERING (Plotly/Streamlit)                      │
│ - Groups by Material, sums Quantity                     │
│ - Sorts descending, takes top 10                        │
│ - Renders bar chart in Streamlit UI                     │
└─────────────────────────────────────────────────────────┘
```

## Key Changes Made

### 1. **Simplified Stage 1 Prompt** (`classify()` method)
   - **Before**: Returned complex intent + filters + aggregations + visualizations
   - **After**: Returns ONLY filters: `{"filters": {"Plant": "1007", "Auth Sell Flag Description": "Yes"}}`
   
### 2. **Updated Filter Extraction Logic**
   - Removed intent classification complexity
   - Focuses purely on extracting column-value pairs
   - Better examples for "auth flag active" → `"Auth Sell Flag Description": "Yes"`

### 3. **Streamlined `generate()` Method**
   - **Before**: Complex branching based on intent (plant_analysis, exceptions, overview, etc.)
   - **After**: Single unified path:
     1. Apply filters to data
     2. Show metrics
     3. Call Stage 2 for charts
     4. Render results

### 4. **Clean Stage 2 Chart Generation**
   - Takes filtered data sample (only 10 rows sent to LLM)
   - Returns chart configurations with exact column names
   - Charts are rendered from configs using Plotly

### 5. **Better UI Feedback**
   - Shows "Stage 1: Extracting filters" spinner
   - Shows "Stage 2: Filtering data and generating charts" spinner
   - Expander shows extracted filters clearly
   - Performance metrics track both stages

## Testing Your Fixed Workflow

Try these queries to see the two-stage workflow in action:

1. **Plant with Auth Flag**:
   ```
   "I want plant 1007 details for auth flag active"
   ```
   Expected:
   - Stage 1: `{"Plant": "1007", "Auth Sell Flag Description": "Yes"}`
   - Stage 2: Bar chart of materials by quantity

2. **Simple Plant Filter**:
   ```
   "Show me plant 7001 data"
   ```
   Expected:
   - Stage 1: `{"Plant": "7001"}`
   - Stage 2: Charts showing plant 7001 analysis

3. **Auth Issues**:
   ```
   "Show me authorization issues"
   ```
   Expected:
   - Stage 1: `{"Auth Sell Flag Description": "No"}`
   - Stage 2: Charts showing failed authorizations

4. **No Filters**:
   ```
   "Show all data"
   ```
   Expected:
   - Stage 1: `{}`
   - Stage 2: Overview charts of entire dataset

## Benefits of This Approach

✅ **Token Efficiency**: Only small data samples sent to LLM (not full 26K rows)
✅ **Clear Separation**: Stage 1 = filters, Stage 2 = visualizations
✅ **Better Debugging**: Can see exactly what filters were extracted
✅ **Performance**: Local filtering is fast, LLM only generates configs
✅ **Maintainability**: Each stage has a single, clear responsibility

## How to Run

```bash
cd /Users/chandrika/Documents/SAPDashBoardCreationUsingAI
python3 -m streamlit run sap_dashboard_agent.py
```

Then try the query:
```
I want plant 1007 details for auth flag active
```

You should see:
1. ✅ Stage 1: Filters extracted: `{"Plant": "1007", "Auth Sell Flag Description": "Yes"}`
2. ✅ Data filtered: 26,000 → 77 records
3. ✅ Stage 2: Charts generated showing top materials by quantity
4. ✅ Visualizations rendered in Streamlit

## Troubleshooting

If Stage 1 doesn't extract filters correctly:
- Check logs for LLM response
- Enable Developer Mode in sidebar to see API calls
- LLM should see column names including "Auth Sell Flag Description"

If Stage 2 doesn't generate charts:
- Check that filtered data sample is being sent
- Verify column names in chart config match actual data columns
- Look for errors in chart rendering logs
