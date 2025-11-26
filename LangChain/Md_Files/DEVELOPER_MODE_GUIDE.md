# Developer Mode Guide

## 🔧 Accessing Developer Tools

The SAP Dashboard now includes comprehensive developer tools for debugging and monitoring.

### **How to Enable Developer Mode:**

1. Run the application:
   ```bash
   python3 -m streamlit run sap_dashboard_agent.py
   ```

2. In the sidebar, check the box: **🔧 Developer Mode**

---

## 📊 Developer Console Features

### **1. API Requests Tab**
Shows all LLM API calls with:
- **Timestamp** - When the call was made
- **Request** - User query sent to LLM
- **Response** - Full JSON response from LLM
- **Duration** - How long the API call took

**Example:**
```json
{
  "timestamp": "2025-11-24 14:30:45",
  "type": "LLM Classification",
  "query": "Show me authorized to sell details",
  "response": {
    "intent": "authorized_to_sell",
    "data_sources": ["auth_yes", "auth_no"],
    "visualizations": ["pie", "bar", "table"]
  },
  "duration": "1.23s"
}
```

---

### **2. Performance Metrics Tab**
Displays timing breakdowns:

- **Init Time** - Application startup time
- **LLM Classification** - Time for intent classification
- **Dashboard Gen** - Time to generate visualizations
- **Total Time** - End-to-end query processing

**Includes visual chart** showing performance breakdown by stage.

---

### **3. Console Logs Tab**
Access to detailed logging:

**Terminal Logs Show:**
```
===============================================================================
LOADING SAP DATA FILES
===============================================================================
Loading: Authorized To Sell Yes.csv
✓ Loaded 301,263 records from Authorized To Sell Yes.csv
Loading: Authorized to Sell No.csv
✓ Loaded 809,689 records from Authorized to Sell No.csv
✓ All data loaded successfully in 2.34 seconds
===============================================================================

===============================================================================
INTENT CLASSIFICATION
===============================================================================
User Query: 'Show me authorized to sell details'
Building LLM prompt...
Sending request to LLM...
✓ LLM Response received in 1.23 seconds
Intent Classification Result:
{
  "intent": "authorized_to_sell",
  "data_sources": ["auth_yes", "auth_no"],
  "visualizations": ["pie", "bar", "table", "metric"]
}
===============================================================================
```

---

### **4. Data Info Tab**
Shows loaded datasets:

For each dataset:
- **Records count**
- **Column names**
- **Memory usage**

**Example:**
```
📁 Authorized Yes
Records: 301,263
Memory: 23.45 MB
Columns: Plant(Location), Material, Auth to sell flag
```

---

## 🎯 What You Can Monitor

### **1. LLM API Calls**
- See exact prompts sent to Llama
- View JSON responses
- Track token usage (indirectly via response size)
- Measure API latency

### **2. Data Processing**
- How long data loading takes
- Memory consumption per dataset
- Total records processed

### **3. Performance Bottlenecks**
- Identify which stage is slowest
- Optimize based on timing data
- Compare different query types

### **4. Error Tracking**
- Full error traces in terminal
- Fallback mechanism activations
- Data validation issues

---

## 🛠️ Developer Tools

### **Clear Logs**
- Click **🗑️ Clear Logs** to reset API call history
- Useful when testing multiple scenarios

### **Refresh Data**
- Click **🔄 Refresh Data** to reload CSV files
- Clears cache and re-reads from disk

---

## 📝 Terminal Output Example

When you run the app, you'll see detailed logs like:

```bash
2025-11-24 14:30:42 - __main__ - INFO - =====================================
2025-11-24 14:30:42 - __main__ - INFO - LOADING SAP DATA FILES
2025-11-24 14:30:42 - __main__ - INFO - =====================================
2025-11-24 14:30:42 - __main__ - INFO - Loading: Authorized To Sell Yes.csv
2025-11-24 14:30:43 - __main__ - INFO - ✓ Loaded 301,263 records
2025-11-24 14:30:43 - __main__ - INFO - Loading: Authorized to Sell No.csv
2025-11-24 14:30:45 - __main__ - INFO - ✓ Loaded 809,689 records
2025-11-24 14:30:45 - __main__ - INFO - ✓ All data loaded in 2.34 seconds

2025-11-24 14:30:50 - __main__ - INFO - =====================================
2025-11-24 14:30:50 - __main__ - INFO - INTENT CLASSIFICATION
2025-11-24 14:30:50 - __main__ - INFO - =====================================
2025-11-24 14:30:50 - __main__ - INFO - User Query: 'Show me authorized to sell'
2025-11-24 14:30:50 - __main__ - INFO - Building LLM prompt...
2025-11-24 14:30:50 - __main__ - INFO - Sending request to LLM...
2025-11-24 14:30:51 - __main__ - INFO - ✓ LLM Response received in 1.23s
2025-11-24 14:30:51 - __main__ - INFO - Intent Classification Result:
{
  "intent": "authorized_to_sell",
  "data_sources": ["auth_yes", "auth_no"]
}

2025-11-24 14:30:51 - __main__ - INFO - =====================================
2025-11-24 14:30:51 - __main__ - INFO - GENERATING DASHBOARD
2025-11-24 14:30:51 - __main__ - INFO - =====================================
2025-11-24 14:30:51 - __main__ - INFO - Dashboard Type: authorized_to_sell
2025-11-24 14:30:52 - __main__ - INFO - ✓ Dashboard generated in 0.45 seconds
```

---

## 🎓 Understanding the Flow

### **Request Flow:**
```
User Query
    ↓
📝 Log: User query received
    ↓
LLM Classification API Call
    ↓
📝 Log: Prompt sent to LLM
📝 Log: Response received (with timing)
    ↓
Data Processing (Pandas)
    ↓
📝 Log: Dashboard type selected
    ↓
Dashboard Generation
    ↓
📝 Log: Visualization created (with timing)
    ↓
Display to User
```

### **What Goes to LLM:**
```json
{
  "system": "You are a SAP data analyst...",
  "user": "Show me authorized to sell details"
}
```

**Size:** ~200 tokens

### **What Comes Back:**
```json
{
  "intent": "authorized_to_sell",
  "data_sources": ["auth_yes", "auth_no"],
  "visualizations": ["pie", "bar", "table"],
  "metrics": ["total", "authorized_count"]
}
```

**Size:** ~50 tokens

### **Total Token Usage Per Query:**
- Input: ~200 tokens
- Output: ~50 tokens
- **Total: ~250 tokens** (FREE with local Llama!)

---

## 🔍 Debugging Tips

### **If LLM is slow:**
- Check terminal for "Sending request to LLM..." timing
- Ensure Ollama is running: `ollama serve`
- Try: `ollama pull llama3.2` to ensure model is downloaded

### **If data loading is slow:**
- Check terminal for file loading times
- Large files (800K+ records) take 2-3 seconds - this is normal
- Data is cached after first load

### **If dashboard generation is slow:**
- Check Performance Metrics tab
- Large datasets may take longer to aggregate
- Consider filtering data before visualization

---

## 🎯 Production Monitoring

For production, you can:

1. **Export logs to file:**
   ```python
   logging.basicConfig(
       filename='app.log',
       level=logging.INFO
   )
   ```

2. **Add metrics to external monitoring:**
   - Send timings to Prometheus
   - Log to Application Insights
   - Track in Datadog

3. **Set up alerts:**
   - Alert if API calls > 5 seconds
   - Alert if data loading fails
   - Monitor memory usage

---

## 📊 Example Developer Session

1. **Enable Developer Mode** ✓
2. **Ask Query:** "Show me authorized to sell details"
3. **Check API Requests tab:**
   - See exact LLM prompt
   - View classification result
   - Note: 1.23s response time
4. **Check Performance tab:**
   - Classification: 1.23s
   - Dashboard: 0.45s
   - Total: 1.68s
5. **Check Terminal:**
   - See detailed logs
   - Verify data loaded correctly
6. **Optimize if needed:**
   - If slow, check bottlenecks
   - Consider caching strategies

---

## 🚀 Advanced Usage

### **Track Custom Metrics:**
Add to session state:
```python
st.session_state.custom_metric = value
```

### **Log Custom Events:**
```python
logger.info("Custom event happened")
```

### **Export API History:**
```python
import json
with open('api_calls.json', 'w') as f:
    json.dump(st.session_state.api_calls, f, indent=2)
```

---

**Developer Mode gives you complete visibility into the AI-powered dashboard!** 🔍
