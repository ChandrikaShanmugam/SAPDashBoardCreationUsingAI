# Manual Testing Guide for Streamlit App

## Setup
1. Make sure all CSV files are in the project directory:
   - `Authorized to Sell No.csv`
   - `Authorized To Sell Yes.csv`
   - `Sales Order Exception report 13 and 14 Nov 2025.csv`
   - `SOException Nov2025.csv`

2. Run the Streamlit app:
   ```bash
   streamlit run sap_dashboard_agent.py
   ```

## Test Queries (in order of complexity)

### Test 1: Basic Filtering (No Aggregation)
**Query:** `I want plant 1007 details for auth flag active`

**Expected Result:**
- Filtered data displayed in table
- Charts showing Plant 1007 + Auth Flag Yes
- No aggregation metrics shown
- Status: "77 records found" (approximate)

---

### Test 2: Single Aggregation (Count)
**Query:** `Provide total records due to Authorized to sell error`

**Expected Result:**
- Filtered data: Auth Sell Flag Description = No
- 📊 **Metric displayed:** "Total Records" with count (should be ~25,000+)
- Charts showing failure distribution
- Status: Records filtered

---

### Test 3: Single Aggregation (Count Unique)
**Query:** `Provide total number of sales order failed due to Authorized to Sell issue`

**Expected Result:**
- Filtered data: Auth Sell Flag Description = No
- 📊 **Metric displayed:** "Failed Sales Orders" with unique count
- Should show number like 15,000-20,000 (approximate)
- Charts showing sales order distribution

---

### Test 4: Dual Aggregation (Count + Sum)
**Query:** `How many failure material description for location 1007 with case qty`

**Expected Result:**
- Filtered data: Plant=1007, Auth Flag=No
- 📊 **Two metrics displayed:**
  1. "Unique Materials" - count_unique of Material Descrption
  2. "Total Case Qty" - sum of Order Quantity Sales Unit
- Charts showing material distribution for Plant 1007
- Status: "X records found after filtering"

---

### Test 5: Complex Dual Filter + Aggregation
**Query:** `Provide total number of sales order for active material but authorised to sell not active`

**Expected Result:**
- Filtered data: Material Status=Active, Auth Flag=No
- 📊 **Metric displayed:** "Sales Orders" with unique count
- Charts showing active materials with auth issues
- Status shows filtered record count

---

### Test 6: Customer Account Query
**Query:** `How many failure material description for customer account 0001234567 with case qty`

**Expected Result:**
- Filtered data: Sold-To Party=0001234567, Auth Flag=No
- 📊 **Two metrics displayed:**
  1. "Unique Materials"
  2. "Total Case Qty"
- Charts for specific customer
- May show "No records found" if customer doesn't exist in data

---

## Verification Checklist

For each test query:
- [ ] Query submitted successfully
- [ ] Stage 1 filter extraction shown in expander (if debug mode enabled)
- [ ] Data filtered correctly (check record count)
- [ ] Aggregation metrics displayed in colored boxes at top
- [ ] Metric values are reasonable (not 0, not None)
- [ ] Charts generated without errors
- [ ] No "Column 'None' not found" errors
- [ ] No Python exceptions in terminal

## Common Issues to Watch For

1. **"Column 'None' not found"** → Stage 2 returning invalid column names (should be fixed)
2. **Aggregations not showing** → Check if `filter_result` contains "aggregations" array
3. **Wrong filter applied** → LLM misunderstood terminology (should be fixed with mappings)
4. **Charts not displaying** → Check if filtered data is empty
5. **Metrics show 0** → Data might not match filter criteria

## Debug Mode
Enable detailed logging by checking the expander sections:
- "🔍 Stage 1: Filter Extraction" → Shows LLM filter result JSON
- "📊 Stage 2: Chart Generation" → Shows LLM chart configuration
- Terminal logs show actual API calls and responses

## Success Criteria
✅ All 6 test queries run without errors  
✅ Aggregation metrics display correctly  
✅ Filtered data matches expected criteria  
✅ Charts render properly  
✅ No None/null column errors  
