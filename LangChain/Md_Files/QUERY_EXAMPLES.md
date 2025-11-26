# SAP Dashboard - Enhanced Query Examples

## 🎯 What's New

Your dashboard now supports **comprehensive column-based queries** with automatic aggregations and detailed metrics!

### Key Enhancements:
1. ✅ **All columns visible to LLM** - Complete column information with data types, sample values, and ranges
2. ✅ **Smart aggregations** - Automatic counting, summing, and statistical calculations
3. ✅ **Flexible filters** - Support for all columns including Plant, Material, Customer Account, etc.
4. ✅ **Custom metrics** - Display exactly what you ask for

---

## 📋 Supported Query Types

### 1. **Material Failure Counts with Quantities**

**Examples:**
- "How many failure material descriptions for plant 1007 with case qty?"
- "How many unique materials failed for location 7001 with total quantity?"
- "Show me material count for plant 1269 with order quantities"

**What happens:**
- Filters data by the specified plant/location
- Counts unique materials (Material Descrption column)
- Sums up Order Quantity Sales Unit
- Displays as dedicated metrics

---

### 2. **Customer Account Analysis**

**Examples:**
- "How many failure material descriptions for customer account 0001234567 with case qty?"
- "Show me materials for Sold-To Party 0001234567 with total quantity"
- "What materials failed for customer 0001234567?"

**What happens:**
- Filters by Sold-To Party (customer account)
- Shows unique material count
- Calculates total case quantities
- Creates customer-specific dashboard

---

### 3. **Authorization Issues**

**Examples:**
- "Provide total number of sales orders failed due to Authorized to Sell issue"
- "Provide total records due to Authorized to sell error"
- "How many orders failed because of auth flag No?"

**What happens:**
- Filters: `Auth Sell Flag Description = "No"`
- Counts unique Sales Order Numbers
- Shows total records affected
- Displays authorization-specific metrics

---

### 4. **Plant/Location Analysis**

**Examples:**
- "Show me plant 7001 details with material count"
- "I want plant 1007 active materials with quantities"
- "Give me plant 1269 exceptions with case qty"

**What happens:**
- Filters by Plant column
- Shows plant-specific metrics
- Generates location-based charts
- Displays material details if requested

---

### 5. **Material Status Queries**

**Examples:**
- "Show active materials for plant 7001"
- "How many inactive materials in plant 1007?"
- "Give me material status breakdown"

**What happens:**
- Filters by Material Status Description
- Shows status distribution
- Creates status-based visualizations

---

## 🔍 Understanding Column Names

The system now has complete visibility into all columns:

| User Says | System Uses Column |
|-----------|-------------------|
| "location", "plant" | `Plant` |
| "customer account", "sold-to party" | `Sold-To Party` |
| "material description" | `Material Descrption` |
| "case qty", "quantity" | `Order Quantity Sales Unit` |
| "auth flag", "authorized" | `Auth Sell Flag Description` |
| "material status" | `Material Status Description` |
| "sales order" | `Sales Order Number` |

---

## 📊 Available Aggregation Functions

The system can automatically calculate:

1. **count** - Total number of records
2. **count_unique** - Number of unique values (e.g., unique materials)
3. **sum** - Total sum (e.g., total quantity)
4. **mean** - Average value
5. **max** - Maximum value
6. **min** - Minimum value

---

## 💡 Smart Query Tips

### Get Specific Metrics:
```
"How many failure material descriptions for plant 1007 with case qty?"
```
→ Shows: Unique Materials + Total Quantity

### Combine Filters:
```
"Show me plant 7001 authorized materials with active status"
```
→ Filters: Plant=7001, Auth Sell Flag=Yes, Status=Active

### Ask for Details:
```
"Give me plant 1007 details with material information"
```
→ Shows: Metrics + Charts + Detailed Material Table

---

## 🎨 What You'll See

### Before:
- Limited column visibility
- Manual interpretation needed
- Generic metrics only

### Now:
- ✅ All columns analyzed by LLM
- ✅ Automatic metric calculation
- ✅ Custom aggregations displayed
- ✅ Smart filter extraction
- ✅ Relevant charts generated

---

## 📈 Sample Output Structure

```
🔍 Filtered Data Analysis
Filters Applied: Plant=1007, Auth Sell Flag Description=No

[Standard Metrics Row]
Total Records: 758
Unique Materials: 77
Unique Plants: 1
Active Materials: 77

[Custom Requested Metrics Row]
Failed Orders: 758
Total Case Qty: 15,234.00
Unique Material Descriptions: 77

[Dynamic Charts - 2x2 Grid]
[Chart 1] [Chart 2]
[Chart 3] [Chart 4]

📋 Material Details
[Detailed Table with Selected Columns]
```

---

## 🚀 Try These Queries Now!

1. **Basic Count:**
   - "How many records for plant 1007?"

2. **With Quantities:**
   - "Show me plant 1007 materials with total case qty"

3. **Authorization Focus:**
   - "Total sales orders failed due to auth to sell issue"

4. **Customer Specific:**
   - "How many materials for customer 0001234567 with quantities?"

5. **Complex Filter:**
   - "Give me plant 7001 active authorized materials with case qty"

---

## 📝 Notes

- The system automatically maps natural language to exact column names
- All columns are now visible and queryable
- Aggregations are calculated on filtered data
- Performance optimized: Only sample data sent to LLM, full aggregation done locally
- Response time: ~10 seconds for complex queries

---

## 🔧 Technical Details

**What Changed:**
1. `_get_columns_info()` - Now shows ALL columns with types, ranges, and samples
2. `intent_system_prompt` - Enhanced with aggregation examples and all column mappings
3. `generate()` - Added aggregation calculation and display logic
4. `_create_exceptions_dashboard_dynamic()` - Now supports custom metrics

**Data Flow:**
1. User Query → LLM analyzes with full column schema
2. LLM extracts: intent + filters + aggregations
3. Python filters data locally (26K records)
4. Python calculates requested aggregations
5. Display metrics + generate charts

---

## ✅ Ready to Use!

Your dashboard is now running at: **http://localhost:8503**

Try the example queries above and see the enhanced analytics in action! 🎉
