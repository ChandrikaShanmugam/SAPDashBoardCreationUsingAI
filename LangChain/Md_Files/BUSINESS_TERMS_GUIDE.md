# Business Terminology Guide - SAP Dashboard

## 🎯 How the LLM Understands Your Questions

### Key Concept: "Failed" / "Error" / "Issue" Mapping

When you say:
- ❌ "failed due to Authorized to Sell"
- ❌ "Authorized to Sell issue"
- ❌ "Authorized to Sell error"
- ❌ "auth to sell problem"
- ❌ "not authorized"

The system understands: **`Auth Sell Flag Description = "No"`**

---

## 📊 Common Questions & How They're Interpreted

### 1. **"Provide total number of sales orders failed due to Authorized to Sell issue"**

**LLM understands:**
```json
{
  "intent": "exceptions",
  "filters": {
    "Auth Sell Flag Description": "No"
  },
  "aggregations": [
    {
      "column": "Sales Order Number",
      "function": "count_unique",
      "label": "Failed Sales Orders"
    }
  ]
}
```

**What you'll see:**
- Filtered to only "No" authorization records
- Metric showing unique count of Sales Order Numbers
- Charts for the filtered data

---

### 2. **"Provide total records due to Authorized to sell error"**

**LLM understands:**
```json
{
  "intent": "exceptions",
  "filters": {
    "Auth Sell Flag Description": "No"
  },
  "aggregations": [
    {
      "column": "Sales Order Number",
      "function": "count",
      "label": "Total Records"
    }
  ]
}
```

**What you'll see:**
- All records where auth flag = "No"
- Total count of records (not unique orders)

---

### 3. **"How many failure material descriptions for plant 1007 with case qty?"**

**LLM understands:**
```json
{
  "intent": "plant_analysis",
  "filters": {
    "Plant": "1007"
  },
  "aggregations": [
    {
      "column": "Material Descrption",
      "function": "count_unique",
      "label": "Unique Materials"
    },
    {
      "column": "Order Quantity Sales Unit",
      "function": "sum",
      "label": "Total Case Qty"
    }
  ]
}
```

**What you'll see:**
- Filtered to Plant 1007
- Count of unique material descriptions
- Sum of all order quantities

---

## 🔑 Column Name Translation Table

| You Say | System Uses |
|---------|-------------|
| "sales order" | `Sales Order Number` |
| "location" or "plant" | `Plant` |
| "customer" or "customer account" or "sold-to" | `Sold-To Party` |
| "material description" | `Material Descrption` |
| "material" or "material code" | `Material` |
| "quantity" or "case qty" | `Order Quantity Sales Unit` |
| "status" or "material status" | `Material Status Description` |
| "auth flag" or "authorization" | `Auth Sell Flag Description` |

---

## 🎲 Understanding Authorization Values

| You Say | Filter Applied |
|---------|----------------|
| "failed due to auth to sell" | `Auth Sell Flag Description = "No"` |
| "authorized to sell issue" | `Auth Sell Flag Description = "No"` |
| "auth flag active" | `Auth Sell Flag Description = "Yes"` |
| "authorized materials" | `Auth Sell Flag Description = "Yes"` |
| "not authorized" | `Auth Sell Flag Description = "No"` |

---

## 📈 Counting & Aggregation Logic

### Count Types:

1. **"How many sales orders"** → `count_unique` on `Sales Order Number`
   - Gives you unique order count

2. **"Total records"** → `count` on any column
   - Gives you total row count

3. **"How many materials"** → `count_unique` on `Material` or `Material Descrption`
   - Gives you unique material count

4. **"Total quantity"** → `sum` on `Order Quantity Sales Unit`
   - Gives you sum of all quantities

---

## 💡 Smart Examples

### Example 1: Authorization Failures
```
Query: "Total orders failed due to auth to sell issue"

System interprets:
- Filter: Auth Sell Flag Description = "No"
- Count: Unique Sales Order Numbers
- Intent: exceptions

Result:
✓ Shows only unauthorized records
✓ Displays unique order count metric
✓ Generates relevant charts
```

### Example 2: Plant with Quantities
```
Query: "How many materials for location 1007 with quantities"

System interprets:
- Filter: Plant = "1007"
- Aggregations:
  * Count unique Materials
  * Sum Order Quantity Sales Unit
- Intent: plant_analysis

Result:
✓ Filters to Plant 1007
✓ Shows unique material count
✓ Shows total case quantity
✓ Displays plant-specific charts
```

### Example 3: Customer Analysis
```
Query: "How many failure materials for customer 0001234567"

System interprets:
- Filter: Sold-To Party = "0001234567"
- Count: Unique Materials
- Intent: customer_analysis

Result:
✓ Filters to specific customer
✓ Shows unique material count
✓ Displays customer-specific data
```

---

## 🚀 Test Queries You Can Try

### Authorization Issues:
1. "Provide total number of sales orders failed due to Authorized to Sell issue"
2. "Provide total records due to Authorized to sell error"
3. "How many orders have auth to sell problems?"
4. "Show me not authorized materials"

### Plant/Location Analysis:
1. "How many failure material descriptions for plant 1007 with case qty"
2. "Show me location 7001 materials with quantities"
3. "What materials failed for plant 1269?"

### Customer Analysis:
1. "How many failure materials for customer account 0001234567 with case qty"
2. "Show me sold-to party 0001234567 details"
3. "What materials failed for customer 0001234567?"

### Combined Filters:
1. "Show me plant 1007 materials failed due to auth to sell issue"
2. "How many authorized materials for location 7001?"
3. "Give me plant 1269 active materials with quantities"

---

## ✅ What Makes This Work

1. **🧠 Comprehensive Business Term Mapping**
   - LLM trained on all common phrasings
   - Maps business language to technical columns

2. **🔍 Context-Aware Interpretation**
   - Understands "failed" = authorization issue
   - Maps "location" = Plant column
   - Recognizes "case qty" = Order Quantity

3. **📊 Smart Aggregation**
   - Automatically chooses count vs count_unique
   - Applies sum for quantities
   - Formats results appropriately

4. **🎯 Intent Classification**
   - Routes to correct dashboard type
   - Applies relevant filters
   - Generates appropriate visualizations

---

## 🔧 Behind the Scenes

When you ask: **"Provide total number of sales orders failed due to Authorized to Sell issue"**

### Step 1: LLM Analysis
```
Input: Full column schema + business term mappings
Process: Parse query → Identify keywords
- "sales orders" → Sales Order Number
- "failed due to Authorized to Sell" → Auth Sell Flag = "No"
- "total number" → count_unique aggregation
```

### Step 2: Filter Application
```python
filtered_data = data[data['Auth Sell Flag Description'] == 'No']
```

### Step 3: Aggregation
```python
unique_orders = filtered_data['Sales Order Number'].nunique()
```

### Step 4: Display
```
📊 Requested Metrics
Failed Sales Orders: 758
```

---

## 📝 Important Notes

1. **ALL data in this system are exceptions** - When you ask about "failures", the system is already working with failure data

2. **"Failed due to auth"** specifically means authorization flag = "No"

3. **Column names are case-sensitive** - System uses exact names from data

4. **Aggregations run on filtered data** - Filters apply first, then counts

5. **Performance optimized** - Only sample sent to LLM, full calculation done locally

---

## 🎉 Ready to Use!

Your dashboard at **http://localhost:8503** now understands natural business terminology!

Try the test queries above and see how the system intelligently maps your questions to the data! 🚀
