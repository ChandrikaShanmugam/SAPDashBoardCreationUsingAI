# JavaScript Conversion of SAP Dashboard Agent

This document describes the conversion of the Python SAP Dashboard Agent to JavaScript/Node.js.

## Overview

The original Python code has been converted to JavaScript with the following key changes:

### Dependencies Replaced
- **pandas** → **csv-parser** for CSV loading, arrays/objects for data manipulation
- **plotly (Python)** → **plotly.js** for chart generation
- **langchain + ChatOllama** → **OpenAI API** direct calls
- **logging** → **console.log** with custom logger
- **classes** → **ES6 classes**

### Key Components Converted

1. **Data Loading** (`loadSapData` function)
   - Loads CSV files using `csv-parser`
   - Normalizes material numbers (removes leading zeros)
   - Returns data as arrays of objects

2. **IntentClassifier** class
   - Uses OpenAI API instead of local LLM
   - Maps column names using fuzzy matching
   - Extracts filters and generates chart configurations

3. **DashboardGenerator** class
   - Applies filters to data arrays
   - Creates Plotly.js chart specifications
   - Supports cross-table joins

4. **Follow-up Questions** (`generateFollowUpQuestions` function)
   - Uses OpenAI API to generate relevant follow-up questions

## Installation

```bash
cd backend
npm install
```

## Environment Variables

Set the following environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

## Usage

### Basic Usage

```javascript
const { loadSapData, IntentClassifier, DashboardGenerator } = require('./core/sap_dashboard_agent');

// Load data
const data = await loadSapData();

// Initialize classifier and generator
const classifier = new IntentClassifier(data);
const dashboardGen = new DashboardGenerator(data, classifier);

// Process a query
const filterResult = await classifier.classify("Show me plant 1007 data");
const dashboard = await dashboardGen.generate(filterResult, "Show me plant 1007 data");
```

### API Server

The backend provides a REST API:

```bash
npm start
```

API Endpoints:
- `POST /api/dashboard/generate` - Generate dashboard from query
- `GET /api/dashboard/follow-up-questions` - Get follow-up questions
- `GET /api/dashboard/data-summary` - Get data summary
- `GET /api/health` - Health check

## Key Differences from Python Version

### Data Structures
- **Python**: pandas DataFrames
- **JavaScript**: Arrays of objects

### Chart Generation
- **Python**: Returns Plotly figures
- **JavaScript**: Returns Plotly.js chart specifications (JSON)

### LLM Integration
- **Python**: Uses PepGenX API wrapper
- **JavaScript**: Direct OpenAI API calls

### Error Handling
- Both versions include comprehensive error handling
- JavaScript version uses async/await patterns

## Limitations

1. **Performance**: JavaScript arrays are slower than pandas for large datasets
2. **Memory Usage**: All data is loaded into memory as objects
3. **Type Safety**: No static typing (TypeScript could be added)
4. **Dependencies**: Requires OpenAI API key instead of local LLM

## Testing

To test the conversion:

```bash
# Test module loading
node -e "require('./core/sap_dashboard_agent')"

# Test data loading
node -e "const {loadSapData} = require('./core/sap_dashboard_agent'); loadSapData().then(d => console.log('Loaded', Object.keys(d).length, 'tables'))"
```

## Integration with Frontend

The JavaScript backend is designed to work with the existing React frontend. The API responses include:

- `charts`: Array of Plotly.js chart specifications
- `tables`: Array of table data with headers and rows
- `data_sample`: Sample of filtered data
- `follow_up_questions`: Suggested next queries

## Future Improvements

1. Add TypeScript for better type safety
2. Implement data streaming for large datasets
3. Add caching for LLM responses
4. Optimize data filtering performance
5. Add unit tests</content>
<parameter name="filePath">/Users/chandrika/Documents/GitHub/SAPDashBoardCreationUsingAI/backend/JAVASCRIPT_CONVERSION_README.md