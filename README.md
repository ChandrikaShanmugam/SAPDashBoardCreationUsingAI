# SAP Dashboard Creation Using AI

This project provides both Streamlit and React implementations of an intelligent SAP dashboard generator that uses AI to create dynamic dashboards from natural language queries.

## Project Structure

```
├── streamlit/          # Streamlit application (working code)
├── react/             # React application (component-based UI)
├── backend/           # FastAPI backend (shared functionality)
└── requirements.txt   # Python dependencies
```

## Features

- **Natural Language Processing**: Ask questions in plain English to generate dashboards
- **Dynamic Chart Generation**: Automatically creates appropriate visualizations based on your data
- **Multi-table Support**: Handles complex queries across multiple SAP data tables
- **Cross-platform UI**: Same functionality available in both Streamlit and React

## Prerequisites

### For Backend (Python)
- Python 3.8+
- Virtual environment (recommended)

### For React Frontend
- Node.js 16+
- npm or yarn

## Installation & Setup

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt
```

### 2. React Frontend Setup

```bash
# Navigate to React directory
cd react

# Install Node.js dependencies
npm install
```

### 3. Streamlit App

The Streamlit app uses the same backend, so no additional setup is needed beyond the backend.

## Running the Applications

### Start the Backend API

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Start the React Frontend

```bash
cd react
npm start
```

The React app will be available at `http://localhost:3000`

### Start the Streamlit App

```bash
cd streamlit/Sap_Dashboard_Creation/src/core
streamlit run sap_dashboard_agent.py
```

The Streamlit app will be available at `http://localhost:8501`

## Usage

1. **Enter a natural language query** in the sidebar, such as:
   - "Show me authorized to sell details"
   - "What are the sales exceptions?"
   - "Give me plant-wise analysis"
   - "Show me data for plant 1007"

2. **Click "Generate Dashboard"** to process your query

3. **View the results** in the main dashboard area with:
   - Applied filters summary
   - Key metrics
   - Dynamic charts and visualizations
   - Data tables

## Architecture

### Backend (FastAPI)
- **Data Loading**: Loads and processes SAP CSV data files
- **Intent Classification**: Uses LLM to understand user queries and extract filters
- **Chart Generation**: Automatically determines appropriate visualizations
- **API Endpoints**: RESTful API for frontend communication

### Frontend Components

#### React App
- **Sidebar**: Query input, settings, example questions
- **Dashboard**: Metrics display, charts, and data tables
- **Chart Component**: Renders various chart types using Plotly.js

#### Streamlit App
- Single-file application with embedded UI and logic
- Direct integration with Python backend functions

## API Endpoints

- `GET /` - API status
- `GET /data_summary` - Available data tables and columns
- `POST /generate_dashboard` - Generate dashboard from query

## Data Sources

The application works with the following SAP data files:
- Sales Order Exception Report
- A1P Location Sequence
- COF Inventory Net Price Material
- COF Material Pricing

## Development

### Adding New Chart Types
1. Update the `Chart.js` component to handle new chart types
2. Modify the backend chart generation logic in `sap_dashboard_agent.py`

### Extending the API
1. Add new endpoints in `backend/main.py`
2. Update frontend components to use new endpoints

### Customizing UI
- **React**: Modify components in `react/src/components/`
- **Streamlit**: Update `sap_dashboard_agent.py`

## Troubleshooting

### Backend Issues
- Ensure all data files are present in `backend/data/`
- Check that the virtual environment is activated
- Verify LLM API credentials are configured

### React Issues
- Run `npm install` to ensure all dependencies are installed
- Check that the backend is running on port 8000
- Clear browser cache if charts don't render

### Streamlit Issues
- Ensure the backend is running
- Check file paths in the Streamlit app

## License

This project is for educational and demonstration purposes.