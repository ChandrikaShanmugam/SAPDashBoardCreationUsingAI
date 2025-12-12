from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import json
import pandas as pd
from typing import Dict, List, Optional

# Add backend directory to path for local imports
backend_dir = '/Users/chandrika/Documents/GitHub/SAPDashBoardCreationUsingAI/backend'
sys.path.insert(0, backend_dir)

# Import core modules
from core import (
    invoke_llm,
    get_text_columns,
    load_exception_csv,
    extract_filters_from_llm,
    apply_filters,
    suggest_charts_from_llm,
    PromptTemplateManager,
    load_sap_data,
    IntentClassifier,
    DashboardGenerator
)

# Import core modules
import core.pepsico_llm
import core.database_schema
import core.exception_handler
import core.prompt_manager
import core.sap_dashboard_agent

from core.pepsico_llm import invoke_llm
from core.database_schema import get_text_columns
from core.exception_handler import load_exception_csv, extract_filters_from_llm, apply_filters, suggest_charts_from_llm
from core.prompt_manager import PromptTemplateManager
from core.sap_dashboard_agent import load_sap_data, IntentClassifier, DashboardGenerator

app = FastAPI(title="SAP Dashboard API", description="API for SAP Dashboard functionality")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for data and models
data = None
classifier = None
dashboard_gen = None

class QueryRequest(BaseModel):
    query: str
    conversation_history: Optional[List[Dict]] = None

class DashboardResponse(BaseModel):
    filters: Dict
    charts: List[Dict]
    tables: List[Dict]
    metrics: Dict
    data_sample: Dict

@app.on_event("startup")
async def startup_event():
    """Initialize data and models on startup"""
    global data, classifier, dashboard_gen
    try:
        print("Loading SAP data...")
        data = load_sap_data()
        print("Initializing classifier and dashboard generator...")
        classifier = IntentClassifier(data)
        dashboard_gen = DashboardGenerator(data, classifier)
        print("API initialized successfully")
    except Exception as e:
        print(f"Error initializing API: {e}")
        raise

@app.post("/generate_dashboard", response_model=DashboardResponse)
async def generate_dashboard(request: QueryRequest):
    """Generate dashboard based on user query"""
    try:
        # Extract filters
        filter_result = classifier.classify(request.query, request.conversation_history or [])
        filters = filter_result.get('filters', {})

        # Apply filters to get data
        combined_data = dashboard_gen._determine_and_apply_filters(filters, request.query)

        # Generate chart configuration
        chart_config = classifier.generate_chart_config(
            request.query,
            combined_data,
            "analysis",
            request.conversation_history or []
        )

        # Calculate metrics
        metrics = {
            "total_records": len(combined_data),
            "unique_materials": combined_data['Material'].nunique() if 'Material' in combined_data.columns else 0,
            "unique_plants": combined_data['Plant'].nunique() if 'Plant' in combined_data.columns else 0,
        }

        # Create data sample for frontend
        sample_size = min(10, len(combined_data))
        data_sample = {
            "shape": combined_data.shape,
            "columns": combined_data.columns.tolist(),
            "sample_rows": combined_data.head(sample_size).to_dict('records'),
        }

        return DashboardResponse(
            filters=filters,
            charts=chart_config.get('charts', []),
            tables=chart_config.get('tables', []),
            metrics=metrics,
            data_sample=data_sample
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data_summary")
async def get_data_summary():
    """Get summary of available data"""
    if data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    return {
        "tables": list(data.keys()),
        "total_records": sum(len(df) for df in data.values()),
        "columns": {name: df.columns.tolist() for name, df in data.items()}
    }

@app.get("/api/data/{table_name}")
async def get_table_data(table_name: str):
    """Get data for a specific table"""
    if data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    if table_name not in data:
        raise HTTPException(status_code=404, detail=f"Table {table_name} not found")

    return data[table_name].to_dict('records')

@app.get("/")
async def root():
    return {"message": "SAP Dashboard API", "status": "running"}