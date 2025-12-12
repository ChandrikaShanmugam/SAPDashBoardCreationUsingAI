/**
 * SAP Dashboard Backend Server
 * Node.js/Express API using converted JavaScript modules
 */

const express = require('express');
const cors = require('cors');
const path = require('path');

// Import converted JavaScript modules
const { loadSapData, IntentClassifier, DashboardGenerator, generateFollowUpQuestions } = require('./core/sap_dashboard_agent');
const { invoke_llm } = require('./core/pepsico_llm');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Global variables for data and models
let data = null;
let classifier = null;
let dashboardGen = null;
let dashboardService = null;

// Initialize services on startup
async function initializeServices() {
    try {
        console.log('Loading SAP data...');
        data = await loadSapData();
        console.log('Initializing classifier and dashboard generator...');
        classifier = new IntentClassifier(data);
        dashboardGen = new DashboardGenerator(data, classifier);
        dashboardService = new DashboardService();
        console.log('✅ API initialized successfully');
    } catch (error) {
        console.error('❌ Error initializing API:', error);
        throw error;
    }
}

class DashboardService {
    constructor() {
        this.classifier = classifier;
        this.dashboardGen = dashboardGen;
        this.data = data;
    }

    async processQuery(query, conversation_history = []) {
        try {
            console.log(`Processing dashboard request: "${query}"`);

            // Extract filters using IntentClassifier
            const filterResult = await this.classifier.classify(query, conversation_history);
            console.log('Extracted filters:', JSON.stringify(filterResult.filters, null, 2));

            // Generate dashboard using DashboardGenerator
            const dashboard = await this.dashboardGen.generate(filterResult, query, conversation_history);

            // Generate follow-up questions
            const followUpQuestions = await generateFollowUpQuestions(query, conversation_history, {
                totalRecords: dashboard.data.length,
                tables: Object.keys(this.data)
            });

            return {
                success: true,
                query: query,
                filters: filterResult,
                total_records: this.data.sales_order.length,
                filtered_records: dashboard.data.length,
                charts: dashboard.charts || [],
                tables: dashboard.tables || [],
                data_sample: dashboard.data.slice(0, 50),
                follow_up_questions: followUpQuestions,
                processing_time: Date.now(),
                summary: dashboard.summary
            };

        } catch (error) {
            console.error('Dashboard processing error:', error);
            return {
                success: false,
                error: error.message,
                query: query
            };
        }
    }
}

// Initialize services on startup
initializeServices().catch(console.error);

// API Routes
app.post('/api/dashboard/generate', async (req, res) => {
    try {
        const { query, conversation_history = [] } = req.body;

        if (!query) {
            return res.status(400).json({ error: 'Query is required' });
        }

        const result = await dashboardService.processQuery(query, conversation_history);
        res.json(result);

    } catch (error) {
        console.error('API error:', error);
        res.status(500).json({
            success: false,
            error: 'Internal server error',
            message: error.message
        });
    }
});

app.get('/api/dashboard/follow-up-questions', async (req, res) => {
    try {
        const { query, conversation_history = [] } = req.query;

        if (!query) {
            return res.status(400).json({ error: 'Query is required' });
        }

        // Generate follow-up questions using the LLM
        const dataSummary = {
            totalRecords: data ? data.sales_order.length : 0,
            tables: data ? Object.keys(data) : []
        };

        const questions = await generateFollowUpQuestions(query, conversation_history, dataSummary);
        res.json(questions);

    } catch (error) {
        console.error('Follow-up questions error:', error);
        res.status(500).json({
            error: 'Failed to generate follow-up questions',
            message: error.message
        });
    }
});

app.get('/api/dashboard/data-summary', async (req, res) => {
    try {
        if (!data) {
            return res.status(503).json({ error: 'Data not loaded yet' });
        }

        const summary = {
            total_records: data.sales_order.length,
            columns: data.sales_order.length > 0 ? Object.keys(data.sales_order[0]) : [],
            tables: Object.keys(data),
            table_counts: Object.fromEntries(
                Object.entries(data).map(([key, tableData]) => [key, tableData.length])
            ),
            key_columns: ['Plant', 'Material', 'Auth Sell Flag Description', 'Sold-to Name'],
            sample_data: data.sales_order.slice(0, 5)
        };

        res.json(summary);

    } catch (error) {
        console.error('Data summary error:', error);
        res.status(500).json({
            error: 'Failed to get data summary',
            message: error.message
        });
    }
});

// API endpoint to get table data as JSON
app.get('/api/data/:tableName', async (req, res) => {
    try {
        const { tableName } = req.params;

        if (!data) {
            return res.status(503).json({ error: 'Data not loaded yet' });
        }

        if (!data[tableName]) {
            return res.status(404).json({ error: `Table ${tableName} not found` });
        }

        res.json(data[tableName]);

    } catch (error) {
        console.error('Table data error:', error);
        res.status(500).json({
            error: 'Failed to get table data',
            message: error.message
        });
    }
});

// LLM API endpoint for direct LLM calls
app.post('/api/llm', async (req, res) => {
    try {
        const payload = req.body;

        if (!payload) {
            return res.status(400).json({ error: 'Payload is required' });
        }

        console.log('Calling LLM with payload:', JSON.stringify(payload, null, 2));
        const result = await invoke_llm(payload);
        console.log('LLM response:', result);

        res.json(result);

    } catch (error) {
        console.error('LLM API error:', error);
        res.status(500).json({
            error: 'LLM call failed',
            message: error.message
        });
    }
});

// Health check
app.get('/api/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        services: {
            llm: 'PepGenX API',
            database: 'CSV files',
            charts: 'LLM-generated'
        }
    });
});

// Serve prompt files
app.get('/api/prompts/:promptName', (req, res) => {
    const promptName = req.params.promptName;
    const promptPath = path.join(__dirname, 'config', 'prompts', `${promptName}.txt`);
    
    if (!require('fs').existsSync(promptPath)) {
        return res.status(404).json({ error: `Prompt ${promptName} not found` });
    }
    
    res.sendFile(promptPath);
});

// Serve static files from data directory
app.use('/api/data', express.static(path.join(__dirname, '../data')));

// Start server
initializeServices().then(() => {
    app.listen(PORT, () => {
        console.log(`🚀 SAP Dashboard Backend API running on port ${PORT}`);
        console.log(`📊 Health check: http://localhost:${PORT}/api/health`);
        console.log(`📱 API endpoints:`);
        console.log(`   POST /api/dashboard/generate - Generate dashboard from query`);
        console.log(`   POST /api/llm - Direct LLM API calls`);
        console.log(`   GET  /api/dashboard/follow-up-questions - Get follow-up questions`);
        console.log(`   GET  /api/dashboard/data-summary - Get data summary`);
    });
}).catch((error) => {
    console.error('❌ Failed to initialize services:', error);
    process.exit(1);
});

module.exports = app;