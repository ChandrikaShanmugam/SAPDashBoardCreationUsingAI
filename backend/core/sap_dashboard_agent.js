/**
 * SAP Data Dashboard Generator using LLM
 * Dynamic dashboard creation based on natural language queries
 */

const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const { invoke_llm } = require('./pepsico_llm');

// Setup logging
const logger = {
  info: (msg) => console.log(`[${new Date().toISOString()}] INFO: ${msg}`),
  error: (msg) => console.error(`[${new Date().toISOString()}] ERROR: ${msg}`),
  warning: (msg) => console.warn(`[${new Date().toISOString()}] WARNING: ${msg}`),
  debug: (msg) => console.debug(`[${new Date().toISOString()}] DEBUG: ${msg}`)
};

// ===========================
// 1. DATA LOADING FUNCTIONS
// ===========================

/**
 * Load all SAP data files including COF inventory tables
 */
async function loadSapData() {
  logger.info('='.repeat(80));
  logger.info('LOADING SAP DATA FILES (4 TABLES)');
  logger.info('='.repeat(80));

  const startTime = Date.now();

  /**
   * Load CSV with encoding handling
   */
  function loadCsvWithEncoding(filepath) {
    return new Promise((resolve, reject) => {
      const results = [];
      const textColumns = getTextColumns('all');

      fs.createReadStream(filepath)
        .pipe(csv({
          mapHeaders: ({ header }) => header.trim(),
          mapValues: ({ header, value }) => {
            // Convert text columns to strings
            if (textColumns.includes(header)) {
              return String(value || '');
            }
            return value;
          }
        }))
        .on('data', (data) => results.push(data))
        .on('end', () => resolve(results))
        .on('error', reject);
    });
  }

  try {
    const dataDir = path.resolve(__dirname, '..', '..', 'Sap_Dashboard_Creation', 'data');

    // Load Sales Order Exception Report (Table 1)
    logger.info('Loading: Sales Order Exception report.csv');
    const exceptionFile = path.join(dataDir, 'Sales Order Exception report.csv');
    logger.info(`Attempting to load from: ${exceptionFile}`);

    let exceptionReport = await loadCsvWithEncoding(exceptionFile);
    logger.info(`✓ Loaded ${exceptionReport.length} records from Sales Order Exception Report`);
    logger.info(`  Columns: ${Object.keys(exceptionReport[0] || {}).length}`);

    // Normalize Material numbers in exception report (strip leading zeros)
    if (exceptionReport.length > 0 && 'Material' in exceptionReport[0]) {
      logger.info('Normalizing Material numbers in exception report (removing leading zeros)');
      exceptionReport = exceptionReport.map(row => ({
        ...row,
        Material_Original: row.Material,
        Material: String(row.Material).split('.')[0].replace(/^0+/, '') || '0'
      }));
      logger.info(`  Sample normalized: ${JSON.stringify(exceptionReport.slice(0, 3).map(r => ({
        Material_Original: r.Material_Original,
        Material: r.Material
      })))}`);
    }

    // Load A1P Location Sequence (Table 2)
    logger.info('Loading: A1P_Locn_Seq_EXPORT.csv');
    const locationFile = path.join(dataDir, 'A1P_Locn_Seq_EXPORT.csv');
    logger.info(`Attempting to load from: ${locationFile}`);

    let locationSequence = await loadCsvWithEncoding(locationFile);
    logger.info(`✓ Loaded ${locationSequence.length} records from A1P Location Sequence`);
    logger.info(`  Columns: ${Object.keys(locationSequence[0] || {}).length}`);

    // Normalize Material numbers in location sequence (strip leading zeros)
    if (locationSequence.length > 0 && 'Material' in locationSequence[0]) {
      logger.info('Normalizing Material numbers in location sequence (removing leading zeros)');
      locationSequence = locationSequence.map(row => ({
        ...row,
        Material_Original: row.Material,
        Material: String(row.Material).split('.')[0].replace(/^0+/, '') || '0'
      }));
      logger.info(`  Sample normalized: ${JSON.stringify(locationSequence.slice(0, 3).map(r => ({
        Material_Original: r.Material_Original,
        Material: r.Material
      })))}`);
    }

    // Load COF Inventory Net Price (Table 3)
    logger.info('Loading: Cof_Inven_NetPrice_Material.csv');
    const cofInventoryFile = path.join(dataDir, 'Cof_Inven_NetPrice_Material.csv');
    let cofInventory = await loadCsvWithEncoding(cofInventoryFile);
    logger.info(`✓ Loaded ${cofInventory.length} records from COF Inventory Net Price`);
    logger.info(`  Columns: ${Object.keys(cofInventory[0] || {}).length}`);

    // Normalize Material numbers in COF inventory (strip leading zeros)
    if (cofInventory.length > 0 && 'Material' in cofInventory[0]) {
      logger.info('Normalizing Material numbers in COF inventory (removing leading zeros)');
      cofInventory = cofInventory.map(row => ({
        ...row,
        Material_Original: row.Material,
        Material: String(row.Material).split('.')[0].replace(/^0+/, '') || '0'
      }));
      logger.info(`  Sample normalized: ${JSON.stringify(cofInventory.slice(0, 3).map(r => ({
        Material_Original: r.Material_Original,
        Material: r.Material
      })))}`);
    }

    // Load COF Material Pricing (Table 4)
    logger.info('Loading: Cof_Inven_NetPrice_MaterialPrice.csv');
    const cofPricingFile = path.join(dataDir, 'Cof_Inven_NetPrice_MaterialPrice.csv');
    let cofPricing = await loadCsvWithEncoding(cofPricingFile);
    logger.info(`✓ Loaded ${cofPricing.length} records from COF Material Pricing`);
    logger.info(`  Columns: ${Object.keys(cofPricing[0] || {}).length}`);

    // Normalize Material numbers in COF pricing (strip leading zeros)
    if (cofPricing.length > 0 && 'Material' in cofPricing[0]) {
      logger.info('Normalizing Material numbers in COF pricing (removing leading zeros)');
      cofPricing = cofPricing.map(row => ({
        ...row,
        Material_Original: row.Material,
        Material: String(row.Material).split('.')[0].replace(/^0+/, '') || '0'
      }));
      logger.info(`  Sample normalized: ${JSON.stringify(cofPricing.slice(0, 3).map(r => ({
        Material_Original: r.Material_Original,
        Material: r.Material
      })))}`);
    }

    const elapsed = (Date.now() - startTime) / 1000;
    logger.info(`✓ All data loaded successfully in ${elapsed.toFixed(2)} seconds`);
    logger.info('='.repeat(80));

    const data = {
      sales_order: exceptionReport,
      location_sequence: locationSequence,
      cof_inventory: cofInventory,
      cof_pricing: cofPricing
    };
    return data;

  } catch (error) {
    logger.error(`✗ Error loading data: ${error.message}`);
    logger.error(`Full error: ${error.stack}`);
    throw error;
  }
}

// Helper function to get text columns (simplified version)
function getTextColumns(table) {
  // This would need to be implemented based on your database schema
  // For now, return common text columns
  return [
    'Material', 'Material Description', 'Plant', 'Sold-to Name',
    'Ship-to Name', 'Auth Sell Flag Description', 'Material Status',
    'Plant(Location)', 'Location Sequence', 'Storage Location',
    'Sales Organization', 'Distribution Channel', 'Division'
  ];
}

// ===========================
// 2. INTENT CLASSIFICATION
// ===========================

class IntentClassifier {
  /**
   * Classify user intent and extract dashboard requirements with column-aware filtering
   */
  constructor(data) {
    logger.info('Initializing IntentClassifier with PepGenX API');
    this.data = data;

    // Set prompt templates directly (since loadPromptTemplate may fail)
    this.intentSystemPrompt = this.intentPromptTemplate; // Will be set below
    this.chartSystemPrompt = this.chartPromptTemplate; // Will be set below

    this.intentPromptTemplate = `You are a SAP data analyst. Classify the user's query and determine what dashboard to create.

Available Data Sources and Their Columns:
{columns_info}

Return JSON with:
- intent: "authorized_to_sell", "exceptions", "plant_analysis", "material_analysis", or "overview"
- data_sources: list of required data sources ["auth_yes", "auth_no", "so_exceptions"]
- filters: object with EXACT column names as keys and filter values. Examples:
  * {{"Plant": "1007"}} - for specific plant
  * {{"Material": "000000000300005846"}} - for specific material
  * {{"Plant(Location)": "1007"}} - use EXACT column name from schema
- show_material_details: true/false - whether to show detailed material information

IMPORTANT: Use EXACT column names from the schema above when creating filters.

Example queries:
- "Show me plant 1007 authorized data" → {{"intent": "plant_analysis", "filters": {{"Plant(Location)": "1007"}}}}
- "Give me plant 1007 authorized data and material details" → {{"intent": "plant_analysis", "filters": {{"Plant(Location)": "1007"}}, "show_material_details": true}}
- "What are the sales exceptions?" → {{"intent": "exceptions", "filters": {{}}}}

{query}`;

    this.chartPromptTemplate = `You are a data visualization expert. Based on the user's query and the filtered data sample, suggest appropriate charts and tables.

Data Sample:
{data_sample}

Return JSON with:
- charts: list of chart configurations, each with:
  * type: "bar", "pie", "line", "scatter", "table"
  * title: chart title
  * x_column: column for x-axis (for bar, line, scatter)
  * y_column: column for y-axis (for bar, line, scatter) or "count" for count aggregation
  * group_by: column to group by (optional)
  * agg_function: "count", "sum", "mean", "max", "min" (for aggregations)
  * limit: number of top items to show (optional, default 10)
- tables: list of table configurations with:
  * columns: list of column names to display
  * title: table title
  * limit: number of rows (optional, default 50)

Example:
{{
  "charts": [
    {{"type": "bar", "title": "Material Count by Plant", "x_column": "Plant", "y_column": "count", "agg_function": "count", "limit": 10}},
    {{"type": "pie", "title": "Authorization Status", "group_by": "Auth Sell Flag Description", "agg_function": "count"}}
  ],
  "tables": [
    {{"columns": ["Material", "Material Descrption", "Plant", "Auth Sell Flag Description"], "title": "Material Details", "limit": 50}}
  ]
}}

{query}`;

    // Now set the system prompts
    this.intentSystemPrompt = this.intentPromptTemplate;
    this.chartSystemPrompt = this.chartPromptTemplate;
  }

  loadPromptTemplate(name) {
    try {
      const promptDir = path.resolve(__dirname, '..', 'config', 'prompts');
      const filePath = path.join(promptDir, `${name}_prompt.txt`);
      return fs.readFileSync(filePath, 'utf8');
    } catch (error) {
      logger.warning(`Could not load prompt template ${name}: ${error.message}`);
      return '';
    }
  }

  /**
   * Map a possibly-misspelled or user-friendly column name to an actual column
   */
  mapColumnName(requested, availableColumns, cutoff = 0.7) {
    if (!requested) return null;

    // Exact match quick path
    if (availableColumns.includes(requested)) return requested;

    // Normalize: strip, lower
    const req = requested.trim().toLowerCase();
    const candidates = availableColumns.map(col => ({ col, low: col.toLowerCase() }));

    // Direct lower-case match
    for (const { col, low } of candidates) {
      if (low === req) return col;
    }

    // Fuzzy match using simple string similarity
    const close = this.findCloseMatches(req, candidates.map(c => c.low), 1, cutoff);
    if (close.length > 0) {
      const matchedLow = close[0];
      const matched = candidates.find(c => c.low === matchedLow);
      if (matched) {
        logger.info(`Mapped requested column '${requested}' → '${matched.col}' using fuzzy match`);
        return matched.col;
      }
    }

    // Try token-based matching
    const reqTokens = new Set(req.replace('/', ' ').replace('-', ' ').split(' '));
    let best = null;
    let bestScore = 0;

    for (const col of availableColumns) {
      const colTokens = new Set(col.toLowerCase().replace('/', ' ').replace('-', ' ').split(' '));
      const score = this.calculateTokenOverlap(reqTokens, colTokens);
      if (score > bestScore) {
        bestScore = score;
        best = col;
      }
    }

    if (bestScore >= 0.5) {
      logger.info(`Mapped requested column '${requested}' → '${best}' using token overlap (score=${bestScore.toFixed(2)})`);
      return best;
    }

    return null;
  }

  findCloseMatches(word, possibilities, n = 3, cutoff = 0.6) {
    const matches = [];

    for (const possibility of possibilities) {
      const ratio = this.similarityRatio(word, possibility);
      if (ratio >= cutoff) {
        matches.push({ word: possibility, ratio });
      }
    }

    matches.sort((a, b) => b.ratio - a.ratio);
    return matches.slice(0, n).map(m => m.word);
  }

  similarityRatio(s1, s2) {
    const longer = s1.length > s2.length ? s1 : s2;
    const shorter = s1.length > s2.length ? s2 : s1;

    if (longer.length === 0) return 1.0;

    const distance = this.levenshteinDistance(longer, shorter);
    return (longer.length - distance) / longer.length;
  }

  levenshteinDistance(s1, s2) {
    const matrix = [];

    for (let i = 0; i <= s2.length; i++) {
      matrix[i] = [i];
    }

    for (let j = 0; j <= s1.length; j++) {
      matrix[0][j] = j;
    }

    for (let i = 1; i <= s2.length; i++) {
      for (let j = 1; j <= s1.length; j++) {
        if (s2.charAt(i - 1) === s1.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            matrix[i][j - 1] + 1,
            matrix[i - 1][j] + 1
          );
        }
      }
    }

    return matrix[s2.length][s1.length];
  }

  calculateTokenOverlap(tokens1, tokens2) {
    const intersection = new Set([...tokens1].filter(x => tokens2.has(x)));
    const union = new Set([...tokens1, ...tokens2]);
    return intersection.size / union.size;
  }

  /**
   * Extract filters from user query - STAGE 1
   */
  async classify(query, conversationHistory = []) {
    logger.info('='.repeat(80));
    logger.info('STAGE 1: FILTER EXTRACTION FROM QUERY');
    logger.info('='.repeat(80));
    logger.info(`User Query: '${query}'`);

    // Add conversation history context if available
    let historyContext = '';
    if (conversationHistory && conversationHistory.length > 0) {
      historyContext = '\nConversation History:\n' + conversationHistory.slice(-3).map(item =>
        `Previous Query ${conversationHistory.indexOf(item) + 1}: ${item.query}\nApplied Filters: ${JSON.stringify(item.filters)}`
      ).join('\n');
      logger.info(`Using conversation history: ${conversationHistory.length} previous interactions`);
    }

    const startTime = Date.now();

    try {
      logger.info('Building LLM prompt with column information...');

      // Get columns info
      const columnsInfo = this.getColumnsInfo();

      // Format the prompt
      const formattedSystemPrompt = this.intentSystemPrompt || this.intentPromptTemplate.replace('{columns_info}', columnsInfo);
      logger.info(`System prompt length: ${formattedSystemPrompt.length} chars`);
      logger.info('✓ Prompt includes cross-table filtering guidance');

      // Include conversation history in the prompt
      const enhancedQuery = historyContext ? `${query}\n\n${historyContext}` : query;

      const payload = {
        "generation_model": "gpt-4o",
        "max_tokens": 500,
        "temperature": 0.0,
        "top_p": 0.01,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "tools": [],
        "tools_choice": "none",
        "system_prompt": formattedSystemPrompt,
        "custom_prompt": [
          {"role": "user", "content": enhancedQuery}
        ],
        "model_provider_name": "openai"
      };
      logger.info(`Sending enhanced query with history: '${enhancedQuery}'`);
      const resp = await invoke_llm(payload);
      logger.info(`Raw API response: ${JSON.stringify(resp)}`);

      // Try to extract JSON from response
      let result;
      if (resp && resp.error) {
        throw new Error(resp.error);
      }

      // If API returns JSON with 'response' field, parse it
      if (resp && resp.response) {
        try {
          let responseText = resp.response;
          // Remove markdown code blocks if present
          if (responseText.includes('```json')) {
            responseText = responseText.split('```json')[1].split('```')[0].trim();
          } else if (responseText.includes('```')) {
            responseText = responseText.split('```')[1].split('```')[0].trim();
          }
          result = JSON.parse(responseText);
          logger.info(`Successfully parsed JSON: ${JSON.stringify(result)}`);
        } catch (parseError) {
          logger.error(`Failed to parse JSON: ${parseError.message}`);
          result = { filters: {} };
        }
      } else {
        result = resp;
      }

      const elapsed = (Date.now() - startTime) / 1000;
      logger.info(`✓ Filter extraction completed in ${elapsed.toFixed(2)} seconds`);
      logger.info(`Extracted intent: ${result.intent}`);
      logger.info(`Filters: ${JSON.stringify(result.filters)}`);

      return result;

    } catch (error) {
      logger.error(`✗ Error in filter extraction: ${error.message}`);
      throw error;
    }
  }

  getColumnsInfo() {
    // Simplified version - would need to implement based on your database schema
    const allColumns = new Set();

    for (const [tableName, tableData] of Object.entries(this.data)) {
      if (tableData && tableData.length > 0) {
        Object.keys(tableData[0]).forEach(col => allColumns.add(col));
      }
    }

    return `All Available Columns (${allColumns.size} total):\n${Array.from(allColumns).join(', ')}`;
  }

  /**
   * Generate chart configuration based on filtered data
   */
  async generateChartConfig(query, dataSample, conversationHistory = []) {
    logger.info('Generating chart configuration...');

    try {
      // Format the chart prompt with dynamic columns from actual data
      const formattedSystemPrompt = this.chartSystemPrompt || this.chartPromptTemplate.replace('{data_sample}', JSON.stringify(dataSample));

      // Include conversation history in the query
      const enhancedQuery = conversationHistory && conversationHistory.length > 0 ?
        `${query}\n\nConversation History:\n${conversationHistory.slice(-3).map(item =>
          `Previous Query: ${item.query}\nApplied Filters: ${JSON.stringify(item.filters)}`
        ).join('\n')}\n\nData Sample:\n${JSON.stringify(dataSample)}` :
        `${query}\n\nData Sample:\n${JSON.stringify(dataSample)}`;

      const payload = {
        "generation_model": "gpt-4o",
        "max_tokens": 1500,
        "temperature": 0.2,
        "top_p": 0.01,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "tools": [],
        "tools_choice": "none",
        "system_prompt": formattedSystemPrompt,
        "custom_prompt": [
          {"role": "user", "content": enhancedQuery}
        ],
        "model_provider_name": "openai"
      };

      const resp = await invoke_llm(payload);
      if (resp && resp.error) {
        throw new Error(resp.error);
      }

      let result;
      if (resp && resp.response) {
        try {
          let responseText = resp.response;
          // Remove markdown code blocks if present
          if (responseText.includes('```json')) {
            responseText = responseText.split('```json')[1].split('```')[0].trim();
          } else if (responseText.includes('```')) {
            responseText = responseText.split('```')[1].split('```')[0].trim();
          }
          result = JSON.parse(responseText);
        } catch (parseError) {
          logger.error(`Failed to parse chart config JSON: ${parseError.message}`);
          result = { charts: [], tables: [] };
        }
      } else {
        result = resp;
      }

      logger.info(`Generated ${result.charts ? result.charts.length : 0} charts and ${result.tables ? result.tables.length : 0} tables`);
      return result;

    } catch (error) {
      logger.error(`Error generating chart config: ${error.message}`);
      throw error;
    }
  }
}

// ===========================
// 3. DASHBOARD GENERATOR
// ===========================

class DashboardGenerator {
  /**
   * Generate dynamic dashboards based on intent with intelligent filtering
   */
  constructor(data, classifier) {
    this.data = data;
    this.classifier = classifier;
    logger.info('DashboardGenerator initialized');
    logger.info(`Available datasets: ${Object.keys(data).join(', ')}`);
    for (const [name, tableData] of Object.entries(data)) {
      logger.info(`  - ${name}: ${tableData.length} records`);
    }
  }

  /**
   * Apply filters to array of objects
   */
  applyFilters(data, filters) {
    if (!filters || Object.keys(filters).length === 0) {
      return data;
    }

    logger.info(`Applying filters: ${JSON.stringify(filters)}`);

    let filteredData = [...data];

    for (const [col, value] of Object.entries(filters)) {
      // Try to map fuzzy column names
      let useCol = col;
      if (this.classifier) {
        const mappedCol = this.classifier.mapColumnName(col, Object.keys(filteredData[0] || {}));
        if (mappedCol) useCol = mappedCol;
      }

      if (filteredData.length > 0 && useCol in filteredData[0]) {
        // Handle both single values and arrays
        if (Array.isArray(value)) {
          const valueStrings = value.map(v => String(v));
          filteredData = filteredData.filter(row => valueStrings.includes(String(row[useCol])));
          logger.info(`  ✓ Filtered by ${useCol} in ${value}, remaining rows: ${filteredData.length}`);
        } else {
          filteredData = filteredData.filter(row => String(row[useCol]) === String(value));
          logger.info(`  ✓ Filtered by ${useCol} = ${value}, remaining rows: ${filteredData.length}`);
        }
      } else {
        logger.warning(`  ⚠️ Column '${col}' not found in data (tried mapping to '${useCol}')`);
      }
    }

    return filteredData;
  }

  /**
   * Apply filters with support for cross-table queries
   */
  applyCrossTableFilters(filterResult) {
    const filters = filterResult.filters || {};
    const requiresJoin = filterResult.requires_join || false;
    const joinOn = filterResult.join_on || ['Plant', 'Material'];

    logger.info(`Cross-table filtering: requires_join=${requiresJoin}`);

    // Start with Sales Order data
    let salesDf = [...this.data.sales_order];

    if (!requiresJoin) {
      // Simple single-table filtering
      logger.info('Single-table query - filtering Sales Order table only');
      return this.applyFilters(salesDf, filters);
    }

    // Cross-table query - need to join
    logger.info('Cross-table query detected - performing JOIN');
    let locationDf = [...this.data.location_sequence];

    // Separate filters by table
    const salesFilters = {};
    const locationFilters = {};

    // Get column names from both tables
    const salesCols = new Set(salesDf.length > 0 ? Object.keys(salesDf[0]) : []);
    const locationCols = new Set(locationDf.length > 0 ? Object.keys(locationDf[0]) : []);

    for (const [col, value] of Object.entries(filters)) {
      if (salesCols.has(col)) {
        salesFilters[col] = value;
        logger.info(`  Filter for Sales Order table: ${col} = ${value}`);
      } else if (locationCols.has(col)) {
        locationFilters[col] = value;
        logger.info(`  Filter for Location table: ${col} = ${value}`);
      } else {
        logger.warning(`  Column '${col}' not found in either table`);
      }
    }

    // Apply filters to each table before joining
    if (Object.keys(salesFilters).length > 0) {
      salesDf = this.applyFilters(salesDf, salesFilters);
      logger.info(`Sales Order filtered: ${salesDf.length} rows`);
    }

    if (Object.keys(locationFilters).length > 0) {
      locationDf = this.applyFilters(locationDf, locationFilters);
      logger.info(`Location Sequence filtered: ${locationDf.length} rows`);
    }

    // Perform the JOIN
    logger.info(`Joining on: Sales.Plant = Location.Plant(Location) AND Sales.Material = Location.Material`);

    const mergedDf = [];
    const salesMap = new Map();

    // Create lookup map for location data
    const locationMap = new Map();
    for (const locRow of locationDf) {
      const key = `${locRow['Plant(Location)']}_${locRow.Material}`;
      locationMap.set(key, locRow);
    }

    // Join sales with location
    for (const salesRow of salesDf) {
      const key = `${salesRow.Plant}_${salesRow.Material}`;
      const locRow = locationMap.get(key);

      if (locRow) {
        mergedDf.push({
          ...salesRow,
          ...locRow,
          _is_joined: true
        });
      }
    }

    logger.info(`✓ Join complete: ${mergedDf.length} rows`);

    return mergedDf;
  }

  /**
   * Generate dashboard using two-stage workflow
   */
  async generate(filterResult, userQuery, conversationHistory = []) {
    logger.info('='.repeat(80));
    logger.info('APPLYING FILTERS AND GENERATING DASHBOARD');
    logger.info('='.repeat(80));

    try {
      // Apply filters to get filtered data
      const filteredData = this.applyCrossTableFilters(filterResult);

      logger.info(`✓ Filtered data contains ${filteredData.length} records`);

      // Generate chart configuration using LLM
      const chartConfig = await this.classifier.generateChartConfig(
        userQuery,
        filteredData.slice(0, 10), // Sample for LLM
        conversationHistory
      );

      return {
        data: filteredData,
        charts: chartConfig.charts || [],
        tables: chartConfig.tables || [],
        summary: {
          totalRecords: filteredData.length,
          filters: filterResult.filters,
          intent: filterResult.intent
        }
      };

    } catch (error) {
      logger.error(`Error generating dashboard: ${error.message}`);
      throw error;
    }
  }

  /**
   * Create charts using Plotly.js
   */
  async createCharts(data, config) {
    const charts = [];
    const tables = [];

    // Process chart configurations
    if (config.charts) {
      for (const chartSpec of config.charts) {
        try {
          const chart = this.createChart(data, chartSpec);
          if (chart) charts.push(chart);
        } catch (error) {
          logger.error(`Error creating chart ${chartSpec.title}: ${error.message}`);
        }
      }
    }

    // Process table configurations
    if (config.tables) {
      for (const tableSpec of config.tables) {
        try {
          const table = this.createTable(data, tableSpec);
          if (table) tables.push(table);
        } catch (error) {
          logger.error(`Error creating table ${tableSpec.title}: ${error.message}`);
        }
      }
    }

    return { charts, tables };
  }

  /**
   * Create a single chart using Plotly.js
   */
  createChart(data, spec) {
    const { type, title, x_column, y_column, group_by, agg_function = 'count', limit = 10 } = spec;

    if (!data || data.length === 0) {
      logger.warning(`No data available for chart: ${title}`);
      return null;
    }

    let plotData = [];

    try {
      switch (type) {
        case 'bar':
          plotData = this.createBarChart(data, spec);
          break;
        case 'pie':
          plotData = this.createPieChart(data, spec);
          break;
        case 'line':
          plotData = this.createLineChart(data, spec);
          break;
        case 'scatter':
          plotData = this.createScatterChart(data, spec);
          break;
        default:
          logger.warning(`Unsupported chart type: ${type}`);
          return null;
      }

      return {
        title,
        type,
        data: plotData,
        layout: {
          title: title,
          showlegend: true
        }
      };

    } catch (error) {
      logger.error(`Error creating ${type} chart: ${error.message}`);
      return null;
    }
  }

  createBarChart(data, spec) {
    const { x_column, y_column, agg_function, limit } = spec;

    if (!x_column || !y_column) return [];

    // Group and aggregate data
    const grouped = this.groupAndAggregate(data, x_column, y_column, agg_function);

    // Sort and limit
    const sorted = Object.entries(grouped)
      .sort(([,a], [,b]) => b - a)
      .slice(0, limit);

    return [{
      type: 'bar',
      x: sorted.map(([key]) => key),
      y: sorted.map(([,value]) => value),
      name: y_column
    }];
  }

  createPieChart(data, spec) {
    const { group_by, agg_function = 'count', limit = 10 } = spec;

    if (!group_by) return [];

    // Group and aggregate data
    const grouped = this.groupAndAggregate(data, group_by, null, agg_function);

    // Sort and limit
    const sorted = Object.entries(grouped)
      .sort(([,a], [,b]) => b - a)
      .slice(0, limit);

    return [{
      type: 'pie',
      labels: sorted.map(([key]) => key),
      values: sorted.map(([,value]) => value),
      name: group_by
    }];
  }

  createLineChart(data, spec) {
    const { x_column, y_column, agg_function, limit } = spec;

    if (!x_column || !y_column) return [];

    // For line charts, we need to sort by x_column
    const grouped = this.groupAndAggregate(data, x_column, y_column, agg_function);

    // Sort by x values (assuming they can be sorted)
    const sorted = Object.entries(grouped)
      .sort(([a], [b]) => String(a).localeCompare(String(b)))
      .slice(0, limit);

    return [{
      type: 'scatter',
      mode: 'lines+markers',
      x: sorted.map(([key]) => key),
      y: sorted.map(([,value]) => value),
      name: y_column
    }];
  }

  createScatterChart(data, spec) {
    const { x_column, y_column, limit = 1000 } = spec;

    if (!x_column || !y_column) return [];

    // Take a sample for scatter plots
    const sample = data.slice(0, limit);

    return [{
      type: 'scatter',
      mode: 'markers',
      x: sample.map(row => row[x_column]),
      y: sample.map(row => row[y_column]),
      name: `${x_column} vs ${y_column}`
    }];
  }

  groupAndAggregate(data, groupColumn, valueColumn, aggFunction) {
    const groups = {};

    for (const row of data) {
      const groupKey = String(row[groupColumn] || 'Unknown');
      const value = valueColumn ? parseFloat(row[valueColumn]) || 0 : 1;

      if (!groups[groupKey]) {
        groups[groupKey] = [];
      }
      groups[groupKey].push(value);
    }

    const result = {};
    for (const [key, values] of Object.entries(groups)) {
      switch (aggFunction) {
        case 'count':
          result[key] = values.length;
          break;
        case 'sum':
          result[key] = values.reduce((a, b) => a + b, 0);
          break;
        case 'mean':
        case 'avg':
          result[key] = values.reduce((a, b) => a + b, 0) / values.length;
          break;
        case 'max':
          result[key] = Math.max(...values);
          break;
        case 'min':
          result[key] = Math.min(...values);
          break;
        default:
          result[key] = values.length; // default to count
      }
    }

    return result;
  }

  /**
   * Create a table
   */
  createTable(data, spec) {
    const { columns, title, limit = 50 } = spec;

    if (!columns || !Array.isArray(columns)) {
      logger.warning(`Invalid table spec: ${JSON.stringify(spec)}`);
      return null;
    }

    // Take limited rows
    const tableData = data.slice(0, limit);

    // Extract specified columns
    const headers = columns;
    const rows = tableData.map(row =>
      columns.map(col => String(row[col] || ''))
    );

    return {
      title,
      type: 'table',
      headers,
      rows,
      totalRows: data.length
    };
  }
}

// ===========================
// FOLLOW-UP QUESTION GENERATION
// ===========================

/**
 * Generate follow-up questions based on user query and conversation history
 */
async function generateFollowUpQuestions(userQuery, conversationHistory, dataSummary) {
  logger.info('Generating follow-up questions...');

  // Create context from conversation history
  let historyContext = '';
  if (conversationHistory && conversationHistory.length > 0) {
    historyContext = '\nPrevious conversation:\n' + conversationHistory.slice(-3).map((item, i) =>
      `User: ${item.query}\nAssistant: Generated dashboard for ${item.query}`
    ).join('\n');
  }

  // Create data summary context
  const dataContext = `Available data summary:
- Total records: ${dataSummary.totalRecords || 'Unknown'}
- Tables: ${Object.keys(dataSummary.tables || {}).join(', ')}
- Key columns: Plant, Material, Auth Sell Flag Description, Material Status, etc.`;

  const followUpPrompt = `You are a SAP data analyst helping users explore their data. Based on the user's current query and conversation history, suggest 4 relevant follow-up questions that would help them gain deeper insights.

Current user query: "${userQuery}"

${historyContext}

${dataContext}

Generate 4 follow-up questions that are:
1. Specific and actionable
2. Related to the current analysis
3. Help explore different aspects of the data
4. Use natural language

Return only a JSON array of 4 question strings, no additional text.

Example format:
["What are the top materials by quantity?", "Show me authorization issues by plant", "How many active materials do we have?", "What are the sales exceptions?"]`;

  try {
    const payload = {
      "generation_model": "gpt-4o",
      "max_tokens": 300,
      "temperature": 0.7,
      "top_p": 0.9,
      "presence_penalty": 0,
      "frequency_penalty": 0,
      "tools": [],
      "tools_choice": "none",
      "system_prompt": "You are a helpful assistant that generates relevant follow-up questions for data analysis.",
      "custom_prompt": [
        {"role": "user", "content": followUpPrompt}
      ],
      "model_provider_name": "openai"
    };

    const resp = await invoke_llm(payload);

    if (resp && resp.error) {
      throw new Error(resp.error);
    }

    let questions;
    if (resp && resp.response) {
      let responseText = resp.response;
      // Clean up response
      if (responseText.includes('```json')) {
        responseText = responseText.split('```json')[1].split('```')[0].trim();
      } else if (responseText.includes('```')) {
        responseText = responseText.split('```')[1].split('```')[0].trim();
      }

      questions = JSON.parse(responseText);
    } else {
      questions = resp;
    }

    if (Array.isArray(questions) && questions.length >= 4) {
      logger.info(`Generated ${questions.length} follow-up questions`);
      return questions.slice(0, 4); // Return first 4
    } else {
      logger.warning('Invalid follow-up questions format, using defaults');
      return getDefaultFollowUpQuestions();
    }

  } catch (error) {
    logger.error(`Error generating follow-up questions: ${error.message}`);
    return getDefaultFollowUpQuestions();
  }
}

function getDefaultFollowUpQuestions() {
  return [
    'Show me authorized to sell details',
    'What are the sales exceptions?',
    'Give me plant-wise analysis',
    'Show overview of all data'
  ];
}

module.exports = {
  loadSapData,
  IntentClassifier,
  DashboardGenerator,
  generateFollowUpQuestions
};