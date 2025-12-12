// import Papa from 'papaparse';
// import * as d3 from 'd3'; // Not currently used
// import Plotly from 'plotly.js/dist/plotly.min.js'; // Will be imported in components that use it

class SAPDashboardService {
  constructor() {
    this.data = {};
    this.llmApiKey = process.env.REACT_APP_OPENAI_API_KEY;
    this.prompts = {}; // Cache for loaded prompts
  }

  // Load prompt from backend
  async loadPrompt(promptName) {
    if (this.prompts[promptName]) {
      return this.prompts[promptName];
    }
    
    try {
      const response = await fetch(`/api/prompts/${promptName}`);
      if (!response.ok) {
        throw new Error(`Failed to load prompt ${promptName}`);
      }
      const promptText = await response.text();
      this.prompts[promptName] = promptText;
      return promptText;
    } catch (error) {
      console.error(`Error loading prompt ${promptName}:`, error);
      throw error;
    }
  }

  // Load CSV data - now from backend API
  async loadCSVData(filename, url) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      this.data[filename] = data;
      return data;
    } catch (error) {
      console.error(`Error loading ${filename}:`, error);
      throw error;
    }
  }

  // Load all data files - now from backend
  async loadAllData() {
    const dataFiles = [
      { name: 'sales_order', url: '/api/data/sales_order' },
      { name: 'cof_inventory', url: '/api/data/cof_inventory' },
      { name: 'cof_pricing', url: '/api/data/cof_pricing' },
      { name: 'location_sequence', url: '/api/data/location_sequence' }
    ];

    const promises = dataFiles.map(file =>
      this.loadCSVData(file.name, file.url)
    );

    await Promise.all(promises);
    return this.data;
  }

  // Get data summary
  getDataSummary() {
    const summary = {};
    Object.keys(this.data).forEach(key => {
      summary[key] = {
        rowCount: this.data[key].length,
        columns: Object.keys(this.data[key][0] || {})
      };
    });
    return summary;
  }

  // Filter data based on conditions
  filterData(data, filters) {
    return data.filter(row => {
      return Object.entries(filters).every(([column, value]) => {
        return row[column] === value;
      });
    });
  }

  // Generate charts from filtered data
  generateCharts(filteredData, query) {
    const charts = [];
    const columns = Object.keys(filteredData[0] || {});

    // Simple chart generation logic
    if (columns.includes('Plant')) {
      charts.push({
        type: 'bar',
        title: 'Count by Plant',
        x_column: 'Plant',
        y_column: 'count',
        agg_function: 'count',
        limit: 10
      });
    }

    if (columns.includes('Sold-to Name')) {
      charts.push({
        type: 'bar',
        title: 'Count by Customer',
        x_column: 'Sold-to Name',
        y_column: 'count',
        agg_function: 'count',
        limit: 10
      });
    }

    return charts;
  }

  // Process query and generate dashboard using direct LLM calls (two-stage workflow)
  async generateDashboard(query, conversationHistory = []) {
    try {
      console.log('Starting dashboard generation with query:', query);
      console.log('Data loaded:', !!this.data && Object.keys(this.data).length > 0);

      // Stage 1: Filter Extraction
      console.log('Stage 1: Extracting filters from query...');

      // Get columns info from loaded data
      const columnsInfo = this.getColumnsInfo();
      const relationshipInfo = this.getRelationshipInfo();

      // Load filter extraction prompt
      const filterPrompt = await this.loadPrompt('filter_extraction_prompt');

      // Prepare intent extraction payload
      const intentPayload = {
        generation_model: "gpt-4o",
        max_tokens: 500,
        temperature: 0.0,
        top_p: 0.01,
        system_prompt: filterPrompt
          .replace('{columns_info}', columnsInfo)
          .replace('{relationship_info}', relationshipInfo),
        custom_prompt: [{ role: 'user', content: query }],
        model_provider_name: "openai"
      };

      const intentResult = await callLLM(intentPayload);
      console.log('Intent extraction result:', intentResult);

      if (!intentResult) {
        throw new Error('Intent extraction failed: no result');
      }

      // Parse intentResult if it's still a string (handle markdown)
      let parsedIntentResult = intentResult;
      if (typeof intentResult === 'string') {
        let responseText = intentResult;
        if (responseText.includes('```json')) {
          responseText = responseText.split('```json')[1].split('```')[0].trim();
        } else if (responseText.includes('```')) {
          responseText = responseText.split('```')[1].split('```')[0].trim();
        }
        try {
          parsedIntentResult = JSON.parse(responseText);
        } catch (e) {
          console.error('Failed to parse intent result:', e);
          parsedIntentResult = { filters: {}, intent: 'overview' };
        }
      }

      const filters = parsedIntentResult.filters || {};
      const intent = parsedIntentResult.intent || 'overview';

      // Stage 2: Apply filters to data
      console.log('Stage 2: Applying filters to data...');
      let filteredData = this.applyFiltersToData(filters);
      console.log(`Filtered data: ${filteredData.length} records`);

      // Stage 3: Generate chart configurations
      console.log('Stage 3: Generating chart configurations...');

      // Create data sample for chart generation
      const sampleSize = Math.min(10, filteredData.length);
      const dataSample = {
        shape: [filteredData.length, Object.keys(filteredData[0] || {}).length],
        columns: Object.keys(filteredData[0] || {}),
        sample_rows: filteredData.slice(0, sampleSize)
      };

      // Load chart generation prompt
      const chartPrompt = await this.loadPrompt('chart_generation_prompt');

      const chartPayload = {
        generation_model: "gpt-4o",
        max_tokens: 1500,
        temperature: 0.2,
        top_p: 0.01,
        system_prompt: chartPrompt
          .replace('{data_sample}', JSON.stringify(dataSample, null, 2))
          .replace('{all_columns}', Object.keys(filteredData[0] || {}).join(', ')),
        custom_prompt: [{ role: 'user', content: query }],
        model_provider_name: "openai"
      };

      const chartResult = await callLLM(chartPayload);
      console.log('Chart generation result:', chartResult);

      if (!chartResult) {
        throw new Error('Chart generation failed: no result');
      }

      // Parse chartResult if it's still a string (handle markdown)
      let parsedChartResult = chartResult;
      if (typeof chartResult === 'string') {
        let responseText = chartResult;
        if (responseText.includes('```json')) {
          responseText = responseText.split('```json')[1].split('```')[0].trim();
        } else if (responseText.includes('```')) {
          responseText = responseText.split('```')[1].split('```')[0].trim();
        }
        try {
          parsedChartResult = JSON.parse(responseText);
        } catch (e) {
          console.error('Failed to parse chart result:', e);
          parsedChartResult = { charts: [], tables: [] };
        }
      }

      // Transform to match frontend expectations
      const transformedData = {
        filters: filters,
        metrics: {
          total_records: filteredData.length,
          filtered_records: filteredData.length,
          unique_materials: this.calculateUniqueCount(filteredData, 'Material'),
          unique_plants: this.calculateUniqueCount(filteredData, 'Plant')
        },
        charts: parsedChartResult.charts || [],
        tables: parsedChartResult.tables ? parsedChartResult.tables.map(table => ({ ...table, type: 'table' })) : [],
        data_sample: {
          sample_rows: filteredData.slice(0, 50),
          shape: [filteredData.length, Object.keys(filteredData[0] || {}).length]
        },
        follow_up_questions: [], // Could implement later
        summary: `Generated dashboard for ${intent} with ${filteredData.length} records`,
        processing_time: Date.now()
      };

      return transformedData;

    } catch (error) {
      console.error('Error generating dashboard:', error);
      throw error;
    }
  }

  // Get columns information for prompts
  getColumnsInfo() {
    const allColumns = new Set();
    Object.values(this.data).forEach(dataset => {
      if (Array.isArray(dataset) && dataset.length > 0) {
        Object.keys(dataset[0]).forEach(col => allColumns.add(col));
      }
    });

    let info = 'Available Data Sources and Their Columns:\n\n';
    Object.entries(this.data).forEach(([name, dataset]) => {
      if (Array.isArray(dataset) && dataset.length > 0) {
        const columns = Object.keys(dataset[0]);
        info += `${name.toUpperCase()}:\n`;
        columns.forEach(col => info += `  - ${col}\n`);
        info += '\n';
      }
    });

    return info;
  }

  // Get relationship information between tables
  getRelationshipInfo() {
    return `Four tables are available:
1. Sales Order Exception Report - Main sales order data
2. A1P Location Sequence - Plant location and material sequence data  
3. COF Inventory Net Price - Customer order fulfillment pricing data
4. COF Inventory Net Price Material - Material pricing conditions and amounts

Tables can be joined using:
- Sales Order ↔ Location: Plant, Material
- Sales Order ↔ COF Inventory: Sold-To Party, Material, UPC, Pespsi Invenid
- COF Inventory ↔ COF Material Pricing: Material`;
  }

  // Apply filters to loaded data (combine all datasets if needed)
  applyFiltersToData(filters) {
    // For simplicity, start with sales_order data
    let combinedData = [...(this.data.sales_order || [])];

    // Apply filters
    if (Object.keys(filters).length > 0) {
      combinedData = this.filterData(combinedData, filters);
    }

    return combinedData;
  }

  // Helper method to calculate unique count for a column
  calculateUniqueCount(data, columnName) {
    if (!Array.isArray(data) || !columnName) return 0;
    const uniqueValues = new Set(data.map(row => row[columnName]).filter(val => val != null));
    return uniqueValues.size;
  }
}

/* ---------------------------
1) API call (fetch) to LLM backend
- Expects your backend to accept the same payload shape used in Python.
- Replace "/api/llm" with your backend endpoint.
--------------------------- */
export async function callLLM(payload) {
  // payload example fields (see prompt constants below):
  // { generation_model, max_tokens, temperature, top_p,
  //   system_prompt, custom_prompt: [{role, content}], model_provider_name, ... }
  const res = await fetch('/api/llm', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`LLM call failed: ${res.status} ${text}`);
  }
  const json = await res.json();

  let result;
  if (json.response) {
    let responseText = json.response;
    if (typeof responseText === 'string') {
      // Remove markdown code blocks if present
      if (responseText.includes('```json')) {
        responseText = responseText.split('```json')[1].split('```')[0].trim();
      } else if (responseText.includes('```')) {
        responseText = responseText.split('```')[1].split('```')[0].trim();
      }
      try {
        result = JSON.parse(responseText);
      } catch (e) {
        console.error('Failed to parse JSON:', e);
        result = { raw: responseText };
      }
    } else {
      // response is already an object
      result = responseText;
    }
  } else {
    // No response key, json is the result
    result = json;
  }

  // Ensure result has expected keys (similar to Python logic)
  if (result && typeof result === 'object' && !result.filters && !result.intent && !result.charts && !result.tables) {
    // If it's a plain object without expected keys, wrap it
    if (payload.system_prompt && payload.system_prompt.includes('intent')) {
      result = { filters: result };
    } else if (payload.system_prompt && payload.system_prompt.includes('chart')) {
      result = { charts: [], tables: [] };
    }
  }

  return result;
}
/* ---------------------------
2) Prompts (system prompts copied from sap_dashboard_agent.py)
Use these as `system_prompt` in the payload. (Intent/filter & Chart generation)
Source: prompt text from repo. :contentReference[oaicite:2]{index=2}
--------------------------- */

/* Example payload assembly (client-side) before calling callLLM:
const payload = {
  generation_model: "gpt-4o",
  max_tokens: 1500,
  temperature: 0.2,
  top_p: 0.01,
  system_prompt: CHART_SYSTEM_PROMPT.replace('{data_sample}', JSON.stringify(dataSample, null, 2)),
  custom_prompt: [{role: 'user', content: enhancedQuery}],
  model_provider_name: "openai"
};
*/
/* ---------------------------
3) Chart generation using Plotly.js
- Inputs:
  * chartConfig: { type, title, x_column, y_column, group_by/color_by, agg_function, limit, ... }
  * data: array of objects [{col1: val, col2: val, ...}, ...]
  * targetId: DOM id where to render the chart (div)
- Mirrors Python _render_dynamic_chart behaviour: counts, top-N, grouped/stacked, tables.
- Requires: Plotly.js (import Plotly from 'plotly.js-dist-min' or include CDN)
--------------------------- */

// Chart generation system prompt
export const CHART_SYSTEM_PROMPT = `
You are a data visualization expert. Based on the user's query and the filtered data sample, suggest appropriate charts and tables.

Data Sample:
{data_sample}

Return JSON with:
- charts: list of chart configurations, each with:
  * type: "bar", "pie", "line", "scatter", "table", "grouped_bar", "stacked_bar"
  * title: chart title
  * x_column: column for x-axis (for bar, line, scatter)
  * y_column: column for y-axis (for bar, line, scatter) or "count" for count aggregation
  * group_by / color_by: column to group by (optional)
  * agg_function: "count", "sum", "mean", "max", "min"
  * limit: number of top items to show (optional, default 10)
- tables: list of table configurations with:
  * columns: list of column names to display
  * title: table title
  * limit: number of rows (optional, default 50)
`;

/* Example payload assembly (client-side) before calling callLLM:
const payload = {
  generation_model: "gpt-4o",
  max_tokens: 1500,
  temperature: 0.2,
  top_p: 0.01,
  system_prompt: CHART_SYSTEM_PROMPT.replace('{data_sample}', JSON.stringify(dataSample, null, 2)),
  custom_prompt: [{role: 'user', content: enhancedQuery}],
  model_provider_name: "openai"
};
*/
/* ---------------------------
3) Chart generation using Plotly.js
- Inputs:
  * chartConfig: { type, title, x_column, y_column, group_by/color_by, agg_function, limit, ... }
  * data: array of objects [{col1: val, col2: val, ...}, ...]
  * targetId: DOM id where to render the chart (div)
- Mirrors Python _render_dynamic_chart behaviour: counts, top-N, grouped/stacked, tables.
- Requires: Plotly.js (import Plotly from 'plotly.js-dist-min' or include CDN)
--------------------------- */

/** helper: aggregate count or aggregate by function */
function aggregateCount(data, key, limit = 10) {
  const counts = data.reduce((acc, row) => {
    const k = row[key] == null ? 'null' : String(row[key]);
    acc[k] = (acc[k] || 0) + 1;
    return acc;
  }, {});
  const arr = Object.entries(counts).map(([k, v]) => ({ key: k, value: v }));
  arr.sort((a, b) => b.value - a.value);
  return arr.slice(0, limit);
}

function aggregateAgg(data, xKey, yKey, agg = 'sum', limit = 10) {
  // Convert y values to numbers when possible
  const groups = {};
  for (const row of data) {
    const x = row[xKey] == null ? 'null' : String(row[xKey]);
    const yRaw = row[yKey];
    const y = typeof yRaw === 'number' ? yRaw : Number(yRaw) || 0;
    if (!groups[x]) groups[x] = [];
    groups[x].push(y);
  }
  const aggregated = Object.entries(groups).map(([k, vals]) => {
    let value = 0;
    if (agg === 'sum') value = vals.reduce((a, b) => a + b, 0);
    else if (agg === 'mean') value = vals.reduce((a, b) => a + b, 0) / vals.length;
    else if (agg === 'max') value = Math.max(...vals);
    else if (agg === 'min') value = Math.min(...vals);
    else value = vals.length;
    return { key: k, value };
  });
  aggregated.sort((a, b) => b.value - a.value);
  return aggregated.slice(0, limit);
}

/** Render bar chart (count or aggregated) - returns config for React component */
export function renderBarChart(chartConfig, data, targetId) {
  const { title = 'Bar', x_column, y_column, agg_function = 'count', limit = 10 } = chartConfig;
  if (!x_column) throw new Error('x_column required for bar chart');

  let series;
  if (y_column === 'count' || agg_function === 'count') {
    series = aggregateCount(data, x_column, limit);
  } else {
    series = aggregateAgg(data, x_column, y_column, agg_function, limit);
  }

  const x = series.map(d => d.key);
  const y = series.map(d => d.value);

  const trace = { x, y, type: 'bar', marker: { autocolorscale: true } };
  const layout = { title, xaxis: { categoryorder: 'total descending' } };

  // For React component usage, return the config
  return { data: [trace], layout };
}

/** Render grouped or stacked bar chart - returns config for React component */
export function renderGroupedStackedBar(chartConfig, data, targetId) {
  const { title = 'Grouped/Stacked', x_column, color_by, y_column = 'count', limit = 10, limit_groups = 5, type = 'grouped_bar' } = chartConfig;
  if (!x_column || !color_by) throw new Error('x_column and color_by required');

  // Determine top X and top groups
  const topX = aggregateCount(data, x_column, limit).map(d => d.key);
  const topGroups = aggregateCount(data, color_by, limit_groups).map(d => d.key);

  // Filter
  const filtered = data.filter(r => topX.includes(String(r[x_column])) && topGroups.includes(String(r[color_by])));

  // Build map x -> group -> aggregated value
  const map = {};
  for (const row of filtered) {
    const x = String(row[x_column] ?? 'null');
    const g = String(row[color_by] ?? 'null');
    const yVal = y_column === 'count' ? 1 : (typeof row[y_column] === 'number' ? row[y_column] : Number(row[y_column]) || 0);
    map[x] = map[x] || {};
    map[x][g] = (map[x][g] || 0) + yVal;
  }

  // Unique groups in a stable order
  const groups = topGroups;
  const xvals = topX;

  const traces = groups.map(g => {
    const yvals = xvals.map(x => map[x]?.[g] ?? 0);
    return { x: xvals, y: yvals, name: g, type: 'bar' };
  });

  const layout = { title, barmode: type === 'grouped_bar' ? 'group' : 'stack', xaxis: { type: 'category' } };

  // For React component usage, return the config
  return { data: traces, layout };
}

/** Render pie chart - returns config for React component */
export function renderPieChart(chartConfig, data, targetId) {
  const { title = 'Pie', group_by } = chartConfig;
  if (!group_by) throw new Error('group_by required for pie chart');

  const counts = aggregateCount(data, group_by, 1000); // large limit to capture all groups
  const labels = counts.map(d => d.key);
  const values = counts.map(d => d.value);

  const trace = { labels, values, type: 'pie', textinfo: 'label+percent' };
  const layout = { title };

  // For React component usage, return the config
  return { data: [trace], layout };
}

/** Render table (simple HTML table) */
export function renderTable(tableConfig, data, targetId) {
  const { title = 'Table', columns = [], limit = 50 } = tableConfig;
  const displayCols = columns.length ? columns : Object.keys(data[0] || {});
  const rows = data.slice(0, limit);

  const container = document.getElementById(targetId);
  if (!container) throw new Error(`Element ${targetId} not found`);

  // Build simple HTML table
  const html = [];
  html.push(`<h3>${title}</h3>`);
  html.push('<div style="overflow:auto"><table class="llm-table" style="border-collapse:collapse;width:100%;">');
  html.push('<thead><tr>');
  for (const c of displayCols) html.push(`<th style="text-align:left;padding:6px;border-bottom:1px solid #ddd">${c}</th>`);
  html.push('</tr></thead>');
  html.push('<tbody>');
  for (const r of rows) {
    html.push('<tr>');
    for (const c of displayCols) html.push(`<td style="padding:6px;border-bottom:1px solid #f1f1f1">${(r[c] ?? '')}</td>`);
    html.push('</tr>');
  }
  html.push('</tbody></table></div>');
  container.innerHTML = html.join('');
}

/** Utility: dispatch chart rendering depending on config.type - returns config for React component */
export function renderChartFromConfig(chartConfig, data, targetId) {
  const t = chartConfig.type || 'table';
  if (t === 'bar') return renderBarChart(chartConfig, data, targetId);
  if (t === 'grouped_bar' || t === 'stacked_bar') return renderGroupedStackedBar(chartConfig, data, targetId);
  if (t === 'pie') return renderPieChart(chartConfig, data, targetId);
  if (t === 'table') return renderTable(chartConfig, data, targetId);
  // fallback: bar
  return renderBarChart(chartConfig, data, targetId);
}

const sapDashboardService = new SAPDashboardService();
export default sapDashboardService;