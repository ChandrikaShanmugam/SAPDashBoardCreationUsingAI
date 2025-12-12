/**
 * SAP Dashboard Agent - JavaScript Implementation
 * Dynamic dashboard creation based on natural language queries
 */

import Papa from 'papaparse';
import * as d3 from 'd3';

class SAPDashboardAgent {
  constructor() {
    this.data = {};
    this.llmApiKey = process.env.REACT_APP_OPENAI_API_KEY;
  }

  // Load CSV data
  async loadCSVData(filename, url) {
    try {
      const response = await fetch(url);
      const csvText = await response.text();

      return new Promise((resolve, reject) => {
        Papa.parse(csvText, {
          header: true,
          dynamicTyping: true,
          complete: (results) => {
            this.data[filename] = results.data;
            resolve(results.data);
          },
          error: (error) => {
            reject(error);
          }
        });
      });
    } catch (error) {
      console.error(`Error loading ${filename}:`, error);
      throw error;
    }
  }

  // Load all data files
  async loadAllData() {
    const dataFiles = [
      { name: 'sales_order', url: '/data/Sales Order Exception report.csv' },
      { name: 'cof_inventory', url: '/data/Cof_Inven_NetPrice_Material.csv' },
      { name: 'cof_pricing', url: '/data/Cof_Inven_NetPrice_MaterialPrice.csv' },
      { name: 'location_sequence', url: '/data/A1P_Locn_Seq_EXPORT.csv' }
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

  // Classify intent from query
  classifyIntent(query) {
    const lowerQuery = query.toLowerCase();

    if (lowerQuery.includes('authorized to sell')) {
      return {
        intent: 'authorized_to_sell',
        data_sources: ['sales_order'],
        filters: { 'Auth Sell Flag Description': 'Yes' }
      };
    } else if (lowerQuery.includes('exception')) {
      return {
        intent: 'exceptions',
        data_sources: ['sales_order'],
        filters: {}
      };
    } else if (lowerQuery.includes('plant')) {
      return {
        intent: 'plant_analysis',
        data_sources: ['sales_order'],
        filters: {}
      };
    } else {
      return {
        intent: 'overview',
        data_sources: ['sales_order'],
        filters: {}
      };
    }
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

  // Process query and generate dashboard
  async generateDashboard(query, conversationHistory = []) {
    try {
      // Load data if not already loaded
      if (Object.keys(this.data).length === 0) {
        await this.loadAllData();
      }

      // Classify intent
      const intentResult = this.classifyIntent(query);
      const filters = intentResult.filters;

      // Apply filters
      let filteredData = [];
      if (intentResult.data_sources.includes('sales_order')) {
        filteredData = this.filterData(this.data.sales_order || [], filters);
      }

      // Generate charts
      const charts = this.generateCharts(filteredData, query);

      // Generate table
      const table = {
        type: 'table',
        columns: ['Plant', 'Sold-to Name', 'Material', 'Order Quantity Sales Unit'],
        title: 'Data Summary',
        limit: 50
      };

      return {
        filters,
        charts,
        tables: [table],
        data_sample: filteredData.slice(0, 5),
        metrics: {
          total_records: filteredData.length,
          filtered_from: this.data.sales_order?.length || 0
        }
      };

    } catch (error) {
      console.error('Error generating dashboard:', error);
      throw error;
    }
  }
}

export default SAPDashboardAgent;