/**
 * Exception Handler Helper for SAP Dashboard
 * JavaScript equivalent of exception_handler.py
 * Provides CSV loading, data filtering, LLM-based filter extraction, and chart suggestions
 */

const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const { invoke_llm } = require('./pepsico_llm');
const { get_text_columns } = require('./database_schema');

// Cache for loaded CSV data
let _csv_cache = null;

/**
 * Get the default CSV path relative to this file's location
 * @returns {string} Absolute path to the default CSV file
 */
function _get_default_csv_path() {
    return path.resolve(__dirname, '../../data/Sales Order Exception report.csv');
}

/**
 * Load the exception CSV and cache it in memory
 * @param {string} filepath - Path to CSV file (optional)
 * @returns {Promise<Array>} Array of row objects
 */
async function load_exception_csv(filepath = null) {
    if (_csv_cache) {
        console.log("Using cached CSV data");
        return _csv_cache;
    }

    if (!filepath) {
        filepath = _get_default_csv_path();
    }

    return new Promise((resolve, reject) => {
        const results = [];
        const text_columns = get_text_columns("sales_order");

        // Create column type mapping for text columns
        const dtypeMap = {};
        text_columns.forEach(col => {
            dtypeMap[col] = 'string';
        });

        fs.createReadStream(filepath)
            .pipe(csv({
                mapHeaders: ({ header }) => header.trim(),
                mapValues: ({ header, value }) => {
                    // Convert to string for text columns
                    if (text_columns.includes(header)) {
                        return String(value || '');
                    }
                    return value;
                }
            }))
            .on('data', (data) => results.push(data))
            .on('end', () => {
                console.log(`Loaded ${results.length} rows from ${filepath}`);
                _csv_cache = results;
                resolve(results);
            })
            .on('error', (error) => {
                console.error(`Error loading CSV: ${error.message}`);
                reject(error);
            });
    });
}

/**
 * Return a short human-friendly description of dataframe columns and samples
 * @param {Array} data - Array of row objects
 * @returns {string} Description of columns and sample values
 */
function get_columns_info(data) {
    if (!data || data.length === 0) {
        return "No data available";
    }

    const columns = Object.keys(data[0]);
    let info_lines = [];
    info_lines.push(`Columns (${columns.length}): ${columns.slice(0, 20).join(', ')}${columns.length > 20 ? '...' : ''}`);

    // Helpful sample values for common columns
    const sample_columns = ['Plant', 'Plant(Location)', 'Material', 'Sales Order Number'];
    sample_columns.forEach(key => {
        if (columns.includes(key)) {
            const uniques = [...new Set(data.map(row => row[key]).filter(val => val != null).slice(0, 8))];
            info_lines.push(`Sample values - ${key}: [${uniques.join(', ')}]`);
        }
    });

    return info_lines.join('\n');
}

/**
 * Extract filters from user query using LLM
 * @param {string} query - User query string
 * @param {string} columns_info - Information about available columns
 * @returns {Promise<Object>} Object mapping column names to filter values
 */
async function extract_filters_from_llm(query, columns_info) {
    const prompt_text = (
        "You are a data assistant. Given this user query and the available columns, " +
        "return a JSON object with a single key 'filters' whose value is an object mapping EXACT column names to filter values. " +
        "If no filters can be found, return {\"filters\": {}}. Use only column names present in the schema.\n\n" +
        "Available columns and samples:\n{columns_info}\n\nUser Query:\n{query}\n"
    ).replace('{columns_info}', columns_info).replace('{query}', query);

    try {
        const payload = {
            "generation_model": "gpt-4o",
            "max_tokens": 700,
            "temperature": 0.0,
            "top_p": 0.01,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "tools": [],
            "tools_choice": "none",
            "system_prompt": prompt_text,
            "custom_prompt": [
                {"role": "user", "content": query}
            ],
            "model_provider_name": "openai"
        };

        const resp = await invoke_llm(payload);

        if (resp.error) {
            console.warn("Pepsico LLM error:", resp.error);
            return _fallback_filter_extraction(query);
        }

        // Parse response
        if (resp.response) {
            try {
                let response_text = resp.response;
                // Remove markdown code blocks if present
                if (response_text.includes('```json')) {
                    response_text = response_text.split('```json')[1].split('```')[0].trim();
                } else if (response_text.includes('```')) {
                    response_text = response_text.split('```')[1].split('```')[0].trim();
                }
                const parsed = JSON.parse(response_text);
                if (parsed && parsed.filters) {
                    return parsed.filters || {};
                }
            } catch (e) {
                console.warn("Failed to parse filter JSON:", e.message);
            }
        }

        if (resp.filters) {
            return resp.filters;
        }

    } catch (error) {
        console.warn("Pepsico LLM invocation failed:", error.message);
    }

    // Fallback extraction
    return _fallback_filter_extraction(query);
}

/**
 * Fallback filter extraction using regex patterns
 * @param {string} query - User query string
 * @returns {Object} Extracted filters
 */
function _fallback_filter_extraction(query) {
    const filters = {};

    // Plant extraction
    const plant_match = query.match(/plant\s*[:=]?\s*['"]?(\d{3,6})['"]?/i);
    if (plant_match) {
        filters['Plant'] = plant_match[1];
    } else {
        const plant_match2 = query.match(/plant\s*(?:number\s*)?(\d{3,6})/i);
        if (plant_match2) {
            filters['Plant'] = plant_match2[1];
        }
    }

    // Material extraction
    const material_match = query.match(/material\s*[:=]?\s*['"]?(\w{4,})['"]?/i);
    if (material_match) {
        filters['Material'] = material_match[1];
    }

    return filters;
}

/**
 * Normalize material number by removing leading zeros
 * @param {string} value - Material number string
 * @returns {string} Normalized material number
 */
function _normalize_material_number(value) {
    if (!value) return value;
    const str_value = String(value).trim();
    // Remove leading zeros but keep at least one digit
    return str_value.replace(/^0+/, '') || '0';
}

/**
 * Apply exact-column filters to data array
 * @param {Array} data - Array of row objects
 * @param {Object} filters - Object mapping column names to filter values
 * @returns {Array} Filtered data array
 */
function apply_filters(data, filters) {
    if (!filters || Object.keys(filters).length === 0) {
        return [...data];
    }

    let filtered = [...data];

    // List of columns that should use normalized comparison (remove leading zeros)
    const material_columns = ['Material', 'Customer Material Number', 'Material Found',
                             'INVENID (Order)', 'Pespsi Invenid', 'Inven Id'];

    for (const [col, value] of Object.entries(filters)) {
        if (filtered.length > 0 && col in filtered[0]) {
            // Check if this is a material-related column that needs normalization
            if (material_columns.includes(col)) {
                // Normalize both the filter value and data values
                const normalized_value = _normalize_material_number(String(value));
                filtered = filtered.filter(row =>
                    _normalize_material_number(String(row[col] || '')) === normalized_value
                );
                console.log(`Applied normalized filter: ${col} = ${value} (normalized: ${normalized_value})`);
            } else {
                // Standard exact match for non-material columns
                const str_value = String(value);
                filtered = filtered.filter(row => String(row[col] || '') === str_value);
            }
        } else {
            console.warn(`Requested filter column '${col}' not found in data`);
        }
    }

    return filtered;
}

/**
 * Ask the LLM for chart suggestions given a data sample
 * @param {string} query - User query
 * @param {Object} data_sample - Sample data structure
 * @param {string} columns_info - Information about available columns
 * @returns {Promise<Object>} Chart suggestions with charts and tables arrays
 */
async function suggest_charts_from_llm(query, data_sample, columns_info) {
    const prompt_text = (
        "You are a data visualization expert. Given the user's query, a small data sample, and columns info, " +
        "return JSON with 'charts' (list) and 'tables' (list). Each chart should include type, title, and relevant columns.\n\n" +
        "Columns Info:\n{columns_info}\n\nData Sample:\n{data_sample}\n\nUser Query:\n{query}\n"
    ).replace('{columns_info}', columns_info)
     .replace('{data_sample}', JSON.stringify(data_sample))
     .replace('{query}', query);

    try {
        const payload = {
            "generation_model": "gpt-4o",
            "max_tokens": 1500,
            "temperature": 0.2,
            "top_p": 0.01,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "tools": [],
            "tools_choice": "none",
            "system_prompt": prompt_text,
            "custom_prompt": [
                {"role": "user", "content": `${query}\n\nData Sample:\n${JSON.stringify(data_sample)}`}
            ],
            "model_provider_name": "openai"
        };

        const resp = await invoke_llm(payload);

        if (resp.error) {
            console.warn("Pepsico LLM error for charts:", resp.error);
            return _fallback_chart_suggestions(data_sample);
        }

        // Parse response
        if (resp.response) {
            try {
                let response_text = resp.response;
                // Remove markdown code blocks if present
                if (response_text.includes('```json')) {
                    response_text = response_text.split('```json')[1].split('```')[0].trim();
                } else if (response_text.includes('```')) {
                    response_text = response_text.split('```')[1].split('```')[0].trim();
                }
                const parsed = JSON.parse(response_text);
                if (parsed && typeof parsed === 'object') {
                    return parsed;
                }
            } catch (e) {
                console.warn("Failed to parse chart suggestion JSON:", e.message);
            }
        }

        if (resp && typeof resp === 'object') {
            return resp;
        }

    } catch (error) {
        console.warn("Pepsico LLM chart invocation failed:", error.message);
    }

    // Fallback heuristic suggestions
    return _fallback_chart_suggestions(data_sample);
}

/**
 * Fallback chart suggestions using heuristics
 * @param {Object} data_sample - Sample data structure
 * @returns {Object} Chart suggestions
 */
function _fallback_chart_suggestions(data_sample) {
    const charts = [];
    const columns = data_sample.columns || [];

    if (columns.includes('Plant') || columns.includes('Plant(Location)')) {
        const plant_col = columns.includes('Plant') ? 'Plant' : 'Plant(Location)';
        charts.push({
            'type': 'bar',
            'title': 'Exceptions by Plant',
            'x_column': plant_col,
            'y_column': 'count',
            'agg_function': 'count',
            'limit': 10
        });
    }

    if (columns.includes('Material')) {
        charts.push({
            'type': 'bar',
            'title': 'Top Materials with Exceptions',
            'x_column': 'Material',
            'y_column': 'count',
            'agg_function': 'count',
            'limit': 10
        });
    }

    // Always include a sample table
    charts.push({
        'type': 'table',
        'title': 'Sample Exceptions',
        'columns': columns.slice(0, 8),
        'limit': 50
    });

    return { 'charts': charts, 'tables': [] };
}

module.exports = {
    load_exception_csv,
    get_columns_info,
    extract_filters_from_llm,
    apply_filters,
    suggest_charts_from_llm,
    _normalize_material_number
};