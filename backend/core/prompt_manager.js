/**
 * Prompt Template Manager
 * JavaScript equivalent of prompt_manager.py
 * Manages loading and formatting of prompt templates from files
 */

const fs = require('fs');
const path = require('path');
const {
    get_foreign_key_relationships,
    get_common_columns,
    get_cof_common_columns,
    get_all_sales_order_columns,
    get_all_location_sequence_columns,
    get_all_cof_inventory_columns,
    get_all_cof_material_columns
} = require('./database_schema');

class PromptTemplateManager {
    /**
     * Initialize prompt manager
     * @param {string} prompt_dir - Directory containing prompt template files
     */
    constructor(prompt_dir = null) {
        if (prompt_dir === null) {
            // Default to prompts directory relative to this file
            prompt_dir = path.resolve(__dirname, '../config/prompts');
        }

        this.prompt_dir = prompt_dir;
        this.templates = {};
        this._load_templates();
    }

    /**
     * Load all prompt templates from directory
     */
    _load_templates() {
        if (!fs.existsSync(this.prompt_dir)) {
            console.warn(`Prompt directory not found: ${this.prompt_dir}`);
            return;
        }

        const template_files = {
            'filter_extraction': 'filter_extraction_prompt.txt',
            'chart_generation': 'chart_generation_prompt.txt'
        };

        for (const [name, filename] of Object.entries(template_files)) {
            const filepath = path.join(this.prompt_dir, filename);
            if (fs.existsSync(filepath)) {
                try {
                    this.templates[name] = fs.readFileSync(filepath, 'utf-8');
                    console.log(`Loaded template: ${name} from ${filename}`);
                } catch (error) {
                    console.error(`Error loading template ${name}: ${error.message}`);
                }
            } else {
                console.warn(`Template file not found: ${filepath}`);
            }
        }
    }

    /**
     * Get a prompt template by name
     * @param {string} template_name - Name of the template
     * @returns {string} Template string
     */
    get_template(template_name) {
        return this.templates[template_name] || "";
    }

    /**
     * Get and format a template with variables
     * @param {string} template_name - Name of the template
     * @param {Object} kwargs - Variables to format into the template
     * @returns {string} Formatted template string
     */
    format_template(template_name, kwargs = {}) {
        let template = this.get_template(template_name);
        if (!template) {
            console.error(`Template not found: ${template_name}`);
            return "";
        }

        try {
            // Replace {variable} placeholders with values
            for (const [key, value] of Object.entries(kwargs)) {
                const placeholder = new RegExp(`{${key}}`, 'g');
                template = template.replace(placeholder, value);
            }
            return template;
        } catch (error) {
            console.error(`Template formatting error: ${error.message}`);
            return template;
        }
    }

    /**
     * Add a custom template at runtime
     * @param {string} name - Template name
     * @param {string} template_content - Template string
     */
    add_custom_template(name, template_content) {
        this.templates[name] = template_content;
        console.log(`Added custom template: ${name}`);
    }

    /**
     * Get formatted relationship information for prompts
     * @returns {string} Formatted string describing table relationships
     */
    get_relationship_info() {
        const fk = get_foreign_key_relationships();
        const common_cols = get_common_columns();
        const cof_cols = get_cof_common_columns();

        const info_parts = [];
        info_parts.push("Four tables are available:");
        info_parts.push("1. Sales Order Exception Report (69 columns) - Main sales order data");
        info_parts.push("2. A1P Location Sequence (15 columns) - Plant location and material sequence data");
        info_parts.push("3. COF Inventory Net Price (7 columns) - Customer order fulfillment pricing data");
        info_parts.push("4. COF Inventory Net Price Material (11 columns) - Material pricing conditions and amounts");
        info_parts.push("");
        info_parts.push("Tables can be joined using:");
        info_parts.push(`- Sales Order ↔ Location: ${common_cols.join(', ')}`);
        info_parts.push(`- Sales Order ↔ COF Inventory: ${cof_cols.join(', ')}`);
        info_parts.push("- COF Inventory ↔ COF Material Pricing: Material");
        info_parts.push("");

        // Sales Order to Location relationship
        const relationship = fk["sales_order_to_location"];
        if (relationship) {
            info_parts.push("Foreign Key Relationships:");
            info_parts.push("[Sales Order → A1P Location]");
            for (const rel of relationship.relationships || []) {
                info_parts.push(`  - ${rel.from_column} → ${rel.to_column} [${rel.relationship_type}]`);
            }
        }

        // Sales Order to COF Inventory relationship
        const cof_relationship = fk["sales_order_to_cof_inventory"];
        if (cof_relationship) {
            info_parts.push("");
            info_parts.push("[Sales Order → COF Inventory]");
            for (const rel of cof_relationship.relationships || []) {
                info_parts.push(`  - ${rel.from_column} → ${rel.to_column} [${rel.relationship_type}]`);
            }
        }

        return info_parts.join("\n");
    }

    /**
     * Get formatted column information for all tables
     * @returns {string} Formatted string with column information
     */
    get_columns_info() {
        const info_parts = [];

        // Sales Order columns
        info_parts.push("SALES ORDER EXCEPTION REPORT COLUMNS:");
        const sales_cols = get_all_sales_order_columns();
        info_parts.push(`Total: ${sales_cols.length} columns`);
        info_parts.push(`Key columns: ${sales_cols.slice(0, 10).join(', ')}... (and ${sales_cols.length - 10} more)`);
        info_parts.push("");

        // Location Sequence columns
        info_parts.push("A1P LOCATION SEQUENCE COLUMNS:");
        const loc_cols = get_all_location_sequence_columns();
        info_parts.push(`Total: ${loc_cols.length} columns`);
        info_parts.push(`All columns: ${loc_cols.join(', ')}`);
        info_parts.push("");

        // COF Inventory columns
        info_parts.push("COF INVENTORY NET PRICE COLUMNS:");
        const cof_cols = get_all_cof_inventory_columns();
        info_parts.push(`Total: ${cof_cols.length} columns`);
        info_parts.push(`All columns: ${cof_cols.join(', ')}`);
        info_parts.push("");

        // COF Material columns
        info_parts.push("COF INVENTORY NET PRICE MATERIAL COLUMNS (Material Pricing):");
        const cof_mat_cols = get_all_cof_material_columns();
        info_parts.push(`Total: ${cof_mat_cols.length} columns`);
        info_parts.push(`All columns: ${cof_mat_cols.join(', ')}`);

        return info_parts.join("\n");
    }

    /**
     * Format the filter extraction prompt with schema information
     * @param {string} user_query - User's natural language query
     * @returns {string} Formatted prompt ready for LLM
     */
    format_filter_extraction_prompt(user_query) {
        const template = this.get_template("filter_extraction");
        if (!template) {
            console.error("Filter extraction template not found");
            return "";
        }

        try {
            return this.format_template("filter_extraction", {
                columns_info: this.get_columns_info(),
                relationship_info: this.get_relationship_info(),
                user_query: user_query
            });
        } catch (error) {
            console.error(`Missing template variable in filter_extraction: ${error.message}`);
            return template;
        }
    }
}

module.exports = {
    PromptTemplateManager
};