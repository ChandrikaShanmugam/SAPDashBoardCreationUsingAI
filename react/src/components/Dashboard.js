import React from 'react';
import Chart from './Chart';
import './Dashboard.css';

function Dashboard({ dashboardData, loading, error, onFollowUpClick }) {
  if (loading) {
    return (
      <div className="dashboard loading">
        <div className="loading-container">
          <div className="loading-spinner">
            <div className="spinner"></div>
            <p>🤖 Processing your query with AI...</p>
            <p className="loading-subtitle">This may take a few moments</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard error">
        <div className="error-container">
          <div className="error-message">
            <h2>❌ Error</h2>
            <p>{error}</p>
            <div className="error-help">
              <p><strong>Troubleshooting:</strong></p>
              <ul>
                <li>Make sure the backend server is running on port 8000</li>
                <li>Check that all data files are present</li>
                <li>Verify your query is clear and specific</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!dashboardData) {
    return (
      <div className="dashboard empty">
        <div className="empty-container">
          <div className="empty-state">
            <h1>🤖 SAP Intelligent Dashboard Generator</h1>
            <p>Ask questions in natural language and get dynamic dashboards!</p>
            <div className="features">
              <h3>✨ Features:</h3>
              <ul>
                <li>🔍 Natural language query processing</li>
                <li>📊 Automatic chart generation</li>
                <li>🔗 Cross-table data analysis</li>
                <li>📈 Interactive visualizations</li>
                <li>💾 Data export capabilities</li>
              </ul>
            </div>
            <p className="instruction">Enter a query in the sidebar to get started.</p>
          </div>
        </div>
      </div>
    );
  }

  const { filters, charts, tables, metrics, data_sample, filtered_data, title } = dashboardData;

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>{title || '🔍 Analysis Results'}</h1>
        {filters && Object.keys(filters).length > 0 && (
          <div className="filters-summary">
            <h3>Applied Filters :</h3>
            <div className="filter-tags">
              {Object.entries(filters).map(([key, value]) => (
                <span key={key} className="filter-tag">
                  <strong>{key}:</strong> {Array.isArray(value) ? value.join(', ') : value}
                </span>
              ))}
            </div>
          </div>
        )}
      </header>

      {metrics && (
        <div className="metrics-grid">
          <div className="metric-card">
            <h3>Total Records</h3>
            <p className="metric-value">{metrics.total_records?.toLocaleString() || 0}</p>
          </div>
          <div className="metric-card">
            <h3>Unique Materials</h3>
            <p className="metric-value">{metrics.unique_materials?.toLocaleString() || 0}</p>
          </div>
          <div className="metric-card">
            <h3>Unique Plants</h3>
            <p className="metric-value">{metrics.unique_plants?.toLocaleString() || 0}</p>
          </div>
          <div className="metric-card">
            <h3>Unique UPCs</h3>
            <p className="metric-value">{metrics.unique_upcs?.toLocaleString() || 0}</p>
          </div>
          <div className="metric-card">
            <h3>Total Quantity</h3>
            <p className="metric-value">{metrics.total_quantity?.toLocaleString() || 0}</p>
          </div>
        </div>
      )}

      {charts && charts.length > 0 && (
        <section className="charts-section">
          <hr />
          <h2>📊 Visualizations</h2>
          <div className="charts-grid">
            {charts.map((chart, index) => (
              <div key={index} className="chart-wrapper">
                <Chart config={chart} data={filtered_data || []} />
              </div>
            ))}
          </div>
        </section>
      )}

      {tables && tables.length > 0 && (
        <section className="tables-section">
          <hr />
          <h2>📋 Data Tables</h2>
          {tables.map((table, index) => (
            <Chart key={index} config={table} data={filtered_data || []} />
          ))}
        </section>
      )}

      {/* Add empty space between sections */}
      <div style={{ marginBottom: '40px' }}></div>

      {/* Download Filtered Data Section */}
      {filtered_data && filtered_data.length > 0 && (
        <section className="download-section">
          <hr />
          <div className="download-container">
            <button 
              className="download-btn"
              onClick={() => downloadFilteredData(filtered_data)}
            >
              📥 Download Filtered Data (CSV)
            </button>
            <p className="download-info">
              Download {filtered_data.length.toLocaleString()} filtered records as CSV file
            </p>
          </div>
        </section>
      )}

    </div>
  );
}

// Helper function to convert data to CSV and download
function downloadFilteredData(data) {
  if (!data || data.length === 0) return;

  // Get all unique keys from the data
  const headers = Array.from(new Set(data.flatMap(row => Object.keys(row))));
  
  // Create CSV content
  const csvContent = [
    headers.join(','), // Header row
    ...data.map(row => 
      headers.map(header => {
        const value = row[header] || '';
        // Escape quotes and wrap in quotes if contains comma, quote, or newline
        if (typeof value === 'string' && (value.includes(',') || value.includes('"') || value.includes('\n'))) {
          return `"${value.replace(/"/g, '""')}"`;
        }
        return value;
      }).join(',')
    )
  ].join('\n');

  // Create and trigger download
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  link.setAttribute('href', url);
  link.setAttribute('download', 'filtered_data.csv');
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

export default Dashboard;