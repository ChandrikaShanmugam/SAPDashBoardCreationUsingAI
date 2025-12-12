import React from 'react';
import Chart from './Chart';
import './Dashboard.css';

function Dashboard({ dashboardData, loading, error }) {
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

  const { filters, charts, tables, metrics, data_sample } = dashboardData;

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>🔍 Analysis Results</h1>
        {filters && Object.keys(filters).length > 0 && (
          <div className="filters-summary">
            <h3>Applied Filters:</h3>
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
            <h3>Data Tables</h3>
            <p className="metric-value">{data_sample?.shape ? data_sample.shape[1] : 0}</p>
            <small>columns</small>
          </div>
        </div>
      )}

      {charts && charts.length > 0 && (
        <section className="charts-section">
          <h2>📊 Visualizations</h2>
          <div className="charts-grid">
            {charts.map((chart, index) => (
              <div key={index} className="chart-wrapper">
                <Chart config={chart} data={data_sample?.sample_rows || []} />
              </div>
            ))}
          </div>
        </section>
      )}

      {tables && tables.length > 0 && (
        <section className="tables-section">
          <h2>📋 Data Tables</h2>
          {tables.map((table, index) => (
            <Chart key={index} config={table} data={data_sample?.sample_rows || []} />
          ))}
        </section>
      )}

      {data_sample && (
        <section className="data-sample-section">
          <h2>📄 Data Preview</h2>
          <div className="data-info">
            <p><strong>Dataset Shape:</strong> {data_sample.shape ? `${data_sample.shape[0]} rows × ${data_sample.shape[1]} columns` : 'Unknown'}</p>
            <p><strong>Available Columns:</strong> {data_sample.columns ? data_sample.columns.join(', ') : 'Unknown'}</p>
            <div className="sample-rows">
              <h4>Sample Data:</h4>
              <div className="sample-table">
                <table>
                  <thead>
                    <tr>
                      {data_sample.columns?.slice(0, 5).map(col => (
                        <th key={col}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data_sample.sample_rows?.slice(0, 3).map((row, idx) => (
                      <tr key={idx}>
                        {data_sample.columns?.slice(0, 5).map(col => (
                          <td key={col}>{String(row[col] || '').substring(0, 20)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

export default Dashboard;