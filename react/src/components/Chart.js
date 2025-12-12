import React from 'react';
import Plot from 'react-plotly.js';
import './Chart.css';

function Chart({ config, data }) {
  const renderChart = () => {
    const chartType = config.type;
    const title = config.title || 'Chart';

    switch (chartType) {
      case 'bar':
      case 'grouped_bar':
        return renderBarChart(config, data, title);
      case 'pie':
        return renderPieChart(config, data, title);
      case 'line':
        return renderLineChart(config, data, title);
      case 'table':
        return renderTable(config, data, title);
      default:
        return <div>Unsupported chart type: {chartType}</div>;
    }
  };

  // Streamlit-like Plotly Express logic for bar/grouped bar
  const renderBarChart = (config, data, title) => {
    const xCol = config.x_column;
    const yCol = config.y_column;
    const groupCol = config.group_by || config.color_by;
    const aggFunc = config.agg_function || 'count';
    const limit = config.limit || 10;

    if (!xCol) {
      return <div className="chart-error">No x_column specified for bar chart</div>;
    }

    // Group and aggregate like Plotly Express
    let traces = [];
    if (groupCol) {
      // Grouped bar: group by groupCol, aggregate by xCol
      const grouped = {};
      data.forEach(row => {
        const x = row[xCol] || 'Unknown';
        const group = row[groupCol] || 'Unknown';
        if (!grouped[group]) grouped[group] = {};
        if (aggFunc === 'count' || yCol === 'count') {
          grouped[group][x] = (grouped[group][x] || 0) + 1;
        } else {
          const yVal = parseFloat(row[yCol]) || 0;
          grouped[group][x] = (grouped[group][x] || 0) + yVal;
        }
      });
      // Limit x values by total count across groups
      const xCounts = {};
      Object.values(grouped).forEach(obj => {
        Object.entries(obj).forEach(([x, val]) => {
          xCounts[x] = (xCounts[x] || 0) + val;
        });
      });
      const sortedX = Object.entries(xCounts)
        .sort(([, a], [, b]) => b - a)
        .slice(0, limit)
        .map(([x]) => x);
      traces = Object.keys(grouped).map(group => ({
        x: sortedX,
        y: sortedX.map(x => grouped[group][x] || 0),
        type: 'bar',
        name: group
      }));
    } else {
      // Simple bar: aggregate by xCol
      const agg = {};
      data.forEach(row => {
        const x = row[xCol] || 'Unknown';
        if (aggFunc === 'count' || yCol === 'count') {
          agg[x] = (agg[x] || 0) + 1;
        } else {
          const yVal = parseFloat(row[yCol]) || 0;
          agg[x] = (agg[x] || 0) + yVal;
        }
      });
      const sorted = Object.entries(agg)
        .sort(([, a], [, b]) => b - a)
        .slice(0, limit);
      traces = [{
        x: sorted.map(([x]) => x),
        y: sorted.map(([, y]) => y),
        type: 'bar',
        name: aggFunc === 'count' ? 'Count' : yCol
      }];
    }

    return (
      <Plot
        data={traces}
        layout={{
          title,
          xaxis: {
            title: xCol,
            tickangle: -45,
            automargin: true
          },
          yaxis: { title: aggFunc === 'count' ? 'Count' : yCol },
          barmode: groupCol ? 'group' : 'relative',
          margin: { t: 50, b: 100, l: 50, r: 50 },
          showlegend: !!groupCol
        }}
        style={{ width: '100%', height: '400px' }}
        config={{ responsive: true }}
      />
    );
  };

  const renderPieChart = (config, data, title) => {
    const groupCol = config.x_column || config.group_by;

    if (!groupCol) {
      return <div className="chart-error">No x_column or group_by specified for pie chart</div>;
    }

    const counts = {};
    data.forEach(row => {
      const val = String(row[groupCol] || 'Unknown');
      counts[val] = (counts[val] || 0) + 1;
    });

    // Sort by count and limit
    const sortedEntries = Object.entries(counts)
      .sort(([,a], [,b]) => b - a)
      .slice(0, config.limit || 10);

    const chartData = [{
      labels: sortedEntries.map(([key]) => key),
      values: sortedEntries.map(([, value]) => value),
      type: 'pie',
      name: 'Distribution',
      textinfo: 'label+percent',
      textposition: 'inside'
    }];

    return (
      <Plot
        data={chartData}
        layout={{
          title,
          margin: { t: 50, b: 50, l: 50, r: 50 }
        }}
        style={{ width: '100%', height: '400px' }}
        config={{ responsive: true }}
      />
    );
  };

  const renderLineChart = (config, data, title) => {
    const xCol = config.x_column;
    const yCol = config.y_column;

    if (!xCol || !yCol) {
      return <div className="chart-error">Missing x_column or y_column for line chart</div>;
    }

    const points = {};
    data.forEach(row => {
      const xVal = String(row[xCol] || 'Unknown');
      const yVal = parseFloat(row[yCol]) || 0;
      if (!points[xVal]) points[xVal] = [];
      points[xVal].push(yVal);
    });

    // Sort x values and limit
    const sortedXValues = Object.keys(points).sort();
    const limitedXValues = sortedXValues.slice(0, config.limit || 50);

    // For simplicity, take average if multiple points per x value
    const xValues = limitedXValues;
    const yValues = xValues.map(x => {
      const vals = points[x];
      return vals.reduce((a, b) => a + b, 0) / vals.length;
    });

    const chartData = [{
      x: xValues,
      y: yValues,
      type: 'scatter',
      mode: 'lines+markers',
      name: yCol
    }];

    return (
      <Plot
        data={chartData}
        layout={{
          title,
          xaxis: { title: xCol },
          yaxis: { title: yCol },
          margin: { t: 50, b: 50, l: 50, r: 50 }
        }}
        style={{ width: '100%', height: '400px' }}
        config={{ responsive: true }}
      />
    );
  };

    const renderTable = (config, data, title) => {
    const columns = config.columns || [];
    const limit = config.limit || 50;
    const sortBy = config.sort_by;

    let displayData = [...data];

    // Sort if sort_by is specified
    if (sortBy && displayData.length > 0) {
      displayData.sort((a, b) => {
        const aVal = a[sortBy];
        const bVal = b[sortBy];

        // Handle different data types
        if (typeof aVal === 'number' && typeof bVal === 'number') {
          return aVal - bVal;
        }
        return String(aVal || '').localeCompare(String(bVal || ''));
      });
    }

    displayData = displayData.slice(0, limit);
    const headers = columns.length > 0 ? columns : (displayData.length > 0 ? Object.keys(displayData[0]) : []);

    return (
      <div className="table-container">
        <h3>{title}</h3>
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                {headers.map((header, index) => (
                  <th key={index}>{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {displayData.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {headers.map((header, colIndex) => (
                    <td key={colIndex}>{String(row[header] || '')}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  return (
    <div className="chart-container">
      {renderChart()}
    </div>
  );
}

export default Chart;