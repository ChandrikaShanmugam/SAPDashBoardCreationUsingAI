import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import sapDashboardService from './services/sapDashboardService';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [devMode, setDevMode] = useState(false);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [followUpQuestions] = useState([
    "Show me authorized to sell details",
    "What are the sales exceptions?",
    "Give me plant-wise analysis",
    "Show overview of all data"
  ]);
  const [dataLoading, setDataLoading] = useState(true);

  // Load data summary on component mount
  useEffect(() => {
    const loadDataSummary = async () => {
      try {
        setDataLoading(true);
        await sapDashboardService.loadAllData();
        // const summary = sapDashboardService.getDataSummary();
        // setDataSummary(summary);
      } catch (err) {
        console.error('Error loading data summary:', err);
        setError('Failed to load data files. Please check that the backend is running.');
      } finally {
        setDataLoading(false);
      }
    };
    loadDataSummary();
  }, []);

  const handleSubmit = async () => {
    if (!query.trim()) return;
    if (dataLoading) {
      setError('Data is still loading. Please wait...');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await sapDashboardService.generateDashboard(query.trim(), conversationHistory);

      setDashboardData(response);

      // Update conversation history
      const newHistory = [...conversationHistory, {
        query: query.trim(),
        filters: response.filters,
        timestamp: new Date().toISOString()
      }];
      setConversationHistory(newHistory);

    } catch (err) {
      console.error('Error generating dashboard:', err);
      setError(err.response?.data?.detail || 'Failed to generate dashboard. Make sure the backend is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (exampleQuery) => {
    setQuery(exampleQuery);
  };

  const clearData = () => {
    setDashboardData(null);
    setConversationHistory([]);
    setQuery('');
    setError(null);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🤖 SAP Intelligent Dashboard Generator</h1>
        <p>Ask questions in natural language and get dynamic dashboards!</p>
        <button className="clear-btn" onClick={clearData}>
          🧹 Clear Data
        </button>
      </header>
      <div className="container">
        <Sidebar
          query={query}
          setQuery={setQuery}
          onSubmit={handleSubmit}
          devMode={devMode}
          setDevMode={setDevMode}
          followUpQuestions={followUpQuestions}
          onExampleClick={handleExampleClick}
          dataLoading={dataLoading}
        />
        <Dashboard
          dashboardData={dashboardData}
          loading={loading}
          error={error}
        />
      </div>
    </div>
  );
}

export default App;