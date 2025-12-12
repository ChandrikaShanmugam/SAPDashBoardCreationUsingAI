import React from 'react';
import './Sidebar.css';

function Sidebar({ query, setQuery, onSubmit, devMode, setDevMode, followUpQuestions, onExampleClick, dataLoading, isFirstTime }) {
  return (
    <div className="sidebar">
      <h2>🎯 Ask Your Question</h2>

      <div className="example-questions">
        <h3>💡 {isFirstTime ? 'Example Queries:' : 'Follow Up Questions:'}</h3>
        {followUpQuestions.map((question, index) => (
          <button
            key={index}
            className="example-btn"
            onClick={() => onExampleClick(question)}
          >
            {question}
          </button>
        ))}
      </div>

      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Enter your query here..."
        className="query-input"
      />

      <button className="generate-btn" onClick={onSubmit} disabled={dataLoading}>
        {dataLoading ? 'Loading Data...' : 'Generate Dashboard'}
      </button>
    </div>
  );
}

export default Sidebar;