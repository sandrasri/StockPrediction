import React, { useState } from 'react';
import axios from 'axios';

const StockForm = ({ setPrediction }) => {
  const [symbol, setSymbol] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      // Correct the URL by passing 'symbol' as a query parameter
      const res = await axios.get(`http://localhost:5001/api/tomorrow?ticker=${symbol}`);
      setPrediction(res.data);
    } catch (err) {
      setError("Failed to fetch prediction. Please try again.");
    }
    setLoading(false);
  };

  return (
    <form onSubmit={handleSubmit} className="mb-6">
      <input
        type="text"
        value={symbol}
        onChange={e => setSymbol(e.target.value)}
        placeholder="Enter Stock Symbol"
        className="p-2 rounded border border-gray-300 mr-2"
      />
      <button
        type="submit"
        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
      >
        {loading ? 'Loading...' : 'Predict'}
      </button>
      {error && <p className="text-red-500 mt-2">{error}</p>}
    </form>
  );
};

export default StockForm;
