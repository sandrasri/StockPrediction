import React, { useState } from 'react';
import StockForm from './components/StockForm';
import PredictionDisplay from './components/PredictionDisplay';

function App() {
  const [prediction, setPrediction] = useState(null);

  return (
    <div className="min-h-screen bg-gray-100 p-6 text-center">
      <h1 className="text-3xl font-bold mb-4"> Stock Forecast Dashboard</h1>
      <StockForm setPrediction={setPrediction} />
      {prediction && <PredictionDisplay prediction={prediction} />}
    </div>
  );
}

export default App;
