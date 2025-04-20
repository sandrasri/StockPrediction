import React from 'react';

const PredictionDisplay = ({ prediction }) => {
  // Handle the prediction directly
  const tomorrowPrediction = prediction ? prediction.prediction : null;

  return (
    <div className="mt-6">
      {/* Display Tomorrow's Prediction */}
      {tomorrowPrediction !== null ? (
        <h2 className="text-xl font-semibold">
          Tomorrow's Prediction for {prediction.ticker}: ${tomorrowPrediction.toFixed(2)}
        </h2>
      ) : (
        <h2 className="text-xl font-semibold">No Prediction Available</h2>
      )}
    </div>
  );
};

export default PredictionDisplay;
