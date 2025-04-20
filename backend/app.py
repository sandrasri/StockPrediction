from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from pytickersymbols import PyTickerSymbols
#from stockprediction import downloadTicker, predictTomorrow, predictShortTerm  # Modularize your logic
from test import downloadTicker, predictTomorrow

app = Flask(__name__)
CORS(app)

@app.route("/api/data", methods=["GET"])
def get_ticker_data():
    ticker = request.args.get("ticker")

    if not ticker:
        return jsonify({"error": "Missing ticker symbol."}), 400

    try:
        df = downloadTicker(ticker.upper())
        json_data = df.to_dict(orient="records")
        return jsonify({"ticker": ticker.upper(), "data": json_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/tomorrow", methods=["GET"])
def get_tomorrow_data():
    ticker = request.args.get("ticker")

    if not ticker:
        return jsonify({"error": "Missing ticker symbol."}), 400

    try:
        tomorrow_pricetf = predictTomorrow(ticker.upper())  # Scalar value
        
        # Return the scalar prediction directly
        return jsonify({"ticker": ticker.upper(), "prediction": np.float64(tomorrow_pricetf)})

    except Exception as e:
        # Instead of returning the exception object itself, return the error message as a string
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)


