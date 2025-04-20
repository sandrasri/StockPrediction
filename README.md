This project is a personal project that is still a work in progress. It aims to predict stock prices using machine learning models and provides an API to retrieve predictions for specific stock symbols. The app consists of a Flask backend that handles predictions and a React frontend for users to input a stock symbol and view the predicted price for tomorrow.

****Project Features****
**Backend:** A flask API that fetches stock data, performs predictions using machine learning models, and returns the predicted stock prices.
**Frontend:** A React-based user interface where users can input a stock symbol and receive the predicted price for the next day.
**Prediction** The backend utilizes various models (e.g. TensorFlow, XGBoost) to predict stock prices.

****Installation****
docker-compose up --build

**To run backend:**
cd backend
pip install -r requirements.txt
python app.py

**To run frontend:**
cd stock-prediction-frontend
npm install
npm start

****How to Use****
1. Enter a stock symbol (e.g., AAPL for Apple)
2. Click on the "Predict" button to get the predicted price for tomorrow (will take 1-2 minutes to load, due to tensorflow)

**Work In Progress**
This project is still being developed. There are plans to add additional features, such as short-term/long-term predictions, graphs, and more machine learning models to get more accurate forecasting. 
