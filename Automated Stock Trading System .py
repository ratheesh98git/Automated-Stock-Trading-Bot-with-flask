import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def fetch_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def preprocess_data(stock_data):
    stock_data['SMA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA200'] = stock_data['Close'].rolling(window=200).mean()
    stock_data['Return'] = stock_data['Close'].pct_change().shift(-1)
    
    stock_data['Target'] = (stock_data['Return'] > 0).astype(int)
    
    stock_data.dropna(inplace=True)
    
    return stock_data

def train_model(stock_data):
    features = ['SMA50', 'SMA200']
    X = stock_data[features]
    y = stock_data['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f'Training Accuracy: {train_accuracy:.2f}')
    print(f'Testing Accuracy: {test_accuracy:.2f}')
    
    return model

def simulate_trading(model, stock_data):
    stock_data['Predicted Signal'] = model.predict(stock_data[['SMA50', 'SMA200']])
    
    stock_data['Daily Return'] = stock_data['Return'] * stock_data['Predicted Signal'].shift(1)
    
    stock_data['Cumulative Return'] = (stock_data['Daily Return'] + 1).cumprod()
    
    return stock_data

@app.route('/')
def home():
    return render_template('automated_stock_trading.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    
    stock_data = fetch_data(ticker, start_date, end_date)
    
    stock_data = preprocess_data(stock_data)
    
    model = train_model(stock_data)
    
    stock_data = simulate_trading(model, stock_data)
    
    plot_data = stock_data[['Cumulative Return']].reset_index()
    plot_data['Date'] = plot_data['Date'].dt.strftime('%Y-%m-%d')
    plot_data = plot_data.set_index('Date')
    plot_data = plot_data.to_dict()['Cumulative Return']
    
    return render_template('automated_result.html', ticker=ticker, plot_data=plot_data)

if __name__ == '__main__':
    app.run(debug=True)
