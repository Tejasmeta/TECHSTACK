from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import yfinance as yf
import gym
import numpy as np
from stable_baselines3 import DQN
import alpaca_trade_api as tradeapi  
from time import time  
import os  # Import os for environment variables

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

default_stock = "TSLA"

# Alpaca API Keys from Environment Variables
ALPACA_API_KEY = os.getenv("api")
ALPACA_SECRET_KEY = os.getenv("key")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

alpaca = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")

try:
    model = DQN.load("stock_trading_dqn")
except:
    model = None  

last_buy_time = 0  

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/set_stock/<symbol>')
def set_stock(symbol):
    global default_stock
    default_stock = symbol.upper()
    return jsonify({"message": f"Stock symbol updated to {default_stock}"}), 200

@app.route('/stock/<symbol>')
def get_stock(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1mo")

    if hist.empty:
        return jsonify({"error": "Invalid stock symbol or data unavailable"}), 404

    prices = hist['Close'].tolist()
    dates = hist.index.strftime('%Y-%m-%d').tolist()

    return jsonify({'symbol': symbol, 'prices': prices, 'dates': dates, 'last_price': prices[-1]})

@app.route('/trade', methods=['GET', 'POST'])
def trade():
    global last_buy_time  

    stock = yf.Ticker(default_stock)
    hist = stock.history(period="1mo")

    if hist.empty:
        return jsonify({"error": "Stock data unavailable"}), 404

    prices = hist['Close'].tolist()
    if model is None:
        return jsonify({"error": "Model not loaded!"})

    env = StockTradingEnv(prices)
    obs = env.reset()
    action, _ = model.predict(obs)
    decision = "BUY" if action == 0 else "SELL" if action == 2 else "HOLD"

    short_window = 5
    long_window = 20
    short_sma = np.mean(prices[-short_window:]) if len(prices) >= short_window else np.mean(prices)
    long_sma = np.mean(prices[-long_window:]) if len(prices) >= long_window else np.mean(prices)
    sma_decision = "BUY" if short_sma > long_sma else "SELL"

    trade_status = "Waiting for user confirmation"
    
    if request.method == 'POST':
        user_decision = request.json.get('user_decision')
        if user_decision not in ["BUY", "SELL"]:
            return jsonify({"error": "Invalid input, must be BUY or SELL"})

        try:
            last_price = prices[-1]
            limit_price = round(last_price * (1.005 if user_decision == "BUY" else 0.995), 2)

            if user_decision == "SELL":
                elapsed_time = time() - last_buy_time
                if elapsed_time < 30:  
                    return jsonify({"error": f"Cannot sell yet! Please wait {30 - int(elapsed_time)}s"}), 400
            
            if user_decision == "BUY":
                last_buy_time = time()  

            alpaca.submit_order(symbol=default_stock, qty=1, side=user_decision.lower(), type="limit", limit_price=limit_price, time_in_force="gtc")
            trade_status = f"{user_decision} order placed for {default_stock} at ${limit_price:.2f}"
        except Exception as e:
            trade_status = f"Trade execution error: {str(e)}"
    
    return jsonify({
        'symbol': default_stock,
        'decision': decision,
        'sma_decision': sma_decision,
        'last_price': prices[-1],
        'trade_status': trade_status
    })

@app.route('/account')
def account():
    try:
        account = alpaca.get_account()
        return jsonify({
            "equity": account.equity,
            "cash": account.cash,
            "buying_power": account.buying_power,
            "portfolio_value": account.portfolio_value
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

class StockTradingEnv(gym.Env):
    def __init__(self, data):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.balance = 10000  
        self.shares_held = 0
        self.action_space = gym.spaces.Discrete(3)  
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(1,))

    def step(self, action):
        current_price = self.data[self.current_step]
        reward = 0

        if action == 0:  
            self.shares_held += 1
            self.balance -= current_price
        elif action == 2 and self.shares_held > 0:  
            self.balance += current_price
            self.shares_held -= 1
            reward = self.balance

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs = np.array([self.data[self.current_step]])
        return obs, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        return np.array([self.data[self.current_step]])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port)
