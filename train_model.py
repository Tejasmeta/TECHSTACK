import gym
import numpy as np
import yfinance as yf
from stable_baselines3 import DQN
from gym import spaces

# ✅ Custom Stock Trading Environment
class StockTradingEnv(gym.Env):
    def __init__(self, data):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.balance = 10000  # Initial cash
        self.shares_held = 0
        self.action_space = spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1,))

    def step(self, action):
        current_price = self.data[self.current_step]
        reward = 0

        if action == 0:  # Buy
            self.shares_held += 1
            self.balance -= current_price
        elif action == 2 and self.shares_held > 0:  # Sell
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

# ✅ Train the RL Model
def train_model():
    stock_data = yf.download("AAPL", period="1y")["Close"].values
    env = StockTradingEnv(stock_data)

    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000)
    model.save("stock_trading_dqn")
    return model

if __name__ == "__main__":
    train_model()
