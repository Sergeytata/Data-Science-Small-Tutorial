import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
# print(currentdir)
parentdir = os.path.dirname(currentdir)
# parentdir = os.path.dirname(parentdir)
# print(parentdir)
sys.path.append(parentdir)

import pandas as pd
import random
import numpy as np
from scipy.stats import pearsonr, spearmanr

from matplotlib import pyplot as plt
## the backtest takes gives the predictor a new price and the strategy returns 


class Backtest:
    """
    This is a backtest class that is used to compute profit for particular strategy.
    """
    def __init__(self, balance_counter, balance_base = 0, order_volume_prop = 0.05) -> None:
        # self.data  = data
        self.balance_base = balance_base
        self.balance_counter = balance_counter
        self.commision = 0.0025 # 0.25%
        self.prop = order_volume_prop
        self._init_balance = balance_counter
        self._last_ticker = None
        self._tickers = []
        self._prices = []
        self._signals = []
        self._orderbook = []
        self._returns = []
        self._base_converted = 0
    
    def process_orders(self):
        N = len(self._orderbook)

        i = 0
        # print('Order book size: {}'.format(N))
        while i < N:
            # print(i)
            order = self._orderbook[i]
            if order['type'] == 'buy':
                if self._last_ticker['close'] < float(order['price']):
                    base_price = self._last_ticker['close']
                    # counter_volume = self.balance_counter * self.prop
                    counter_volume = order['volume']
                    base_volume = round(counter_volume/base_price,8)
                    self.balance_base += base_volume
                    # self.balance_counter -= counter_volume
                    self._orderbook.pop(i)
                    N -= 1
                elif order['stop_price'] != '':
                    if self._last_ticker['close'] > float(order['stop_price']):
                        base_price = self._last_ticker['close']
                        # counter_volume = self.balance_counter * self.prop
                        counter_volume = order['volume']
                        base_volume = round(counter_volume/base_price,8)
                        self.balance_base += base_volume
                        # self.balance_counter -= counter_volume
                        self._orderbook.pop(i)
                        N -= 1
                else:
                    i += 1
            
            elif order['type'] == 'sell':
                
                if self._last_ticker['close'] > float(order['price']):
                    base_price = self._last_ticker['close']
                    # base_volume = self.balance_base * self.prop
                    base_volume = order['volume']
                    counter_volume = round(base_price*base_volume,8)
                    # self.balance_base -= base_volume
                    self.balance_counter += counter_volume
                    self._orderbook.pop(i)
                    N -= 1
                elif order['stop_price'] != '':
                    if self._last_ticker['close'] < float(order['stop_price']):
                        base_price = self._last_ticker['close']
                        # base_volume = self.balance_base * self.prop
                        base_volume = order['volume']
                        counter_volume = round(base_price*base_volume,8)
                        # self.balance_base -= base_volume
                        self.balance_counter += counter_volume
                        self._orderbook.pop(i)
                        N -= 1
                else:
                    i += 1

        
    
    def add_order(self, order):
        # order = {
        # 'timestamp': timestamp,
        # 'pair': pair,
        # 'type': order_type, # "buy" "sell"
        # 'post_only': None,
        # 'volume': volume, # in counter currency
        # 'price': '',
        # 'stop_price': '',
        # 'stop_direction': '',
        # 'base_account_id': base_acc_id,
        # 'counter_account_id': counter_acc_id,
        # 'wallet': self.wallet
        # }
        if order['type'] == 'buy':
            if order['volume'] == None:
                counter_volume = self.balance_counter * self.prop
                self.balance_counter -= counter_volume
                # save commisioned volume
                order['volume'] = counter_volume*(1-self.commision)
            else:
                self.balance_counter -= order['volume']
                order['volume'] = order['volume']*(1-self.commision)

        elif order['type'] == 'sell':
            if order['volume'] == None:
                base_volume = self.balance_base * self.prop
                self.balance_base -= base_volume
                order['volume'] = base_volume*(1-self.commision)
            else:
                self.balance_base -= order['volume']
                order['volume'] = order['volume']*(1-self.commision)
        self._orderbook.append(order)
    
    def get_order_book(self):
        return self._orderbook

    def get_balance(self, ticker):
        """
        This function returns the balance of the account in the counter currency.
        """
        
        # return self.balance_base*(1-self.commision) * ticker['close'] + self.balance_counter
        return self.balance_base* ticker['close'] + self.balance_counter
        
    def _evaluate_signal(self, ticker, signal):
        """
        Evaluate the given signal
        Type:
        | (Dict, int) -> None
        | (DataFrame, int) -> int
        .
        """
        
        base_price = ticker['close']
        
        self._tickers.append(ticker)
        self._prices.append(base_price)
        self._signals.append(signal)

        if signal == 1:
            # buy
            counter_volume = self.balance_counter * self.prop
            base_volume = round(counter_volume/base_price*(1-self.commision),8)
            self.balance_base += base_volume
            self.balance_counter -= counter_volume

        elif signal == -1:
            base_volume = self.balance_base * self.prop
            counter_volume = round(base_price*base_volume*(1-self.commision),8)
            self.balance_base -= base_volume
            self.balance_counter += counter_volume

        elif signal == 0:
            pass
    
    def run(self, ticker, signal = None, order = None):
        if not self._base_converted:
            self.balance_base = round((self.balance_counter*0.5)/ticker['close'],8)
            self.balance_counter -= self.balance_counter*0.5
            self._base_converted = 1
        self._last_ticker = ticker
        if signal != None:
            self._evaluate_signal(ticker, signal)
        
        if order != None:
            self.add_order(order)
        
        self.process_orders()
        self._returns.append(self.get_return())
    
    def get_return(self):
        """
        This function computes and returns current return based on the initial balance.
        """
        
        curr_balance = self.get_balance(self._last_ticker)
        return (curr_balance - self._init_balance)/self._init_balance

    def get_returns(self):
        return np.array(self._returns)

    def get_tickers(self):
        return pd.DataFrame(self._tickers)
    
    def get_signals(self):
        return np.array(self._signals)

    def visualise(self):
        prices = np.array(self._prices)
        signals = np.array(self._signals)
        N = len(prices)
        buys = signals == 1
        sells = signals == -1
        idles = signals == 0

        hold_returns = (prices - prices[0])/prices[0]

        ns = np.arange(N)
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(prices)
        ax1.scatter(ns[buys],prices[buys],c='b')
        ax1.scatter(ns[sells],prices[sells],c='r')
        # plt.plot(returns)
        ax1.legend(['Price', 'Buys', 'Sells'])
        ax1.set_title('Buys/Sells')
        ax2.plot(self.get_returns())
        ax2.plot(hold_returns)
        ax2.legend(['SS returns', 'Hold returns'])
        ax2.set_title('Returns')
        return fig
    
    def total_trades(self):
        signals = np.array(self._signals)
        trades_N = np.sum(signals != 0)
        return trades_N

    def get_correlation(self):
        prices = np.array(self._prices)
        hold_returns = (prices - prices[0])/prices[0]
        # corr, _ = pearsonr(self.get_returns(), hold_returns)
        corr, _ = spearmanr(self.get_returns(), hold_returns)
        return corr

class Random_Strategy:
    def __init__(self, *args, **kwargs):
        self._last_signal = None
    
    def get_signal(self,ticker):
        return random.randint(-1,1)


class High_Bet_Strategy:
    def __init__(self, *args, **kwargs):
        self._last_signal = None
    
    def get_last_signal(self):
        return self._last_signal[-1]

    def run(self, ticker):
        signal = random.randint(-1,1)
        
        if signal != 0:
            price = ticker['close']
            if signal == 1:
                order_type = 'buy'
                price = price*0.95
            else:
                order_type = 'sell'
                price = price*1.05

            order = {
            'timestamp': None,
            'pair': None,
            'type': order_type, # "buy" "sell"
            'post_only': None,
            'volume': None, # in counter currency
            'price': str(price),
            'stop_price': '',
            'stop_direction': '',
            'base_account_id': None,
            'counter_account_id': None,
            'wallet': None
            }
            return None, order
        else:
            return None, None

def plot_prices_signals(prices, signals):
    N = len(prices)
    buys = signals == 1
    sells = signals == -1
    idles = signals == 0
    

    ns = np.arange(N)
    fig = plt.figure()
    plt.plot(prices)
    plt.scatter(ns[buys],prices[buys],c='b')
    plt.scatter(ns[sells],prices[sells],c='r')
    # plt.plot(returns)
    plt.legend(['Price', 'Buys', 'Sells'])
    return fig


################################################
##############      TESTs           ############
################################################
def test_generated_data():
    N = 100
    prices = np.random.randn(N,1)*0.5 + 5

    
    # print(np.linspace(0,1,N).shape)
    print(prices.shape)
    data = pd.DataFrame(data=prices, columns=['close'])
    # print(data.loc[0])
    bt = Backtest(100,0.1)

    signals = []
    returns = []
    for i in range(len(data)):
        ticker = data.loc[i]
        # print(ticker['close'])    
        signal = random_strategy(ticker)
        signals.append(signal)
        bt._evaluate_signal(ticker, signal)
        returns.append(bt.get_return())
    
    signals = np.array(signals)
    buys = signals == 1
    sells = signals == -1
    idles = signals == 0
    # buys_x = []
    # buys_y = []
    # sells_x = []
    # sells_y = []
    # idles_x = []
    # idles_y = []

    # for i in range(N):
    #     signal = signals[i]
    #     if signal == 1:
    #         buys_x.append(i)
    #         buys_y = prices[i]
    #     elif signal == -1:


    print('Profit: {}'.format(bt.get_return()))
    ns = np.arange(N)
    plt.plot(prices)
    plt.scatter(ns[buys],prices[buys],c='b')
    plt.scatter(ns[sells],prices[sells],c='r')
    # plt.plot(returns)
    plt.legend(['Price', 'Buys', 'Sells'])
    plt.show()
    # print(bt.get_balance(ticker))

def test_with_year_prices():
    data = pd.read_csv('data\ETHBTC_1d.csv')
    print(data.head())
    N = len(data)
    initial_balance = 100
    bt = Backtest(initial_balance)
    random_strategy = Random_Strategy()

    for i in range(200,N):
        ticker = data.loc[i]
        signal = random_strategy.get_signal(ticker)
        bt.run(ticker,signal)

    signals = bt.get_signals()
    print(bt.get_return())
    fig = plot_prices_signals(bt.get_tickers()['close'].to_numpy(),signals)
    plt.show()


def test_order_book():
    data = pd.read_csv('data\ETHBTC_1d.csv')
    print(data.head())
    N = len(data)
    initial_balance = 100
    bt = Backtest(initial_balance)
    strategy = High_Bet_Strategy()
    for i in range(200,N):
        ticker = data.loc[i]
        _, order = strategy.run(ticker)
        bt.run(ticker, None, order)

    print(bt.get_return())
    print(len(bt.get_order_book()))

if __name__ == '__main__':
    test_with_year_prices()
    # test_order_book()