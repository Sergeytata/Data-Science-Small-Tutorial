import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
# print(currentdir)
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
# print(parentdir)
sys.path.append(parentdir)


import numpy as np
from simulation.exceptions import ValueLengthIsNot4, InvalidKey
import pandas as pd
# from processing.indicators.ma import ma
from processing.indicators.ema import ema, EMA
from processing.indicators.wma import wma
from typing import List
from simulation.binance_market import Market
from matplotlib import pyplot as plt


class DMI():
    def __init__(self, diPlus: np.ndarray, diMinus: np.ndarray, adx: np.ndarray, candles, lag: int = 37):
        # Indicator.__init__(self)
        self.diPlus = diPlus
        self.diMinus = diMinus
        self.adx = adx
        self.lag = lag
        self.candles = candles # DMI has more candles than diPlus such that we can always reupdate to a new DMI
        self.timestamps = candles.timestamps[self.lag:] # we start from lag to much the number of candles

    def __getitem__(self, key):
        if isinstance(key, int):
            return np.array([self.timestamps[key], self.diPlus[key], self.diMinus[key], self.adx[key]])
        elif isinstance(key, slice):
            return DMI(self.diPlus[key], self.diMinus[key], self.adx[key], self.candles[key])
        else:
            raise InvalidKey
    
    def __setitem__(self, key, new_value: np.ndarray):
        if len(new_value) != 4:
            raise ValueLengthIsNot4
        else:
            self.timestamps[key] = new_value[0]
            self.diPlus[key] = new_value[1]
            self.diMinus[key] = new_value[2]
            self.adx[key] = new_value[3]
        
    def __delitem__(self, key):
        pass

    def __setslice__(self, start, end, new_values):
        pass
    def __delslice__(self, start, endl):
        pass
    def __str__(self):
        d = {
            "diPlus": self.diPlus,
            "diMinus": self.diMinus,
            "adx": self.adx,
            "timestamp": self.timestamps
        }
        return pd.DataFrame(d).to_string()
    
    def __len__(self):
        return len(self.adx)

    def compute_from_candles(self):
        pass
    
    def update(self, new_candle):
        self.candles = self.candles[1:].append(new_candle)
        # print(type(self.candles))
        # print("New Candles len is {}".format(len(self.candles)))
        self.timestamps = np.append(self.timestamps[1::], new_candle.timestamp)
        new_dmi = from_candles(self.candles)
        self.diPlus = new_dmi.diPlus
        self.diMinus = new_dmi.diMinus
        self.adx = new_dmi.adx
        del(new_dmi)
        return np.array([self.timestamps[-1], self.diPlus[-1], self.diPlus[-1], self.adx[-1]])
    
    def plot(self):
        return self.timestamps, np.concatenate(([self.diPlus], [self.diMinus], [self.adx]), axis = 0)




def from_candles(candles, smoothing = 'wma', mode = 'more', return_type = 'pandas', lag = 37) -> DMI:
    
    upMove = candles.max[1:] - candles.max[:-1]
    downMove = candles.min[1:] - candles.min[:-1]  

    dmPlus = (upMove > 0)*(upMove > downMove)*upMove
    dmMinus = (downMove > 0)*(downMove > upMove)*downMove

    max_min = candles.max[1:] - candles.min[1:]
    max_lastprev = np.absolute(candles.max[1:] - candles.close[:-1])
    min_lastprev = np.absolute(candles.min[1:] - candles.close[:-1])

    tr = np.maximum(max_min, max_lastprev, min_lastprev)

    # print(dmPlus)
    # print(dmMinus)
    
    filter_size = (lag - 1 )//3 + 2

    # AT this point we have a lag of 1 data

    if smoothing == 'wma':
        # atr = np.mean(tr)
        # offset = 0
        atr = np.array(wma(tr, filter_size)) # wma lags for filter_size - 2
        offset = len(dmPlus) - len(atr)
        # print(offset)
        diPlus = 100*np.array(wma(dmPlus[offset::]/atr, filter_size)) # lags for filter_size - 2
        diMinus = 100*np.array(wma(dmMinus[offset::]/atr, filter_size))
        dx = abs((diPlus - diMinus)/(diPlus + diMinus))
        adx = 100*np.array(wma(dx, filter_size)) # lags for filter_size - 2
        # 3*filter_size - 6
    
    elif smoothing == 'ema':
        # atr = np.mean(tr)
        # offset = 0
        atr = np.array(ema(tr, filter_size))
        offset = len(dmPlus) - len(atr)
        # print(offset)
        # logging.debug('DMPLUS first term is is {}'.format(dmPlus[0]))
        # logging.debug('DMMINUS first term is is {}'.format(dmMinus[0]))
        diPlus = 100*ema(dmPlus[offset::]/atr, filter_size)
        diMinus = 100*ema(dmMinus[offset::]/atr, filter_size)
        # logging.debug('DIPLUS first term is is {}'.format(diPlus[0]))
        # logging.debug('DIMINUS first term is is {}'.format(diMinus[0]))
        # old_err_state = np.seterr(divide='raise')
        # ignored_states = np.seterr(**old_err_state)
        dx = abs(np.divide((diPlus - diMinus),(diPlus + diMinus), out=np.zeros_like((diPlus - diMinus)), where=(diPlus + diMinus)!=0))
        # dx = np.nan_to_num(dx)
        # logging.info('DX is {}'.format(dx))
        adx = 100*ema(dx[offset::], filter_size)
        # logging.info('ADX is {}'.format(adx))
    
    diMinus = diMinus[(filter_size-2):]
    diPlus = diPlus[(filter_size-2):]
    # print(len(adx))
    # print(len(diMinus))

    # timestamps = candles.timestamps[lag:]
    # timestamps = (timestamps[len(candles) - len(adx):])
    return DMI(diPlus, diMinus, adx, candles, lag)


def from_ndarray(x: np.ndarray) -> DMI:
    return None



def from_dataframe(x: pd.DataFrame, smoothing = 'ema', mode = 'more', return_type = 'pandas', lag = 15) -> DMI:
    def computeTR(candle, prev_candle):
        max_min = float(candle["max"]) - float(candle["min"])
        max_lastprev = abs(float(candle["max"]) - float(prev_candle["close"]))
        min_lastprev = abs(float(candle["min"]) - float(prev_candle["close"]))
        tr_instance = max(max_min, max_lastprev, min_lastprev)
        return tr_instance
        
    def computeDM(candle, prev_candle):
        current_max = candle['max']
        current_min = candle['min']
        prev_max = prev_candle['max']
        prev_min = prev_candle['min']

        upMove = float(current_max) - float(prev_max)
        downMove = float(prev_min) - float(current_min)
        return upMove, downMove

    dmPlus = np.array([])
    dmMinus = np.array([])
    tr = np.array([])

    for i in range(1,len(x)):
        candle = x.iloc[i]
        prev_candle = x.iloc[i-1]
        # print(candle)
        upMove, downMove = computeDM(candle, prev_candle)
        if upMove > downMove and upMove > 0:
            dmPlus = np.append(dmPlus, upMove)
        else:
            dmPlus = np.append(dmPlus, 0)
        if downMove > upMove and downMove > 0:
            dmMinus = np.append(dmMinus, downMove)
        else: 
            dmMinus = np.append(dmMinus, 0)
        tr = np.append(tr, computeTR(candle, prev_candle))
        
    filter_size = lag - 1

    # not working properly
    if smoothing == 'wma':
        # atr = np.mean(tr)
        # offset = 0
        atr = np.array(wma(tr, filter_size))
        offset = len(dmPlus) - len(atr)
        diPlus = 100*np.array(wma(dmPlus[offset::]/atr, filter_size))
        diMinus = 100*np.array(wma(dmMinus[offset::]/atr, filter_size))
        dx = abs((diPlus - diMinus)/(diPlus + diMinus))
        adx = 100*np.array(wma(dx, filter_size))
    
    elif smoothing == 'ema':
        # atr = np.mean(tr)
        # offset = 0
        atr = np.array(ema(tr, filter_size))
        offset = len(dmPlus) - len(atr)
        # print(offset)
        # logging.debug('DMPLUS first term is is {}'.format(dmPlus[0]))
        # logging.debug('DMMINUS first term is is {}'.format(dmMinus[0]))
        diPlus = 100*ema(dmPlus[offset::]/atr, filter_size)
        diMinus = 100*ema(dmMinus[offset::]/atr, filter_size)
        # logging.debug('DIPLUS first term is is {}'.format(diPlus[0]))
        # logging.debug('DIMINUS first term is is {}'.format(diMinus[0]))
        # old_err_state = np.seterr(divide='raise')
        # ignored_states = np.seterr(**old_err_state)
        dx = abs(np.divide((diPlus - diMinus),(diPlus + diMinus), out=np.zeros_like((diPlus - diMinus)), where=(diPlus + diMinus)!=0))
        # dx = np.nan_to_num(dx)
        # logging.info('DX is {}'.format(dx))
        adx = 100*ema(dx[offset::], filter_size)
        # logging.info('ADX is {}'.format(adx))

    # if return_type == 'pandas':
    # timestamps = x['timestamp']
    # timestamps = (timestamps[len(x) - len(adx):])
    #     # print(len(timestamps))
    #     # print(len(adx))
    #     # print(len(diPlus))
    #     # print(len(diMinus))
        

    #     df = pd.DataFrame({'diPlus': diPlus, 'diMinus': diMinus, 'ADX': adx, 'timestamp': timestamps})
    #     return df
    # else:
    
    timestamps = x['timestamp']
    timestamps = (timestamps[len(x) - len(adx):].reset_index(drop=True))
    # print((timestamps).head())
    # print(len(diPlus))
    out = pd.DataFrame({})
    out['diPlus'] = diPlus
    out['diMinus'] = diMinus
    out['adx'] = adx
    out['timestamp'] = timestamps

    return out, lag


# Update Function
def dmi_update_function(candles):
    new_value = dmi(candles, smoothing='ema', return_type = 'pandas')
    return new_value



#####################################################################################
############################        TESTS       #####################################
#####################################################################################
# This test does not work
def test_dmi_from_candles():
    market = Market()
    candles = market.get_n_klines("ETHBTC", '1h', 300)
    # print(candles)
    dmi = from_candles(candles)
    # print(dmi)
    x, y = dmi.plot()
    print(y.shape)
    # print(y[0])

    fig, (ax1) = plt.subplots(1,1)
    ax1.plot(x, y[0] - y[1])
    # ax1.plot(x, y[1])
    ax1.plot(x, y[2])
    ax1.legend(['diPlus', 'diMinus', 'adx'])
    plt.show()



def test_dmi():
    # data = pd.read_csv('data\ETHBTC_4h.csv')
    data = pd.read_csv('data\BTCUSDT-4h-from-2019-10-01-to-2020-07-01.csv')
    # data = data[-300:].reset_index(drop=True)
    # print(data.head())
    dmi_signal, lag = from_dataframe(data)
    # print(dmi_signal.head())
    prices = data['close']
    print(prices.mean())
    prices = (prices - prices.mean())/prices.std()

    # plt.plot(prices)
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.plot(dmi_signal['adx'])
    ax1.plot(dmi_signal['diPlus'])
    ax1.plot(dmi_signal['diMinus'])
    ax1.legend(['adx', 'diPlus', 'diMinus'])
    ax2.plot(prices[1:])
    plt.show()


if __name__ == "__main__":
    # print("DMI FILE!!!")
    # test_dmi_from_candles()
    test_dmi()