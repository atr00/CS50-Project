import pandas as pd
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
import seaborn as sns
from helper import *
from deribit import client


class Option:

    def __init__(self, underlying, opt_type, strike, expiry, size = 1.0, side = 'buy'):
        '''
        Represents an option. Pricing and greeks using Black-76 and Garman & Kohlhagen models.
        Pairs are viewed as foreign/domestic (cur1/cur2).
        F = S * exp((r_dom-r_fgn)*T)

        Parameters:
          - underlying  : string
          - opt_type    : string equal to "call" or "put"
          - strike      : float or int
          - expiry      : can be a tenor of type string (e.g. 1d, 1w, 1m, 1y, 1m1y), a datetime.date or datetime.datetime
          - size        : float or int (float, default = 1.0)
          - side        : string equal to "buy" or "sell" (default = "buy")
        '''
        assert isinstance(underlying, str), 'underlying should be a string.'
        assert opt_type == 'call' or opt_type == 'put', 'opt_type should be "call" or "put".'
        assert isinstance(strike, int) or isinstance(strike, float), 'strike should be a int or a float.'
        assert is_date_or_tenor(expiry), 'expiry should be a dt.date, dt.datetime or a string representing a tenor (e.g. "1Y").'
        assert isinstance(size, int) or isinstance(size, float), 'size should be a int or a float.'
        assert size >= 0.0, 'size must be positive.'
        assert isinstance(side, str), 'side should be a string equal to "buy" or "sell".'
        assert side == 'buy' or side == 'sell', 'side must be "sell" or "buy".'

        self.opt_type = opt_type
        self.K = float(strike)
        self.underlying = underlying
        self.size = float(size)
        self.side = side

        if isinstance(expiry, str):
            self.expiry_date = tenor2date(expiry)
        elif isinstance(expiry, dt.date) or isinstance(expiry, dt.datetime):        
            self.expiry_date = pd.to_datetime(expiry)
        else:
            raise ValueError('Wrong date format provided.')


    def __repr__(self):
        str_repr = 'Option({side},{undl},{size},{opt_type},{strike},{expi})'
        str_expiry = self.expiry_date.strftime('%Y-%m-%d')
        return str_repr.format(side     = self.side,
                               undl     = self.underlying,
                               size     = self.size,
                               opt_type = self.opt_type,
                               strike   = self.K,
                               expi     = str_expiry)

    def __str__(self):
        str_repr = '{side:<9s} {undl:<9s} {size:<9.2f} {opt_type:<9s} {strike:<9.2f} {expi}'
        str_expiry = self.expiry_date.strftime('%Y-%m-%d')
        return str_repr.format(side     = self.side,
                               undl     = self.underlying,
                               size     = self.size,
                               opt_type = self.opt_type,
                               strike   = self.K,
                               expi     = str_expiry)

    @property
    def deribit_undl(self):
        return self.underlying[:3].upper()

    @property
    def deribit_exp(self):
        return dt.datetime.strftime(pd.to_datetime(self.expiry_date), "%d%b%y").upper()
    
    @property
    def T(self):
        delta = self.expiry_date - dt.datetime.combine(dt.date.today(), dt.datetime.min.time())
        return delta.days / 365.25

    @property
    def expiry(self):
        return self.expiry_date
        
    @property
    def cur1(self):
        return self.underlying[:3]

    @property
    def cur2(self):
        return self.underlying[-3:]    

    @staticmethod
    def N(x):
        return norm.cdf(x)

    @staticmethod
    def N_der(x):
        return np.exp(-x**2/2) / np.sqrt(2 * np.pi)

    def _d1(self, F, σ, r_dom, r_fgn):
        return (np.log(F/self.K) + σ**2/2*self.T) / \
           (σ*np.sqrt(self.T))

    def _d2(self, F, σ, r_dom, r_fgn):
        return self._d1(F, σ, r_dom, r_fgn) - σ * np.sqrt(self.T)

    def _call_value_cur2(self, F, σ, r_dom, r_fgn):
        '''
        returns the value of a call. Value in currency 2 and notional in currency 1.
        '''    
        return (F * self.N(self._d1(F, σ, r_dom, r_fgn)) - self.K * self.N(self._d2(F, σ, r_dom, r_fgn))) *\
                np.exp(-r_dom*self.T) * self.size


    def _call_value_cur1(self, F, σ, r_dom, r_fgn):
        '''
        returns the value of a call. Value in currency 1 and notional in currency 1.
        '''    
        return self._call_value_cur2(F, σ, r_dom, r_fgn)  / F * np.exp((r_dom-r_fgn)*self.T)
                  
    def _put_value_cur2(self, F, σ, r_dom, r_fgn):
        '''
        returns the value of a put. Value in currency 2 and notional in currency 1.
        '''        
        return (self.K * self.N(-self._d2(F, σ, r_dom, r_fgn)) -  F * self.N(-self._d1(F, σ, r_dom, r_fgn)) ) *\
                np.exp(-r_dom*self.T) * self.size

                  
    def _put_value_cur1(self, F, σ, r_dom, r_fgn):
        '''
        returns the value of a put. Value in currency 1 and notional in currency 1.
        '''        
        return self._put_value_cur2(F, σ, r_dom, r_fgn)  / F * np.exp((r_dom-r_fgn)*self.T)

    def _call_delta(self, F, σ, r_dom, r_fgn):
        '''
        returns the delta of a call. Value in currency 1.
        '''        
        return np.exp(-r_fgn*self.T) * self.N(self._d1(F, σ, r_dom, r_fgn)) * self.size

    def _put_delta(self, F, σ, r_dom, r_fgn):
        '''
        returns the delta of a put. Value in currency 1.
        '''      
        return (self.N(self._d1(F, σ, r_dom, r_fgn)) - 1) * self.size

    def _vega(self, F, σ, r_dom, r_fgn):
        '''
        returns the vega for 1% of move in implied volatility. Value in currency 2.
        '''
        return 0.01 * F * self.N_der(self._d1(F, σ, r_dom, r_fgn)) * np.sqrt(self.T) * np.exp(-r_dom*self.T) * self.size

    def _gamma(self, F, σ, r_dom, r_fgn):
        '''
        returns the change of delta for 1% move of the spot. Value in currency 1.
        '''
        return 0.01 * self.N_der(self._d1(F, σ, r_dom, r_fgn)) * np.exp(-r_fgn*self.T) / (σ * np.sqrt(self.T)) * self.size

    def _theta(self, F, σ, r_dom, r_fgn):
        '''
        returns the decay per day. Value in currency 2.
        '''
        return  -((-F * self.N_der(self._d1(F, σ, r_dom, r_fgn)) * np.exp(-r_dom * self.T) * σ) / (2 * np.sqrt(self.T)) +\
                r_fgn * F * self.N(self._d1(F, σ, r_dom, r_fgn)) * np.exp(-r_dom * self.T) -\
                r_dom * self.K * self.N(self._d2(F, σ, r_dom, r_fgn)) * np.exp(-r_dom * self.T)) * self.size / (self.T * 365.25)

    def price(self, F, σ, r_dom = 0.0, r_fgn = 0.0):
        res = {}
        if self.opt_type.upper() == 'CALL':
            res['price1'] = {
              'value': self._call_value_cur1(F, σ, r_dom, r_fgn),
              'quote': self.cur1
            }
            res['price2'] = {
              'value': self._call_value_cur2(F, σ, r_dom, r_fgn),
              'quote': self.cur2
            }      
            res['delta'] = {
              'value': self._call_delta(F, σ, r_dom, r_fgn),
              'quote': self.cur1
            }
        elif self.opt_type.upper() == 'PUT':
            res['price1'] = {
              'value': self._put_value_cur1(F, σ, r_dom, r_fgn),
              'quote': self.cur1
            }
            res['price2'] = {
              'value': self._put_value_cur2(F, σ, r_dom, r_fgn),
              'quote': self.cur2
            }      
            res['delta'] = {
              'value': self._put_delta(F, σ, r_dom, r_fgn),
              'quote': self.cur1
            }      

        res['vega'] = {
            'value': self._vega(F, σ, r_dom, r_fgn),
            'quote': self.cur2
        }
        res['gamma'] = {
            'value': self._gamma(F, σ, r_dom, r_fgn),
            'quote': self.cur1
        }
        res['theta'] = {
            'value': self._gamma(F, σ, r_dom, r_fgn),
            'quote': self.cur2
        } 

        return pd.DataFrame(res)


class OptionPortfolio:

    def __init__(self):
        self.strategy = []
        
    def __repr__(self):
        obj = 'object' if len(self) < 2 else 'objects'
        str_repr = 'OptionPortfolio({} {})'
        return str_repr.format(len(self), obj)

    def __str__(self):
        return self.to_df.to_string()
    
    def __len__(self):
        return len(self.strategy)

    @property
    def unique_undls(self):
        df = self.to_df
        return list(df.underlying.unique())

    @property
    def unique_expiries(self):
        res = {}
        for undl in self.unique_undls:
            undl_df = self.to_df[self.to_df['underlying'] == undl]
            res[undl] = list(undl_df.expiry.unique())
        return res

    @property
    def to_df(self):
        sides     = []
        undls     = []
        sizes     = []
        types     = []
        strikes   = []
        expiries  = []
        for opt in self.strategy:
            sides.append(opt.side)
            undls.append(opt.underlying)
            sizes.append(opt.size)
            types.append(opt.opt_type)
            strikes.append(opt.K)
            expiries.append(opt.expiry_date)

        opt_dict = {
            'side': sides , 
            'underlying': undls,
            'size': sizes,
            'option_type': types,
            'strike': strikes,
            'expiry': expiries
        }

        return pd.DataFrame.from_dict(opt_dict)


    def deribit_price(self):
        """
        Aggregated values by expiries
        """
        expiries = self.unique_expiries
        undls    = self.unique_undls
        price1   = []
        price2   = []
        delta    = []
        vega     = []
        gamma    = []
        theta    = []

        res = {}
        for undl in undls: 
            exp_dict = {}
            for expiry in expiries[undl]:
                tmp_dict = {}
                _price1  = 0
                _price2  = 0
                _delta   = 0
                _vega    = 0
                _gamma   = 0
                _theta   = 0
                
                options = [opt for opt in self.strategy if opt.expiry == expiry and opt.underlying == undl]
                for opt in options:
                    deribit_inst = client.get_opt_instrument(opt.deribit_undl, opt.deribit_exp, int(opt.K), opt.opt_type)
                    opt_info = client.get_option_info(deribit_inst)
                    iv = opt_info['mark_iv'] / 100
                    f = opt_info['underlying_price']
                    price = opt.price(f, iv)
                    _price1 += price['price1'].value
                    _price2 += price['price2'].value
                    _delta  += price['delta'].value
                    _vega   += price['vega'].value
                    _gamma  += price['gamma'].value
                    _theta  += price['theta'].value
                    
                tmp_dict = {'price1': _price1,
                            'price2': _price2,
                            'delta' : _delta,
                            'vega'  : _vega,
                            'gamma' : _gamma,
                            'theta' : _theta}

                exp_dict[expiry] = tmp_dict
                
            exp_dict['quote'] = {'price1': price['price1'].quote,
                                 'price2': price['price2'].quote,
                                 'delta' : price['delta'].quote,
                                 'vega'  : price['vega'].quote,
                                 'gamma' : price['gamma'].quote,
                                 'theta' : price['theta'].quote}

            
            res[undl] = exp_dict

        return res


    def add(self, opt):
        if type(opt) is not list:
            self.strategy.append(opt)
        else:
            for opt_tmp in opt:
                self.strategy.append(opt_tmp)

    def remove(self, idx):
        del self.strategy[idx]

    def get_payoff(self, x_axis, side, size, opt_type, strike):
        if x_axis is None:
            return []
        else:
            mult = 1.0 if side == 'buy' else -1.0
        if opt_type == 'call':
            y_axis = np.clip(x_axis - strike, a_min=0, a_max=None) * mult * size
        elif  opt_type == 'put':
            y_axis = np.clip(strike - x_axis, a_min=0, a_max=None) * mult * size
        else:
            raise ValueError('Wrong options details provided.')

        return y_axis

    def get_payoffs_sum(self, x_axis, undl, expiry):
        mask = (self.to_df['underlying'] == undl) & (self.to_df['expiry'] == expiry)
        filtered_df = self.to_df[mask]
        filtered_df = filtered_df.reset_index()
        y_axis = np.zeros(len(x_axis))

        for idx, opt in filtered_df.iterrows():
            side      = opt['side']
            size      = opt['size']
            opt_type  = opt['option_type']
            strike    = opt['strike']
            y_axis    = y_axis + self.get_payoff(x_axis, side, size, opt_type, strike)

        return y_axis

    def plot(self, undl):
        unique_exp = self.unique_expiries[undl]
        subplots_nb = len(unique_exp)
        df = self.to_df[self.to_df['underlying'] == undl]
        a_min = max(df['strike'].min() * 0.85, 0)
        a_max = df['strike'].max() * 1.15

        if subplots_nb == 1:
            x_axis = np.linspace(a_min, a_max, 1000)
            y_axis = self.get_payoffs_sum(x_axis, undl, unique_exp[0])
            sns.lineplot(x = x_axis, y = y_axis)
        else:
            i = 0
            y_axes = []
            fig, axes = plt.subplots(subplots_nb, 1, sharey='col', figsize=(10, 8))
            #fig.suptitle('Payoff for each Expiry')
            plt.tight_layout()

            for exp in unique_exp:
                df_temp = df[df['expiry'] == exp]
                x_axis = np.linspace(a_min, a_max, 1000)

                y_axis = self.get_payoffs_sum(x_axis, undl, exp)
                y_axes.append(y_axis)
                sns.lineplot(ax = axes[i], x = x_axis, y = y_axes[i])
                str_title = pd.to_datetime(exp).strftime('%d %b %Y')
                axes[i].set_title(str_title)
                i += 1


class BullishStrategy(OptionPortfolio):
    def __init__(self, undl, schedule, call_K, put_at_maturity = False, put_K = None, size = 1.0, leverage = 1.0):
        assert isinstance(call_K, int) or isinstance(call_K, float) or isinstance(call_K, list), 'call_K must be an intance of int, float or a list with 2 elements.'
        assert isinstance(undl, str), 'undl must be an instance of str.'
        if isinstance(call_K, list):
            assert len(call_K) == 2, 'call_K must have 2 elements only.'
        assert isinstance(call_K[0], int) or isinstance(call_K[0], float), 'call_K elments must be instances of int or float.'
        assert isinstance(call_K[1], int) or isinstance(call_K[1], float), 'call_K elments must be instances of int or float.'
        assert isinstance(put_at_maturity, bool), 'put_at_maturity should be a boolean.'
        if put_at_maturity:
            assert isinstance(put_K, int) or isinstance(put_K, float) or isinstance(put_K, list), 'put_K must be an intance of int, float or a list with 2 elements.'
            if isinstance(put_K, list):
                assert len(put_K) == 2, 'call_K must have 2 elements only.'
                assert isinstance(put_K[0], int) or isinstance(put_K[0], float), 'put_K elments must be instances of int or float.'
                assert isinstance(put_K[1], int) or isinstance(put_K[1], float), 'put_K elments must be instances of int or float.'
            assert isinstance(leverage, int) or leverage(call_K, float), 'leverage must be an int or a float.'
        super().__init__()

        self.undl = undl
        self.schedule = schedule
        self.call_K = call_K
        self.put_at_maturity = put_at_maturity
        self.put_K = put_K
        self.size = size
        self.leverage = leverage

        for tenor_date in schedule:
            if isinstance(call_K, list):
                self.add(Option(self.undl, 'call', call_K[0], tenor_date, size, 'buy'))
                self.add(Option(self.undl, 'call', call_K[1], tenor_date, size, 'sell'))        
            else:
                self.add(Option(self.undl, 'call', call_K, tenor_date, size, 'buy'))

            if put_at_maturity and tenor_date == schedule[-1]:
                if isinstance(put_K, list):
                    self.add(Option(self.undl, 'put', put_K[0], tenor_date, size * leverage, 'sell'))
                    self.add(Option(self.undl, 'put', put_K[1], tenor_date, size * leverage, 'buy'))             
                else:
                    self.add(Option(self.undl, 'put', put_K, tenor_date, size * leverage, 'sell'))             