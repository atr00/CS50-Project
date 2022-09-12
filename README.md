# FX OPTIONS PRICER
### Video Demo:  <URL HERE>
## Description
This is a Jupyter notebook based options pricer applied to FX and more precisely to crypto options using Deribit. This tool will allow users to build strategy, price them and aggregate the risks to have a global view of the portfolio. This tool leverages on IPython widgets which makes notebooks interactive while adding widgets like buttons, droplists or observers.

## Project files
1. deribit.py
2. options.py
3. helper.py
4. OptionsView.ipynb

## deribit.py
This file consists of an API to access Deribit's market data. The connection to the server is done via WebSocket. Users will need to enter their private keys in the file credentials.py if they want to use private functions. On this version, all the data needed to run the project are using public functions.
I used the following project which I modified to add more functions: https://github.com/Jimmy-sha256/deribit_websocket_v2.
The main use cases of this file is to connect to Deribit and download data on futures and options.

## options.py
This file is where classes are defined. The main 2 classes are:
- Option
- OptionPortfolio

#### Option
Represents an option. Pricing and greeks using Black-76 and Garman & Kohlhagen models.
Pairs are viewed as foreign/domestic (cur1/cur2).
F = S * exp((r_dom-r_fgn)*T)

Parameters:
    - underlying  : string
    - opt_type    : string equal to "call" or "put"
    - strike      : float or int
    - expiry      : can be a tenor of type string (e.g. 1d, 1w, 1m, 1y, 1m1y), a datetime.datetime
    - size        : float or int (float, default = 1.0)
    - side

The Option class creates an option object defined by it underlying, type, strike, expiry, size, side.
The underlying can be anything but in order to apply to Deribit, it has to be a string as follows: 'eth-usd' or 'btc-usd'. However this is a generic class and any option can be created.
The class itself has methods that will price the option and give its principal risks: delta, vega, gamma and theta. The inputs of the price methods are the current underlying (forward) price and the current implied volatility. The return object is a pandas DataFrame.
A few other methods are a bridge to Deribit and give Deribit-like instruments names or expiry date format.

#### OptionPortfolio
The OptionPortfolio class creates a portfolio object which is an aggregation of option objects. It adds the computation of the portfolio's risks and will aggregator the results by underlying and by expiry dates. Therefore it gives the overview of the risks of the portfolio.
Users can add, remove options from the portfolio. Users can also plot the payoff (one chart per expiry date).
It also allows inheritance and users will be able to create their own fixed strategies. I built a BullishStragery that inherits from OptionPortfolio as an example. It takes a schedule as an input, strikes and if we want the stragegy to be risky (i.e. selling options to add more leverage).

## helper.py
Several helper functions are in this file:
- deribitExp2Dt: convert deribit expiry format (DDMMMYY) to a datetime.datetime object.
- dtExp2Deribit: the reverse function, converts a datetime.time object to a Deribit date string.
- tenor2date: converts a string tenor (e.g. 1M, 1Y) to a datetime.datetime object
- is_single_tenor: erifies that the period provided in the form of a string conforms to the rules.
- is_valid_tenor(tenor): verifies that the expiry provided in the form of a tenor conforms to the rules.
- is_date(date): verifies that the date provided is an instance of dt.date or dt.datetime or a valid tenor.
- generate_schedule: Return an array of pd.Timesteamp. e.g. sched = generate_schedule('0d', '3M', '1M')
- is_notebook: returns True if the file is run from a notebook, False otherwise.