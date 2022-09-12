import asyncio
import websockets
import json
import pandas as pd
import re

from credentials import client_id, client_secret, client_url

# create a websocket object
class WS_Client(object):
    def __init__(self, client_id=None, client_secret=None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.client_url = 'wss://test.deribit.com/ws/api/v2'
        self.json = {
            'jsonrpc' : '2.0',
            'id' : 1,
            'method' : None,
        }

    # send an authentication request
    async def private_api(self, request):
        options = {
            'grant_type' : 'client_credentials',
            'client_id' : self.client_id,
            'client_secret' : self.client_secret
        }

        self.json['method'] = 'public/auth'
        self.json['params'] = options

        async with websockets.connect(self.client_url) as websocket:
            await websocket.send(json.dumps(self.json))
            while websocket.open:
                response = await websocket.recv()

                # send a private subscription request
                if 'private/subscribe' in request:
                    await websocket.send(request)
                    while websocket.open:
                        response = await websocket.recv()
                        response = json.loads(response)
                        print(response)

                # send a private method request
                else:
                    await websocket.send(request)
                    response = await websocket.recv()
                    response = json.loads(response)
                    break
            return response

    # send a public method request
    async def public_api(self, request):
        async with websockets.connect(self.client_url) as websocket:
            await websocket.send(request)
            response = await websocket.recv()
            response = json.loads(response)
            return response

    # send a public subscription request
    async def public_sub(self, request):
        async with websockets.connect(self.client_url) as websocket:
            await websocket.send(request)
            while websocket.open:
                response = await websocket.recv()
                response = json.loads(response)
                print(response)

    # create an asyncio event loop
    def loop(self, api, request):
        response = asyncio.get_event_loop().run_until_complete(
            api(json.dumps(request)))
        return response

    def index(self, currency):
        options = {'currency': currency}
        self.json['method'] = 'public/get_index'
        self.json['params'] = options
        return self.loop(self.public_api, self.json)

    def ticker(self, instrument_name):
        options = {'instrument_name': instrument_name}
        self.json['method'] = 'public/ticker'
        self.json['params'] = options
        return self.loop(self.public_api, self.json)

    def get_index_price(self, index_name):
        options = {'index_name': index_name}
        self.json['method'] = 'public/get_index_price'
        self.json['params'] = options
        return self.loop(self.public_api, self.json)

    def get_instruments(self, currency, inst_type):
        options = {'currency': currency, 'kind': inst_type, 'expired': False}
        self.json['method'] = 'public/get_instruments'
        self.json['params'] = options
        return self.loop(self.public_api, self.json)

    def get_order_book(self, instrument_name):
        options = {'instrument_name': instrument_name}
        self.json['method'] = 'public/get_order_book'
        self.json['params'] = options
        return self.loop(self.public_api, self.json)

    def mark_price(self, instrument_name):
        return self.get_order_book(instrument_name)['result']['mark_price']

    def index_price(self, instrument_name):
        return self.get_order_book(instrument_name)['result']['index_price']

    def get_option_info(self, instrument_name):
        info = {}
        order_book = self.get_order_book(instrument_name)['result']
        info['index_price'] = order_book['index_price']
        info['instrument_name'] = order_book['instrument_name']
        info['mark_iv'] = order_book['mark_iv']
        info['mark_price'] = order_book['mark_price']
        info['underlying_index'] = order_book['underlying_index']
        info['underlying_price'] = order_book['underlying_price']
        info['timestamp'] = order_book['timestamp']
        return info


    def download_instr_data(self, currency, instrument_type):
        """
        Download data from Deribit. Please refer to https://docs.deribit.com/#public-get_instruments for help.
        Arguments:
          - currency (string):        'ETH', 'BTC', 'USDC'
          - instrument_type (string): 'future', 'option'
        Returns: pd.DataFrame
        """
        data = self.get_instruments(currency, instrument_type)

        if instrument_type == 'future':
            fut_regex = r'(.+)-(.+)'
            get_exp = lambda x: re.findall(fut_regex, x)[0][1]

            df = pd.DataFrame.from_dict(data['result'])
            df = df[df['settlement_period'] != 'perpetual']
            df['expiry'] = df['instrument_name'].apply(get_exp)
            df['expiry_date'] = pd.to_datetime(df['expiry'], infer_datetime_format=True)
            df = df.sort_values(by = ['expiry_date']).reset_index()

            return df

        elif instrument_type == 'option':
            opt_regex = r'(.+)-(.+)-(.+)-(.+)'
            get_exp = lambda x: re.findall(opt_regex, x)[0][1]
            get_str = lambda x: re.findall(opt_regex, x)[0][2]

            df = pd.DataFrame.from_dict(data['result'])
            df['expiry'] = df['instrument_name'].apply(get_exp)
            df['expiry_date'] = pd.to_datetime(df['expiry'], infer_datetime_format=True)
            df['strike'] = df['instrument_name'].apply(get_str).astype(float, errors = 'raise')
            df = df.sort_values(by = ['expiry_date', 'strike']).reset_index()

            return df

        return 0

    @staticmethod
    def get_fut_exp(futures_df, underlying):
        """
        Returns the list of expiries for a given underlying.
        Arguments:
          - futures_df (pd.DataFrame):  Deribit data in DataFrame format
          - underlying (string):        'btc_usd', 'eth_usd'
        """
        df = futures_df[futures_df['price_index'] == underlying]
        df = df.sort_values(by = 'expiry_date')

        return df['expiry'].to_list()

    @staticmethod
    def get_strike(options_df, underlying, option_type, expiry_str):
        """
        Returns the list of strikes for a given underlying, type and expiry.
        Arguments:
          - options_df (pd.DataFrame):  Deribit data in DataFrame format
          - underlying (string):        'btc_usd', 'eth_usd'
          - option_type (string):       'call', 'put'
          - expiry_str (string):        DDMMMYY format (e.g. 09APR22)
        """
        df = options_df[options_df['price_index'] == underlying]
        df = df[df['option_type'] == option_type]
        df = df[df['expiry'] == expiry_str]
        df = df.sort_values(by = 'strike')

        return df['strike'].to_list()


    def download_all_instr_list(self):
        """
        Download the futures and options instruments list from Deribit (BTC and ETH)
        """
        eth_futures = self.download_instr_data('ETH', 'future')
        eth_options = self.download_instr_data('ETH', 'option')
        btc_futures = self.download_instr_data('BTC', 'future')
        btc_options = self.download_instr_data('BTC', 'option')
        futures = pd.concat([eth_futures, btc_futures])
        options = pd.concat([eth_options, btc_options])

        return futures, options

    def download_fut_expiry(self, undl):
        """
        Returns the list of expiries for a given future underlying.
        Arguments:
        - ws_client:                        Deribit API client
        - underlying (tuple(str, str)):     list of tuple in the format (option_underlying, future underlying)
                                            e.g. ('eth_usd', 'ETH')
        """
        futures = self.download_instr_data(self, undl[1], 'future')
        return get_fut_exp(futures, undl[0])

    @staticmethod
    def get_opt_instrument(undl, expiry, strike, opt_type):
        """
        Returns the correct format for an option instrument.
        Arguments:
         - undl (string):       deribit currency e.g. 'ETH'
         - expiry (string):     deribit expiry e.g. 30JUN23
         - strike (integer:     option strike
         - opt_type (string):   'C' or 'call" or 'P' or 'put'
        """
        if opt_type.lower() == 'call':
            opt_type = 'C'
        elif opt_type.lower() == 'put':
            opt_type = 'P'

        return '{}-{}-{}-{}'.format(undl, expiry, strike, opt_type)

client = WS_Client(client_id, client_secret)
