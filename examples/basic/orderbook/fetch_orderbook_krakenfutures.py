# -*- coding: utf-8 -*-

import os
import sys
from pprint import pprint
import ccxt  # noqa: E402

print("CCXT Version:", ccxt.__version__)

print ("exchange = ccxt.krakenfutures()")
exchange = ccxt.krakenfutures()
markets = exchange.load_markets()
# exchange.verbose = True  # uncomment for debugging purposes if necessary
print(exchange.name, "supports the following methods:")
pprint(exchange.has)
print(exchange.name, "supports the following trading symbols:")
for symbol in exchange.symbols:
   print(symbol)
symbol = 'BTC/USD:USD'
pprint(symbol)
orderbook = exchange.fetch_order_book(symbol)
pprint(orderbook) 
