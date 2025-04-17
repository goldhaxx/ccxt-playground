#!/usr/bin/env python3
"""
Basic example of fetching OHLCV (Open, High, Low, Close, Volume) data from Alpaca.
This script demonstrates how to:
1. Initialize the Alpaca exchange with API credentials
2. Fetch OHLCV data for a specific symbol
3. Convert the data to a pandas DataFrame for analysis
4. Handle basic error cases

Note: You need to set your Alpaca API key and secret as environment variables:
export ALPACA_API_KEY='your_api_key'
export ALPACA_SECRET_KEY='your_secret_key'
"""

import ccxt
import pandas as pd
from datetime import datetime
import pytz
from rich import print
from rich.console import Console
from rich.table import Table
import time
import os

def fetch_ohlcv(symbol='SPY/USD', timeframe='1h', limit=100, max_retries=3):
    """
    Fetch OHLCV data from Alpaca for a given symbol and timeframe.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'SPY/USD' for stocks, 'BTC/USD' for crypto)
        timeframe (str): Candle timeframe (e.g., '1m', '5m', '1h', '1d')
        limit (int): Number of candles to fetch
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        pandas.DataFrame: OHLCV data with columns [timestamp, open, high, low, close, volume]
        None: If data cannot be fetched or is invalid
    """
    exchange = None
    for attempt in range(max_retries):
        try:
            # Check for API credentials
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            
            if not api_key or not secret_key:
                print("[red]Error: Alpaca API credentials not found in environment variables[/red]")
                print("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
                return None

            # Initialize Alpaca exchange with configuration
            exchange = ccxt.alpaca({
                'apiKey': api_key,
                'secret': secret_key,
                'enableRateLimit': True,
                'timeout': 30000,  # 30 seconds
                'verbose': True,  # Enable request/response debugging
                'options': {
                    'defaultType': 'spot',  # use spot trading
                    'adjustForTimeDifference': True,  # handle server time differences
                    'warnOnFetchOHLCVLimitArgument': True,  # warn about OHLCV limits
                },
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
                }
            })
            
            # Print exchange capabilities
            print(f"\n[bold]Exchange Capabilities:[/bold]")
            print(f"Has fetchOHLCV: {exchange.has['fetchOHLCV']}")
            print(f"Timeframes: {exchange.timeframes}")
            
            # Validate timeframe
            if timeframe not in exchange.timeframes:
                print(f"[red]Invalid timeframe: {timeframe}[/red]")
                print(f"Available timeframes: {', '.join(exchange.timeframes)}")
                return None
            
            print(f"\n[yellow]Loading markets (attempt {attempt + 1}/{max_retries})...[/yellow]")
            exchange.load_markets()
            
            # Convert symbol to exchange specific format if needed
            if symbol not in exchange.markets:
                print(f"[red]Symbol {symbol} not found in available markets![/red]")
                available_symbols = [s for s in exchange.markets.keys() if 'USD' in s][:5]
                print(f"Available USD pairs: {', '.join(available_symbols)}...")
                return None
                
            exchange_symbol = exchange.market_id(symbol)
            print(f"\n[bold]Using exchange symbol:[/bold] {exchange_symbol}")
            
            print(f"[yellow]Fetching OHLCV data (attempt {attempt + 1}/{max_retries})...[/yellow]")
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Validate OHLCV data structure
            if not isinstance(ohlcv, list):
                print(f"[red]Invalid OHLCV data structure. Expected list, got {type(ohlcv)}[/red]")
                return None
                
            if not ohlcv:
                print(f"[red]No data returned for {symbol}[/red]")
                return None
            
            # Validate OHLCV data format
            if not all(len(candle) == 6 for candle in ohlcv):
                print("[red]Invalid OHLCV data format. Each candle should have 6 values.[/red]")
                return None
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Validate numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if not pd.to_numeric(df[col], errors='coerce').notna().all():
                    print(f"[red]Invalid numeric data in {col} column[/red]")
                    return None
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            print(f"[green]Successfully fetched {len(df)} candles[/green]")
            return df
        
        except ccxt.NetworkError as e:
            print(f"[red]Network error occurred (attempt {attempt + 1}/{max_retries}): {str(e)}[/red]")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"[yellow]Waiting {wait_time} seconds before retrying...[/yellow]")
                time.sleep(wait_time)
            else:
                print("[red]Max retries reached. Please check your internet connection and try again.[/red]")
                return None
                
        except ccxt.ExchangeError as e:
            print(f"[red]Exchange error occurred: {str(e)}[/red]")
            print("[yellow]This might be due to:")
            print("1. Invalid API credentials")
            print("2. Rate limiting")
            print("3. Invalid symbol")
            print("4. Exchange maintenance[/yellow]")
            return None
            
        except Exception as e:
            print(f"[red]An unexpected error occurred: {str(e)}[/red]")
            if exchange:
                print(f"Exchange info: {exchange.id}, {exchange.version}")
            return None

def display_ohlcv(df):
    """
    Display OHLCV data in a formatted table using Rich.
    """
    if df is None:
        print("[red]No data to display[/red]")
        return
    
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    
    # Add columns
    table.add_column("Timestamp")
    table.add_column("Open", justify="right")
    table.add_column("High", justify="right")
    table.add_column("Low", justify="right")
    table.add_column("Close", justify="right")
    table.add_column("Volume", justify="right")
    
    # Add rows (last 10 entries)
    for _, row in df.tail(10).iterrows():
        table.add_row(
            row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            f"{row['open']:.2f}",
            f"{row['high']:.2f}",
            f"{row['low']:.2f}",
            f"{row['close']:.2f}",
            f"{row['volume']:.2f}"
        )
    
    console.print(table)

def main():
    # Example usage
    symbol = 'SPY/USD'  # Using SPY (S&P 500 ETF) as default
    timeframe = '1h'
    limit = 100
    
    print(f"[yellow]Fetching {timeframe} OHLCV data for {symbol} from Alpaca...[/yellow]")
    
    df = fetch_ohlcv(symbol, timeframe, limit)
    if df is not None and not df.empty:
        print("\n[bold]Last 10 candles:[/bold]")
        display_ohlcv(df)
        
        # Print some basic statistics
        print("\n[bold]Basic Statistics:[/bold]")
        print(f"Latest close price: {df['close'].iloc[-1]:.2f} USD")
        
        # Calculate 24h price change only if we have enough data
        if len(df) >= 24:
            price_change = ((df['close'].iloc[-1] - df['close'].iloc[-24]) / df['close'].iloc[-24] * 100)
            print(f"24h price change: {price_change:.2f}%")
        else:
            print(f"[yellow]Not enough data for 24h price change calculation (need 24 candles, got {len(df)})[/yellow]")
            
        print(f"Total volume: {df['volume'].sum():.2f}")
    else:
        print("[red]Failed to fetch OHLCV data[/red]")

if __name__ == "__main__":
    main() 