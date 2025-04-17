#!/usr/bin/env python3
"""
Basic example of fetching OHLCV (Open, High, Low, Close, Volume) data from Bybit Spot.
This script demonstrates how to:
1. Initialize the Bybit exchange
2. Fetch OHLCV data for a specific symbol
3. Convert the data to a pandas DataFrame for analysis
4. Handle basic error cases

Note: Bybit primarily uses USDT pairs for spot trading, with some USDC pairs available.
Note: Bybit is not accessible from the US so will not work from a US based IP address.
"""

import ccxt
import pandas as pd
from datetime import datetime
import pytz
from rich import print
from rich.console import Console
from rich.table import Table
import time

def fetch_ohlcv(symbol='BTC/USDT', timeframe='1h', limit=100, max_retries=3):
    """
    Fetch OHLCV data from Bybit for a given symbol and timeframe.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
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
            # Initialize Bybit exchange with configuration
            exchange = ccxt.bybit({
                'enableRateLimit': True,
                'timeout': 30000,  # 30 seconds
                'options': {
                    'defaultType': 'spot',  # ensure we're using spot trading
                    'adjustForTimeDifference': True,  # handle server time differences
                    'recvWindow': 5000,  # Bybit specific parameter
                },
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
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
            
            # First check if the exact symbol exists
            original_symbol = symbol
            if symbol in exchange.markets:
                print(f"[green]Found exact symbol: {symbol}[/green]")
            # Try alternatives if symbol not found
            else:
                base = symbol.split('/')[0]
                # Bybit primarily uses USDT, then USDC
                alternative_symbols = [f"{base}/USDT", f"{base}/USDC"]
                print(f"[yellow]Symbol not found. Trying alternatives: {', '.join(alternative_symbols)}[/yellow]")
                
                # Try each alternative symbol
                for alt_symbol in alternative_symbols:
                    if alt_symbol in exchange.markets:
                        symbol = alt_symbol
                        print(f"[green]Found alternative pair: {symbol}[/green]")
                        break
            
            # If still not found, show available pairs
            if symbol not in exchange.markets:
                print(f"[red]Symbol {original_symbol} and its alternatives not found in available markets![/red]")
                # Show some popular USDT and USDC pairs
                usdt_pairs = [s for s in exchange.markets.keys() if s.endswith('/USDT')][:5]
                usdc_pairs = [s for s in exchange.markets.keys() if s.endswith('/USDC')][:5]
                print(f"Popular USDT pairs: {', '.join(usdt_pairs)}...")
                if usdc_pairs:
                    print(f"Popular USDC pairs: {', '.join(usdc_pairs)}...")
                return None
                
            exchange_symbol = exchange.market_id(symbol)
            print(f"\n[bold]Using exchange symbol:[/bold] {exchange_symbol}")
            
            # Calculate since timestamp (24 hours ago by default)
            since = exchange.milliseconds() - (86400 * 1000)  # 24 hours ago
            
            print(f"[yellow]Fetching OHLCV data (attempt {attempt + 1}/{max_retries})...[/yellow]")
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            except ccxt.BadRequest as e:
                print(f"[red]Bad request error: {str(e)}[/red]")
                print("[yellow]Trying without explicit timestamps...[/yellow]")
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
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    print(f"[red]Error converting {col} to numeric: {str(e)}[/red]")
                    return None
            
            # Check for any NaN values after conversion
            if df[numeric_cols].isna().any().any():
                print("[red]Found NaN values after numeric conversion[/red]")
                return None
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Store the quote currency for display
            quote_currency = symbol.split('/')[1]
            df.attrs['quote_currency'] = quote_currency
            
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
            print("1. Rate limiting")
            print("2. Invalid symbol")
            print("3. Exchange maintenance")
            print("4. API key requirements (some endpoints may need authentication)[/yellow]")
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
    
    quote_currency = df.attrs.get('quote_currency', 'USDT')
    
    # Add columns
    table.add_column("Timestamp")
    table.add_column(f"Open ({quote_currency})", justify="right")
    table.add_column(f"High ({quote_currency})", justify="right")
    table.add_column(f"Low ({quote_currency})", justify="right")
    table.add_column(f"Close ({quote_currency})", justify="right")
    table.add_column("Volume", justify="right")
    
    # Add rows (last 10 entries)
    for _, row in df.tail(10).iterrows():
        table.add_row(
            row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            f"{row['open']:.2f}",  # 8 decimal places for crypto prices
            f"{row['high']:.2f}",
            f"{row['low']:.2f}",
            f"{row['close']:.2f}",
            f"{row['volume']:.8f}"  # 8 decimal places for volume
        )
    
    console.print(table)

def main():
    # Example usage
    symbol = 'BTC/USDT'  # Bybit primarily uses USDT pairs
    timeframe = '1h'
    limit = 100
    
    print(f"[yellow]Fetching {timeframe} OHLCV data for {symbol} from Bybit Spot...[/yellow]")
    
    df = fetch_ohlcv(symbol, timeframe, limit)
    if df is not None and not df.empty:
        print("\n[bold]Last 10 candles:[/bold]")
        display_ohlcv(df)
        
        # Get the actual quote currency used
        quote_currency = df.attrs.get('quote_currency', 'USDT')
        
        # Print some basic statistics
        print("\n[bold]Basic Statistics:[/bold]")
        print(f"Latest close price: {df['close'].iloc[-1]:.8f} {quote_currency}")
        
        # Calculate 24h price change only if we have enough data
        if len(df) >= 24:
            price_change = ((df['close'].iloc[-1] - df['close'].iloc[-24]) / df['close'].iloc[-24] * 100)
            print(f"24h price change: {price_change:.2f}%")
        else:
            print(f"[yellow]Not enough data for 24h price change calculation (need 24 candles, got {len(df)})[/yellow]")
            
        print(f"Total volume: {df['volume'].sum():.8f}")
    else:
        print("[red]Failed to fetch OHLCV data[/red]")

if __name__ == "__main__":
    main() 