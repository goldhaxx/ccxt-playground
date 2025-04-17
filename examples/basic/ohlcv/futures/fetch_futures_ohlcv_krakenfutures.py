#!/usr/bin/env python3
"""
Basic example of fetching OHLCV (Open, High, Low, Close, Volume) data from Kraken Futures.
This script demonstrates how to:
1. Initialize the Kraken Futures exchange connection
2. Fetch OHLCV data for a specific futures contract
3. Convert the data to a pandas DataFrame for analysis
4. Handle basic error cases

Note:
    Kraken Futures uses different symbol formats than Kraken spot:
    - Perpetual futures use format 'SYMBOL/USD:USD' (e.g., 'BTC/USD:USD')
    - Quarterly futures use similar format with month codes (e.g., 'BTC/USD:USTM24' for June 2024)
"""

import ccxt
import pandas as pd
from datetime import datetime
import pytz
from rich import print
from rich.console import Console
from rich.table import Table
import time

def fetch_ohlcv(symbol='BTC/USD:USD', timeframe='1h', limit=100, max_retries=3):
    """
    Fetch OHLCV data from Kraken Futures for a given symbol and timeframe.
    
    Args:
        symbol (str): Trading pair symbol in Kraken Futures format (e.g., 'BTC/USD:USD' for perpetual)
                     For quarterly futures, use format like 'BTC/USD:USTM24' (June 2024)
        timeframe (str): Candle timeframe (e.g., '1m', '5m', '1h', '1d')
        limit (int): Number of candles to fetch
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        pandas.DataFrame: OHLCV data with columns [timestamp, open, high, low, close, volume]
        None: If data cannot be fetched or is invalid
        
    Note:
        Kraken Futures uses different symbol notation than Kraken spot:
        - Perpetual contracts use 'SYMBOL/USD:USD' format
        - Quarterly futures use month codes (M: March, U: September, etc.)
    """
    exchange = None
    for attempt in range(max_retries):
        try:
            # Initialize Kraken Futures exchange with configuration
            exchange = ccxt.krakenfutures({
                'enableRateLimit': True,
                'timeout': 30000,  # 30 seconds
                'verbose': True,  # Enable request/response debugging
                'options': {
                    'adjustForTimeDifference': True,  # handle server time differences
                    'fetchOHLCVWarning': False,  # Suppress OHLCV warning
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
            
            # Save original symbol for error reporting
            original_symbol = symbol
            
            # Check if symbol exists in exchange markets
            if symbol not in exchange.markets:
                print(f"[red]Symbol {original_symbol} not found in Kraken Futures markets![/red]")
                
                # Try to find alternative perpetual contracts
                if symbol == 'BTC/USD:USD':
                    alternatives = ['ETH/USD:USD', 'SOL/USD:USD']
                    print(f"[yellow]USD perpetual not found. Trying alternatives: {', '.join(alternatives)}[/yellow]")
                    
                    # Try each alternative symbol
                    for alt_symbol in alternatives:
                        if alt_symbol in exchange.markets:
                            symbol = alt_symbol
                            print(f"[green]Found alternative contract: {symbol}[/green]")
                            break
                
                # If still not found
                if symbol not in exchange.markets:
                    # List some available perpetual contracts
                    perp_pairs = [s for s in exchange.markets.keys() if s.endswith(':USD')][:5]
                    futures_pairs = [s for s in exchange.markets.keys() if ':UST' in s][:5]
                    
                    if perp_pairs:
                        print(f"Available perpetual contracts: {', '.join(perp_pairs)}...")
                    if futures_pairs:
                        print(f"Available futures contracts: {', '.join(futures_pairs)}...")
                    
                    return None
            
            print(f"[green]Using contract: {symbol}[/green]")
            
            # Show contract details for debugging
            if symbol in exchange.markets:
                market = exchange.markets[symbol]
                contract_type = market.get('type', 'unknown')
                expiry = market.get('expiry', 'perpetual')
                print(f"[blue]Contract type: {contract_type}, Expiry: {expiry}[/blue]")
            
            # Calculate since timestamp (24 hours ago by default)
            since = exchange.milliseconds() - (86400 * 1000)  # 24 hours ago in milliseconds
            
            print(f"[yellow]Fetching OHLCV data (attempt {attempt + 1}/{max_retries})...[/yellow]")
            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit
                )
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
            
            # Store contract info for display
            df.attrs['contract_type'] = contract_type if 'contract_type' in locals() else 'unknown'
            df.attrs['expiry'] = expiry if 'expiry' in locals() else 'unknown'
            
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
            print("2. Invalid contract symbol")
            print("3. Exchange maintenance")
            print("4. API key requirements (some endpoints may need authentication)")
            print("5. Contract expiry (for quarterly futures)[/yellow]")
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
            f"{row['open']:.2f}",  # 2 decimal places for futures prices
            f"{row['high']:.2f}",
            f"{row['low']:.2f}",
            f"{row['close']:.2f}",
            f"{row['volume']:.8f}"  # 8 decimal places for volume
        )
    
    console.print(table)

def main():
    # Example usage
    symbol = 'BTC/USD:USD'  # Use Kraken Futures format (perpetual contract)
    timeframe = '1h'
    limit = 100
    
    print(f"[yellow]Fetching {timeframe} OHLCV data for {symbol} from Kraken Futures...[/yellow]")
    print("[yellow]Note: For quarterly futures, use format like 'BTC/USD:USTM24' (June 2024)[/yellow]")
    
    df = fetch_ohlcv(symbol, timeframe, limit)
    if df is not None and not df.empty:
        print("\n[bold]Last 10 candles:[/bold]")
        display_ohlcv(df)
        
        # Get contract information
        contract_type = df.attrs.get('contract_type', 'unknown')
        expiry = df.attrs.get('expiry', 'unknown')
        
        # Print some basic statistics
        print("\n[bold]Basic Statistics:[/bold]")
        print(f"Contract type: {contract_type}")
        if expiry != 'perpetual':
            print(f"Expiry: {expiry}")
        print(f"Latest close price: {df['close'].iloc[-1]:.2f} USD")
        
        # Calculate 24h price change only if we have enough data
        if len(df) >= 24:
            price_change = ((df['close'].iloc[-1] - df['close'].iloc[-24]) / df['close'].iloc[-24] * 100)
            print(f"24h price change: {price_change:.2f}%")
        else:
            print(f"[yellow]Not enough data for 24h price change calculation (need 24 candles, got {len(df)})[/yellow]")
            
        print(f"Total volume: {df['volume'].sum():.8f}")
        
        # Calculate funding rate if available
        if contract_type == 'perpetual':
            print("[yellow]Note: Funding rates can be fetched separately using the exchange's API[/yellow]")
    else:
        print("[red]Failed to fetch OHLCV data[/red]")

if __name__ == "__main__":
    main() 