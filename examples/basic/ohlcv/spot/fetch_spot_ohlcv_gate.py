#!/usr/bin/env python3
"""
Fetch OHLCV data from Gate.io using the CCXT library.

Gate.io primarily uses USDT as the quote currency, with some pairs also available in USDC.
The exchange has specific rate limits and requires proper error handling for network issues.

Example usage:
    python examples/basic/fetch_ohlcv_gate.py
"""

import sys
import os
from datetime import datetime, timedelta
import time
import pandas as pd
from rich import print as rprint
from rich.console import Console
from rich.table import Table
import ccxt

# Add parent directory to path to allow imports from parent
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)


def fetch_ohlcv(symbol='BTC/USDT', timeframe='1h', limit=10, retries=3):
    """
    Fetch OHLCV data from Gate.io for a given symbol and timeframe.

    Args:
        symbol (str): Trading pair symbol (default: 'BTC/USDT')
        timeframe (str): Timeframe for candles (default: '1h')
        limit (int): Number of candles to fetch (default: 10)
        retries (int): Number of retry attempts for failed requests (default: 3)

    Returns:
        tuple: (pandas DataFrame with OHLCV data, quote currency string)
    """
    exchange = ccxt.gate({
        'enableRateLimit': True,
        'timeout': 30000,  # 30 seconds
        'options': {
            'defaultType': 'spot',  # Use spot markets by default
            'createMarketBuyOrderRequiresPrice': True,  # Gate.io requires the price for market buy orders
        }
    })

    # Gate.io has specific rate limits, so we'll add a small delay between retries
    retry_delay = 2  # seconds

    while retries > 0:
        try:
            # Validate timeframe
            timeframes = exchange.timeframes
            if timeframe not in timeframes:
                raise ValueError(f"Timeframe '{timeframe}' not supported. Available timeframes: {timeframes}")

            # Try to fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) == 0:
                # If the symbol is not found, try to find alternative pairs
                markets = exchange.load_markets()
                available_symbols = list(markets.keys())
                
                # Gate.io primarily uses USDT, try USDT pairs first
                usdt_pairs = [s for s in available_symbols if s.endswith('/USDT')]
                usdc_pairs = [s for s in available_symbols if s.endswith('/USDC')]
                
                if symbol not in available_symbols:
                    alternatives = []
                    base = symbol.split('/')[0]
                    # Look for alternative quote currencies
                    alternatives.extend([s for s in usdt_pairs if s.startswith(f"{base}/")])
                    alternatives.extend([s for s in usdc_pairs if s.startswith(f"{base}/")])
                    
                    if alternatives:
                        raise ValueError(f"Symbol '{symbol}' not found. Did you mean one of these? {alternatives}")
                    else:
                        raise ValueError(f"Symbol '{symbol}' not found. Available USDT pairs: {usdt_pairs[:5]}")

            # Convert to pandas DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Validate numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Check for any NaN values that would indicate conversion errors
            if df[numeric_columns].isna().any().any():
                raise ValueError("Error converting data: some values could not be converted to numbers")
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Extract quote currency for display
            quote_currency = symbol.split('/')[1] if '/' in symbol else 'USDT'
            
            return df, quote_currency

        except ccxt.NetworkError as e:
            retries -= 1
            if retries > 0:
                rprint(f"[yellow]Network error: {str(e)}. Retrying in {retry_delay} seconds... ({retries} attempts remaining)[/yellow]")
                time.sleep(retry_delay)
            else:
                raise Exception(f"Failed to fetch data after multiple retries: {str(e)}")
                
        except ccxt.ExchangeError as e:
            retries -= 1
            if retries > 0:
                rprint(f"[yellow]Exchange error: {str(e)}. Retrying in {retry_delay} seconds... ({retries} attempts remaining)[/yellow]")
                time.sleep(retry_delay)
            else:
                raise Exception(f"Exchange error: {str(e)}")
                
        except ValueError as e:
            raise e
        
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")


def display_ohlcv(df, quote_currency):
    """
    Display OHLCV data in a formatted table.

    Args:
        df (pandas.DataFrame): DataFrame containing OHLCV data
        quote_currency (str): Quote currency for price formatting
    """
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    
    # Add columns
    table.add_column("Timestamp", style="dim")
    table.add_column(f"Open ({quote_currency})", justify="right")
    table.add_column(f"High ({quote_currency})", justify="right")
    table.add_column(f"Low ({quote_currency})", justify="right")
    table.add_column(f"Close ({quote_currency})", justify="right")
    table.add_column(f"Volume", justify="right")
    
    # Format and add rows
    for _, row in df.iterrows():
        table.add_row(
            row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            f"{row['open']:.8f}",  # Gate.io prices often need more decimal places
            f"{row['high']:.8f}",
            f"{row['low']:.8f}",
            f"{row['close']:.8f}",
            f"{row['volume']:.8f}"
        )
    
    console.print(table)


def main():
    """
    Main function to fetch and display OHLCV data from Gate.io.
    """
    try:
        # Fetch OHLCV data
        df, quote_currency = fetch_ohlcv()
        
        # Display the data
        rprint("\n[bold green]Last 10 candles:[/bold green]")
        display_ohlcv(df, quote_currency)
        
        # Display some basic statistics
        rprint("\n[bold green]Basic statistics:[/bold green]")
        stats = df.describe()
        rprint(f"Average close price: {stats['close']['mean']:.8f} {quote_currency}")
        rprint(f"Highest price: {stats['high']['max']:.8f} {quote_currency}")
        rprint(f"Lowest price: {stats['low']['min']:.8f} {quote_currency}")
        rprint(f"Total volume: {df['volume'].sum():.8f}")  # Calculate sum directly from the DataFrame
        
    except Exception as e:
        rprint(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 