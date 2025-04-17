#!/usr/bin/env python3
"""
Aggregate OHLCV data from multiple exchanges asynchronously using ccxt.async_support.

This single script:
1. Defines a list of target exchanges.
2. Initializes ccxt exchange instances asynchronously.
3. Fetches OHLCV data concurrently, handling exchange-specific quirks.
4. Aggregates the results into a single pandas DataFrame.
5. Displays summary statistics using the rich library.
"""

import asyncio
import ccxt.async_support as ccxt_async # Use async version of ccxt
import pandas as pd
import os
import time
import traceback
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from datetime import datetime

# --- Configuration ---
EXCHANGES_TO_FETCH = [
    'binanceus',
    'bingx',
    'bitmart',
    'bitmex', # Note: Uses XBT internally, ccxt handles normalization
    'blofin', # Note: Requires :USDT suffix for spot symbols
    'btcalpha', # Note: Primarily USD, check alternatives
    'coinbase', # Note: Primarily USD
    'coinbaseexchange', # Advanced Trade API, Primarily USD
    'coinbaseinternational', # Primarily USD
    'gate', # Note: Primarily USDT/USDC
    'gemini', # Note: Primarily USD
    'kraken', # Note: Uses XBT/various quotes, ccxt handles normalization
    'kucoin', # Note: Primarily USDT/USDC/BTC
    'mexc',   # Note: Primarily USDT/USDC
    'okx',
    'poloniex' # Note: Primarily USDT, some legacy BTC
]

DEFAULT_SYMBOL = 'SOL/USD'
DEFAULT_TIMEFRAME = '1d'
DEFAULT_LIMIT = 30
MAX_RETRIES = 2
RETRY_DELAY = 3 # seconds

# Dictionary for exchange-specific configurations
EXCHANGE_CONFIGS = {
    'binanceus': {'options': {'defaultType': 'spot', 'adjustForTimeDifference': True}, 'timeout': 30000},
    'bingx': {'options': {'defaultType': 'spot', 'adjustForTimeDifference': True, 'createMarketBuyOrderRequiresPrice': False}, 'timeout': 30000},
    'bitmart': {'options': {'defaultType': 'spot', 'adjustForTimeDifference': True}, 'timeout': 30000},
    'bitmex': {'options': {'defaultType': 'spot', 'adjustForTimeDifference': True, 'fetchOHLCVWarning': False}, 'timeout': 30000},
    'blofin': {'options': {'defaultType': 'spot', 'adjustForTimeDifference': True, 'createMarketBuyOrderRequiresPrice': True}, 'timeout': 30000},
    'btcalpha': {'options': {'defaultType': 'spot', 'adjustForTimeDifference': True}, 'timeout': 30000},
    'coinbase': {'options': {'defaultType': 'spot', 'adjustForTimeDifference': True, 'createMarketBuyOrderRequiresPrice': True, 'fetchOHLCVWarning': False}, 'timeout': 30000},
    'coinbaseexchange': {'options': {'defaultType': 'spot', 'adjustForTimeDifference': True, 'createMarketBuyOrderRequiresPrice': True, 'fetchOHLCVWarning': False}, 'timeout': 30000},
    'coinbaseinternational': {'options': {'defaultType': 'spot', 'adjustForTimeDifference': True, 'createMarketBuyOrderRequiresPrice': True, 'fetchOHLCVWarning': False}, 'timeout': 30000},
    'coinlist': {
        'options': {'defaultType': 'spot', 'adjustForTimeDifference': True, 'createMarketBuyOrderRequiresPrice': True, 'fetchOHLCVWarning': False},
        'timeout': 30000,
        'valid_timeframe': '30m' # Override: Coinlist supports '1m', '5m', '30m'
        },
    'gate': {'options': {'defaultType': 'spot', 'adjustForTimeDifference': True, 'createMarketBuyOrderRequiresPrice': True}, 'timeout': 30000},
    'gemini': {'options': {'defaultType': 'spot', 'adjustForTimeDifference': True, 'fetchOHLCVWarning': False}, 'timeout': 30000},
    'kraken': {'options': {'defaultType': 'spot', 'adjustForTimeDifference': True, 'fetchOHLCVWarning': False}, 'timeout': 30000},
    'kucoin': {'options': {'defaultType': 'spot', 'adjustForTimeDifference': True, 'fetchOHLCVWarning': False}, 'timeout': 30000},
    'mexc': {'options': {'defaultType': 'spot', 'adjustForTimeDifference': True, 'createMarketBuyOrderRequiresPrice': True, 'fetchOHLCVWarning': False}, 'timeout': 30000},
    'okx': {'options': {'defaultType': 'spot', 'adjustForTimeDifference': True}, 'timeout': 30000},
    'poloniex': {'options': {'defaultType': 'spot', 'adjustForTimeDifference': True, 'fetchOHLCVWarning': False}, 'timeout': 30000},
}
# --- End Configuration ---

async def fetch_single_exchange_ohlcv(exchange_id, symbol, default_timeframe, limit):
    """
    Asynchronously fetches OHLCV data for a single exchange using ccxt.async_support.

    Args:
        exchange_id (str): The ccxt ID of the exchange.
        symbol (str): Trading pair symbol (e.g., 'BTC/USDT').
        default_timeframe (str): Default candle timeframe (e.g., '1h'). Can be overridden by config.
        limit (int): Number of candles to fetch.

    Returns:
        pandas.DataFrame: OHLCV data with an 'exchange' column, or None on failure.
    """
    exchange_class = getattr(ccxt_async, exchange_id, None)
    if not exchange_class:
        rprint(f"[red]Exchange ID '{exchange_id}' not found in ccxt.async_support.[/red]")
        return None

    exchange_specific_config = EXCHANGE_CONFIGS.get(exchange_id, {})

    # Base config + exchange-specific config
    config = {
        'enableRateLimit': True,
        'verbose': False, # Keep verbose off by default in aggregate runs
        'headers': { # Generic User-Agent
             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    }
    # Deep merge options if present
    if 'options' in exchange_specific_config:
        config.setdefault('options', {}).update(exchange_specific_config['options'])
    if 'timeout' in exchange_specific_config:
        config['timeout'] = exchange_specific_config['timeout']

    # Determine effective timeframe, allowing overrides
    effective_timeframe = exchange_specific_config.get('valid_timeframe', default_timeframe)

    exchange = exchange_class(config)
    rprint(f"[cyan]Attempting fetch from {exchange.id.upper()} (Timeframe: {effective_timeframe})...[/cyan]")

    current_symbol = symbol # Start with the default symbol
    ohlcv_data = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            await exchange.load_markets()

            # --- Handle Exchange Specific Symbol Logic ---
            original_symbol = current_symbol
            if exchange_id == 'blofin' and current_symbol.endswith('/USDT') and ':' not in current_symbol:
                # Check if the adjusted symbol actually exists before assigning it
                adjusted_blofin_symbol = f"{current_symbol}:USDT"
                if adjusted_blofin_symbol in exchange.markets:
                    current_symbol = adjusted_blofin_symbol
                    rprint(f"[{exchange.id.upper()}] Adjusted symbol to Blofin format: {current_symbol}")
                # If not, we'll rely on the alternative finding logic below


            # --- Symbol Availability Check & Alternatives ---
            if current_symbol not in exchange.markets:
                 rprint(f"[{exchange.id.upper()}] Symbol '{current_symbol}' not found directly. Trying alternatives...")
                 base, quote = symbol.split('/') if '/' in symbol else (symbol, None)
                 # Prioritize common quotes based on exchange tendencies if known, else use a standard list
                 if exchange_id in ['coinbase', 'coinbaseexchange', 'coinbaseinternational', 'gemini', 'kraken', 'btcalpha']:
                     possible_quotes = ['USD', 'USDT', 'USDC', 'BTC', 'EUR']
                 else: # Default for most others that heavily use USDT
                      possible_quotes = ['USDT', 'USDC', 'BTC', 'USD']

                 found_alternative = False
                 for alt_quote in possible_quotes:
                     if not base or not alt_quote: continue # Skip if split failed
                     alt_symbol = f"{base}/{alt_quote}"

                     # Handle Blofin's specific format again if trying USDT
                     if exchange_id == 'blofin' and alt_quote == 'USDT':
                         # Need to check the market list for the :USDT suffix version
                         if f"{alt_symbol}:USDT" in exchange.markets:
                            alt_symbol = f"{alt_symbol}:USDT"
                         else:
                             continue # Skip if the :USDT version doesn't exist

                     if alt_symbol in exchange.markets:
                         current_symbol = alt_symbol
                         rprint(f"[green][{exchange.id.upper()}] Found alternative symbol: {current_symbol}[/green]")
                         found_alternative = True
                         break

                 if not found_alternative:
                      rprint(f"[red][{exchange.id.upper()}] Symbol '{original_symbol}' and common alternatives not found.[/red]")
                      # List first few available markets for debugging help
                      available_markets = list(exchange.markets.keys())
                      rprint(f"[{exchange.id.upper()}] Available markets sample: {available_markets[:5]}")
                      await exchange.close()
                      return None # Give up for this exchange

            # --- Timeframe Validation ---
            if effective_timeframe not in exchange.timeframes:
                rprint(f"[red][{exchange.id.upper()}] Invalid timeframe: {effective_timeframe}. Available: {list(exchange.timeframes.keys())}[/red]")
                await exchange.close()
                return None

            # --- Fetch OHLCV Data ---
            # Always calculate 'since' based on limit and timeframe for robustness
            try:
                 since_ms = exchange.milliseconds() - exchange.parse_timeframe(effective_timeframe) * 1000 * limit
            except Exception as time_parse_e:
                 rprint(f"[red][{exchange.id.upper()}] Error parsing timeframe '{effective_timeframe}': {time_parse_e}. Cannot calculate 'since'.[/red]")
                 await exchange.close()
                 return None

            # Prepare params dictionary, potentially needed for some exchanges
            params = {}
            if exchange_id == 'coinlist':
                 # Coinlist might still prefer explicit start/end, calculate based on since_ms
                 now_ms = exchange.milliseconds()
                 params = {
                     'start_time': exchange.iso8601(since_ms),
                     'end_time': exchange.iso8601(now_ms), # End time can be current time
                 }

            ohlcv_data = await exchange.fetch_ohlcv(current_symbol, effective_timeframe, since=since_ms, limit=limit, params=params)

            # --- Data Validation ---
            if not isinstance(ohlcv_data, list):
                raise ValueError(f"Invalid data structure received. Expected list, got {type(ohlcv_data)}")
            if not ohlcv_data:
                # It's possible an exchange returns empty if data for the *exact* 'since' is unavailable
                # but might have slightly later data. Could add logic here to retry with slightly adjusted 'since' if needed.
                rprint(f"[yellow][{exchange.id.upper()}] No OHLCV data returned for {current_symbol} with since={datetime.fromtimestamp(since_ms/1000)}.[/yellow]")
                await exchange.close()
                return None # No data, not necessarily an error to retry

            # Filter out potential malformed candles (e.g., Coinlist nulls, incorrect length)
            valid_ohlcv = []
            for candle in ohlcv_data:
                 if isinstance(candle, list) and len(candle) == 6 and all(c is not None for c in candle[1:5]): # Check OHLC not None
                    ts = candle[0]
                    if isinstance(ts, str): # Handle string timestamps
                        try:
                           # Attempt parsing with pandas, robust for various ISO-like formats
                           ts = int(pd.to_datetime(ts, errors='coerce').timestamp() * 1000)
                           if pd.isna(ts): continue # Skip if parsing failed
                           candle[0] = ts
                        except Exception:
                            continue # Skip candle with bad timestamp format
                    elif not isinstance(ts, (int, float)):
                        continue # Skip candle with non-numeric timestamp

                    valid_ohlcv.append(candle)

            if not valid_ohlcv:
                rprint(f"[yellow][{exchange.id.upper()}] No valid candles found after filtering for {current_symbol}.[/yellow]")
                await exchange.close()
                return None

            # --- DataFrame Conversion and Validation ---
            df = pd.DataFrame(valid_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Check for NaNs *after* conversion attempt
            if df[numeric_cols].isnull().values.any():
                 nan_rows = df[df[numeric_cols].isnull().any(axis=1)]
                 rprint(f"[yellow][{exchange.id.upper()}] Found NaN values after numeric conversion for {current_symbol}. Dropping {len(nan_rows)} rows.[/yellow]")
                 #rprint(nan_rows) # Optional: print rows with NaNs for debugging
                 df.dropna(subset=numeric_cols, inplace=True)
                 if df.empty:
                     rprint(f"[yellow][{exchange.id.upper()}] DataFrame empty after dropping NaN rows. Skipping exchange.[/yellow]")
                     await exchange.close()
                     return None

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['exchange'] = exchange.id.upper()

            # Final check on candle count vs limit requested
            if len(df) < limit:
                 rprint(f"[yellow][{exchange.id.upper()}] Fetched {len(df)} candles, less than requested limit {limit}.[/yellow]")


            rprint(f"[green][{exchange.id.upper()}] Successfully fetched and processed {len(df)} candles for {current_symbol}.[/green]")
            await exchange.close()
            return df # Success

        except (ccxt_async.NetworkError, ccxt_async.ExchangeNotAvailable, ccxt_async.RateLimitExceeded, asyncio.TimeoutError) as e:
            if attempt < MAX_RETRIES:
                rprint(f"[yellow][{exchange.id.upper()}] Attempt {attempt + 1} failed ({type(e).__name__}): {str(e)[:150]}. Retrying in {RETRY_DELAY}s...[/yellow]") # Truncate long errors
                await asyncio.sleep(RETRY_DELAY + attempt) # Increase delay slightly on retries
            else:
                rprint(f"[red][{exchange.id.upper()}] Max retries reached. Failed due to {type(e).__name__}: {str(e)[:150]}[/red]")
                break # Exit retry loop
        except ccxt_async.ExchangeError as e:
            rprint(f"[red][{exchange.id.upper()}] Exchange error for {current_symbol}: {e}[/red]")
            # Specific handling for common symbol errors
            if 'symbol' in str(e).lower() or 'pair' in str(e).lower():
                 rprint(f"[yellow][{exchange.id.upper()}] Suggests symbol issue. Check market availability or symbol format ({current_symbol}).[/yellow]")
            break # Don't retry most exchange errors unless clearly temporary (like maintenance)
        except Exception as e:
            rprint(f"[red][{exchange.id.upper()}] Unexpected error processing {current_symbol}: {e}[/red]")
            rprint(f"[grey]{traceback.format_exc()}[/grey]")
            break # Exit loop on unexpected error

    await exchange.close() # Ensure closed even if loops break
    return None # Return None if all retries fail or loop breaks


async def aggregate_all_ohlcv(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME, limit=DEFAULT_LIMIT):
    """
    Fetches OHLCV data from all configured exchanges concurrently and aggregates it.

    Args:
        symbol (str): Trading pair symbol.
        timeframe (str): Default candle timeframe (can be overridden per exchange).
        limit (int): Number of candles.

    Returns:
        pandas.DataFrame: Aggregated OHLCV data from all successful fetches.
    """
    tasks = [
        fetch_single_exchange_ohlcv(exchange_id, symbol, timeframe, limit)
        for exchange_id in EXCHANGES_TO_FETCH
    ]

    rprint(f"\n[bold blue]Starting fetches for {len(tasks)} exchanges...[/bold blue]")
    results = await asyncio.gather(*tasks, return_exceptions=True) # Capture exceptions

    successful_dfs = []
    failed_count = 0
    for i, result in enumerate(results):
        exchange_id = EXCHANGES_TO_FETCH[i]
        if isinstance(result, pd.DataFrame) and not result.empty:
            successful_dfs.append(result)
        elif isinstance(result, Exception):
            rprint(f"[red]Fetch failed for {exchange_id.upper()} with exception: {result}[/red]")
            failed_count += 1
        else: # result was None or an empty DataFrame
            # Error/reason should have been logged in fetch_single_exchange_ohlcv
            failed_count += 1

    if not successful_dfs:
        rprint("[bold red]Failed to fetch data from any exchange.[/bold red]")
        return pd.DataFrame() # Return empty DataFrame

    # Concatenate all successful DataFrames
    all_data = pd.concat(successful_dfs, ignore_index=True)

    # Sort by timestamp just in case
    all_data.sort_values(by='timestamp', inplace=True)

    rprint(f"\n[bold green]Successfully aggregated data from {len(successful_dfs)} exchanges. ({failed_count} failures)[/bold green]")
    return all_data

def display_summary_statistics(df, symbol, default_timeframe):
    """
    Calculates and displays summary statistics for the aggregated OHLCV data.

    Args:
        df (pandas.DataFrame): Aggregated OHLCV data with 'exchange' column.
        symbol (str): The symbol used for fetching.
        default_timeframe (str): The default timeframe used (overrides noted in logs).
    """
    if df.empty:
        rprint("[yellow]No data available to display statistics.[/yellow]")
        return

    console = Console()
    title = f"Aggregated OHLCV Summary for {symbol} (Default Timeframe: {default_timeframe})"
    console.print(f"\n[bold underline magenta]{title}[/bold underline magenta]")

    # Ensure numeric types (should be handled earlier, but good practice)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)

    if df.empty:
        rprint("[yellow]No valid numeric data available after cleaning.[/yellow]")
        return

    # Aggregate Statistics
    if df.empty: # Check again after potential dropna
        rprint("[yellow]DataFrame became empty after removing non-numeric rows.[/yellow]")
        return

    overall_open_row = df.loc[df['timestamp'].idxmin()]
    overall_close_row = df.loc[df['timestamp'].idxmax()]
    overall_open = overall_open_row['open']
    overall_close = overall_close_row['close']

    overall_high = df['high'].max()
    overall_low = df['low'].min()
    total_volume = df['volume'].sum()
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    total_candles = len(df)
    unique_exchanges = df['exchange'].nunique()

    summary_table = Table(title="Overall Market Statistics", show_header=False, box=None, padding=(0,1))
    summary_table.add_column("Metric", style="cyan", no_wrap=True)
    summary_table.add_column("Value", style="white")

    summary_table.add_row("Time Range Start:", str(start_time))
    summary_table.add_row("Time Range End:", str(end_time))
    summary_table.add_row("Total Candles Aggregated:", f"{total_candles:,}")
    summary_table.add_row("Number of Exchanges:", str(unique_exchanges))
    summary_table.add_row("Overall Period Open:", f"{overall_open:.4f} (from {overall_open_row['exchange']} at {overall_open_row['timestamp']})")
    summary_table.add_row("Overall Period High:", f"{overall_high:.4f}")
    summary_table.add_row("Overall Period Low:", f"{overall_low:.4f}")
    summary_table.add_row("Overall Period Close:", f"{overall_close:.4f} (from {overall_close_row['exchange']} at {overall_close_row['timestamp']})")
    summary_table.add_row("Total Aggregated Volume:", f"{total_volume:,.4f}")

    console.print(summary_table)

    # Per-Exchange Summary
    console.print("\n[bold underline magenta]Per-Exchange Summary (Latest Data & Totals)[/bold underline magenta]")
    exchange_summary_table = Table(title="Statistics per Exchange", show_header=True, header_style="bold blue")
    exchange_summary_table.add_column("Exchange", style="cyan", footer="AVERAGE")
    exchange_summary_table.add_column("Latest Timestamp", style="dim")
    exchange_summary_table.add_column("Latest Close", justify="right", footer_style="bold")
    exchange_summary_table.add_column("Total Volume", justify="right", footer_style="bold")
    exchange_summary_table.add_column("Candle Count", justify="right", footer_style="bold")
    exchange_summary_table.add_column("Avg Volume / Candle", justify="right", footer_style="bold")

    total_vol_sum = 0
    latest_close_sum = 0
    total_candle_count = 0
    valid_closes = 0

    exchanges_sorted = sorted(df['exchange'].unique()) # Sort for consistent display

    for exchange_name in exchanges_sorted:
        group = df[df['exchange'] == exchange_name]
        if group.empty: continue # Should not happen if df is checked earlier

        latest_candle = group.loc[group['timestamp'].idxmax()]
        exchange_total_volume = group['volume'].sum()
        candle_count = len(group)
        avg_vol_per_candle = exchange_total_volume / candle_count if candle_count > 0 else 0

        # Check for NaN close before adding to sum
        if pd.notna(latest_candle['close']):
            latest_close_sum += latest_candle['close']
            valid_closes += 1

        total_vol_sum += exchange_total_volume
        total_candle_count += candle_count

        exchange_summary_table.add_row(
            exchange_name,
            str(latest_candle['timestamp']),
            f"{latest_candle['close']:.4f}" if pd.notna(latest_candle['close']) else "[red]N/A[/red]",
            f"{exchange_total_volume:,.4f}",
            f"{candle_count:,}",
            f"{avg_vol_per_candle:.4f}"
        )

    # Calculate averages for the footer
    avg_latest_close = latest_close_sum / valid_closes if valid_closes > 0 else 0
    # Average volume across exchanges (might be less meaningful than total)
    #avg_total_volume = total_vol_sum / len(exchanges_sorted) if exchanges_sorted else 0
    # Let's show Total Volume in footer instead of Average Total Volume per Exchange
    avg_candle_count = total_candle_count / len(exchanges_sorted) if exchanges_sorted else 0
    avg_vol_per_candle_overall = total_vol_sum / total_candle_count if total_candle_count > 0 else 0

    # Set footer values explicitly for Rich Table
    exchange_summary_table.columns[2].footer = f"{avg_latest_close:.4f}"
    exchange_summary_table.columns[3].footer = f"{total_vol_sum:,.4f} (TOTAL)" # Show total volume
    exchange_summary_table.columns[4].footer = f"{avg_candle_count:,.1f}"
    exchange_summary_table.columns[5].footer = f"{avg_vol_per_candle_overall:.4f}"

    console.print(exchange_summary_table)


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fetch and aggregate OHLCV data from multiple exchanges.")
    parser.add_argument('--symbol', type=str, default=DEFAULT_SYMBOL, help=f"Trading symbol (default: {DEFAULT_SYMBOL})")
    parser.add_argument('--timeframe', type=str, default=DEFAULT_TIMEFRAME, help=f"Default candle timeframe (default: {DEFAULT_TIMEFRAME})")
    parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT, help=f"Number of candles (default: {DEFAULT_LIMIT})")
    args = parser.parse_args()

    symbol = args.symbol
    timeframe = args.timeframe # This is the default, might be overridden
    limit = args.limit

    rprint(f"[bold yellow]Starting aggregation for {symbol} (Default Timeframe: {timeframe}, Limit: {limit})...[/bold yellow]")
    start_run_time = time.time()

    aggregated_data = await aggregate_all_ohlcv(symbol, timeframe, limit)

    end_run_time = time.time()
    rprint(f"\n[italic]Data fetching and aggregation took {end_run_time - start_run_time:.2f} seconds.[/italic]")

    if not aggregated_data.empty:
        display_summary_statistics(aggregated_data, symbol, timeframe)
        # Optionally save the aggregated data
        # filename = f"aggregated_{symbol.replace('/', '-')}_{timeframe}.csv"
        # try:
        #     aggregated_data.to_csv(filename, index=False)
        #     rprint(f"\n[italic blue]Aggregated data saved to {filename}[/italic blue]")
        # except Exception as save_e:
        #     rprint(f"[red]Error saving file {filename}: {save_e}[/red]")
    else:
        rprint("[bold red]No data was successfully aggregated.[/bold red]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        rprint("\n[bold orange_red1]Execution interrupted by user.[/bold orange_red1]")
    except Exception as e:
         rprint(f"\n[bold red]An error occurred during execution:[/bold red] {e}")
         rprint(f"[grey]{traceback.format_exc()}[/grey]")