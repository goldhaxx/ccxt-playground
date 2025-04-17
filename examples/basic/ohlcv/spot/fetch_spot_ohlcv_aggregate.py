#!/usr/bin/env python3
"""
Aggregate OHLCV data from multiple exchanges asynchronously using ccxt.async_support.

This single script:
1. Defines a list of target exchanges.
2. Initializes ccxt exchange instances asynchronously.
3. Identifies all markets matching BASE/<STABLE_QUOTE> for the requested symbol on each exchange.
4. Fetches OHLCV data concurrently for *all* identified pairs per exchange.
5. Aggregates the results into a single pandas DataFrame.
6. Displays summary statistics using the rich library, grouped by exchange and symbol.
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
    # 'btcalpha', # Often fails or has limited pairs
    'coinbase', # Note: Primarily USD
    'coinbaseexchange', # Advanced Trade API, Primarily USD
    'coinbaseinternational', # Primarily USD
    # 'coinlist', # Very limited API, often fails timeframe/pair validation
    'gate', # Note: Primarily USDT/USDC
    'gemini', # Note: Primarily USD
    'kraken', # Note: Uses XBT/various quotes, ccxt handles normalization
    'kucoin', # Note: Primarily USDT/USDC/BTC
    'mexc',   # Note: Primarily USDT/USDC
    'okx',
    'poloniex' # Note: Primarily USDT, some legacy BTC
]

# Define the "like" quote currencies to search for
STABLE_QUOTES = {'USD', 'USDT', 'USDC', 'DAI', 'TUSD', 'USDP', 'FDUSD'}

DEFAULT_SYMBOL = 'FARTCOIN/USD' # Input symbol primarily used to determine the BASE asset
DEFAULT_TIMEFRAME = '1d'
DEFAULT_LIMIT = 30
MAX_RETRIES = 1 # Reduce retries further due to increased call volume
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
    'coinlist': { # Often problematic, limited timeframes/pairs
        'options': {'defaultType': 'spot', 'adjustForTimeDifference': True, 'createMarketBuyOrderRequiresPrice': True, 'fetchOHLCVWarning': False},
        'timeout': 30000,
        'valid_timeframe': '30m'
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

async def _fetch_and_process_symbol(exchange, symbol_to_fetch, timeframe, limit, since_ms, params):
    """Internal helper to fetch and process OHLCV for a single symbol."""
    try:
        ohlcv_data = await exchange.fetch_ohlcv(symbol_to_fetch, timeframe, since=since_ms, limit=limit, params=params)

        if not isinstance(ohlcv_data, list):
            rprint(f"[yellow][{exchange.id.upper()}] Invalid data structure for {symbol_to_fetch}. Expected list, got {type(ohlcv_data)}[/yellow]")
            return None
        if not ohlcv_data:
            rprint(f"[yellow][{exchange.id.upper()}] No OHLCV data returned for {symbol_to_fetch}.[/yellow]")
            return None

        valid_ohlcv = []
        for candle in ohlcv_data:
            if isinstance(candle, list) and len(candle) == 6 and all(c is not None for c in candle[1:5]):
                ts = candle[0]
                if isinstance(ts, str):
                    try:
                        ts = int(pd.to_datetime(ts, errors='coerce').timestamp() * 1000)
                        if pd.isna(ts): continue
                        candle[0] = ts
                    except Exception: continue
                elif not isinstance(ts, (int, float)): continue
                valid_ohlcv.append(candle)

        if not valid_ohlcv:
            rprint(f"[yellow][{exchange.id.upper()}] No valid candles found after filtering for {symbol_to_fetch}.[/yellow]")
            return None

        df = pd.DataFrame(valid_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if df[numeric_cols].isnull().values.any():
            nan_count = df[numeric_cols].isnull().any(axis=1).sum()
            rprint(f"[yellow][{exchange.id.upper()}] Found NaNs after conversion for {symbol_to_fetch}. Dropping {nan_count} rows.[/yellow]")
            df.dropna(subset=numeric_cols, inplace=True)
            if df.empty:
                rprint(f"[yellow][{exchange.id.upper()}] DataFrame empty for {symbol_to_fetch} after dropping NaNs.[/yellow]")
                return None

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # Add symbol column *before* returning
        df['symbol'] = symbol_to_fetch
        rprint(f"[green][{exchange.id.upper()}] Processed {len(df)} candles for {symbol_to_fetch}.[/green]")
        return df

    except (ccxt_async.NetworkError, ccxt_async.ExchangeNotAvailable, ccxt_async.RateLimitExceeded, asyncio.TimeoutError) as e:
        # Let retry logic be handled by the outer function for the whole exchange
        rprint(f"[yellow][{exchange.id.upper()}] Network/RateLimit error fetching {symbol_to_fetch}: {str(e)[:100]}...[/yellow]")
        raise # Re-raise to trigger retry in the outer loop if applicable
    except ccxt_async.ExchangeError as e:
        rprint(f"[red][{exchange.id.upper()}] Exchange error fetching {symbol_to_fetch}: {e}[/red]")
        return None # Don't retry symbol-specific exchange errors usually
    except Exception as e:
        rprint(f"[red][{exchange.id.upper()}] Unexpected error fetching/processing {symbol_to_fetch}: {e}[/red]")
        #rprint(f"[grey]{traceback.format_exc()}[/grey]") # Can be noisy
        return None


async def fetch_single_exchange_ohlcv(exchange_id, base_asset, default_timeframe, limit):
    """
    Asynchronously fetches OHLCV data for all 'like' pairs (BASE/STABLE_QUOTE)
    for a single exchange using ccxt.async_support.

    Args:
        exchange_id (str): The ccxt ID of the exchange.
        base_asset (str): The base asset symbol (e.g., 'SOL').
        default_timeframe (str): Default candle timeframe (e.g., '1h'). Can be overridden by config.
        limit (int): Number of candles to fetch.

    Returns:
        pandas.DataFrame: Concatenated OHLCV data for all successful pairs
                          on this exchange, with 'exchange' and 'symbol' columns,
                          or None on failure.
    """
    exchange_class = getattr(ccxt_async, exchange_id, None)
    if not exchange_class:
        rprint(f"[red]Exchange ID '{exchange_id}' not found.[/red]")
        return None

    exchange_specific_config = EXCHANGE_CONFIGS.get(exchange_id, {})
    config = {
        'enableRateLimit': True, 'verbose': False,
        'headers': {'User-Agent': 'Mozilla/5.0 ...'} # Simplified
    }
    if 'options' in exchange_specific_config:
        config.setdefault('options', {}).update(exchange_specific_config['options'])
    if 'timeout' in exchange_specific_config:
        config['timeout'] = exchange_specific_config['timeout']

    effective_timeframe = exchange_specific_config.get('valid_timeframe', default_timeframe)
    exchange = exchange_class(config)
    rprint(f"[cyan]Processing {exchange.id.upper()} (Default Timeframe: {effective_timeframe})...[/cyan]")

    all_exchange_dfs = []
    for attempt in range(MAX_RETRIES + 1):
        try:
            await exchange.load_markets()

            # --- Identify all target "like" symbols ---
            target_symbols = []
            for quote in STABLE_QUOTES:
                symbol_guess = f"{base_asset}/{quote}"
                # Handle Blofin's :USDT suffix specifically
                if exchange_id == 'blofin' and quote == 'USDT':
                    if f"{symbol_guess}:USDT" in exchange.markets:
                        target_symbols.append(f"{symbol_guess}:USDT")
                elif symbol_guess in exchange.markets:
                    target_symbols.append(symbol_guess)

            if not target_symbols:
                rprint(f"[yellow][{exchange.id.upper()}] No markets found for base '{base_asset}' against quotes {STABLE_QUOTES}.[/yellow]")
                await exchange.close()
                return None

            rprint(f"[{exchange.id.upper()}] Found target symbols: {target_symbols}")

            # --- Timeframe Validation (check against the effective one) ---
            if effective_timeframe not in exchange.timeframes:
                rprint(f"[red][{exchange.id.upper()}] Invalid timeframe: {effective_timeframe}. Available: {list(exchange.timeframes.keys())}[/red]")
                await exchange.close()
                return None

            # --- Calculate 'since' once ---
            try:
                 since_ms = exchange.milliseconds() - exchange.parse_timeframe(effective_timeframe) * 1000 * limit
            except Exception as time_parse_e:
                 rprint(f"[red][{exchange.id.upper()}] Error parsing timeframe '{effective_timeframe}': {time_parse_e}.[/red]")
                 await exchange.close()
                 return None

            # --- Prepare Params (example for Coinlist) ---
            params = {}
            # if exchange_id == 'coinlist': # Coinlist often fails anyway, consider removing it
            #      now_ms = exchange.milliseconds()
            #      params = {'start_time': exchange.iso8601(since_ms), 'end_time': exchange.iso8601(now_ms)}

            # --- Fetch data for all target symbols concurrently ---
            symbol_tasks = [
                _fetch_and_process_symbol(exchange, sym, effective_timeframe, limit, since_ms, params)
                for sym in target_symbols
            ]
            symbol_results = await asyncio.gather(*symbol_tasks, return_exceptions=True)

            # --- Collect successful results for this exchange ---
            successful_symbol_dfs = []
            for i, res in enumerate(symbol_results):
                if isinstance(res, pd.DataFrame):
                    successful_symbol_dfs.append(res)
                elif isinstance(res, Exception):
                     # Logged inside _fetch_and_process_symbol or here if needed
                     rprint(f"[yellow][{exchange.id.upper()}] Error during gather for {target_symbols[i]}: {res}[/yellow]")
                     # Decide if this constitutes a retryable failure for the whole exchange
                     # For now, we just collect successes

            if not successful_symbol_dfs:
                rprint(f"[yellow][{exchange.id.upper()}] No data successfully fetched for any target symbol.[/yellow]")
                # Consider if this is a scenario to retry (e.g., if all failed with NetworkError)
                # For simplicity, we'll currently fail the exchange if no symbols succeed on an attempt
                # Re-raise a caught exception if we want the outer retry loop to catch it
                # if any(isinstance(res, (ccxt_async.NetworkError, asyncio.TimeoutError)) for res in symbol_results):
                #     raise ccxt_async.NetworkError(f"Network errors fetching symbols on {exchange.id}")
                await exchange.close()
                return None # No successful data from this exchange on this attempt


            # --- Combine results for the exchange ---
            exchange_df = pd.concat(successful_symbol_dfs, ignore_index=True)
            exchange_df['exchange'] = exchange.id.upper() # Add exchange ID column

            await exchange.close()
            return exchange_df # Success for this exchange

        except (ccxt_async.NetworkError, ccxt_async.ExchangeNotAvailable, ccxt_async.RateLimitExceeded, asyncio.TimeoutError) as e:
            if attempt < MAX_RETRIES:
                rprint(f"[yellow][{exchange.id.upper()}] Attempt {attempt + 1} failed ({type(e).__name__}). Retrying whole exchange in {RETRY_DELAY}s...[/yellow]")
                await exchange.close() # Close before sleep
                await asyncio.sleep(RETRY_DELAY + attempt)
            else:
                rprint(f"[red][{exchange.id.upper()}] Max retries reached for exchange. Failed due to {type(e).__name__}.[/red]")
                await exchange.close()
                return None # Failed all retries for the exchange
        except ccxt_async.ExchangeError as e:
            rprint(f"[red][{exchange.id.upper()}] Exchange error during setup/market load: {e}[/red]")
            await exchange.close()
            return None # Don't retry these usually
        except Exception as e:
            rprint(f"[red][{exchange.id.upper()}] Unexpected error processing exchange: {e}[/red]")
            rprint(f"[grey]{traceback.format_exc()}[/grey]")
            await exchange.close()
            return None # Unexpected error

    # Should only reach here if all retries fail
    await exchange.close()
    return None


async def aggregate_all_ohlcv(input_symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME, limit=DEFAULT_LIMIT):
    """
    Fetches OHLCV data for all 'like' pairs from configured exchanges
    concurrently and aggregates it.

    Args:
        input_symbol (str): Input symbol like 'SOL/USD' used to find the base asset.
        timeframe (str): Default candle timeframe.
        limit (int): Number of candles.

    Returns:
        pandas.DataFrame: Aggregated OHLCV data.
    """
    # Extract base asset
    base_asset = input_symbol.split('/')[0].upper() if '/' in input_symbol else input_symbol.upper()
    if not base_asset:
        rprint("[red]Could not determine base asset from input symbol.[/red]")
        return pd.DataFrame()

    rprint(f"[bold blue]Determined Base Asset: {base_asset}[/bold blue]")

    tasks = [
        fetch_single_exchange_ohlcv(exchange_id, base_asset, timeframe, limit)
        for exchange_id in EXCHANGES_TO_FETCH
    ]

    rprint(f"\n[bold blue]Starting fetches for {len(tasks)} exchanges...[/bold blue]")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful_exchange_dfs = []
    failed_count = 0
    for i, result in enumerate(results):
        exchange_id = EXCHANGES_TO_FETCH[i]
        if isinstance(result, pd.DataFrame) and not result.empty:
            successful_exchange_dfs.append(result)
        elif isinstance(result, Exception):
            rprint(f"[red]Gather failed for {exchange_id.upper()} with exception: {result}[/red]")
            failed_count += 1
        else: # result was None or an empty DataFrame
            failed_count += 1

    if not successful_exchange_dfs:
        rprint("[bold red]Failed to fetch data from any exchange.[/bold red]")
        return pd.DataFrame()

    all_data = pd.concat(successful_exchange_dfs, ignore_index=True)
    all_data.sort_values(by=['exchange', 'symbol', 'timestamp'], inplace=True)

    rprint(f"\n[bold green]Successfully aggregated data from {len(successful_exchange_dfs)} exchanges. ({failed_count} failures)[/bold green]")
    return all_data

def display_summary_statistics(df, base_asset, default_timeframe):
    """
    Calculates and displays summary statistics for the aggregated OHLCV data.
    Now groups by Exchange and Symbol.

    Args:
        df (pandas.DataFrame): Aggregated OHLCV data with 'exchange' and 'symbol' columns.
        base_asset (str): The base asset used for fetching.
        default_timeframe (str): The default timeframe used.
    """
    if df.empty:
        rprint("[yellow]No data available to display statistics.[/yellow]")
        return

    console = Console()
    title = f"Aggregated OHLCV Summary for {base_asset}/* (Default Timeframe: {default_timeframe})"
    console.print(f"\n[bold underline magenta]{title}[/bold underline magenta]")

    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)

    if df.empty:
        rprint("[yellow]No valid numeric data available after cleaning.[/yellow]")
        return

    # Overall Aggregate Statistics (calculated across *all* fetched data)
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
    unique_symbols = df['symbol'].nunique()

    summary_table = Table(title="Overall Market Statistics (Aggregated Across All Pairs)", show_header=False, box=None, padding=(0,1))
    summary_table.add_column("Metric", style="cyan", no_wrap=True)
    summary_table.add_column("Value", style="white")
    summary_table.add_row("Time Range Start:", str(start_time))
    summary_table.add_row("Time Range End:", str(end_time))
    summary_table.add_row("Total Candles Aggregated:", f"{total_candles:,}")
    summary_table.add_row("Number of Exchanges:", str(unique_exchanges))
    summary_table.add_row("Number of Symbols Fetched:", str(unique_symbols))
    summary_table.add_row("Overall Period Open:", f"{overall_open:.4f} (from {overall_open_row['exchange']}/{overall_open_row['symbol']} at {overall_open_row['timestamp']})")
    summary_table.add_row("Overall Period High:", f"{overall_high:.4f}")
    summary_table.add_row("Overall Period Low:", f"{overall_low:.4f}")
    summary_table.add_row("Overall Period Close:", f"{overall_close:.4f} (from {overall_close_row['exchange']}/{overall_close_row['symbol']} at {overall_close_row['timestamp']})")
    summary_table.add_row("Total Aggregated Volume:", f"{total_volume:,.4f}")
    console.print(summary_table)

    # Per-Exchange-Symbol Summary
    console.print("\n[bold underline magenta]Per-Exchange & Symbol Summary (Latest Data & Totals)[/bold underline magenta]")
    ex_sym_summary_table = Table(title="Statistics per Exchange/Symbol", show_header=True, header_style="bold blue")
    ex_sym_summary_table.add_column("Exchange", style="cyan")
    ex_sym_summary_table.add_column("Symbol", style="yellow")
    ex_sym_summary_table.add_column("Latest Timestamp", style="dim")
    ex_sym_summary_table.add_column("Latest Close", justify="right")
    ex_sym_summary_table.add_column("Total Volume", justify="right")
    ex_sym_summary_table.add_column("Candle Count", justify="right")
    ex_sym_summary_table.add_column("Avg Volume / Candle", justify="right")

    # Group by both exchange and symbol
    grouped = df.groupby(['exchange', 'symbol'])

    # Sort groups for consistent display (optional, but nice)
    sorted_groups = sorted(grouped.groups.keys())

    for ex, sym in sorted_groups:
        group = grouped.get_group((ex, sym))
        if group.empty: continue

        latest_candle = group.loc[group['timestamp'].idxmax()]
        exchange_total_volume = group['volume'].sum()
        candle_count = len(group)
        avg_vol_per_candle = exchange_total_volume / candle_count if candle_count > 0 else 0

        ex_sym_summary_table.add_row(
            ex,
            sym,
            str(latest_candle['timestamp']),
            f"{latest_candle['close']:.4f}" if pd.notna(latest_candle['close']) else "[red]N/A[/red]",
            f"{exchange_total_volume:,.4f}",
            f"{candle_count:,}",
            f"{avg_vol_per_candle:.4f}"
        )

    console.print(ex_sym_summary_table)


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fetch and aggregate OHLCV data from multiple exchanges for all stable pairs of a base asset.")
    parser.add_argument('--symbol', type=str, default=DEFAULT_SYMBOL, help=f"Input symbol (e.g., 'SOL/USD') to identify the base asset (default: {DEFAULT_SYMBOL})")
    parser.add_argument('--timeframe', type=str, default=DEFAULT_TIMEFRAME, help=f"Default candle timeframe (default: {DEFAULT_TIMEFRAME})")
    parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT, help=f"Number of candles (default: {DEFAULT_LIMIT})")
    args = parser.parse_args()

    input_symbol = args.symbol
    timeframe = args.timeframe # Default, might be overridden per exchange
    limit = args.limit

    rprint(f"[bold yellow]Starting aggregation for base asset derived from {input_symbol} (Default Timeframe: {timeframe}, Limit: {limit})...[/bold yellow]")
    start_run_time = time.time()

    aggregated_data = await aggregate_all_ohlcv(input_symbol, timeframe, limit)

    end_run_time = time.time()
    rprint(f"\n[italic]Data fetching and aggregation took {end_run_time - start_run_time:.2f} seconds.[/italic]")

    if not aggregated_data.empty:
        base_asset = input_symbol.split('/')[0].upper()
        display_summary_statistics(aggregated_data, base_asset, timeframe)
        # Optionally save
        # filename = f"aggregated_{base_asset}_stablepairs_{timeframe}.csv"
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