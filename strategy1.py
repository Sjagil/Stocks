import datetime as dt
import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.stats import norm
import yfinance as yf

pd.set_option('display.max_columns', None)

def get_data(symbol, start, end):
    """
    Haalt historische OHLCV data op via yfinance.
    Ondersteunt indices/aandelen (bijv. 'SPY') en crypto via yfinance tickers (bijv. 'BTC-USD').
    """
    print(f"Historische data ophalen voor {symbol} van {start} tot {end}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval="1d")
    
    if df.empty:
        raise ValueError(f"Geen data gevonden voor symbool {symbol} in deze periode.")
    
    # Reset index om 'Date' als kolom te krijgen en kolommen naar lowercase te forceren
    df = df.reset_index()
    df.columns = [col.lower() for col in df.columns]
    return df

def calculate_technical_setup(df):
    """
    Berekent alle technische indicatoren voor de setup op de dataframe (df).
    df moet de kolommen bevatten: 'open', 'high', 'low', 'close', 'volume'
    """
    # Zorg dat de kolommen kleine letters zijn voor pandas_ta
    df.columns = [col.lower() for col in df.columns]

    # 1. RSI(2)
    df["rsi_2"] = ta.rsi(df["close"], length=2)

    # 2. ADX(5)
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=5)
    df["adx_5"] = adx_df["ADX_5"]

    # 3. Bollinger Bands (20, 2)
    bb_df = ta.bbands(df["close"], length=20, std=2)
    df["bb_lower"] = bb_df["BBL_20_2.0"]
    df["bb_upper"] = bb_df["BBU_20_2.0"]

    # 4. SMA 200 (Grote trend filter)
    df["sma_200"] = ta.sma(df["close"], length=200)

    # 5. ATR (Voor de Stop-Loss afstand)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    return df

def calculate_gex(ticker_symbol):
    """Berekent de totale Gamma Exposure (GEX) per strike prijs."""
    ticker = yf.Ticker(ticker_symbol)

    # Pak de huidige koers van het onderliggende aandeel
    history = ticker.history(period="1d")
    if history.empty:
        return {"net_gex": 0, "put_wall": None, "call_wall": None}
        
    spot_price = history["Close"].iloc[-1]

    # Pak de eerste beschikbare expiratiedatum
    try:
        options = ticker.options
        if not options:
            return {"net_gex": 0, "put_wall": None, "call_wall": None}
        exp_date = options[0]
    except (IndexError, AttributeError):
        return {"net_gex": 0, "put_wall": None, "call_wall": None}

    opt_chain = ticker.option_chain(exp_date)
    calls = opt_chain.calls.copy()
    puts = opt_chain.puts.copy()

    # Schatten van de Grieken (Gamma) via Black-Scholes (versimpeld voor Open Interest)
    T = 1 / 365    # Tijd tot expiratie (1 dag = 1/365, aanname voor daily opties)
    r = 0.045      # Risicovrije rentevoet (~4.5%)
    sigma = 0.20   # Impliciete volatiliteit aanname (20%)

    def calculate_gamma(row, is_call=True):
        K = row["strike"]
        if pd.isna(row["openInterest"]) or row["openInterest"] == 0:
            return 0
        
        # Voorkom zero-division errors in logaritmes
        if spot_price <= 0 or K <= 0:
            return 0
            
        d1 = (np.log(spot_price / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (spot_price * sigma * np.sqrt(T))

        # GEX Formule: Open Interest * Gamma * Spot^2 * Contract Size (100)
        gex = row["openInterest"] * gamma * (spot_price**2) * 100
        return gex if is_call else -gex

    calls["GEX"] = calls.apply(calculate_gamma, axis=1, is_call=True)
    puts["GEX"] = puts.apply(calculate_gamma, axis=1, is_call=False)

    # Combineer en groepeer per Strike om de Walls te vinden
    total_gex = pd.concat([calls[["strike", "GEX"]], puts[["strike", "GEX"]]])
    gex_profile = total_gex.groupby("strike").sum().reset_index()

    if gex_profile.empty:
        return {"net_gex": 0, "put_wall": None, "call_wall": None}

    net_gex = gex_profile["GEX"].sum()
    put_wall = gex_profile.loc[gex_profile["GEX"].idxmin()]["strike"]
    call_wall = gex_profile.loc[gex_profile["GEX"].idxmax()]["strike"]

    return {
        "net_gex": net_gex,   # Positief = Mean Reversion, Negatief = Squeeze
        "put_wall": put_wall, # Zwaarste support
        "call_wall": call_wall # Zwaarste weerstand
    }

def check_orderflow_trigger(footprint_candle, min_imbalance_ratio=3.0):
    """Controleert een footprint candle op agressieve koop- of verkoopimbalances."""
    buy_imbalances = 0
    sell_imbalances = 0

    # Sorteer de prijzen om diagonaal te vergelijken (Bid vs Ask van 1 tick hoger)
    sorted_prices = sorted(footprint_candle.keys())

    for i in range(len(sorted_prices) - 1):
        current_price = sorted_prices[i]
        next_price = sorted_prices[i + 1]

        bid_volume = footprint_candle[next_price]["bid_vol"]  # Verkopers slaan toe op bid
        ask_volume = footprint_candle[current_price]["ask_vol"] # Kopers slaan toe op ask

        # Check voor Koop Imbalance
        if bid_volume > 0 and (ask_volume / bid_volume) >= min_imbalance_ratio:
            buy_imbalances += 1

        # Check voor Verkoop Imbalance
        if ask_volume > 0 and (bid_volume / ask_volume) >= min_imbalance_ratio:
            sell_imbalances += 1

    # Bepaal het signaal op basis van gestapelde imbalances (minimaal 3 boven elkaar)
    if buy_imbalances >= 3:
        return "BUY_TRIGGER"
    elif sell_imbalances >= 3:
        return "SELL_TRIGGER"

    return "NO_TRIGGER"

# Mock functies om de live loop gesloten en werkend te krijgen zonder actieve API-sleutels:
def simulated_live_footprint():
    """Simuleert een Footprint Candle met een actieve Buy Imbalance op 3 niveaus."""
    return {
        3950.00: {'bid_vol': 10, 'ask_vol': 45},  # Ratio 4.5 (>3)
        3951.00: {'bid_vol': 5,  'ask_vol': 25},  # Ratio 5.0 (>3)
        3952.00: {'bid_vol': 12, 'ask_vol': 50},  # Ratio 4.1 (>3)
        3953.00: {'bid_vol': 20, 'ask_vol': 15}
    }

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
def main():
    raise RuntimeError(
        "LEGACY_INSPIRATION_ONLY: options, tick/orderflow and standalone "
        "execution are forbidden; use main.py research commands"
    )
    # Instellingen voor de test-run
    target_ticker = "SPY"
    start_date = (dt.datetime.now() - dt.timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = dt.datetime.now().strftime('%Y-%m-%d')

    # 1. Macro Check (Eenmalig per dag/sessie)
    print("Stap 1: Berekenen Gamma Exposure (GEX) klimaat...")
    gex_data = calculate_gex(target_ticker)
    print(f"GEX Resultaten voor {target_ticker}:")
    print(f" -> Net GEX: {gex_data['net_gex']:,.2f}")
    print(f" -> Put Wall (Support): {gex_data['put_wall']}")
    print(f" -> Call Wall (Weerstand): {gex_data['call_wall']}\n")

    # 2. Historische/Live Data ophalen en indicatoren berekenen
    print("Stap 2: Technische markt-setup controleren...")
    try:
        df = get_data(target_ticker, start=start_date, end=end_date)
        df_with_indicators = calculate_technical_setup(df)
        
        # Pak de meest recente gesloten candle
        last_row = df_with_indicators.dropna().iloc[-1]
        print(f"Laatste data info - Close: {last_row['close']:.2f} | SMA200: {last_row['sma_200']:.2f}")
        print(f"RSI(2): {last_row['rsi_2']:.2f} | ADX(5): {last_row['adx_5']:.2f} | BB Lower: {last_row['bb_lower']:.2f}\n")
    except Exception as e:
        print(f"Fout bij data verwerking: {e}")
        return

    # 3. Controleer de Retail Setup condities (Positive GEX Mean-Reversion)
    print("Stap 3: Conditie-vergelijking starten...")
    
    # Handmatige overschrijving voor demonstratie-doeleinden indien je markt niet live oversold is
    market_setup_ready = (
        gex_data["net_gex"] > 0
        and last_row["close"] > last_row["sma_200"]
        and last_row["rsi_2"] < 10
        and last_row["close"] <= last_row["bb_lower"]
    )

    if market_setup_ready:
        print("[GROEN LICHT] Technische setup voldoet aan de eisen. Wachten op Orderflow trigger...")
        
        # 4. Schakel over naar de live Footprint/Tick data stream
        print("Stap 4: Orderflow Footprint analyseren op Imbalances...")
        live_footprint = simulated_live_footprint() # Vervang dit live met je websocket data feed
        trigger = check_orderflow_trigger(live_footprint)

        if trigger == "BUY_TRIGGER":
            stop_loss_level = last_row['close'] - (1.5 * last_row['atr'])
            print(f"\n[!!!] ORDERFLOW BEVESTIGD! {trigger} GEDECTEERD.")
            print(f"-> Actie: Verzend BUY order op market of huidige prijs ({last_row['close']:.2f})")
            print(f"-> Risicobeheer: Stop-loss strikt instellen op: {stop_loss_level:.2f} (1.5x ATR)")
        else:
            print("[STANDBY] Geen orderflow imbalance gevonden. Order gecanceld.")
    else:
        print("[GEEN SETUP] Markt is momenteel niet oververhit of bevindt zich in het verkeerde GEX-klimaat.")

if __name__ == "__main__":
    main()
