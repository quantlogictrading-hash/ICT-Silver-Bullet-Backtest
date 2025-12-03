#SilverBullet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
FILE_PATH = 'NQ2016_2022.csv'  
# Time Adjustment:
# 10:00 AM NY = 17:00 in the data.
SESSION_START_HOUR = 17
SESSION_END_HOUR = 18
RISK_RATIOS = [1, 2, 3, 4, 5]


# ==========================================
# 1. DATA LOADING AND CLEANING
# ==========================================
def load_data(filepath):
    """
    Reads the CSV file with specific formatting and cleans it.
    """
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                # Remove quotes at start/end if they exist
                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1]
                data.append(line.split('\t'))

        columns = data[0]
        df = pd.DataFrame(data[1:], columns=columns)

        # Type conversion
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y.%m.%d %H:%M:%S')
        cols_numeric = ['Open', 'High', 'Low', 'Close', 'Volume', 'TickVolume']
        for c in cols_numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c])

        df = df.sort_values('DateTime').reset_index(drop=True)
        df['Date'] = df['DateTime'].dt.date
        df['Hour'] = df['DateTime'].dt.hour
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# ==========================================
# 2. FVG IDENTIFICATION (Fair Value Gaps)
# ==========================================
def identify_fvgs(df):
    """
    Identifies Bullish and Bearish FVGs by vectorizing the DataFrame.
    Rules:
    - Bearish: Low(i-2) > High(i). Entry: High(i). SL: Max(High(i-1), High(i-2))
    - Bullish: High(i-2) < Low(i). Entry: Low(i). SL: Min(Low(i-1), Low(i-2))
    """
    # Shifts to compare previous candles
    df['High_shifted_1'] = df['High'].shift(1)
    df['High_shifted_2'] = df['High'].shift(2)
    df['Low_shifted_1'] = df['Low'].shift(1)
    df['Low_shifted_2'] = df['Low'].shift(2)

    # --- BEARISH FVG ---
    df['is_bear_fvg'] = df['Low_shifted_2'] > df['High']
    df['bear_entry'] = df['High']
    df['bear_sl'] = df[['High_shifted_1', 'High_shifted_2']].max(axis=1)

    # --- BULLISH FVG ---
    df['is_bull_fvg'] = df['High_shifted_2'] < df['Low']
    df['bull_entry'] = df['Low']
    df['bull_sl'] = df[['Low_shifted_1', 'Low_shifted_2']].min(axis=1)

    # Filter only rows that are FVGs for faster processing
    fvg_rows = df[(df['is_bear_fvg']) | (df['is_bull_fvg'])].copy()
    fvg_rows['fvg_type'] = np.where(fvg_rows['is_bear_fvg'], 'bear', 'bull')
    fvg_rows['entry'] = np.where(fvg_rows['is_bear_fvg'], fvg_rows['bear_entry'], fvg_rows['bull_entry'])
    fvg_rows['sl'] = np.where(fvg_rows['is_bear_fvg'], fvg_rows['bear_sl'], fvg_rows['bull_sl'])

    # Group by day in a dictionary for fast access during backtest
    from collections import defaultdict
    fvgs_by_day = defaultdict(list)
    for idx, row in fvg_rows.iterrows():
        day_str = row['DateTime'].strftime('%Y-%m-%d')
        fvgs_by_day[day_str].append({
            'creation_index': idx,
            'fvg_type': row['fvg_type'],
            'entry': row['entry'],
            'sl': row['sl'],
            'creation_time': row['DateTime']
        })

    return fvgs_by_day


# ==========================================
# 3. BACKTEST ENGINE (Silver Bullet Logic)
# ==========================================
def run_backtest(df, fvgs_by_day):
    unique_days = df['Date'].unique()
    unique_days.sort()

    results_log = []
    print(f"Starting simulation on {len(unique_days)} days...")

    for day in unique_days:
        day_str = day.strftime('%Y-%m-%d')

        # Get data only for this day
        day_mask = (df['Date'] == day)
        day_data = df[day_mask]

        # Filter only the session hour (10 AM - 11 AM NY)
        session_data = day_data[day_data['Hour'] == SESSION_START_HOUR]
        if session_data.empty: continue

        session_start_idx = session_data.index[0]

        # Get FVGs for the day
        day_fvgs = fvgs_by_day.get(day_str, [])
        if not day_fvgs: continue

        # List of active FVGs. Filter those created BEFORE or DURING the session.
        # (For those created before, simple validation to check they weren't invalidated)
        active_fvgs = []
        for f in day_fvgs:
            # If created before the session
            if f['creation_index'] < session_start_idx:
                # Check if price hit SL or Entry between creation and session start
                check_slice = day_data.loc[f['creation_index'] + 1: session_start_idx - 1]

                is_valid = True
                if not check_slice.empty:
                    if f['fvg_type'] == 'bull':
                        if (check_slice['Low'] <= f['sl']).any() or (check_slice['Low'] <= f['entry']).any():
                            is_valid = False
                    else:  # Bear
                        if (check_slice['High'] >= f['sl']).any() or (check_slice['High'] >= f['entry']).any():
                            is_valid = False

                if is_valid: active_fvgs.append(f)
            else:
                # If created during the session, add it (validated candle by candle)
                active_fvgs.append(f)

        trade_taken = None

        # CANDLE BY CANDLE loop within the session (M5)
        for idx, row in session_data.iterrows():
            # We can only use FVGs created before this current candle
            available = [f for f in active_fvgs if f['creation_index'] < idx]
            if not available: continue

            triggered = []
            for f in available:
                # Trigger Logic (Price Touch)
                if f['fvg_type'] == 'bull':
                    if row['Low'] <= f['entry']: triggered.append(f)
                else:  # Bear
                    if row['High'] >= f['entry']: triggered.append(f)

            if triggered:
                # RULE: If multiple, pick the one closest to Open price
                triggered.sort(key=lambda x: abs(row['Open'] - x['entry']))
                chosen = triggered[0]

                trade_taken = {
                    'EntryPrice': chosen['entry'],
                    'SL': chosen['sl'],
                    'Type': chosen['fvg_type'],
                    'Index': idx,
                    'Time': row['DateTime']
                }
                break  # We only take 1 trade per day (the first one)

        # If trade taken, calculate outcome
        if trade_taken:
            outcome_slice = df.loc[trade_taken['Index']:]  # Data from entry to future
            risk = abs(trade_taken['EntryPrice'] - trade_taken['SL'])
            if risk == 0: risk = 0.1  # Div/0 protection

            row_result = {
                'Date': day_str,
                'DayOfWeek': trade_taken['Time'].day_name()
            }

            # Check TP for each ratio (1:1 to 1:5)
            for r in RISK_RATIOS:
                ratio_col = f'1:{r}'
                tp_dist = risk * r

                if trade_taken['Type'] == 'bear':
                    tp_price = trade_taken['EntryPrice'] - tp_dist
                    # Indices where SL (High >= SL) or TP (Low <= TP) is hit
                    sl_hits = np.where(outcome_slice['High'] >= trade_taken['SL'])[0]
                    tp_hits = np.where(outcome_slice['Low'] <= tp_price)[0]
                else:  # Bull
                    tp_price = trade_taken['EntryPrice'] + tp_dist
                    # Indices where SL (Low <= SL) or TP (High >= TP) is hit
                    sl_hits = np.where(outcome_slice['Low'] <= trade_taken['SL'])[0]
                    tp_hits = np.where(outcome_slice['High'] >= tp_price)[0]

                first_sl = sl_hits[0] if len(sl_hits) > 0 else 999999
                first_tp = tp_hits[0] if len(tp_hits) > 0 else 999999

                # Result: 1 if TP hit before SL, 0 otherwise
                row_result[ratio_col] = 1 if first_tp < first_sl else 0

            results_log.append(row_result)

    return pd.DataFrame(results_log)


# ==========================================
# 4. RESULTS ANALYSIS
# ==========================================
def analyze_results(results_df):
    if results_df.empty:
        print("No trades generated.")
        return

    # A. General Winrate
    print("\n" + "=" * 40)
    print(" GENERAL WINRATE")
    print("=" * 40)
    general_wr = results_df[[f'1:{r}' for r in RISK_RATIOS]].mean() * 100
    print(general_wr.round(2).to_string())

    # B. Winrate by Day of Week
    print("\n" + "=" * 40)
    print(" WINRATE BY DAY OF WEEK")
    print("=" * 40)
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    results_df['DayOfWeek'] = pd.Categorical(results_df['DayOfWeek'], categories=days_order, ordered=True)
    dow_summary = results_df.groupby('DayOfWeek')[[f'1:{r}' for r in RISK_RATIOS]].mean() * 100
    print(dow_summary.round(2))

    # C. Streaks (Consecutive)
    print("\n" + "=" * 40)
    print(" MAX STREAKS (CONSECUTIVE)")
    print("=" * 40)
    stats_streaks = []
    for r in RISK_RATIOS:
        col = f'1:{r}'
        series = results_df[col]

        # Group consecutive equal values
        groups = series.ne(series.shift()).cumsum()
        counts = series.groupby(groups).transform('size')

        max_win = counts[series == 1].max() if (series == 1).any() else 0
        max_loss = counts[series == 0].max() if (series == 0).any() else 0
        stats_streaks.append([f'1:{r}', max_win, max_loss])

    print(pd.DataFrame(stats_streaks, columns=['Ratio', 'Max Win Streak', 'Max Loss Streak']))

    # D. Equity Curve Simulation
    plot_equity(general_wr['1:1'] / 100, general_wr['1:2'] / 100)


def plot_equity(wr1, wr2):
    """Generates a Monte Carlo simulation of 1500 trades"""
    n_trades = 1500
    # Generate random sequence based on winrate
    outcomes1 = np.random.choice([1, -1], size=n_trades, p=[wr1, 1 - wr1])
    outcomes2 = np.random.choice([2, -1], size=n_trades, p=[wr2, 1 - wr2])

    equity1 = np.cumsum(outcomes1)
    equity2 = np.cumsum(outcomes2)

    plt.figure(figsize=(12, 6))
    plt.plot(equity1, label=f'Ratio 1:1 (WR {wr1 * 100:.1f}%)', color='#1f77b4', linewidth=1.5)
    plt.plot(equity2, label=f'Ratio 1:2 (WR {wr2 * 100:.1f}%)', color='#ff7f0e', alpha=0.8, linewidth=1.5)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title('Equity Curve Simulation (1500 Random Trades)', fontsize=14)
    plt.xlabel('Number of Trades')
    plt.ylabel('Accumulated Return (R)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Loading data...")
    df = load_data(FILE_PATH)

    if not df.empty:
        print("Identifying FVGs...")
        fvgs = identify_fvgs(df)

        results = run_backtest(df, fvgs)
        analyze_results(results)
    else:
        print("Could not load DataFrame.")