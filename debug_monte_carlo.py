#!/usr/bin/env python3
"""
Debug Monte Carlo Analysis to identify why no trades are occurring
"""
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf

def debug_single_simulation():
    """Debug a single simulation to understand trade logic"""
    
    # Load real BTC data
    start_date = datetime.strptime("2020-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2024-12-31", "%Y-%m-%d")
    
    ticker = yf.Ticker("BTC-USD")
    btc_data = ticker.history(
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        interval="1d"
    )
    
    # Use actual historical prices for debugging
    synthetic_prices = btc_data['Close'].values
    
    print(f"📊 Debugging with {len(synthetic_prices)} days of data")
    print(f"   Price range: ${synthetic_prices.min():.0f} - ${synthetic_prices.max():.0f}")
    
    # Strategy parameters
    strategy_params = {
        'min_drawdown': -15,
        'max_fear_greed': 40,
        'min_days_since_ath': 5,
        'fibonacci_levels': {
            '23.6%': {'ratio': 0.236, 'base_multiplier': 1.8, 'entry_window': (-10.0, -0.5)},
            '38.2%': {'ratio': 0.382, 'base_multiplier': 2.8, 'entry_window': (-15.0, -0.5)},
            '50.0%': {'ratio': 0.500, 'base_multiplier': 4.5, 'entry_window': (-18.0, -0.5)},
            '61.8%': {'ratio': 0.618, 'base_multiplier': 7.0, 'entry_window': (-22.0, -0.5)},
            '78.6%': {'ratio': 0.786, 'base_multiplier': 11.0, 'entry_window': (-25.0, -0.5)}
        },
        'base_leverage': 2.8,
        'max_leverage': 16.0,
        'cooldown_hours': 36,
        'min_range_pct': 0.04,
        'score_threshold': 4.0,
        'max_position_pct': 0.18,
        'min_capital': 75
    }
    
    # Calculate indicators
    price_series = pd.Series(synthetic_prices)
    rolling_highs = price_series.rolling(60, min_periods=1).max()
    drawdowns = (price_series - rolling_highs) / rolling_highs * 100
    
    # Fear & Greed simulation
    fear_greed = np.clip(50 + drawdowns * 0.8 + np.random.normal(0, 5, len(drawdowns)), 0, 100)
    
    # Days since ATH
    days_since_ath = np.zeros(len(synthetic_prices))
    for i in range(1, len(synthetic_prices)):
        if synthetic_prices[i] >= rolling_highs.iloc[i]:
            days_since_ath[i] = 0
        else:
            days_since_ath[i] = days_since_ath[i-1] + 1
    
    # Debug counters
    total_days_checked = 0
    cooldown_blocks = 0
    drawdown_fails = 0
    fear_greed_fails = 0
    days_ath_fails = 0
    range_fails = 0
    score_fails = 0
    trades_made = 0
    
    total_capital = 100000
    capital_deployed = 0
    last_trade_time = None
    
    print(f"\n🔍 DEBUGGING ENTRY CONDITIONS:")
    print(f"   Min drawdown: {strategy_params['min_drawdown']}%")
    print(f"   Max fear & greed: {strategy_params['max_fear_greed']}")
    print(f"   Min days since ATH: {strategy_params['min_days_since_ath']}")
    print(f"   Min range %: {strategy_params['min_range_pct']*100}%")
    print(f"   Score threshold: {strategy_params['score_threshold']}")
    
    # Sample some days for detailed analysis
    sample_days = [100, 500, 1000, 1500] if len(synthetic_prices) > 1500 else [len(synthetic_prices)//4, len(synthetic_prices)//2, 3*len(synthetic_prices)//4]
    
    for day, (price, drawdown, fg, days_ath) in enumerate(zip(
        synthetic_prices, drawdowns, fear_greed, days_since_ath
    )):
        total_days_checked += 1
        
        # Sample day analysis
        if day in sample_days:
            print(f"\n📅 Day {day} Analysis:")
            print(f"   Price: ${price:.0f}")
            print(f"   Drawdown: {drawdown:.1f}%")
            print(f"   Fear & Greed: {fg:.0f}")
            print(f"   Days since ATH: {days_ath:.0f}")
        
        # Check cooldown
        if last_trade_time is not None:
            cooldown_days = strategy_params['cooldown_hours'] / 24
            if day - last_trade_time < cooldown_days:
                cooldown_blocks += 1
                if day in sample_days:
                    print(f"   ❌ Cooldown active ({day - last_trade_time:.1f} < {cooldown_days:.1f} days)")
                continue
        
        # Check entry criteria
        if drawdown > strategy_params['min_drawdown']:
            drawdown_fails += 1
            if day in sample_days:
                print(f"   ❌ Drawdown too small: {drawdown:.1f}% > {strategy_params['min_drawdown']}%")
            continue
            
        if fg > strategy_params['max_fear_greed']:
            fear_greed_fails += 1
            if day in sample_days:
                print(f"   ❌ Fear & Greed too high: {fg:.0f} > {strategy_params['max_fear_greed']}")
            continue
            
        if days_ath < strategy_params['min_days_since_ath']:
            days_ath_fails += 1
            if day in sample_days:
                print(f"   ❌ Not enough days since ATH: {days_ath:.0f} < {strategy_params['min_days_since_ath']}")
            continue
        
        if day in sample_days:
            print(f"   ✅ Basic criteria passed!")
        
        # Fibonacci analysis
        lookback_start = max(0, day - 90)
        recent_high = price_series.iloc[lookback_start:day+1].max()
        recent_low = price_series.iloc[lookback_start:day+1].min()
        price_range = recent_high - recent_low
        
        if day in sample_days:
            print(f"   Range analysis: ${recent_low:.0f} - ${recent_high:.0f} (${price_range:.0f})")
            print(f"   Range %: {price_range/recent_high*100:.1f}% (need {strategy_params['min_range_pct']*100:.1f}%)")
        
        if price_range <= recent_high * strategy_params['min_range_pct']:
            range_fails += 1
            if day in sample_days:
                print(f"   ❌ Range too small")
            continue
        
        # Check Fibonacci levels
        best_score = 0
        best_level = None
        
        for level_name, level_config in strategy_params['fibonacci_levels'].items():
            fib_price = recent_high - (price_range * level_config['ratio'])
            distance_pct = (price - fib_price) / fib_price * 100
            
            entry_window = level_config['entry_window']
            if entry_window[0] <= distance_pct <= entry_window[1]:
                leverage = min(
                    strategy_params['base_leverage'] + abs(drawdown) * 0.25,
                    strategy_params['max_leverage']
                )
                
                score = level_config['base_multiplier'] * leverage * abs(distance_pct)
                
                if day in sample_days:
                    print(f"   Fib {level_name}: price=${fib_price:.0f}, distance={distance_pct:.1f}%, score={score:.1f}")
                
                if score > best_score:
                    best_score = score
                    best_level = level_config
        
        if day in sample_days:
            print(f"   Best score: {best_score:.1f} (need {strategy_params['score_threshold']:.1f})")
        
        if best_score <= strategy_params['score_threshold']:
            score_fails += 1
            if day in sample_days:
                print(f"   ❌ Score too low")
            continue
        
        # Execute trade
        remaining_capital = total_capital - capital_deployed
        if remaining_capital > strategy_params['min_capital']:
            trades_made += 1
            last_trade_time = day
            capital_deployed += min(remaining_capital * 0.18, 15000)
            
            if day in sample_days:
                print(f"   ✅ TRADE EXECUTED! #{trades_made}")
    
    print(f"\n📊 DEBUGGING RESULTS:")
    print(f"   Total days checked: {total_days_checked}")
    print(f"   Cooldown blocks: {cooldown_blocks}")
    print(f"   Drawdown fails: {drawdown_fails}")
    print(f"   Fear & Greed fails: {fear_greed_fails}")
    print(f"   Days since ATH fails: {days_ath_fails}")
    print(f"   Range fails: {range_fails}")
    print(f"   Score fails: {score_fails}")
    print(f"   TRADES MADE: {trades_made}")
    
    years = (end_date - start_date).days / 365.25
    print(f"   Trades per year: {trades_made / years:.1f}")

if __name__ == "__main__":
    debug_single_simulation()