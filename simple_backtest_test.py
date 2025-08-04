#!/usr/bin/env python3
"""
🌸 Simple Nanpin Backtest Test
Quick validation of the Enhanced Nanpin strategy performance
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import aiohttp
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleNanpinBacktest:
    """Simple backtest for Enhanced Nanpin strategy"""
    
    def __init__(self, start_date="2020-01-01", end_date="2024-12-31"):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.base_amount = 100.0  # $100 base position size
        
        # Fibonacci levels and multipliers
        self.fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.fib_names = ['23.6%', '38.2%', '50.0%', '61.8%', '78.6%']
        self.fib_multipliers = {name: mult for name, mult in zip(self.fib_names, [1, 2, 3, 5, 8])}
        
        # Performance tracking
        self.trades = []
        self.portfolio_history = pd.DataFrame()
        self.btc_data = pd.DataFrame()
        
    async def run_simple_backtest(self):
        """Run a simplified version of the Enhanced Nanpin backtest"""
        try:
            logger.info("🚀 Starting Simple Nanpin Backtest")
            
            # Step 1: Load BTC data
            await self._load_btc_data()
            
            # Step 2: Calculate Fibonacci levels
            self._calculate_simple_fibonacci_levels()
            
            # Step 3: Simulate strategy
            self._simulate_simple_strategy()
            
            # Step 4: Calculate performance
            results = self._calculate_simple_performance()
            
            # Step 5: Display results
            self._display_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Simple backtest failed: {e}")
            raise
    
    async def _load_btc_data(self):
        """Load BTC price data from Yahoo Finance"""
        try:
            logger.info("📊 Loading BTC data...")
            
            # Add buffer for calculations
            buffer_start = self.start_date - timedelta(days=200)
            
            btc = yf.Ticker("BTC-USD")
            data = btc.history(
                start=buffer_start.strftime("%Y-%m-%d"),
                end=self.end_date.strftime("%Y-%m-%d"),
                interval="1d"
            )
            
            if data.empty:
                raise Exception("No BTC data retrieved")
            
            # Clean and process data
            self.btc_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            self.btc_data.index = pd.to_datetime(self.btc_data.index.date)  # Remove timezone
            self.btc_data = self.btc_data.dropna()
            
            logger.info(f"✅ Loaded {len(self.btc_data)} days of BTC data")
            logger.info(f"   Price range: ${self.btc_data['Low'].min():,.0f} - ${self.btc_data['High'].max():,.0f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load BTC data: {e}")
            raise
    
    def _calculate_simple_fibonacci_levels(self):
        """Calculate Fibonacci retracement levels for each day"""
        try:
            logger.info("📐 Calculating Fibonacci levels...")
            
            # Filter data to backtest period
            backtest_data = self.btc_data[
                (self.btc_data.index >= pd.Timestamp(self.start_date)) &
                (self.btc_data.index <= pd.Timestamp(self.end_date))
            ].copy()
            
            fibonacci_levels = []
            
            for date in backtest_data.index:
                # Get 90-day lookback period
                lookback_start = date - pd.Timedelta(days=90)
                period_data = self.btc_data[
                    (self.btc_data.index >= lookback_start) & 
                    (self.btc_data.index <= date)
                ]
                
                if len(period_data) < 30:
                    fibonacci_levels.append({})
                    continue
                
                # Calculate swing high and low
                swing_high = period_data['High'].max()
                swing_low = period_data['Low'].min()
                price_range = swing_high - swing_low
                
                # Calculate Fibonacci levels
                levels = {}
                for i, (level_name, ratio) in enumerate(zip(self.fib_names, self.fib_levels)):
                    fib_price = swing_high - (price_range * ratio)
                    levels[level_name] = {
                        'price': fib_price,
                        'swing_high': swing_high,
                        'swing_low': swing_low,
                        'multiplier': self.fib_multipliers[level_name]
                    }
                
                fibonacci_levels.append(levels)
            
            # Add to backtest data
            backtest_data['Fibonacci_Levels'] = fibonacci_levels
            
            # Add simple macro indicators
            backtest_data['VIX'] = np.random.normal(20, 5, len(backtest_data))  # Simplified VIX
            backtest_data['Fear_Greed'] = np.clip(100 - backtest_data['VIX'] * 2, 0, 100)
            backtest_data['Macro_Scaling'] = np.where(backtest_data['Fear_Greed'] < 30, 2.0, 1.0)  # 2x scaling during fear
            
            self.backtest_data = backtest_data
            logger.info(f"✅ Fibonacci levels calculated for {len(self.backtest_data)} days")
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate Fibonacci levels: {e}")
            raise
    
    def _simulate_simple_strategy(self):
        """Simulate the Enhanced Nanpin strategy"""
        try:
            logger.info("💰 Simulating Enhanced Nanpin strategy...")
            
            total_btc = 0.0
            total_invested = 0.0
            trade_count = 0
            last_trade_date = None
            cooldown_days = 3  # 3 days between trades
            
            portfolio_values = []
            self.trades = []
            
            for date, row in self.backtest_data.iterrows():
                current_price = row['Close']
                fibonacci_levels = row['Fibonacci_Levels']
                macro_scaling = row['Macro_Scaling']
                
                # Calculate portfolio value
                portfolio_value = total_btc * current_price
                portfolio_values.append({
                    'date': date,
                    'btc': total_btc,
                    'value': portfolio_value,
                    'invested': total_invested,
                    'price': current_price
                })
                
                # Skip if no Fibonacci levels or too soon after last trade
                if (not fibonacci_levels or 
                    (last_trade_date and (date - last_trade_date).days < cooldown_days)):
                    continue
                
                # Check for trading opportunities
                best_opportunity = None
                best_score = 0
                
                for level_name, level_data in fibonacci_levels.items():
                    target_price = level_data['price']
                    multiplier = level_data['multiplier']
                    
                    # Calculate distance from target
                    distance_pct = (current_price - target_price) / target_price * 100
                    
                    # Entry condition: price is 1% to 8% below Fibonacci level
                    if -8.0 <= distance_pct <= -1.0:
                        # Calculate opportunity score
                        opportunity_score = multiplier * macro_scaling * abs(distance_pct)
                        
                        if opportunity_score > best_score:
                            best_score = opportunity_score
                            best_opportunity = {
                                'level': level_name,
                                'target_price': target_price,
                                'multiplier': multiplier,
                                'macro_scaling': macro_scaling,
                                'distance_pct': distance_pct
                            }
                
                # Execute trade if opportunity found
                if best_opportunity:
                    total_multiplier = best_opportunity['multiplier'] * best_opportunity['macro_scaling']
                    trade_amount = self.base_amount * total_multiplier
                    btc_quantity = trade_amount / current_price
                    
                    # Record trade
                    trade = {
                        'date': date,
                        'level': best_opportunity['level'],
                        'price': current_price,
                        'amount_usd': trade_amount,
                        'btc_quantity': btc_quantity,
                        'multiplier': total_multiplier,
                        'distance_pct': best_opportunity['distance_pct']
                    }
                    
                    self.trades.append(trade)
                    total_btc += btc_quantity
                    total_invested += trade_amount
                    trade_count += 1
                    last_trade_date = date
                    
                    if trade_count <= 10:  # Log first 10 trades
                        logger.info(f"Trade {trade_count}: {best_opportunity['level']} @ ${current_price:,.0f} "
                                   f"(${trade_amount:.0f}, {total_multiplier:.1f}x)")
            
            # Create portfolio history DataFrame
            self.portfolio_history = pd.DataFrame(portfolio_values).set_index('date')
            
            logger.info(f"✅ Strategy simulation completed")
            logger.info(f"   Total trades: {len(self.trades)}")
            logger.info(f"   Final BTC: {total_btc:.6f}")
            logger.info(f"   Total invested: ${total_invested:,.0f}")
            if total_btc > 0:
                logger.info(f"   Average price: ${total_invested/total_btc:,.0f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to simulate strategy: {e}")
            raise
    
    def _calculate_simple_performance(self):
        """Calculate basic performance metrics"""
        try:
            logger.info("📊 Calculating performance metrics...")
            
            if self.portfolio_history.empty or len(self.trades) == 0:
                logger.warning("No trades executed")
                return {
                    'total_return': 0,
                    'annual_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'total_trades': 0,
                    'final_value': 0,
                    'total_invested': 0
                }
            
            # Basic metrics
            final_value = self.portfolio_history['value'].iloc[-1]
            total_invested = self.portfolio_history['invested'].iloc[-1]
            
            # Returns
            total_return = (final_value - total_invested) / total_invested if total_invested > 0 else 0
            
            # Time calculation
            start_date = self.portfolio_history.index[0]
            end_date = self.portfolio_history.index[-1]
            years_traded = (end_date - start_date).days / 365.25
            
            annual_return = (1 + total_return) ** (1 / years_traded) - 1 if years_traded > 0 else 0
            
            # Simple risk metrics
            daily_returns = self.portfolio_history['value'].pct_change().dropna()
            if len(daily_returns) > 1:
                volatility = daily_returns.std() * np.sqrt(365)
                sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Maximum drawdown
            rolling_max = self.portfolio_history['value'].expanding().max()
            drawdown = (self.portfolio_history['value'] - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0
            
            # Buy & Hold comparison
            start_price = self.backtest_data['Close'].iloc[0]
            end_price = self.backtest_data['Close'].iloc[-1]
            buy_hold_return = (end_price - start_price) / start_price
            buy_hold_annual = (1 + buy_hold_return) ** (1 / years_traded) - 1 if years_traded > 0 else 0
            
            results = {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': len(self.trades),
                'final_value': final_value,
                'total_invested': total_invested,
                'years_traded': years_traded,
                'buy_hold_annual': buy_hold_annual,
                'outperformance': annual_return - buy_hold_annual
            }
            
            logger.info("✅ Performance calculation completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate performance: {e}")
            return {}
    
    def _display_results(self, results):
        """Display backtest results"""
        print("\n" + "="*60)
        print("🌸 ENHANCED NANPIN STRATEGY BACKTEST RESULTS 🌸")
        print("="*60)
        
        if not results:
            print("❌ No results to display")
            return
        
        print(f"📅 Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print(f"📊 Total Trades: {results['total_trades']}")
        print(f"💰 Total Invested: ${results['total_invested']:,.0f}")
        print(f"💎 Final Portfolio Value: ${results['final_value']:,.0f}")
        print(f"📈 Total Return: {results['total_return']:+.1%}")
        print(f"📈 Annual Return: {results['annual_return']:+.1%}")
        print(f"📉 Max Drawdown: {results['max_drawdown']:.1%}")
        print(f"⚖️ Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"🎯 Years Traded: {results['years_traded']:.1f}")
        
        print(f"\n📊 BENCHMARK COMPARISON:")
        print(f"💼 Buy & Hold Annual: {results['buy_hold_annual']:+.1%}")
        print(f"🚀 Strategy Outperformance: {results['outperformance']:+.1%}")
        
        # Target comparison
        target_annual = 2.454  # 245.4%
        vs_target = results['annual_return'] / target_annual if target_annual > 0 else 0
        
        print(f"\n🎯 TARGET COMPARISON:")
        print(f"🏆 Target (Simple Trump Era): +245.4%")
        print(f"📊 Our Strategy: {results['annual_return']:+.1%}")
        print(f"🎯 vs Target: {vs_target:.1%} ({'+' if results['annual_return'] > target_annual else '-'})")
        
        # Performance grade
        if results['annual_return'] > target_annual:
            grade = "A+ 🎉"
            recommendation = "EXCELLENT - Strategy beats target!"
        elif results['annual_return'] > target_annual * 0.8:
            grade = "A 👍"
            recommendation = "STRONG - Close to target performance"
        elif results['annual_return'] > target_annual * 0.6:
            grade = "B 📊"
            recommendation = "GOOD - Solid performance but room for improvement"
        else:
            grade = "C ⚠️"
            recommendation = "NEEDS OPTIMIZATION"
        
        print(f"\n🏆 PERFORMANCE GRADE: {grade}")
        print(f"💡 RECOMMENDATION: {recommendation}")
        print("\n" + "="*60)

async def main():
    """Run the simple backtest"""
    try:
        print("🌸 Enhanced Nanpin Strategy - Simple Backtest")
        print("=" * 50)
        
        # Run backtest for different periods
        periods = [
            {"name": "Full Period", "start": "2020-01-01", "end": "2024-12-31"},
            {"name": "COVID Era", "start": "2020-01-01", "end": "2021-12-31"},
            {"name": "Recent Period", "start": "2023-01-01", "end": "2024-12-31"}
        ]
        
        all_results = {}
        
        for period in periods:
            print(f"\n🔄 Testing {period['name']} ({period['start']} to {period['end']})...")
            
            backtester = SimpleNanpinBacktest(period['start'], period['end'])
            results = await backtester.run_simple_backtest()
            all_results[period['name']] = results
        
        print(f"\n🎉 ALL BACKTESTS COMPLETED!")
        
        # Summary
        print(f"\n📋 SUMMARY:")
        for period_name, results in all_results.items():
            if results:
                print(f"   {period_name}: {results['annual_return']:+.1%} annual return")
        
    except Exception as e:
        print(f"❌ Simple backtest failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())