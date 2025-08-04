#!/usr/bin/env python3
"""
🌸 Ultimate Nanpin Backtest v3.0
Maximum performance optimization with advanced multi-timeframe analysis
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
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltimateNanpinBacktest:
    """Ultimate Nanpin backtester with maximum optimization"""
    
    def __init__(self, start_date="2020-01-01", end_date="2024-12-31"):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.base_amount = 100.0  # $100 base position size
        
        # Advanced Fibonacci configuration
        self.fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.fib_names = ['23.6%', '38.2%', '50.0%', '61.8%', '78.6%']
        self.fib_multipliers = {name: mult for name, mult in zip(self.fib_names, [1, 2, 3, 5, 8])}
        
        # Ultimate optimized entry criteria (based on historical analysis)
        self.entry_thresholds = {
            '23.6%': (-1.5, -0.3),   # Tighter for shallow retracements
            '38.2%': (-2.5, -0.8),   # More aggressive for medium retracements
            '50.0%': (-4.0, -1.2),   # Wider range for major retracements
            '61.8%': (-6.0, -2.0),   # Golden ratio - prime opportunity
            '78.6%': (-10.0, -3.5)   # Deep retracement - maximum opportunity
        }
        
        # Multi-timeframe analysis periods
        self.timeframes = {
            'short': 30,    # 30 days for immediate trends
            'medium': 90,   # 90 days for intermediate trends
            'long': 180     # 180 days for major trends
        }
        
        # Performance tracking
        self.trades = []
        self.portfolio_history = pd.DataFrame()
        self.btc_data = pd.DataFrame()
        
    async def run_ultimate_backtest(self):
        """Run ultimate optimized backtest"""
        try:
            logger.info("🚀 Starting Ultimate Nanpin Backtest v3.0")
            
            # Step 1: Load optimized dataset
            await self._load_ultimate_data()
            
            # Step 2: Multi-timeframe Fibonacci analysis
            self._calculate_multi_timeframe_fibonacci()
            
            # Step 3: Advanced macro intelligence
            self._add_ultimate_macro_intelligence()
            
            # Step 4: Simulate ultimate strategy
            self._simulate_ultimate_strategy()
            
            # Step 5: Calculate ultimate performance
            results = self._calculate_ultimate_performance()
            
            # Step 6: Display ultimate results
            self._display_ultimate_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Ultimate backtest failed: {e}")
            raise
    
    async def _load_ultimate_data(self):
        """Load optimized BTC dataset with enhanced indicators"""
        try:
            logger.info("📊 Loading ultimate dataset...")
            
            # Extended buffer for better analysis
            buffer_start = self.start_date - timedelta(days=365)
            
            btc = yf.Ticker("BTC-USD")
            data = btc.history(
                start=buffer_start.strftime("%Y-%m-%d"),
                end=self.end_date.strftime("%Y-%m-%d"),
                interval="1d"
            )
            
            self.btc_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            self.btc_data.index = pd.to_datetime(self.btc_data.index.date)
            self.btc_data = self.btc_data.dropna()
            
            # Add comprehensive technical indicators
            self._add_comprehensive_indicators()
            
            logger.info(f"✅ Ultimate data loaded: {len(self.btc_data)} days")
            logger.info(f"   Price range: ${self.btc_data['Low'].min():,.0f} - ${self.btc_data['High'].max():,.0f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load ultimate data: {e}")
            raise
    
    def _add_comprehensive_indicators(self):
        """Add comprehensive technical indicators"""
        try:
            # Price-based indicators
            self.btc_data['Returns'] = self.btc_data['Close'].pct_change()
            self.btc_data['Log_Returns'] = np.log(self.btc_data['Close'] / self.btc_data['Close'].shift(1))
            
            # Multiple moving averages
            for period in [7, 14, 21, 50, 100, 200]:
                self.btc_data[f'MA_{period}'] = self.btc_data['Close'].rolling(period).mean()
                self.btc_data[f'MA_{period}_Ratio'] = self.btc_data['Close'] / self.btc_data[f'MA_{period}']
            
            # Volatility measures (multiple timeframes)
            for window in [7, 14, 30, 60]:
                self.btc_data[f'Volatility_{window}d'] = (
                    self.btc_data['Returns'].rolling(window).std() * np.sqrt(365)
                )
            
            # Momentum indicators
            for period in [7, 14, 30, 60]:
                self.btc_data[f'Momentum_{period}d'] = (
                    self.btc_data['Close'] / self.btc_data['Close'].shift(period) - 1
                )
            
            # Support/Resistance levels
            for window in [20, 50, 90]:
                self.btc_data[f'Rolling_Max_{window}d'] = self.btc_data['High'].rolling(window).max()
                self.btc_data[f'Rolling_Min_{window}d'] = self.btc_data['Low'].rolling(window).min()
                self.btc_data[f'Price_Position_{window}d'] = (
                    (self.btc_data['Close'] - self.btc_data[f'Rolling_Min_{window}d']) /
                    (self.btc_data[f'Rolling_Max_{window}d'] - self.btc_data[f'Rolling_Min_{window}d'])
                )
            
            # Advanced drawdown analysis
            self.btc_data['ATH'] = self.btc_data['High'].expanding().max()
            self.btc_data['Drawdown_from_ATH'] = (self.btc_data['Close'] - self.btc_data['ATH']) / self.btc_data['ATH']
            
            # Volume analysis
            self.btc_data['Volume_MA_20'] = self.btc_data['Volume'].rolling(20).mean()
            self.btc_data['Volume_Ratio'] = self.btc_data['Volume'] / self.btc_data['Volume_MA_20']
            
            logger.debug("✅ Comprehensive indicators added")
            
        except Exception as e:
            logger.warning(f"Failed to add comprehensive indicators: {e}")
    
    def _calculate_multi_timeframe_fibonacci(self):
        """Calculate multi-timeframe Fibonacci levels"""
        try:
            logger.info("📐 Calculating multi-timeframe Fibonacci levels...")
            
            # Filter to backtest period
            backtest_data = self.btc_data[
                (self.btc_data.index >= pd.Timestamp(self.start_date)) &
                (self.btc_data.index <= pd.Timestamp(self.end_date))
            ].copy()
            
            fibonacci_data = []
            
            for date in backtest_data.index:
                day_fib_data = {}
                
                # Calculate Fibonacci for each timeframe
                for tf_name, tf_days in self.timeframes.items():
                    lookback_start = date - pd.Timedelta(days=tf_days)
                    period_data = self.btc_data[
                        (self.btc_data.index >= lookback_start) & 
                        (self.btc_data.index <= date)
                    ]
                    
                    if len(period_data) < tf_days // 2:
                        continue
                    
                    # Enhanced swing point detection
                    swing_high, swing_low = self._find_optimal_swing_points(period_data, tf_name)
                    
                    if swing_high > swing_low:
                        price_range = swing_high - swing_low
                        
                        # Calculate Fibonacci levels for this timeframe
                        tf_levels = {}
                        for level_name, ratio in zip(self.fib_names, self.fib_levels):
                            fib_price = swing_high - (price_range * ratio)
                            
                            # Multi-timeframe confluence score
                            confluence = self._calculate_multi_tf_confluence(
                                period_data, fib_price, tf_name, date
                            )
                            
                            tf_levels[level_name] = {
                                'price': fib_price,
                                'swing_high': swing_high,
                                'swing_low': swing_low,
                                'timeframe': tf_name,
                                'confluence_score': confluence,
                                'strength': self._calculate_level_strength(period_data, fib_price)
                            }
                        
                        day_fib_data[tf_name] = tf_levels
                
                # Combine timeframes for best opportunities
                combined_levels = self._combine_timeframe_levels(day_fib_data, date)
                fibonacci_data.append(combined_levels)
            
            backtest_data['Fibonacci_Levels'] = fibonacci_data
            self.backtest_data = backtest_data
            
            logger.info(f"✅ Multi-timeframe Fibonacci calculated for {len(self.backtest_data)} days")
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate multi-timeframe Fibonacci: {e}")
            raise
    
    def _find_optimal_swing_points(self, period_data: pd.DataFrame, timeframe: str) -> Tuple[float, float]:
        """Find optimal swing high and low points based on timeframe"""
        if timeframe == 'short':
            # More responsive for short-term
            swing_high = period_data['High'].rolling(5, center=True).max().max()
            swing_low = period_data['Low'].rolling(5, center=True).min().min()
        elif timeframe == 'medium':
            # Balanced approach
            swing_high = period_data['High'].rolling(10, center=True).max().max()
            swing_low = period_data['Low'].rolling(10, center=True).min().min()
        else:  # long
            # More stable for long-term
            swing_high = period_data['High'].rolling(20, center=True).max().max()
            swing_low = period_data['Low'].rolling(20, center=True).min().min()
        
        return swing_high, swing_low
    
    def _calculate_multi_tf_confluence(self, period_data: pd.DataFrame, 
                                     fib_price: float, timeframe: str, date: pd.Timestamp) -> float:
        """Calculate multi-timeframe confluence score"""
        try:
            confluence = 1.0
            
            # Moving average confluence
            ma_periods = [20, 50, 100, 200]
            ma_hits = 0
            for period in ma_periods:
                if f'MA_{period}' in self.btc_data.columns and date in self.btc_data.index:
                    ma_value = self.btc_data.loc[date, f'MA_{period}']
                    if pd.notna(ma_value) and abs(fib_price - ma_value) / ma_value < 0.03:
                        ma_hits += 1
            
            confluence += ma_hits * 0.1
            
            # Volume confirmation
            if date in self.btc_data.index:
                volume_ratio = self.btc_data.loc[date, 'Volume_Ratio']
                if pd.notna(volume_ratio) and volume_ratio > 1.2:
                    confluence += 0.15
            
            # Previous support/resistance
            tolerance = fib_price * 0.02
            historical_touches = period_data[
                abs(period_data['Close'] - fib_price) <= tolerance
            ]
            if len(historical_touches) > 2:
                confluence += 0.2 * min(len(historical_touches) / 10, 0.5)
            
            # Round number proximity
            if fib_price % 1000 < 150 or fib_price % 1000 > 850:
                confluence += 0.1
            elif fib_price % 5000 < 250 or fib_price % 5000 > 4750:
                confluence += 0.15
            
            # Timeframe weight
            tf_weights = {'short': 0.8, 'medium': 1.0, 'long': 1.2}
            confluence *= tf_weights.get(timeframe, 1.0)
            
            return min(confluence, 3.0)  # Cap at 3.0
            
        except Exception:
            return 1.0
    
    def _calculate_level_strength(self, period_data: pd.DataFrame, fib_price: float) -> float:
        """Calculate Fibonacci level strength"""
        try:
            strength = 1.0
            
            # Test count (how many times price touched this level)
            tolerance = fib_price * 0.015  # 1.5% tolerance
            touches = period_data[
                (period_data['Low'] <= fib_price + tolerance) &
                (period_data['High'] >= fib_price - tolerance)
            ]
            
            test_count = len(touches)
            if test_count > 1:
                strength += min(test_count * 0.1, 0.5)
            
            # Reaction strength (how much price moved away after touching)
            if not touches.empty:
                reactions = []
                for touch_date in touches.index:
                    # Look at next 5 days for reaction
                    future_data = period_data[period_data.index > touch_date].head(5)
                    if not future_data.empty:
                        max_move = max(
                            abs(future_data['High'].max() - fib_price),
                            abs(fib_price - future_data['Low'].min())
                        )
                        reaction_pct = max_move / fib_price
                        reactions.append(reaction_pct)
                
                if reactions:
                    avg_reaction = np.mean(reactions)
                    strength += min(avg_reaction * 5, 1.0)  # Max 1.0 bonus
            
            return min(strength, 2.5)  # Cap at 2.5
            
        except Exception:
            return 1.0
    
    def _combine_timeframe_levels(self, day_fib_data: Dict, date: pd.Timestamp) -> Dict:
        """Combine multiple timeframe levels for best opportunities"""
        try:
            combined = {}
            
            # For each Fibonacci level, find the best timeframe
            for level_name in self.fib_names:
                best_level = None
                best_score = 0
                
                for tf_name, tf_levels in day_fib_data.items():
                    if level_name in tf_levels:
                        level_data = tf_levels[level_name]
                        
                        # Calculate combined score
                        score = (
                            level_data['confluence_score'] * 
                            level_data['strength'] * 
                            self.fib_multipliers[level_name]
                        )
                        
                        if score > best_score:
                            best_score = score
                            best_level = level_data.copy()
                            best_level['combined_score'] = score
                            best_level['multiplier'] = self.fib_multipliers[level_name]
                
                if best_level:
                    combined[level_name] = best_level
            
            return combined
            
        except Exception:
            return {}
    
    def _add_ultimate_macro_intelligence(self):
        """Add ultimate macro intelligence"""
        try:
            logger.info("🔮 Adding ultimate macro intelligence...")
            
            # Simulate advanced macro indicators based on BTC price action and volatility
            regimes = []
            position_scaling = []
            risk_adjustments = []
            dynamic_cooldowns = []
            fear_greed_scores = []
            
            for date in self.backtest_data.index:
                # Enhanced regime detection based on multiple factors
                current_price = self.backtest_data.loc[date, 'Close']
                volatility_30d = self.backtest_data.loc[date, 'Volatility_30d']
                drawdown = self.backtest_data.loc[date, 'Drawdown_from_ATH']
                momentum_30d = self.backtest_data.loc[date, 'Momentum_30d']
                
                # Handle NaN values
                volatility_30d = volatility_30d if pd.notna(volatility_30d) else 0.3
                drawdown = drawdown if pd.notna(drawdown) else 0
                momentum_30d = momentum_30d if pd.notna(momentum_30d) else 0
                
                # Synthetic VIX based on BTC volatility
                synthetic_vix = min(volatility_30d * 50, 80)  # Scale volatility to VIX-like range
                
                # Enhanced Fear & Greed calculation
                fear_greed = 50  # Neutral starting point
                
                # Volatility component (inverse)
                fear_greed -= min(synthetic_vix - 20, 30)
                
                # Drawdown component
                fear_greed += drawdown * 100  # Drawdown is negative, so this reduces fear_greed
                
                # Momentum component
                if momentum_30d > 0.5:  # Strong positive momentum
                    fear_greed += 20
                elif momentum_30d < -0.3:  # Strong negative momentum
                    fear_greed -= 20
                
                # Clamp to 0-100 range
                fear_greed = np.clip(fear_greed, 0, 100)
                fear_greed_scores.append(fear_greed)
                
                # Advanced regime classification
                if (synthetic_vix > 50 or drawdown < -0.5 or fear_greed < 15):
                    regime = 'crisis'
                    scaling = 3.0
                    risk_adj = 1.3
                    cooldown = 1
                elif (synthetic_vix > 35 or drawdown < -0.3 or fear_greed < 25):
                    regime = 'recession'
                    scaling = 2.2
                    risk_adj = 1.2
                    cooldown = 1
                elif (synthetic_vix < 15 and momentum_30d > 0.2 and fear_greed > 75):
                    regime = 'bubble'
                    scaling = 0.4
                    risk_adj = 0.7
                    cooldown = 7
                elif (drawdown > -0.1 and momentum_30d > 0.1):
                    regime = 'recovery'
                    scaling = 1.4
                    risk_adj = 1.1
                    cooldown = 2
                else:
                    regime = 'expansion'
                    scaling = 1.0
                    risk_adj = 1.0
                    cooldown = 3
                
                # Additional scaling based on Fear & Greed
                if fear_greed < 10:
                    scaling *= 2.0  # Extreme fear
                elif fear_greed < 20:
                    scaling *= 1.6  # Strong fear
                elif fear_greed < 35:
                    scaling *= 1.3  # Fear
                elif fear_greed > 90:
                    scaling *= 0.3  # Extreme greed
                elif fear_greed > 80:
                    scaling *= 0.6  # Greed
                
                regimes.append(regime)
                position_scaling.append(min(scaling, 4.0))  # Cap at 4x
                risk_adjustments.append(risk_adj)
                dynamic_cooldowns.append(cooldown)
            
            self.backtest_data['Regime'] = regimes
            self.backtest_data['Position_Scaling'] = position_scaling
            self.backtest_data['Risk_Adjustment'] = risk_adjustments
            self.backtest_data['Dynamic_Cooldown'] = dynamic_cooldowns
            self.backtest_data['Fear_Greed'] = fear_greed_scores
            
            logger.info("✅ Ultimate macro intelligence added")
            logger.info(f"   Crisis periods: {(pd.Series(regimes) == 'crisis').sum()} days")
            logger.info(f"   Bubble periods: {(pd.Series(regimes) == 'bubble').sum()} days")
            logger.info(f"   Avg position scaling: {np.mean(position_scaling):.2f}x")
            logger.info(f"   Avg Fear/Greed: {np.mean(fear_greed_scores):.1f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to add ultimate macro intelligence: {e}")
            raise
    
    def _simulate_ultimate_strategy(self):
        """Simulate ultimate optimized strategy"""
        try:
            logger.info("💰 Simulating Ultimate Nanpin Strategy v3.0...")
            
            total_btc = 0.0
            total_invested = 0.0
            trade_count = 0
            last_trade_date = None
            
            portfolio_values = []
            self.trades = []
            
            for date, row in self.backtest_data.iterrows():
                current_price = row['Close']
                fibonacci_levels = row['Fibonacci_Levels']
                macro_scaling = row['Position_Scaling']
                risk_adjustment = row['Risk_Adjustment']
                dynamic_cooldown = int(row['Dynamic_Cooldown'])
                fear_greed = row['Fear_Greed']
                
                # Calculate portfolio value
                portfolio_value = total_btc * current_price
                portfolio_values.append({
                    'date': date,
                    'btc': total_btc,
                    'value': portfolio_value,
                    'invested': total_invested,
                    'price': current_price
                })
                
                # Advanced cooldown logic
                if last_trade_date:
                    days_since_trade = (date - last_trade_date).days
                    if days_since_trade < dynamic_cooldown:
                        # Allow override for extreme opportunities
                        if not (fear_greed < 15 and macro_scaling > 2.5):
                            continue
                
                # Skip if no Fibonacci levels
                if not fibonacci_levels:
                    continue
                
                # Ultimate opportunity analysis
                best_opportunity = None
                best_score = 0
                
                for level_name, level_data in fibonacci_levels.items():
                    target_price = level_data['price']
                    base_multiplier = level_data['multiplier']
                    confluence_score = level_data.get('confluence_score', 1.0)
                    strength = level_data.get('strength', 1.0)
                    combined_score = level_data.get('combined_score', 1.0)
                    
                    # Calculate distance from target
                    distance_pct = (current_price - target_price) / target_price * 100
                    
                    # Ultimate entry criteria
                    min_threshold, max_threshold = self.entry_thresholds[level_name]
                    
                    if min_threshold <= distance_pct <= max_threshold:
                        # Ultimate opportunity scoring
                        opportunity_score = (
                            base_multiplier * 
                            macro_scaling * 
                            confluence_score * 
                            strength *
                            abs(distance_pct) * 
                            risk_adjustment *
                            (1 + combined_score / 10)  # Boost from combined score
                        )
                        
                        # Bonus for extreme fear
                        if fear_greed < 20:
                            opportunity_score *= 1.5
                        elif fear_greed < 35:
                            opportunity_score *= 1.2
                        
                        if opportunity_score > best_score:
                            best_score = opportunity_score
                            best_opportunity = {
                                'level': level_name,
                                'target_price': target_price,
                                'base_multiplier': base_multiplier,
                                'macro_scaling': macro_scaling,
                                'confluence_score': confluence_score,
                                'strength': strength,
                                'risk_adjustment': risk_adjustment,
                                'distance_pct': distance_pct,
                                'combined_score': combined_score,
                                'total_multiplier': min(base_multiplier * macro_scaling * risk_adjustment, 20.0)
                            }
                
                # Execute ultimate trade
                if best_opportunity and best_score > 3.0:  # Higher quality threshold
                    total_multiplier = best_opportunity['total_multiplier']
                    trade_amount = self.base_amount * total_multiplier
                    btc_quantity = trade_amount / current_price
                    
                    # Ultimate trade record
                    trade = {
                        'date': date,
                        'level': best_opportunity['level'],
                        'price': current_price,
                        'amount_usd': trade_amount,
                        'btc_quantity': btc_quantity,
                        'base_multiplier': best_opportunity['base_multiplier'],
                        'macro_scaling': best_opportunity['macro_scaling'],
                        'risk_adjustment': best_opportunity['risk_adjustment'],
                        'total_multiplier': total_multiplier,
                        'confluence_score': best_opportunity['confluence_score'],
                        'strength': best_opportunity['strength'],
                        'distance_pct': best_opportunity['distance_pct'],
                        'regime': row['Regime'],
                        'fear_greed': fear_greed,
                        'opportunity_score': best_score,
                        'combined_score': best_opportunity['combined_score']
                    }
                    
                    self.trades.append(trade)
                    total_btc += btc_quantity
                    total_invested += trade_amount
                    trade_count += 1
                    last_trade_date = date
                    
                    if trade_count <= 20:  # Log first 20 trades
                        logger.info(
                            f"Trade {trade_count}: {best_opportunity['level']} @ ${current_price:,.0f} "
                            f"(${trade_amount:.0f}, {total_multiplier:.1f}x, {row['Regime']}, "
                            f"F&G:{fear_greed:.0f})"
                        )
            
            # Create portfolio history DataFrame
            self.portfolio_history = pd.DataFrame(portfolio_values).set_index('date')
            
            logger.info(f"✅ Ultimate strategy simulation completed")
            logger.info(f"   Total trades: {len(self.trades)}")
            logger.info(f"   Final BTC: {total_btc:.6f}")
            logger.info(f"   Total invested: ${total_invested:,.0f}")
            if total_btc > 0:
                logger.info(f"   Average price: ${total_invested/total_btc:,.0f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to simulate ultimate strategy: {e}")
            raise
    
    def _calculate_ultimate_performance(self):
        """Calculate ultimate performance metrics"""
        try:
            logger.info("📊 Calculating ultimate performance metrics...")
            
            if self.portfolio_history.empty or len(self.trades) == 0:
                return {
                    'total_return': 0, 'annual_return': 0, 'sharpe_ratio': 0,
                    'max_drawdown': 0, 'total_trades': 0, 'final_value': 0,
                    'total_invested': 0, 'ultimate_metrics': {}
                }
            
            # Basic metrics
            final_value = self.portfolio_history['value'].iloc[-1]
            total_invested = self.portfolio_history['invested'].iloc[-1]
            total_return = (final_value - total_invested) / total_invested if total_invested > 0 else 0
            
            # Time calculation
            start_date = self.portfolio_history.index[0]
            end_date = self.portfolio_history.index[-1]
            years_traded = (end_date - start_date).days / 365.25
            annual_return = (1 + total_return) ** (1 / years_traded) - 1 if years_traded > 0 else 0
            
            # Advanced risk metrics
            daily_returns = self.portfolio_history['value'].pct_change().dropna()
            if len(daily_returns) > 1:
                volatility = daily_returns.std() * np.sqrt(365)
                sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
                
                # Sortino ratio (downside deviation)
                downside_returns = daily_returns[daily_returns < 0]
                downside_vol = downside_returns.std() * np.sqrt(365) if len(downside_returns) > 0 else 0
                sortino_ratio = (annual_return - 0.02) / downside_vol if downside_vol > 0 else 0
                
                # Calmar ratio
                rolling_max = self.portfolio_history['value'].expanding().max()
                drawdown = (self.portfolio_history['value'] - rolling_max) / rolling_max
                max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0
                calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
            else:
                sharpe_ratio = sortino_ratio = calmar_ratio = 0
                volatility = max_drawdown = 0
            
            # Buy & Hold comparison
            start_price = self.backtest_data['Close'].iloc[0]
            end_price = self.backtest_data['Close'].iloc[-1]
            buy_hold_return = (end_price - start_price) / start_price
            buy_hold_annual = (1 + buy_hold_return) ** (1 / years_traded) - 1 if years_traded > 0 else 0
            
            # Ultimate analysis
            trades_df = pd.DataFrame(self.trades)
            ultimate_metrics = {}
            
            if not trades_df.empty:
                ultimate_metrics = {
                    'avg_confluence_score': trades_df['confluence_score'].mean(),
                    'avg_strength': trades_df['strength'].mean(),
                    'avg_macro_scaling': trades_df['macro_scaling'].mean(),
                    'avg_combined_score': trades_df['combined_score'].mean(),
                    'crisis_trades': (trades_df['regime'] == 'crisis').sum(),
                    'recession_trades': (trades_df['regime'] == 'recession').sum(),
                    'bubble_trades': (trades_df['regime'] == 'bubble').sum(),
                    'extreme_fear_trades': (trades_df['fear_greed'] < 20).sum(),
                    'high_score_trades': (trades_df['opportunity_score'] > 10.0).sum(),
                    'avg_opportunity_score': trades_df['opportunity_score'].mean(),
                    'golden_ratio_trades': (trades_df['level'] == '61.8%').sum(),
                    'deep_retracement_trades': (trades_df['level'] == '78.6%').sum()
                }
            
            results = {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'total_trades': len(self.trades),
                'final_value': final_value,
                'total_invested': total_invested,
                'years_traded': years_traded,
                'buy_hold_annual': buy_hold_annual,
                'outperformance': annual_return - buy_hold_annual,
                'ultimate_metrics': ultimate_metrics
            }
            
            logger.info("✅ Ultimate performance calculation completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate ultimate performance: {e}")
            return {}
    
    def _display_ultimate_results(self, results):
        """Display ultimate backtest results"""
        print("\n" + "="*80)
        print("🌸 ULTIMATE NANPIN STRATEGY v3.0 BACKTEST RESULTS 🌸")
        print("="*80)
        
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
        print(f"📊 Volatility: {results['volatility']:.1%}")
        print(f"⚖️ Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"📊 Sortino Ratio: {results['sortino_ratio']:.2f}")
        print(f"📊 Calmar Ratio: {results['calmar_ratio']:.2f}")
        
        print(f"\n📊 BENCHMARK COMPARISON:")
        print(f"💼 Buy & Hold Annual: {results['buy_hold_annual']:+.1%}")
        print(f"🚀 Strategy Outperformance: {results['outperformance']:+.1%}")
        
        # Ultimate metrics
        if results['ultimate_metrics']:
            um = results['ultimate_metrics']
            print(f"\n🔧 ULTIMATE ANALYSIS:")
            print(f"🎯 Avg Confluence Score: {um.get('avg_confluence_score', 0):.2f}")
            print(f"💪 Avg Strength Score: {um.get('avg_strength', 0):.2f}")
            print(f"📊 Avg Macro Scaling: {um.get('avg_macro_scaling', 0):.2f}x")
            print(f"🔗 Avg Combined Score: {um.get('avg_combined_score', 0):.2f}")
            print(f"🚨 Crisis Trades: {um.get('crisis_trades', 0)}")
            print(f"📉 Recession Trades: {um.get('recession_trades', 0)}")
            print(f"🎈 Bubble Trades: {um.get('bubble_trades', 0)}")
            print(f"😱 Extreme Fear Trades: {um.get('extreme_fear_trades', 0)}")
            print(f"⭐ High Score Trades: {um.get('high_score_trades', 0)}")
            print(f"🎯 Avg Opportunity Score: {um.get('avg_opportunity_score', 0):.1f}")
            print(f"🏆 Golden Ratio (61.8%) Trades: {um.get('golden_ratio_trades', 0)}")
            print(f"💎 Deep Retracement (78.6%) Trades: {um.get('deep_retracement_trades', 0)}")
        
        # Target comparison
        target_annual = 2.454  # 245.4%
        vs_target = results['annual_return'] / target_annual if target_annual > 0 else 0
        
        print(f"\n🎯 TARGET COMPARISON:")
        print(f"🏆 Target (Simple Trump Era): +245.4%")
        print(f"📊 Ultimate Strategy: {results['annual_return']:+.1%}")
        print(f"🎯 vs Target: {vs_target:.1%} ({'+' if results['annual_return'] > target_annual else '-'})")
        
        # Ultimate performance grade
        if results['annual_return'] > target_annual * 1.2 and results['sharpe_ratio'] > 2.5:
            grade = "S+ 🌟"
            recommendation = "LEGENDARY - Strategy crushes all targets!"
        elif results['annual_return'] > target_annual and results['sharpe_ratio'] > 2.0:
            grade = "A+ 🎉"
            recommendation = "EXCEPTIONAL - Strategy significantly beats target!"
        elif results['annual_return'] > target_annual * 0.9 and results['sharpe_ratio'] > 1.5:
            grade = "A 🎯"
            recommendation = "EXCELLENT - Strategy beats target!"
        elif results['annual_return'] > target_annual * 0.7:
            grade = "A- 👍"
            recommendation = "STRONG - Very close to target performance"
        elif results['annual_return'] > target_annual * 0.5:
            grade = "B+ 📊"
            recommendation = "GOOD - Solid performance, optimization successful"
        elif results['annual_return'] > target_annual * 0.3:
            grade = "B 📈"
            recommendation = "MODERATE - Decent improvement achieved"
        else:
            grade = "C ⚠️"
            recommendation = "NEEDS FURTHER OPTIMIZATION"
        
        print(f"\n🏆 ULTIMATE PERFORMANCE GRADE: {grade}")
        print(f"💡 RECOMMENDATION: {recommendation}")
        
        # Success indicators
        if results['annual_return'] > target_annual:
            print(f"\n🎊 SUCCESS INDICATORS:")
            print(f"✅ Target beaten by {(vs_target - 1) * 100:+.1f}%")
            print(f"✅ Risk-adjusted performance: Sharpe {results['sharpe_ratio']:.2f}")
            print(f"✅ Downside protection: Sortino {results['sortino_ratio']:.2f}")
            print(f"✅ Drawdown management: {results['max_drawdown']:.1%} max")
        
        print("\n" + "="*80)

async def main():
    """Run the ultimate backtest"""
    try:
        print("🌸 Ultimate Nanpin Strategy v3.0 - Maximum Performance Optimization")
        print("=" * 70)
        
        # Run ultimate backtest for key periods
        periods = [
            {"name": "Full Period (Ultimate)", "start": "2020-01-01", "end": "2024-12-31"},
            {"name": "COVID Era (Ultimate)", "start": "2020-01-01", "end": "2021-12-31"},
            {"name": "Bull Run 2021", "start": "2020-10-01", "end": "2021-11-30"},
            {"name": "Bear Market 2022", "start": "2021-11-01", "end": "2022-12-31"},
            {"name": "Recovery 2023-24", "start": "2023-01-01", "end": "2024-12-31"}
        ]
        
        all_results = {}
        
        for period in periods:
            print(f"\n🔄 Testing {period['name']} ({period['start']} to {period['end']})...")
            
            backtester = UltimateNanpinBacktest(period['start'], period['end'])
            results = await backtester.run_ultimate_backtest()
            all_results[period['name']] = results
        
        print(f"\n🎉 ALL ULTIMATE BACKTESTS COMPLETED!")
        
        # Ultimate summary
        print(f"\n📋 ULTIMATE STRATEGY SUMMARY:")
        for period_name, results in all_results.items():
            if results:
                sharpe = results.get('sharpe_ratio', 0)
                outperf = results.get('outperformance', 0)
                print(f"   {period_name}: {results['annual_return']:+.1%} annual "
                      f"(Sharpe: {sharpe:.2f}, vs BH: {outperf:+.1%})")
        
        # Find best performing period
        best_period = max(all_results.items(), 
                         key=lambda x: x[1].get('annual_return', 0) if x[1] else 0)
        if best_period[1]:
            print(f"\n🏆 ULTIMATE BEST PERFORMANCE: {best_period[0]}")
            print(f"   Annual Return: {best_period[1]['annual_return']:+.1%}")
            print(f"   Sharpe Ratio: {best_period[1]['sharpe_ratio']:.2f}")
            print(f"   Total Trades: {best_period[1]['total_trades']}")
            print(f"   vs Target (+245.4%): {best_period[1]['annual_return']/2.454:.1%}")
        
        # Check if any period beat the target
        target_beaters = [name for name, results in all_results.items() 
                         if results and results.get('annual_return', 0) > 2.454]
        
        if target_beaters:
            print(f"\n🎯 TARGET ACHIEVED IN: {', '.join(target_beaters)}")
        else:
            print(f"\n📊 OPTIMIZATION STATUS: Target not yet achieved, further tuning needed")
        
    except Exception as e:
        print(f"❌ Ultimate backtest failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())