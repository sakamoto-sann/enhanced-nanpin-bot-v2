#!/usr/bin/env python3
"""
🌸 Enhanced Nanpin Backtest v2.0
Advanced validation with real FRED data and optimized parameters
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

class RealTimeDataFetcher:
    """Enhanced data fetcher with real FRED and market data"""
    
    def __init__(self):
        self.fred_api_key = "7aa42875026454682d22f3e02afff1b2"
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        # Key FRED series IDs
        self.fred_series = {
            'FEDFUNDS': 'Federal Funds Rate',
            'CPILFESL': 'Core CPI',
            'VIXCLS': 'VIX Volatility Index',
            'DGS10': '10-Year Treasury',
            'DGS2': '2-Year Treasury',
            'UNRATE': 'Unemployment Rate',
            'M2SL': 'M2 Money Supply',
            'DEXUSEU': 'USD/EUR Exchange Rate'
        }
    
    async def fetch_fred_data(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from FRED API"""
        try:
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'observation_start': start_date,
                'observation_end': end_date,
                'frequency': 'd',  # Daily frequency
                'sort_order': 'asc'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.fred_base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'observations' in data:
                            df = pd.DataFrame(data['observations'])
                            df['date'] = pd.to_datetime(df['date'])
                            df['value'] = pd.to_numeric(df['value'], errors='coerce')
                            df = df.dropna().set_index('date')
                            return df[['value']].rename(columns={'value': series_id})
                        
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"Failed to fetch FRED data for {series_id}: {e}")
            return pd.DataFrame()
    
    async def fetch_fear_greed_index(self) -> pd.DataFrame:
        """Fetch Fear & Greed Index (simulated with enhanced VIX correlation)"""
        try:
            # Use VIX as proxy for Fear & Greed
            vix_data = await self.fetch_fred_data('VIXCLS', '2020-01-01', '2024-12-31')
            if not vix_data.empty:
                # Convert VIX to Fear & Greed scale (inverse correlation)
                fear_greed = 100 - (vix_data['VIXCLS'] - 10) * 2.5
                fear_greed = np.clip(fear_greed, 0, 100)
                return pd.DataFrame({'Fear_Greed': fear_greed}, index=vix_data.index)
            
        except Exception as e:
            logger.warning(f"Failed to fetch Fear & Greed data: {e}")
        
        return pd.DataFrame()
    
    async def fetch_all_macro_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch comprehensive macro dataset"""
        try:
            logger.info("🔮 Fetching real-time macro data from FRED...")
            
            # Fetch all FRED series
            macro_data = pd.DataFrame()
            
            for series_id, name in self.fred_series.items():
                data = await self.fetch_fred_data(series_id, start_date, end_date)
                if not data.empty:
                    if macro_data.empty:
                        macro_data = data
                    else:
                        macro_data = macro_data.join(data, how='outer')
                    logger.info(f"   ✅ {name}: {len(data)} data points")
                else:
                    logger.warning(f"   ❌ Failed to fetch {name}")
            
            # Fetch Fear & Greed Index
            fg_data = await self.fetch_fear_greed_index()
            if not fg_data.empty:
                macro_data = macro_data.join(fg_data, how='outer')
            
            # Forward fill missing values
            macro_data = macro_data.fillna(method='ffill')
            
            logger.info(f"✅ Macro data fetched: {len(macro_data)} days")
            return macro_data
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch macro data: {e}")
            return pd.DataFrame()

class EnhancedNanpinBacktest:
    """Enhanced Nanpin backtester with real macro data and optimizations"""
    
    def __init__(self, start_date="2020-01-01", end_date="2024-12-31"):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.base_amount = 100.0  # $100 base position size
        
        # Enhanced Fibonacci levels and multipliers
        self.fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.fib_names = ['23.6%', '38.2%', '50.0%', '61.8%', '78.6%']
        self.fib_multipliers = {name: mult for name, mult in zip(self.fib_names, [1, 2, 3, 5, 8])}
        
        # Enhanced entry criteria (optimization #2)
        self.entry_thresholds = {
            '23.6%': (-1.0, -0.5),   # 0.5% to 1% below
            '38.2%': (-2.0, -1.0),   # 1% to 2% below  
            '50.0%': (-3.0, -1.5),   # 1.5% to 3% below
            '61.8%': (-5.0, -2.0),   # 2% to 5% below
            '78.6%': (-8.0, -3.0)    # 3% to 8% below
        }
        
        # Performance tracking
        self.trades = []
        self.portfolio_history = pd.DataFrame()
        self.btc_data = pd.DataFrame()
        self.macro_data = pd.DataFrame()
        
        # Data fetcher
        self.data_fetcher = RealTimeDataFetcher()
        
    async def run_enhanced_backtest(self):
        """Run enhanced backtest with all optimizations"""
        try:
            logger.info("🚀 Starting Enhanced Nanpin Backtest v2.0")
            
            # Step 1: Load BTC and macro data
            await self._load_enhanced_data()
            
            # Step 2: Calculate enhanced Fibonacci levels
            self._calculate_enhanced_fibonacci_levels()
            
            # Step 3: Add macro intelligence
            self._add_macro_intelligence()
            
            # Step 4: Simulate enhanced strategy
            self._simulate_enhanced_strategy()
            
            # Step 5: Calculate performance
            results = self._calculate_enhanced_performance()
            
            # Step 6: Display results
            self._display_enhanced_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Enhanced backtest failed: {e}")
            raise
    
    async def _load_enhanced_data(self):
        """Load BTC and real macro data"""
        try:
            logger.info("📊 Loading enhanced dataset...")
            
            # Load BTC data
            buffer_start = self.start_date - timedelta(days=200)
            btc = yf.Ticker("BTC-USD")
            data = btc.history(
                start=buffer_start.strftime("%Y-%m-%d"),
                end=self.end_date.strftime("%Y-%m-%d"),
                interval="1d"
            )
            
            self.btc_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            self.btc_data.index = pd.to_datetime(self.btc_data.index.date)
            self.btc_data = self.btc_data.dropna()
            
            # Load real macro data
            self.macro_data = await self.data_fetcher.fetch_all_macro_data(
                self.start_date.strftime("%Y-%m-%d"),
                self.end_date.strftime("%Y-%m-%d")
            )
            
            logger.info(f"✅ Enhanced data loaded:")
            logger.info(f"   BTC data: {len(self.btc_data)} days")
            logger.info(f"   Macro data: {len(self.macro_data)} days")
            
        except Exception as e:
            logger.error(f"❌ Failed to load enhanced data: {e}")
            raise
    
    def _calculate_enhanced_fibonacci_levels(self):
        """Calculate enhanced Fibonacci levels with dynamic lookback"""
        try:
            logger.info("📐 Calculating enhanced Fibonacci levels...")
            
            # Filter to backtest period
            backtest_data = self.btc_data[
                (self.btc_data.index >= pd.Timestamp(self.start_date)) &
                (self.btc_data.index <= pd.Timestamp(self.end_date))
            ].copy()
            
            fibonacci_levels = []
            
            for date in backtest_data.index:
                # Dynamic lookback period based on volatility (optimization #3)
                if date in self.macro_data.index and 'VIXCLS' in self.macro_data.columns:
                    vix = self.macro_data.loc[date, 'VIXCLS']
                    if pd.notna(vix):
                        # Higher volatility = shorter lookback (more responsive)
                        if vix > 40:
                            lookback_days = 60  # Crisis mode
                        elif vix > 25:
                            lookback_days = 75  # High volatility
                        else:
                            lookback_days = 90  # Normal conditions
                    else:
                        lookback_days = 90
                else:
                    lookback_days = 90
                
                lookback_start = date - pd.Timedelta(days=lookback_days)
                period_data = self.btc_data[
                    (self.btc_data.index >= lookback_start) & 
                    (self.btc_data.index <= date)
                ]
                
                if len(period_data) < 30:
                    fibonacci_levels.append({})
                    continue
                
                # Enhanced swing point detection
                swing_high = period_data['High'].max()
                swing_low = period_data['Low'].min()
                price_range = swing_high - swing_low
                
                # Calculate Fibonacci levels with confidence scoring
                levels = {}
                for i, (level_name, ratio) in enumerate(zip(self.fib_names, self.fib_levels)):
                    fib_price = swing_high - (price_range * ratio)
                    
                    # Add confluence factors for confidence
                    confluence_score = self._calculate_confluence_score(
                        period_data, fib_price, date
                    )
                    
                    levels[level_name] = {
                        'price': fib_price,
                        'swing_high': swing_high,
                        'swing_low': swing_low,
                        'multiplier': self.fib_multipliers[level_name],
                        'confluence_score': confluence_score,
                        'lookback_days': lookback_days
                    }
                
                fibonacci_levels.append(levels)
            
            backtest_data['Fibonacci_Levels'] = fibonacci_levels
            self.backtest_data = backtest_data
            
            logger.info(f"✅ Enhanced Fibonacci levels calculated for {len(self.backtest_data)} days")
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate enhanced Fibonacci levels: {e}")
            raise
    
    def _calculate_confluence_score(self, period_data: pd.DataFrame, fib_price: float, date: pd.Timestamp) -> float:
        """Calculate confluence score for Fibonacci level strength"""
        try:
            score = 1.0  # Base score
            
            # Moving average confluence
            if len(period_data) >= 50:
                ma_50 = period_data['Close'].tail(50).mean()
                if abs(fib_price - ma_50) / ma_50 < 0.02:  # Within 2%
                    score += 0.3
            
            # Volume-weighted support/resistance
            if len(period_data) >= 20:
                price_tolerance = fib_price * 0.02  # 2% tolerance
                nearby_prices = period_data[
                    abs(period_data['Close'] - fib_price) <= price_tolerance
                ]
                if len(nearby_prices) > 2:
                    score += 0.2 * (len(nearby_prices) / 20)  # Max 0.2 bonus
            
            # Round number proximity
            if fib_price % 1000 < 100 or fib_price % 1000 > 900:  # Near $1000 intervals
                score += 0.15
            
            return min(score, 2.0)  # Cap at 2.0
            
        except Exception:
            return 1.0
    
    def _add_macro_intelligence(self):
        """Add macro intelligence to backtest data"""
        try:
            logger.info("🔮 Adding macro intelligence...")
            
            if self.macro_data.empty:
                logger.warning("No macro data available, using simplified indicators")
                # Fallback to simplified indicators
                self.backtest_data['Regime'] = 'expansion'
                self.backtest_data['Position_Scaling'] = 1.0
                self.backtest_data['Risk_Adjustment'] = 1.0
                self.backtest_data['Dynamic_Cooldown'] = 3
                return
            
            # Align macro data with backtest dates
            macro_aligned = self.macro_data.reindex(
                self.backtest_data.index, method='ffill'
            )
            
            # Economic regime classification (optimization #1)
            regimes = []
            position_scaling = []
            risk_adjustments = []
            dynamic_cooldowns = []
            
            for date in self.backtest_data.index:
                if date in macro_aligned.index:
                    # Get macro indicators
                    vix = macro_aligned.loc[date, 'VIXCLS'] if 'VIXCLS' in macro_aligned.columns else 20
                    fed_rate = macro_aligned.loc[date, 'FEDFUNDS'] if 'FEDFUNDS' in macro_aligned.columns else 2
                    unemployment = macro_aligned.loc[date, 'UNRATE'] if 'UNRATE' in macro_aligned.columns else 4
                    fear_greed = macro_aligned.loc[date, 'Fear_Greed'] if 'Fear_Greed' in macro_aligned.columns else 50
                    
                    # Handle NaN values
                    vix = vix if pd.notna(vix) else 20
                    fed_rate = fed_rate if pd.notna(fed_rate) else 2
                    unemployment = unemployment if pd.notna(unemployment) else 4
                    fear_greed = fear_greed if pd.notna(fear_greed) else 50
                    
                    # Classify regime
                    if vix > 40 or unemployment > 8 or fear_greed < 20:
                        regime = 'crisis'
                        scaling = 2.5
                        risk_adj = 1.2
                        cooldown = 1  # Aggressive during crisis
                    elif vix > 30 or unemployment > 6 or fear_greed < 35:
                        regime = 'recession'
                        scaling = 2.0
                        risk_adj = 1.1
                        cooldown = 2
                    elif fed_rate < 1 and unemployment < 5:
                        regime = 'recovery'
                        scaling = 1.3
                        risk_adj = 1.0
                        cooldown = 2
                    elif fear_greed > 80 and vix < 15:
                        regime = 'bubble'
                        scaling = 0.6
                        risk_adj = 0.8
                        cooldown = 5  # Conservative during bubbles
                    else:
                        regime = 'expansion'
                        scaling = 1.0
                        risk_adj = 1.0
                        cooldown = 3
                    
                    # Additional scaling based on Fear & Greed
                    if fear_greed < 20:
                        scaling *= 1.8  # Extreme fear opportunity
                    elif fear_greed < 35:
                        scaling *= 1.4  # Fear opportunity
                    elif fear_greed > 85:
                        scaling *= 0.5  # Extreme greed caution
                    
                    regimes.append(regime)
                    position_scaling.append(min(scaling, 3.0))  # Cap at 3x
                    risk_adjustments.append(risk_adj)
                    dynamic_cooldowns.append(cooldown)
                else:
                    regimes.append('expansion')
                    position_scaling.append(1.0)
                    risk_adjustments.append(1.0)
                    dynamic_cooldowns.append(3)
            
            self.backtest_data['Regime'] = regimes
            self.backtest_data['Position_Scaling'] = position_scaling
            self.backtest_data['Risk_Adjustment'] = risk_adjustments
            self.backtest_data['Dynamic_Cooldown'] = dynamic_cooldowns
            
            logger.info("✅ Macro intelligence added")
            logger.info(f"   Crisis periods: {(pd.Series(regimes) == 'crisis').sum()} days")
            logger.info(f"   Avg position scaling: {np.mean(position_scaling):.2f}x")
            
        except Exception as e:
            logger.error(f"❌ Failed to add macro intelligence: {e}")
            raise
    
    def _simulate_enhanced_strategy(self):
        """Simulate enhanced strategy with all optimizations"""
        try:
            logger.info("💰 Simulating Enhanced Nanpin Strategy v2.0...")
            
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
                
                # Calculate portfolio value
                portfolio_value = total_btc * current_price
                portfolio_values.append({
                    'date': date,
                    'btc': total_btc,
                    'value': portfolio_value,
                    'invested': total_invested,
                    'price': current_price
                })
                
                # Check dynamic cooldown (optimization #4)
                if (last_trade_date and 
                    (date - last_trade_date).days < dynamic_cooldown):
                    continue
                
                # Skip if no Fibonacci levels
                if not fibonacci_levels:
                    continue
                
                # Enhanced opportunity analysis
                best_opportunity = None
                best_score = 0
                
                for level_name, level_data in fibonacci_levels.items():
                    target_price = level_data['price']
                    base_multiplier = level_data['multiplier']
                    confluence_score = level_data.get('confluence_score', 1.0)
                    
                    # Calculate distance from target
                    distance_pct = (current_price - target_price) / target_price * 100
                    
                    # Enhanced entry criteria (optimization #2)
                    min_threshold, max_threshold = self.entry_thresholds[level_name]
                    
                    if min_threshold <= distance_pct <= max_threshold:
                        # Enhanced opportunity scoring
                        opportunity_score = (
                            base_multiplier * 
                            macro_scaling * 
                            confluence_score * 
                            abs(distance_pct) * 
                            risk_adjustment
                        )
                        
                        if opportunity_score > best_score:
                            best_score = opportunity_score
                            best_opportunity = {
                                'level': level_name,
                                'target_price': target_price,
                                'base_multiplier': base_multiplier,
                                'macro_scaling': macro_scaling,
                                'confluence_score': confluence_score,
                                'risk_adjustment': risk_adjustment,
                                'distance_pct': distance_pct,
                                'total_multiplier': base_multiplier * macro_scaling * risk_adjustment
                            }
                
                # Execute enhanced trade
                if best_opportunity and best_score > 2.0:  # Minimum quality threshold
                    total_multiplier = min(best_opportunity['total_multiplier'], 15.0)  # Cap at 15x
                    trade_amount = self.base_amount * total_multiplier
                    btc_quantity = trade_amount / current_price
                    
                    # Enhanced trade record
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
                        'distance_pct': best_opportunity['distance_pct'],
                        'regime': row['Regime'],
                        'opportunity_score': best_score
                    }
                    
                    self.trades.append(trade)
                    total_btc += btc_quantity
                    total_invested += trade_amount
                    trade_count += 1
                    last_trade_date = date
                    
                    if trade_count <= 15:  # Log first 15 trades
                        logger.info(
                            f"Trade {trade_count}: {best_opportunity['level']} @ ${current_price:,.0f} "
                            f"(${trade_amount:.0f}, {total_multiplier:.1f}x, {row['Regime']})"
                        )
            
            # Create portfolio history DataFrame
            self.portfolio_history = pd.DataFrame(portfolio_values).set_index('date')
            
            logger.info(f"✅ Enhanced strategy simulation completed")
            logger.info(f"   Total trades: {len(self.trades)}")
            logger.info(f"   Final BTC: {total_btc:.6f}")
            logger.info(f"   Total invested: ${total_invested:,.0f}")
            if total_btc > 0:
                logger.info(f"   Average price: ${total_invested/total_btc:,.0f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to simulate enhanced strategy: {e}")
            raise
    
    def _calculate_enhanced_performance(self):
        """Calculate enhanced performance metrics"""
        try:
            logger.info("📊 Calculating enhanced performance metrics...")
            
            if self.portfolio_history.empty or len(self.trades) == 0:
                return {
                    'total_return': 0, 'annual_return': 0, 'sharpe_ratio': 0,
                    'max_drawdown': 0, 'total_trades': 0, 'final_value': 0,
                    'total_invested': 0, 'enhancement_metrics': {}
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
            
            # Enhanced risk metrics
            daily_returns = self.portfolio_history['value'].pct_change().dropna()
            if len(daily_returns) > 1:
                volatility = daily_returns.std() * np.sqrt(365)
                sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
                
                # Sortino ratio
                downside_returns = daily_returns[daily_returns < 0]
                downside_vol = downside_returns.std() * np.sqrt(365) if len(downside_returns) > 0 else 0
                sortino_ratio = (annual_return - 0.02) / downside_vol if downside_vol > 0 else 0
            else:
                sharpe_ratio = 0
                sortino_ratio = 0
                volatility = 0
            
            # Maximum drawdown
            rolling_max = self.portfolio_history['value'].expanding().max()
            drawdown = (self.portfolio_history['value'] - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0
            
            # Buy & Hold comparison
            start_price = self.backtest_data['Close'].iloc[0]
            end_price = self.backtest_data['Close'].iloc[-1]
            buy_hold_return = (end_price - start_price) / start_price
            buy_hold_annual = (1 + buy_hold_return) ** (1 / years_traded) - 1 if years_traded > 0 else 0
            
            # Enhancement analysis
            trades_df = pd.DataFrame(self.trades)
            enhancement_metrics = {}
            
            if not trades_df.empty:
                enhancement_metrics = {
                    'avg_confluence_score': trades_df['confluence_score'].mean(),
                    'avg_macro_scaling': trades_df['macro_scaling'].mean(),
                    'crisis_trades': (trades_df['regime'] == 'crisis').sum(),
                    'recession_trades': (trades_df['regime'] == 'recession').sum(),
                    'high_score_trades': (trades_df['opportunity_score'] > 5.0).sum(),
                    'avg_opportunity_score': trades_df['opportunity_score'].mean()
                }
            
            results = {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'total_trades': len(self.trades),
                'final_value': final_value,
                'total_invested': total_invested,
                'years_traded': years_traded,
                'buy_hold_annual': buy_hold_annual,
                'outperformance': annual_return - buy_hold_annual,
                'enhancement_metrics': enhancement_metrics
            }
            
            logger.info("✅ Enhanced performance calculation completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate enhanced performance: {e}")
            return {}
    
    def _display_enhanced_results(self, results):
        """Display enhanced backtest results"""
        print("\n" + "="*70)
        print("🌸 ENHANCED NANPIN STRATEGY v2.0 BACKTEST RESULTS 🌸")
        print("="*70)
        
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
        
        print(f"\n📊 BENCHMARK COMPARISON:")
        print(f"💼 Buy & Hold Annual: {results['buy_hold_annual']:+.1%}")
        print(f"🚀 Strategy Outperformance: {results['outperformance']:+.1%}")
        
        # Enhancement metrics
        if results['enhancement_metrics']:
            em = results['enhancement_metrics']
            print(f"\n🔧 ENHANCEMENT ANALYSIS:")
            print(f"🎯 Avg Confluence Score: {em.get('avg_confluence_score', 0):.2f}")
            print(f"📊 Avg Macro Scaling: {em.get('avg_macro_scaling', 0):.2f}x")
            print(f"🚨 Crisis Trades: {em.get('crisis_trades', 0)}")
            print(f"📉 Recession Trades: {em.get('recession_trades', 0)}")
            print(f"⭐ High Score Trades: {em.get('high_score_trades', 0)}")
            print(f"🎯 Avg Opportunity Score: {em.get('avg_opportunity_score', 0):.1f}")
        
        # Target comparison
        target_annual = 2.454  # 245.4%
        vs_target = results['annual_return'] / target_annual if target_annual > 0 else 0
        
        print(f"\n🎯 TARGET COMPARISON:")
        print(f"🏆 Target (Simple Trump Era): +245.4%")
        print(f"📊 Enhanced Strategy: {results['annual_return']:+.1%}")
        print(f"🎯 vs Target: {vs_target:.1%} ({'+' if results['annual_return'] > target_annual else '-'})")
        
        # Enhanced performance grade
        if results['annual_return'] > target_annual and results['sharpe_ratio'] > 2.0:
            grade = "A+ 🎉"
            recommendation = "EXCEPTIONAL - Strategy significantly beats target!"
        elif results['annual_return'] > target_annual:
            grade = "A 🎯"
            recommendation = "EXCELLENT - Strategy beats target!"
        elif results['annual_return'] > target_annual * 0.8 and results['sharpe_ratio'] > 1.5:
            grade = "A- 👍"
            recommendation = "STRONG - Very close to target with good risk metrics"
        elif results['annual_return'] > target_annual * 0.6:
            grade = "B+ 📊"
            recommendation = "GOOD - Solid performance, further optimization possible"
        elif results['annual_return'] > target_annual * 0.4:
            grade = "B 📈"
            recommendation = "MODERATE - Decent performance but needs improvement"
        else:
            grade = "C ⚠️"
            recommendation = "NEEDS OPTIMIZATION"
        
        print(f"\n🏆 ENHANCED PERFORMANCE GRADE: {grade}")
        print(f"💡 RECOMMENDATION: {recommendation}")
        print("\n" + "="*70)

async def main():
    """Run the enhanced backtest"""
    try:
        print("🌸 Enhanced Nanpin Strategy v2.0 - Real Macro Data Backtest")
        print("=" * 60)
        
        # Run enhanced backtest for key periods
        periods = [
            {"name": "Full Period (Enhanced)", "start": "2020-01-01", "end": "2024-12-31"},
            {"name": "COVID Era (Enhanced)", "start": "2020-01-01", "end": "2021-12-31"},
            {"name": "Recent Period (Enhanced)", "start": "2023-01-01", "end": "2024-12-31"}
        ]
        
        all_results = {}
        
        for period in periods:
            print(f"\n🔄 Testing {period['name']} ({period['start']} to {period['end']})...")
            
            backtester = EnhancedNanpinBacktest(period['start'], period['end'])
            results = await backtester.run_enhanced_backtest()
            all_results[period['name']] = results
        
        print(f"\n🎉 ALL ENHANCED BACKTESTS COMPLETED!")
        
        # Summary comparison
        print(f"\n📋 ENHANCED STRATEGY SUMMARY:")
        for period_name, results in all_results.items():
            if results:
                sharpe = results.get('sharpe_ratio', 0)
                print(f"   {period_name}: {results['annual_return']:+.1%} annual (Sharpe: {sharpe:.2f})")
        
        # Best period analysis
        best_period = max(all_results.items(), 
                         key=lambda x: x[1].get('annual_return', 0) if x[1] else 0)
        if best_period[1]:
            print(f"\n🏆 BEST PERFORMANCE: {best_period[0]}")
            print(f"   Annual Return: {best_period[1]['annual_return']:+.1%}")
            print(f"   Sharpe Ratio: {best_period[1]['sharpe_ratio']:.2f}")
            print(f"   Total Trades: {best_period[1]['total_trades']}")
        
    except Exception as e:
        print(f"❌ Enhanced backtest failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())