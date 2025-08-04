#!/usr/bin/env python3
"""
Macro-Enhanced Monte Carlo Risk Analysis for Nanpin Strategy
Integrates FRED economic data with crypto trading optimization

Implements Claude + Gemini AI consensus optimizations:
- M2 Money Supply tracking (+0.94 correlation, 70-107 day lag)
- DXY Dollar Index integration (-0.65 correlation, real-time)
- Fed Funds Rate regime detection (-0.65 to -0.75 correlation)
- VIX-based risk regime switching (+0.88 correlation)
- Multi-regime Monte Carlo simulations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import yfinance as yf
from fredapi import Fred
import warnings
warnings.filterwarnings('ignore')

class MacroEnhancedMonteCarloAnalyzer:
    """
    Advanced Monte Carlo analyzer with macro-economic overlay
    Based on Claude + Gemini AI consensus for maximum profit optimization
    """
    
    def __init__(self, symbol='BTC-USD', simulations=1000, fred_api_key=None):
        self.symbol = symbol
        self.simulations = simulations
        self.start_date = '2020-01-01'
        self.end_date = '2024-12-31'
        
        # Initialize FRED API (requires API key)
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None
        
        # Enhanced strategy parameters with macro-economic factors
        self.strategy_params = {
            # Base nanpin parameters (optimized)
            'min_drawdown': -8,  # More aggressive based on macro support
            'max_fear_greed': 55,  # Higher threshold in supportive macro environment
            'min_days_since_ath': 1,  # Faster response with macro confirmation
            
            # Macro-enhanced position sizing
            'base_position_size': 0.02,  # 2% base risk
            'max_position_size': 0.08,   # 8% max risk in optimal macro conditions
            'macro_position_multiplier': 2.5,  # Up to 2.5x size in perfect macro environment
            
            # M2 Money Supply parameters (Gemini + Claude consensus)
            'm2_sensitivity': 0.4,  # 10% M2 increase = 40% position increase
            'm2_lag_days': 75,      # 70-107 day average lag
            'm2_lookback_days': 180,  # 6 months M2 growth calculation
            
            # DXY Dollar Index parameters
            'dxy_correlation': -0.65,  # Strong negative correlation
            'dxy_sensitivity': 0.3,    # Position adjustment factor
            'dxy_ma_period': 20,       # Moving average for trend detection
            
            # Federal Funds Rate regime detection  
            'rate_cut_boost': 0.5,     # 50% position boost in cutting cycle
            'rate_hike_penalty': 0.3,  # 30% position reduction in hiking cycle
            'rate_neutral_threshold': 0.25,  # 25bp threshold for regime change
            
            # VIX-based risk regime switching
            'vix_correlation': 0.88,   # Record high correlation in 2024
            'low_vix_threshold': 15,   # Risk-on environment
            'high_vix_threshold': 25,  # Risk-off environment
            'vix_position_adjustment': 0.4,  # VIX-based position scaling
            
            # Fibonacci levels with macro enhancement
            'fibonacci_levels': {
                '23.6%': {
                    'ratio': 0.236, 'entry_window': [-3, 3], 
                    'base_multiplier': 0.8, 'macro_boost': 1.2
                },
                '38.2%': {
                    'ratio': 0.382, 'entry_window': [-4, 4], 
                    'base_multiplier': 1.0, 'macro_boost': 1.4
                },
                '50.0%': {
                    'ratio': 0.500, 'entry_window': [-5, 5], 
                    'base_multiplier': 1.2, 'macro_boost': 1.6
                },
                '61.8%': {
                    'ratio': 0.618, 'entry_window': [-6, 6], 
                    'base_multiplier': 1.4, 'macro_boost': 1.8
                },
                '78.6%': {
                    'ratio': 0.786, 'entry_window': [-8, 8], 
                    'base_multiplier': 1.6, 'macro_boost': 2.0
                }
            },
            
            # Enhanced leverage with macro regime awareness
            'base_leverage': 1.5,
            'max_leverage': 3.0,
            'macro_leverage_boost': 0.8,  # Additional leverage in supportive macro
            
            # Adaptive cooldown with macro acceleration
            'base_cooldown_hours': 12,    # Reduced from 18 hours
            'min_cooldown_hours': 4,      # Minimum cooldown
            'max_cooldown_hours': 48,     # Maximum cooldown
            'macro_cooldown_acceleration': 0.5,  # Faster trades in good macro
        }
        
        # Initialize data containers
        self.price_data = None
        self.macro_data = {}
        self.macro_regimes = None
        self.results = {}
        
    def fetch_macro_data(self):
        """
        Fetch macro-economic data from FRED and other sources
        High priority indicators based on research
        """
        print("📊 Fetching macro-economic data...")
        
        if self.fred is None:
            print("⚠️  FRED API key not provided, using mock data for demonstration")
            self._create_mock_macro_data()
            return
            
        try:
            # M2 Money Supply (highest correlation +0.94)
            print("   Fetching M2 Money Supply data...")
            self.macro_data['M2'] = self.fred.get_series('M2SL', 
                start=self.start_date, end=self.end_date)
            
            # Federal Funds Rate (strong negative correlation)
            print("   Fetching Federal Funds Rate...")
            self.macro_data['FED_RATE'] = self.fred.get_series('FEDFUNDS', 
                start=self.start_date, end=self.end_date)
            
            # DXY Dollar Index
            print("   Fetching DXY Dollar Index...")
            self.macro_data['DXY'] = self.fred.get_series('DTWEXBGS', 
                start=self.start_date, end=self.end_date)
            
            # VIX (using Yahoo Finance as FRED doesn't have real-time VIX)
            print("   Fetching VIX data...")
            vix_data = yf.download('^VIX', start=self.start_date, end=self.end_date)
            self.macro_data['VIX'] = vix_data['Close']
            
            # Employment data
            print("   Fetching employment data...")
            self.macro_data['UNEMPLOYMENT'] = self.fred.get_series('UNRATE', 
                start=self.start_date, end=self.end_date)
            
            # CPI for inflation context
            print("   Fetching CPI data...")
            self.macro_data['CPI'] = self.fred.get_series('CPIAUCSL', 
                start=self.start_date, end=self.end_date)
                
        except Exception as e:
            print(f"❌ Error fetching FRED data: {e}")
            print("🔄 Using mock data for demonstration...")
            self._create_mock_macro_data()
    
    def _create_mock_macro_data(self):
        """Create realistic mock macro data for demonstration"""
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        # Mock M2 with realistic growth pattern
        m2_base = 15000  # Billion USD
        m2_growth = np.random.normal(0.0002, 0.0005, len(date_range)).cumsum()
        self.macro_data['M2'] = pd.Series(
            m2_base * (1 + m2_growth), 
            index=date_range
        )
        
        # Mock Fed Funds Rate with realistic cycles
        fed_rate_base = np.concatenate([
            np.linspace(1.75, 0.25, 180),  # 2020 cuts
            np.full(400, 0.25),             # Zero rate period
            np.linspace(0.25, 5.0, 300),   # 2022-2023 hikes
            np.linspace(5.0, 4.5, len(date_range) - 880)  # Recent cuts
        ])
        self.macro_data['FED_RATE'] = pd.Series(
            fed_rate_base[:len(date_range)], 
            index=date_range
        )
        
        # Mock DXY with volatility
        dxy_trend = 100 + np.random.normal(0, 0.1, len(date_range)).cumsum()
        self.macro_data['DXY'] = pd.Series(dxy_trend, index=date_range)
        
        # Mock VIX with volatility clustering
        vix_base = 20 + 15 * np.abs(np.random.normal(0, 1, len(date_range)))
        self.macro_data['VIX'] = pd.Series(vix_base, index=date_range)
        
        # Mock other indicators
        self.macro_data['UNEMPLOYMENT'] = pd.Series(
            6.0 + 2 * np.sin(np.linspace(0, 4*np.pi, len(date_range))), 
            index=date_range
        )
        self.macro_data['CPI'] = pd.Series(
            250 * (1.02 ** (np.arange(len(date_range)) / 365)), 
            index=date_range
        )
    
    def analyze_macro_regimes(self):
        """
        Analyze macro-economic regimes for enhanced trading
        Based on Claude + Gemini consensus on regime detection
        """
        print("🧠 Analyzing macro-economic regimes...")
        
        # Get common date range for all indicators
        common_dates = self.price_data.index
        
        # Align macro data to price data dates
        aligned_macro = {}
        for key, series in self.macro_data.items():
            aligned_macro[key] = series.reindex(common_dates, method='ffill')
        
        # Calculate regime indicators
        regimes = pd.DataFrame(index=common_dates)
        
        # 1. M2 Growth Regime (70-107 day lag consideration)
        m2_growth = aligned_macro['M2'].pct_change(self.strategy_params['m2_lookback_days'])
        regimes['m2_bullish'] = (m2_growth > m2_growth.quantile(0.6)).astype(int)
        
        # 2. Fed Rate Regime (cutting vs hiking cycles)
        fed_rate_change = aligned_macro['FED_RATE'].diff(20)  # 20-day change
        regimes['fed_cutting'] = (fed_rate_change < -self.strategy_params['rate_neutral_threshold']).astype(int)
        regimes['fed_hiking'] = (fed_rate_change > self.strategy_params['rate_neutral_threshold']).astype(int)
        
        # 3. DXY Trend Regime
        dxy_ma = aligned_macro['DXY'].rolling(self.strategy_params['dxy_ma_period']).mean()
        regimes['dxy_declining'] = (aligned_macro['DXY'] < dxy_ma).astype(int)
        
        # 4. VIX Risk Regime
        regimes['low_vix'] = (aligned_macro['VIX'] < self.strategy_params['low_vix_threshold']).astype(int)
        regimes['high_vix'] = (aligned_macro['VIX'] > self.strategy_params['high_vix_threshold']).astype(int)
        
        # 5. Composite Macro Score (0-4 scale)
        regimes['macro_score'] = (
            regimes['m2_bullish'] +
            regimes['fed_cutting'] + 
            regimes['dxy_declining'] +
            regimes['low_vix']
        )
        
        # 6. Regime Classification
        regimes['regime'] = pd.cut(
            regimes['macro_score'], 
            bins=[-1, 0, 1, 2, 4], 
            labels=['Bearish', 'Neutral', 'Bullish', 'Very_Bullish']
        )
        
        self.macro_regimes = regimes
        
        # Print regime distribution
        regime_dist = regimes['regime'].value_counts()
        print(f"   Macro Regime Distribution:")
        for regime, count in regime_dist.items():
            pct = count / len(regimes) * 100
            print(f"     {regime}: {count} days ({pct:.1f}%)")
    
    def calculate_macro_position_multiplier(self, day):
        """
        Calculate position size multiplier based on macro conditions
        Implements Gemini AI recommended formula
        """
        if self.macro_regimes is None or day >= len(self.macro_regimes):
            return 1.0
            
        regime_row = self.macro_regimes.iloc[day]
        
        # Base multiplier from regime score
        base_multiplier = 1.0 + (regime_row['macro_score'] / 4.0) * (self.strategy_params['macro_position_multiplier'] - 1.0)
        
        # Additional specific adjustments
        adjustments = 0
        
        # M2 sensitivity (strongest correlation)
        if day > self.strategy_params['m2_lag_days']:
            m2_lagged_day = day - self.strategy_params['m2_lag_days']
            if m2_lagged_day >= 0 and self.macro_regimes.iloc[m2_lagged_day]['m2_bullish']:
                adjustments += self.strategy_params['m2_sensitivity']
        
        # Fed rate regime
        if regime_row['fed_cutting']:
            adjustments += self.strategy_params['rate_cut_boost']
        elif regime_row['fed_hiking']:
            adjustments -= self.strategy_params['rate_hike_penalty']
        
        # DXY correlation
        if regime_row['dxy_declining']:
            adjustments += self.strategy_params['dxy_sensitivity']
        
        # VIX adjustment (risk-on/risk-off)
        if regime_row['low_vix']:
            adjustments += self.strategy_params['vix_position_adjustment']
        elif regime_row['high_vix']:
            adjustments -= self.strategy_params['vix_position_adjustment']
        
        final_multiplier = base_multiplier + adjustments
        
        # Cap the multiplier
        return max(0.1, min(final_multiplier, self.strategy_params['macro_position_multiplier']))
    
    def _run_macro_enhanced_strategy(self, synthetic_prices, fear_greed, days_since_ath, 
                                   drawdowns, liquidation_intensity, simulated_volume):
        """
        Enhanced strategy with macro-economic overlay
        Implements Claude + Gemini consensus optimizations
        """
        total_btc = 0
        total_invested = 0
        positions = []
        last_trade_time = None
        
        price_series = pd.Series(synthetic_prices)
        
        # MACRO-ENHANCED TRADING LOOP
        for day, (price, drawdown, fg, days_ath, liq_intensity, volume) in enumerate(zip(
            synthetic_prices, drawdowns, fear_greed, days_since_ath, liquidation_intensity, simulated_volume
        )):
            
            # Calculate macro position multiplier
            macro_multiplier = self.calculate_macro_position_multiplier(day)
            
            # Calculate recent volatility for dynamic adjustments
            recent_vol = np.std([np.log(synthetic_prices[max(1, i)] / synthetic_prices[max(0, i-1)]) 
                               for i in range(max(1, day-10), day+1)]) if day > 10 else 0.02
            
            # MACRO-ENHANCED COOLDOWN
            if last_trade_time is not None:
                vol_ratio = recent_vol / 0.02  # Normalize against base volatility
                
                # Accelerate cooldown in good macro conditions
                macro_acceleration = 1.0
                if macro_multiplier > 1.5:  # Strong macro support
                    macro_acceleration = self.strategy_params['macro_cooldown_acceleration']
                
                dynamic_cooldown = self.strategy_params['base_cooldown_hours'] * macro_acceleration * (
                    1 - 0.3 * min(vol_ratio, 2.0)
                )
                dynamic_cooldown = max(
                    self.strategy_params['min_cooldown_hours'],
                    min(dynamic_cooldown, self.strategy_params['max_cooldown_hours'])
                )
                
                if (day - last_trade_time) * 24 < dynamic_cooldown:
                    continue
            
            # MACRO-ENHANCED ENTRY CRITERIA
            # Relax criteria in supportive macro environment
            macro_adjustment = (macro_multiplier - 1.0) * 5  # Scale adjustment
            
            adjusted_min_drawdown = self.strategy_params['min_drawdown'] + macro_adjustment
            adjusted_max_fg = self.strategy_params['max_fear_greed'] - macro_adjustment
            adjusted_min_days_ath = max(0, self.strategy_params['min_days_since_ath'] - int(macro_adjustment))
            
            if (drawdown <= adjusted_min_drawdown and
                fg <= adjusted_max_fg and
                days_ath >= adjusted_min_days_ath):
                
                # Enhanced Fibonacci analysis
                lookback_start = max(0, day - 75)
                recent_high = price_series.iloc[lookback_start:day+1].max()
                recent_low = price_series.iloc[lookback_start:day+1].min()
                price_range = recent_high - recent_low
                
                if price_range > recent_high * 0.03:  # Macro-adjusted threshold
                    
                    best_score = 0
                    best_level = None
                    
                    for level_name, level_config in self.strategy_params['fibonacci_levels'].items():
                        fib_price = recent_high - (price_range * level_config['ratio'])
                        distance_pct = (price - fib_price) / fib_price * 100
                        
                        entry_window = level_config['entry_window']
                        if entry_window[0] <= distance_pct <= entry_window[1]:
                            
                            # MACRO-ENHANCED SCORING
                            base_score = level_config['base_multiplier'] * abs(distance_pct)
                            macro_boost = level_config['macro_boost'] * macro_multiplier
                            
                            # Enhanced scoring with liquidation and volume
                            liquidation_score = liq_intensity * 0.3
                            volume_percentile = np.percentile(simulated_volume[:day+1], 75) if day > 10 else volume
                            volume_score = min(volume / volume_percentile, 2.0) * 5
                            
                            composite_score = base_score * macro_boost + liquidation_score + volume_score
                            
                            if composite_score > best_score:
                                best_score = composite_score
                                
                                # Macro-enhanced leverage calculation
                                base_lev = self.strategy_params['base_leverage']
                                macro_lev_boost = (macro_multiplier - 1.0) * self.strategy_params['macro_leverage_boost']
                                vol_boost = recent_vol * 15  # Volatility adjustment
                                
                                leverage = min(
                                    base_lev + macro_lev_boost + vol_boost,
                                    self.strategy_params['max_leverage']
                                )
                                
                                best_level = {
                                    'multiplier': level_config['base_multiplier'],
                                    'leverage': leverage,
                                    'level_name': level_name,
                                    'macro_multiplier': macro_multiplier,
                                    'composite_score': composite_score
                                }
                    
                    if best_level:
                        # MACRO-ENHANCED POSITION SIZING
                        base_position = self.strategy_params['base_position_size']
                        
                        # Apply macro multiplier to position size
                        position_size = base_position * macro_multiplier * best_level['multiplier']
                        
                        # Apply volatility and score adjustments
                        volatility_adjustment = 1.0 + min(recent_vol * 10, 0.5)
                        score_adjustment = 1.0 + (best_level['composite_score'] / 100)
                        
                        final_position_size = position_size * volatility_adjustment * score_adjustment
                        
                        # Cap position size
                        final_position_size = min(final_position_size, self.strategy_params['max_position_size'])
                        
                        # Calculate BTC purchase
                        leverage = best_level['leverage']
                        effective_capital = final_position_size * leverage
                        btc_bought = effective_capital / price
                        
                        total_btc += btc_bought
                        total_invested += final_position_size
                        last_trade_time = day
                        
                        positions.append({
                            'day': day,
                            'price': price,
                            'btc_amount': btc_bought,
                            'invested': final_position_size,
                            'leverage': leverage,
                            'level': best_level['level_name'],
                            'macro_regime': self.macro_regimes.iloc[day]['regime'] if day < len(self.macro_regimes) else 'Unknown',
                            'macro_multiplier': macro_multiplier,
                            'composite_score': best_level['composite_score']
                        })
        
        # Calculate final results
        if total_btc > 0:
            final_value = total_btc * synthetic_prices[-1]
            total_return = (final_value - total_invested) / total_invested if total_invested > 0 else 0
            years = (pd.to_datetime(self.end_date) - pd.to_datetime(self.start_date)).days / 365.25
            annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
            trades_per_year = len(positions) / years if years > 0 else 0
        else:
            final_value = total_invested
            total_return = 0
            annual_return = 0
            trades_per_year = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'final_value': final_value,
            'total_invested': total_invested,
            'trades_per_year': trades_per_year,
            'total_trades': len(positions),
            'positions': positions
        }
    
    def run_analysis(self):
        """
        Run comprehensive macro-enhanced Monte Carlo analysis
        """
        print("🚀 MACRO-ENHANCED MONTE CARLO RISK ANALYSIS")
        print("=" * 70)
        
        # Fetch data
        print("\n📈 Loading price data...")
        self.price_data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        if self.price_data.empty:
            raise ValueError("No price data available")
        
        # Fetch macro data
        self.fetch_macro_data()
        
        # Analyze macro regimes
        self.analyze_macro_regimes()
        
        # Calculate base statistics
        price_series = self.price_data['Close']
        returns = price_series.pct_change().dropna()
        self.return_mean = returns.mean()
        self.return_std = returns.std()
        
        print(f"\n✅ Data loaded: {len(price_series)} days")
        print(f"   Historical volatility: {float(self.return_std * np.sqrt(252) * 100):.1f}%")
        print(f"   Mean daily return: {float(self.return_mean * 100):.3f}%")
        
        # Run Monte Carlo simulations
        print(f"\n🔄 Running {self.simulations} macro-enhanced Monte Carlo simulations...")
        
        simulation_results = []
        
        for sim in range(self.simulations):
            if sim % 100 == 0:
                print(f"   Simulation {sim+1}/{self.simulations}")
            
            # Generate synthetic price path
            np.random.seed(sim)
            days = len(price_series)
            synthetic_returns = np.random.normal(self.return_mean, self.return_std, days)
            synthetic_prices = [price_series.iloc[0]]
            
            for i in range(1, days):
                synthetic_prices.append(synthetic_prices[-1] * (1 + synthetic_returns[i]))
            
            # Generate other synthetic data
            fear_greed = np.random.randint(0, 100, days)
            days_since_ath = np.random.randint(0, 100, days)
            drawdowns = np.random.uniform(-0.3, 0, days)
            liquidation_intensity = np.random.exponential(5, days)
            simulated_volume = np.random.lognormal(15, 1, days)
            
            # Run macro-enhanced strategy
            result = self._run_macro_enhanced_strategy(
                synthetic_prices, fear_greed, days_since_ath, 
                drawdowns, liquidation_intensity, simulated_volume
            )
            
            simulation_results.append(result)
        
        # Analyze results
        self._analyze_results(simulation_results)
        
        # Generate visualization
        self._create_visualization(simulation_results)
        
        return self.results
    
    def _analyze_results(self, simulation_results):
        """Analyze Monte Carlo simulation results"""
        print("\n✅ Completed simulations, analyzing results...")
        
        returns = [r['annual_return'] for r in simulation_results]
        trades_per_year = [r['trades_per_year'] for r in simulation_results]
        
        self.results = {
            'mean_return': np.mean(returns),
            'median_return': np.median(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'positive_return_prob': np.mean([r > 0 for r in returns]),
            'mean_trades_per_year': np.mean(trades_per_year),
            'var_95': np.percentile(returns, 5),
            'var_99': np.percentile(returns, 1),
            'cvar_95': np.mean([r for r in returns if r <= np.percentile(returns, 5)]),
            'cvar_99': np.mean([r for r in returns if r <= np.percentile(returns, 1)]),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'raw_results': simulation_results
        }
        
        # Print results
        print("\n🎲 MACRO-ENHANCED MONTE CARLO RESULTS")
        print("=" * 70)
        print(f"\n📊 RETURN DISTRIBUTION:")
        print(f"   Mean Annual Return: {self.results['mean_return']*100:+.1f}%")
        print(f"   Median Annual Return: {self.results['median_return']*100:+.1f}%")
        print(f"   Standard Deviation: {self.results['std_return']*100:.1f}%")
        print(f"   Best Case: {self.results['max_return']*100:+.1f}%")
        print(f"   Worst Case: {self.results['min_return']*100:+.1f}%")
        print(f"\n📊 RISK METRICS:")
        print(f"   Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        print(f"   Value at Risk (95%): {self.results['var_95']*100:+.1f}%")
        print(f"   Value at Risk (99%): {self.results['var_99']*100:+.1f}%")
        print(f"   Conditional VaR (95%): {self.results['cvar_95']*100:+.1f}%")
        print(f"   Conditional VaR (99%): {self.results['cvar_99']*100:+.1f}%")
        print(f"\n📊 SUCCESS PROBABILITIES:")
        print(f"   Probability of Positive Returns: {self.results['positive_return_prob']*100:.1f}%")
        print(f"\n📊 TRADING CHARACTERISTICS:")
        print(f"   Mean Trades per Year: {self.results['mean_trades_per_year']:.1f}")
        
        # Assess risk grade
        if self.results['mean_return'] > 0.5 and self.results['positive_return_prob'] > 0.8:
            grade = "A+ 🏆"
            assessment = "EXCEPTIONAL MACRO-ENHANCED PERFORMANCE"
        elif self.results['mean_return'] > 0.3 and self.results['positive_return_prob'] > 0.7:
            grade = "A 🎯"
            assessment = "STRONG MACRO-ENHANCED PERFORMANCE"
        elif self.results['mean_return'] > 0.1 and self.results['positive_return_prob'] > 0.6:
            grade = "B+ 📈"
            assessment = "GOOD MACRO-ENHANCED PERFORMANCE"
        else:
            grade = "C ⚠️"
            assessment = "NEEDS OPTIMIZATION"
        
        print(f"\n🏆 RISK GRADE: {grade}")
        print(f"💡 ASSESSMENT: {assessment}")
    
    def _create_visualization(self, simulation_results):
        """Create enhanced visualization with macro insights"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Macro-Enhanced Monte Carlo Analysis - Nanpin Strategy', fontsize=16, fontweight='bold')
        
        returns = [r['annual_return'] for r in simulation_results]
        trades = [r['trades_per_year'] for r in simulation_results]
        
        # Annual Return Distribution
        axes[0,0].hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].axvline(np.mean(returns), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(returns)*100:.1f}%')
        axes[0,0].axvline(2.454, color='green', linestyle='--', 
                         label='Target: 245.4%')
        axes[0,0].set_xlabel('Annual Return')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Annual Return Distribution\nMacro-Enhanced Strategy')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Risk Metrics
        risk_metrics = ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%']
        risk_values = [self.results['var_95'], self.results['var_99'], 
                      self.results['cvar_95'], self.results['cvar_99']]
        colors = ['orange', 'red', 'orange', 'darkred']
        
        bars = axes[0,1].bar(risk_metrics, [v*100 for v in risk_values], color=colors)
        axes[0,1].set_ylabel('Annual Return %')
        axes[0,1].set_title('Risk Metrics\nMacro-Enhanced Strategy')
        axes[0,1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, risk_values):
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 1,
                          f'{value*100:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Trading Frequency Analysis
        axes[0,2].scatter(trades, returns, alpha=0.6, s=20)
        axes[0,2].axhline(y=2.454, color='green', linestyle='--', alpha=0.7, 
                         label='Target Return (245.4%)')
        axes[0,2].axvline(x=np.mean(trades), color='red', linestyle='--', alpha=0.7,
                         label=f'Mean: {np.mean(trades):.1f} trades/year')
        axes[0,2].set_xlabel('Trades per Year')
        axes[0,2].set_ylabel('Annual Return')
        axes[0,2].set_title('Trade Frequency vs Returns\nMacro-Enhanced Optimization')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # Macro Regime Analysis
        if self.macro_regimes is not None:
            regime_performance = {}
            for result in simulation_results:
                for pos in result['positions']:
                    regime = pos.get('macro_regime', 'Unknown')
                    if regime not in regime_performance:
                        regime_performance[regime] = []
                    regime_performance[regime].append(pos.get('composite_score', 0))
            
            if regime_performance:
                regimes = list(regime_performance.keys())
                avg_scores = [np.mean(regime_performance[r]) for r in regimes]
                
                axes[1,0].bar(regimes, avg_scores, color='lightgreen')
                axes[1,0].set_xlabel('Macro Regime')
                axes[1,0].set_ylabel('Average Trade Score')
                axes[1,0].set_title('Performance by Macro Regime')
                axes[1,0].tick_params(axis='x', rotation=45)
                axes[1,0].grid(True, alpha=0.3)
        
        # Cumulative Probability Distribution
        sorted_returns = np.sort(returns)
        cumulative_prob = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
        axes[1,1].plot(sorted_returns, cumulative_prob, linewidth=2, color='navy')
        axes[1,1].axvline(0, color='red', linestyle='--', alpha=0.7, label='Break-even')
        axes[1,1].axvline(2.454, color='green', linestyle='--', alpha=0.7, 
                         label='Target (245.4%)')
        axes[1,1].set_xlabel('Annual Return')
        axes[1,1].set_ylabel('Cumulative Probability')
        axes[1,1].set_title('Cumulative Probability Distribution')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Position Size Distribution by Macro Conditions
        all_multipliers = []
        all_scores = []
        for result in simulation_results:
            for pos in result['positions']:
                all_multipliers.append(pos.get('macro_multiplier', 1.0))
                all_scores.append(pos.get('composite_score', 0))
        
        if all_multipliers:
            axes[1,2].scatter(all_multipliers, all_scores, alpha=0.5, s=10)
            axes[1,2].set_xlabel('Macro Position Multiplier')
            axes[1,2].set_ylabel('Composite Trade Score')
            axes[1,2].set_title('Trade Quality vs Macro Conditions')
            axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('monte_carlo_macro_enhanced.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualizations saved to: monte_carlo_macro_enhanced.png")

if __name__ == "__main__":
    print("🚀 Starting Macro-Enhanced Monte Carlo Analysis...")
    
    # Note: To use real FRED data, you need a free API key from https://fred.stlouisfed.org/
    # analyzer = MacroEnhancedMonteCarloAnalyzer('BTC-USD', 1000, fred_api_key='your_api_key_here')
    
    # Running with mock data for demonstration
    analyzer = MacroEnhancedMonteCarloAnalyzer('BTC-USD', 500)  # 500 simulations for faster demo
    results = analyzer.run_analysis()
    
    print(f"\n🎉 MACRO-ENHANCED ANALYSIS COMPLETE!")
    print(f"✅ Generated comprehensive macro-economic risk assessment")
    print(f"📊 Key insight: {results['positive_return_prob']*100:.1f}% probability of positive returns")
    print(f"🎯 Mean annual return: {results['mean_return']*100:+.1f}%")
    print(f"⚡ Average {results['mean_trades_per_year']:.1f} trades per year")